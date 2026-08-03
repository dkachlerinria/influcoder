#!/usr/bin/env python3
"""Unified Fig 2 (Counterfact) timing + accuracy harness -- ONE file, every
method's parameters in ONE place, run ONE method per invocation so partial
progress survives a GPU preemption (results accumulate in
results/fig2_counterfact.json, keyed by method, never overwritten wholesale).

Fairness rule for `wall_s` (the number that goes in the figure): every method
here computes its score fresh, from the checkpoint, with no reusable trained
artifact -- except InfluCoder, which has a one-time setup cost (teacher-
gradient extraction + distillation) that amortizes across every future query.
So InfluCoder's `wall_s` is INFERENCE ONLY (embed+score), matching what every
other method's number already is by construction. The setup cost is measured
and saved too (as `setup_s`), NOT dropped -- present it separately, don't
average it into `wall_s` and don't hide it either.

Methods: bm25, repsim, graddot, gradsim, less, datainf, ekfac, influcoder.

Usage:
    python methods/run_fig2_counterfact.py --method bm25
    python methods/run_fig2_counterfact.py --method repsim
    python methods/run_fig2_counterfact.py --method graddot
    python methods/run_fig2_counterfact.py --method gradsim
    python methods/run_fig2_counterfact.py --method less
    python methods/run_fig2_counterfact.py --method datainf
    python methods/run_fig2_counterfact.py --method ekfac
    python methods/run_fig2_counterfact.py --method influcoder
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import types
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "evaluation"))

# ---------------------------------------------------------------------------
# Harmless import-time-only stubs for litgpt/lightning, injected directly into
# sys.modules so this script needs no external _stubs/ dir or PYTHONPATH hack.
# methods/model_utils.py (imported transitively for dattri-backed methods)
# imports these at module level for a sibling litgpt checkpoint loader that
# the HF/peft path used here never calls -- confirmed safe.
# ---------------------------------------------------------------------------
if "litgpt" not in sys.modules:
    litgpt = types.ModuleType("litgpt")
    litgpt_lora = types.ModuleType("litgpt.lora")
    litgpt_utils = types.ModuleType("litgpt.utils")

    class _StubGPT:
        pass

    class _StubConfig:
        pass

    def _stub_raise(*a, **k):
        raise NotImplementedError("stub: real litgpt not installed, not used by this script's HF/peft path")

    litgpt_lora.GPT = _StubGPT
    litgpt_lora.Config = _StubConfig
    litgpt_lora.lora_filter = _stub_raise
    litgpt_lora.merge_lora_weights = _stub_raise
    litgpt_utils.extend_checkpoint_dir = _stub_raise
    litgpt.lora = litgpt_lora
    litgpt.utils = litgpt_utils
    sys.modules["litgpt"] = litgpt
    sys.modules["litgpt.lora"] = litgpt_lora
    sys.modules["litgpt.utils"] = litgpt_utils

if "lightning" not in sys.modules:
    sys.modules["lightning"] = types.ModuleType("lightning")
# ---------------------------------------------------------------------------

import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from datamodules.load_data import get_dataset, prepare_chat_format  # noqa: E402
from evaluate_application import get_fact_indices_counterfact, evaluate_fact, read_json  # noqa: E402

# ============================================================================
# ONE PLACE for every method's parameters. Change a number here, not in five
# different scripts.
# ============================================================================
TASK = "Counterfact"
SUBSET = "Pythia-1b"
BASE_MODEL = "EleutherAI/pythia-1b"
CHECKPOINT = "DataAttributionEval/Pythia-1b-counterfactual"
MAX_LEN = 1024
SEED = 0

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "fig2_counterfact.json"
GRAD_CACHE_DIR = RESULTS_DIR / "_grad_cache"

# Populated by run_influcoder(), merged into the saved record by main().
EXTRA_RECORD_FIELDS = {}

# dattri.py-backed methods (Grad_Dot/Grad_Sim/LESS/DataInf/EKFAC): method name
# as dattri.py expects it, plus method-specific hyperparameters. batch_size=1
# throughout (dattri's own convention, not overridden here).
DATTRI_METHOD_NAME = {
    "graddot": "Grad_Dot",
    "gradsim": "Grad_Sim",
    "less": "LESS",
    "datainf": "DataInf",
    "ekfac": "EKFAC",
}
DATTRI_EXTRA_PARAMS = {
    "graddot": {},
    "gradsim": {},
    "less": {"proj_dim": 8192},
    "datainf": {"regularization": 1e-5, "fim_estimate_data_ratio": 1.0},
    "ekfac": {"damping": 1e-7},
}

# Rep-Sim: forward-pass-only, last-token hidden state, cosine similarity.
REPSIM_BATCH_SIZE = 1

# BM25: pure lexical, no model/GPU needed at all.
BM25_STOPWORDS = "en"

# Per-method meaning of setup_s. Both InfluCoder and Semantic report an
# inference-only wall_s with a separately-measured setup_s, so their wall_s
# values bracket the identical embed+score region and are directly comparable;
# what each pays up front differs, hence the differing notes.
SETUP_NOTES = {
    "influcoder": "one-time cost (get_dataset + pool building + teacher-grad extraction + encoder load + distillation), NOT included in wall_s",
    "repsim": "checkpoint load + get_dataset + dataset construction, NOT included in wall_s -- same convention as InfluCoder/Semantic so all three wall_s values bracket compute only",
    "semantic": "get_dataset + encoder load, NOT included in wall_s -- excluded to match InfluCoder's inference-only wall_s convention (Semantic is InfluCoder's untrained-encoder control doing identical embed+score work)",
}


# InfluCoder: BEST known config, found via the hard_ratio + encoder-size sweep
# in EXP2.md -- ettin-400m, moredata scale, hard_ratio=0.25
# (Recall@50=0.4689, MRR=0.8739 the first time this exact config was measured;
# note EXP2.md also documents a GPU-dependent reproducibility gap on this
# exact config, so don't be surprised by a few points of drift on a different
# card).
INFLUCODER_ENCODER_MODEL = "jhu-clsp/ettin-encoder-400m"
INFLUCODER_ENCODER_MAX_LEN = 1024
INFLUCODER_PROJ_DIM = 8192
INFLUCODER_EXTERNAL_REPO = "NeelNanda/counterfact-tracing"
INFLUCODER_N_EXT_ANCHORS = 900
INFLUCODER_N_TEACHER_TRAIN = 2000
INFLUCODER_N_EVAL_TRAIN = 500
INFLUCODER_EPOCHS = 8
INFLUCODER_HARD_RATIO = 0.25
INFLUCODER_LR = 5e-5
INFLUCODER_SELECT_BEST_ON = "aggregated"


# ============================================================================
# Shared helpers
# ============================================================================
def detect_gpu_label() -> str:
    """Never hardcode a GPU model name -- auto-detect so a result run on
    whichever card happens to be free is labeled correctly, not silently
    mislabeled with whatever GPU an earlier run used."""
    try:
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
    except Exception:
        name = "CPU-only (no GPU used)"
    return f"{name} ({socket.getfqdn()})"


def example_text(d: dict) -> str:
    return f"{d['prompt'].strip()}\n{d['response'].strip()}"


def evaluate(score_path: Path) -> tuple[float, float]:
    """Uses evaluate_application.py's own Counterfact ground-truth/scoring
    functions directly (imported, not reimplemented) -- get_fact_indices_counterfact
    is the real ground-truth definition; never guess at it."""
    train, ref = get_dataset(TASK, SUBSET)
    score = read_json(str(score_path))
    fact_indices_list = get_fact_indices_counterfact(train, ref)
    return evaluate_fact(score, fact_indices_list, k=50)


def save_summary(method_key: str, record: dict):
    """flock-protected read-modify-write: with several methods now running in
    true parallel across separate GPUs and writing to this SAME summary file,
    an unprotected read+write here would race -- two processes reading the
    old file before either writes back loses whichever write lands first.
    The lock covers only this tiny read/modify/write, not the (multi-minute)
    compute that produced `record`, so parallelism is unaffected."""
    import fcntl
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = SUMMARY_PATH.with_suffix(".lock")
    with open(lock_path, "w") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        try:
            summary = {}
            if SUMMARY_PATH.exists():
                summary = json.loads(SUMMARY_PATH.read_text())
            summary[method_key] = record
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        finally:
            fcntl.flock(lockfile, fcntl.LOCK_UN)
    print(f"wrote {SUMMARY_PATH} (method={method_key})")


# ============================================================================
# BM25 -- pure lexical, CPU only, no checkpoint/GPU needed
# ============================================================================
def run_bm25():
    import bm25s

    # get_dataset() timed IN, matching Rep-Sim/dattri-backed methods (dattri's
    # own attribute() calls get_dataset() internally as its first step) --
    # keeps wall_s meaning the same thing across every method here. Previously
    # excluded, giving BM25/InfluCoder/semantic a small unearned head start;
    # found during a fairness review, fixed going forward (not retroactively
    # applied to already-completed runs).
    t0 = time.perf_counter()
    local_train, local_ref = get_dataset(TASK, SUBSET)
    train_texts = [example_text(d) for d in local_train]
    ref_texts = [example_text(d) for d in local_ref]

    train_tokens = bm25s.tokenize(train_texts, stopwords=BM25_STOPWORDS)
    retriever = bm25s.BM25()
    retriever.index(train_tokens)

    # get_scores() takes ONE query's raw string tokens at a time (not the
    # batch-tokenized/ID-mapped object bm25s.tokenize() returns by default),
    # so tokenize with return_ids=False and loop -- gives the full
    # [n_ref, n_train] score matrix (every train doc, in original order, per
    # ref query), NOT top-k retrieval, which would drop/reorder docs and
    # break the Recall@50/MRR convention every other method here uses.
    query_tokens = bm25s.tokenize(ref_texts, stopwords=BM25_STOPWORDS, return_ids=False)
    scores = np.stack([retriever.get_scores(qt) for qt in query_tokens])  # [n_ref, n_train]
    wall_s = time.perf_counter() - t0

    save_path = RESULTS_DIR / "fig2-counterfact-bm25" / "BM25.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(np.asarray(scores).tolist(), f)

    return wall_s, None, save_path


# ============================================================================
# Rep-Sim -- forward-pass-only, no backward pass
# ============================================================================
def run_repsim():
    from torch.nn.functional import normalize
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from datamodules.data_utils import MessageDatasetRepSim
    from methods.model_utils import checkpoints_load_func

    def collate_fn(tokenizer):
        def _fn(batch):
            input_ids = [torch.tensor(x["input_ids"]) for x in batch]
            attention_masks = [torch.tensor(x["attention_mask"]) for x in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
            return {"input_ids": input_ids, "attention_mask": attention_masks}
        return _fn

    def collect_reps(loader, model, desc):
        device = next(model.parameters()).device
        reps = []
        for batch in tqdm(loader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                ids = torch.arange(len(input_ids), device=device)
                pos = attention_mask.sum(dim=1) - 1
                batch_reps = out.hidden_states[-1][ids, pos]
            reps.append(batch_reps.float().cpu())
        return normalize(torch.cat(reps), dim=1)

    t_setup0 = time.perf_counter()
    tokenizer, model = checkpoints_load_func(None, CHECKPOINT, BASE_MODEL)
    model.eval()

    local_train, local_ref = get_dataset(TASK, SUBSET)
    train_chat = prepare_chat_format(local_train)
    ref_chat = prepare_chat_format(local_ref)

    collate = collate_fn(tokenizer)
    train_ds = MessageDatasetRepSim(train_chat, tokenizer=tokenizer, max_length=MAX_LEN)
    ref_ds = MessageDatasetRepSim(ref_chat, tokenizer=tokenizer, max_length=MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=REPSIM_BATCH_SIZE, shuffle=False, collate_fn=collate)
    ref_loader = DataLoader(ref_ds, batch_size=REPSIM_BATCH_SIZE, shuffle=False, collate_fn=collate)
    setup_s = time.perf_counter() - t_setup0

    # Same two-phase convention as run_semantic()/run_influcoder(): setup_s is
    # checkpoint load + get_dataset + dataset/loader construction; wall_s is the
    # forward passes + scoring only. Tokenization stays inside wall_s (it happens
    # lazily during DataLoader iteration), matching Semantic, where
    # sentence-transformers tokenizes inside embed(). NOTE the dattri-backed
    # methods CANNOT be split this way -- they load the checkpoint inside
    # upstream's attribute(), which we keep byte-identical -- so their wall_s
    # still carries a ~5s load. Negligible at their 340-11000s scale.
    t_inf0 = time.perf_counter()

    train_reps = collect_reps(train_loader, model, desc="repsim train reps")
    ref_reps = collect_reps(ref_loader, model, desc="repsim ref reps")
    sim = train_reps @ ref_reps.T  # [n_train, n_ref]
    wall_s = time.perf_counter() - t_inf0

    save_path = RESULTS_DIR / "fig2-counterfact-repsim" / "Rep_Sim.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(sim.T.tolist(), f)  # [n_ref, n_train], matches dattri's convention

    return wall_s, setup_s, save_path


# ============================================================================
# dattri.py-backed methods -- Grad_Dot, Grad_Sim, LESS, DataInf, EKFAC
# ============================================================================
def run_dattri(method_key: str):
    from methods.dattri import attribute

    dattri_method = DATTRI_METHOD_NAME[method_key]
    save_dir = RESULTS_DIR / f"fig2-counterfact-{method_key}"
    config = OmegaConf.create({
        "method": dattri_method,
        "task": TASK,
        "subset": SUBSET,
        "base_model_path": BASE_MODEL,
        "checkpoint": CHECKPOINT,
        "device": "cuda",
        "save_path": str(save_dir),
        **DATTRI_EXTRA_PARAMS[method_key],
    })

    t0 = time.perf_counter()
    attribute(config)
    wall_s = time.perf_counter() - t0

    save_path = save_dir / f"{dattri_method}.pt"
    return wall_s, None, save_path


# ============================================================================
# InfluCoder -- setup (teacher grads + distillation) timed SEPARATELY from
# inference (embed+score). wall_s == inference only; setup_s recorded too.
# ============================================================================
def run_influcoder():
    from methods.influcoder import _bootstrap  # noqa: F401
    from methods.influcoder.teacher_grads import (
        compute_gradient_features,
        free_teacher_model,
        load_teacher_model,
        sample_valid_indices,
        tokenize_chat_examples,
    )
    from influcoder.encoder import distill, embed, load_encoder
    from influcoder.metrics import spearman_metrics
    from methods.influcoder.grad_cache import install_grad_cache

    # Teacher gradients are deterministic given the seeded pools + fixed
    # checkpoint, so they are computed once and reused across every later run
    # (sweeps, stability reps, re-timings). setup_s is corrected below so a
    # cached run still reports the true from-scratch cost.
    _grad_stats = install_grad_cache(CHECKPOINT, GRAD_CACHE_DIR)

    def build_clean_external_pool(local_train, local_ref):
        ext = load_dataset(INFLUCODER_EXTERNAL_REPO)["train"]
        local_rows = list(local_train) + list(local_ref)
        local_prompts = set(r["prompt"].strip() for r in local_rows)
        ext_by_prompt = {}
        for i, r in enumerate(ext):
            ext_by_prompt.setdefault(r["prompt"].strip(), []).append(i)
        local_subjects = set()
        n_unmatched = 0
        for r in local_rows:
            cand = ext_by_prompt.get(r["prompt"].strip())
            if not cand:
                n_unmatched += 1
                continue
            for idx in cand:
                local_subjects.add(ext[idx]["subject"].strip().lower())
        assert n_unmatched == 0, f"{n_unmatched} local rows did not join to {INFLUCODER_EXTERNAL_REPO}"
        pool = []
        for r in ext:
            if r["prompt"].strip() in local_prompts:
                continue
            if r["subject"].strip().lower() in local_subjects:
                continue
            pool.append({"prompt": r["prompt"].strip(), "response": r["target_true"].strip()})
        needed = INFLUCODER_N_EXT_ANCHORS + INFLUCODER_N_TEACHER_TRAIN + INFLUCODER_N_EVAL_TRAIN
        assert len(pool) >= needed, f"clean pool too small ({len(pool)} < {needed})"
        return pool

    random.seed(SEED)

    # ---- SETUP: teacher gradients + distillation (measured, saved, NOT counted in wall_s) ----
    # get_dataset() timed IN here (setup, paid once), matching the fairness
    # fix applied to run_bm25()/run_semantic() -- previously excluded from
    # both setup_s and wall_s entirely, an inconsistency vs. Rep-Sim/dattri.
    t_setup0 = time.perf_counter()
    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)

    pool = build_clean_external_pool(local_train, local_ref)
    perm = list(range(len(pool)))
    random.Random(SEED).shuffle(perm)

    tokenizer, teacher = load_teacher_model(BASE_MODEL, CHECKPOINT, device="cuda")
    pool_chat = prepare_chat_format(pool)

    ext_anchor_idx = sample_valid_indices(pool_chat, tokenizer, MAX_LEN, perm, INFLUCODER_N_EXT_ANCHORS)
    teacher_idx = sample_valid_indices(pool_chat, tokenizer, MAX_LEN, perm, INFLUCODER_N_TEACHER_TRAIN,
                                       exclude=set(ext_anchor_idx))
    eval_idx = sample_valid_indices(pool_chat, tokenizer, MAX_LEN, perm, INFLUCODER_N_EVAL_TRAIN,
                                    exclude=set(ext_anchor_idx) | set(teacher_idx))

    anchor_toks = tokenize_chat_examples([pool_chat[i] for i in ext_anchor_idx], tokenizer, MAX_LEN)
    g_anchor = compute_gradient_features(teacher, anchor_toks, proj_dim=INFLUCODER_PROJ_DIM,
                                         device="cuda", desc="anchor grads")
    teacher_toks = tokenize_chat_examples([pool_chat[i] for i in teacher_idx], tokenizer, MAX_LEN)
    g_teacher = compute_gradient_features(teacher, teacher_toks, proj_dim=INFLUCODER_PROJ_DIM,
                                          device="cuda", desc="candidate grads")
    targets = g_anchor @ g_teacher.T

    eval_toks = tokenize_chat_examples([pool_chat[i] for i in eval_idx], tokenizer, MAX_LEN)
    g_eval = compute_gradient_features(teacher, eval_toks, proj_dim=INFLUCODER_PROJ_DIM,
                                       device="cuda", desc="eval grads")
    gt_eval = (g_anchor @ g_eval.T).numpy()
    eval_pool_texts = [example_text(pool[i]) for i in eval_idx]

    free_teacher_model(teacher)

    enc = load_encoder(INFLUCODER_ENCODER_MODEL, max_seq_len=INFLUCODER_ENCODER_MAX_LEN)
    anchor_texts = [example_text(pool[i]) for i in ext_anchor_idx]
    pool_texts = [example_text(pool[i]) for i in teacher_idx]

    def epoch_eval():
        pred = embed(enc, anchor_texts) @ embed(enc, eval_pool_texts).T
        return spearman_metrics(pred, gt_eval)

    # restore_best=False: train a FIXED number of epochs and keep the final
    # weights. Best-epoch restoration (distill's default) argmaxes over a noisy
    # eval metric, so a hair of bf16/GPU floating-point difference flips WHICH
    # epoch wins -- a discrete jump to an entirely different weight set, not a
    # small perturbation. Measured directly: two back-to-back runs of the same
    # config, same seed, same GPU, same cached teacher gradients differed by
    # 0.077 AUPRC (0.4961 vs 0.4188), and the unseeded torch.randperm was ruled
    # out as the cause (the pair ran at hard_ratio=0.0, where randperm is never
    # reached). Fixing the epoch removes the selection step entirely.
    distill(enc, anchor_texts, pool_texts, targets, epochs=INFLUCODER_EPOCHS,
           hard_ratio=INFLUCODER_HARD_RATIO, lr=INFLUCODER_LR, seed=SEED,
           epoch_eval=epoch_eval, select_best_on=INFLUCODER_SELECT_BEST_ON,
           restore_best=False)

    setup_s_this_run = time.perf_counter() - t_setup0
    # Add back what the cache skipped, so setup_s always means "cost from
    # scratch" and is comparable across cached and uncached runs.
    setup_s = setup_s_this_run - _grad_stats["grad_s_this_run"] + _grad_stats["grad_s"]
    EXTRA_RECORD_FIELDS["grad_collection_s"] = _grad_stats["grad_s"]
    EXTRA_RECORD_FIELDS["setup_s_this_run"] = setup_s_this_run

    # ---- INFERENCE: embed+score the full local train/ref (the fair, comparable number) ----
    t_inf0 = time.perf_counter()
    all_train_emb = embed(enc, [example_text(d) for d in local_train])
    all_ref_emb = embed(enc, [example_text(d) for d in local_ref])
    scores = all_train_emb @ all_ref_emb.T
    inference_s = time.perf_counter() - t_inf0

    save_path = RESULTS_DIR / "fig2-counterfact-influcoder" / "InfluCoder.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(scores.T.tolist(), f)

    return inference_s, setup_s, save_path


# ============================================================================
# Semantic baseline -- the SAME encoder InfluCoder distills (ettin-400m), but
# UNTRAINED (straight off the shelf, zero-shot). Isolates how much of
# InfluCoder's result is the distillation itself vs. just having a small,
# cheap encoder to embed with. Structurally single-phase like Rep-Sim (no
# setup/inference split -- there's no training step to amortize, so wall_s
# includes model loading, matching Rep-Sim's convention), NOT two-phase like
# InfluCoder. Expect inference-comparable wall_s to InfluCoder's (same size
# encoder, same embed+score operation) since neither does a backward pass.
# ============================================================================
def run_semantic():
    from methods.influcoder import _bootstrap  # noqa: F401 -- sys.path side effect, must run first
    from influcoder.encoder import embed, load_encoder

    # Timed in two phases mirroring run_influcoder() exactly: setup_s covers
    # get_dataset()+load_encoder(), wall_s covers ONLY embed+score. Semantic is
    # InfluCoder's ablation control -- same encoder architecture, same texts,
    # same embed+matmul, only the weights differ (distilled vs off-the-shelf) --
    # so the two wall_s numbers are comparable only if the timer brackets the
    # identical region. wall_s previously also swallowed get_dataset()+
    # load_encoder(), which InfluCoder's inference timer excludes; on this task
    # that load is ~10s against a ~4.5s inference, so it dominated the number
    # and made an identical-cost encoder look 3x slower than InfluCoder.
    t_setup0 = time.perf_counter()
    local_train, local_ref = get_dataset(TASK, SUBSET)
    enc = load_encoder(INFLUCODER_ENCODER_MODEL, max_seq_len=INFLUCODER_ENCODER_MAX_LEN)
    setup_s = time.perf_counter() - t_setup0

    t_inf0 = time.perf_counter()
    all_train_emb = embed(enc, [example_text(d) for d in local_train])
    all_ref_emb = embed(enc, [example_text(d) for d in local_ref])
    scores = all_train_emb @ all_ref_emb.T
    wall_s = time.perf_counter() - t_inf0

    save_path = RESULTS_DIR / "fig2-counterfact-semantic" / "Semantic.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(scores.T.tolist(), f)

    return wall_s, setup_s, save_path


# ============================================================================
DISPATCH = {
    "bm25": run_bm25,
    "repsim": run_repsim,
    "graddot": lambda: run_dattri("graddot"),
    "gradsim": lambda: run_dattri("gradsim"),
    "less": lambda: run_dattri("less"),
    "datainf": lambda: run_dattri("datainf"),
    "ekfac": lambda: run_dattri("ekfac"),
    "influcoder": run_influcoder,
    "semantic": run_semantic,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(DISPATCH.keys()))
    args = parser.parse_args()

    print(f"=== Fig 2 (Counterfact/Pythia-1b): {args.method} ===")
    wall_s, setup_s, score_path = DISPATCH[args.method]()

    if setup_s is not None:
        print(f"{args.method} setup (NOT in wall_s): {setup_s:.1f}s")
    print(f"{args.method} wall_s (fair/comparable number): {wall_s:.1f}s")

    recall50, mrr = evaluate(score_path)
    print(f"{args.method} Recall@50={recall50:.4f} MRR={mrr:.4f}")

    record = {
        "task": TASK,
        "wall_s": wall_s,
        "recall_at_50": recall50,
        "mrr": mrr,
        "gpu": detect_gpu_label(),
    }
    if setup_s is not None:
        record["setup_s"] = setup_s
        record["setup_note"] = SETUP_NOTES[args.method]

    record.update(EXTRA_RECORD_FIELDS)
    save_summary(args.method, record)


if __name__ == "__main__":
    main()
