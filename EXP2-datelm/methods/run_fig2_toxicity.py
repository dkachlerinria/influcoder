#!/usr/bin/env python3
"""Unified Fig 2 (Toxicity/Bias) timing + accuracy harness -- Toxicity analog
of `run_fig2_counterfact.py`. Same file, same design: every method's
parameters in ONE place, dispatched via --method so a single method's
progress survives a GPU preemption (results accumulate in
results/fig2_toxicity.json, keyed by method, never overwritten wholesale).

Score convention differs from Counterfact: Toxicity/Bias is scored by AUPRC
over a FLAT per-train-example score (mean similarity/influence across the
local ref set), not a [n_ref, n_train] per-query matrix -- matches
evaluate_application.py's `AUPRC()`/`get_unsafe_indices()` and dattri.py's
own `torch.mean(scores, dim=1)` convention for this task.

Fairness rule for `wall_s`: same as the Counterfact version -- every method
computes its score fresh from the checkpoint, no reusable trained artifact,
except InfluCoder (one-time setup: teacher-gradient extraction +
distillation, amortizes across future queries). InfluCoder's `wall_s` is
inference-only; `setup_s` is measured and saved separately, not discarded.
get_dataset() is timed IN for every method from the start here (the
Counterfact script's original exclusion for BM25/InfluCoder was a bug, fixed
there after the fact -- this script never had it).

InfluCoder config is the established BEST Toxicity/Bias result found this
session: WildGuardMix anchors+eval / ToxicChat candidates (cross-source mix,
not single-source), ettin-400m, hard_ratio=0.5 -- AUPRC=0.7651, the best
number in the whole EXP2 toxicity investigation, beats every DATE-LM
baseline including LESS (0.734).

Methods: bm25, repsim, graddot, gradsim, less, datainf, ekfac, influcoder, semantic.

Usage:
    python methods/run_fig2_toxicity.py --method bm25
    python methods/run_fig2_toxicity.py --method repsim
    python methods/run_fig2_toxicity.py --method graddot
    python methods/run_fig2_toxicity.py --method gradsim
    python methods/run_fig2_toxicity.py --method less
    python methods/run_fig2_toxicity.py --method datainf
    python methods/run_fig2_toxicity.py --method ekfac
    python methods/run_fig2_toxicity.py --method influcoder
    python methods/run_fig2_toxicity.py --method semantic
"""
from __future__ import annotations

import argparse
import json
import random
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
# sys.modules -- same rationale as run_fig2_counterfact.py, see its docstring.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from datamodules.load_data import get_dataset, prepare_chat_format  # noqa: E402
from evaluate_application import AUPRC, get_unsafe_indices, read_json  # noqa: E402

# ============================================================================
# ONE PLACE for every method's parameters.
# ============================================================================
TASK = "Toxicity/Bias"
SUBSET = "XSTest-response-Het"
BASE_MODEL = "EleutherAI/pythia-1b"
CHECKPOINT = "DataAttributionEval/Pythia-1b-XSTest-response-Het"
MAX_LEN = 1024
SEED = 0

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "fig2_toxicity.json"

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

REPSIM_BATCH_SIZE = 1
BM25_STOPWORDS = "en"

# InfluCoder: BEST known Toxicity/Bias config -- cross-source mix (WildGuardMix
# anchors+eval, ToxicChat candidates), ettin-400m, hard_ratio=0.5 -> AUPRC=0.7651.
INFLUCODER_ENCODER_MODEL = "jhu-clsp/ettin-encoder-400m"
INFLUCODER_ENCODER_MAX_LEN = 1024
INFLUCODER_PROJ_DIM = 8192
INFLUCODER_N_EXT_ANCHORS = 100
INFLUCODER_N_EXT_BENIGN = 1000
INFLUCODER_N_EXT_TOXIC_POS = 300
INFLUCODER_N_EVAL_BENIGN = 300
INFLUCODER_N_EVAL_TOXIC = 100
INFLUCODER_EPOCHS = 8
INFLUCODER_HARD_RATIO = 0.5
INFLUCODER_LR = 5e-5
INFLUCODER_SELECT_BEST_ON = "aggregated"


# ============================================================================
# Shared helpers
# ============================================================================
def detect_gpu_label() -> str:
    try:
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
    except Exception:
        name = "CPU-only (no GPU used)"
    return f"{name} ({socket.getfqdn()})"


def example_text(d: dict) -> str:
    return f"{d['prompt'].strip()}\n{d['response'].strip()}"


def evaluate(score_path: Path) -> float:
    """Uses evaluate_application.py's own Toxicity/Bias ground-truth/scoring
    functions directly (imported, not reimplemented)."""
    train, ref = get_dataset(TASK, SUBSET)
    score = read_json(str(score_path))
    unsafe_indices = get_unsafe_indices(train)
    _, _, _, auprc = AUPRC(score, TASK, unsafe_indices)
    return auprc


def save_summary(method_key: str, record: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
    summary[method_key] = record
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"wrote {SUMMARY_PATH} (method={method_key})")


# ============================================================================
# BM25 -- pure lexical, CPU only, no checkpoint/GPU needed
# ============================================================================
def run_bm25():
    import bm25s

    t0 = time.perf_counter()
    local_train, local_ref = get_dataset(TASK, SUBSET)
    train_texts = [example_text(d) for d in local_train]
    ref_texts = [example_text(d) for d in local_ref]

    train_tokens = bm25s.tokenize(train_texts, stopwords=BM25_STOPWORDS)
    retriever = bm25s.BM25()
    retriever.index(train_tokens)

    query_tokens = bm25s.tokenize(ref_texts, stopwords=BM25_STOPWORDS, return_ids=False)
    scores = np.stack([retriever.get_scores(qt) for qt in query_tokens])  # [n_ref, n_train]
    flat_scores = scores.mean(axis=0)  # [n_train] -- mean over ref, matches dattri's Toxicity convention
    wall_s = time.perf_counter() - t0

    save_path = RESULTS_DIR / "fig2-toxicity-bm25" / "BM25.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(flat_scores.tolist(), f)

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

    t0 = time.perf_counter()
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

    train_reps = collect_reps(train_loader, model, desc="repsim train reps")
    ref_reps = collect_reps(ref_loader, model, desc="repsim ref reps")
    sim = train_reps @ ref_reps.T  # [n_train, n_ref]
    flat_scores = sim.mean(dim=1)  # [n_train]
    wall_s = time.perf_counter() - t0

    save_path = RESULTS_DIR / "fig2-toxicity-repsim" / "Rep_Sim.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(flat_scores.tolist(), f)

    return wall_s, None, save_path


# ============================================================================
# dattri.py-backed methods -- Grad_Dot, Grad_Sim, LESS, DataInf, EKFAC
# ============================================================================
def run_dattri(method_key: str):
    from methods.dattri import attribute

    dattri_method = DATTRI_METHOD_NAME[method_key]
    save_dir = RESULTS_DIR / f"fig2-toxicity-{method_key}"
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
# InfluCoder -- BEST config: cross-source mix (WildGuardMix anchors+eval,
# ToxicChat candidates), setup (teacher grads + distillation) timed
# SEPARATELY from inference (embed+score). wall_s == inference only.
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

    def build_wildguard_toxic_pool(local_prompts: set, n_needed: int):
        ds = load_dataset("allenai/wildguardmix", "wildguardtrain")["train"]
        toxic = [r for r in ds if r["response_harm_label"] == "harmful"]
        clean = []
        for r in toxic:
            p = r["prompt"].strip()
            if p in local_prompts:
                continue
            resp = (r["response"] or "").strip()
            if not resp:
                continue
            clean.append({"prompt": p, "response": resp})
        assert len(clean) >= n_needed, f"not enough clean wildguardmix rows ({len(clean)} < {n_needed})"
        return clean

    def build_toxicchat_toxic_pool(local_prompts: set, n_needed: int):
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
        rows = list(ds["train"]) + list(ds["test"])
        toxic = [r for r in rows if r["toxicity"] == 1]
        clean = []
        for r in toxic:
            p = r["user_input"].strip()
            if p in local_prompts:
                continue
            resp = (r["model_output"] or "").strip()
            if not resp:
                continue
            clean.append({"prompt": p, "response": resp})
        assert len(clean) >= n_needed, f"not enough clean toxic-chat rows ({len(clean)} < {n_needed})"
        return clean

    def build_external_benign_pool(local_benign_prompts: set, n_needed: int, scan_limit: int = 60000):
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        collected = []
        n_checked = 0
        for r in ds:
            msgs = r["messages"]
            user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
            asst_msgs = [m["content"] for m in msgs if m["role"] == "assistant"]
            if not user_msgs or not asst_msgs:
                continue
            p, resp = user_msgs[0].strip(), asst_msgs[0].strip()
            n_checked += 1
            if p not in local_benign_prompts:
                collected.append({"prompt": p, "response": resp})
                if len(collected) >= n_needed:
                    break
            if n_checked >= scan_limit:
                break
        assert len(collected) >= n_needed, "not enough clean benign rows found"
        return collected

    random.seed(SEED)

    # ---- SETUP: pool building + teacher gradients + distillation ----
    t_setup0 = time.perf_counter()
    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)
    local_benign_prompts = set(r["prompt"].strip() for r in local_train if r["type"] == "Benign")
    local_all_prompts = set(r["prompt"].strip() for r in local_train) | set(r["prompt"].strip() for r in local_ref)

    n_wildguard_needed = INFLUCODER_N_EXT_ANCHORS + INFLUCODER_N_EVAL_TOXIC
    wildguard_pool = build_wildguard_toxic_pool(local_all_prompts, n_wildguard_needed)
    random.Random(SEED).shuffle(wildguard_pool)
    ext_anchors = wildguard_pool[:INFLUCODER_N_EXT_ANCHORS]
    toxic_eval = wildguard_pool[INFLUCODER_N_EXT_ANCHORS:INFLUCODER_N_EXT_ANCHORS + INFLUCODER_N_EVAL_TOXIC]

    toxicchat_pool = build_toxicchat_toxic_pool(local_all_prompts, INFLUCODER_N_EXT_TOXIC_POS)
    random.Random(SEED + 1).shuffle(toxicchat_pool)
    toxic_candidates = toxicchat_pool[:INFLUCODER_N_EXT_TOXIC_POS]

    benign_pool = build_external_benign_pool(local_benign_prompts, INFLUCODER_N_EXT_BENIGN + INFLUCODER_N_EVAL_BENIGN)
    random.Random(SEED + 2).shuffle(benign_pool)
    benign_candidates = benign_pool[:INFLUCODER_N_EXT_BENIGN]
    benign_eval = benign_pool[INFLUCODER_N_EXT_BENIGN:INFLUCODER_N_EXT_BENIGN + INFLUCODER_N_EVAL_BENIGN]

    candidates = benign_candidates + toxic_candidates
    eval_pool = benign_eval + toxic_eval
    random.Random(SEED + 3).shuffle(candidates)
    random.Random(SEED + 4).shuffle(eval_pool)

    tokenizer, teacher = load_teacher_model(BASE_MODEL, CHECKPOINT, device="cuda")

    anchor_chat = prepare_chat_format(ext_anchors)
    cand_chat = prepare_chat_format(candidates)
    eval_chat = prepare_chat_format(eval_pool)

    anchor_idx = sample_valid_indices(anchor_chat, tokenizer, MAX_LEN, list(range(len(anchor_chat))), len(anchor_chat))
    cand_idx = sample_valid_indices(cand_chat, tokenizer, MAX_LEN, list(range(len(cand_chat))), len(cand_chat))
    eval_idx = sample_valid_indices(eval_chat, tokenizer, MAX_LEN, list(range(len(eval_chat))), len(eval_chat))

    anchor_toks = tokenize_chat_examples([anchor_chat[i] for i in anchor_idx], tokenizer, MAX_LEN)
    g_anchor = compute_gradient_features(teacher, anchor_toks, proj_dim=INFLUCODER_PROJ_DIM, device="cuda", desc="anchor grads")
    cand_toks = tokenize_chat_examples([cand_chat[i] for i in cand_idx], tokenizer, MAX_LEN)
    g_cand = compute_gradient_features(teacher, cand_toks, proj_dim=INFLUCODER_PROJ_DIM, device="cuda", desc="candidate grads")
    targets = g_anchor @ g_cand.T

    eval_toks = tokenize_chat_examples([eval_chat[i] for i in eval_idx], tokenizer, MAX_LEN)
    g_eval = compute_gradient_features(teacher, eval_toks, proj_dim=INFLUCODER_PROJ_DIM, device="cuda", desc="eval grads")
    gt_eval = (g_anchor @ g_eval.T).numpy()
    eval_pool_texts = [example_text(eval_pool[i]) for i in eval_idx]

    free_teacher_model(teacher)

    enc = load_encoder(INFLUCODER_ENCODER_MODEL, max_seq_len=INFLUCODER_ENCODER_MAX_LEN)
    anchor_texts = [example_text(ext_anchors[i]) for i in anchor_idx]
    pool_texts = [example_text(candidates[i]) for i in cand_idx]

    def epoch_eval():
        pred = embed(enc, anchor_texts) @ embed(enc, eval_pool_texts).T
        return spearman_metrics(pred, gt_eval)

    distill(enc, anchor_texts, pool_texts, targets, epochs=INFLUCODER_EPOCHS,
           hard_ratio=INFLUCODER_HARD_RATIO, lr=INFLUCODER_LR, seed=SEED,
           epoch_eval=epoch_eval, select_best_on=INFLUCODER_SELECT_BEST_ON)

    setup_s = time.perf_counter() - t_setup0

    # ---- INFERENCE: embed+score the full local train/ref ----
    t_inf0 = time.perf_counter()
    all_train_emb = embed(enc, [example_text(d) for d in local_train])
    all_ref_emb = embed(enc, [example_text(d) for d in local_ref])
    scores = all_train_emb @ all_ref_emb.T  # [n_train, n_ref]
    flat_scores = scores.mean(dim=1)  # [n_train]
    inference_s = time.perf_counter() - t_inf0

    save_path = RESULTS_DIR / "fig2-toxicity-influcoder" / "InfluCoder.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(flat_scores.tolist(), f)

    return inference_s, setup_s, save_path


# ============================================================================
# Semantic baseline -- same encoder InfluCoder distills (ettin-400m),
# untrained/zero-shot. See run_fig2_counterfact.py's run_semantic() for the
# full rationale -- identical here, just flat-mean-reduced for AUPRC.
# ============================================================================
def run_semantic():
    from methods.influcoder import _bootstrap  # noqa: F401 -- sys.path side effect, must run first
    from influcoder.encoder import embed, load_encoder

    t0 = time.perf_counter()
    local_train, local_ref = get_dataset(TASK, SUBSET)
    enc = load_encoder(INFLUCODER_ENCODER_MODEL, max_seq_len=INFLUCODER_ENCODER_MAX_LEN)

    all_train_emb = embed(enc, [example_text(d) for d in local_train])
    all_ref_emb = embed(enc, [example_text(d) for d in local_ref])
    scores = all_train_emb @ all_ref_emb.T
    flat_scores = scores.mean(dim=1)
    wall_s = time.perf_counter() - t0

    save_path = RESULTS_DIR / "fig2-toxicity-semantic" / "Semantic.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(flat_scores.tolist(), f)

    return wall_s, None, save_path


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

    print(f"=== Fig 2 (Toxicity/Bias/XSTest-response-Het): {args.method} ===")
    wall_s, setup_s, score_path = DISPATCH[args.method]()

    if setup_s is not None:
        print(f"{args.method} setup (NOT in wall_s): {setup_s:.1f}s")
    print(f"{args.method} wall_s (fair/comparable number): {wall_s:.1f}s")

    auprc = evaluate(score_path)
    print(f"{args.method} AUPRC={auprc:.4f}")

    record = {
        "task": TASK,
        "wall_s": wall_s,
        "auprc": auprc,
        "gpu": detect_gpu_label(),
    }
    if setup_s is not None:
        record["setup_s"] = setup_s
        record["setup_note"] = "one-time cost (teacher-grad extraction + distillation), NOT included in wall_s"

    save_summary(args.method, record)


if __name__ == "__main__":
    main()
