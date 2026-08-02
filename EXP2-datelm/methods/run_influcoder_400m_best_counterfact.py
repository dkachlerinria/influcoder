#!/usr/bin/env python3
"""Timed, single-config run of InfluCoder's best Counterfact result -- ettin-
encoder-400m, moredata scale (900 anchors/2000 candidates/500 held-out eval
from NeelNanda/counterfact-tracing), HARD_RATIO=0.25 -- the exact config that
scored Recall@50=0.4689/MRR=0.8739
(`influcoder_attribute_noleak_counterfact_extquery_moredata_400m_hardsweep.py`'s
0.25 branch). That script sweeps hard_ratio in [0.25, 0.75] in one process, so
its wall-clock isn't a clean "just the winning config" number; this is the
same logic with only hard_ratio=0.25 run, for a real, isolated timing
measurement -- deterministic (same SEED=0, same data draw), so the result
score file should reproduce the existing 0.4689/0.8739 exactly, which also
serves as a re-verification.

Same base checkpoint (`DataAttributionEval/Pythia-1b-counterfactual`) as every
other Counterfact run in this doc -- "the checkpoint from our best result"
IS this one, already used throughout; nothing to swap in.

Usage:
    python methods/run_influcoder_400m_best_counterfact.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from datasets import load_dataset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from datamodules.load_data import get_dataset, prepare_chat_format  # noqa: E402

from methods.influcoder import _bootstrap  # noqa: F401,E402
from methods.influcoder.teacher_grads import (  # noqa: E402
    compute_gradient_features,
    free_teacher_model,
    load_teacher_model,
    sample_valid_indices,
    tokenize_chat_examples,
)
from influcoder.encoder import distill, embed, load_encoder  # noqa: E402
from influcoder.metrics import spearman_metrics  # noqa: E402

TASK = "Counterfact"
SUBSET = "Pythia-1b"
BASE_MODEL = "EleutherAI/pythia-1b"
CHECKPOINT = "DataAttributionEval/Pythia-1b-counterfactual"
EXTERNAL_REPO = "NeelNanda/counterfact-tracing"

ENCODER_MODEL = "jhu-clsp/ettin-encoder-400m"
ENCODER_MAX_LEN = 1024
MAX_LEN = 1024
PROJ_DIM = 8192
N_EXT_ANCHORS = 900
N_TEACHER_TRAIN = 2000
N_EVAL_TRAIN = 500
EPOCHS = 8
HARD_RATIO = 0.25
LR = 5e-5
SEED = 0
SELECT_BEST_ON = "aggregated"

SAVE_PATH = Path("results/factual-attribution-influcoder-noleak-extquery-moredata-400m-hard025-timed/InfluCoder.pt")


def example_text(d: dict) -> str:
    return f"{d['prompt'].strip()}\n{d['response'].strip()}"


def build_clean_external_pool(local_train, local_ref):
    print(f"loading external pool: {EXTERNAL_REPO} ...")
    ext = load_dataset(EXTERNAL_REPO)["train"]
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
    assert n_unmatched == 0
    pool = []
    for r in ext:
        if r["prompt"].strip() in local_prompts:
            continue
        if r["subject"].strip().lower() in local_subjects:
            continue
        pool.append({"prompt": r["prompt"].strip(), "response": r["target_true"].strip()})
    needed = N_EXT_ANCHORS + N_TEACHER_TRAIN + N_EVAL_TRAIN
    assert len(pool) >= needed
    return pool


def main():
    t_start = time.perf_counter()
    random.seed(SEED)
    print(f"InfluCoder 400m (BEST config, hard_ratio={HARD_RATIO}) / DATE-LM | "
          f"task={TASK} subset={SUBSET} | base_model={BASE_MODEL} checkpoint={CHECKPOINT}\n")

    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)

    pool = build_clean_external_pool(local_train, local_ref)
    perm = list(range(len(pool)))
    random.Random(SEED).shuffle(perm)

    tokenizer, teacher = load_teacher_model(BASE_MODEL, CHECKPOINT, device="cuda")
    pool_chat = prepare_chat_format(pool)

    ext_anchor_idx = sample_valid_indices(pool_chat, tokenizer, MAX_LEN, perm, N_EXT_ANCHORS)
    teacher_idx = sample_valid_indices(pool_chat, tokenizer, MAX_LEN, perm, N_TEACHER_TRAIN,
                                       exclude=set(ext_anchor_idx))
    eval_idx = sample_valid_indices(pool_chat, tokenizer, MAX_LEN, perm, N_EVAL_TRAIN,
                                    exclude=set(ext_anchor_idx) | set(teacher_idx))

    anchor_toks = tokenize_chat_examples([pool_chat[i] for i in ext_anchor_idx], tokenizer, MAX_LEN)
    g_anchor = compute_gradient_features(teacher, anchor_toks, proj_dim=PROJ_DIM, device="cuda", desc="anchor grads")
    teacher_toks = tokenize_chat_examples([pool_chat[i] for i in teacher_idx], tokenizer, MAX_LEN)
    g_teacher = compute_gradient_features(teacher, teacher_toks, proj_dim=PROJ_DIM, device="cuda", desc="candidate grads")
    targets = g_anchor @ g_teacher.T

    eval_toks = tokenize_chat_examples([pool_chat[i] for i in eval_idx], tokenizer, MAX_LEN)
    g_eval = compute_gradient_features(teacher, eval_toks, proj_dim=PROJ_DIM, device="cuda", desc="eval grads")
    gt_eval = (g_anchor @ g_eval.T).numpy()
    eval_pool_texts = [example_text(pool[i]) for i in eval_idx]

    free_teacher_model(teacher)

    print(f"\ndistilling {ENCODER_MODEL} ({EPOCHS} epochs, hard_ratio={HARD_RATIO})")
    enc = load_encoder(ENCODER_MODEL, max_seq_len=ENCODER_MAX_LEN)
    anchor_texts = [example_text(pool[i]) for i in ext_anchor_idx]
    pool_texts = [example_text(pool[i]) for i in teacher_idx]

    def epoch_eval():
        pred = embed(enc, anchor_texts) @ embed(enc, eval_pool_texts).T
        return spearman_metrics(pred, gt_eval)

    log = distill(enc, anchor_texts, pool_texts, targets, epochs=EPOCHS,
                 hard_ratio=HARD_RATIO, lr=LR, seed=SEED,
                 epoch_eval=epoch_eval, select_best_on=SELECT_BEST_ON)
    restored = log["epoch_metrics"][log["best_epoch"]]
    print(f"  restored (best-epoch={log['best_epoch'] + 1}/{EPOCHS}) fidelity: "
          f"agg rho={restored['aggregated']:+.4f}")

    t_embed0 = time.perf_counter()
    all_train_emb = embed(enc, [example_text(d) for d in local_train])
    all_ref_emb = embed(enc, [example_text(d) for d in local_ref])
    embed_s = time.perf_counter() - t_embed0
    scores = all_train_emb @ all_ref_emb.T
    print(f"  embed+score wall time: {embed_s:.1f}s")

    final_scores = scores.T.tolist()
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(final_scores, f)

    total_s = time.perf_counter() - t_start
    print(f"\nwrote {SAVE_PATH}")
    print(f"TOTAL wall time (model load + teacher grads + distill + embed+score + save): {total_s:.1f}s")


if __name__ == "__main__":
    main()
