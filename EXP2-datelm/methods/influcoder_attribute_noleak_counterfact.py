#!/usr/bin/env python3
"""Leak-free InfluCoder distillation for DATE-LM's Counterfact / Pythia-1b task.

`methods/influcoder_attribute.py` distills InfluCoder's encoder using teacher
gradients computed on a 500-example SUBSET OF DATE-LM's OWN `train` split, then
scores/evaluates that same `train` split (all 5,473 examples, including those
500). That's leakage: the encoder is directly fit to reproduce the true
gradient-similarity of those 500 examples against the exact `ref` set the task
evaluates against, then those same 500 examples are included in the Recall@50/
MRR computation. Measured overlap: 100/1007 (9.9%) of the ground-truth
fact-matches for the 66 ref queries fall inside that 500-example fit set, and
all 66 ref queries have >=1 of their matches inside it.

This script distills entirely on EXTERNAL data instead: DATE-LM's local
Counterfact/Pythia-1b split (5,473 train examples) is confirmed (by exact
`prompt` text join, see below) to be a verbatim subset of the public
`NeelNanda/counterfact-tracing` dataset (21,919 rows total -- the ROME/
CounterFact dataset, Meng et al. 2022, NeurIPS). We build a clean pool by
excluding from that 21,919:
  (a) any row whose `prompt` exactly matches a local train OR ref row
      (all 5,539 local rows join 1:1 -- confirmed empirically), and
  (b) any row whose `subject` matches ANY subject appearing in local train
      OR ref (5,421 distinct subjects) -- so even a *different* fact about
      the same real-world entity being evaluated is excluded, not just the
      identical fact instance.
This leaves ~15,648 clean rows -- comfortably enough for the same
n_teacher_train=500 / n_eval_train=250 split InfluCoder normally uses.

The encoder is then distilled against REAL teacher gradients (same
checkpoint, `DataAttributionEval/Pythia-1b-counterfactual`) computed on this
external pool, and evaluated by embedding DATE-LM's ORIGINAL, untouched
train+ref split -- so the final saved scores go through
`evaluation/evaluate_application.py` (DATE-LM's native scoring code)
unchanged, directly comparable to the paper's Table 6 Pythia-1B numbers.

Usage:
    python methods/influcoder_attribute_noleak_counterfact.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
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

ENCODER_MODEL = "jhu-clsp/ettin-encoder-68m"
ENCODER_MAX_LEN = 1024
MAX_LEN = 1024
PROJ_DIM = 8192
N_TEACHER_TRAIN = 500
N_EVAL_TRAIN = 250
EPOCHS = 8
HARD_RATIO = 0.5
LR = 5e-5
SEED = 0
SELECT_BEST_ON = "aggregated"

SAVE_PATH = Path("results/factual-attribution-influcoder-noleak/InfluCoder.pt")


def example_text(d: dict) -> str:
    return f"{d['prompt'].strip()}\n{d['response'].strip()}"


def build_clean_external_pool(local_train, local_ref):
    """External CounterFact rows with zero row- or subject-level overlap
    with DATE-LM's local train+ref, converted to the local {prompt,response}
    schema so they tokenize identically to local examples."""
    print(f"loading external pool: {EXTERNAL_REPO} ...")
    ext = load_dataset(EXTERNAL_REPO)["train"]

    local_rows = list(local_train) + list(local_ref)
    local_prompts = set(r["prompt"].strip() for r in local_rows)

    # Recover each local row's `subject` by exact-prompt join against the
    # external dataset (local rows don't carry a `subject` field of their own).
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
    assert n_unmatched == 0, (
        f"{n_unmatched} local rows did not join to the external dataset by exact "
        "prompt text -- the disjointness guarantee below depends on this join "
        "being complete, refusing to proceed with a partial join"
    )
    print(f"  local rows: {len(local_rows)}, joined subjects: {len(local_subjects)}")

    pool = []
    excl_prompt = excl_subject = 0
    for r in ext:
        if r["prompt"].strip() in local_prompts:
            excl_prompt += 1
            continue
        if r["subject"].strip().lower() in local_subjects:
            excl_subject += 1
            continue
        pool.append({"prompt": r["prompt"].strip(), "response": r["target_true"].strip()})

    print(f"  external total: {len(ext)}, excluded (exact prompt): {excl_prompt}, "
          f"excluded (subject overlap): {excl_subject}, clean pool: {len(pool)}")
    assert len(pool) >= N_TEACHER_TRAIN + N_EVAL_TRAIN, "clean pool too small"
    return pool


def main():
    random.seed(SEED)

    print(f"InfluCoder (leak-free) / DATE-LM | task={TASK} subset={SUBSET} "
          f"| base_model={BASE_MODEL} checkpoint={CHECKPOINT}\n")

    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)
    print(f"local train={n_train} ref={n_ref} (untouched by distillation, scored at the end)\n")

    pool = build_clean_external_pool(local_train, local_ref)
    perm = list(range(len(pool)))
    random.Random(SEED).shuffle(perm)

    tokenizer, teacher = load_teacher_model(BASE_MODEL, CHECKPOINT, device="cuda")

    local_ref_chat = prepare_chat_format(local_ref)
    ref_idx = sample_valid_indices(local_ref_chat, tokenizer, MAX_LEN, list(range(n_ref)), n_ref)
    if len(ref_idx) < n_ref:
        print(f"  [ref] skipped {n_ref - len(ref_idx)}/{n_ref} examples with no valid loss span")

    pool_chat = prepare_chat_format(pool)
    teacher_idx = sample_valid_indices(pool_chat, tokenizer, MAX_LEN, perm, N_TEACHER_TRAIN)
    eval_idx = sample_valid_indices(pool_chat, tokenizer, MAX_LEN, perm, N_EVAL_TRAIN, exclude=set(teacher_idx))

    print(f"\n########## teacher gradients: ref (n={len(ref_idx)}) ##########")
    ref_toks = tokenize_chat_examples([local_ref_chat[i] for i in ref_idx], tokenizer, MAX_LEN)
    g_ref = compute_gradient_features(teacher, ref_toks, proj_dim=PROJ_DIM, device="cuda", desc="ref grads")

    print(f"\n########## teacher gradients: EXTERNAL pool subset (n={len(teacher_idx)}) ##########")
    teacher_toks = tokenize_chat_examples([pool_chat[i] for i in teacher_idx], tokenizer, MAX_LEN)
    g_teacher = compute_gradient_features(teacher, teacher_toks, proj_dim=PROJ_DIM, device="cuda",
                                          desc="external-pool grads")
    targets = g_ref @ g_teacher.T  # [n_ref, N_TEACHER_TRAIN] -- anchors=local ref, pool=EXTERNAL examples

    eval_anchor_texts = eval_pool_texts = gt_eval = None
    if eval_idx:
        print(f"\n########## teacher gradients: held-out EXTERNAL eval subset (n={len(eval_idx)}) ##########")
        eval_toks = tokenize_chat_examples([pool_chat[i] for i in eval_idx], tokenizer, MAX_LEN)
        g_eval = compute_gradient_features(teacher, eval_toks, proj_dim=PROJ_DIM, device="cuda",
                                           desc="external-pool eval grads")
        gt_eval = (g_ref @ g_eval.T).numpy()
        eval_anchor_texts = [example_text(local_ref[i]) for i in ref_idx]
        eval_pool_texts = [example_text(pool[i]) for i in eval_idx]

    free_teacher_model(teacher)

    print(f"\n########## distilling {ENCODER_MODEL} ({EPOCHS} epochs) on EXTERNAL data only ##########")
    enc = load_encoder(ENCODER_MODEL, max_seq_len=ENCODER_MAX_LEN)
    anchor_texts = [example_text(local_ref[i]) for i in ref_idx]
    pool_texts = [example_text(pool[i]) for i in teacher_idx]

    epoch_eval = None
    if gt_eval is not None:
        def epoch_eval():
            pred = embed(enc, eval_anchor_texts) @ embed(enc, eval_pool_texts).T
            return spearman_metrics(pred, gt_eval)

    log = distill(enc, anchor_texts, pool_texts, targets, epochs=EPOCHS,
                 hard_ratio=HARD_RATIO, lr=LR, seed=SEED,
                 epoch_eval=epoch_eval, select_best_on=SELECT_BEST_ON)
    if gt_eval is not None and log["epoch_metrics"]:
        restored = log["epoch_metrics"][log["best_epoch"]]
        print(f"  restored (best-epoch={log['best_epoch'] + 1}/{EPOCHS}) fidelity vs held-out "
              f"EXTERNAL teacher gradients: agg rho={restored['aggregated']:+.4f} "
              f"per-anchor rho={restored['per_anchor_mean']:+.4f}")

    print(f"\n########## embedding LOCAL train (n={n_train}) + ref (n={n_ref}) -- never seen in distillation ##########")
    t0 = time.perf_counter()
    all_train_emb = embed(enc, [example_text(d) for d in local_train])
    all_ref_emb = embed(enc, [example_text(d) for d in local_ref])
    embed_s = time.perf_counter() - t0
    scores = all_train_emb @ all_ref_emb.T
    print(f"  embed+score wall time: {embed_s:.1f}s")

    final_scores = scores.T.tolist()  # matches dattri.py's Counterfact convention

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(SAVE_PATH, "w") as f:
        json.dump(final_scores, f)
    print(f"\nwrote {SAVE_PATH}")


if __name__ == "__main__":
    main()
