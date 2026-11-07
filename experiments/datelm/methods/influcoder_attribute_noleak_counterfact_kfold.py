#!/usr/bin/env python3
"""Fully leak-free InfluCoder distillation for DATE-LM's Counterfact / Pythia-1b
task -- fixes BOTH leaks found in `methods/influcoder_attribute.py`:

  1. TRAIN-side leak (fixed the same way as
     `influcoder_attribute_noleak_counterfact.py`): distillation targets come
     from an external, disjoint CounterFact pool (`NeelNanda/counterfact-tracing`,
     21,919 rows minus any row/subject overlap with DATE-LM's local train+ref --
     see that script's docstring for the exact exclusion logic), never from
     DATE-LM's own scored `train` split.

  2. REF-side leak (new fix, this script): the base recipe distills the
     encoder to reproduce true gradient-similarity for the *exact* 66 ref
     queries it is later scored against -- the encoder gets to specialize to
     the precise queries it's graded on, an advantage no other DATE-LM method
     (Grad_Dot/Grad_Sim/LESS/DataInf/EKFAC) has, since those compute exact
     gradients fresh for any query with no fitting step at all. It's also
     inconsistent with EXP1's own convention of disjoint train/eval anchor
     sets. Fixed via 2-fold cross-validation over the 66 ref queries: for each
     fold, distill an encoder using the OTHER fold's ref queries as anchors
     (plus the external pool), then score/evaluate ONLY the held-out fold
     with that fold's encoder. Every query is scored by a model that never
     saw it during distillation. The two folds' held-out scores are stitched
     back into a single [n_train, 66] matrix in original ref order, so the
     final saved file has the same shape DATE-LM's native
     `evaluation/evaluate_application.py` expects -- directly comparable to
     the paper's Table 6 Pythia-1B numbers.

The external pool's teacher gradients (`g_teacher`/`g_eval`, computed against
the *complementary* ref fold each time... actually: pool-side gradients don't
depend on which ref fold is being used as anchors, only the anchor-side
(ref) gradients change per fold -- so the pool's teacher/eval gradients are
computed ONCE and reused across both folds, only the ref-anchor gradient
computation (on ~33 examples) and the encoder distillation are repeated.

Usage:
    python methods/influcoder_attribute_noleak_counterfact_kfold.py
"""
from __future__ import annotations

import json
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
N_FOLDS = 2

SAVE_PATH = Path("results/factual-attribution-influcoder-noleak-2fold/InfluCoder.pt")


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
    assert n_unmatched == 0, (
        f"{n_unmatched} local rows did not join to the external dataset by exact "
        "prompt text -- refusing to proceed with a partial join"
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

    print(f"InfluCoder (fully leak-free, {N_FOLDS}-fold ref CV) / DATE-LM | "
          f"task={TASK} subset={SUBSET} | base_model={BASE_MODEL} checkpoint={CHECKPOINT}\n")

    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)
    print(f"local train={n_train} ref={n_ref}\n")

    pool = build_clean_external_pool(local_train, local_ref)
    perm = list(range(len(pool)))
    random.Random(SEED).shuffle(perm)

    ref_order = list(range(n_ref))
    random.Random(SEED + 1).shuffle(ref_order)
    fold_of = {}
    for i, ref_i in enumerate(ref_order):
        fold_of[ref_i] = i % N_FOLDS
    folds = [[i for i in range(n_ref) if fold_of[i] == f] for f in range(N_FOLDS)]
    for f, idxs in enumerate(folds):
        print(f"  fold {f}: {len(idxs)} ref queries -> {idxs}")

    tokenizer, teacher = load_teacher_model(BASE_MODEL, CHECKPOINT, device="cuda")

    local_ref_chat = prepare_chat_format(local_ref)
    pool_chat = prepare_chat_format(pool)

    teacher_idx = sample_valid_indices(pool_chat, tokenizer, MAX_LEN, perm, N_TEACHER_TRAIN)
    eval_idx = sample_valid_indices(pool_chat, tokenizer, MAX_LEN, perm, N_EVAL_TRAIN, exclude=set(teacher_idx))

    print(f"\n########## teacher gradients: EXTERNAL pool subset (n={len(teacher_idx)}) -- shared across folds ##########")
    teacher_toks = tokenize_chat_examples([pool_chat[i] for i in teacher_idx], tokenizer, MAX_LEN)
    g_teacher = compute_gradient_features(teacher, teacher_toks, proj_dim=PROJ_DIM, device="cuda",
                                          desc="external-pool grads")

    g_eval = None
    eval_pool_texts = None
    if eval_idx:
        print(f"\n########## teacher gradients: held-out EXTERNAL eval subset (n={len(eval_idx)}) -- shared across folds ##########")
        eval_toks = tokenize_chat_examples([pool_chat[i] for i in eval_idx], tokenizer, MAX_LEN)
        g_eval = compute_gradient_features(teacher, eval_toks, proj_dim=PROJ_DIM, device="cuda",
                                           desc="external-pool eval grads")
        eval_pool_texts = [example_text(pool[i]) for i in eval_idx]

    pool_texts = [example_text(pool[i]) for i in teacher_idx]

    all_train_emb_cache = {}  # keyed by encoder id -> avoid re-embedding train per fold if reused (it isn't, but keep simple)
    full_scores = np.zeros((n_train, n_ref), dtype=np.float64)

    for f in range(N_FOLDS):
        held_out = folds[f]
        anchor_idx = [i for i in range(n_ref) if i not in held_out]
        print(f"\n===================== FOLD {f}: distill on {len(anchor_idx)} anchors, "
              f"score held-out {len(held_out)} queries =====================")

        anchor_ref_idx = sample_valid_indices(local_ref_chat, tokenizer, MAX_LEN, anchor_idx, len(anchor_idx))
        if len(anchor_ref_idx) < len(anchor_idx):
            print(f"  [fold {f} anchors] skipped {len(anchor_idx) - len(anchor_ref_idx)} with no valid loss span")

        print(f"  teacher gradients: fold-{f} anchor ref (n={len(anchor_ref_idx)})")
        anchor_toks = tokenize_chat_examples([local_ref_chat[i] for i in anchor_ref_idx], tokenizer, MAX_LEN)
        g_anchor = compute_gradient_features(teacher, anchor_toks, proj_dim=PROJ_DIM, device="cuda",
                                             desc=f"fold-{f} anchor-ref grads")
        targets = g_anchor @ g_teacher.T  # [len(anchor_ref_idx), N_TEACHER_TRAIN]

        gt_eval = None
        anchor_texts = [example_text(local_ref[i]) for i in anchor_ref_idx]
        if g_eval is not None:
            gt_eval = (g_anchor @ g_eval.T).numpy()

        print(f"  distilling {ENCODER_MODEL} ({EPOCHS} epochs) for fold {f} ...")
        enc = load_encoder(ENCODER_MODEL, max_seq_len=ENCODER_MAX_LEN)

        epoch_eval = None
        if gt_eval is not None:
            def epoch_eval(_anchor_texts=anchor_texts, _eval_pool_texts=eval_pool_texts, _gt_eval=gt_eval):
                pred = embed(enc, _anchor_texts) @ embed(enc, _eval_pool_texts).T
                return spearman_metrics(pred, _gt_eval)

        log = distill(enc, anchor_texts, pool_texts, targets, epochs=EPOCHS,
                     hard_ratio=HARD_RATIO, lr=LR, seed=SEED,
                     epoch_eval=epoch_eval, select_best_on=SELECT_BEST_ON)
        if gt_eval is not None and log["epoch_metrics"]:
            restored = log["epoch_metrics"][log["best_epoch"]]
            print(f"  fold {f} restored (best-epoch={log['best_epoch'] + 1}/{EPOCHS}) fidelity: "
                  f"agg rho={restored['aggregated']:+.4f} per-anchor rho={restored['per_anchor_mean']:+.4f}")

        print(f"  embedding LOCAL train (n={n_train}) + fold-{f} held-out ref (n={len(held_out)})")
        t0 = time.perf_counter()
        train_emb = embed(enc, [example_text(d) for d in local_train])
        held_out_ref_texts = [example_text(local_ref[i]) for i in held_out]
        held_out_emb = embed(enc, held_out_ref_texts)
        embed_s = time.perf_counter() - t0
        fold_scores = train_emb @ held_out_emb.T  # [n_train, len(held_out)]
        print(f"  embed+score wall time: {embed_s:.1f}s")

        for col, ref_i in enumerate(held_out):
            full_scores[:, ref_i] = fold_scores[:, col]

        del enc
        torch.cuda.empty_cache()

    free_teacher_model(teacher)

    final_scores = full_scores.T.tolist()  # [n_ref, n_train], matches dattri.py's Counterfact convention

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(final_scores, f)
    print(f"\nwrote {SAVE_PATH}")


if __name__ == "__main__":
    main()
