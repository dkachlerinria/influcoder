#!/usr/bin/env python3
"""`ettin-encoder-400m` variant of `influcoder_attribute_noleak_counterfact_extquery_bigger.py`
(which found Recall@50=0.4448 at 1800/4000/1000 on the 68m encoder, barely
above the moredata 68m baseline of 0.4411 -- data scale alone doesn't move
Recall@50 much on 68m). Since the encoder-size sweep separately found 400m
beats 68m by +0.02 Recall@50 at the moredata (900/2000/500) scale, this tests
whether that gain compounds with the doubled data scale, independent of the
hard_ratio question tested elsewhere. ONLY CHANGE vs. the 68m `_bigger.py`
script: ENCODER_MODEL. Same 1800/4000/1000 sample counts, HARD_RATIO=0.5,
EPOCHS=8, SEED=0. Pool has 15,648 clean rows; this uses 6,800, still well
within budget. See EXP2.md for the result.

Fully leak-free InfluCoder distillation for DATE-LM's Counterfact / Pythia-1b
task -- same fixes as `influcoder_attribute_noleak_counterfact_kfold.py`, but
instead of 2-fold cross-validating over the local 66 ref queries, this pulls
EXTRA queries from the same external, disjoint CounterFact pool to serve as
distillation-time anchors. The encoder is trained end-to-end on external
anchors x external candidates only, and never sees ANY local data (train OR
ref) until the final embed+score step. That means:

  - No folding/stitching: all 66 local ref queries are scored in one pass by
    a single encoder, directly comparable to the paper's Table 6 number
    (rather than a 2-fold average over two separately-trained encoders).
  - Arguably a better generalization test than the 2-fold version too: the
    encoder sees ~300 distinct practice queries during training instead of
    33, so its "does this candidate matter for this query" mapping has to
    generalize across more query diversity, not just transfer from one
    specific 33-query half to the other.

Data provenance / exclusion logic is identical to the sibling scripts (see
`influcoder_attribute_noleak_counterfact.py`'s docstring for the full
row+subject exclusion argument): local Counterfact/Pythia-1b train+ref
(5,473+66=5,539 rows) join 1:1 by exact `prompt` text to
`NeelNanda/counterfact-tracing` (21,919 rows, the ROME/CounterFact dataset).
Excluding those rows AND any row sharing a `subject` with local train+ref
leaves ~15,648 clean rows -- this script draws N_EXT_ANCHORS (as "queries"),
N_TEACHER_TRAIN (as distillation candidates), and N_EVAL_TRAIN (as a held-out
internal fidelity check) disjoint slices from that pool.

Usage:
    python methods/influcoder_attribute_noleak_counterfact_extquery.py
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

ENCODER_MODEL = "jhu-clsp/ettin-encoder-400m"
ENCODER_MAX_LEN = 1024
MAX_LEN = 1024
PROJ_DIM = 8192
N_EXT_ANCHORS = 1800  # phase-B: 2x the moredata run's 900
N_TEACHER_TRAIN = 4000  # phase-B: 2x the moredata run's 2000
N_EVAL_TRAIN = 1000    # phase-B: 2x the moredata run's 500
EPOCHS = 8             # held fixed -- FINDINGS.md: more epochs is a dead lever
HARD_RATIO = 0.5       # held fixed -- sweep didn't find a confident winner elsewhere
LR = 5e-5
SEED = 0
SELECT_BEST_ON = "aggregated"

SAVE_PATH = Path("results/factual-attribution-influcoder-noleak-extquery-bigger-400m/InfluCoder.pt")


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
    needed = N_EXT_ANCHORS + N_TEACHER_TRAIN + N_EVAL_TRAIN
    assert len(pool) >= needed, f"clean pool too small ({len(pool)} < {needed})"
    return pool


def main():
    random.seed(SEED)

    print(f"InfluCoder (fully leak-free, external-query) / DATE-LM | "
          f"task={TASK} subset={SUBSET} | base_model={BASE_MODEL} checkpoint={CHECKPOINT}\n")

    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)
    print(f"local train={n_train} ref={n_ref} (BOTH untouched by distillation -- "
          f"encoder never sees any local data until the final embed+score step)\n")

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
    print(f"external anchors={len(ext_anchor_idx)} candidates={len(teacher_idx)} "
          f"held-out eval={len(eval_idx)} (all disjoint slices of the same clean pool)")

    print(f"\n########## teacher gradients: external anchors/'queries' (n={len(ext_anchor_idx)}) ##########")
    anchor_toks = tokenize_chat_examples([pool_chat[i] for i in ext_anchor_idx], tokenizer, MAX_LEN)
    g_anchor = compute_gradient_features(teacher, anchor_toks, proj_dim=PROJ_DIM, device="cuda",
                                         desc="external-anchor grads")

    print(f"\n########## teacher gradients: external candidates (n={len(teacher_idx)}) ##########")
    teacher_toks = tokenize_chat_examples([pool_chat[i] for i in teacher_idx], tokenizer, MAX_LEN)
    g_teacher = compute_gradient_features(teacher, teacher_toks, proj_dim=PROJ_DIM, device="cuda",
                                          desc="external-candidate grads")
    targets = g_anchor @ g_teacher.T  # [N_EXT_ANCHORS, N_TEACHER_TRAIN]

    gt_eval = None
    eval_pool_texts = None
    if eval_idx:
        print(f"\n########## teacher gradients: held-out external eval candidates (n={len(eval_idx)}) ##########")
        eval_toks = tokenize_chat_examples([pool_chat[i] for i in eval_idx], tokenizer, MAX_LEN)
        g_eval = compute_gradient_features(teacher, eval_toks, proj_dim=PROJ_DIM, device="cuda",
                                           desc="external eval-candidate grads")
        gt_eval = (g_anchor @ g_eval.T).numpy()
        eval_pool_texts = [example_text(pool[i]) for i in eval_idx]

    free_teacher_model(teacher)

    print(f"\n########## distilling {ENCODER_MODEL} ({EPOCHS} epochs) on EXTERNAL anchors x candidates only ##########")
    enc = load_encoder(ENCODER_MODEL, max_seq_len=ENCODER_MAX_LEN)
    anchor_texts = [example_text(pool[i]) for i in ext_anchor_idx]
    pool_texts = [example_text(pool[i]) for i in teacher_idx]

    epoch_eval = None
    if gt_eval is not None:
        def epoch_eval():
            pred = embed(enc, anchor_texts) @ embed(enc, eval_pool_texts).T
            return spearman_metrics(pred, gt_eval)

    log = distill(enc, anchor_texts, pool_texts, targets, epochs=EPOCHS,
                 hard_ratio=HARD_RATIO, lr=LR, seed=SEED,
                 epoch_eval=epoch_eval, select_best_on=SELECT_BEST_ON)
    if gt_eval is not None and log["epoch_metrics"]:
        restored = log["epoch_metrics"][log["best_epoch"]]
        print(f"  restored (best-epoch={log['best_epoch'] + 1}/{EPOCHS}) fidelity vs held-out "
              f"EXTERNAL teacher gradients: agg rho={restored['aggregated']:+.4f} "
              f"per-anchor rho={restored['per_anchor_mean']:+.4f}")

    print(f"\n########## embedding LOCAL train (n={n_train}) + LOCAL ref (n={n_ref}) -- "
          f"neither ever seen during distillation ##########")
    t0 = time.perf_counter()
    all_train_emb = embed(enc, [example_text(d) for d in local_train])
    all_ref_emb = embed(enc, [example_text(d) for d in local_ref])
    embed_s = time.perf_counter() - t0
    scores = all_train_emb @ all_ref_emb.T
    print(f"  embed+score wall time: {embed_s:.1f}s")

    final_scores = scores.T.tolist()  # matches dattri.py's Counterfact convention

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(final_scores, f)
    print(f"\nwrote {SAVE_PATH}")


if __name__ == "__main__":
    main()
