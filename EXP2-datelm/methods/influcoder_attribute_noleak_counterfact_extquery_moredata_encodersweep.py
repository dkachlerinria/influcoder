#!/usr/bin/env python3
"""Encoder-size sweep on top of the moredata Counterfact config
(`influcoder_attribute_noleak_counterfact_extquery_moredata.py`) -- IDENTICAL
data provenance/exclusion/sample-count logic (900 anchors / 2000 candidates /
500 held-out eval from the same clean `NeelNanda/counterfact-tracing` pool,
HARD_RATIO=0.5, EPOCHS=8), the only thing that varies is ENCODER_MODEL.

The teacher (Pythia-1B checkpoint) gradient computation for the anchor/
candidate/eval sets does not depend on which student encoder is being
distilled into, so it's computed ONCE and reused across every encoder size in
SIZES below (per FINDINGS.md's documented time-saver for exactly this kind of
sweep) -- only the load_encoder -> distill -> embed+score step repeats per
size. The existing 68m result (`influcoder_attribute_noleak_counterfact_extquery_moredata.py`,
Recall@50=0.4411, MRR=0.8226) is NOT recomputed here; SIZES below covers only
the new sizes being tested.

Usage:
    python methods/influcoder_attribute_noleak_counterfact_extquery_moredata_encodersweep.py
"""
from __future__ import annotations

import gc
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

ENCODER_MAX_LEN = 1024
MAX_LEN = 1024
PROJ_DIM = 8192
N_EXT_ANCHORS = 900
N_TEACHER_TRAIN = 2000
N_EVAL_TRAIN = 500
EPOCHS = 8
HARD_RATIO = 0.5
LR = 5e-5
SEED = 0
SELECT_BEST_ON = "aggregated"

SIZES = [
    ("150m", "jhu-clsp/ettin-encoder-150m"),
    ("400m", "jhu-clsp/ettin-encoder-400m"),
    ("1b", "jhu-clsp/ettin-encoder-1b"),
]


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

    print(f"InfluCoder encoder-size sweep / DATE-LM | task={TASK} subset={SUBSET} | "
          f"base_model={BASE_MODEL} checkpoint={CHECKPOINT}\n")

    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)
    print(f"local train={n_train} ref={n_ref} (untouched by distillation)\n")

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
          f"held-out eval={len(eval_idx)} (all disjoint slices of the same clean pool, "
          f"IDENTICAL selection to the 68m moredata run -- same SEED, same pool construction)")

    print(f"\n########## teacher gradients: external anchors/'queries' (n={len(ext_anchor_idx)}) "
          f"[computed ONCE, reused for every encoder size below] ##########")
    anchor_toks = tokenize_chat_examples([pool_chat[i] for i in ext_anchor_idx], tokenizer, MAX_LEN)
    g_anchor = compute_gradient_features(teacher, anchor_toks, proj_dim=PROJ_DIM, device="cuda",
                                         desc="external-anchor grads")

    print(f"\n########## teacher gradients: external candidates (n={len(teacher_idx)}) ##########")
    teacher_toks = tokenize_chat_examples([pool_chat[i] for i in teacher_idx], tokenizer, MAX_LEN)
    g_teacher = compute_gradient_features(teacher, teacher_toks, proj_dim=PROJ_DIM, device="cuda",
                                          desc="external-candidate grads")
    targets = g_anchor @ g_teacher.T

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
    torch.cuda.empty_cache()

    anchor_texts = [example_text(pool[i]) for i in ext_anchor_idx]
    pool_texts = [example_text(pool[i]) for i in teacher_idx]

    summary = {}
    for tag, encoder_model in SIZES:
        print(f"\n{'=' * 70}\ndistilling {encoder_model} ({EPOCHS} epochs) on EXTERNAL anchors x candidates "
              f"only\n{'=' * 70}")
        t_size0 = time.perf_counter()
        enc = load_encoder(encoder_model, max_seq_len=ENCODER_MAX_LEN)

        epoch_eval = None
        if gt_eval is not None:
            def epoch_eval(_enc=enc):
                pred = embed(_enc, anchor_texts) @ embed(_enc, eval_pool_texts).T
                return spearman_metrics(pred, gt_eval)

        log = distill(enc, anchor_texts, pool_texts, targets, epochs=EPOCHS,
                     hard_ratio=HARD_RATIO, lr=LR, seed=SEED,
                     epoch_eval=epoch_eval, select_best_on=SELECT_BEST_ON)
        if gt_eval is not None and log["epoch_metrics"]:
            per_epoch = [f"{m['aggregated']:+.3f}" for m in log["epoch_metrics"]]
            print(f"  per-epoch agg rho: {per_epoch}")
            restored = log["epoch_metrics"][log["best_epoch"]]
            print(f"  restored (best-epoch={log['best_epoch'] + 1}/{EPOCHS}) fidelity vs held-out "
                  f"EXTERNAL teacher gradients: agg rho={restored['aggregated']:+.4f} "
                  f"per-anchor rho={restored['per_anchor_mean']:+.4f}")

        print(f"  embedding LOCAL train (n={n_train}) + LOCAL ref (n={n_ref}) -- "
              f"neither ever seen during distillation")
        t0 = time.perf_counter()
        all_train_emb = embed(enc, [example_text(d) for d in local_train])
        all_ref_emb = embed(enc, [example_text(d) for d in local_ref])
        embed_s = time.perf_counter() - t0
        scores = all_train_emb @ all_ref_emb.T
        print(f"  embed+score wall time: {embed_s:.1f}s")

        final_scores = scores.T.tolist()  # matches dattri.py's Counterfact convention
        save_path = Path(f"results/factual-attribution-influcoder-noleak-extquery-moredata-{tag}/InfluCoder.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(final_scores, f)
        wall = time.perf_counter() - t_size0
        print(f"  wrote {save_path} (this size's wall time: {wall:.1f}s)")
        summary[tag] = {"save_path": str(save_path), "wall_s": wall,
                        "best_epoch": log.get("best_epoch"),
                        "distill_agg_rho": log["epoch_metrics"][log["best_epoch"]]["aggregated"]
                                          if gt_eval is not None and log["epoch_metrics"] else None}

        del enc, all_train_emb, all_ref_emb, scores
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}\nSWEEP SUMMARY\n{'=' * 70}")
    for tag, s in summary.items():
        print(f"  {tag}: {s}")


if __name__ == "__main__":
    main()
