#!/usr/bin/env python3
"""Leak-free InfluCoder distillation for DATE-LM's Toxicity/Bias task,
WildGuardMix-sourced, sample-matched to the ToxicChat `_moredata` ablation --
identical to `influcoder_attribute_noleak_toxicity_wildguard.py` except the
five sample-count constants are raised to the same values used in
`influcoder_attribute_noleak_toxicity_moredata.py`
(N_EXT_ANCHORS=100/N_EXT_BENIGN=1000/N_EXT_TOXIC_POS=300/N_EVAL_BENIGN=300/
N_EVAL_TOXIC=100), with HARD_RATIO/EPOCHS held fixed as always.

This isolates one question: does more data help WildGuardMix the same way
it helped ToxicChat (0.4242 -> 0.5845)? Toxic anchors, candidates, AND
held-out eval all still come from WildGuardMix alone here -- no cross-source
mixing yet (see `influcoder_attribute_noleak_toxicity_mixed.py` for that).
Toxic side needs 100+300+100=500 of WildGuardMix's ~8,368 usable harmful
rows -- plenty of headroom.

Usage:
    python methods/influcoder_attribute_noleak_toxicity_wildguard_moredata.py
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

TASK = "Toxicity/Bias"
SUBSET = "XSTest-response-Het"
BASE_MODEL = "EleutherAI/pythia-1b"
CHECKPOINT = "DataAttributionEval/Pythia-1b-XSTest-response-Het"

ENCODER_MODEL = "jhu-clsp/ettin-encoder-68m"
ENCODER_MAX_LEN = 1024
MAX_LEN = 1024
PROJ_DIM = 8192
N_EXT_ANCHORS = 100     # moredata: 2.5x the 40 used in the recommended/wildguard runs
N_EXT_BENIGN = 1000     # moredata: ~3.3x the 300 used in the recommended/wildguard runs
N_EXT_TOXIC_POS = 300   # moredata: 3x the 100 used in the recommended/wildguard runs
N_EVAL_BENIGN = 300     # moredata: 3x the 100 used in the recommended/wildguard runs
N_EVAL_TOXIC = 100      # moredata: 2x the 50 used in the recommended/wildguard runs
EPOCHS = 8              # held fixed -- FINDINGS.md: more epochs is a dead lever
HARD_RATIO = 0.5        # held fixed on purpose, same convention as every other run in this doc
LR = 5e-5
SEED = 0
SELECT_BEST_ON = "aggregated"

SAVE_PATH = Path("results/toxicity-bias-influcoder-noleak-wildguard-moredata/InfluCoder.pt")


def example_text(d: dict) -> str:
    return f"{d['prompt'].strip()}\n{d['response'].strip()}"


def build_external_benign_pool(local_benign_prompts: set, n_needed: int, scan_limit: int = 60000):
    print(f"streaming HuggingFaceH4/ultrachat_200k for {n_needed} clean benign rows "
          f"(disjoint from local's 10,000 benign prompts)...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    collected = []
    n_checked = n_overlap = 0
    for r in ds:
        msgs = r["messages"]
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        asst_msgs = [m["content"] for m in msgs if m["role"] == "assistant"]
        if not user_msgs or not asst_msgs:
            continue
        p, resp = user_msgs[0].strip(), asst_msgs[0].strip()
        n_checked += 1
        if p in local_benign_prompts:
            n_overlap += 1
        else:
            collected.append({"prompt": p, "response": resp})
            if len(collected) >= n_needed:
                break
        if n_checked >= scan_limit:
            break
    print(f"  checked {n_checked}, overlap with local benign {n_overlap}, collected {len(collected)}")
    assert len(collected) >= n_needed, "not enough clean benign rows found"
    return collected


def build_external_toxic_pool(local_prompts: set, n_needed: int):
    print("loading allenai/wildguardmix (wildguardtrain) for toxic rows "
          "(disjoint task/source from local XSTest data)...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain")["train"]
    toxic = [r for r in ds if r["response_harm_label"] == "harmful"]
    print(f"  wildguardmix total response_harm_label=='harmful' rows: {len(toxic)}")
    clean = []
    n_overlap = 0
    for r in toxic:
        p = r["prompt"].strip()
        if p in local_prompts:
            n_overlap += 1
            continue
        resp = (r["response"] or "").strip()
        if not resp:
            continue
        clean.append({"prompt": p, "response": resp})
    print(f"  overlap with local prompts: {n_overlap}, usable toxic rows: {len(clean)}")
    assert len(clean) >= n_needed, f"not enough clean toxic rows ({len(clean)} < {n_needed})"
    return clean


def main():
    random.seed(SEED)
    print(f"InfluCoder (leak-free, WildGuardMix, moredata) / DATE-LM | task={TASK} subset={SUBSET} "
          f"| base_model={BASE_MODEL} checkpoint={CHECKPOINT}\n")

    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)
    local_benign_prompts = set(r["prompt"].strip() for r in local_train if r["type"] == "Benign")
    local_all_prompts = set(r["prompt"].strip() for r in local_train) | set(r["prompt"].strip() for r in local_ref)
    print(f"local train={n_train} ref={n_ref} (both untouched by distillation)\n")

    n_toxic_needed = N_EXT_ANCHORS + N_EXT_TOXIC_POS + N_EVAL_TOXIC
    n_benign_needed = N_EXT_BENIGN + N_EVAL_BENIGN

    toxic_pool = build_external_toxic_pool(local_all_prompts, n_toxic_needed)
    benign_pool = build_external_benign_pool(local_benign_prompts, n_benign_needed)

    random.Random(SEED).shuffle(toxic_pool)
    random.Random(SEED + 1).shuffle(benign_pool)

    ext_anchors = toxic_pool[:N_EXT_ANCHORS]
    toxic_candidates = toxic_pool[N_EXT_ANCHORS:N_EXT_ANCHORS + N_EXT_TOXIC_POS]
    toxic_eval = toxic_pool[N_EXT_ANCHORS + N_EXT_TOXIC_POS:N_EXT_ANCHORS + N_EXT_TOXIC_POS + N_EVAL_TOXIC]
    benign_candidates = benign_pool[:N_EXT_BENIGN]
    benign_eval = benign_pool[N_EXT_BENIGN:N_EXT_BENIGN + N_EVAL_BENIGN]

    candidates = benign_candidates + toxic_candidates
    eval_pool = benign_eval + toxic_eval
    random.Random(SEED + 2).shuffle(candidates)
    random.Random(SEED + 3).shuffle(eval_pool)
    print(f"\nexternal anchors={len(ext_anchors)} (all toxic) | candidates={len(candidates)} "
          f"({len(benign_candidates)} benign + {len(toxic_candidates)} toxic) | "
          f"held-out eval={len(eval_pool)} ({len(benign_eval)} benign + {len(toxic_eval)} toxic)")

    tokenizer, teacher = load_teacher_model(BASE_MODEL, CHECKPOINT, device="cuda")

    anchor_chat = prepare_chat_format(ext_anchors)
    cand_chat = prepare_chat_format(candidates)
    eval_chat = prepare_chat_format(eval_pool)

    anchor_idx = sample_valid_indices(anchor_chat, tokenizer, MAX_LEN, list(range(len(anchor_chat))), len(anchor_chat))
    cand_idx = sample_valid_indices(cand_chat, tokenizer, MAX_LEN, list(range(len(cand_chat))), len(cand_chat))
    eval_idx = sample_valid_indices(eval_chat, tokenizer, MAX_LEN, list(range(len(eval_chat))), len(eval_chat))
    print(f"after valid-loss-span filtering: anchors={len(anchor_idx)} candidates={len(cand_idx)} eval={len(eval_idx)}")

    print(f"\n########## teacher gradients: external anchors/'queries' (n={len(anchor_idx)}) ##########")
    anchor_toks = tokenize_chat_examples([anchor_chat[i] for i in anchor_idx], tokenizer, MAX_LEN)
    g_anchor = compute_gradient_features(teacher, anchor_toks, proj_dim=PROJ_DIM, device="cuda", desc="external-anchor grads")

    print(f"\n########## teacher gradients: external candidates (n={len(cand_idx)}) ##########")
    cand_toks = tokenize_chat_examples([cand_chat[i] for i in cand_idx], tokenizer, MAX_LEN)
    g_cand = compute_gradient_features(teacher, cand_toks, proj_dim=PROJ_DIM, device="cuda", desc="external-candidate grads")
    targets = g_anchor @ g_cand.T

    gt_eval = None
    eval_pool_texts = None
    if eval_idx:
        print(f"\n########## teacher gradients: held-out external eval (n={len(eval_idx)}) ##########")
        eval_toks = tokenize_chat_examples([eval_chat[i] for i in eval_idx], tokenizer, MAX_LEN)
        g_eval = compute_gradient_features(teacher, eval_toks, proj_dim=PROJ_DIM, device="cuda", desc="external eval grads")
        gt_eval = (g_anchor @ g_eval.T).numpy()
        eval_pool_texts = [example_text(eval_pool[i]) for i in eval_idx]

    free_teacher_model(teacher)

    print(f"\n########## distilling {ENCODER_MODEL} ({EPOCHS} epochs) on EXTERNAL data only ##########")
    enc = load_encoder(ENCODER_MODEL, max_seq_len=ENCODER_MAX_LEN)
    anchor_texts = [example_text(ext_anchors[i]) for i in anchor_idx]
    pool_texts = [example_text(candidates[i]) for i in cand_idx]

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

    final_scores = np.mean(scores, axis=1).tolist()  # matches influcoder_attribute.py's Toxicity/Bias convention

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(final_scores, f)
    print(f"\nwrote {SAVE_PATH}")


if __name__ == "__main__":
    main()
