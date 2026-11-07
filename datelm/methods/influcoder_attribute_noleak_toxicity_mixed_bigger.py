#!/usr/bin/env python3
"""Scaled-up version of `influcoder_attribute_noleak_toxicity_mixed.py` --
same WildGuardMix-anchor / ToxicChat-candidate cross-source design (see that
script's docstring for the full rationale), only the sample counts are
raised:

  N_EXT_ANCHORS  100 -> 300   (WildGuardMix, pool has 8,368 usable rows)
  N_EXT_TOXIC_POS 300 -> 500  (ToxicChat, pool has only 746 usable rows total
                                -- this is deliberately NOT pushed further)
  N_EVAL_TOXIC   100 -> 200   (WildGuardMix)
  N_EXT_BENIGN  1000 -> 2500  (UltraChat, effectively unlimited via streaming)
  N_EVAL_BENIGN  300 -> 600   (UltraChat)

EPOCHS held fixed at 8 (FINDINGS.md: more epochs is a dead lever for this
distill() code path). HARD_RATIO is DELIBERATELY set to 0.0 here -- a
one-off deviation from the 0.5 used everywhere else in this doc, per
explicit instruction, to directly test whether hard-negative mining is
itself doing something analogous to what the same-source anchor/pool
design turned out to do (or not). This is a real risk, not a free
scale-up: EXP1's FINDINGS.md documented that scaling up candidate volume
WITHOUT proportionally raising hard_ratio can regress quality via pool
dilution (easy negatives swamp the hard-negative signal) -- and that
finding was specifically about hard_ratio=0 at scale, which is exactly the
regime this run is in (2500+500=3000 candidates at hard_ratio=0.0). So
this could plausibly underperform the hard_ratio=0.5 mixed result (0.6160)
even with more data. Report whatever actually happens -- this is a genuine
open question, not a foregone conclusion either way.

Adds a printed spot-check of a few raw example rows from each of the three
external sources right after the pools are built (before the expensive GPU
gradient step), so a garbled/mis-encoded/truncated row would show up in the
log rather than only being caught after burning GPU time.

Usage:
    python methods/influcoder_attribute_noleak_toxicity_mixed_bigger.py
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
N_EXT_ANCHORS = 300     # WildGuardMix
N_EXT_BENIGN = 2500     # UltraChat
N_EXT_TOXIC_POS = 500   # ToxicChat -- DIFFERENT source than the anchors
N_EVAL_BENIGN = 600     # UltraChat
N_EVAL_TOXIC = 200      # WildGuardMix -- same source as anchors, disjoint slice
EPOCHS = 8              # held fixed -- FINDINGS.md: more epochs is a dead lever
HARD_RATIO = 0.0        # DEVIATION from this doc's 0.5 convention, per explicit instruction --
                        # see module docstring for why, and the real dilution-regression risk this carries
LR = 5e-5
SEED = 0
SELECT_BEST_ON = "aggregated"

SAVE_PATH = Path("results/toxicity-bias-influcoder-noleak-mixed-bigger/InfluCoder.pt")


def example_text(d: dict) -> str:
    return f"{d['prompt'].strip()}\n{d['response'].strip()}"


def spot_check(label: str, rows: list[dict], n: int = 4):
    print(f"\n---- spot-check: {label} ({len(rows)} rows) ----")
    for r in rows[:n]:
        p = r["prompt"].replace("\n", " \\n ")[:150]
        resp = r["response"].replace("\n", " \\n ")[:150]
        print(f"  PROMPT:   {p!r}")
        print(f"  RESPONSE: {resp!r}")
    print("----")


def build_external_benign_pool(local_benign_prompts: set, n_needed: int, scan_limit: int = 200000):
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


def build_wildguard_toxic_pool(local_prompts: set, n_needed: int):
    print("loading allenai/wildguardmix (wildguardtrain) for toxic ANCHOR+EVAL rows "
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
    assert len(clean) >= n_needed, f"not enough clean wildguardmix rows ({len(clean)} < {n_needed})"
    return clean


def build_toxicchat_toxic_pool(local_prompts: set, n_needed: int):
    print("loading lmsys/toxic-chat for toxic CANDIDATE rows "
          "(disjoint task/source from local XSTest data, and a DIFFERENT source than the anchors)...")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    rows = list(ds["train"]) + list(ds["test"])
    toxic = [r for r in rows if r["toxicity"] == 1]
    print(f"  toxic-chat total toxic rows: {len(toxic)}")
    clean = []
    n_overlap = 0
    for r in toxic:
        p = r["user_input"].strip()
        if p in local_prompts:
            n_overlap += 1
            continue
        resp = (r["model_output"] or "").strip()
        if not resp:
            continue
        clean.append({"prompt": p, "response": resp})
    print(f"  overlap with local prompts: {n_overlap}, usable toxic rows: {len(clean)}")
    assert len(clean) >= n_needed, f"not enough clean toxic-chat rows ({len(clean)} < {n_needed})"
    return clean


def main():
    random.seed(SEED)
    print(f"InfluCoder (leak-free, CROSS-SOURCE mixed, BIGGER) / DATE-LM | task={TASK} subset={SUBSET} "
          f"| base_model={BASE_MODEL} checkpoint={CHECKPOINT}\n")
    print("toxic anchors <- WildGuardMix | toxic candidates <- ToxicChat | "
          "toxic held-out eval <- WildGuardMix (disjoint from anchors)\n")

    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)
    local_benign_prompts = set(r["prompt"].strip() for r in local_train if r["type"] == "Benign")
    local_all_prompts = set(r["prompt"].strip() for r in local_train) | set(r["prompt"].strip() for r in local_ref)
    print(f"local train={n_train} ref={n_ref} (both untouched by distillation)\n")

    # anchors + held-out eval both draw from ONE shuffled WildGuardMix pool (disjoint slices)
    n_wildguard_needed = N_EXT_ANCHORS + N_EVAL_TOXIC
    wildguard_pool = build_wildguard_toxic_pool(local_all_prompts, n_wildguard_needed)
    random.Random(SEED).shuffle(wildguard_pool)
    ext_anchors = wildguard_pool[:N_EXT_ANCHORS]
    toxic_eval = wildguard_pool[N_EXT_ANCHORS:N_EXT_ANCHORS + N_EVAL_TOXIC]

    # candidates draw from an INDEPENDENT ToxicChat pool -- different source entirely
    toxicchat_pool = build_toxicchat_toxic_pool(local_all_prompts, N_EXT_TOXIC_POS)
    random.Random(SEED + 1).shuffle(toxicchat_pool)
    toxic_candidates = toxicchat_pool[:N_EXT_TOXIC_POS]

    benign_pool = build_external_benign_pool(local_benign_prompts, N_EXT_BENIGN + N_EVAL_BENIGN)
    random.Random(SEED + 2).shuffle(benign_pool)
    benign_candidates = benign_pool[:N_EXT_BENIGN]
    benign_eval = benign_pool[N_EXT_BENIGN:N_EXT_BENIGN + N_EVAL_BENIGN]

    spot_check("WildGuardMix anchors", ext_anchors)
    spot_check("ToxicChat candidates", toxic_candidates)
    spot_check("UltraChat benign candidates", benign_candidates)

    candidates = benign_candidates + toxic_candidates
    eval_pool = benign_eval + toxic_eval
    random.Random(SEED + 3).shuffle(candidates)
    random.Random(SEED + 4).shuffle(eval_pool)
    print(f"\nexternal anchors={len(ext_anchors)} (all WildGuardMix toxic) | candidates={len(candidates)} "
          f"({len(benign_candidates)} UltraChat benign + {len(toxic_candidates)} ToxicChat toxic) | "
          f"held-out eval={len(eval_pool)} ({len(benign_eval)} UltraChat benign + {len(toxic_eval)} WildGuardMix toxic)")

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
