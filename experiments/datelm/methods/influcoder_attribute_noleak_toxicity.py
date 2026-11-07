#!/usr/bin/env python3
"""Leak-free InfluCoder distillation for DATE-LM's Toxicity/Bias task
(XSTest-response-Het subset), same philosophy as
`influcoder_attribute_noleak_counterfact_extquery.py`: train the encoder
entirely on external data, embed+score DATE-LM's own untouched train+ref only
at the very end.

DATE-LM's local data (10,187 train + 10 ref) is:
  - 10,000 Benign train rows, verbatim from HuggingFaceH4/ultrachat_200k
    (confirmed: 742/15,000 streamed rows matched local benign prompts by
    exact text -- this IS the source, or shares heavy overlap with it).
  - 66 Unsafe + 121 Safety-Aligned train rows + all 10 ref rows, verbatim
    from allenai/xstest-response's `response_harmfulness` split (446 rows,
    gated -- requires HF auth). All 197 local rows matched 1:1 by exact
    (prompt, response) text.

Unlike Counterfact's CounterFact-tracing source (21,919 rows, only ~5,539
used -- huge headroom), XSTest-response is nearly saturated: of its 249
unmatched rows, 247 are "unharmful"/"prompt_safe" and only 2 are "harmful" --
nowhere near enough fresh unsafe material to build query-analogs or positive
training candidates.

Per explicit instruction, this uses lmsys/toxic-chat instead for the unsafe
side -- one of DATE-LM's own alternate sources for this exact task (the
ToxicChat-Het/Hom subsets use it directly), just not the specific subset
being evaluated here. It has 746 toxicity=1 rows across train+test, real
headroom. This is a weaker distributional match than the Counterfact fix
(same task family, not the same narrow source) -- disclosed, not hidden.

External pools used (all disjoint from local data by construction: different
source datasets entirely for the toxic side, verified exact-text disjoint
sampling for the benign side):
  - external anchors ("queries"): N_EXT_ANCHORS toxic ToxicChat rows
  - external candidates (pool): N_EXT_BENIGN fresh UltraChat rows +
    N_EXT_TOXIC_POS toxic ToxicChat rows (disjoint from the anchors)
  - external held-out eval: same composition, disjoint from both above

Usage:
    python methods/influcoder_attribute_noleak_toxicity.py
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
N_EXT_ANCHORS = 40      # external toxic "queries" (4x local ref's 10)
N_EXT_BENIGN = 300      # external benign candidates (fresh UltraChat)
N_EXT_TOXIC_POS = 100   # external toxic candidates (fresh ToxicChat)
N_EVAL_BENIGN = 100
N_EVAL_TOXIC = 50
EPOCHS = 8
HARD_RATIO = 0.5
LR = 5e-5
SEED = 0
SELECT_BEST_ON = "aggregated"

SAVE_PATH = Path("results/toxicity-bias-influcoder-noleak/InfluCoder.pt")


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
    print(f"loading lmsys/toxic-chat for toxic rows (disjoint task/source from local XSTest data)...")
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
    assert len(clean) >= n_needed, f"not enough clean toxic rows ({len(clean)} < {n_needed})"
    return clean


def main():
    random.seed(SEED)
    print(f"InfluCoder (leak-free) / DATE-LM | task={TASK} subset={SUBSET} "
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
