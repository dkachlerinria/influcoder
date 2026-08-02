#!/usr/bin/env python3
"""Encoder-size sweep on top of the winning cross-source toxicity mix
(`influcoder_attribute_noleak_toxicity_mixed.py`) -- IDENTICAL data
provenance/exclusion/sample-count logic (WildGuardMix anchors(100)+eval(100),
ToxicChat candidates(300), UltraChat benign candidates(1000)+eval(300),
HARD_RATIO=0.5, EPOCHS=8), the only thing that varies is ENCODER_MODEL.

The teacher (Pythia-1B checkpoint) gradient computation for the fixed anchor/
candidate/eval draw does not depend on which student encoder is being
distilled into, so it is computed ONCE and reused across every size in SIZES
below (same pattern as
`influcoder_attribute_noleak_counterfact_extquery_moredata_encodersweep.py`).
The existing 68m result (AUPRC=0.6160, current best in the doc) is NOT
recomputed here.

Risk this sweep is specifically watching for: this config's total external
data (~1,800 rows) is notably smaller than the Counterfact moredata sweep's
3,400, closer to the scale FINDINGS.md (EXP1) documented 400m collapsing at
with the default lr=5e-5. Per-epoch eval Spearman is printed for every size so
a collapse (metric craters after an early peak despite falling train loss) is
visible rather than only reporting a blind final AUPRC.

Usage:
    python methods/influcoder_attribute_noleak_toxicity_mixed_encodersweep.py
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

TASK = "Toxicity/Bias"
SUBSET = "XSTest-response-Het"
BASE_MODEL = "EleutherAI/pythia-1b"
CHECKPOINT = "DataAttributionEval/Pythia-1b-XSTest-response-Het"

ENCODER_MAX_LEN = 1024
MAX_LEN = 1024
PROJ_DIM = 8192
N_EXT_ANCHORS = 100
N_EXT_BENIGN = 1000
N_EXT_TOXIC_POS = 300
N_EVAL_BENIGN = 300
N_EVAL_TOXIC = 100
EPOCHS = 8
HARD_RATIO = 0.5
LR = 5e-5
LR_FALLBACK = 1e-5  # FINDINGS.md's documented fix for bigger-encoder collapse at small data
SEED = 0
SELECT_BEST_ON = "aggregated"

SIZES = [
    ("150m", "jhu-clsp/ettin-encoder-150m"),
    ("400m", "jhu-clsp/ettin-encoder-400m"),
    ("1b", "jhu-clsp/ettin-encoder-1b"),
]


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


def run_one_size(tag, encoder_model, anchor_texts, pool_texts, targets,
                  eval_pool_texts, gt_eval, local_train, local_ref, n_train, n_ref,
                  lr, save_tag_suffix=""):
    print(f"\n{'=' * 70}\ndistilling {encoder_model} (lr={lr}) ({EPOCHS} epochs) on EXTERNAL "
          f"anchors x candidates only\n{'=' * 70}")
    t_size0 = time.perf_counter()
    enc = load_encoder(encoder_model, max_seq_len=ENCODER_MAX_LEN)

    epoch_eval = None
    if gt_eval is not None:
        def epoch_eval(_enc=enc):
            pred = embed(_enc, anchor_texts) @ embed(_enc, eval_pool_texts).T
            return spearman_metrics(pred, gt_eval)

    log = distill(enc, anchor_texts, pool_texts, targets, epochs=EPOCHS,
                 hard_ratio=HARD_RATIO, lr=lr, seed=SEED,
                 epoch_eval=epoch_eval, select_best_on=SELECT_BEST_ON)
    collapsed = False
    if gt_eval is not None and log["epoch_metrics"]:
        per_epoch = [f"{m['aggregated']:+.3f}" for m in log["epoch_metrics"]]
        print(f"  per-epoch agg rho: {per_epoch}")
        vals = [m["aggregated"] for m in log["epoch_metrics"]]
        peak = max(vals)
        final = vals[-1]
        if peak > 0.05 and final < 0.3 * peak:
            collapsed = True
            print(f"  ** COLLAPSE PATTERN DETECTED ** peak={peak:+.3f} final={final:+.3f}")
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

    final_scores = np.mean(scores, axis=1).tolist()  # matches toxicity_mixed.py's convention
    save_path = Path(f"results/toxicity-bias-influcoder-noleak-mixed-{tag}{save_tag_suffix}/InfluCoder.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(final_scores, f)
    wall = time.perf_counter() - t_size0
    print(f"  wrote {save_path} (this size's wall time: {wall:.1f}s)")

    del enc, all_train_emb, all_ref_emb, scores
    gc.collect()
    torch.cuda.empty_cache()
    return collapsed, str(save_path)


def main():
    random.seed(SEED)
    print(f"InfluCoder (leak-free, CROSS-SOURCE mixed) encoder-size sweep / DATE-LM | "
          f"task={TASK} subset={SUBSET} | base_model={BASE_MODEL} checkpoint={CHECKPOINT}\n")

    local_train, local_ref = get_dataset(TASK, SUBSET)
    n_train, n_ref = len(local_train), len(local_ref)
    local_benign_prompts = set(r["prompt"].strip() for r in local_train if r["type"] == "Benign")
    local_all_prompts = set(r["prompt"].strip() for r in local_train) | set(r["prompt"].strip() for r in local_ref)
    print(f"local train={n_train} ref={n_ref} (both untouched by distillation)\n")

    n_wildguard_needed = N_EXT_ANCHORS + N_EVAL_TOXIC
    wildguard_pool = build_wildguard_toxic_pool(local_all_prompts, n_wildguard_needed)
    random.Random(SEED).shuffle(wildguard_pool)
    ext_anchors = wildguard_pool[:N_EXT_ANCHORS]
    toxic_eval = wildguard_pool[N_EXT_ANCHORS:N_EXT_ANCHORS + N_EVAL_TOXIC]

    toxicchat_pool = build_toxicchat_toxic_pool(local_all_prompts, N_EXT_TOXIC_POS)
    random.Random(SEED + 1).shuffle(toxicchat_pool)
    toxic_candidates = toxicchat_pool[:N_EXT_TOXIC_POS]

    benign_pool = build_external_benign_pool(local_benign_prompts, N_EXT_BENIGN + N_EVAL_BENIGN)
    random.Random(SEED + 2).shuffle(benign_pool)
    benign_candidates = benign_pool[:N_EXT_BENIGN]
    benign_eval = benign_pool[N_EXT_BENIGN:N_EXT_BENIGN + N_EVAL_BENIGN]

    candidates = benign_candidates + toxic_candidates
    eval_pool = benign_eval + toxic_eval
    random.Random(SEED + 3).shuffle(candidates)
    random.Random(SEED + 4).shuffle(eval_pool)
    print(f"\nexternal anchors={len(ext_anchors)} (all WildGuardMix toxic) | candidates={len(candidates)} "
          f"({len(benign_candidates)} UltraChat benign + {len(toxic_candidates)} ToxicChat toxic) | "
          f"held-out eval={len(eval_pool)} ({len(benign_eval)} UltraChat benign + {len(toxic_eval)} WildGuardMix toxic) "
          f"-- IDENTICAL selection to the 68m mixed run (same SEED, same pool construction)")

    tokenizer, teacher = load_teacher_model(BASE_MODEL, CHECKPOINT, device="cuda")

    anchor_chat = prepare_chat_format(ext_anchors)
    cand_chat = prepare_chat_format(candidates)
    eval_chat = prepare_chat_format(eval_pool)

    anchor_idx = sample_valid_indices(anchor_chat, tokenizer, MAX_LEN, list(range(len(anchor_chat))), len(anchor_chat))
    cand_idx = sample_valid_indices(cand_chat, tokenizer, MAX_LEN, list(range(len(cand_chat))), len(cand_chat))
    eval_idx = sample_valid_indices(eval_chat, tokenizer, MAX_LEN, list(range(len(eval_chat))), len(eval_chat))
    print(f"after valid-loss-span filtering: anchors={len(anchor_idx)} candidates={len(cand_idx)} eval={len(eval_idx)}")

    print(f"\n########## teacher gradients: external anchors/'queries' (n={len(anchor_idx)}) "
          f"[computed ONCE, reused for every encoder size below] ##########")
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
    torch.cuda.empty_cache()

    anchor_texts = [example_text(ext_anchors[i]) for i in anchor_idx]
    pool_texts = [example_text(candidates[i]) for i in cand_idx]

    summary = {}
    for tag, encoder_model in SIZES:
        collapsed, save_path = run_one_size(
            tag, encoder_model, anchor_texts, pool_texts, targets,
            eval_pool_texts, gt_eval, local_train, local_ref, n_train, n_ref, LR)
        summary[tag] = {"lr": LR, "collapsed": collapsed, "save_path": save_path}

        if collapsed:
            print(f"\n  {tag} collapsed at lr={LR} -- retrying once at lr={LR_FALLBACK} "
                  f"per FINDINGS.md's documented fix ##########")
            collapsed2, save_path2 = run_one_size(
                tag, encoder_model, anchor_texts, pool_texts, targets,
                eval_pool_texts, gt_eval, local_train, local_ref, n_train, n_ref,
                LR_FALLBACK, save_tag_suffix="-lrfix")
            summary[f"{tag}-lrfix"] = {"lr": LR_FALLBACK, "collapsed": collapsed2, "save_path": save_path2}

    print(f"\n{'=' * 70}\nSWEEP SUMMARY\n{'=' * 70}")
    for tag, s in summary.items():
        print(f"  {tag}: {s}")


if __name__ == "__main__":
    main()
