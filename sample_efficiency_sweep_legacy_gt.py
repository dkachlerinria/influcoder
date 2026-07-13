#!/usr/bin/env python3
"""Re-run of sample_efficiency_sweep.py's Part A degradation curve, scored
against tis-ie's ACTUAL ground truth (influcoder.legacy_gt -- plain-SGD LoRA
gradients, TRAK Rademacher projection via fast_jl/BasicProjector, dropout
active) instead of just this repo's own CountSketch ground truth.

What's unchanged from the original sweep: the training recipe (BASELINE),
the nested ratio-preserving train-size subsets (SIZE_TRIALS), and the
CountSketch-computed distillation targets the encoder is trained to
reproduce -- that's a training-time choice, separate from which ground
truth the resulting encoder gets *measured* against, which is what this
script changes. Best-epoch checkpoint selection during training also still
uses CountSketch (so trials pick the identical checkpoint the original
sweep would have); legacy GT is only used to re-score the baseline and
final checkpoints, alongside CountSketch, not to pick between them.

The fixed 100x100 eval set never shrinks across trials (same as the
original), so legacy GT is computed ONCE and reused for all 7 size points --
it is cached to disk since tis-ie's own ground-truth definition has dropout
active during gradient collection (model.train()) and is therefore
stochastic; caching means every subsequent point in the curve is compared
against the exact same legacy-GT draw, not a fresh noisy one per trial.

Usage:
    python sample_efficiency_sweep_legacy_gt.py
"""

import os

import torch

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from influcoder.data import disjoint_splits, load_bbh, load_dolly
from influcoder.encoder import distill, embed, load_encoder
from influcoder.legacy_gt import legacy_gt_scores
from influcoder.metrics import spearman_metrics
from sample_efficiency_sweep import (BASELINE, GRAD_MAX_LEN, GRAD_MODEL, N_EVAL_A,
                                     N_EVAL_P, SIZE_TRIALS, compute_or_load_gradients)


def run_trial_dual(cache, gts: dict, na: int, np_: int, cfg: dict, seed: int,
                   encoder_model: str) -> dict:
    """Like sample_efficiency_sweep.run_trial, but scores the baseline and
    final checkpoint against every (name -> gt_matrix) in `gts` instead of a
    single ground truth. Training itself (targets, best-epoch selection) is
    unchanged -- always CountSketch -- so this is purely an additional
    evaluation, not a different training run."""
    enc = load_encoder(encoder_model)
    g_a = cache["g_train_a"][:na]
    g_p = cache["g_train_p"][:np_]
    targets = g_a @ g_p.T  # CountSketch distillation targets -- unchanged
    train_a_texts = cache["train_a_texts"][:na]
    train_p_texts = cache["train_p_texts"][:np_]

    def eval_all():
        pred = embed(enc, cache["eval_a_texts"]) @ embed(enc, cache["eval_p_texts"]).T
        return {name: spearman_metrics(pred, gt) for name, gt in gts.items()}

    def eval_countsketch_only():
        # Used as distill()'s epoch_eval -- keeps best-epoch selection
        # identical to the original sweep (single metric dict expected).
        pred = embed(enc, cache["eval_a_texts"]) @ embed(enc, cache["eval_p_texts"]).T
        return spearman_metrics(pred, gts["countsketch"])

    baseline = eval_all()
    t0 = time.time()
    train_log = distill(enc, train_a_texts, train_p_texts, targets,
                        epochs=cfg["epochs"], k_anchors=cfg["k_anchors"],
                        m_candidates=cfg["m_candidates"], lr=cfg["lr"],
                        hard_ratio=cfg["hard_ratio"], alpha=cfg["alpha"],
                        seed=seed, epoch_eval=eval_countsketch_only)
    final = eval_all()
    elapsed = time.time() - t0
    del enc
    torch.cuda.empty_cache()
    return {"baseline": baseline, "trained": final, "best_epoch": train_log["best_epoch"],
            "elapsed_s": round(elapsed, 1), "n_train_a": na, "n_train_p": np_, "config": cfg}


def compute_or_load_legacy_gt(seed: int, out_dir: Path, proj_dim: int, lora_dropout: float):
    cache_path = out_dir / "legacy_gt_100x100.pt"
    if cache_path.exists():
        print(f"loading cached legacy GT from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    splits = disjoint_splits(
        load_bbh("data/eval/bbh", seed=42), load_dolly("dolly/dolly_data.jsonl", seed=42),
        N_EVAL_A, 0, N_EVAL_P, 0,
    )
    print(f"computing legacy (tis-ie) ground truth on the fixed {N_EVAL_A}x{N_EVAL_P} "
          f"eval set (proj_dim={proj_dim}, lora_dropout={lora_dropout})...")
    t0 = time.time()
    legacy_gt = legacy_gt_scores(splits["eval_anchors"], splits["eval_pool"], GRAD_MODEL,
                                 proj_dim=proj_dim, lora_dropout=lora_dropout,
                                 lora_seed=seed, max_len=GRAD_MAX_LEN)
    print(f"legacy GT: {time.time() - t0:.1f}s")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(legacy_gt, cache_path)
    print(f"cached legacy GT to {cache_path} (reused on re-run)")
    return legacy_gt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder_model", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs_out/sample_eff")
    ap.add_argument("--legacy_proj_dim", type=int, default=65536,
                    help="tis-ie GT_PROJ_DIM default")
    ap.add_argument("--legacy_lora_dropout", type=float, default=0.1,
                    help="tis-ie stock value (stochastic GT); pass 0.0 for determinism")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = compute_or_load_gradients(args.seed)
    countsketch_gt = (cache["g_eval_a"] @ cache["g_eval_p"].T).numpy()
    legacy_gt = compute_or_load_legacy_gt(args.seed, out_dir, args.legacy_proj_dim,
                                          args.legacy_lora_dropout)

    gt_agreement = spearman_metrics(countsketch_gt, legacy_gt)
    print(f"\nCountSketch GT vs. legacy GT (fixed {N_EVAL_A}x{N_EVAL_P} eval): "
          f"per-anchor rho={gt_agreement['per_anchor_mean']:+.4f}  "
          f"agg rho={gt_agreement['aggregated']:+.4f}")

    gts = {"countsketch": countsketch_gt, "legacy": legacy_gt}

    results = {
        "baseline_recipe": BASELINE,
        "legacy_gt_config": {"proj_dim": args.legacy_proj_dim,
                             "lora_dropout": args.legacy_lora_dropout},
        "gt_agreement": gt_agreement,
        "size_trials": [],
    }
    print(f"\n{'n_train_a x n_train_p':24s}{'agg rho (CS)':>14s}{'agg rho (legacy)':>18s}{'time(s)':>9s}")
    for na, np_ in SIZE_TRIALS:
        cfg = {**BASELINE, "n_train_a": na, "n_train_p": np_}
        r = run_trial_dual(cache, gts, na, np_, cfg, args.seed, args.encoder_model)
        results["size_trials"].append(r)
        cs = r["trained"]["countsketch"]["aggregated"]
        lg = r["trained"]["legacy"]["aggregated"]
        print(f"{f'{na}x{np_}':24s}{cs:>+14.4f}{lg:>+18.4f}{r['elapsed_s']:>9.0f}")

    out_path = out_dir / "part_a_legacy_gt_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
