#!/usr/bin/env python3
"""InfluCoder pipeline runner.

  1. gradient features on disjoint eval/train splits (influcoder.gradients)
  2. eval-split gradient cosines  = ground truth
     train-split gradient cosines = distillation targets
  3. train the bi-encoder on the targets (influcoder.encoder)
  4. Spearman of encoder scores vs. ground truth, against the untrained
     encoder as baseline (influcoder.metrics)

Usage:
    python run.py --preset sanity   # smallest end-to-end check + projection fidelity
    python run.py --preset tiny     # small but non-degenerate reproduction
"""

import os

import torch

# Pre-Ampere GPUs have no triton support; disable dynamo before transformers loads.
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
from influcoder.gradients import GradientFeaturizer, hardware_profile, projection_fidelity
from influcoder.metrics import spearman_metrics

PRESETS = {
    #            eval matrix        train matrix          encoder          gradients
    "sanity": dict(n_eval_a=6,  n_eval_p=12, n_train_a=16,  n_train_p=32,
                   epochs=2, proj_dim=2048, grad_max_len=512, check_projection=True,
                   hard_ratio=0.0),
    "tiny":   dict(n_eval_a=15, n_eval_p=40, n_train_a=120, n_train_p=240,
                   epochs=8, proj_dim=8192, grad_max_len=1024, check_projection=False,
                   hard_ratio=0.5),
    # Same eval set as "tiny" (numbers stay comparable) with a bigger/longer
    # train side and hard-negative mining -- more of the gradient-compute
    # budget is spent teaching the encoder outright, not proving the pipe works.
    # Verified: per-anchor rho -0.22 (untrained) -> +0.44 (best epoch 9/20).
    "push":   dict(n_eval_a=15, n_eval_p=40, n_train_a=200, n_train_p=400,
                   epochs=20, proj_dim=8192, grad_max_len=1024, check_projection=False,
                   hard_ratio=0.5),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="sanity")
    ap.add_argument("--grad_model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--encoder_model", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="override preset")
    ap.add_argument("--out_dir", default=None, help="default: runs_out/<preset>")
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    epochs = args.epochs or cfg["epochs"]
    out_dir = Path(args.out_dir or f"runs_out/{args.preset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype, attn = hardware_profile()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"hardware: {gpu} -> {dtype}, attn={attn}")

    timings: dict[str, float] = {}
    t0 = time.time()

    # -- data ---------------------------------------------------------------
    splits = disjoint_splits(
        load_bbh("data/eval/bbh", seed=42),
        load_dolly("dolly/dolly_data.jsonl", seed=42),
        cfg["n_eval_a"], cfg["n_train_a"], cfg["n_eval_p"], cfg["n_train_p"],
    )
    print(f"data: eval {cfg['n_eval_a']}x{cfg['n_eval_p']}, "
          f"train {cfg['n_train_a']}x{cfg['n_train_p']}")
    timings["data"] = time.time() - t0

    # -- gradient features ----------------------------------------------------
    t0 = time.time()
    feat = GradientFeaturizer(args.grad_model, lora_rank=args.lora_rank,
                              lora_seed=args.seed, proj_dim=cfg["proj_dim"],
                              max_len=cfg["grad_max_len"])
    n_full = 8 if cfg["check_projection"] else 0
    g_eval_a, fulls = feat.features(splits["eval_anchors"], "grads eval anchors", n_full)
    g_eval_p, _ = feat.features(splits["eval_pool"], "grads eval pool")
    g_train_a, _ = feat.features(splits["train_anchors"], "grads train anchors")
    g_train_p, _ = feat.features(splits["train_pool"], "grads train pool")
    lora_params = feat.n_params
    feat.close()
    timings["gradients"] = time.time() - t0

    proj_check = None
    if fulls:
        proj_check = projection_fidelity(fulls, g_eval_a)
        print(f"projection check ({proj_check['pairs']} pairs): "
              f"mean|err|={proj_check['mean_abs_err']:.4f} "
              f"max|err|={proj_check['max_abs_err']:.4f} "
              f"spearman={proj_check['spearman']:.4f}")
        if proj_check["max_abs_err"] > 0.15:
            print("  WARNING: sketch too lossy at this proj_dim -- raise it")

    gt = (g_eval_a @ g_eval_p.T).numpy()   # ground truth (eval split)
    targets = g_train_a @ g_train_p.T      # distillation targets (train split)

    # -- encoder --------------------------------------------------------------
    t0 = time.time()
    enc = load_encoder(args.encoder_model)
    eval_a_texts = [s.text for s in splits["eval_anchors"]]
    eval_p_texts = [s.text for s in splits["eval_pool"]]

    def eval_spearman():
        pred = embed(enc, eval_a_texts) @ embed(enc, eval_p_texts).T
        return spearman_metrics(pred, gt)

    baseline = eval_spearman()
    print(f"untrained encoder: per-anchor rho={baseline['per_anchor_mean']:+.4f} "
          f"agg rho={baseline['aggregated']:+.4f}")
    timings["baseline_eval"] = time.time() - t0

    t0 = time.time()
    train_log = distill(enc, [s.text for s in splits["train_anchors"]],
                        [s.text for s in splits["train_pool"]],
                        targets, epochs=epochs, seed=args.seed,
                        hard_ratio=cfg["hard_ratio"], epoch_eval=eval_spearman)
    timings["encoder_training"] = time.time() - t0

    t0 = time.time()
    final = eval_spearman()  # re-measured post-restore; should match the best epoch's logged value
    timings["final_eval"] = time.time() - t0

    # -- report ----------------------------------------------------------------
    print("\n=== results ===")
    print(f"{'':24s}{'per-anchor rho':>16s}{'agg rho':>10s}")
    print(f"{'untrained encoder':24s}{baseline['per_anchor_mean']:>+16.4f}"
          f"{baseline['aggregated']:>+10.4f}")
    print(f"{'trained (influcoder)':24s}{final['per_anchor_mean']:>+16.4f}"
          f"{final['aggregated']:>+10.4f}")
    print("timings: " + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items())
          + f", total={sum(timings.values()):.1f}s")

    results = {
        "preset": args.preset,
        "hardware": {"gpu": gpu, "dtype": str(dtype), "attn": attn},
        "config": {**cfg, "epochs": epochs, "grad_model": args.grad_model,
                   "encoder_model": args.encoder_model, "lora_rank": args.lora_rank,
                   "lora_params": lora_params, "seed": args.seed},
        "projection_check": proj_check,
        "baseline": baseline,
        "trained": final,
        "best_epoch": train_log["best_epoch"],
        "epoch_losses": train_log["epoch_losses"],
        "epoch_metrics": train_log["epoch_metrics"],
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "total_s": round(sum(timings.values()), 2),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
