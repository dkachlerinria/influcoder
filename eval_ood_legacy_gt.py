#!/usr/bin/env python3
"""Two out-of-distribution checks, both scored against this repo's own
CountSketch ground truth AND tis-ie's actual/legacy ground truth:

  1. OOD pool (re-run of the earlier eval_ood.py experiment, GT swap added):
     BBH anchors (in-distribution, unchanged) x FineWeb pool
     (out-of-distribution, never seen in training). Original CountSketch-only
     result: untrained +0.10 -> trained +0.04 per-anchor rho -- training made
     OOD-pool ranking *worse*.

  2. OOD query (NEW): FineWeb anchors (out-of-distribution query) x Dolly
     pool (kept in-distribution -- the same pool distribution the encoder
     was actually trained on). Tests the other axis: does an unfamiliar
     QUERY still get ranked sensibly against a familiar candidate pool?

Both use the same trained encoder (runs_out/big/encoder: BBH anchors /
Dolly pool, 500x1000 training) and the same untrained baseline, at 100x100.
The in-distribution reference number (BBH anchors x Dolly pool) is not
recomputed here -- it's already in runs_out/big/legacy_gt_retest.json.

Usage:
    python eval_ood_legacy_gt.py
"""

import os

import torch

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

import argparse
import json
import time
from pathlib import Path

from influcoder.data import load_bbh, load_dolly, load_fineweb
from influcoder.encoder import embed, load_encoder
from influcoder.gradients import GradientFeaturizer
from influcoder.legacy_gt import legacy_gt_scores
from influcoder.metrics import spearman_metrics


def eval_variant(name, anchors, pool, trained_enc, base_enc, args):
    print(f"\n=== {name}: {len(anchors)} anchors x {len(pool)} pool ===")

    feat = GradientFeaturizer(args.grad_model, lora_rank=args.lora_rank, lora_seed=args.seed,
                              proj_dim=args.proj_dim, max_len=args.grad_max_len)
    g_a, _ = feat.features(anchors, f"{name}: CountSketch anchors")
    g_p, _ = feat.features(pool, f"{name}: CountSketch pool")
    feat.close()
    countsketch_gt = (g_a @ g_p.T).numpy()

    t0 = time.time()
    legacy_gt = legacy_gt_scores(anchors, pool, args.grad_model, proj_dim=args.legacy_proj_dim,
                                 lora_dropout=args.legacy_lora_dropout, lora_seed=args.seed,
                                 max_len=args.grad_max_len)
    print(f"  legacy GT: {time.time() - t0:.1f}s")

    anchor_texts = [s.text for s in anchors]
    pool_texts = [s.text for s in pool]
    base_pred = embed(base_enc, anchor_texts) @ embed(base_enc, pool_texts).T
    pred = embed(trained_enc, anchor_texts) @ embed(trained_enc, pool_texts).T

    result = {
        "untrained_countsketch": spearman_metrics(base_pred, countsketch_gt),
        "trained_countsketch": spearman_metrics(pred, countsketch_gt),
        "untrained_legacy": spearman_metrics(base_pred, legacy_gt),
        "trained_legacy": spearman_metrics(pred, legacy_gt),
    }
    for k, m in result.items():
        print(f"  {k:24s} per-anchor={m['per_anchor_mean']:+.4f}  agg={m['aggregated']:+.4f}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder_dir", default="runs_out/big/encoder")
    ap.add_argument("--base_encoder_model", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--grad_model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--proj_dim", type=int, default=8192)
    ap.add_argument("--grad_max_len", type=int, default=1024)
    ap.add_argument("--legacy_proj_dim", type=int, default=65536)
    ap.add_argument("--legacy_lora_dropout", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs_out/ood_legacy_gt")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.encoder_dir).exists():
        raise FileNotFoundError(f"{args.encoder_dir} not found -- train it first "
                                f"(python run.py --preset big).")

    bbh = load_bbh("data/eval/bbh", seed=42)[:args.n]
    dolly = load_dolly("dolly/dolly_data.jsonl", seed=42)[:args.n]
    fineweb = load_fineweb(seed=42)[:args.n]
    print(f"anchors/pool source sizes: BBH={args.n}, Dolly={args.n}, FineWeb={args.n}")

    base_enc = load_encoder(args.base_encoder_model)
    trained_enc = load_encoder(args.encoder_dir)

    results = {
        "ood_pool": eval_variant("OOD pool (BBH anchors x FineWeb pool)",
                                 bbh, fineweb, trained_enc, base_enc, args),
        "ood_query": eval_variant("OOD query (FineWeb anchors x Dolly pool)",
                                  fineweb, dolly, trained_enc, base_enc, args),
    }

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
