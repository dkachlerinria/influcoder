#!/usr/bin/env python3
"""Out-of-distribution pool check: does a trained InfluCoder encoder's
Spearman agreement with gradient ground truth hold up on a pool distribution
it never saw during distillation?

The encoder was trained (see run.py) on Dolly-formatted candidates. Here the
candidate ("pool") side is swapped for FineWeb -- generic web text, neither
Dolly's chat format nor BBH's Q/A format -- while anchors stay BBH (same
eval_anchors the encoder's training run was scored against, so the only
variable that changes is the pool distribution). Ground truth is recomputed
from scratch on this new pool (gradients are never reused across datasets);
only the trained encoder and its embedding call are reused.

Usage:
    python eval_ood.py --encoder_dir runs_out/big/encoder --n_anchors 100 --n_pool 100
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
from pathlib import Path

import numpy as np

from influcoder.data import load_bbh, load_fineweb
from influcoder.encoder import embed, load_encoder
from influcoder.gradients import GradientFeaturizer, hardware_profile
from influcoder.metrics import spearman_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder_dir", required=True,
                    help="Trained encoder from a run.py run, e.g. runs_out/big/encoder")
    ap.add_argument("--base_encoder_model", default="jhu-clsp/ettin-encoder-68m",
                    help="Untrained baseline, for comparison -- must match what the "
                         "trained encoder started from.")
    ap.add_argument("--grad_model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--proj_dim", type=int, default=8192)
    ap.add_argument("--grad_max_len", type=int, default=1024)
    ap.add_argument("--n_anchors", type=int, default=100,
                    help="Same BBH eval_anchors slice as the training run (seed=42, "
                         "front slice) so this is the identical anchor set, just a new pool.")
    ap.add_argument("--n_pool", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    anchors = load_bbh("data/eval/bbh", seed=42)[: args.n_anchors]
    pool = load_fineweb(seed=42)[: args.n_pool]
    print(f"anchors: {len(anchors)} (BBH, in-distribution -- unchanged from training)")
    print(f"pool:    {len(pool)} (FineWeb, out-of-distribution -- never seen in training)")

    feat = GradientFeaturizer(args.grad_model, lora_rank=args.lora_rank,
                              lora_seed=args.seed, proj_dim=args.proj_dim,
                              max_len=args.grad_max_len)
    g_anchors, _ = feat.features(anchors, "grads anchors (BBH)")
    g_pool, _ = feat.features(pool, "grads pool (FineWeb)")
    feat.close()
    gt = (g_anchors @ g_pool.T).numpy()

    anchor_texts = [s.text for s in anchors]
    pool_texts = [s.text for s in pool]

    print(f"\nbaseline encoder: {args.base_encoder_model} (untrained)")
    base_enc = load_encoder(args.base_encoder_model)
    base_pred = embed(base_enc, anchor_texts) @ embed(base_enc, pool_texts).T
    base_metrics = spearman_metrics(base_pred, gt)
    del base_enc
    torch.cuda.empty_cache()

    print(f"trained encoder:  {args.encoder_dir}")
    enc = load_encoder(args.encoder_dir)
    pred = embed(enc, anchor_texts) @ embed(enc, pool_texts).T
    trained_metrics = spearman_metrics(pred, gt)

    print("\n=== OOD pool results (FineWeb, unseen) ===")
    print(f"{'':24s}{'per-anchor rho':>16s}{'agg rho':>10s}")
    print(f"{'untrained encoder':24s}{base_metrics['per_anchor_mean']:>+16.4f}"
          f"{base_metrics['aggregated']:>+10.4f}")
    print(f"{'trained (influcoder)':24s}{trained_metrics['per_anchor_mean']:>+16.4f}"
          f"{trained_metrics['aggregated']:>+10.4f}")

    out_path = Path(args.out or f"{args.encoder_dir.rstrip('/').removesuffix('/encoder')}/ood_fineweb.json")
    results = {
        "encoder_dir": args.encoder_dir,
        "pool_dataset": "HuggingFaceFW/fineweb (sample-10BT, shard 000_00000)",
        "n_anchors": len(anchors), "n_pool": len(pool),
        "baseline": base_metrics, "trained": trained_metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
