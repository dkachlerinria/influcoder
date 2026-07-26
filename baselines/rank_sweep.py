#!/usr/bin/env python3
"""Competitor reference levels across their rank hyperparameter.

LESS and LoGRA do no distillation, so "training-set size" is undefined for them.
What they *do* have is a rank knob controlling how large a gradient
representation each candidate is compressed to -- their main quality/cost dial.
Sweeping it gives the band of levels the InfluCoder scaling curve has to cross,
rather than a single arbitrary operating point.

  * LoGRA: `rank` of the LoGra projection.
  * LESS:  the LoRA rank whose gradient is random-projected (the LESS paper's
           default is 128).

Scored on the identical fixed eval + GT as every other number in this repo.

    python -m baselines.rank_sweep --logra_ranks 2 4 8 16 32 --less_ranks 8 16 32 128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baselines.common import DEFAULT_GRAD_MODEL, ground_truth
from influcoder.metrics import spearman_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="big")
    ap.add_argument("--logra_ranks", type=int, nargs="*", default=[2, 4, 8, 16, 32])
    ap.add_argument("--less_ranks", type=int, nargs="*", default=[8, 16, 32, 128])
    ap.add_argument("--grad_model", default=DEFAULT_GRAD_MODEL)
    ap.add_argument("--lora_rank", type=int, default=8, help="GT featurizer rank")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"baselines/out/{args.preset}/rank_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    gt, splits, cfg = ground_truth(args.preset, seed=0, grad_model=args.grad_model,
                                   lora_rank=args.lora_rank)
    print(f"GT {gt.shape[0]}x{gt.shape[1]} (fixed)")

    results = {"logra_raw": {}, "logra_fim": {}, "less": {}}

    for r in args.logra_ranks:
        from baselines.logra.score import score_logra
        t0 = time.time()
        try:
            variants = score_logra(splits, args.grad_model, lora_rank=r,
                                   max_len=cfg["grad_max_len"])
        except Exception as e:  # a rank incompatible with the layer dims
            print(f"  logra rank={r}: SKIPPED ({type(e).__name__}: {e})")
            continue
        for name, scores in variants.items():
            m = spearman_metrics(scores.numpy(), gt)
            results[name][str(r)] = {"aggregated": m["aggregated"],
                                     "per_anchor_mean": m["per_anchor_mean"]}
            print(f"  {name} rank={r}: agg {m['aggregated']:+.4f} "
                  f"per-anchor {m['per_anchor_mean']:+.4f} ({time.time() - t0:.0f}s)")

    for r in args.less_ranks:
        from baselines.less.score import score_less
        t0 = time.time()
        try:
            scores = score_less(splits, args.grad_model, proj_dim=cfg["proj_dim"],
                                max_len=cfg["grad_max_len"], lora_rank=r,
                                lora_alpha=4 * r)
        except Exception as e:
            print(f"  less rank={r}: SKIPPED ({type(e).__name__}: {e})")
            continue
        m = spearman_metrics(scores.numpy(), gt)
        results["less"][str(r)] = {"aggregated": m["aggregated"],
                                   "per_anchor_mean": m["per_anchor_mean"]}
        print(f"  less rank={r}: agg {m['aggregated']:+.4f} "
              f"per-anchor {m['per_anchor_mean']:+.4f} ({time.time() - t0:.0f}s)")

    payload = {"config": {"preset": args.preset, "gt_shape": list(gt.shape),
                          "grad_model": args.grad_model,
                          "proj_dim": cfg["proj_dim"],
                          "note": "LESS lora_alpha pinned to 4*rank (its default ratio)"},
               "results": results}
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
