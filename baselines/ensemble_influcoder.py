#!/usr/bin/env python3
"""Train N seeds of a fixed InfluCoder config and score-average their eval
predictions, to see if ensembling squeezes out gains beyond the best single
seed. Purely a training/inference-side change (average of independently
distilled encoders' score matrices) -- the eval GT and LoGRA target are
untouched.

    python -m baselines.ensemble_influcoder --gt_preset paper200x3 \
        --n_train_a 1500 --n_train_p 3000 --hard_ratio 0.6 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baselines.common import ground_truth
from baselines.scaling_sweep import train_features
from influcoder.encoder import distill, embed, load_encoder
from influcoder.metrics import spearman_metrics

MODEL = "Qwen/Qwen3-4B"
OUT_DIR = Path("baselines/out/qwen4b_tuning")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gt_preset", default="paper200x3")
    ap.add_argument("--gt_lora_rank", type=int, default=16)
    ap.add_argument("--n_train_a", type=int, default=1500)
    ap.add_argument("--n_train_p", type=int, default=3000)
    ap.add_argument("--encoder_model", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hard_ratio", type=float, default=0.6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt, splits, cfg = ground_truth(args.gt_preset, seed=0, grad_model=MODEL,
                                   lora_rank=args.gt_lora_rank)
    g_ta, g_tp = train_features(splits, cfg, MODEL, args.gt_lora_rank, 0)
    n_a, n_p = min(args.n_train_a, g_ta.shape[0]), min(args.n_train_p, g_tp.shape[0])
    if n_a < args.n_train_a or n_p < args.n_train_p:
        raise SystemExit(f"cached train features only cover {g_ta.shape[0]}x{g_tp.shape[0]}")
    targets = g_ta[:n_a] @ g_tp[:n_p].T

    eval_a = [s.text for s in splits["eval_anchors"]]
    eval_p = [s.text for s in splits["eval_pool"]]

    score_mats = []
    solo_metrics = []
    for seed in args.seeds:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        enc = load_encoder(args.encoder_model)

        def ev():
            return spearman_metrics(embed(enc, eval_a) @ embed(enc, eval_p).T, gt)

        distill(enc, [s.text for s in splits["train_anchors"][:n_a]],
               [s.text for s in splits["train_pool"][:n_p]], targets,
               epochs=args.epochs, seed=seed, hard_ratio=args.hard_ratio,
               epoch_eval=ev, select_best_on="aggregated")
        score = embed(enc, eval_a) @ embed(enc, eval_p).T
        m = spearman_metrics(score, gt)
        print(f"seed {seed}: agg {m['aggregated']:+.4f} per-anchor {m['per_anchor_mean']:+.4f}")
        score_mats.append(score)
        solo_metrics.append(m)
        del enc
        torch.cuda.empty_cache()

    # Average the raw cosine-score matrices (same scale across seeds, no
    # renormalization needed) then re-score against the same GT.
    ens_score = np.mean(score_mats, axis=0)
    ens_m = spearman_metrics(ens_score, gt)
    print(f"\n### {args.name}")
    for seed, m in zip(args.seeds, solo_metrics):
        print(f"  solo seed {seed}: agg {m['aggregated']:+.4f}")
    print(f"  ensemble ({len(args.seeds)} seeds): agg {ens_m['aggregated']:+.4f} "
          f"per-anchor {ens_m['per_anchor_mean']:+.4f}")

    record = {
        "name": args.name, "args": vars(args),
        "solo_aggs": [m["aggregated"] for m in solo_metrics],
        "ensemble_agg": ens_m["aggregated"], "ensemble_per_anchor": ens_m["per_anchor_mean"],
    }
    with open(OUT_DIR / "ensemble_log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
