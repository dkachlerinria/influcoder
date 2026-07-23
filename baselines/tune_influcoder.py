#!/usr/bin/env python3
"""One InfluCoder training experiment against the fixed Qwen3-4B paper200 GT.

Built to iterate fast: the eval split + GT and the train-side gradient
features are cached (baselines/common.ground_truth, baselines.scaling_sweep.
train_features), so after the first call every experiment pays only for
encoder training -- no repeat 4B forward/backward passes.

Every run appends one JSON line to baselines/out/qwen4b_tuning/log.jsonl
(never overwritten, so the full experiment history survives) and, if it beats
the best `agg` seen so far (tracked in baselines/out/qwen4b_tuning/best.json),
saves the encoder to runs_out/qwen4b_tuning/<name>/encoder.

    python -m baselines.tune_influcoder --name baseline
    python -m baselines.tune_influcoder --name accum4 --grad_accum_steps 4
    python -m baselines.tune_influcoder --name big2x --n_train_a 1000 --n_train_p 2000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from baselines.common import ground_truth
from baselines.scaling_sweep import train_features
from influcoder.encoder import distill, embed, load_encoder
from influcoder.metrics import spearman_metrics

MODEL = "Qwen/Qwen3-4B"
OUT_DIR = Path("baselines/out/qwen4b_tuning")
ENC_DIR = Path("runs_out/qwen4b_tuning")
LOG_PATH = OUT_DIR / "log.jsonl"
BEST_PATH = OUT_DIR / "best.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="experiment id, used as log key + save dir")
    ap.add_argument("--notes", default="", help="one-line rationale for this experiment")
    ap.add_argument("--gt_preset", default="paper200",
                    help="preset defining the eval split + GT fidelity (unchanged unless told otherwise)")
    ap.add_argument("--gt_lora_rank", type=int, default=16)
    ap.add_argument("--n_train_a", type=int, default=500)
    ap.add_argument("--n_train_p", type=int, default=1000)
    ap.add_argument("--encoder_model", default="jhu-clsp/ettin-encoder-68m")
    ap.add_argument("--encoder_max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--k_anchors", type=int, default=8)
    ap.add_argument("--m_candidates", type=int, default=16)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--lr_schedule", default="linear", choices=["linear", "cosine"])
    ap.add_argument("--alpha", type=float, default=0.5, help="Pearson-loss weight")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--hard_ratio", type=float, default=0.0)
    ap.add_argument("--hard_ratio_end", type=float, default=None,
                    help="if set, ramp hard_ratio -> this value over training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select_best_on", default="aggregated",
                    choices=["aggregated", "per_anchor_mean"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    gt, splits, cfg = ground_truth(args.gt_preset, seed=0, grad_model=MODEL,
                                   lora_rank=args.gt_lora_rank)
    g_ta, g_tp = train_features(splits, cfg, MODEL, args.gt_lora_rank, 0)
    n_a, n_p = min(args.n_train_a, g_ta.shape[0]), min(args.n_train_p, g_tp.shape[0])
    if n_a < args.n_train_a or n_p < args.n_train_p:
        raise SystemExit(f"cached train features only cover {g_ta.shape[0]}x{g_tp.shape[0]}; "
                         f"requested {args.n_train_a}x{args.n_train_p}")
    targets = g_ta[:n_a] @ g_tp[:n_p].T

    eval_a = [s.text for s in splits["eval_anchors"]]
    eval_p = [s.text for s in splits["eval_pool"]]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    enc = load_encoder(args.encoder_model, max_seq_len=args.encoder_max_len)

    def ev():
        return spearman_metrics(embed(enc, eval_a) @ embed(enc, eval_p).T, gt)

    untrained = ev()
    log = distill(
        enc, [s.text for s in splits["train_anchors"][:n_a]],
        [s.text for s in splits["train_pool"][:n_p]], targets,
        epochs=args.epochs, k_anchors=args.k_anchors, m_candidates=args.m_candidates,
        lr=args.lr, seed=args.seed, hard_ratio=args.hard_ratio, epoch_eval=ev,
        select_best_on=args.select_best_on, grad_accum_steps=args.grad_accum_steps,
        alpha=args.alpha, temperature=args.temperature, weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm, warmup_frac=args.warmup_frac,
        lr_schedule=args.lr_schedule, hard_ratio_end=args.hard_ratio_end,
    )
    best = log["epoch_metrics"][log["best_epoch"]]
    final = log["epoch_metrics"][-1]
    elapsed = time.time() - t0

    print(f"\n### {args.name}")
    print(f"    untrained   agg {untrained['aggregated']:+.4f}")
    print(f"    best-epoch  agg {best['aggregated']:+.4f}  per-anchor {best['per_anchor_mean']:+.4f}"
          f"  (epoch {log['best_epoch'] + 1}/{args.epochs})")
    print(f"    final-epoch agg {final['aggregated']:+.4f}  per-anchor {final['per_anchor_mean']:+.4f}")
    print(f"    elapsed {elapsed:.0f}s")

    record = {
        "name": args.name, "notes": args.notes, "elapsed_s": round(elapsed, 1),
        "args": vars(args),
        "untrained_agg": untrained["aggregated"],
        "best_epoch": log["best_epoch"] + 1,
        "best_agg": best["aggregated"], "best_per_anchor": best["per_anchor_mean"],
        "final_agg": final["aggregated"], "final_per_anchor": final["per_anchor_mean"],
        "epoch_trace_agg": [m["aggregated"] for m in log["epoch_metrics"]],
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    best_so_far = None
    if BEST_PATH.exists():
        best_so_far = json.loads(BEST_PATH.read_text())
    if best_so_far is None or best["aggregated"] > best_so_far["best_agg"]:
        save_dir = ENC_DIR / args.name / "encoder"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        enc.save(str(save_dir))
        record["encoder_dir"] = str(save_dir)
        BEST_PATH.write_text(json.dumps(record, indent=2))
        print(f"    *** NEW BEST *** saved encoder to {save_dir}")

    print(f"wrote {LOG_PATH}")


if __name__ == "__main__":
    main()
