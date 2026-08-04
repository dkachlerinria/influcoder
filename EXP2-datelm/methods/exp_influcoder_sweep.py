#!/usr/bin/env python3
"""Sweep InfluCoder training hyperparameters, looking for a stable+good config.

Writes to its own results dir; never touches the official fig2_*.json.
Reuses the teacher-gradient disk cache, so only the first run per task pays
the gradient cost.

Usage:
    python methods/exp_influcoder_sweep.py --task het --epochs 8 --lr 5e-5 \
        --hard-ratio 0.5 --tag base --rep 1
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

OUT_DIR = Path("results_influcoder_sweep")
SUMMARY = OUT_DIR / "summary.json"

TASK_MODULES = {
    "cf": "methods.run_fig2_counterfact",
    "het": "methods.run_fig2_toxicity",
    "hom": "methods.run_fig2_toxicity_hom",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_MODULES))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hard-ratio", type=float, default=None)
    ap.add_argument("--tag", required=True, help="short label for this config")
    ap.add_argument("--encoder", type=str, default=None)
    ap.add_argument("--rep", type=int, default=1)
    args = ap.parse_args()

    mod = importlib.import_module(TASK_MODULES[args.task])

    # Keep sweep artifacts out of the official results tree.
    key = f"{args.task}_{args.tag}_rep{args.rep}"
    mod.RESULTS_DIR = OUT_DIR / key
    # ...but SHARE the gradient cache with the official runs: gradients depend
    # only on the seeded pools + checkpoint, not on any training hyperparameter
    # being swept here, so every config can reuse the same tensors.
    mod.GRAD_CACHE_DIR = Path("results/_grad_cache")

    if args.epochs is not None:
        mod.INFLUCODER_EPOCHS = args.epochs
    if args.lr is not None:
        mod.INFLUCODER_LR = args.lr
    if args.hard_ratio is not None:
        mod.INFLUCODER_HARD_RATIO = args.hard_ratio
    if args.encoder is not None:
        mod.INFLUCODER_ENCODER_MODEL = args.encoder

    print(f"=== sweep {key}: epochs={mod.INFLUCODER_EPOCHS} lr={mod.INFLUCODER_LR} "
          f"hard_ratio={mod.INFLUCODER_HARD_RATIO} enc={mod.INFLUCODER_ENCODER_MODEL} ===")

    t0 = time.perf_counter()
    wall_s, setup_s, score_path = mod.run_influcoder()
    total_s = time.perf_counter() - t0

    metrics = mod.evaluate(score_path)
    rec = ({"recall_at_50": metrics[0], "mrr": metrics[1]}
           if isinstance(metrics, tuple) else {"auprc": metrics})
    rec.update({
        "task": args.task, "tag": args.tag, "rep": args.rep,
        "epochs": mod.INFLUCODER_EPOCHS, "lr": mod.INFLUCODER_LR,
        "hard_ratio": mod.INFLUCODER_HARD_RATIO,
        "encoder": mod.INFLUCODER_ENCODER_MODEL,
        "wall_s": wall_s, "setup_s": setup_s, "total_s": total_s,
        "gpu": mod.detect_gpu_label(),
    })
    print(f"    -> {json.dumps({k: v for k, v in rec.items() if k != 'gpu'})}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    summary[key] = rec
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"wrote {SUMMARY} (key={key})")


if __name__ == "__main__":
    main()
