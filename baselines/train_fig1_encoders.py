#!/usr/bin/env python3
"""Train the 3 InfluCoder encoder sizes needed for Figure 1, all under the
`fig1` preset (400x400 eval, 1500x3000 train, Qwen3-4B GT). Unlike
tune_influcoder.py, this ALWAYS saves every size (not just new global
bests) -- Figure 1 needs the 400m/150m/68m rows side by side, not just
whichever one happens to score highest.

GT and train-side gradient features are built/cached once and shared across
all 3 encoder trainings (only the encoder differs), so this pays the
Qwen3-4B featurization cost a single time.

Recipe (the plateau-quality config found during tuning, without hard
mining per the decision to drop it): lr=5e-5, epochs=8, hard_ratio=0.0,
alpha=0.5 (default -- the alpha=0.3 gain was inside the noise band).

    python -m baselines.train_fig1_encoders
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baselines.common import PRESETS, ground_truth
from baselines.scaling_sweep import train_features
from influcoder.encoder import distill, embed, load_encoder
from influcoder.metrics import spearman_metrics

MODEL = "Qwen/Qwen3-4B"
GT_LORA_RANK = 16

ENCODERS = {
    "68m": "jhu-clsp/ettin-encoder-68m",
    "150m": "jhu-clsp/ettin-encoder-150m",
    "400m": "jhu-clsp/ettin-encoder-400m",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fig1", choices=list(PRESETS))
    args = ap.parse_args()
    preset = args.preset
    out_dir = Path("runs_out") / preset
    results_path = Path("baselines/out") / preset / "train_results.json"

    results_path.parent.mkdir(parents=True, exist_ok=True)
    gt, splits, cfg = ground_truth(preset, seed=0, grad_model=MODEL, lora_rank=GT_LORA_RANK)
    print(f"GT ready: {gt.shape[0]}x{gt.shape[1]}")
    g_ta, g_tp = train_features(splits, cfg, MODEL, GT_LORA_RANK, 0)
    n_a, n_p = cfg["n_train_a"], cfg["n_train_p"]
    targets = g_ta[:n_a] @ g_tp[:n_p].T
    print(f"train features ready: anchors {tuple(g_ta.shape)}, pool {tuple(g_tp.shape)}")

    eval_a = [s.text for s in splits["eval_anchors"]]
    eval_p = [s.text for s in splits["eval_pool"]]
    train_a = [s.text for s in splits["train_anchors"][:n_a]]
    train_p = [s.text for s in splits["train_pool"][:n_p]]

    results = {}
    for size, model_name in ENCODERS.items():
        print(f"\n########## training InfluCoder ({size}) ##########")
        t0 = time.time()
        enc = load_encoder(model_name, max_seq_len=cfg.get("encoder_max_len", 512))

        def ev():
            return spearman_metrics(embed(enc, eval_a) @ embed(enc, eval_p).T, gt)

        untrained = ev()
        log = distill(enc, train_a, train_p, targets, epochs=cfg["epochs"],
                      hard_ratio=cfg["hard_ratio"], epoch_eval=ev,
                      select_best_on="aggregated", seed=0)
        best = log["epoch_metrics"][log["best_epoch"]]
        elapsed = time.time() - t0

        save_dir = out_dir / f"encoder_{size}"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        enc.save(str(save_dir))
        print(f"  untrained agg {untrained['aggregated']:+.4f}  "
              f"best-epoch agg {best['aggregated']:+.4f} "
              f"(epoch {log['best_epoch'] + 1}/{cfg['epochs']})  "
              f"elapsed {elapsed:.0f}s  saved -> {save_dir}")

        results[size] = {
            "model": model_name, "encoder_dir": str(save_dir),
            "untrained_agg": untrained["aggregated"],
            "best_epoch": log["best_epoch"] + 1,
            "best_agg": best["aggregated"], "best_per_anchor": best["per_anchor_mean"],
            "elapsed_s": round(elapsed, 1),
        }
        del enc
        import torch
        torch.cuda.empty_cache()

    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
