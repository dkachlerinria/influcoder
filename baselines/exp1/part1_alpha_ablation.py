#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

import os as _os
if _os.environ.get("EXP1_CONFIG") == "biggpu":
    from baselines.exp1 import config_biggpu as cfg
else:
    from baselines.exp1 import config as cfg
from baselines.exp1 import data, train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=50, help="Eval size (default 50 for quick test)")
    ap.add_argument("--epochs", type=int, default=4, help="Epochs to train")
    args = ap.parse_args()

    gt, splits, preset_cfg = data.load_gt_and_splits(n_eval=args.n_eval)
    eval_a_texts = [s.text for s in splits["eval_anchors"]]
    eval_p_texts = [s.text for s in splits["eval_pool"]]

    g_ta, g_tp = data.load_train_features(splits, preset_cfg)
    targets = g_ta[:cfg.N_TRAIN_A] @ g_tp[:cfg.N_TRAIN_P].T
    train_a = [s.text for s in splits["train_anchors"][:cfg.N_TRAIN_A]]
    train_p = [s.text for s in splits["train_pool"][:cfg.N_TRAIN_P]]

    alphas = [0.5, 0.75, 1.0]
    seeds = [0, 1, 2]
    all_metrics = {}

    print(f"Resuming alpha ablation: {alphas} over seeds {seeds} for Influcoder 68m ({args.epochs} epochs)")
    
    for alpha in alphas:
        all_metrics[alpha] = {}
        for seed in seeds:
            if alpha == 0.5 and seed == 0:
                continue
            print(f"\n########## Alpha = {alpha}, Seed = {seed} ##########")
            enc, log, final, untrained = train.train_influcoder(
                cfg.ENCODER_MODELS["68m"], train_a, train_p, targets,
                eval_anchor_texts=eval_a_texts, eval_pool_texts=eval_p_texts, gt=gt,
                alpha=alpha, epochs=args.epochs, seed=seed
            )
            if final:
                print(f"Alpha {alpha} (seed {seed}): agg rho={final['aggregated']:+.4f} (best epoch {log['best_epoch']+1})")
                all_metrics[alpha][seed] = {
                    "agg_rho": final["aggregated"],
                    "best_epoch": log["best_epoch"] + 1,
                    "per_anchor": final["per_anchor_mean"]
                }
            train.free(enc)
            train.free(untrained)
            del enc, log, final, untrained
            import torch
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    OUT = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / "alpha_ablation_seeds_3.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nWrote ablation results to {OUT}")
    
    print("\nSummary:")
    for a, seeds_dict in all_metrics.items():
        avg_rho = np.mean([res["agg_rho"] for res in seeds_dict.values()])
        print(f"  alpha={a}: avg agg rho={avg_rho:+.4f} (across {len(seeds_dict)} seeds)")
        for s, res in seeds_dict.items():
            print(f"    seed={s}: {res['agg_rho']:+.4f}")

if __name__ == "__main__":
    main()
