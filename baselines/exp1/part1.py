#!/usr/bin/env python3
"""EXP1 Part 1: Cost vs Quality -- every method, one eval, one config.

For a fixed eval, how good is each method (aggregate Spearman rho vs. the
Qwen3-4B gradient-influence ground truth) and how expensive is it per
candidate (inference ms/sample)? Every method is scored through
`baselines.exp1.methods`'s canonical wrappers, so every rank/attn/max_len
value comes from `baselines.exp1.config` -- see that module's docstring for
why this matters.

InfluCoder's 68m/150m checkpoints are trained once (at config.N_TRAIN_A/
N_TRAIN_P, the "anchor point" Part 2's sweep also passes through -- see
EXP1.md section 4.2.2) and reused on subsequent runs; pass --retrain to force
retraining even if a checkpoint already exists at runs_out/<preset>/encoder_*.

    python -m baselines.exp1.part1
    python -m baselines.exp1.part1 --n_eval 50   # fast peek, not the default
    python -m baselines.exp1.part1 --retrain
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baselines.cost import CostMeter, summarize, sync
from baselines.common import report

import os as _os
if _os.environ.get("EXP1_CONFIG") == "biggpu":
    from . import config_biggpu as cfg  # BIG_GPU_FINAL_EXP1 -- see that module's docstring
else:
    from . import config as cfg
from . import data, methods, train

ENC_DIR = Path("runs_out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED)
OUT = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED) / "exp1_part1.json"


def ensure_checkpoint(size: str, encoder_model: str, splits, full_gt_eval_texts, retrain: bool):
    save_dir = ENC_DIR / f"encoder_{size}"
    if save_dir.exists() and not retrain:
        print(f"  [{size}] checkpoint already exists at {save_dir}, skipping training "
              f"(pass --retrain to force)")
        return save_dir

    print(f"  [{size}] training fresh checkpoint: {cfg.N_TRAIN_A}x{cfg.N_TRAIN_P} "
          f"anchors/pool, {cfg.INFLUCODER_EPOCHS} epochs")
    g_ta, g_tp = data.load_train_features(splits, full_gt_eval_texts["preset_cfg"])
    targets = g_ta[:cfg.N_TRAIN_A] @ g_tp[:cfg.N_TRAIN_P].T
    train_a = [s.text for s in splits["train_anchors"][:cfg.N_TRAIN_A]]
    train_p = [s.text for s in splits["train_pool"][:cfg.N_TRAIN_P]]

    eval_a, eval_p, gt = full_gt_eval_texts["eval_a"], full_gt_eval_texts["eval_p"], full_gt_eval_texts["gt"]

    enc, log, final, _untrained = train.train_influcoder(
        encoder_model, train_a, train_p, targets,
        eval_anchor_texts=eval_a, eval_pool_texts=eval_p, gt=gt)
    print(f"  [{size}] final-epoch agg={final['aggregated']:+.4f} "
          f"(vs. best_epoch={log['best_epoch'] + 1}/{cfg.INFLUCODER_EPOCHS} internally)")

    save_dir.parent.mkdir(parents=True, exist_ok=True)
    enc.save(str(save_dir))
    train.free(enc)
    return save_dir


def run_row(name, fn, gt, n_samples, all_metrics):
    print(f"\n########## {name} ##########")
    meter = CostMeter()
    t0 = time.perf_counter()
    scores = fn(meter)
    sync()
    total_time = time.perf_counter() - t0
    cost = summarize(0, meter, total_time, n_samples)
    m = report(name, scores, gt)
    m.update(cost)
    all_metrics[name] = m
    print(f"  {name}: {cost['time_per_sample_ms']:.2f} ms/sample "
          f"(inference {cost['inference_time_s']:.1f}s, load {cost['load_time_s']:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_eval", type=int, default=cfg.N_EVAL,
                    help=f"eval slice size (default: {cfg.N_EVAL}, the full fig1_dolci eval)")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain InfluCoder checkpoints even if they already exist")
    args = ap.parse_args()

    gt, splits, preset_cfg = data.load_gt_and_splits(n_eval=args.n_eval)
    n_samples = gt.shape[0] + gt.shape[1]
    print(f"EXP1 Part 1 | {args.n_eval}x{args.n_eval} eval slice of the cached "
          f"{preset_cfg['n_eval_a']}x{preset_cfg['n_eval_p']} fig1_dolci eval | "
          f"attn={cfg.ATTN} max_len={cfg.MAX_LEN} | LESS rank={cfg.LESS_RANK} | "
          f"LoGRA rank={cfg.LOGRA_RANK}\n")

    eval_a_texts = [s.text for s in splits["eval_anchors"]]
    eval_p_texts = [s.text for s in splits["eval_pool"]]
    ctx = {"preset_cfg": preset_cfg, "eval_a": eval_a_texts, "eval_p": eval_p_texts, "gt": gt}

    all_metrics = {}

    # -- InfluCoder + untrained encoder (68m, 150m) ---------------------------
    for size, model_name in cfg.ENCODER_MODELS.items():
        encoder_dir = ensure_checkpoint(size, model_name, splits, ctx, args.retrain)
        run_row(f"influcoder_{size}", lambda meter, d=str(encoder_dir):
               methods.run_influcoder(splits, d, meter=meter), gt, n_samples, all_metrics)
        run_row(f"untrained_{size}", lambda meter, m=model_name:
               methods.run_untrained(splits, m, meter=meter), gt, n_samples, all_metrics)

    # -- LESS (4B, 1.7B, 0.6B), same rank across sizes ------------------------
    for label, model_name in cfg.LESS_MODEL_SIZES.items():
        run_row(f"less_{label}", lambda meter, m=model_name:
               methods.run_less(splits, m, meter=meter), gt, n_samples, all_metrics)

    # -- LoGRA (4B, 1.7B, 0.6B), same rank across sizes -----------------------
    for label, model_name in cfg.LOGRA_MODEL_SIZES.items():
        run_row(f"logra_{label}", lambda meter, m=model_name:
               methods.run_logra(splits, m, meter=meter), gt, n_samples, all_metrics)

    # -- RDS+ ------------------------------------------------------------------
    run_row("rdsplus", lambda meter: methods.run_rdsplus(splits, meter=meter),
           gt, n_samples, all_metrics)

    # -- TF-IDF ------------------------------------------------------------------
    run_row("tfidf", lambda meter: methods.run_tfidf(splits, meter=meter),
           gt, n_samples, all_metrics)

    # less_fingerprint/logra_fingerprint are the single source of truth for
    # "what config produced this row" -- Part 2 (and anything else that might
    # reuse a LESS/LoGRA row instead of recomputing it) compares against
    # exactly these same functions, so the writer and the reuse-checker can
    # never drift out of sync with each other. They share several keys
    # (n_eval/preset/gt_model/gt_lora_rank/attn/max_len/seed) with identical
    # values, so merging is safe.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "config": {**methods.less_fingerprint(args.n_eval),
                  **methods.logra_fingerprint(args.n_eval),
                  "n_train_a": cfg.N_TRAIN_A, "n_train_p": cfg.N_TRAIN_P,
                  "influcoder_epochs": cfg.INFLUCODER_EPOCHS,
                  "influcoder_lr": cfg.INFLUCODER_LR,
                  "influcoder_hard_ratio": cfg.INFLUCODER_HARD_RATIO},
        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "per_anchor"}
                   for k, v in all_metrics.items()},
    }, indent=2))
    print(f"\nwrote {OUT}")

    print(f"\n{'method':24s}{'agg rho':>10s}{'ms/sample':>14s}")
    for name, m in all_metrics.items():
        print(f"{name:24s}{m['aggregated']:>+10.4f}{m['time_per_sample_ms']:>14.3f}")


if __name__ == "__main__":
    main()
