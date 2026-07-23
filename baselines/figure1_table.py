#!/usr/bin/env python3
"""Figure 1, table 1: inference speed (ms/sample) vs. aggregate Spearman,
every method scored on the SAME 400x400 eval (`fig1` preset, GT from
Qwen/Qwen3-4B, LoRA rank 16). One seed.

Methods:
  - LESS                          (grad_model = Qwen3-4B, full fidelity)
  - LoGRA (rank=8)                 (grad_model = Qwen3-4B, raw variant)
  - LoGRA proxy (1.7B)              (grad_model = Qwen3-1.7B, same-family cheap proxy)
  - LoGRA proxy (0.6B)              (grad_model = Qwen3-0.6B, cheaper still)
  - InfluCoder (400m/150m/68m)      (trained encoders from train_fig1_encoders.py)
  - untrained encoder (400m/150m/68m) (the same 3 encoders pre-distillation)
  - RDS+                           (grad_model = Qwen3-4B, no gradients -- hidden states)
  - TF-IDF                         (no model)

Cost measured the same way as eval_baselines.py: a clean timing pass + a
FLOP-counted pass (FlopCounterMode inflates wall-clock, so never trust
timing under it), analytic terms added for anything the counter can't see
(LoGRA's pinv).

    python -m baselines.figure1_table
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from baselines.common import PRESETS, ground_truth, report
from baselines.cost import CostMeter, flop_counter, summarize, sync

MODEL = "Qwen/Qwen3-4B"
GT_LORA_RANK = 16
PRESET = "fig1"
OUT_DIR = Path("baselines/out/fig1")
ENC_DIR = Path("runs_out/fig1")

ENCODER_MODELS = {
    "68m": "jhu-clsp/ettin-encoder-68m",
    "150m": "jhu-clsp/ettin-encoder-150m",
    "400m": "jhu-clsp/ettin-encoder-400m",
}


def run_row(name, fn, splits, cfg, gt, n_samples, all_metrics, measure_flops=False):
    """measure_flops=False (current default): timing-only, one clean pass.

    FLOPs are deferred to an appendix experiment -- the FlopCounterMode pass
    roughly 5x's wall time for instrumentation alone and, for a speed-vs-
    quality story, ends up telling the same tale as ms/sample anyway. Set
    measure_flops=True to restore the old clean-pass + FLOP-counted-pass
    behavior when that appendix experiment happens.
    """
    print(f"\n########## {name} ##########")
    meter = CostMeter()
    t0 = time.perf_counter()
    scores = fn(meter)
    sync()
    total_time = time.perf_counter() - t0

    measured = 0
    if measure_flops:
        flop_meter = CostMeter()
        with flop_counter() as fc:
            fn(flop_meter)
            sync()
        measured = fc.total()
        meter.extra_flops = flop_meter.extra_flops
        meter.notes = flop_meter.notes

    cost = summarize(measured, meter, total_time, n_samples)
    m = report(name, scores, gt)
    m.update(cost)
    all_metrics[name] = m
    print(f"  {name}: {cost['time_per_sample_ms']:.2f} ms/sample"
          + (f", {cost['flops_per_sample']:.3e} FLOPs/sample" if measure_flops else ""))
    for note in meter.notes:
        print(f"  [analytic] {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=PRESET, choices=list(PRESETS))
    args = ap.parse_args()
    preset = args.preset
    out_dir = Path("baselines/out") / preset
    enc_dir = Path("runs_out") / preset

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"building ground truth for preset '{preset}' (Qwen3-4B, rank {GT_LORA_RANK}) ...")
    gt, splits, cfg = ground_truth(preset, seed=0, grad_model=MODEL, lora_rank=GT_LORA_RANK)
    print(f"GT ready: {gt.shape[0]} anchors x {gt.shape[1]} pool")
    n_samples = gt.shape[0] + gt.shape[1]

    all_metrics = {}

    # -- LESS -----------------------------------------------------------------
    from baselines.less.score import score_less
    run_row("less", lambda meter: score_less(
        splits, MODEL, proj_dim=8192, lora_rank=128, lora_alpha=512,
        max_len=cfg["grad_max_len"], meter=meter),
        splits, cfg, gt, n_samples, all_metrics)

    # -- LoGRA (rank=8), full-fidelity + two cheaper same-family proxies ------
    # Proxies use rank=32, not 8: a rank sweep (baselines/logra_proxy_rank_sweep.py)
    # found the 1.7B proxy genuinely improves with rank (+0.287 @ r8 -> +0.503 @
    # r16 -> +0.511 @ r32, plateauing), while the 0.6B proxy stays pinned near
    # zero at every rank tested -- its failure isn't a rank/capacity problem.
    # r32 is used for both (matching cost, since rank barely moves LoGRA's wall
    # time -- the dominant cost is the rank-independent backbone forward+
    # backward) so the pair is directly comparable. raw only: logra_fim was
    # consistently much worse than raw for both proxies at every rank tested.
    from baselines.logra.score import score_logra
    for label, grad_model, rank in [("logra_r8", MODEL, 8),
                                    ("logra_proxy_1.7B", "Qwen/Qwen3-1.7B", 32),
                                    ("logra_proxy_0.6B", "Qwen/Qwen3-0.6B", 32)]:
        def fn(meter, gm=grad_model, r=rank):
            variants = score_logra(splits, gm, lora_rank=r,
                                   max_len=cfg["grad_max_len"], seed=0, meter=meter)
            return variants["logra_raw"]
        run_row(label, fn, splits, cfg, gt, n_samples, all_metrics)

    # -- InfluCoder (3 sizes, trained encoders) + untrained encoder baseline --
    from baselines.influcoder.score import score_influcoder
    from baselines.semantic.score import score_semantic
    for size, model_name in ENCODER_MODELS.items():
        encoder_dir = enc_dir / f"encoder_{size}"
        run_row(f"influcoder_{size}", lambda meter, d=str(encoder_dir):
               score_influcoder(splits, d, max_len=512, meter=meter),
               splits, cfg, gt, n_samples, all_metrics)
        run_row(f"untrained_{size}", lambda meter, m=model_name:
               score_semantic(splits, m, max_len=512, meter=meter),
               splits, cfg, gt, n_samples, all_metrics)

    # -- RDS+ -------------------------------------------------------------------
    from baselines.rdsplus.score import score_rdsplus
    run_row("rdsplus", lambda meter: score_rdsplus(
        splits, MODEL, max_len=cfg["grad_max_len"], meter=meter),
        splits, cfg, gt, n_samples, all_metrics)

    # -- TF-IDF -----------------------------------------------------------------
    from baselines.tfidf.score import score_tfidf
    run_row("tfidf", lambda meter: score_tfidf(splits, meter=meter),
           splits, cfg, gt, n_samples, all_metrics)

    results = {
        "config": {"preset": preset, "pool": cfg.get("pool", "dolly"),
                   "grad_model": MODEL, "gt_lora_rank": GT_LORA_RANK,
                   "gt_shape": list(gt.shape), "seed": 0},
        "methods": {name: {k: v for k, v in m.items() if k != "per_anchor"}
                    for name, m in all_metrics.items()},
    }
    with open(out_dir / "table1.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_dir / 'table1.json'}")

    print(f"\n{'method':24s}{'agg rho':>10s}{'ms/sample':>14s}")
    for name, m in all_metrics.items():
        print(f"{name:24s}{m['aggregated']:>+10.4f}{m['time_per_sample_ms']:>14.3f}")


if __name__ == "__main__":
    main()
