#!/usr/bin/env python3
"""Evaluate competitor baselines on the *same* eval run.py reports.

Builds the identical eval split + gradient-influence ground truth for a preset
(cached), runs the requested method to produce an [anchors x pool] score matrix,
and prints its Spearman agreement with GT in the same table shape as run.py.

    python -m baselines.eval_baselines --method logra --preset big
    python -m baselines.eval_baselines --method less semantic rdsplus --preset big
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from baselines.common import DEFAULT_GRAD_MODEL, ground_truth, report
from baselines.cost import CostMeter, flop_counter, summarize, sync

METHODS = ["influcoder", "logra", "less", "semantic", "rdsplus", "tfidf", "random"]


def run_method(method: str, splits, cfg, args, meter) -> dict:
    """Returns {variant_name: score_tensor}. A method may emit >1 variant."""
    if method == "influcoder":
        from baselines.influcoder.score import score_influcoder
        enc_dir = args.encoder_dir or f"runs_out/{args.preset}/encoder"
        return {"influcoder": score_influcoder(splits, enc_dir,
                                               max_len=args.encoder_max_len,
                                               meter=meter)}
    if method == "logra":
        from baselines.logra.score import score_logra
        return score_logra(splits, args.grad_model, lora_rank=args.logra_rank,
                           max_len=cfg["grad_max_len"], seed=args.seed, meter=meter)
    if method == "less":
        from baselines.less.score import score_less
        # LESS gets its OWN projection dim, not the GT's. The old pipeline ran
        # GT at 65536 and LESS at 8192 on purpose; scoring LESS against a GT
        # sketched at its own width makes it a near-self-comparison.
        return {"less": score_less(splits, args.grad_model,
                                   proj_dim=args.less_proj_dim,
                                   lora_rank=args.less_lora_rank,
                                   lora_alpha=args.less_lora_alpha,
                                   max_len=cfg["grad_max_len"], meter=meter)}
    if method == "semantic":
        from baselines.semantic.score import score_semantic
        return {"semantic": score_semantic(splits, args.encoder_model,
                                           max_len=args.encoder_max_len,
                                           meter=meter)}
    if method == "rdsplus":
        from baselines.rdsplus.score import score_rdsplus
        return {"rdsplus": score_rdsplus(splits, args.grad_model,
                                         max_len=cfg["grad_max_len"], meter=meter)}
    if method == "tfidf":
        from baselines.tfidf.score import score_tfidf
        return {"tfidf": score_tfidf(splits, meter=meter)}
    if method == "random":
        from baselines.random.score import score_random
        return {"random": score_random(splits, seed=args.seed, meter=meter)}
    raise ValueError(f"unknown method: {method}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", nargs="+", choices=METHODS, required=True)
    ap.add_argument("--preset", default="big", choices=list(__import__("run").PRESETS))
    ap.add_argument("--grad_model", default=DEFAULT_GRAD_MODEL)
    ap.add_argument("--encoder_model", default="jhu-clsp/ettin-encoder-68m")
    # Distinct knobs on purpose -- the old config_influence.sh set LORA_RANK=16
    # for the GT/LESS featurizer but LOGRA_RANK=8, and LESS_PROJ_DIM=8192
    # against GT_PROJ_DIM=65536. Driving them all off one flag silently changes
    # the experiment.
    ap.add_argument("--lora_rank", type=int, default=8, help="GT featurizer LoRA rank")
    ap.add_argument("--logra_rank", type=int, default=8, help="LoGRA projection rank")
    ap.add_argument("--less_proj_dim", type=int, default=8192)
    ap.add_argument("--less_lora_rank", type=int, default=128)
    ap.add_argument("--less_lora_alpha", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder_dir", default=None,
                    help="trained influcoder encoder; default runs_out/<preset>/encoder")
    ap.add_argument("--no_flops", action="store_true",
                    help="skip the FLOP-counting pass (timing only, ~2x faster)")
    ap.add_argument("--encoder_max_len", type=int, default=512,
                    help="matches run.py's load_encoder default, so the semantic / "
                         "influcoder rows reproduce run.py's numbers exactly")
    ap.add_argument("--out_dir", default=None, help="default: baselines/out/<preset>")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or f"baselines/out/{args.preset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"building ground truth for preset '{args.preset}' ...")
    gt, splits, cfg = ground_truth(args.preset, seed=args.seed,
                                   grad_model=args.grad_model, lora_rank=args.lora_rank)
    print(f"GT ready: {gt.shape[0]} anchors x {gt.shape[1]} pool")

    n_samples = gt.shape[0] + gt.shape[1]  # every sample must be featurized once
    all_metrics = {}
    for method in args.method:
        print(f"\n########## {method} ##########")

        # Two passes on purpose. FlopCounterMode is a TorchDispatchMode: it
        # intercepts every aten op, which inflates wall-clock several-fold. Timing
        # under the counter would report the counter's overhead, not the method's
        # cost -- so time is measured in a clean pass and FLOPs in a counted one.
        # Scores come from the clean pass; both passes are deterministic.
        meter = CostMeter()
        t0 = time.perf_counter()
        variants = run_method(method, splits, cfg, args, meter)
        sync()  # CUDA is async; settle before stopping the clock
        total_time = time.perf_counter() - t0

        measured = 0
        if not args.no_flops:
            flop_meter = CostMeter()
            with flop_counter() as fc:
                run_method(method, splits, cfg, args, flop_meter)
                sync()
            measured = fc.total()
            meter.extra_flops = flop_meter.extra_flops
            meter.notes = flop_meter.notes

        cost = summarize(measured, meter, total_time, n_samples)
        # Variants of one method share the encoding pass, so they share its cost.
        for name, scores in variants.items():
            m = report(name, scores, gt)
            m.update(cost)
            all_metrics[name] = m
            print(f"  {name}: {cost['flops_per_sample']:.3e} FLOPs/sample, "
                  f"{cost['time_per_sample_ms']:.2f} ms/sample "
                  f"(inference {cost['inference_time_s']:.1f}s, "
                  f"load {cost['load_time_s']:.1f}s)")
        for note in meter.notes:
            print(f"  [analytic] {note}")

    results = {
        "config": {"preset": args.preset, "grad_model": args.grad_model,
                   "gt_shape": list(gt.shape), "seed": args.seed},
        "methods": {name: {k: v for k, v in m.items() if k != "per_anchor"}
                    for name, m in all_metrics.items()},
    }
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
