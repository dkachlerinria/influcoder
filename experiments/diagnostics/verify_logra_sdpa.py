#!/usr/bin/env python3
"""Two correctness checks before adopting sdpa/batching for LoGRA's official
timing numbers:

  1. eager batch=1 vs sdpa batch=1 -- pure attention-kernel swap, no loss-
     dilution possible (batch of one), should be near-identical. If it isn't,
     something is actually wrong, not just "different but fine."
  2. sdpa batch=1 vs sdpa batch=4 -- does the raw-matrix drift we already
     measured actually move the AGGREGATE SPEARMAN metric we report, or does
     rank correlation wash it out since it only cares about ordering?

Reports aggregated rho (the number that goes in table1.json), not just raw
matrix delta, so this is a decision on the actual reported metric.
"""
from __future__ import annotations

from baselines.common import ground_truth, report
from baselines.logra.score import score_logra

PRESET = "fig1"
GRAD_MODEL = "Qwen/Qwen3-0.6B"
RANK = 32


def main():
    gt, splits, cfg = ground_truth(PRESET, seed=0, grad_model="Qwen/Qwen3-4B", lora_rank=16)

    configs = [
        ("eager_bs1", "eager", 1),
        ("sdpa_bs1", "sdpa", 1),
        ("sdpa_bs4", "sdpa", 4),
    ]
    results = {}
    for name, attn, bs in configs:
        out = score_logra(splits, GRAD_MODEL, lora_rank=RANK, max_len=cfg["grad_max_len"],
                          seed=0, batch_size=bs, attn_implementation=attn)
        m = report(name, out["logra_raw"], gt)
        results[name] = (out["logra_raw"], m["aggregated"])
        print(f"{name}: aggregated rho = {m['aggregated']:+.4f}")

    print("\n=== deltas ===")
    e1, e1_agg = results["eager_bs1"]
    s1, s1_agg = results["sdpa_bs1"]
    s4, s4_agg = results["sdpa_bs4"]
    print(f"eager_bs1 vs sdpa_bs1: max|raw delta| = {(e1 - s1).abs().max().item():.2e}, "
          f"aggregated rho delta = {s1_agg - e1_agg:+.4f}")
    print(f"sdpa_bs1 vs sdpa_bs4: max|raw delta| = {(s1 - s4).abs().max().item():.2e}, "
          f"aggregated rho delta = {s4_agg - s1_agg:+.4f}")


if __name__ == "__main__":
    main()
