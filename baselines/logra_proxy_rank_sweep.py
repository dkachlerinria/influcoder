#!/usr/bin/env python3
"""Do the LoGRA proxies (1.7B, 0.6B) get better at rank 16/32?

table1.json's rank=8 result: proxy 1.7B agg +0.287, proxy 0.6B agg -0.158 --
both far below logra_r8 (4B, +0.889). Hypothesis: a smaller model needs more
LoRA rank to carry a comparable amount of gradient signal. Same fig1 eval
splits/GT as everything else; only grad_model and rank vary. Reports both
logra_raw and logra_fim so we can see if FIM-preconditioning behaves
differently as rank grows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from baselines.common import ground_truth, report
from baselines.logra.score import score_logra

MODEL = "Qwen/Qwen3-4B"
PRESET = "fig1"
OUT = Path("baselines/out/fig1/logra_proxy_rank_sweep.json")

PROXIES = {"1.7B": "Qwen/Qwen3-1.7B", "0.6B": "Qwen/Qwen3-0.6B"}
RANKS = [16, 32]


def main():
    gt, splits, cfg = ground_truth(PRESET, seed=0, grad_model=MODEL, lora_rank=16)
    results = {}
    for size, grad_model in PROXIES.items():
        for rank in RANKS:
            key = f"proxy_{size}_r{rank}"
            print(f"\n########## {key} ##########")
            t0 = time.perf_counter()
            variants = score_logra(splits, grad_model, lora_rank=rank,
                                   max_len=cfg["grad_max_len"], seed=0)
            dt = time.perf_counter() - t0
            raw_m = report(f"{key}_raw", variants["logra_raw"], gt)
            fim_m = report(f"{key}_fim", variants["logra_fim"], gt)
            results[key] = {
                "grad_model": grad_model, "rank": rank, "wall_s": dt,
                "logra_raw": {"aggregated": raw_m["aggregated"],
                             "per_anchor_mean": raw_m["per_anchor_mean"]},
                "logra_fim": {"aggregated": fim_m["aggregated"],
                             "per_anchor_mean": fim_m["per_anchor_mean"]},
            }
            print(f"  raw: agg {raw_m['aggregated']:+.4f}  "
                  f"fim: agg {fim_m['aggregated']:+.4f}  ({dt:.0f}s)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    print(f"\n{'config':16s}{'raw agg':>10s}{'fim agg':>10s}")
    for key, r in results.items():
        print(f"{key:16s}{r['logra_raw']['aggregated']:>+10.4f}"
              f"{r['logra_fim']['aggregated']:>+10.4f}")


if __name__ == "__main__":
    main()
