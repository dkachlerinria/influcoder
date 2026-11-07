#!/usr/bin/env python3
"""Quick empirical check: does LoGRA rank actually move wall-clock ms/sample?

Timing-only (no FlopCounterMode pass -- that instrumentation itself roughly
doubles wall time and isn't needed to answer "is r=4 faster than r=8").
Same fig1 eval splits, same grad_model, same GT cache -- rank is the only
variable.
"""
from __future__ import annotations

import time

from baselines.common import ground_truth
from baselines.cost import sync
from baselines.logra.score import score_logra

MODEL = "Qwen/Qwen3-4B"
PRESET = "fig1"

def main():
    gt, splits, cfg = ground_truth(PRESET, seed=0, grad_model=MODEL, lora_rank=16)
    n_samples = gt.shape[0] + gt.shape[1]

    for rank in [8, 4]:
        t0 = time.perf_counter()
        score_logra(splits, MODEL, lora_rank=rank, max_len=cfg["grad_max_len"], seed=0)
        sync()
        dt = time.perf_counter() - t0
        print(f"rank={rank}: {dt:.2f}s total, {1000*dt/n_samples:.3f} ms/sample")

if __name__ == "__main__":
    main()
