#!/usr/bin/env python3
"""Does batch_size move LoGRA's wall-clock ms/sample, and does the achievable
batch size actually scale down with model size the way it should?

figure1_table.py times every LoGRA row (r8/4B, proxy 1.7B, proxy 0.6B) at the
same hardcoded batch_size=1 -- apples-to-apples for isolating the rank/model
effect, but NOT representative of each model's real achievable throughput,
since a 0.6B model has far more unused GPU memory at batch_size=1 than a 4B
model does and should be run at a correspondingly larger batch in practice.
This sweeps batch_size per model (catching OOM) and checks scores don't drift
from the batch_size=1 reference, so a real win here isn't silently wrong.

    python -m baselines.time_logra_batch --grad_model Qwen/Qwen3-0.6B --rank 32 \
        --batch_sizes 1 4 8 16 32 64
"""
from __future__ import annotations

import argparse
import time

import torch

from baselines.common import ground_truth
from baselines.cost import sync
from baselines.logra.score import score_logra

PRESET = "fig1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grad_model", default="Qwen/Qwen3-4B")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--attn_implementation", default="sdpa")
    args = ap.parse_args()

    gt, splits, cfg = ground_truth(PRESET, seed=0, grad_model="Qwen/Qwen3-4B", lora_rank=16)
    n_samples = gt.shape[0] + gt.shape[1]

    ref = None
    for bs in args.batch_sizes:
        t0 = time.perf_counter()
        try:
            out = score_logra(splits, args.grad_model, lora_rank=args.rank,
                              max_len=cfg["grad_max_len"], seed=0, batch_size=bs,
                              attn_implementation=args.attn_implementation)
        except torch.cuda.OutOfMemoryError:
            print(f"model={args.grad_model} attn={args.attn_implementation} batch_size={bs}: OOM")
            torch.cuda.empty_cache()
            continue
        sync()
        dt = time.perf_counter() - t0
        raw = out["logra_raw"]
        drift = "" if ref is None else f", max|delta| vs bs={args.batch_sizes[0]}: {(raw - ref).abs().max().item():.2e}"
        if ref is None:
            ref = raw
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"model={args.grad_model} attn={args.attn_implementation} batch_size={bs:3d}: {dt:.2f}s total, "
              f"{1000*dt/n_samples:.3f} ms/sample, peak {peak_mem:.1f}GB{drift}")
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
