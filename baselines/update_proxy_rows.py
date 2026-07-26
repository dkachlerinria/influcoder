#!/usr/bin/env python3
"""Patch table1.json in place: re-score the three LoGRA rows (r8/4B, proxy
1.7B, proxy 0.6B at rank=32; raw only -- see figure1_table.py's comment for
why) instead of re-running the full 12-row table. Timing-only
(measure_flops=False) -- FLOPs are deferred to an appendix experiment, see
run_row's docstring. score_logra now defaults attn_implementation="sdpa"
(verified in baselines/verify_logra_sdpa.py: aggregated rho moves only
+0.0046 vs the old eager default -- noise-level -- while cutting ms/sample
~10-28%); batch_size stays at 1 (verified unsafe to raise -- see
INFLUCODER_FINDINGS.md's batching section).
"""
from __future__ import annotations

import json
from pathlib import Path

from baselines.common import ground_truth
from baselines.figure1_table import MODEL, GT_LORA_RANK, PRESET, OUT_DIR, run_row
from baselines.logra.score import score_logra

TABLE = OUT_DIR / "table1.json"


def main():
    data = json.loads(TABLE.read_text())
    gt, splits, cfg = ground_truth(PRESET, seed=0, grad_model=MODEL, lora_rank=GT_LORA_RANK)
    n_samples = gt.shape[0] + gt.shape[1]

    updated = {}
    for label, grad_model, rank in [("logra_r8", MODEL, 8),
                                    ("logra_proxy_1.7B", "Qwen/Qwen3-1.7B", 32),
                                    ("logra_proxy_0.6B", "Qwen/Qwen3-0.6B", 32)]:
        def fn(meter, gm=grad_model, r=rank):
            variants = score_logra(splits, gm, lora_rank=r,
                                   max_len=cfg["grad_max_len"], seed=0, meter=meter)
            return variants["logra_raw"]
        run_row(label, fn, splits, cfg, gt, n_samples, updated)

    for label, m in updated.items():
        data["methods"][label] = {k: v for k, v in m.items() if k != "per_anchor"}

    TABLE.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {TABLE}")
    for label, m in updated.items():
        print(f"  {label}: agg {m['aggregated']:+.4f}, {m['time_per_sample_ms']:.3f} ms/sample")


if __name__ == "__main__":
    main()
