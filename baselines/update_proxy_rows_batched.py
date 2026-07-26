#!/usr/bin/env python3
"""Patch table1.json in place: re-score the two LoGRA PROXY rows (1.7B, 0.6B
at rank=32) at their max-feasible batch size instead of the official
batch_size=1. batch_size=4 for both -- the ceiling found in
baselines/time_logra_batch.py's sweep (both OOM at batch=8; the main
logra_r8/4B row already OOMs at batch=4, so it is left untouched at
batch_size=1 and this script does not re-score it).

User-requested override, NOT the default methodology: batching moves the
reported aggregated Spearman rho by a real, non-noise amount (see
INFLUCODER_FINDINGS.md's batching section / verify_logra_sdpa.py --
aggregated rho delta -0.0130 at batch=4 vs batch=1 for the 0.6B proxy). This
means these two rows are no longer directly comparable to logra_r8's
batch=1 number in the same table, or to the Dolly-pool fig1 table (which
uses batch=1 throughout). Kept as a separate script rather than folded into
figure1_table.py's default so the apples-to-apples batch=1 methodology stays
the one-command reproducible path.

    python -m baselines.update_proxy_rows_batched --preset fig1_dolci
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.common import PRESETS, ground_truth
from baselines.figure1_table import MODEL, GT_LORA_RANK, run_row
from baselines.logra.score import score_logra

BATCH_SIZE = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fig1_dolci", choices=list(PRESETS))
    args = ap.parse_args()
    table = Path("baselines/out") / args.preset / "table1.json"

    data = json.loads(table.read_text())
    gt, splits, cfg = ground_truth(args.preset, seed=0, grad_model=MODEL, lora_rank=GT_LORA_RANK)
    n_samples = gt.shape[0] + gt.shape[1]

    updated = {}
    for label, grad_model in [("logra_proxy_1.7B", "Qwen/Qwen3-1.7B"),
                              ("logra_proxy_0.6B", "Qwen/Qwen3-0.6B")]:
        def fn(meter, gm=grad_model):
            variants = score_logra(splits, gm, lora_rank=32, max_len=cfg["grad_max_len"],
                                   seed=0, batch_size=BATCH_SIZE, meter=meter)
            return variants["logra_raw"]
        run_row(label, fn, splits, cfg, gt, n_samples, updated)

    for label, m in updated.items():
        data["methods"][label] = {k: v for k, v in m.items() if k != "per_anchor"}
        data["methods"][label]["batch_size"] = BATCH_SIZE

    table.write_text(json.dumps(data, indent=2))
    print(f"\nwrote {table}")
    for label, m in updated.items():
        print(f"  {label}: agg {m['aggregated']:+.4f}, {m['time_per_sample_ms']:.3f} ms/sample "
              f"(batch_size={BATCH_SIZE})")


if __name__ == "__main__":
    main()
