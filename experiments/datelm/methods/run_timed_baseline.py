#!/usr/bin/env python3
"""Unified timing harness for DATE-LM's dattri.py-backed attribution methods
(Grad_Dot, Grad_Sim, DataInf, EKFAC, LESS) on the Counterfact application
task -- one script, one code path, one timing methodology for every method,
so the numbers are directly comparable to each other and to the separately
-hosted Rep-Sim (`run_repsim_counterfact.py`) and InfluCoder
(`run_influcoder_400m_best_counterfact.py`) timings on the same GPU.

Imports `attribute()` from `dattri.py` directly (no subprocess) and times
exactly the checkpoint-load + attribute() call, matching what those two
sibling scripts measure. Evaluates immediately via evaluate_application.py's
own scoring function (imported directly, not shelled out, to avoid a second
process-start cost skewing anything) and appends a structured record to
results/timing_summary.json so results accumulate across methods/runs.

Usage:
    python methods/run_timed_baseline.py --method Grad_Dot
    python methods/run_timed_baseline.py --method Grad_Sim
    python methods/run_timed_baseline.py --method DataInf
    python methods/run_timed_baseline.py --method EKFAC
    python methods/run_timed_baseline.py --method LESS
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


def detect_gpu_label() -> str:
    """Auto-detect the actual GPU this run is on -- never hardcode a model
    name, since this script is meant to run on whichever GPU is reserved at
    the time, and a stale hardcoded label would silently mislabel results."""
    try:
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        name = "UNKNOWN"
    return f"{name} ({socket.getfqdn()})"

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from methods.dattri import attribute  # noqa: E402

CONFIG_MAP = {
    "Grad_Dot": "configs/factual-attribution-graddot.yaml",
    "Grad_Sim": "configs/factual-attribution-gradsim.yaml",
    "DataInf": "configs/factual-attribution-datainf.yaml",
    "EKFAC": "configs/factual-attribution-ekfac.yaml",
    "LESS": "configs/factual-attribution-less.yaml",
}

SUMMARY_PATH = Path("results/timing_summary.json")


sys.path.insert(0, str(project_root / "evaluation"))
from evaluate_application import (  # noqa: E402
    get_fact_indices_counterfact,
    evaluate_fact,
    read_json,
)


def evaluate(score_path: Path):
    """Uses evaluate_application.py's own Counterfact ground-truth/scoring
    functions directly (imported, not reimplemented) -- get_fact_indices_counterfact
    is the real ground-truth definition; do not guess at it."""
    from datamodules.load_data import get_dataset

    train, ref = get_dataset("Counterfact", "Pythia-1b")
    score = read_json(str(score_path))
    fact_indices_list = get_fact_indices_counterfact(train, ref)
    recall50, mrr = evaluate_fact(score, fact_indices_list, k=50)
    return recall50, mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(CONFIG_MAP.keys()))
    args = parser.parse_args()

    config_path = CONFIG_MAP[args.method]
    config = OmegaConf.load(config_path)

    print(f"=== timing {args.method} (Counterfact/Pythia-1b) ===")
    t0 = time.perf_counter()
    attribute(config)
    wall_s = time.perf_counter() - t0
    print(f"{args.method} wall time: {wall_s:.1f}s")

    score_path = Path(config.save_path) / f"{args.method}.pt"
    recall50, mrr = evaluate(score_path)
    print(f"{args.method} Recall@50={recall50:.4f} MRR={mrr:.4f}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
    summary[args.method] = {
        "task": "Counterfact",
        "wall_s": wall_s,
        "recall_at_50": recall50,
        "mrr": mrr,
        "gpu": detect_gpu_label(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
