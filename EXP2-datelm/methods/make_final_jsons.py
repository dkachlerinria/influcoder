#!/usr/bin/env python3
"""Regenerate the three exp2-*-final.json files from the live fig2_*.json
results, filtered to the kept method set. Safe to re-run any time -- pure
read/filter/write, never mutates the source fig2_*.json files.

Kept methods (per decision): bm25, repsim, gradsim, less, datainf, semantic,
influcoder. Dropped: graddot, ekfac.
"""
import json
from pathlib import Path

KEEP = ["bm25", "repsim", "gradsim", "less", "datainf", "semantic", "influcoder"]

SOURCES = {
    "results/fig2_counterfact.json": "exp2-counter-final.json",
    "results/fig2_toxicity.json": "exp2-het-final.json",
    "results/fig2_toxicity_hom.json": "exp2-hom-final.json",
}

for src, dst in SOURCES.items():
    d = json.loads(Path(src).read_text())
    final = {m: d[m] for m in KEEP if m in d}
    missing = [m for m in KEEP if m not in d]
    Path(dst).write_text(json.dumps(final, indent=2))
    print(f"{dst}: {len(final)}/{len(KEEP)} methods" + (f"  (missing: {missing})" if missing else ""))
