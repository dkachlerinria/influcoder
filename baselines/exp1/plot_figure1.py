#!/usr/bin/env python3
"""EXP1 Figure 1 (BIG_GPU_FINAL): the combined 3-panel figure.

(A) Part 1 -- cost vs quality, every method, one eval.
(B) Part 2 -- InfluCoder training-sample scaling, same eval, vs LESS/LoGRA
    reference lines.
(C) Part 3 -- GPU-time vs samples-processed amortization, InfluCoder vs the
    LESS/LoGRA sizes it actually beats in quality (1.7B; see config.py's
    PART3_LESS_MODELS/PART3_LOGRA_MODELS -- deliberately not the 4B variants,
    which beat InfluCoder on quality and so aren't a meaningful "does it
    eventually win" comparison) plus RDS+ (single size, no family the way
    LESS/LoGRA have).

Reuses each part's own `draw(ax, ...)` function (plot_part1.draw /
plot_part2.draw / plot_part3.draw) so there is exactly one implementation of
each panel's drawing logic -- this script and the three standalone
`plot_part{1,2,3}.py` scripts can never visually drift apart from each other.

Reads the single consolidated EXP1_results.json (written by
collect_results.py) ONCE and passes its "part1"/"part2"/"part3" sections
straight to each draw() -- no script (this one or the standalone
plot_part{1,2,3}.py) ever re-derives a number that isn't already sitting in
that file, so the figure is always exactly what's on disk, not a script's
in-memory recomputation of it.

    python -m baselines.exp1.plot_figure1
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if os.environ.get("EXP1_CONFIG") == "biggpu":
    from . import config_biggpu as cfg  # BIG_GPU_FINAL_EXP1 -- see that module's docstring
else:
    from . import config as cfg
from . import plot_part1, plot_part2, plot_part3

IN_RESULTS = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED) / "EXP1_results.json"
OUT_PATH = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED) / "figure1"


def main():
    results = json.loads(IN_RESULTS.read_text())
    p1_data = results["part1"]
    p2_data = results["part2"]
    less_logra = results["part3"]["less_logra"]
    influcoder = results["part3"]["influcoder"]

    fig, axes = plt.subplots(1, 3, figsize=(23, 7.5))

    plot_part1.draw(axes[0], p1_data)
    plot_part2.draw(axes[1], p2_data)
    cross_less, cross_logra, cross_rdsplus = plot_part3.draw(axes[2], less_logra, influcoder)

    for ax, letter in zip(axes, "ABC"):
        ax.text(-0.10, 1.05, letter, transform=ax.transAxes, fontsize=17,
                fontweight="bold", color="#0b0b0b", va="bottom", ha="left")

    fig.suptitle("EXP1 Figure 1 (BIG_GPU_FINAL): InfluCoder vs gradient-influence "
                "baselines -- cost/quality, scaling, and amortization",
                fontsize=13.5, y=1.04)

    fig.tight_layout()
    fig.savefig(f"{OUT_PATH}.png", dpi=160, bbox_inches="tight")
    fig.savefig(f"{OUT_PATH}.pdf", bbox_inches="tight")
    print(f"wrote {OUT_PATH}.png / .pdf")

    if cross_less:
        print(f"InfluCoder overtakes LESS at {cross_less[0]:.0f}s ({cross_less[1]:.0f} samples)")
    if cross_logra:
        print(f"InfluCoder overtakes LoGRA at {cross_logra[0]:.0f}s ({cross_logra[1]:.0f} samples)")
    if cross_rdsplus:
        print(f"InfluCoder overtakes RDS+ at {cross_rdsplus[0]:.0f}s ({cross_rdsplus[1]:.0f} samples)")


if __name__ == "__main__":
    main()
