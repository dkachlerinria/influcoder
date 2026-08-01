#!/usr/bin/env python3
"""EXP1 Part 1 (BIG_GPU_FINAL) plot: cost vs quality.

Reads `baselines/out/<preset>/exp1_part1.json` (written by `part1.py`) and
produces `baselines/out/<preset>/exp1_part1_figure.{png,pdf}`. Same visual
language as this session's earlier draft figures (see EXP1.md section 5.5/5.6
and `.tuning_logs/plot_vercors18_part1.py`, which this script's design is
carried forward from): categorical color by cost-class (gradient fwd+bwd =
blue, single forward = orange, no model = aqua), marker shape by family
(LESS circle / LoGRA square / InfluCoder+untrained diamond / RDS+ triangle),
marker size by model size, filled=trained vs hollow=untrained, solid line
connecting each proxy family in size order. TF-IDF has no model forward
pass, so its "cost" isn't comparable to the others on this axis -- drawn as
a horizontal reference line (its quality, at any cost) rather than a point.
Sqrt-x, starting at 1ms/sample (per explicit request): this run's cost range
(0.15ms TF-IDF -> 479ms LESS-4B) spans over 3 orders of magnitude, but TF-IDF
is drawn as a horizontal line rather than a point (its cost isn't comparable
to the model-based methods' anyway), so the axis only needs to actually
accommodate the model-based methods' range (~1.8ms InfluCoder -> 479ms
LESS-4B) -- starting the visible axis at 1ms avoids wasting space on the
sub-1ms region nothing but TF-IDF's now-irrelevant literal x-position would
have occupied.

    python -m baselines.exp1.plot_part1
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if os.environ.get("EXP1_CONFIG") == "biggpu":
    from . import config_biggpu as cfg  # BIG_GPU_FINAL_EXP1 -- see that module's docstring
else:
    from . import config as cfg

C_GRAD = "#2a78d6"      # LoGRA (uniform r32) + LESS r32
C_GRAD_R8 = "#8fb8e8"   # LESS r8 -- lighter shade of the same hue (still
                        # reads as "LESS/gradient", lightness carries the
                        # rank sub-category) so the r8-vs-r32 comparison is a
                        # single-glance color contrast, not just label text.
C_FWD = "#eb6834"
C_FREE = "#1baf7a"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

# Reads from the consolidated EXP1_results.json (written by
# collect_results.py from Part 1/2/3's raw outputs), not exp1_part1.json
# directly -- one file every plot script reads its numbers from, so there is
# exactly one place to look to see what actually drove Figure 1.
IN_RESULTS = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED) / "EXP1_results.json"
OUT_PATH = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED) / "exp1_part1_figure"

# (label, marker, size_pt, color, filled, family, size_rank)
# LESS is now TWO full families -- "less_r8" (practical) and "less_r32"
# (ceiling) -- each its own connecting line in size order, deliberately kept
# SEPARATE (not merged into one "less" family) specifically so the
# rank-starvation effect at 0.6B (agg rho +0.128 at r8 vs +0.253 at r32) shows
# up as a visible gap between the two lines rather than a single zigzagging
# line that hides which point came from which rank.
SPEC = {
    "less_4B_r8": ("LESS (4B) r8", "o", 150, C_GRAD_R8, True, "less_r8", 3),
    "less_1.7B_r8": ("LESS (1.7B) r8", "o", 105, C_GRAD_R8, True, "less_r8", 2),
    "less_0.6B_r8": ("LESS (0.6B) r8", "o", 65, C_GRAD_R8, True, "less_r8", 1),
    "less_4B_r32": ("LESS (4B) r32", "o", 150, C_GRAD, True, "less_r32", 3),
    "less_1.7B_r32": ("LESS (1.7B) r32", "o", 105, C_GRAD, True, "less_r32", 2),
    "less_0.6B_r32": ("LESS (0.6B) r32", "o", 65, C_GRAD, True, "less_r32", 1),
    "logra_4B": ("LoGRA (4B)", "s", 150, C_GRAD, True, "logra", 3),
    "logra_1.7B": ("LoGRA (1.7B)", "s", 105, C_GRAD, True, "logra", 2),
    "logra_0.6B": ("LoGRA (0.6B)", "s", 65, C_GRAD, True, "logra", 1),
    "influcoder_68m": ("InfluCoder 68m", "D", 90, C_FWD, True, None, 0),
    "influcoder_150m": ("InfluCoder 150m", "D", 120, C_FWD, True, None, 0),
    "untrained_68m": ("untrained 68m", "D", 90, C_FWD, False, None, 0),
    "untrained_150m": ("untrained 150m", "D", 120, C_FWD, False, None, 0),
    "rdsplus": ("RDS+ (4B)", "^", 120, C_FWD, True, None, 0),
}

# Label nudges: (dx, dy, ha) in points, tuned to avoid collisions.
OFFSETS = {
    "less_4B_r8": (10, -14, "left"),
    "less_1.7B_r8": (10, -14, "left"),
    "less_0.6B_r8": (10, -14, "left"),
    "less_4B_r32": (10, 6, "left"),
    "less_1.7B_r32": (10, 6, "left"),
    "less_0.6B_r32": (10, 6, "left"),
    "logra_4B": (10, -14, "left"),
    "logra_1.7B": (10, 6, "left"),
    "logra_0.6B": (10, 6, "left"),
    "influcoder_68m": (-10, -14, "right"),
    "influcoder_150m": (10, 6, "left"),
    "untrained_68m": (10, 6, "left"),
    "untrained_150m": (-10, -14, "right"),
    "rdsplus": (10, 6, "left"),
}


def draw(ax, data):
    """Draw the full Part 1 cost-vs-quality panel onto `ax`. Factored out of
    `main()` so `plot_figure1.py` can reuse this exact drawing code for its
    combined panel instead of a second, driftable copy of it."""
    methods = data["methods"]
    n_eval = data["config"]["n_eval"]
    less_ranks = data["config"]["less_ranks"]
    logra_rank = data["config"]["logra_rank"]

    for family, color, ls in [("less_r8", C_GRAD_R8, (0, (4, 2))),
                              ("less_r32", C_GRAD, "solid"),
                              ("logra", C_GRAD, "solid")]:
        pts = sorted(
            ((methods[k]["time_per_sample_ms"], methods[k]["aggregated"])
             for k, v in SPEC.items() if v[5] == family),
            key=lambda p: p[0],
        )
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
               linewidth=1.3, alpha=0.45, zorder=2, linestyle=ls)

    for key, (label, marker, size, color, filled, family, _) in SPEC.items():
        m = methods[key]
        x, y = m["time_per_sample_ms"], m["aggregated"]
        ax.scatter(x, y, s=size, marker=marker,
                  facecolors=color if filled else "none",
                  edgecolors=color, linewidths=1.8, zorder=3)
        dx, dy, ha = OFFSETS[key]
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                   ha=ha, fontsize=8.5, color=INK, zorder=4)

    ax.axhline(0, color=INK_MUTED, linewidth=1.0, alpha=0.5, zorder=1)

    tfidf_y = methods["tfidf"]["aggregated"]
    ax.axhline(tfidf_y, color=C_FREE, linewidth=1.6, linestyle=(0, (5, 3)),
              alpha=0.8, zorder=2)
    ax.annotate("TF-IDF (no model)", (0.995, tfidf_y),
               xycoords=("axes fraction", "data"), textcoords="offset points",
               xytext=(0, 4), ha="right", fontsize=8.5, color=C_FREE, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("inference cost (ms/sample, log scale)", fontsize=10.5, color=INK)
    ax.set_ylabel(f"aggregate Spearman ρ vs. {data['config']['gt_model']}/"
                 f"r{data['config']['gt_lora_rank']} ground truth", fontsize=10.5, color=INK)
    less_ranks_str = "/".join(str(r) for r in less_ranks)
    ax.set_title(f"EXP1 Part 1 (BIG_GPU_FINAL) — {n_eval}x{n_eval} eval\n"
                f"LESS r={less_ranks_str} (every size, both ranks), LoGRA r={logra_rank} (uniform, batched)",
                fontsize=11, color=INK, pad=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlim(1, 300)
    ax.set_xticks([1, 5, 20, 50, 100, 200, 300])

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_GRAD_R8,
              markeredgecolor=C_GRAD_R8, markersize=9, label="LESS r8 (gradient fwd+bwd)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_GRAD,
              markeredgecolor=C_GRAD, markersize=9, label="LESS r32 (gradient fwd+bwd)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_GRAD,
              markeredgecolor=C_GRAD, markersize=9, label="LoGRA (gradient fwd+bwd)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C_FWD,
              markeredgecolor=C_FWD, markersize=9, label="InfluCoder (single fwd, filled=trained)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=C_FWD,
              markeredgecolor=C_FWD, markersize=9, label="RDS+ (single fwd)"),
        Line2D([0], [0], color=C_FREE, linewidth=1.6, linestyle=(0, (5, 3)),
              label="TF-IDF (no model, cost not comparable)"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=8, frameon=False)


def main():
    data = json.loads(IN_RESULTS.read_text())["part1"]
    fig, ax = plt.subplots(figsize=(8, 8))
    draw(ax, data)
    fig.tight_layout()
    fig.savefig(f"{OUT_PATH}.png", dpi=160)
    fig.savefig(f"{OUT_PATH}.pdf")
    print(f"wrote {OUT_PATH}.png / .pdf")


if __name__ == "__main__":
    main()
