#!/usr/bin/env python3
"""EXP1 Part 2 (BIG_GPU_FINAL) plot: InfluCoder training-sample scaling.

Reads `baselines/out/<preset>/<profile>/exp1_part2.json` (written by
`part2.py`) and produces `exp1_part2_figure.{png,pdf}` in the same directory.
Same visual language as `plot_part1.py`: InfluCoder in C_FWD (single-forward
cost class), LESS/LoGRA gradient reference lines in C_GRAD -- drawn as
horizontal lines (like Part 1's TF-IDF line) since Part 2 doesn't re-sweep
their training-set size, just the InfluCoder curve against them at THIS
sweep's own eval slice (see part2.py's docstring on why these are recomputed
fresh here rather than borrowed from Part 1). Solid=4B, dashed=1.7B; LESS vs
LoGRA distinguished by label text and line weight, not a second hue -- both
are "gradient fwd+bwd" cost-class, matching Part 1's convention.

    python -m baselines.exp1.plot_part2
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if os.environ.get("EXP1_CONFIG") == "biggpu":
    from . import config_biggpu as cfg  # BIG_GPU_FINAL_EXP1 -- see that module's docstring
else:
    from . import config as cfg

C_GRAD = "#2a78d6"
C_FWD = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

# Reads from the consolidated EXP1_results.json (written by
# collect_results.py from Part 1/2/3's raw outputs), not exp1_part2.json
# directly -- one file every plot script reads its numbers from, so there is
# exactly one place to look to see what actually drove Figure 1.
IN_RESULTS = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED) / "EXP1_results.json"
OUT_PATH = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED) / "exp1_part2_figure"

# label, linestyle, linewidth, label_dy (points -- nudged to avoid collisions
# when two reference lines land close together in data units)
REF_SPEC = {
    "less_4B_r32": ("LESS (4B) r32", "solid", 1.8, 6),
    "less_1.7B_r8": ("LESS (1.7B) r8", "solid", 1.1, 6),
    "logra_4B": ("LoGRA (4B)", (0, (5, 2)), 1.8, -13),
    "logra_1.7B": ("LoGRA (1.7B)", (0, (5, 2)), 1.1, -13),
}


def draw(ax, data):
    """Draw the full Part 2 InfluCoder-scaling panel onto `ax`. Factored out
    of `main()` so `plot_figure1.py` can reuse this exact drawing code for
    its combined panel instead of a second, driftable copy of it."""
    points = sorted(data["points"], key=lambda p: p["n_train_total"])
    refs = data["reference_lines"]
    cfg_d = data["config"]

    xs = [p["n_train_total"] for p in points]
    ys = [p["agg"] for p in points]

    # Untrained (0 samples) drawn as a real point ON the scaling curve, not a
    # separate flat reference line -- log-x can't place it at literal x=0, so
    # it sits at a small placeholder x left of the first real point (matching
    # Part 1's hollow-diamond = untrained convention) and the curve connects
    # through it, showing the actual 0-sample -> 4500-sample trajectory.
    untrained_y = points[0]["untrained_agg"]
    untrained_x = xs[0] / 3
    full_xs, full_ys = [untrained_x] + xs, [untrained_y] + ys

    ax.plot(full_xs, full_ys, color=C_FWD, linewidth=1.6, alpha=0.85, zorder=2)
    ax.scatter(xs, ys, s=90, marker="D", facecolors=C_FWD,
              edgecolors=C_FWD, linewidths=1.5, zorder=3, label="InfluCoder 68m")
    ax.scatter(untrained_x, untrained_y, s=90, marker="D", facecolors="none",
              edgecolors=C_FWD, linewidths=1.5, zorder=3)
    for p in points:
        ax.annotate(f"{p['n_train_total']}", (p["n_train_total"], p["agg"]),
                   textcoords="offset points", xytext=(0, 10), ha="center",
                   fontsize=8, color=INK, zorder=4)
    ax.annotate("untrained\n(0 samples)", (untrained_x, untrained_y),
               textcoords="offset points", xytext=(0, -26), ha="center",
               fontsize=8, color=INK_MUTED, zorder=4)

    for key, (label, ls, lw, dy) in REF_SPEC.items():
        y = refs[key]
        ax.axhline(y, color=C_GRAD, linewidth=lw, linestyle=ls, alpha=0.75, zorder=1)
        ax.annotate(label, (0.995, y), xycoords=("axes fraction", "data"),
                   textcoords="offset points", xytext=(0, dy), ha="right",
                   fontsize=8.5, color=C_GRAD, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("InfluCoder training set size (anchors + pool, log scale)",
                 fontsize=10.5, color=INK)
    ax.set_ylabel(f"aggregate Spearman ρ vs. {cfg_d['gt_model']}/"
                 f"r{cfg_d['gt_lora_rank']} ground truth", fontsize=10.5, color=INK)
    ax.set_title(f"EXP1 Part 2 (BIG_GPU_FINAL) — InfluCoder scaling, "
                f"{cfg_d['n_eval']}x{cfg_d['n_eval']} eval\n"
                f"epochs={cfg_d['epochs']}, hard_ratio={cfg_d['hard_ratio']}, "
                f"lr={cfg_d['lr']}", fontsize=11, color=INK, pad=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_ylim(-0.05, 1.05)

    legend_elems = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C_FWD,
              markeredgecolor=C_FWD, markersize=9, label="InfluCoder 68m (single fwd)"),
        Line2D([0], [0], color=C_GRAD, linewidth=1.8, linestyle="solid",
              label="LESS-4B r32 [benchmaxxer] / LoGRA-4B r32 (this eval slice)"),
        Line2D([0], [0], color=C_GRAD, linewidth=1.1, linestyle=(0, (5, 2)),
              label="LESS-1.7B r8 / LoGRA-1.7B r32 (this eval slice)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="none",
              markeredgecolor=C_FWD, markersize=9,
              label="untrained InfluCoder (0 training samples)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, frameon=False)


def main():
    data = json.loads(IN_RESULTS.read_text())["part2"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    draw(ax, data)
    fig.tight_layout()
    fig.savefig(f"{OUT_PATH}.png", dpi=160)
    fig.savefig(f"{OUT_PATH}.pdf")
    print(f"wrote {OUT_PATH}.png / .pdf")


if __name__ == "__main__":
    main()
