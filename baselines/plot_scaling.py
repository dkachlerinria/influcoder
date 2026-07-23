#!/usr/bin/env python3
"""InfluCoder's Spearman vs. distillation-set size, against competitor levels.

x is the size of the distillation set (anchors; pool = 2x, the fixed 1:2 ratio);
y is aggregate Spearman against the same fixed 100x100 gradient-influence GT
every other number in this repo is scored on.

Design choices:
  * Line + markers, because size is a continuous ordered quantity and the
    question is a trend, not a set of independent comparisons.
  * A min-max band across seeds is drawn rather than hidden. Run-to-run drift at
    identical settings is the same order as some of the gaps being read off this
    chart, so suppressing it would overstate what the curve resolves.
  * Competitors are horizontal because distillation-set size is undefined for
    them -- they are a level to cross, not a curve to race.
  * Colour encodes METHOD FAMILY (3 slots, all-pairs validated), not rank.
    Within the LoGRA family the ranks share one hue and are separated by direct
    labels, so no reader has to distinguish five shades of the same colour.
  * Log x: sizes span 20x and the interesting behaviour is at the low end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette (all-pairs safe).
C_INFLU = "#2a78d6"   # blue
C_LOGRA = "#eb6834"   # orange
C_LESS = "#1baf7a"    # aqua
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
SURFACE = "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling", default="baselines/out/big/scaling.json")
    ap.add_argument("--ranks", default="baselines/out/big/rank_sweep.json")
    ap.add_argument("--out", default="baselines/out/big/scaling_vs_competitors")
    ap.add_argument("--metric", default="agg", choices=["agg", "per_anchor"])
    args = ap.parse_args()

    sc = json.loads(Path(args.scaling).read_text())
    pts = sc["points"]
    xs = [p["n_train_anchors"] for p in pts]
    if args.metric == "agg":
        ys = [p["agg_mean"] for p in pts]
        lo = [p["agg_min"] for p in pts]
        hi = [p["agg_max"] for p in pts]
        ylabel = "aggregate Spearman $\\rho$ vs. gradient-influence GT"
        rkey = "aggregated"
    else:
        ys = [p["per_anchor_mean"] for p in pts]
        lo = [p["per_anchor_min"] for p in pts]
        hi = [p["per_anchor_max"] for p in pts]
        ylabel = "per-anchor mean Spearman $\\rho$"
        rkey = "per_anchor_mean"

    ranks = json.loads(Path(args.ranks).read_text())["results"]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    all_y = list(ys) + list(lo) + list(hi)

    # Horizontal levels bunch up near the top (LoGRA r=16/r=32 and LESS sit
    # within 0.05 of each other). Stagger their labels along x so close levels
    # never overprint; `place` returns the x fraction to use for a given value.
    placed: list[tuple[float, float]] = []

    # The curve itself is an obstacle too, not just the other labels: a level
    # that passes close to the curve must be labelled where the curve is far
    # away in y, or the text lands on the data.
    lx0, lx1 = min(xs), max(xs)

    def curve_at(xfrac):
        """Curve y at an axes x-fraction (log x, so interpolate in log space)."""
        import math
        t = math.log(lx0) + xfrac * (math.log(lx1) - math.log(lx0))
        xv = math.exp(t)
        for i in range(len(xs) - 1):
            if xs[i] <= xv <= xs[i + 1]:
                f = ((math.log(xv) - math.log(xs[i]))
                     / (math.log(xs[i + 1]) - math.log(xs[i])))
                return ys[i] + f * (ys[i + 1] - ys[i])
        return ys[0] if xv < lx0 else ys[-1]

    def place(value, min_gap=0.055, steps=(0.012, 0.30, 0.58)):
        for x in steps:
            clear_of_labels = all(abs(value - v) > min_gap or abs(x - px) > 1e-9
                                  for v, px in placed)
            clear_of_curve = abs(value - curve_at(x)) > min_gap
            if clear_of_labels and clear_of_curve:
                placed.append((value, x))
                return x
        # Nothing clear: take the step whose curve clearance is largest.
        x = max(steps, key=lambda s: abs(value - curve_at(s)))
        placed.append((value, x))
        return x

    # -- competitor levels ---------------------------------------------------
    less_vals = [v[rkey] for v in ranks.get("less", {}).values()]
    if less_vals:
        # LESS is essentially flat in rank; draw the band and one labelled line
        # rather than four indistinguishable lines stacked on each other.
        if max(less_vals) - min(less_vals) > 0.004:
            ax.axhspan(min(less_vals), max(less_vals), color=C_LESS, alpha=0.13, zorder=1)
        lvl = sum(less_vals) / len(less_vals)
        ax.axhline(lvl, color=C_LESS, linewidth=2.0, linestyle="--", zorder=2)
        ax.annotate(f"LESS (rank 8–128, flat)  {lvl:+.3f}", (1.0, lvl),
                    xycoords=("axes fraction", "data"), textcoords="offset points",
                    xytext=(-6, 5), ha="right", fontsize=9, color=INK, zorder=5)
        placed.append((lvl, 1.0))  # reserve the right edge so levels avoid it
        all_y += less_vals

    logra = ranks.get("logra_raw", {})
    for r in sorted(logra, key=int):
        v = logra[r][rkey]
        ax.axhline(v, color=C_LOGRA, linewidth=1.6, linestyle=(0, (5, 3)),
                   alpha=0.85, zorder=2)
        ax.annotate(f"LoGRA r={r}  {v:+.3f}", (place(v), v),
                    xycoords=("axes fraction", "data"), textcoords="offset points",
                    xytext=(6, 4), ha="left", fontsize=8.5, color=INK, zorder=5)
        all_y.append(v)

    # -- influcoder scaling curve --------------------------------------------
    ax.fill_between(xs, lo, hi, color=C_INFLU, alpha=0.18, zorder=3,
                    linewidth=0)
    ax.plot(xs, ys, color=C_INFLU, linewidth=2.2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
            label="InfluCoder")
    ax.annotate("InfluCoder\n(mean of 3 seeds, band = min–max)",
                (xs[-1], ys[-1]), textcoords="offset points", xytext=(-8, -34),
                ha="right", fontsize=9.5, color=INK, fontweight="bold", zorder=5)

    # untrained-encoder floor: the same encoder before distillation
    untrained = pts[0].get("untrained_agg_mean")
    if untrained is not None and args.metric == "agg":
        ax.axhline(untrained, color=INK_MUTED, linewidth=1.2, linestyle=":",
                   alpha=0.8, zorder=2)
        ax.annotate(f"untrained encoder  {untrained:+.3f}", (0.0, untrained),
                    xycoords=("axes fraction", "data"), textcoords="offset points",
                    xytext=(6, 4), ha="left", fontsize=8.5, color=INK_MUTED, zorder=5)
        all_y.append(untrained)

    ax.axhline(0, color=INK_MUTED, linewidth=1.0, alpha=0.45, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    span = max(all_y) - min(all_y) or 1.0
    ax.set_ylim(min(min(all_y), 0) - 0.06 * span, max(all_y) + 0.10 * span)
    ax.set_xlim(min(xs) * 0.78, max(xs) * 1.30)

    ax.set_xlabel("distillation-set size: training anchors (pool = 2x)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=10.5, color=INK)
    # Title derives from the results file, never hardcoded -- this script renders
    # several settings and a stale eval size or model in the title is a lie.
    cfg = sc.get("config", {})
    shape = cfg.get("gt_shape", [])
    model = str(cfg.get("grad_model", "")).split("/")[-1]
    setting = f"{shape[0]}x{shape[1]}" if len(shape) == 2 else ""
    ax.set_title("InfluCoder scaling vs. competitor levels\n"
                 f"fixed {setting} BBH x Dolly eval, GT from {model}",
                 fontsize=12, color=INK, pad=12)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    handles = [
        plt.Line2D([], [], color=C_INFLU, linewidth=2.2, marker="o", markersize=7,
                   label="InfluCoder (distilled encoder)"),
        plt.Line2D([], [], color=C_LOGRA, linewidth=1.6, linestyle=(0, (5, 3)),
                   label="LoGRA (raw), by rank"),
        plt.Line2D([], [], color=C_LESS, linewidth=2.0, linestyle="--",
                   label="LESS (mean over ranks)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
              labelcolor=INK)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{args.out}.{ext}"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
