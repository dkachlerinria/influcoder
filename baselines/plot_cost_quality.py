#!/usr/bin/env python3
"""Cost-vs-quality figures: FLOPs/sample and ms/sample against aggregate Spearman.

Reads baseline_results.json (written by baselines.eval_baselines) and renders a
two-panel scatter. Each point is a method; x is what it costs to featurize one
candidate at selection time, y is how well its ranking agrees with the
gradient-influence ground truth.

Design choices:
  * Scatter, because the question is a trade-off between two continuous
    quantities with method identity attached -- not a magnitude comparison.
  * Log x: per-sample cost spans several orders of magnitude, so a linear axis
    would collapse every cheap method onto the origin.
  * Colour encodes *cost class* (gradient fwd+bwd / single forward / no model),
    not one hue per method. Three categorical slots is also the documented
    all-pairs cap for CVD-safe scatter; eight would not validate. Method
    identity is carried by direct labels on every point, never by colour alone.
  * A Pareto frontier is drawn so "best quality per unit cost" is readable
    directly rather than inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette (all-pairs safe).
C_GRAD = "#2a78d6"   # blue   -- forward + backward per candidate
C_FWD = "#eb6834"    # orange -- single forward per candidate
C_FREE = "#1baf7a"   # aqua   -- no model at all
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

COST_CLASS = {
    "less": ("gradient (fwd+bwd)", C_GRAD),
    "logra_raw": ("gradient (fwd+bwd)", C_GRAD),
    "logra_fim": ("gradient (fwd+bwd)", C_GRAD),
    "influcoder": ("single forward", C_FWD),
    "semantic": ("single forward", C_FWD),
    "rdsplus": ("single forward", C_FWD),
    "tfidf": ("no model", C_FREE),
    "random": ("no model", C_FREE),
}

LABEL = {
    "less": "LESS", "logra_raw": "LoGRA (raw)", "logra_fim": "LoGRA (FIM)",
    "influcoder": "InfluCoder", "semantic": "semantic (untrained enc.)",
    "rdsplus": "RDS+", "tfidf": "TF-IDF", "random": "random",
}

# Nudge labels where the default placement would collide: (dx, dy, ha).
# Points near the right edge are labelled leftwards so the text stays inside the
# axes; points near the floor are labelled upwards for the same reason.
OFFSETS = {
    # raw and fim share an x (same encode pass): label the upper one upward and
    # the lower one downward, or the two texts converge on each other.
    "logra_raw": (9, -14, "left"),
    "logra_fim": (9, 5, "left"),
    "less": (-9, 7, "right"),
    "influcoder": (11, -4, "left"),
    "semantic": (10, -5, "left"),
    "rdsplus": (-9, -14, "right"),
    "tfidf": (9, 5, "left"),
    "random": (9, 5, "left"),
}


def pareto_front(points):
    """(x, y) pairs that nothing beats on both axes (cheaper AND better)."""
    front, best = [], float("-inf")
    for x, y in sorted(points, key=lambda p: p[0]):
        if y > best:
            front.append((x, y))
            best = y
    return front


def panel(ax, rows, xkey, xlabel, title, floor_note):
    xs = [r[xkey] for r in rows if r[xkey] > 0]
    floor = min(xs) / 8 if xs else 1.0

    # Methods with no model do zero counted FLOPs; clamp them to a floor so they
    # remain visible on a log axis, and mark them hollow so the reader can see
    # the value is a floor, not a measurement.
    for r in rows:
        clamped = r[xkey] <= 0
        x = floor if clamped else r[xkey]
        _, color = COST_CLASS[r["name"]]
        ax.scatter(x, r["aggregated"], s=110, color="none" if clamped else color,
                   edgecolors=color, linewidths=2.0, zorder=3)
        dx, dy, ha = OFFSETS.get(r["name"], (9, 5, "left"))
        weight = "bold" if r["name"] == "influcoder" else "normal"
        ax.annotate(LABEL[r["name"]], (x, r["aggregated"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=9, color=INK, fontweight=weight, zorder=4)

    front = pareto_front([(r[xkey] if r[xkey] > 0 else floor, r["aggregated"])
                          for r in rows])
    if len(front) > 1:
        ax.plot([p[0] for p in front], [p[1] for p in front], color=INK_MUTED,
                linewidth=1.2, linestyle="--", alpha=0.55, zorder=2,
                label="Pareto frontier")

    ax.axhline(0, color=INK_MUTED, linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xscale("log")

    # Headroom so offset labels stay inside the axes rather than overflowing
    # into the caption. Labels sit ~14pt from their point, so pad in data units.
    ys = [r["aggregated"] for r in rows]
    span = max(ys) - min(ys) or 1.0
    ax.set_ylim(min(ys) - 0.14 * span, max(ys) + 0.12 * span)
    hi = max(max(xs), floor)
    ax.set_xlim(floor / 2.5, hi * 4.0)

    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    if floor_note:
        # Top-left: the bottom-left corner is where the cheap/weak methods and
        # their labels live, so a note there would collide with them.
        ax.text(0.02, 0.97, floor_note, transform=ax.transAxes, fontsize=7.5,
                color=INK_MUTED, va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="baselines/out/big/baseline_results.json")
    ap.add_argument("--out", default="baselines/out/big/cost_vs_quality")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    rows = []
    for name, m in data["methods"].items():
        if name not in COST_CLASS:
            continue
        rows.append({"name": name, "aggregated": m["aggregated"],
                     "flops_per_sample": m.get("flops_per_sample", 0.0),
                     "time_per_sample_ms": m.get("time_per_sample_ms", 0.0)})
    if not rows:
        raise SystemExit("no methods found in results file")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    panel(axes[0], rows, "flops_per_sample", "FLOPs per sample (log)",
          "Compute cost vs. ranking quality",
          "hollow = no counted FLOPs, clamped to axis floor")
    panel(axes[1], rows, "time_per_sample_ms", "inference time per sample, ms (log)",
          "Wall-clock cost vs. ranking quality",
          "measured on one RTX 6000 Ada; excludes model load")

    axes[0].set_ylabel("aggregate Spearman $\\rho$ vs. gradient-influence GT",
                       fontsize=10, color=INK)

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                          markerfacecolor=c, markeredgecolor=c,
                          markeredgewidth=2.0, label=l)
               for l, c in [("gradient (fwd+bwd per candidate)", C_GRAD),
                            ("single forward per candidate", C_FWD),
                            ("no model", C_FREE)]]
    handles.append(plt.Line2D([], [], color=INK_MUTED, linestyle="--",
                              linewidth=1.2, label="Pareto frontier"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02), labelcolor=INK)

    # Title is derived from the results file, never hardcoded -- the same script
    # renders several settings and a stale eval size in the title is a lie.
    cfg = data.get("config", {})
    shape = cfg.get("gt_shape", [])
    model = str(cfg.get("grad_model", "")).split("/")[-1]
    setting = f"{shape[0]}x{shape[1]} eval" if len(shape) == 2 else "eval"
    fig.suptitle("Selection-time cost vs. agreement with gradient influence "
                 f"(BBH x Dolly, {setting}, GT from {model})",
                 fontsize=12, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    for ext in ("png", "pdf"):
        path = f"{args.out}.{ext}"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
