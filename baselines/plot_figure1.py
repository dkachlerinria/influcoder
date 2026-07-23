#!/usr/bin/env python3
"""Figure 1, table 1 rendered: inference cost vs. agreement with gradient
influence, every method on the SAME 400x400 eval (GT from Qwen3-4B, rank 16).

Same design language as plot_cost_quality.py (reused, not reinvented): log-x
scatter, color by cost class (validated categorical slots 1-3), Pareto
frontier, direct labels. Two new encodings for this table's extra structure:
  * marker SIZE steps with encoder scale (68m < 150m < 400m) for both
    InfluCoder and the untrained-encoder baseline it is measured against.
  * marker FILL: solid = trained (InfluCoder), hollow = untrained encoder --
    same color, so a reader can see exactly what distillation buys at each
    encoder size (compare same-size solid vs. hollow pairs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same validated categorical slots 1-3 as plot_cost_quality.py.
C_GRAD = "#2a78d6"   # blue   -- forward + backward per candidate
C_FWD = "#eb6834"    # orange -- single forward per candidate
C_FREE = "#1baf7a"   # aqua   -- no model at all
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

# name -> (label, cost-class color, marker size, filled?)
ROWS = {
    "less":              ("LESS",                    C_GRAD, 110, True),
    "logra_r8":          ("LoGRA (r=8, 4B)",          C_GRAD, 110, True),
    "logra_proxy_1.7B":  ("LoGRA proxy (1.7B, r=32)", C_GRAD, 110, True),
    "logra_proxy_0.6B":  ("LoGRA proxy (0.6B, r=32)", C_GRAD, 110, True),
    "influcoder_68m":    ("InfluCoder 68m",           C_FWD,   70, True),
    "influcoder_150m":   ("InfluCoder 150m",          C_FWD,  110, True),
    "influcoder_400m":   ("InfluCoder 400m",          C_FWD,  160, True),
    "untrained_68m":     ("untrained enc. 68m",       C_FWD,   70, False),
    "untrained_150m":    ("untrained enc. 150m",      C_FWD,  110, False),
    "untrained_400m":    ("untrained enc. 400m",      C_FWD,  160, False),
    "rdsplus":           ("RDS+",                     C_FWD,  110, True),
}
# TF-IDF is a horizontal reference line, not a scatter point: its cost is
# ~0 (no model at all), so its x-position under a log-cost axis doesn't mean
# anything -- it's more honestly read as "the quality floor a free method
# gets you", spanning the whole plot.
TFIDF_LABEL = "TF-IDF (no model, ~0 cost)"

# (dx, dy, ha) label offsets, tuned to avoid collisions in this specific table.
# The 68m/150m pairs sit almost on top of each other on both x axes (same
# forward-pass cost within a size, and adjacent sizes only ~2x apart), so
# each pair is split into an "up" label (trained, bold) and a "down" label
# (untrained) rather than left/right, which collided with the neighbouring
# size's cluster. rdsplus and logra_proxy_1.7B sit almost exactly on top of
# EACH OTHER (nearly identical cost and quality) so they get opposite
# vertical pushes too.
OFFSETS = {
    "less": (10, 2, "left"),
    "logra_r8": (10, 2, "left"),
    "logra_proxy_1.7B": (10, -16, "left"),
    "logra_proxy_0.6B": (10, -5, "left"),
    "influcoder_68m": (10, 20, "left"),
    "influcoder_150m": (10, 6, "left"),
    "influcoder_400m": (12, 8, "left"),
    "untrained_68m": (10, -10, "left"),
    "untrained_150m": (10, -24, "left"),
    "untrained_400m": (12, -24, "left"),
    "rdsplus": (10, 20, "left"),
}
# dolci-instruct's encoder-cluster scores land much more tightly bunched
# (agg rho all within ~0.12-0.18, vs. a much wider spread for Dolly) and the
# trained/untrained ordering even flips at 68m/150m -- OFFSETS above puts
# labels on top of each other here, so this preset gets its own fan-out.
OFFSETS_DOLCI = {
    **OFFSETS,
    "untrained_68m": (-4, 4, "right"),
    "influcoder_68m": (-6, -20, "right"),
    "untrained_150m": (0, 34, "center"),
    "influcoder_150m": (0, -34, "center"),
    "influcoder_400m": (6, 22, "left"),
    "untrained_400m": (6, -16, "left"),
}


def pareto_front(points):
    front, best = [], float("-inf")
    for x, y in sorted(points, key=lambda p: p[0]):
        if y > best:
            front.append((x, y))
            best = y
    return front


def panel(ax, rows, xkey, xlabel, title, tfidf_agg=None, xlim_min=None, offsets=OFFSETS):
    xs = [r[xkey] for r in rows if r[xkey] > 0]
    floor = min(xs) / 8 if xs else 1.0

    for r in rows:
        clamped = r[xkey] <= 0
        x = floor if clamped else r[xkey]
        label, color, size, filled = ROWS[r["name"]]
        face = color if filled else "none"
        ax.scatter(x, r["aggregated"], s=size, color=face, edgecolors=color,
                  linewidths=2.0, zorder=3)
        dx, dy, ha = offsets.get(r["name"], (9, 5, "left"))
        weight = "bold" if r["name"].startswith("influcoder") else "normal"
        ax.annotate(label, (x, r["aggregated"]), textcoords="offset points",
                   xytext=(dx, dy), ha=ha, fontsize=8.5, color=INK,
                   fontweight=weight, zorder=4)

    front = pareto_front([(r[xkey] if r[xkey] > 0 else floor, r["aggregated"])
                          for r in rows])
    if len(front) > 1:
        ax.plot([p[0] for p in front], [p[1] for p in front], color=INK_MUTED,
                linewidth=1.2, linestyle="--", alpha=0.55, zorder=2)

    ax.axhline(0, color=INK_MUTED, linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xscale("log")

    ys = [r["aggregated"] for r in rows]
    if tfidf_agg is not None:
        ys = ys + [tfidf_agg]
    span = max(ys) - min(ys) or 1.0
    ax.set_ylim(min(ys) - 0.16 * span, max(ys) + 0.14 * span)
    hi = max(max(xs), floor)
    ax.set_xlim(xlim_min if xlim_min is not None else floor / 2.5, hi * 4.0)

    if tfidf_agg is not None:
        ax.axhline(tfidf_agg, color=C_FREE, linewidth=1.6, linestyle=(0, (5, 3)),
                  alpha=0.85, zorder=2)
        ax.annotate(TFIDF_LABEL, (ax.get_xlim()[1], tfidf_agg),
                   textcoords="offset points", xytext=(-6, -18), ha="right",
                   fontsize=8.5, color=C_FREE, fontweight="bold", zorder=4)

    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


POOL_LABELS = {"dolly": "Dolly", "dolci_instruct": "dolci-instruct"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fig1")
    args = ap.parse_args()
    results_path = Path("baselines/out") / args.preset / "table1.json"
    out = str(Path("baselines/out") / args.preset / "figure1")
    data = json.loads(results_path.read_text())
    rows = []
    tfidf_agg = None
    for name, m in data["methods"].items():
        if name == "tfidf":
            tfidf_agg = m["aggregated"]
            continue
        if name not in ROWS:
            continue
        rows.append({"name": name, "aggregated": m["aggregated"],
                     "time_per_sample_ms": m.get("time_per_sample_ms", 0.0)})

    # FLOPs panel dropped for now -- deferred to an appendix experiment (see
    # figure1_table.run_row's measure_flops flag). It told the same story as
    # ms/sample anyway, at ~5x the wall-clock cost to measure.
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    offsets = OFFSETS_DOLCI if data.get("config", {}).get("pool") == "dolci_instruct" else OFFSETS
    panel(ax, rows, "time_per_sample_ms", "inference time per sample, ms (log)",
         "Selection-time cost vs. ranking quality", tfidf_agg=tfidf_agg, xlim_min=1,
         offsets=offsets)
    ax.set_ylabel("aggregate Spearman $\\rho$ vs. gradient-influence GT",
                  fontsize=10, color=INK)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                  markerfacecolor=C_GRAD, markeredgecolor=C_GRAD, markeredgewidth=2.0,
                  label="gradient (fwd+bwd per candidate)"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                  markerfacecolor=C_FWD, markeredgecolor=C_FWD, markeredgewidth=2.0,
                  label="single forward per candidate"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                  markerfacecolor=INK_MUTED, markeredgecolor=INK_MUTED,
                  label="filled = trained, hollow = untrained"),
        plt.Line2D([], [], color=INK_MUTED, linestyle="--", linewidth=1.2,
                  label="Pareto frontier"),
        plt.Line2D([], [], color=C_FREE, linestyle=(0, (5, 3)), linewidth=1.6,
                  label="TF-IDF (no model, ~0 cost)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
              fontsize=8.5, bbox_to_anchor=(0.5, -0.06), labelcolor=INK)

    cfg = data.get("config", {})
    shape = cfg.get("gt_shape", [])
    model = str(cfg.get("grad_model", "")).split("/")[-1]
    setting = f"{shape[0]}x{shape[1]} eval" if len(shape) == 2 else "eval"
    pool_label = POOL_LABELS.get(cfg.get("pool", "dolly"), cfg.get("pool", "Dolly"))
    fig.suptitle("Figure 1: selection-time cost vs. agreement with gradient "
                f"influence (BBH x {pool_label}, {setting}, GT from {model})",
                fontsize=12, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    for ext in ("png", "pdf"):
        path = f"{out}.{ext}"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
