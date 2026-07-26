#!/usr/bin/env python3
"""LoGRA rank curve at the paper setting, with InfluCoder placed against it.

Setting reproduces tis-ie runs/influence_spearman/config_influence_tiny.sh:
SmolLM2-1.7B native (no proxy), 200x200 BBH x Dolly eval, GT at proj 65536 /
LoRA rank 16. Every point is scored against that same ground truth.

Design choices:
  * Lines over rank, because rank is an ordered continuous knob and the
    question is a trend (does more rank keep buying quality?).
  * InfluCoder is horizontal: its cost and quality do not depend on LoGRA's
    rank, so it is a level to compare against, not a curve.
  * Colour encodes method (3 slots, all-pairs validated). Direct labels on
    every line, so identity never rests on colour alone.
  * Linear y through 0 with the untrained-encoder floor marked, so "how far
    above uninformative" is readable directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_INFLU = "#2a78d6"   # blue   -- slot 1
C_RAW = "#eb6834"     # orange -- slot 2
C_FIM = "#1baf7a"     # aqua   -- slot 3
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
SURFACE = "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logra", default="baselines/out/paper200_logra_native.json")
    ap.add_argument("--influcoder_agg", type=float, default=0.4978)
    ap.add_argument("--influcoder_pa", type=float, default=0.3922)
    ap.add_argument("--untrained_agg", type=float, default=-0.0406)
    ap.add_argument("--metric", default="agg", choices=["agg", "per_anchor"])
    ap.add_argument("--out", default="baselines/out/paper200_rank_curve")
    args = ap.parse_args()

    d = json.loads(Path(args.logra).read_text())
    key = "aggregated" if args.metric == "agg" else "per_anchor_mean"
    influ = args.influcoder_agg if args.metric == "agg" else args.influcoder_pa

    ranks, raw, fim = [], [], []
    for r in [4, 8, 16]:
        kr, kf = f"logra_raw_r{r}", f"logra_fim_r{r}"
        if kr in d and kf in d:
            ranks.append(r)
            raw.append(d[kr][key])
            fim.append(d[kf][key])
    if not ranks:
        raise SystemExit("no logra rows found")

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(ranks, raw, color=C_RAW, linewidth=2.2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.plot(ranks, fim, color=C_FIM, linewidth=2.2, marker="s", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.annotate("LoGRA (raw)", (ranks[-1], raw[-1]), textcoords="offset points",
                xytext=(10, -2), fontsize=10, color=INK, fontweight="bold", zorder=5)
    ax.annotate("LoGRA (FIM)", (ranks[-1], fim[-1]), textcoords="offset points",
                xytext=(10, -10), fontsize=10, color=INK, fontweight="bold", zorder=5)

    ax.axhline(influ, color=C_INFLU, linewidth=2.4, linestyle="--", zorder=3)
    ax.annotate(f"InfluCoder (68M encoder)  {influ:+.3f}", (ranks[0], influ),
                textcoords="offset points", xytext=(-4, 8), ha="left",
                fontsize=10, color=INK, fontweight="bold", zorder=5)

    ax.axhline(args.untrained_agg, color=INK_MUTED, linewidth=1.2,
               linestyle=":", alpha=0.85, zorder=2)
    ax.annotate(f"untrained encoder  {args.untrained_agg:+.3f}",
                (ranks[0], args.untrained_agg), textcoords="offset points",
                xytext=(-4, -14), ha="left", fontsize=8.5, color=INK_MUTED, zorder=5)

    ax.axhline(0, color=INK_MUTED, linewidth=1.0, alpha=0.45, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlim(ranks[0] * 0.82, ranks[-1] * 1.55)
    ys = raw + fim + [influ, args.untrained_agg, 0.0]
    span = max(ys) - min(ys)
    ax.set_ylim(min(ys) - 0.10 * span, max(ys) + 0.12 * span)

    ax.set_xlabel("LoGRA rank", fontsize=10.5, color=INK)
    ax.set_ylabel(("aggregate" if args.metric == "agg" else "per-anchor mean")
                  + " Spearman $\\rho$ vs. gradient-influence GT",
                  fontsize=10.5, color=INK)
    ax.set_title("Paper setting: SmolLM2-1.7B native, 200x200 eval, GT proj 65536\n"
                 "LoGRA needs rank to compete; InfluCoder scores with a 68M encoder",
                 fontsize=12, color=INK, pad=12)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    # Legend below the axes: the plot interior is fully occupied (curves above,
    # the untrained-encoder floor below), so any in-axes placement collides.
    handles = [
        plt.Line2D([], [], color=C_INFLU, linewidth=2.4, linestyle="--",
                   label="InfluCoder — 68M enc., 1 forward"),
        plt.Line2D([], [], color=C_RAW, linewidth=2.2, marker="o", markersize=8,
                   label="LoGRA raw — 1.7B, fwd+bwd"),
        plt.Line2D([], [], color=C_FIM, linewidth=2.2, marker="s", markersize=8,
                   label="LoGRA FIM — 1.7B, fwd+bwd"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.015), labelcolor=INK)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        path = f"{args.out}.{ext}"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
