#!/usr/bin/env python3
"""EXP1 Part 3 (BIG_GPU_FINAL) plot: GPU-time vs samples-processed amortization.

Reads `baselines/out/<preset>/<profile>/exp1_part3_{less_logra,influcoder}.json`
(written by `part3.py`) and produces `exp1_part3_figure.{png,pdf}`. Same visual
language as this session's earlier standalone/combined Part 3 figures (see
EXP1.md section 7.6/8.4 and `.tuning_logs/plot_part3.py`): log-x cumulative
GPU time, log-y cumulative samples processed (a log axis can't include exactly
0, so each curve's pre-start "zero samples" segment is drawn as a flat
FLOOR=0.5 segment during setup/load, a plotting convention only). Filled
markers = actually measured at that n; hollow = extrapolated from the
steady-state per-sample rate. Star markers + labels mark where InfluCoder's
cumulative-samples line crosses LESS's/LoGRA's.

InfluCoder's steady-state rate uses the LARGEST measured n (100,000 this run
-- genuinely measured, not extrapolated, unlike earlier sessions where 100K
was itself an extrapolation from n=10,000). LESS/LoGRA are only measured at
n=100/1,000 (cost-prohibitive beyond that) and extrapolated from the n=1,000
rate for the 10K/100K/1M marker points and the crossover computation.

    python -m baselines.exp1.plot_part3
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

C_INFLUCODER = "#eb6834"
C_LESS = "#2a78d6"
C_LOGRA = "#7b3fb0"
C_RDSPLUS = "#1a9c76"
C_COLLECT = "#a8c8ef"
C_TRAIN = "#f5c9a8"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

# Reads from the consolidated EXP1_results.json (written by
# collect_results.py from Part 1/2/3's raw outputs), not
# exp1_part3_{less_logra,influcoder}.json directly -- one file every plot
# script reads its numbers from, so there is exactly one place to look to
# see what actually drove Figure 1.
IN_RESULTS = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED) / "EXP1_results.json"
OUT_PATH = Path("baselines/out") / cfg.PRESET / cfg.PROFILE / cfg.seed_dir(cfg.SEED) / "exp1_part3_figure"

FLOOR = 0.5
X_MIN = 5
MARK_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]


def samples_at(t, load_s, rate_s_per_sample):
    """Cumulative samples processed by wall-clock time t, given a fixed
    per-sample rate starting after `load_s` of setup/load time."""
    return np.maximum(0.0, (t - load_s) / rate_s_per_sample)


def find_crossover(load_a, rate_a, load_b, rate_b, t_max):
    """First t > max(load_a, load_b) where samples_at(t, a) == samples_at(t, b)."""
    t = np.linspace(max(load_a, load_b), t_max, 2_000_000)
    sa = samples_at(t, load_a, rate_a)
    sb = samples_at(t, load_b, rate_b)
    diff = sa - sb
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    i = sign_changes[0]
    return float(t[i]), float(sa[i])


def draw(ax, less_logra, influcoder):
    """Draw the full Part 3 amortization panel onto `ax`, returning
    `(cross_less, cross_logra, cross_rdsplus)` for the caller's summary print.
    Factored out of `main()` so `plot_figure1.py` can reuse this exact
    drawing code for its combined panel instead of a second, driftable copy
    of it. `cross_rdsplus` is None (and RDS+ isn't drawn at all) if `less_logra`
    predates RDS+ being added to Part 3 -- older result files stay plottable."""
    less_label, less_data = next(iter(less_logra["less"].items()))
    logra_label, logra_data = next(iter(less_logra["logra"].items()))
    rdsplus_item = next(iter(less_logra.get("rdsplus", {}).items()), None)

    # Steady-state rates: InfluCoder uses its LARGEST measured n (least
    # first-call-warmup-affected); LESS/LoGRA use n=1000 (the larger of
    # their two measured points).
    infl_points = {p["n"]: p for p in influcoder["process_points"]}
    infl_load_s = influcoder["setup_time_s"]
    infl_collect_s = influcoder["collect_time_s"]
    infl_rate = infl_points[max(infl_points)]["ms_per_sample"] / 1000.0

    less_points = {p["n"]: p for p in less_data["points"]}
    less_load_s = less_data["load_time_s"]
    less_rate = less_points[max(less_points)]["ms_per_sample"] / 1000.0

    logra_points = {p["n"]: p for p in logra_data["points"]}
    logra_load_s = logra_data["load_time_s"]
    logra_rate = logra_points[max(logra_points)]["ms_per_sample"] / 1000.0

    if rdsplus_item is not None:
        rdsplus_label, rdsplus_data = rdsplus_item
        rdsplus_points = {p["n"]: p for p in rdsplus_data["points"]}
        rdsplus_load_s = rdsplus_data["load_time_s"]
        rdsplus_rate = rdsplus_points[max(rdsplus_points)]["ms_per_sample"] / 1000.0

    t_max = infl_load_s + 1_000_000 * infl_rate
    cross_less = find_crossover(infl_load_s, infl_rate, less_load_s, less_rate, t_max)
    cross_logra = find_crossover(infl_load_s, infl_rate, logra_load_s, logra_rate, t_max)
    cross_rdsplus = (find_crossover(infl_load_s, infl_rate, rdsplus_load_s, rdsplus_rate, t_max)
                     if rdsplus_item is not None else None)

    ax.axvspan(X_MIN, infl_collect_s, color=C_COLLECT, alpha=0.5, zorder=0,
              label="InfluCoder: gradient collection")
    ax.axvspan(infl_collect_s, infl_load_s, color=C_TRAIN, alpha=0.5, zorder=0,
              label="InfluCoder: distillation training")

    def plot_curve(load_s, rate, color, label, measured_ns):
        t_line = np.geomspace(max(load_s, X_MIN), t_max, 500)
        s_line = samples_at(t_line, load_s, rate)
        s_line = np.maximum(s_line, FLOOR) if load_s <= X_MIN else s_line
        # Flat FLOOR segment before this method starts producing samples.
        t_pre = np.geomspace(X_MIN, max(load_s, X_MIN), 50)
        ax.plot(t_pre, np.full_like(t_pre, FLOOR), color=color, linewidth=1.6, alpha=0.35, zorder=1)
        ax.plot(t_line, s_line, color=color, linewidth=1.8, zorder=2, label=label)

        for n in MARK_SIZES:
            t_n = load_s + n * rate
            if t_n > t_max * 1.01:
                continue
            measured = n in measured_ns
            ax.scatter(t_n, n, s=70, marker="o",
                      facecolors=color if measured else "none",
                      edgecolors=color, linewidths=1.8, zorder=4)

    plot_curve(infl_load_s, infl_rate, C_INFLUCODER, "InfluCoder 68m", set(infl_points))
    plot_curve(less_load_s, less_rate, C_LESS, f"LESS ({less_label})", set(less_points))
    plot_curve(logra_load_s, logra_rate, C_LOGRA, f"LoGRA ({logra_label})", set(logra_points))
    if rdsplus_item is not None:
        plot_curve(rdsplus_load_s, rdsplus_rate, C_RDSPLUS, f"RDS+ ({rdsplus_label})", set(rdsplus_points))

    for cross, color, name in [(cross_less, C_LESS, "LESS"), (cross_logra, C_LOGRA, "LoGRA"),
                               (cross_rdsplus, C_RDSPLUS, "RDS+")]:
        if cross is None:
            continue
        t_c, s_c = cross
        ax.scatter([t_c], [s_c], marker="*", s=260, color=INK, edgecolors="white",
                  linewidths=0.8, zorder=5)
        ax.annotate(f"{t_c:.0f}s", (t_c, s_c), textcoords="offset points",
                   xytext=(8, -12), fontsize=8.5, color=INK, zorder=6)

    # x = cumulative GPU time: sqrt (not log) -- the y-axis already carries
    # the heavy-tailed compression (0.5 -> 2M samples, 6+ orders of
    # magnitude), and sqrt keeps time differences in the 0-1000s range (where
    # the setup cost and both crossovers actually land) more visually
    # distinguishable than log does, at the cost of compressing the long tail
    # out to t_max.
    ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.set_yscale("log")
    ax.set_xlim(X_MIN, t_max)
    ax.set_ylim(FLOOR, 2_000_000)
    ax.set_xlabel("cumulative GPU time (seconds, sqrt scale)", fontsize=10.5, color=INK)
    ax.set_ylabel("cumulative samples processed (log scale)", fontsize=10.5, color=INK)
    ax.set_title("EXP1 Part 3 (BIG_GPU_FINAL) — GPU-time vs samples-processed amortization\n"
                f"InfluCoder rate measured directly through n=100,000 (not extrapolated)",
                fontsize=11, color=INK, pad=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    legend_elems = [
        Line2D([0], [0], color=C_INFLUCODER, linewidth=1.8, label="InfluCoder 68m"),
        Line2D([0], [0], color=C_LESS, linewidth=1.8, label=f"LESS ({less_label})"),
        Line2D([0], [0], color=C_LOGRA, linewidth=1.8, label=f"LoGRA ({logra_label})"),
    ]
    if rdsplus_item is not None:
        legend_elems.append(Line2D([0], [0], color=C_RDSPLUS, linewidth=1.8,
                                   label=f"RDS+ ({rdsplus_label})"))
    legend_elems += [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK_MUTED,
              markeredgecolor=INK_MUTED, markersize=8, label="measured"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
              markeredgecolor=INK_MUTED, markersize=8, label="extrapolated"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=INK,
              markeredgecolor=INK, markersize=13, label="InfluCoder overtakes"),
        Line2D([0], [0], color=C_COLLECT, linewidth=8, alpha=0.5, label="InfluCoder: gradient collection"),
        Line2D([0], [0], color=C_TRAIN, linewidth=8, alpha=0.5, label="InfluCoder: distillation training"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=7.5, frameon=False)

    return cross_less, cross_logra, cross_rdsplus


def main():
    part3 = json.loads(IN_RESULTS.read_text())["part3"]
    less_logra, influcoder = part3["less_logra"], part3["influcoder"]

    fig, ax = plt.subplots(figsize=(8, 8))
    cross_less, cross_logra, cross_rdsplus = draw(ax, less_logra, influcoder)
    fig.tight_layout()
    fig.savefig(f"{OUT_PATH}.png", dpi=160)
    fig.savefig(f"{OUT_PATH}.pdf")
    print(f"wrote {OUT_PATH}.png / .pdf")

    if cross_less:
        print(f"InfluCoder overtakes LESS at {cross_less[0]:.0f}s ({cross_less[1]:.0f} samples)")
    if cross_logra:
        print(f"InfluCoder overtakes LoGRA at {cross_logra[0]:.0f}s ({cross_logra[1]:.0f} samples)")
    if cross_rdsplus:
        print(f"InfluCoder overtakes RDS+ at {cross_rdsplus[0]:.0f}s ({cross_rdsplus[1]:.0f} samples)")


if __name__ == "__main__":
    main()
