"""Figure: the operating plane. What each routing threshold buys, and what it costs.

One panel, because the argument is one sentence: every candidate routing
threshold is a point in (cost saved, attacks caught); up-and-right is better;
the shipped threshold has a calibration-selected alternative up and to its right,
so it is dominated.

x  measured latency reduction against M2-on-every-input (the cost axis)
y  end-to-end detection rate at a 1% FPR budget (the detection axis)
   each dot is one theta_safe; the curve is the sweep, annotated with the
   threshold values so a position maps back to a setting.

The reference point at (0, DR of M2-on-every-input) is "no cascade at all".
Anything above its horizontal line detects more than the strong stage alone.

Reads results/analysis/theta_safe_sweep.json only (produced by
scripts/eval/sweep_theta_safe.py). No logits, no GPU, no dataset text.

Usage:
    python scripts/figures/make_figure_theta_sweep.py --out results/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
SWEEP = "results/analysis/theta_safe_sweep.json"
COLOURS = {"Qwen-1.5B": BLUE, "Llama-1B": ORANGE}

# Which thresholds get a printed label, and where the label sits relative to its
# point, per arm. Labelling all 12 is unreadable; these span the range.
ANNOT = {
    "Qwen-1.5B": {0.0001: (2, 11), 0.005: (-19, 9), 0.316: (-2, 10)},
    "Llama-1B": {1e-05: (13, -3), 0.005: (-6, -11), 0.316: (-3, -11)},
}

# One Qwen point sits far below the plotted band; showing it would spend most of
# the vertical space on a region strictly worse than having no cascade at all.
# It is excluded from the axes and called out in text instead.
YLIM = (0.3985, 0.4375)


def style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 8.5, "xtick.labelsize": 7.7, "ytick.labelsize": 7.7,
        "legend.fontsize": 7.0, "legend.frameon": False,
        "axes.linewidth": 0.6, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _mark(rows, key):
    for i, r in enumerate(rows):
        if key in (r.get("note") or ""):
            return i
    return None


def _fmt_theta(t: float) -> str:
    """5e-03 -> $5{\\times}10^{-3}$, 1e-04 -> $10^{-4}$."""
    e = int(np.floor(np.log10(t)))
    m = t / 10 ** e
    if abs(m - 1.0) < 0.02:
        return rf"$10^{{{e}}}$"
    return rf"${m:.2f}".rstrip("0").rstrip(".") + rf"{{\times}}10^{{{e}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()
    style()

    D = json.loads((ROOT / a.sweep).read_text())
    s2_only, frozen = D["stage2_only_dr"], D["frozen_theta_safe"]

    fig, ax = plt.subplots(figsize=(4.9, 3.4))

    # "no cascade" reference: pay the guard on every input, save nothing.
    ax.axhline(s2_only, color=INK_MUTED, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.plot([0.0], [s2_only], "o", ms=5.0, mfc="white", mec=INK, mew=1.0, zorder=6)
    ax.annotate("M2 on every input (no cascade)", xy=(0.0, s2_only),
                xytext=(0.011, s2_only - 0.0008), fontsize=6.8, color=INK_2,
                va="top", ha="left")

    for name, arm in D["sweep"].items():
        col = COLOURS.get(name, INK_2)
        # theta_safe = 1 escalates everything and costs more than the guard alone
        # (both stages run); it is off-plane and excluded.
        rows = [r for r in arm["rows"] if r["theta_safe"] < 1.0]
        th = np.array([r["theta_safe"] for r in rows])
        dr = np.array([r["e2e_dr"] for r in rows])
        lat = np.array([r["latency_reduction"] for r in rows])
        i_cal, i_frz = _mark(rows, "CAL-SELECTED"), _mark(rows, "FROZEN")

        ax.plot(lat, dr, "-", color=col, lw=1.1, marker="o", ms=3.0,
                mfc=col, mec="none", label=name, zorder=3)

        for t, (dx, dy) in ANNOT.get(name, {}).items():
            k = int(np.argmin(np.abs(th - t)))
            ax.annotate(_fmt_theta(th[k]), xy=(lat[k], dr[k]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=6.3, color=col, ha="center",
                        va="center", zorder=4)

        if i_frz is not None:
            ax.plot(lat[i_frz], dr[i_frz], "s", ms=7.5, mfc="none", mec=RED,
                    mew=1.2, zorder=7)
        if i_cal is not None:
            ax.plot(lat[i_cal], dr[i_cal], "D", ms=6.6, mfc="none", mec=GREEN,
                    mew=1.2, zorder=7)
        if i_frz is not None and i_cal is not None:
            ax.annotate("", xy=(lat[i_cal], dr[i_cal]), xytext=(lat[i_frz], dr[i_frz]),
                        arrowprops=dict(arrowstyle="-|>", lw=0.9, color=GREEN,
                                        shrinkA=7.0, shrinkB=7.0, mutation_scale=8),
                        zorder=5)

    ax.set_xlabel("cost saved vs running the guard on every input")
    ax.set_ylabel("attacks caught (DR at 1% FPR)")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(-0.032, 0.575)
    ax.set_ylim(*YLIM)

    # name the excluded point rather than silently dropping it
    off = [(n, r) for n, arm in D["sweep"].items() for r in arm["rows"]
           if r["theta_safe"] < 1.0 and r["e2e_dr"] < YLIM[0]]
    for k, (n, r) in enumerate(off):
        ax.annotate(rf"off-scale: $\theta_{{\mathrm{{safe}}}}={_fmt_theta(r['theta_safe'])[1:-1]}$, "
                    rf"{r['latency_reduction']:.0%}, DR {r['e2e_dr']:.3f}",
                    xy=(0.045, 0.72 - 0.05 * k), xycoords="axes fraction",
                    fontsize=6.3, color=COLOURS.get(n, INK_2), ha="left", va="bottom")

    # direction-of-good, so the plane reads without the caption
    ax.annotate("", xy=(0.170, 0.40), xytext=(0.055, 0.26), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK_MUTED,
                                mutation_scale=7))
    ax.annotate("better", xy=(0.180, 0.410), xycoords="axes fraction", fontsize=6.8,
                color=INK_MUTED, ha="left", va="center", style="italic")

    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls="none", marker="s", ms=7.0, mfc="none", mec=RED,
                           mew=1.2, label=rf"shipped ($\theta_{{\mathrm{{safe}}}}={frozen:g}$)"),
                plt.Line2D([], [], ls="none", marker="D", ms=6.2, mfc="none", mec=GREEN,
                           mew=1.2, label="calibration-selected, log grid")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.015, 1.02),
              labelspacing=0.35, handletextpad=0.5)

    outdir = ROOT / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = outdir / f"theta_sweep.{fmt}"
        fig.savefig(p, dpi=a.dpi if fmt == "png" else None)
        print("wrote", p)


if __name__ == "__main__":
    main()
