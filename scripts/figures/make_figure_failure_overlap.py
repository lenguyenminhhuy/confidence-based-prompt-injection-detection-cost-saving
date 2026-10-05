"""Figure: attacks Stage 1 waves through are attacks Stage 2 also misses.

Panel A  Stage-2 detection rate against the Stage-1 routing score, on attacks
         only. If routing discarded catchable attacks this curve would be flat.
Panel B  Stage-2 injection-score distribution for the attacks Stage 1 escalated
         versus the ones it auto-passed, with Stage-2's own 1%-FPR threshold.
         The auto-passed mass sits far below the threshold: Stage 2 scores those
         inputs benign too.

Usage:
    python scripts/figures/make_figure_failure_overlap.py --out results/figures
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
FPR_TARGET, THETA_SAFE = 0.01, 0.005
S2 = "results/stage2/mistral-7b-v0.1/eval_logits.jsonl"
ARMS = [("Qwen2.5-1.5B", "results/stage1_prec/bf16/qwen2.5-1.5b/eval_logits.jsonl", BLUE),
        ("Llama3.2-1B",  "results/stage1_prec/bf16/llama3.2-1b/eval_logits.jsonl",  ORANGE)]
EDGES = [0, 1e-4, 1e-3, 5e-3, 2e-2, 1e-1, 1.0 + 1e-9]
LABELS = [r"$<10^{-4}$", r"$10^{-4}$–$10^{-3}$", r"$10^{-3}$–$5{\times}10^{-3}$",
          r"$5{\times}10^{-3}$–$0.02$", r"$0.02$–$0.1$", r"$>0.1$"]


def style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 7.6, "xtick.labelsize": 7.7, "ytick.labelsize": 7.7,
        "legend.fontsize": 6.4, "legend.frameon": False,
        "axes.linewidth": 0.6, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def p_safe(p): return np.asarray([json.loads(l)["p_safe"] for l in open(ROOT / p) if l.strip()], float)


def thr(score, y, target):
    b = score[y == 0]
    return max((t for t in np.unique(b) if (b >= t).mean() <= target),
               key=lambda t: (score[y == 1] >= t).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()
    style()

    y = np.array([int(json.loads(l)["label"]) for l in open(ROOT / a.eval_split) if l.strip()])
    s2 = 1.0 - p_safe(S2)
    t2 = thr(s2, y, FPR_TARGET)
    atk = y == 1
    f2 = s2 >= t2
    dr_all = f2[atk].mean()

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 2.75))

    # ---- Panel A ----
    w = 0.38
    for j, (name, path, col) in enumerate(ARMS):
        v = p_safe(path)[atk]
        g = f2[atk]
        xs, hs, ns = [], [], []
        for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
            m = (v >= lo) & (v < hi)
            xs.append(i + (j - 0.5) * w)
            hs.append(g[m].mean() if m.sum() else np.nan)
            ns.append(int(m.sum()))
        axA.bar(xs, hs, width=w, color=col, label=name, linewidth=0)
        for x, h, nn in zip(xs, hs, ns):
            if np.isnan(h):
                continue
            axA.text(x, h + 0.012, str(nn), ha="center", va="bottom",
                     fontsize=5.6, color=INK_MUTED, rotation=90)
    axA.axhline(dr_all, color=INK_2, lw=0.7, ls=(0, (4, 2)))
    axA.text(len(EDGES) - 1.55, dr_all + 0.012,
             f"Stage 2 on all attacks ({dr_all:.3f})",
             fontsize=6.4, color=INK_2, ha="right")
    axA.axvline(2.5, color=RED, lw=0.7, ls=(0, (2, 2)))
    axA.text(2.62, 0.30, "auto-passed\n" + r"($p_{\mathrm{safe}}\!\geq\!\theta_{\mathrm{safe}}$)",
             fontsize=6.4, color=RED, ha="left", va="top", linespacing=1.35)
    axA.set_xticks(range(len(LABELS)))
    axA.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=6.2)
    axA.set_xlabel(r"Stage-1 routing score $p_{\mathrm{safe}}$ (attacks only)")
    axA.set_ylabel("Stage-2 detection rate")
    axA.set_ylim(0, 0.60)
    axA.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02))
    axA.set_title("(a) Stage 2 fails where Stage 1\n     is confident-benign", loc="left")

    # ---- Panel B ----
    name, path, col = ARMS[0]
    v = p_safe(path)
    esc = atk & (v < THETA_SAFE)
    leak = atk & (v >= THETA_SAFE)
    bins = np.linspace(0, 1, 51)
    axB.hist(s2[esc], bins=bins, color=BLUE, alpha=0.85, linewidth=0,
             label=f"escalated attacks (n={esc.sum()})", density=True)
    axB.hist(s2[leak], bins=bins, color=RED, alpha=0.80, linewidth=0,
             label=f"auto-passed attacks (n={leak.sum()})", density=True)
    axB.set_yscale("log")
    axB.set_ylim(top=axB.get_ylim()[1] * 12)
    axB.axvline(t2, color=INK, lw=0.8)
    axB.annotate(f"Stage-2 threshold\n@1% FPR ({t2:.2f})",
                 xy=(t2, 3.0), xytext=(t2 + 0.10, 6.0),
                 fontsize=6.2, color=INK, va="center", ha="left", linespacing=1.3,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    axB.set_xlabel("Stage-2 injection score")
    axB.set_ylabel("density")
    axB.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03))
    axB.set_title("(b) what Stage 1 waves through\n     is benign to Stage 2 too", loc="left")

    # ---- Panel C: per-channel, is the strong stage complementary? ----
    ch = np.array([json.loads(l).get("channel") for l in open(ROOT / a.eval_split) if l.strip()], object)
    name, path, _ = ARMS[0]
    s1 = 1.0 - p_safe(path)
    t1 = thr(s1, y, FPR_TARGET)
    f1 = s1 >= t1
    chans = ["direct", "document", "tool"]
    xs = np.arange(len(chans))
    w3 = 0.26
    s1dr = [f1[atk & (ch == c)].mean() for c in chans]
    s2dr = [f2[atk & (ch == c)].mean() for c in chans]
    resc = [f2[atk & (ch == c) & ~f1].mean() for c in chans]
    axC.bar(xs - w3, s1dr, width=w3, color=BLUE, linewidth=0, label="Stage-1 DR")
    axC.bar(xs,      s2dr, width=w3, color=GREEN, linewidth=0, label="Stage-2 DR")
    axC.bar(xs + w3, resc, width=w3, color=RED, linewidth=0,
            label="Stage 2 rescue rate\nof Stage-1 misses")
    for x, v in zip(xs + w3, resc):
        axC.text(x, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=6.0, color=RED)
    axC.set_xticks(xs)
    axC.set_xticklabels([f"{c}\n(n={int((atk & (ch == c)).sum())})" for c in chans], fontsize=6.6)
    axC.set_ylabel("rate")
    axC.set_ylim(0, 1.30)
    axC.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.03), fontsize=5.9,
               labelspacing=0.2, handlelength=1.2, handletextpad=0.4)
    axC.set_title("(c) stages complement each other\n     only on direct injection", loc="left")

    fig.subplots_adjust(wspace=0.46)
    outdir = ROOT / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        p = outdir / f"failure_overlap.{fmt}"
        fig.savefig(p, dpi=a.dpi if fmt == "png" else None)
        print("wrote", p)


if __name__ == "__main__":
    main()
