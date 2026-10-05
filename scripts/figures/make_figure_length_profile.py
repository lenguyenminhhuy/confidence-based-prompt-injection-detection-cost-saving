#!/usr/bin/env python3
"""Rendered-length profile of the two evaluation sets, as a figure.

Replaces the two request-share columns that made Table 3 (the latency-versus-length
profile) ten columns wide. Panel (a) gives the share of requests in each of the
bands that table uses, so the figure maps onto it row for row; panel (b) gives the
full distribution as an ECDF, which the banded columns could not show.

Both sets are tokenised with the Llama Stage-1 tokeniser through the same prompt
template used at inference, so the lengths are the ones the cost model prices.

PAYLOAD HYGIENE: reads label and rendered token count only. No input
text is stored, printed or plotted.

    PYTHONPATH=. python scripts/figures/make_figure_length_profile.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tokenizers import Tokenizer                                              # noqa: E402
from src.models.prompt_template import load_model_config, format_prompt       # noqa: E402

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SPLIT_COLOR = {"in_dist": BLUE, "ood": ORANGE}

MODEL, TOK = "llama3.2-1b", "results/stage1/llama3.2-1b/adapter/tokenizer.json"
EVAL, VAL = "data/eval_proposal/eval.jsonl", "data/train_proposal/val.jsonl"
EDGES = [77, 128, 192, 256, 384, 512, 1024, 2048]
LABELS = ["floor\n71--77", "128", "192", "256", "384", "512", "1024", "2048", ">2048"]
OUT = ROOT / "results" / "figures" / "length_profile_double.pdf"


def style() -> None:
    base = 8.5
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base - 1.3, "ytick.labelsize": base - 0.8,
        "legend.fontsize": base - 1.3, "legend.frameon": False,
        "legend.handlelength": 1.6, "legend.labelspacing": 0.28,
        "axes.linewidth": 0.6, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    })


def ntok(path: str) -> np.ndarray:
    cfg = load_model_config(MODEL)
    tok = Tokenizer.from_file(str(ROOT / TOK))
    texts = []
    with open(ROOT / path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                texts.append(format_prompt(cfg, r.get("text") or r.get("input") or ""))
    return np.array([len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)], float)


def shares(n: np.ndarray) -> np.ndarray:
    out, lo = [], 0
    for hi in EDGES:
        out.append(float(((n > lo) & (n <= hi)).mean()) if lo else float((n <= hi).mean()))
        lo = hi
    out.append(float((n > EDGES[-1]).mean()))
    return np.array(out)


def main() -> None:
    style()
    ne, nv = ntok(EVAL), ntok(VAL)
    se, sv = shares(ne), shares(nv)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))

    ax = axes[0]
    x = np.arange(len(LABELS)); w = 0.38
    ax.bar(x - w / 2, se * 100, w, color=SPLIT_COLOR["ood"], label="Evaluation (OOD)")
    ax.bar(x + w / 2, sv * 100, w, color=SPLIT_COLOR["in_dist"], label="Validation (in-dist.)")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_xlabel("Rendered prompt length (tokens), band upper edge")
    ax.set_ylabel("Requests (\\%)" if mpl.rcParams["text.usetex"] else "Requests (%)")
    ax.set_yticks([0, 20, 40, 60])
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.set_title("(a) Share of requests per length band", loc="left", pad=4)

    ax = axes[1]
    for n, key, lab in ((ne, "ood", "Evaluation (OOD)"), (nv, "in_dist", "Validation (in-dist.)")):
        xs = np.sort(n)
        ax.step(xs, np.arange(1, xs.size + 1) / xs.size * 100, where="post",
                color=SPLIT_COLOR[key], label=lab)
        ax.axvline(np.median(n), color=SPLIT_COLOR[key], linewidth=0.7, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlim(60, 4000)
    ax.set_xticks([77, 128, 256, 512, 1024, 2048])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("Rendered prompt length (tokens)")
    ax.set_ylabel("Requests at or below (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.axvline(512, color=INK_2, linewidth=0.6, linestyle="--")
    ax.annotate("512", (512, 4), xytext=(2, 0), textcoords="offset points",
                color=INK_2, fontsize=mpl.rcParams["xtick.labelsize"])
    ax.legend(loc="lower right")
    ax.set_title("(b) Cumulative length distribution", loc="left", pad=4)

    fig.tight_layout(pad=0.4, w_pad=1.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT); fig.savefig(OUT.with_suffix(".png"), dpi=200)

    data = {
        "tokenizer": MODEL, "bands": LABELS,
        "eval": {"n": int(ne.size), "median": float(np.median(ne)), "mean": float(ne.mean()),
                 "at_floor": float((ne <= 77).mean()), "ge_512": float((ne >= 512).mean()),
                 "shares": se.round(4).tolist()},
        "val": {"n": int(nv.size), "median": float(np.median(nv)), "mean": float(nv.mean()),
                "at_floor": float((nv <= 77).mean()), "ge_512": float((nv >= 512).mean()),
                "shares": sv.round(4).tolist()},
    }
    (ROOT / "results/analysis/length_profile_figure_data.json").write_text(json.dumps(data, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "shares"}
                      for k, v in data.items() if k in ("eval", "val")}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
