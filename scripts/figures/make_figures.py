#!/usr/bin/env python3
"""Paper figures for the cascade prompt-injection detector (E1-E7).

Every chart is regenerated from the frozen per-sample score files, through the
SAME code paths that produced the reported numbers
(``src.evaluation.metrics.detection_rate_at_fpr`` and
``scripts.eval.eval_cascade._cascade_point``), so a figure cannot drift from the
tables in the write-up.

PAYLOAD HYGIENE: reads only ``id`` / ``label`` / ``channel`` and
``len(text)`` from the split files, plus per-row scores from the logit files.
Input text is never stored, printed or plotted.

Usage
-----
    PYTHONPATH=. python scripts/figures/make_figures.py                     # all figs, both widths
    PYTHONPATH=. python scripts/figures/make_figures.py --only roc,pr
    PYTHONPATH=. python scripts/figures/make_figures.py --width double --formats pdf

Writes ``results/figures/<name>_<width>.{pdf,png}`` and
``results/figures/figure_data.json`` (every plotted number, for LaTeX tables,
caption text and the accessible table view).

Design notes
------------
* Colour is assigned per ENTITY (a detector keeps its hue in every figure) from
  a CVD-validated categorical order; each series also carries a distinct
  linestyle + marker, so identity survives greyscale printing and colour-vision
  deficiency.  Panels are faceted so no panel exceeds four coloured series.
* Bars carry direct value labels (required relief for the low-contrast hues) and
  a white 0.6pt edge as the inter-bar spacer.
* No chart titles: the LaTeX caption is the title.  Panels are labelled (a)/(b).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.eval.eval_cascade import _cascade_point, _read_labels, _read_logits  # noqa: E402
from src.evaluation.metrics import detection_rate_at_fpr, ece  # noqa: E402

FPR_TARGET = 0.01
OUT_DEFAULT = ROOT / "results/figures"

# ---------------------------------------------------------------------------
# palette / entity registry
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#d8d7d2"
SURFACE = "#ffffff"

# CVD-validated categorical order (all-pairs PASS for the 4 slots used per panel).
BLUE, ORANGE, AQUA, VIOLET, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e34948"

# entity -> (label, colour, linestyle, marker)
ENT = {
    "mistral7b":    ("M2 (Mistral-7B, Stage 2)",      BLUE,   "-",  "o"),
    "qwen":         ("Qwen-2.5-1.5B (Stage 1)",       ORANGE, "--", "s"),
    "llama":        ("Llama-3.2-1B (Stage 1)",        AQUA,   "-.", "^"),
    "granite":      ("Granite-Guardian-2B (Stage 1)", VIOLET, ":",  "D"),
    "protectai":    ("ProtectAI DeBERTa-v3 (v2)",     RED,    "--", "v"),
    "promptguard2": ("PromptGuard 2 (86M)",           YELLOW, "-.", "P"),
    "datasentinel": ("DataSentinel-7B (verdict)",     INK_2,  "",   "X"),
    "cascade_llama": ("Cascade: Llama-1B -> M2",      AQUA,   "-",  "^"),
    "cascade_qwen":  ("Cascade: Qwen-1.5B -> M2",     ORANGE, "-",  "s"),
}
# compact labels for the narrow single-column variant
SHORT = {"mistral7b": "M2 (Mistral-7B)", "qwen": "Qwen-2.5-1.5B", "llama": "Llama-3.2-1B",
         "granite": "Granite-Guardian-2B", "protectai": "ProtectAI DeBERTa",
         "promptguard2": "PromptGuard 2", "datasentinel": "DataSentinel-7B",
         "cascade_llama": "Cascade: Llama-1B", "cascade_qwen": "Cascade: Qwen-1.5B"}


def elabel(k: str, width: str) -> str:
    return SHORT[k] if width == "single" else ENT[k][0]


CHANNEL_COLOR = {"direct": BLUE, "document": ORANGE, "tool": AQUA}
SPLIT_COLOR = {"in_dist": BLUE, "ood": ORANGE}

OURS = ["mistral7b", "qwen", "llama", "granite"]
OFFSHELF = ["protectai", "promptguard2"]

WIDTHS = {"double": 7.0, "single": 3.4}


def style(width: str) -> None:
    base = 8.5 if width == "double" else 7.2
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base - 0.8, "ytick.labelsize": base - 0.8,
        "legend.fontsize": base - 1.3, "legend.frameon": False,
        "legend.handlelength": 2.0, "legend.labelspacing": 0.28,
        "legend.borderpad": 0.2, "legend.handletextpad": 0.5,
        "axes.linewidth": 0.6, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 4.0,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.grid": False,
    })


def grid(ax, axis="both"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, linestyle="-")


def panels(width: str, n: int, h_double: float, h_single: float):
    """Facet layout: side-by-side at double width, stacked at single width."""
    if width == "double":
        fig, axes = plt.subplots(1, n, figsize=(WIDTHS["double"], h_double))
    else:
        fig, axes = plt.subplots(n, 1, figsize=(WIDTHS["single"], h_single * n))
    return fig, (np.atleast_1d(axes).ravel().tolist())


def plabel(ax, txt):
    ax.set_title(txt, loc="left", fontsize=mpl.rcParams["axes.titlesize"], color=INK, pad=4)


def thin(x, y, n=1200):
    """Thin a step curve for compact vector output, keeping endpoints/extremes."""
    if len(x) <= n:
        return x, y
    idx = np.unique(np.concatenate([[0, len(x) - 1], np.linspace(0, len(x) - 1, n).astype(int)]))
    return x[idx], y[idx]


def barlabel(ax, rects, fmt="{:.2f}", dy=0.012, na_mask=None):
    for i, r in enumerate(rects):
        h = r.get_height()
        if na_mask is not None and na_mask[i]:
            ax.text(r.get_x() + r.get_width() / 2, dy, "n/a", ha="center", va="bottom",
                    fontsize=mpl.rcParams["legend.fontsize"] - 0.3, color=INK_MUTED, rotation=90)
            continue
        ax.text(r.get_x() + r.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom",
                fontsize=mpl.rcParams["legend.fontsize"] - 0.3, color=INK_2)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
STAGE1_DIRS = {"llama": "results/stage1/llama3.2-1b",
               "qwen": "results/stage1/qwen2.5-1.5b",
               "granite": "results/stage1/granite-guardian-2b"}
STAGE2_DIR = "results/stage2/mistral-7b-v0.1"
EVAL_SPLIT = "data/eval_proposal/eval.jsonl"
CAL_SPLIT = "data/train_proposal/cal.jsonl"
VAL_SPLIT = "data/train_proposal/val.jsonl"
CASCADES = {"cascade_llama": ("llama", "results/analysis/cascade_nf4_llama3.2-1b/cascade_summary.json"),
            "cascade_qwen": ("qwen", "results/analysis/cascade_nf4_qwen2.5-1.5b/cascade_summary.json")}


def _meta(path: Path):
    """id / label / channel / len(text) only -- never the text itself."""
    ids, lab, ch, tlen = [], [], [], []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ids.append(r["id"]); lab.append(int(r["label"]))
            ch.append(r.get("channel") or "")
            tlen.append(len(r.get("text") or ""))
    return ids, np.asarray(lab, int), np.asarray(ch, object), np.asarray(tlen, int)


# Escalation rate e is a property of the Stage-1 scores, not of the latency measurement,
# and it differs between quantization regimes (bf16 and NF4 scores of the same adapter
# differ by ~5.2% on eval, so the same theta_safe routes a different fraction). It is
# therefore sourced from the scoring artifacts, never carried inside a latency file.
BF16_COST_ARTIFACTS = {"llama3.2-1b": "results/analysis/cost_lwfull_llama3.2-1b_bf16.json",
                       "qwen2.5-1.5b": "results/analysis/cost_lwfull_qwen2.5-1.5b_bf16.json"}
SECONDS_PER_HOUR = 3600.0


def _cascade_block(doc: dict, e_by_model: dict, tokens: int = 512) -> dict:
    """{model: {k1_ms, k2_ms, e, k1_over_k2, breakeven_e, reduction_at_e, ...}} at one length.

    Accepts either latency schema: the original files carry this block precomputed at 512
    tokens; scripts/cost/benchmark_latency_curve.py records a full profile instead, from which
    the same quantities are derived here. Deriving them keeps e out of the latency file,
    so a curve cannot silently pair one regime's latencies with another regime's routing.
    """
    key = str(tokens)
    per = doc["per_model"]
    s2 = next(m for m, spec in per.items() if spec.get("role") == "stage2")
    k2 = per[s2]["by_length"][key]["median_ms"]
    usd = doc.get("gpu_hourly_usd_assumption", 1.20)
    per_m = lambda ms: ms / 1000.0 / SECONDS_PER_HOUR * usd * 1e6
    out = {}
    for m, spec in per.items():
        if spec.get("role") != "stage1" or m not in e_by_model:
            continue
        k1 = spec["by_length"][key]["median_ms"]
        e = e_by_model[m]
        r = k1 / k2
        out[m] = {"k1_ms": k1, "k2_ms": k2, "e": e, "k1_over_k2": round(r, 4),
                  "breakeven_e": round(1.0 - r, 4), "reduction_at_e": 1.0 - r - e,
                  "cascade_cost": {"latency_ms": k1 + e * k2,
                                   "usd_per_1m": per_m(k1 + e * k2)},
                  "guard_cost": {"latency_ms": k2, "usd_per_1m": per_m(k2)}}
    return out


def load(latency: str, latency_bf16: str) -> dict:
    d: dict = {}
    ids, y, ch, tlen = _meta(ROOT / EVAL_SPLIT)
    d["ids"], d["y"], d["ch"], d["tlen"] = ids, y, ch, tlen

    # generative detectors: p_safe -> injection score
    d["psafe"], d["score"] = {}, {}
    for k, sub in list(STAGE1_DIRS.items()) + [("mistral7b", STAGE2_DIR)]:
        _, _, ps = _read_logits(ROOT / sub / "eval_logits.jsonl")
        if len(ps) != len(y):
            raise SystemExit(f"ROW MISMATCH {sub}/eval_logits={len(ps)} vs eval={len(y)}")
        d["psafe"][k] = ps
        d["score"][k] = 1.0 - ps

    # Optional off-the-shelf baselines (roc/pr/channel figures only; not in the paper).
    # No script in this repository produces these prediction files.
    for k, sub in [("promptguard2", "results/baselines/predictions/promptguard2_86m.jsonl"),
                   ("protectai", "results/baselines/predictions/protectai_deberta_v2.jsonl")]:
        if not (ROOT / sub).exists():
            continue
        rows = [json.loads(l) for l in (ROOT / sub).open() if l.strip()]
        if [r["id"] for r in rows] != ids:
            rows = sorted(rows, key=lambda r: ids.index(r["id"]))
        d["score"][k] = np.asarray([r["score"] for r in rows], float)

    # DataSentinel-7B: verdict only (score is null) -> one operating point
    ds_path = ROOT / "results/predictions/datasentinel_7b.jsonl"
    if ds_path.exists():
        ds = {json.loads(l)["id"]: json.loads(l) for l in ds_path.open() if l.strip()}
        d["ds_pred"] = np.asarray([int(ds[i]["pred"]) for i in ids], int)

    # cal / val (calibration + generalisation-gap panels)
    d["cal"] = {"y": _read_labels(ROOT / CAL_SPLIT)}
    d["val"] = {"y": _read_labels(ROOT / VAL_SPLIT)}
    for split, sub in (("cal", "cal_logits"), ("val", "val_logits")):
        for k, sd in list(STAGE1_DIRS.items()) + [("mistral7b", STAGE2_DIR)]:
            p = ROOT / sd / f"{sub}.jsonl"
            if p.exists():
                lb, li, ps = _read_logits(p)
                if len(ps) == len(d[split]["y"]):
                    d[split][k] = {"psafe": ps, "score": 1.0 - ps, "logp": (lb, li)}
    lb, li, _ = _read_logits(ROOT / STAGE2_DIR / "eval_logits.jsonl")
    d["eval_logp"] = {"mistral7b": (lb, li)}
    for k, sd in STAGE1_DIRS.items():
        lb, li, _ = _read_logits(ROOT / sd / "eval_logits.jsonl")
        d["eval_logp"][k] = (lb, li)

    d["summaries"] = {k: json.loads((ROOT / p).read_text()) for k, (_, p) in CASCADES.items()}
    lat = json.loads((ROOT / latency).read_text())
    lat_bf = json.loads((ROOT / latency_bf16).read_text())
    e_nf4 = {mname: d["summaries"][k]["headline_cal_frozen"]["on_eval"]["escalation_rate"]
             for k, mname in (("cascade_llama", "llama3.2-1b"), ("cascade_qwen", "qwen2.5-1.5b"))}
    e_bf16 = {m: json.loads((ROOT / p).read_text())["escalation_rate_measured"]
              for m, p in BF16_COST_ARTIFACTS.items()}
    if "cascade_k1k2" not in lat or "k1_ms" not in next(iter(lat["cascade_k1k2"].values()), {}):
        lat["cascade_k1k2"] = _cascade_block(lat, e_nf4)
    if "cascade_k1k2" not in lat_bf or "k1_ms" not in next(iter(lat_bf["cascade_k1k2"].values()), {}):
        lat_bf["cascade_k1k2"] = _cascade_block(lat_bf, e_bf16)
    d["latency"], d["latency_bf16"] = lat, lat_bf
    d["stage1_sel"] = json.loads((ROOT / "results/metrics/stage1_selection.json").read_text())
    return d


def dr_at(y, s, target=FPR_TARGET):
    r = detection_rate_at_fpr(y, s, (target,))[str(target)]
    return r["dr"], r["threshold"], r["achieved_fpr"]


def cascade_preds(D, key):
    """Final cascade decision per row at the frozen (cal-selected) operating point."""
    s1_key, _ = CASCADES[key]
    summ = D["summaries"][key]
    th_safe = summ["headline_cal_frozen"]["theta_safe"]
    s1_ps, s2 = D["psafe"][s1_key], D["score"]["mistral7b"]
    pt = _cascade_point(s1_ps, s2, D["y"], th_safe, FPR_TARGET)
    esc = s1_ps < th_safe
    pred = np.zeros(len(D["y"]), int)
    if pt["theta2"] is not None:
        pred[esc & (s2 >= pt["theta2"])] = 1
    return pred, pt


def channel_recall(y, ch, pred):
    out = {}
    for c in ("direct", "document", "tool"):
        m = (y == 1) & (ch == c)
        out[c] = float(pred[m].mean()) if m.sum() else float("nan")
    out["overall"] = float(pred[y == 1].mean())
    return out


def recall_by_channel_at_1pct(y, ch, s):
    _, thr, _ = dr_at(y, s)
    return channel_recall(y, ch, (np.asarray(s) >= thr).astype(int))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_roc(D, width):
    fig, ax = panels(width, 2, 3.05, 2.85)
    data = {}
    for k in D["score"]:
        fpr, tpr, _ = roc_curve(D["y"], D["score"][k])
        data[k] = {"auroc": float(roc_auc_score(D["y"], D["score"][k]))}
        D.setdefault("_roc", {})[k] = (fpr, tpr)
    groups = [(0, OURS, "(a) Models fine-tuned in this work"),
              (1, ["mistral7b"] + OFFSHELF, "(b) Off-the-shelf detectors")]
    for i, keys, title in groups:
        a = ax[i]
        a.plot([0, 1], [0, 1], color=INK_MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        a.text(0.60, 0.53, "chance", color=INK_MUTED, rotation=38, fontsize=mpl.rcParams["legend.fontsize"] - 0.5)
        for k in keys:
            _, col, ls, mk = ENT[k]; lab = elabel(k, width)
            fpr, tpr = thin(*D["_roc"][k])
            ref = (i == 1 and k == "mistral7b")
            a.plot(fpr, tpr, color=col, ls=(0, (5, 2)) if ref else ls, lw=1.2 if ref else 1.5,
                   label=f"{lab} — AUROC {data[k]['auroc']:.3f}", zorder=3)
            dr, thr, afpr = dr_at(D["y"], D["score"][k])
            a.plot([afpr], [dr], marker=mk, color=col, ms=5.0, mec=SURFACE, mew=0.8, ls="none", zorder=4)
            data[k]["dr_at_1pct_fpr"] = float(dr)
        if i == 1:
            y1 = float(D["ds_pred"][D["y"] == 1].mean()); x1 = float(D["ds_pred"][D["y"] == 0].mean())
            _, col, _, mk = ENT["datasentinel"]; lab = elabel("datasentinel", width)
            a.plot([x1], [y1], marker=mk, color=col, ms=6.5, ls="none", zorder=5,
                   label=f"{lab} — FPR {x1:.2f}")
            a.annotate(f"flags {x1:.0%} of benign", xy=(x1, y1), xytext=(0.26, 0.56), color=INK_2,
                       fontsize=mpl.rcParams["legend.fontsize"] - 0.3, ha="left",
                       arrowprops=dict(arrowstyle="-", lw=0.6, color=INK_MUTED))
            data["datasentinel"] = {"tpr": y1, "fpr": x1}
        a.set_xlim(0, 1); a.set_ylim(0, 1.02)
        a.set_xlabel("False positive rate"); a.set_ylabel("Detection rate (TPR)")
        grid(a)
        plabel(a, title)
        a.legend(loc="lower right", markerscale=0.9)
    fig.tight_layout(pad=0.4)
    return fig, {"note": "ROC on the 25,747-row OOD eval; marker = DR at 1% FPR", "detectors": data}


def fig_roc_lowfpr(D, width):
    fig, ax = panels(width, 2, 3.05, 2.85)
    data = {}
    groups = [(0, OURS, "(a) Models fine-tuned in this work"),
              (1, ["mistral7b"] + OFFSHELF, "(b) Off-the-shelf detectors")]
    for i, keys, title in groups:
        a = ax[i]
        a.axvline(FPR_TARGET, color=INK_MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        a.text(FPR_TARGET * 1.2, 0.62, "1% FPR", color=INK_2, fontsize=mpl.rcParams["legend.fontsize"] - 0.3)
        for k in keys:
            _, col, ls, mk = ENT[k]; lab = elabel(k, width)
            fpr, tpr, _ = roc_curve(D["y"], D["score"][k])
            m = fpr > 0
            fx, ty = thin(fpr[m], tpr[m])
            ref = (i == 1 and k == "mistral7b")
            dr, _, afpr = dr_at(D["y"], D["score"][k])
            a.plot(fx, ty, color=col, ls=(0, (5, 2)) if ref else ls, lw=1.2 if ref else 1.5,
                   label=f"{lab} — DR {dr:.3f}", zorder=3)
            a.plot([afpr], [dr], marker=mk, color=col, ms=5.0, mec=SURFACE, mew=0.8, ls="none", zorder=4)
            data[k] = {"dr_at_1pct_fpr": float(dr), "achieved_fpr": float(afpr)}
        a.set_xscale("log"); a.set_xlim(1e-4, 1.0); a.set_ylim(0, 1.02)
        a.set_xlabel("False positive rate (log)"); a.set_ylabel("Detection rate (TPR)")
        grid(a); plabel(a, title)
        a.legend(loc="upper left", markerscale=0.9)
    fig.tight_layout(pad=0.4)
    return fig, {"note": "low-FPR region; legend reports DR at the 1% FPR operating point", "detectors": data}


def fig_pr(D, width):
    fig, ax = panels(width, 2, 3.05, 2.85)
    prev = float((D["y"] == 1).mean())
    data = {"prevalence": prev}
    groups = [(0, OURS, "(a) Models fine-tuned in this work"),
              (1, ["mistral7b"] + OFFSHELF, "(b) Off-the-shelf detectors")]
    for i, keys, title in groups:
        a = ax[i]
        a.axhline(prev, color=INK_MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        a.text(0.02, prev + 0.02, f"prevalence {prev:.2f}", color=INK_2,
               fontsize=mpl.rcParams["legend.fontsize"] - 0.3)
        for k in keys:
            _, col, ls, mk = ENT[k]; lab = elabel(k, width)
            pr, rc, _ = precision_recall_curve(D["y"], D["score"][k])
            ap = float(average_precision_score(D["y"], D["score"][k]))
            rx, py = thin(rc, pr)
            ref = (i == 1 and k == "mistral7b")
            a.plot(rx, py, color=col, ls=(0, (5, 2)) if ref else ls, lw=1.2 if ref else 1.5,
                   label=f"{lab} — AP {ap:.3f}", zorder=3)
            data.setdefault("detectors", {})[k] = {"average_precision": ap}
        if i == 1:
            yy = D["y"]; p = D["ds_pred"]
            tp = int(((p == 1) & (yy == 1)).sum()); fp = int(((p == 1) & (yy == 0)).sum())
            prec, rec = tp / (tp + fp), tp / int((yy == 1).sum())
            _, col, _, mk = ENT["datasentinel"]; lab = elabel("datasentinel", width)
            a.plot([rec], [prec], marker=mk, color=col, ms=6.5, ls="none", zorder=5,
                   label=f"{lab} — P {prec:.2f}")
            data["detectors"]["datasentinel"] = {"precision": prec, "recall": rec}
        a.set_xlim(0, 1); a.set_ylim(0, 1.02)
        a.set_xlabel("Recall"); a.set_ylabel("Precision")
        grid(a); plabel(a, title)
        a.legend(loc="lower left", markerscale=0.9)
    fig.tight_layout(pad=0.4)
    return fig, data


def fig_generalization_gap(D, width):
    sel = D["stage1_sel"]["rows"]
    ind_val, ood = {}, {}
    for k, name in (("llama", "llama3.2-1b"), ("qwen", "qwen2.5-1.5b"), ("granite", "granite-guardian-2b")):
        ind_val[k] = sel[name]["val"]["dr_at_1pct_fpr"]; ood[k] = sel[name]["eval"]["dr_at_1pct_fpr"]
    ind_val["mistral7b"] = dr_at(D["val"]["y"], D["val"]["mistral7b"]["score"])[0]
    ood["mistral7b"] = dr_at(D["y"], D["score"]["mistral7b"])[0]

    order = ["granite", "llama", "qwen", "mistral7b"]   # the four detectors in the paper
    short = {"promptguard2": "PromptGuard 2\n(86M)", "protectai": "ProtectAI\nDeBERTa-v3",
             "granite": "Granite\nGuardian-2B", "llama": "Llama-3.2\n-1B",
             "qwen": "Qwen-2.5\n-1.5B", "mistral7b": "M2 Mistral-7B\n(Stage 2)"}
    fig, ax = panels(width, 1, 3.0, 3.2)
    a = ax[0]
    x = np.arange(len(order)); w = 0.38
    v_in = np.array([ind_val[k] for k in order], float)
    v_oo = np.array([ood[k] for k in order], float)
    na = np.isnan(v_in)
    r1 = a.bar(x - w / 2, np.nan_to_num(v_in), w, color=SPLIT_COLOR["in_dist"], edgecolor=SURFACE,
               linewidth=0.6, label="In-distribution (validation split)")
    r2 = a.bar(x + w / 2, v_oo, w, color=SPLIT_COLOR["ood"], edgecolor=SURFACE, linewidth=0.6,
               label="Our OOD eval (25,747 rows)")
    barlabel(a, r1, na_mask=na); barlabel(a, r2)
    a.set_xticks(x); a.set_xticklabels([short[k] for k in order])
    a.set_ylim(0, 1.12); a.set_ylabel("Detection rate @ 1% FPR")
    a.set_xlabel("")
    grid(a, axis="y")
    a.legend(loc="upper center", ncol=1 if width == "single" else 2, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(pad=0.4)
    return fig, {"in_distribution": {k: ind_val[k] for k in order},
                 "ood_eval": {k: ood[k] for k in order}}


def fig_channel_dr(D, width):
    rows = {}
    for k in ("promptguard2", "protectai", "mistral7b"):
        rows[k] = recall_by_channel_at_1pct(D["y"], D["ch"], D["score"][k])
    for k in CASCADES:
        pred, pt = cascade_preds(D, k)
        rows[k] = channel_recall(D["y"], D["ch"], pred)
        rows[k]["_achieved_fpr"] = pt["e2e_fpr"]
    order = ["promptguard2", "protectai", "mistral7b", "cascade_llama", "cascade_qwen"]
    short = {"promptguard2": "PromptGuard 2", "protectai": "ProtectAI", "mistral7b": "M2 (Stage 2)",
             "cascade_llama": "Cascade\nLlama-1B", "cascade_qwen": "Cascade\nQwen-1.5B"}
    fig, ax = panels(width, 1, 3.0, 3.2)
    a = ax[0]
    x = np.arange(len(order)); w = 0.21
    for j, c in enumerate(("direct", "document", "tool")):
        v = np.array([rows[k][c] for k in order], float)
        r = a.bar(x + (j - 1.5) * w, v, w, color=CHANNEL_COLOR[c], edgecolor=SURFACE, linewidth=0.6,
                  label=f"{c} injection")
        barlabel(a, r, fmt="{:.2f}")
    ov = np.array([rows[k]["overall"] for k in order], float)
    r = a.bar(x + 1.5 * w, ov, w, color=INK_MUTED, edgecolor=SURFACE, linewidth=0.6,
              hatch="///", label="all channels (aggregate)")
    barlabel(a, r, fmt="{:.2f}")
    a.set_xticks(x); a.set_xticklabels([short[k] for k in order])
    a.set_ylim(0, 1.10); a.set_ylabel("Detection rate @ 1% FPR")
    grid(a, axis="y")
    a.legend(loc="upper center", ncol=2 if width == "single" else 4, bbox_to_anchor=(0.5, 1.17))
    fig.tight_layout(pad=0.4)
    return fig, {"per_channel_recall_at_1pct_fpr": rows,
                 "n_attacks_by_channel": {c: int(((D["y"] == 1) & (D["ch"] == c)).sum())
                                          for c in ("direct", "document", "tool")}}


def fig_channel_roc(D, width):
    fig, ax = panels(width, 2, 3.05, 2.85)
    a = ax[0]
    y, ch = D["y"], D["ch"]
    s = D["score"]["mistral7b"]
    data = {"per_channel_roc": {}}
    a.axvline(FPR_TARGET, color=INK_MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    a.text(FPR_TARGET * 1.2, 0.30, "1% FPR", color=INK_2, fontsize=mpl.rcParams["legend.fontsize"] - 0.3)
    for c, ls, mk in (("direct", "-", "o"), ("document", "--", "s"), ("tool", "-.", "^")):
        m = (y == 0) | ((y == 1) & (ch == c))
        fpr, tpr, _ = roc_curve(y[m], s[m])
        au = float(roc_auc_score(y[m], s[m])); dr, _, afpr = dr_at(y[m], s[m])
        keep = fpr > 0
        fx, ty = thin(fpr[keep], tpr[keep])
        a.plot(fx, ty, color=CHANNEL_COLOR[c], ls=ls, label=f"{c} — AUROC {au:.3f}", zorder=3)
        a.plot([afpr], [dr], marker=mk, color=CHANNEL_COLOR[c], ms=5.0, mec=SURFACE, mew=0.8, ls="none", zorder=4)
        data["per_channel_roc"][c] = {"auroc": au, "dr_at_1pct_fpr": float(dr)}
    a.set_xscale("log"); a.set_xlim(1e-4, 1.0); a.set_ylim(0, 1.02)
    a.set_xlabel("False positive rate (log)"); a.set_ylabel("Detection rate (TPR)")
    grid(a); plabel(a, "(a) M2 (Stage 2), by injection channel")
    a.legend(loc="upper left")

    # (b) contamination re-score: drop the document channel, keep every benign row
    b = ax[1]
    keys = ["mistral7b", "cascade_llama", "cascade_qwen"]
    short = {"mistral7b": "M2 (Stage 2)", "cascade_llama": "Cascade\nLlama-1B", "cascade_qwen": "Cascade\nQwen-1.5B"}
    keep_doc = ~((y == 1) & (ch == "document"))
    allc, nodoc = [], []
    for k in keys:
        if k == "mistral7b":
            allc.append(dr_at(y, s)[0]); nodoc.append(dr_at(y[keep_doc], s[keep_doc])[0])
        else:
            pred, _ = cascade_preds(D, k)
            allc.append(float(pred[y == 1].mean()))
            m = keep_doc & (y == 1)
            nodoc.append(float(pred[m].mean()))
    x = np.arange(len(keys)); w = 0.38
    r1 = b.bar(x - w / 2, allc, w, color=BLUE, edgecolor=SURFACE, linewidth=0.6, label="all channels")
    r2 = b.bar(x + w / 2, nodoc, w, color=ORANGE, edgecolor=SURFACE, linewidth=0.6, label="document channel removed")
    barlabel(b, r1); barlabel(b, r2)
    b.set_xticks(x); b.set_xticklabels([short[k] for k in keys])
    b.set_ylim(0, 0.85); b.set_ylabel("Detection rate @ 1% FPR")
    grid(b, axis="y"); plabel(b, "(b) Re-scored without the document channel")
    b.legend(loc="upper left")
    data["contamination_rescore"] = {k: {"all_channels": allc[i], "document_removed": nodoc[i]}
                                     for i, k in enumerate(keys)}
    fig.tight_layout(pad=0.4)
    return fig, data


def _frontier(D, key, step=0.005):
    s1_key, _ = CASCADES[key]
    s1_ps, s2, y = D["psafe"][s1_key], D["score"]["mistral7b"], D["y"]
    es, tprs, ths = [], [], []
    for th in np.arange(0.0, 1.0 + 1e-9, step):
        pt = _cascade_point(s1_ps, s2, y, float(th), FPR_TARGET)
        if not np.isfinite(pt["e2e_tpr"]):
            continue
        es.append(pt["escalation_rate"]); tprs.append(pt["e2e_tpr"]); ths.append(float(th))
    o = np.argsort(es)
    return np.asarray(es)[o], np.asarray(tprs)[o], np.asarray(ths)[o]


def fig_cost_frontier(D, width):
    fig, ax = panels(width, 2, 3.05, 2.9)
    a, b = ax[0], ax[1]
    m2_dr = dr_at(D["y"], D["score"]["mistral7b"])[0]
    data = {"stage2_only_dr_at_1pct_fpr": float(m2_dr), "cascades": {}}
    a.axhline(m2_dr, color=BLUE, lw=1.2, ls=(0, (5, 2)), zorder=2)
    a.text(0.02, m2_dr + 0.028, f"M2 on every input — DR {m2_dr:.3f}", color=BLUE,
           fontsize=mpl.rcParams["legend.fontsize"] - 0.3)
    for k in ("cascade_llama", "cascade_qwen"):
        _, col, ls, mk = ENT[k]; lab = elabel(k, width)
        e, tpr, _ = _frontier(D, k)
        a.plot(e, tpr, color=col, ls="-", lw=1.5, label=lab, zorder=3)
        summ = D["summaries"][k]["headline_cal_frozen"]["on_eval"]
        boot = D["summaries"][k]["headline_cal_frozen"]["bootstrap_eval"]["gap_cascade_minus_stage2"]
        a.plot([summ["escalation_rate"]], [summ["e2e_tpr"]], marker=mk, color=col, ms=6.5,
               mec=SURFACE, mew=0.9, ls="none", zorder=5)
        dx, dy = (-0.22, -0.075) if k == "cascade_qwen" else (0.09, -0.13)
        a.annotate(f"$e$ = {summ['escalation_rate']:.2f}", xy=(summ["escalation_rate"], summ["e2e_tpr"]),
                   xytext=(summ["escalation_rate"] + dx, summ["e2e_tpr"] + dy), color=col,
                   fontsize=mpl.rcParams["legend.fontsize"] - 0.3,
                   arrowprops=dict(arrowstyle="-", lw=0.6, color=col))
        data["cascades"][k] = {"operating_escalation": summ["escalation_rate"],
                               "operating_e2e_dr": summ["e2e_tpr"],
                               "gap_ci": [boot["lo"], boot["hi"]]}
    a.set_xlim(0, 1); a.set_ylim(0, 0.49)
    a.set_xlabel("Deferral rate $e$ (fraction sent to Stage 2)")
    a.set_ylabel("End-to-end detection @ 1% FPR")
    grid(a); plabel(a, "(a) Accuracy–cost frontier (frozen point marked)")
    a.legend(loc="lower right")

    k2 = D["latency"]["cascade_k1k2"]["llama3.2-1b"]["k2_ms"]
    e_grid = np.linspace(0, 1, 201)
    b.axhline(0.0, color=INK_MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    for k, mname in (("cascade_llama", "llama3.2-1b"), ("cascade_qwen", "qwen2.5-1.5b")):
        _, col, _, mk = ENT[k]; lab = elabel(k, width)
        blk = D["latency"]["cascade_k1k2"][mname]
        k1 = blk["k1_ms"]
        red = 1.0 - (k1 + e_grid * k2) / k2
        b.plot(e_grid, red * 100, color=col, ls="-", lw=1.5, label=lab, zorder=3)
        b.plot([blk["e"]], [blk["reduction_at_e"] * 100], marker=mk, color=col, ms=6.5,
               mec=SURFACE, mew=0.9, ls="none", zorder=5)
        tx, ty = (0.03, 10.0) if mname.startswith("qwen") else (0.63, 52.0)
        b.annotate(f"+{blk['reduction_at_e'] * 100:.1f}% at $e$ = {blk['e']:.2f}",
                   xy=(blk["e"], blk["reduction_at_e"] * 100), xytext=(tx, ty), color=col,
                   fontsize=mpl.rcParams["legend.fontsize"] - 0.3,
                   arrowprops=dict(arrowstyle="-", lw=0.6, color=col))
        b.plot([blk["breakeven_e"]], [0.0], marker="|", ms=9, mew=1.2, color=col, ls="none", zorder=5)
        b.annotate(f"break-even $e^*$ = {blk['breakeven_e']:.2f}", xy=(blk["breakeven_e"], 0.0),
                   xytext=(0.05, -19.0 if mname.startswith("qwen") else -31.0), color=col, ha="left",
                   va="center", fontsize=mpl.rcParams["legend.fontsize"] - 0.3,
                   arrowprops=dict(arrowstyle="-", lw=0.6, color=col))
        bf = D["latency_bf16"]["cascade_k1k2"][mname] if mname in D["latency_bf16"].get("cascade_k1k2", {}) else None
        data["cascades"].setdefault(k, {}).update({
            "k1_ms": k1, "k2_ms": k2, "breakeven_e": blk["breakeven_e"],
            "reduction_at_operating_e_pct": blk["reduction_at_e"] * 100,
            "usd_per_1m_cascade": blk["cascade_cost"]["usd_per_1m"],
            "usd_per_1m_stage2_only": blk["guard_cost"]["usd_per_1m"],
            "reduction_bf16_stage1_pct": (bf["reduction_at_e"] * 100 if bf else None)})
    b.set_xlim(0, 1); b.set_ylim(-40, 100)
    b.set_xlabel("Deferral rate $e$")
    b.set_ylabel("Inference-cost reduction (%)")
    grid(b); plabel(b, "(b) Measured cost reduction (A10G, batch = 1, 512 tok)")
    b.legend(loc="upper right")
    fig.tight_layout(pad=0.4)
    return fig, data


def _reliability(ax, y, p, color, ls, marker, nbins=10):
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, nbins - 1)
    xs, ys = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 5:
            continue
        xs.append(float(p[m].mean())); ys.append(float(y[m].mean()))
    e = float(ece(y, p))
    ax.plot(xs, ys, color=color, ls=ls, marker=marker, ms=4.0, mec=SURFACE, mew=0.6, zorder=3)
    return e


def fig_calibration(D, width):
    fig, ax = panels(width, 3, 2.55, 2.5)
    data = {}
    for i, k in enumerate(("mistral7b", "qwen")):
        a = ax[i]
        a.plot([0, 1], [0, 1], color=INK_MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        a.text(0.58, 0.50, "perfect", color=INK_MUTED, rotation=38,
               fontsize=mpl.rcParams["legend.fontsize"] - 0.5)
        e_cal = _reliability(a, D["cal"]["y"], D["cal"][k]["score"], BLUE, "-", "o")
        e_ev = _reliability(a, D["y"], D["score"][k], ORANGE, "--", "s")
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.set_xlabel("Predicted P(injection)"); a.set_ylabel("Observed attack fraction")
        grid(a); plabel(a, f"({'ab'[i]}) {SHORT[k]} reliability")
        ex, ey, eha, eva = (0.97, 0.04, "right", "bottom") if i == 0 else (0.03, 0.97, "left", "top")
        a.text(ex, ey, f"ECE {e_cal:.3f} \u2192 {e_ev:.3f}", transform=a.transAxes, ha=eha,
               va=eva, color=INK_2, fontsize=mpl.rcParams["legend.fontsize"])
        data[k] = {"ece_cal": e_cal, "ece_eval": e_ev}

    c = ax[2]
    y_c, s_c = D["cal"]["y"], D["cal"]["mistral7b"]["score"]
    y_e, s_e = D["y"], D["score"]["mistral7b"]
    _, thr, _ = dr_at(y_c, s_c)
    for yy, ss, col, ls in ((y_c, s_c, BLUE, "-"), (y_e, s_e, ORANGE, "--")):
        b = np.sort(ss[yy == 0])
        c.plot(np.maximum(b, 1e-8), 1.0 - np.arange(len(b)) / len(b), color=col, ls=ls, zorder=3)
    fpr_eval = float((s_e[y_e == 0] >= thr).mean())
    c.axvline(thr, color=INK_2, lw=0.8, ls=(0, (1, 1.5)), zorder=2)
    c.axhline(FPR_TARGET, color=INK_MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    c.plot([thr], [FPR_TARGET], marker="o", ms=4.5, color=BLUE, mec=SURFACE, mew=0.8, ls="none", zorder=5)
    c.plot([thr], [fpr_eval], marker="s", ms=4.5, color=ORANGE, mec=SURFACE, mew=0.8, ls="none", zorder=5)
    c.annotate(f"same threshold,\n{fpr_eval:.1%} FPR on eval", xy=(thr, fpr_eval), xytext=(3e-5, 0.28),
               color=ORANGE, fontsize=mpl.rcParams["legend.fontsize"] - 0.3, ha="left",
               arrowprops=dict(arrowstyle="-", lw=0.6, color=ORANGE))
    c.text(3e-1, FPR_TARGET * 1.4, "1% FPR target", color=INK_2, ha="right",
           fontsize=mpl.rcParams["legend.fontsize"] - 0.3)
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlim(1e-7, 1.0); c.set_ylim(1e-4, 1.3)
    c.set_xlabel("M2 injection score (log)"); c.set_ylabel("Fraction of benign above threshold")
    grid(c); plabel(c, "(c) Operating point does not transfer")
    data["threshold_transfer"] = {"cal_threshold_at_1pct_fpr": float(thr), "eval_fpr_at_that_threshold": fpr_eval}
    fig.tight_layout(pad=0.4)
    handles = [mpl.lines.Line2D([], [], color=BLUE, ls="-", marker="o", ms=4.0),
               mpl.lines.Line2D([], [], color=ORANGE, ls="--", marker="s", ms=4.0)]
    fig.legend(handles, ["calibration split (in-distribution)", "OOD eval"],
               loc="lower center" if width == "single" else "upper center",
               bbox_to_anchor=(0.5, -0.02) if width == "single" else (0.5, 1.10),
               ncol=2, columnspacing=1.6)
    return fig, data


def fig_datasentinel_length(D, width):
    y, pred, tl = D["y"], D["ds_pred"], D["tlen"]
    ben = y == 0
    q = np.quantile(tl[ben], [0.25, 0.5, 0.75])
    binidx = np.digitize(tl, q)
    labels = [f"Q1\n(<{int(q[0])})", f"Q2\n({int(q[0])}–{int(q[1])})",
              f"Q3\n({int(q[1])}–{int(q[2])})", f"Q4\n(>{int(q[2])})"]
    fprs, ns = [], []
    for b in range(4):
        m = ben & (binidx == b)
        fprs.append(float(pred[m].mean())); ns.append(int(m.sum()))
    fig, ax = panels(width, 1, 2.9, 3.0)
    a = ax[0]
    r = a.bar(np.arange(4), fprs, 0.58, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    barlabel(a, r)
    overall = float(pred[ben].mean())
    a.axhline(overall, color=ORANGE, lw=1.2, ls=(0, (5, 2)), zorder=4)
    a.text(3.42, overall + 0.02, f"overall {overall:.2f}", color=ORANGE, ha="right",
           fontsize=mpl.rcParams["legend.fontsize"] - 0.3)
    a.axhline(FPR_TARGET, color=INK_MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    a.text(-0.42, FPR_TARGET + 0.02, "1% FPR budget", color=INK_2, ha="left",
           fontsize=mpl.rcParams["legend.fontsize"] - 0.3)
    a.set_xticks(np.arange(4)); a.set_xticklabels(labels)
    a.set_xlabel("Benign input length quartile (characters)")
    a.set_ylabel("DataSentinel-7B false positive rate")
    a.set_ylim(0, 1.05)
    grid(a, axis="y")
    fig.tight_layout(pad=0.4)
    return fig, {"fpr_by_benign_length_quartile": dict(zip(["q1", "q2", "q3", "q4"], fprs)),
                 "n_by_quartile": ns, "overall_benign_fpr": overall,
                 "quartile_edges_chars": [int(v) for v in q]}


FIGURES = {
    "roc": fig_roc,
    "roc_lowfpr": fig_roc_lowfpr,
    "pr": fig_pr,
    "generalization_gap": fig_generalization_gap,
    "channel_dr": fig_channel_dr,
    "channel_roc": fig_channel_roc,
    "cost_frontier": fig_cost_frontier,
    "calibration": fig_calibration,
    "datasentinel_length": fig_datasentinel_length,
}


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate paper figures from frozen score files")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--width", default="both", choices=["both", "double", "single"])
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--only", default="", help="comma-separated figure names")
    ap.add_argument("--latency", default="results/analysis/latency_curve_full_nf4.json",
                   help="NF4 latency measurement backing the cost panel")
    ap.add_argument("--latency-bf16", default="results/analysis/latency_curve_full_bf16.json",
                   help="bf16 Stage-1 latency measurement (Stage 2 is NF4 in both)")
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(FIGURES)
    bad = [n for n in names if n not in FIGURES]
    if bad:
        raise SystemExit(f"unknown figure(s) {bad}; available: {list(FIGURES)}")
    args.out = args.out if args.out.is_absolute() else (ROOT / args.out)
    widths = ["double", "single"] if args.width == "both" else [args.width]
    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]

    print("loading scores ...")
    D = load(args.latency, args.latency_bf16)
    print(f"  eval n={len(D['y'])}  attacks={int((D['y'] == 1).sum())}  benign={int((D['y'] == 0).sum())}")
    args.out.mkdir(parents=True, exist_ok=True)

    meta = {}
    for name in names:
        for w in widths:
            style(w)
            fig, data = FIGURES[name](D, w)
            for fmt in fmts:
                p = args.out / f"{name}_{w}.{fmt}"
                fig.savefig(p, dpi=args.dpi if fmt == "png" else None)
                print(f"  wrote {_rel(p)}")
            plt.close(fig)
            if w == widths[0]:
                meta[name] = data

    meta["_provenance"] = {
        "eval_split": EVAL_SPLIT, "cal_split": CAL_SPLIT, "val_split": VAL_SPLIT,
        "stage1_dirs": STAGE1_DIRS, "stage2_dir": STAGE2_DIR,
        "fpr_target": FPR_TARGET,
        "n": int(len(D["y"])), "n_attacks": int((D["y"] == 1).sum()), "n_benign": int((D["y"] == 0).sum()),
        "cascade_summaries": {k: p for k, (_, p) in CASCADES.items()},
        "metric_code": "src.evaluation.metrics.detection_rate_at_fpr / scripts.eval.eval_cascade._cascade_point",
    }
    (args.out / "figure_data.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"  wrote {_rel(args.out / 'figure_data.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
