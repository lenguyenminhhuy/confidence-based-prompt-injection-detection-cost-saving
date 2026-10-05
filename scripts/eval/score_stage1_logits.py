"""Score a Stage-1 logit dump against its split — NO GPU, payload-safe.

Joins a ``*_logits.jsonl`` file (per-row {logp_benign, logp_injection, p_safe},
as written by train_stage1.py) with its source split by ROW ORDER, then reports
the E2 selection metrics using the repo's own src/evaluation/metrics.py:

  - AUROC (rank-based, numpy only)
  - accuracy / precision / recall / F1 / FPR at the p_safe=0.5 boundary
  - detection rate (TPR) at fixed FPR (0.1% / 0.5% / 1%) — the security metric
  - ECE (calibration)
  - per-source and per-channel AUROC + accuracy (leakage / trivial-separability check)

PAYLOAD HYGIENE: reads only label/source/channel from the split —
never the 'input'/'text' field. Prints only aggregates: counts and metrics.

    PYTHONPATH=. python scripts/eval/score_stage1_logits.py \
        --logits results/stage1/qwen2.5-1.5b/val_logits.jsonl \
        --split  data/train_proposal/val.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# metrics.py does `from evaluation.per_channel import ...`, so src/ must be on path
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.evaluation.metrics import binary_metrics, detection_rate_at_fpr, ece  # noqa: E402

_SAFE = {"safe", "benign", 0, "0"}
_UNSAFE = {"unsafe", "injection", "injected", 1, "1"}  # test splits use 'injected'


def _label_to_int(raw) -> int:
    if raw in _SAFE:
        return 0
    if raw in _UNSAFE:
        return 1
    raise ValueError(f"unrecognized label {raw!r}")


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), tie-aware, numpy only."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks within tied score groups
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_rows(logits_path: Path, split_path: Path):
    """Return (labels, scores, sources, channels) joined by row order."""
    with logits_path.open() as f:
        logit_rows = [json.loads(l) for l in f if l.strip()]
    labels, sources, channels = [], [], []
    with split_path.open() as f:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)  # only pull label/source/channel — never r['input']
            labels.append(_label_to_int(r.get("label")))
            sources.append(r.get("source"))
            channels.append(r.get("channel"))
    if len(logit_rows) != len(labels):
        raise SystemExit(
            f"ROW COUNT MISMATCH: {logits_path.name}={len(logit_rows)} vs "
            f"{split_path.name}={len(labels)} — cannot join by index."
        )
    p_safe = np.array([r["p_safe"] for r in logit_rows], dtype=float)
    scores = 1.0 - p_safe  # higher = more injection
    n_bad = int(np.isnan(scores).sum() + np.isinf(scores).sum())
    if n_bad:
        print(f"WARNING: {n_bad} NaN/Inf scores present")
    return np.array(labels, dtype=int), scores, sources, channels


def _fmt(x) -> str:
    return "nan" if x != x else f"{x:.4f}"


def _slice_breakdown(labels, scores, keys) -> dict:
    out = {}
    for key in sorted(set(keys), key=lambda x: (x is None, str(x))):
        m = np.array([k == key for k in keys])
        yl, sl = labels[m], scores[m]
        n_pos, n_neg = int((yl == 1).sum()), int((yl == 0).sum())
        out[str(key)] = {
            "auroc": auroc(yl, sl) if (n_pos and n_neg) else float("nan"),
            "accuracy": float(((sl > 0.5).astype(int) == yl).mean()),
            "n": int(m.sum()), "pos": n_pos, "neg": n_neg,
        }
    return out


def compute_metrics(labels, scores, sources=None, channels=None) -> dict:
    """All E2 selection metrics as a dict — no printing. Reused by the ranker."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores > 0.5).astype(int)  # p_safe < 0.5 -> injection
    m = {
        "n": int(len(labels)),
        "n_benign": int((labels == 0).sum()),
        "n_injection": int((labels == 1).sum()),
        "auroc": auroc(labels, scores),
        **binary_metrics(labels, preds),
        "ece": ece(labels, scores),
        "detection_rate_at_fpr": detection_rate_at_fpr(labels, scores),
    }
    if sources is not None:
        m["per_source"] = _slice_breakdown(labels, scores, sources)
    if channels is not None:
        m["per_channel"] = _slice_breakdown(labels, scores, channels)
    return m


def report(labels, scores, sources, channels) -> dict:
    m = compute_metrics(labels, scores, sources, channels)
    print(f"\nn={m['n']}  benign={m['n_benign']}  injection={m['n_injection']}")
    print(f"AUROC              {_fmt(m['auroc'])}")
    print(f"accuracy@0.5       {_fmt(m['accuracy'])}")
    print(f"precision/recall   {_fmt(m['precision'])} / {_fmt(m['recall'])}   "
          f"F1 {_fmt(m['f1'])}")
    print(f"FPR@0.5            {_fmt(m['fpr'])}")
    print(f"ECE                {_fmt(m['ece'])}")
    print("detection-rate @ fixed FPR (threshold set on benign):")
    for k, v in m["detection_rate_at_fpr"].items():
        flag = "" if v["resolvable"] else "  [under-powered: too few benign]"
        print(f"  FPR={float(k)*100:>4.1f}%   DR(TPR)={_fmt(v['dr'])}   "
              f"achieved_fpr={_fmt(v['achieved_fpr'])}{flag}")
    for name in ("source", "channel"):
        print(f"\nper-{name} (AUROC / acc / n):")
        for key, s in m[f"per_{name}"].items():
            print(f"  {key:<24} {_fmt(s['auroc']):>7} / {_fmt(s['accuracy']):>7} / "
                  f"{s['n']:>5}   (pos={s['pos']}, neg={s['neg']})")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a Stage-1 logit dump (no GPU)")
    ap.add_argument("--logits", type=Path, required=True,
                    help="*_logits.jsonl from train_stage1.py")
    ap.add_argument("--split", type=Path, required=True,
                    help="matching split jsonl (val/cal) — labels joined by row order")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional: write metrics JSON here")
    args = ap.parse_args()

    labels, scores, sources, channels = load_rows(args.logits, args.split)
    print(f"=== {args.logits.name}  vs  {args.split.name} ===")
    metrics = report(labels, scores, sources, channels)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote metrics -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
