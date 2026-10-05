"""Temperature-scaling calibration of Stage-1 (no GPU, payload-safe).

Fits a scalar temperature ``T`` on the calibration split by NLL, reports ECE + NLL
before/after on cal AND val, writes reliability diagrams, freezes ``T``, and emits
three schema-identical logit dirs for each test split:

  raw/       original scores
  calfit_T/  cal-fit T applied to the test split   (the realistic lever)
  oracle_T/  T re-fit per test split on itself      (monotone-map ceiling)

Temperature scaling is monotone, so it changes ECE but not the routing frontier.
The routing threshold theta_safe is selected separately by scripts/eval/eval_cascade.py.

PAYLOAD HYGIENE: reads only label + logit fields; never text. Emits
only counts / metrics / logits.

    PYTHONPATH=. python scripts/train/calibrate.py \
        --logits-dir results/stage1/llama3.2-1b \
        --out-dir results/analysis/calibration/llama3.2-1b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.calibration.temperature_scaling import (  # noqa: E402
    IsotonicScaler,
    TemperatureScaler,
    _nll_injection,
    fit_temperature,
    gap_from_logps,
    nll_curvature,
)
from src.evaluation.metrics import ece  # noqa: E402
from src.utils.io import read_jsonl, write_jsonl  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from scripts.eval.score_stage1_logits import _label_to_int, auroc  # noqa: E402

log = get_logger("calibrate")

DEFAULT_TEST_SPLITS = [ROOT / "data/eval_proposal/eval.jsonl"]


def load_split(logits_path: Path, split_path: Path):
    """Return (logp_benign, logp_injection, p_safe, labels) joined by row order."""
    rows = [r for r in read_jsonl(logits_path)]
    labels = [_label_to_int(r.get("label")) for r in read_jsonl(split_path)]
    if len(rows) != len(labels):
        raise SystemExit(
            f"ROW MISMATCH: {logits_path.name}={len(rows)} vs "
            f"{split_path.name}={len(labels)} — cannot join by index."
        )
    lb = np.array([r["logp_benign"] for r in rows], dtype=float)
    li = np.array([r["logp_injection"] for r in rows], dtype=float)
    ps = np.array([r["p_safe"] for r in rows], dtype=float)
    return lb, li, ps, np.array(labels, dtype=int)


def _ece_nll(labels, lb, li, T):
    """ECE (confidence = p_injection = 1 - p_safe) and mean injection NLL at temp T."""
    z = gap_from_logps(lb, li)
    p_safe = 1.0 / (1.0 + np.exp(-z / T))
    return {
        "ece": ece(labels, 1.0 - p_safe),
        "nll": _nll_injection(T, z, labels),
        "auroc": auroc(labels, 1.0 - p_safe),  # rank-only; invariant to T (sanity)
    }


def _emit(out_dir: Path, split: str, lb, li, p_safe) -> None:
    """Write <split>_logits.jsonl with the frozen {logp_benign, logp_injection, p_safe}."""
    recs = ({"logp_benign": float(a), "logp_injection": float(b), "p_safe": float(c)}
            for a, b, c in zip(lb, li, p_safe))
    write_jsonl(out_dir / f"{split}_logits.jsonl", recs)


def _reliability_png(labels, p_safe, path: Path, title: str, n_bins: int = 10) -> bool:
    """10-bin reliability diagram on the injection-confidence axis (mirrors ece())."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log.warning("matplotlib unavailable (%s) — skipping %s", e, path.name)
        return False
    y = np.asarray(labels, dtype=int)
    conf = 1.0 - np.asarray(p_safe, dtype=float)  # p_injection
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        xs.append(float(conf[m].mean())); ys.append(float(y[m].mean())); ws.append(int(m.sum()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect")
    ax.scatter(xs, ys, s=[max(12, w / 6) for w in ws], color="#3f6bd6", alpha=0.85, zorder=3)
    ax.plot(xs, ys, color="#3f6bd6", lw=1.2, zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted P(injection)"); ax.set_ylabel("empirical P(injection)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temperature-scaling calibration (no GPU)")
    p.add_argument("--logits-dir", type=Path, required=True,
                   help="dir with cal_logits.jsonl, val_logits.jsonl, <split>_logits.jsonl")
    p.add_argument("--cal-split", type=Path, default=ROOT / "data/train_proposal/cal.jsonl")
    p.add_argument("--val-split", type=Path, default=ROOT / "data/train_proposal/val.jsonl")
    p.add_argument("--test-splits", nargs="+", type=Path, default=DEFAULT_TEST_SPLITS,
                   help="split jsonl(s); logits are read from <logits-dir>/<stem>_logits.jsonl")
    p.add_argument("--method", choices=["temperature", "isotonic", "both"], default="temperature")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/analysis/calibration")
    p.add_argument("--figures-dir", type=Path, default=ROOT / "results/figures")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- fit T on cal ---------------------------------------------------------
    cal_lb, cal_li, cal_ps, cal_y = load_split(
        args.logits_dir / "cal_logits.jsonl", args.cal_split)
    scaler = TemperatureScaler().fit(cal_lb, cal_li, cal_y)
    z_cal = gap_from_logps(cal_lb, cal_li)
    curv = nll_curvature(z_cal, cal_y, scaler.T)
    log.info("fit T=%.4f on cal (n=%d, NLL=%.5f, curvature=%.3e)",
             scaler.T, scaler.n_fit, scaler.fit_nll, curv)
    if abs(scaler.T - 1.0) < 0.05 or curv < 1e-3:
        log.warning("T~1 and/or flat NLL surface: cal is near-saturated — expect POOR "
                    "transfer of this T to shifted splits (report this).")
    scaler.to_json(args.out_dir / "temperature.json")

    # --- E3 acceptance: ECE + NLL before/after on cal AND val -----------------
    val_lb, val_li, val_ps, val_y = load_split(
        args.logits_dir / "val_logits.jsonl", args.val_split)
    accept = {
        "cal": {"raw": _ece_nll(cal_y, cal_lb, cal_li, 1.0),
                "tempered": _ece_nll(cal_y, cal_lb, cal_li, scaler.T)},
        "val": {"raw": _ece_nll(val_y, val_lb, val_li, 1.0),
                "tempered": _ece_nll(val_y, val_lb, val_li, scaler.T)},
    }
    if args.method in ("isotonic", "both"):
        iso = IsotonicScaler().fit(cal_ps, cal_y)
        accept["cal"]["isotonic_ece"] = ece(cal_y, 1.0 - iso.transform(cal_ps))
        accept["val"]["isotonic_ece"] = ece(val_y, 1.0 - iso.transform(val_ps))

    # --- reliability diagrams -------------------------------------------------
    if not args.no_plots:
        for split, (lb, li, y) in {"cal": (cal_lb, cal_li, cal_y),
                                   "val": (val_lb, val_li, val_y)}.items():
            z = gap_from_logps(lb, li)
            _reliability_png(y, 1 / (1 + np.exp(-z)),
                             args.figures_dir / f"reliability_{split}_raw.png",
                             f"{split} — raw (ECE={accept[split]['raw']['ece']:.3f})")
            _reliability_png(y, 1 / (1 + np.exp(-z / scaler.T)),
                             args.figures_dir / f"reliability_{split}_tempered.png",
                             f"{split} — T={scaler.T:.2f} (ECE={accept[split]['tempered']['ece']:.3f})")

    # --- emit raw / calfit_T / oracle_T logit dirs for the test splits --------
    per_split = {}
    for split_path in args.test_splits:
        split = split_path.stem
        lb, li, ps, y = load_split(args.logits_dir / f"{split}_logits.jsonl", split_path)
        # raw
        _emit(args.out_dir / "raw", split, lb, li, ps)
        # calfit: cal-fit T (realistic)
        c_lb, c_li = scaler.calibrated_logps(lb, li)
        _emit(args.out_dir / "calfit_T", split, c_lb, c_li, scaler.transform(lb, li))
        # oracle: T re-fit on THIS split (ceiling of any monotone map, fixed grid)
        o = TemperatureScaler().fit(lb, li, y)
        o_lb, o_li = o.calibrated_logps(lb, li)
        _emit(args.out_dir / "oracle_T", split, o_lb, o_li, o.transform(lb, li))
        per_split[split] = {
            "n": int(len(y)), "n_attacks": int((y == 1).sum()), "n_benign": int((y == 0).sum()),
            "auroc": auroc(y, 1.0 - ps),
            "ece_raw": ece(y, 1.0 - ps),
            "ece_calfit": ece(y, 1.0 - scaler.transform(lb, li)),
            "ece_oracle": ece(y, 1.0 - o.transform(lb, li)),
            "T_oracle": o.T,
        }

    summary = {
        "T_cal": scaler.T, "cal_nll": scaler.fit_nll, "cal_nll_curvature": curv,
        "n_cal": scaler.n_fit, "acceptance": accept, "per_test_split": per_split,
        "note": ("Monotone calibration is rank-preserving: on a wide+fine routing grid the "
                 "cost frontier is invariant (raw≈oracle). calfit_T = realistic (T fit on "
                 "in-dist cal); oracle_T = per-split ceiling."),
    }
    (args.out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2))

    # --- console table --------------------------------------------------------
    print(f"\nfrozen T (cal) = {scaler.T:.4f}   [curvature {curv:.2e}]")
    print(f"{'split':<20} {'ECE raw':>9} {'ECE temp':>9} {'NLL raw':>9} {'NLL temp':>9}")
    print("-" * 60)
    for s in ("cal", "val"):
        r, t = accept[s]["raw"], accept[s]["tempered"]
        print(f"{s:<20} {r['ece']:>9.4f} {t['ece']:>9.4f} {r['nll']:>9.4f} {t['nll']:>9.4f}")
    print(f"\n{'test split':<20} {'ECE raw':>9} {'ECE calfit':>11} {'ECE oracle':>11} {'T_oracle':>9}")
    print("-" * 64)
    for s, m in per_split.items():
        print(f"{s:<20} {m['ece_raw']:>9.4f} {m['ece_calfit']:>11.4f} "
              f"{m['ece_oracle']:>11.4f} {m['T_oracle']:>9.3f}")
    print(f"\nwrote calibrated dirs + summary -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
