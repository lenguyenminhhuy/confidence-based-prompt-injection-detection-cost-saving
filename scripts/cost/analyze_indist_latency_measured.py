"""Compose measured cascade cost on the validation split from per-row timings.

Inputs: results/analysis/indist_latency/val_<model>_nf4.jsonl written by
scripts/cost/measure_latency_indist.py on the A10G (one file per model: the Stage-2
M2 and each NF4 Stage-1 candidate), plus the stored NF4 Stage-1 val logits used
for Table tab:indist so the routing decisions are exactly the reported ones.

For each Stage-1 candidate at the frozen theta_safe:
  * measured   : baseline = sum k2_meas ; cascade = sum k1_meas + esc * k2_meas
  * priced     : same with k1/k2 read off the synthetic-filler latency curve at
                 each row's rendered length (what tab:indist currently reports)
  * both at the split's own prevalence and re-weighted to the OOD benign share
  * per-request latency distribution (median / p95 / p99) for M2-alone and cascade
  * interpolation error: (priced - measured) / measured per row
  * pipeline identity check: measured p_safe vs stored p_safe; routing agreement

No GPU, no text. Run from the repository root:
    PYTHONPATH=. python scripts/cost/analyze_indist_latency_measured.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.eval.eval_cascade import _read_logits                                       # noqa: E402
from scripts.eval.eval_cascade_indist import (STAGE1, STAGE2, STAGE2_DIR, THETA_FROZEN,   # noqa: E402
                                         OOD_BENIGN_SHARE, _lat, _weights)

IN_DIR = "results/analysis/indist_latency"


def _load_meas(path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    rows.sort(key=lambda r: r["idx"])
    idx = np.array([r["idx"] for r in rows], int)
    assert len(np.unique(idx)) == len(idx), f"duplicate idx in {path}"
    return {"idx": idx,
            "label": np.array([r["label"] for r in rows], int),
            "ntok": np.array([r["ntok"] for r in rows], float),
            "ms": np.array([r["ms_median"] for r in rows], float),
            "p_safe": np.array([r["p_safe"] for r in rows], float)}


def _wq(v: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(v); cw = np.cumsum(w[o]); cw /= cw[-1]
    return float(v[o][np.searchsorted(cw, q)])


def _dist(v: np.ndarray, w: np.ndarray) -> dict:
    return {"mean_ms": float((w * v).sum() / w.sum()), "median_ms": _wq(v, w, 0.5),
            "p95_ms": _wq(v, w, 0.95), "p99_ms": _wq(v, w, 0.99)}


def _cost(k1, k2, esc, w) -> dict:
    base = float((w * k2).sum()); casc = float((w * (k1 + esc * k2)).sum())
    return {"baseline_ms_per_req": base, "cascade_ms_per_req": casc,
            "reduction": 1 - casc / base, "r_effective": float((w * k1).sum() / base)}


def _relerr(priced: np.ndarray, meas: np.ndarray) -> dict:
    e = (priced - meas) / meas
    return {"mean": float(e.mean()), "median": float(np.median(e)),
            "p95_abs": float(np.percentile(np.abs(e), 95)), "max_abs": float(np.abs(e).max()),
            "total_bias": float(priced.sum() / meas.sum() - 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=IN_DIR)
    ap.add_argument("--theta", type=float, default=THETA_FROZEN)
    ap.add_argument("--out", default=f"{IN_DIR}/measured_cost")
    a = ap.parse_args()
    d = ROOT / a.in_dir

    m2 = _load_meas(d / f"val_{STAGE2}_nf4.jsonl")
    n = len(m2["idx"]); y = m2["label"]
    _, _, m2_ps = _read_logits(ROOT / STAGE2_DIR / "val_logits.jsonl")
    m2_ps = m2_ps[m2["idx"]]
    res = {"split": "val", "n_timed": n, "n_benign": int((y == 0).sum()), "n_attack": int((y == 1).sum()),
           "theta_safe": a.theta, "ood_benign_share": OOD_BENIGN_SHARE,
           "m2": {"identity_max_abs_dp": float(np.abs(m2["p_safe"] - m2_ps).max()),
                  "measured": {}, "stage1": {}}}
    w_val = np.full(n, 1 / n); w_ood = _weights(y, OOD_BENIGN_SHARE)
    res["m2"]["measured"] = {"val_prevalence": _dist(m2["ms"], w_val), "ood_share": _dist(m2["ms"], w_ood),
                             "total_sec": float(m2["ms"].sum() / 1000),
                             "by_label": {"benign": _dist(m2["ms"][y == 0], np.ones((y == 0).sum())),
                                          "attack": _dist(m2["ms"][y == 1], np.ones((y == 1).sum()))}}

    lines = ["| | " + " | ".join(f"**{k} → M2**" for k in STAGE1) + " |", "|---|" + "---|" * len(STAGE1)]
    table: dict[str, dict[str, str]] = {}

    def put(row: str, name: str, val: str) -> None:
        table.setdefault(row, {})[name] = val

    for name, spec in STAGE1.items():
        f = d / f"val_{name}_nf4.jsonl"
        if not f.exists():
            print(f"[{name}] no measured file yet ({f.name}); skipped"); continue
        s1 = _load_meas(f)
        common = np.intersect1d(s1["idx"], m2["idx"])
        sel1 = np.isin(s1["idx"], common); sel2 = np.isin(m2["idx"], common)
        k1m, k2m, yy = s1["ms"][sel1], m2["ms"][sel2], m2["label"][sel2]
        assert (s1["label"][sel1] == yy).all()
        nn = len(common)
        _, _, ps_stored = _read_logits(ROOT / spec["dir"] / "val_logits.jsonl")
        ps_stored = ps_stored[common]
        esc = (ps_stored < a.theta).astype(float)
        esc_meas = (s1["p_safe"][sel1] < a.theta).astype(float)

        lat = json.load(open(ROOT / spec["lat"]))["per_model"]
        k1p = _lat(lat, name, s1["ntok"][sel1]); k2p = _lat(lat, STAGE2, m2["ntok"][sel2])
        wv = np.full(nn, 1 / nn); wo = _weights(yy, OOD_BENIGN_SHARE)
        casc_ms = k1m + esc * k2m

        row = {
            "n": nn, "escalation_rate": float(esc.mean()),
            "identity": {"max_abs_dp_safe": float(np.abs(s1["p_safe"][sel1] - ps_stored).max()),
                         "routing_disagreements": int((esc != esc_meas).sum())},
            "measured": {"val_prevalence": _cost(k1m, k2m, esc, wv), "ood_share": _cost(k1m, k2m, esc, wo),
                         "stage1_total_sec": float(k1m.sum() / 1000),
                         "cascade_total_sec": float(casc_ms.sum() / 1000), "baseline_total_sec": float(k2m.sum() / 1000)},
            "priced": {"val_prevalence": _cost(k1p, k2p, esc, wv), "ood_share": _cost(k1p, k2p, esc, wo)},
            "interp_error": {"k1": _relerr(k1p, k1m), "k2": _relerr(k2p, k2m)},
            "per_request": {"val_prevalence": {"m2_alone": _dist(k2m, wv), "cascade": _dist(casc_ms, wv)},
                            "ood_share": {"m2_alone": _dist(k2m, wo), "cascade": _dist(casc_ms, wo)}},
            "stage1_alone": {"val_prevalence": _dist(k1m, wv)},
        }
        res["m2"]["stage1"][name] = row
        mv, mo = row["measured"]["val_prevalence"], row["measured"]["ood_share"]
        pv, po = row["priced"]["val_prevalence"], row["priced"]["ood_share"]
        put("Measured reduction, split prevalence", name, f"{mv['reduction']:+.1%}")
        put("Priced reduction, split prevalence (tab:indist)", name, f"{pv['reduction']:+.1%}")
        put("Measured reduction, 72.4% benign", name, f"{mo['reduction']:+.1%}")
        put("Priced reduction, 72.4% benign (tab:indist)", name, f"{po['reduction']:+.1%}")
        put("Measured ms/request, split prevalence (M2 / cascade)", name,
            f"{mv['baseline_ms_per_req']:.0f} / {mv['cascade_ms_per_req']:.0f}")
        pr = row["per_request"]["val_prevalence"]
        put("Per-request median ms (M2 / cascade)", name, f"{pr['m2_alone']['median_ms']:.0f} / {pr['cascade']['median_ms']:.0f}")
        put("Per-request p95 ms (M2 / cascade)", name, f"{pr['m2_alone']['p95_ms']:.0f} / {pr['cascade']['p95_ms']:.0f}")
        put("Interpolation bias on Stage-1 total", name, f"{row['interp_error']['k1']['total_bias']:+.1%}")
        put("Interpolation bias on Stage-2 total", name, f"{row['interp_error']['k2']['total_bias']:+.1%}")
        print(f"[{name}] n={nn} e={esc.mean():.3f} measured red val={mv['reduction']:+.3f} ood={mo['reduction']:+.3f} | "
              f"priced val={pv['reduction']:+.3f} ood={po['reduction']:+.3f} | k1 bias {row['interp_error']['k1']['total_bias']:+.3f} "
              f"k2 bias {row['interp_error']['k2']['total_bias']:+.3f} | routing disagreements {row['identity']['routing_disagreements']}")

    for r, cells in table.items():
        lines.append(f"| {r} | " + " | ".join(cells.get(k, "—") for k in STAGE1) + " |")
    out = ROOT / a.out
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("wrote", out.with_suffix(".json"), out.with_suffix(".md"))


if __name__ == "__main__":
    main()
