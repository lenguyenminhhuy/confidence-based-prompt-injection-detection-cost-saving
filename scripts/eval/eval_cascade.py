"""End-to-end cascade evaluation from saved Stage-1 and Stage-2 logits (no GPU, payload-safe).

Consumes the per-row Stage-1 and Stage-2 scores on the SAME eval rows and computes
the end-to-end cascade.

Design (justified by the results, not assumed):
  * ONE-SIDED "obvious-benign filter": Stage-1 auto-passes rows with p_safe >=
    theta_safe (predict benign, skip Stage-2); everything else escalates to
    Stage-2, which decides at threshold theta2. Two-sided auto-block is never
    budget-positive here — a Stage-1 auto-block of benign spends the global FP
    budget that Stage-2 could spend more accurately, and this Stage-1 is badly
    miscalibrated OOD (eval ECE ~0.26-0.35) — so the optimum collapses to the
    one-sided filter. Stage-1 provides cost savings by passing benign, not by
    blocking attacks.
  * OVERALL FPR == 1%: theta2 is tuned so the whole cascade's benign false-positive
    rate hits the target. Because auto-passed benign contribute zero FPs, the entire
    0.01*n_benign budget is spent on the (smaller) escalated-benign pool. This is
    the SAME "threshold set on eval benign to hit 1% FPR" convention used for the
    single-stage baselines here, so the cascade-vs-single-stage comparison is
    apples-to-apples (each sets exactly one eval threshold).

Honesty guards (added after review):
  * OPERATING POINT theta_safe is SELECTED ON CAL and FROZEN, then reported on eval
    (no test-set argmax): the theta_safe with the lowest cal escalation rate whose
    cascade keeps at least --keep-frac (default 0.99) of Stage-2's cal detection
    rate at the target FPR. The eval-optimal frontier is also reported, clearly
    labelled as an in-sample upper bound.
  * BOOTSTRAP 95% CIs on eval for cascade TPR, Stage-2-only DR, and the
    (cascade - Stage-2) gap — so "matches" vs "exceeds" is decided by the CI.
  * COST is reported here as a reduction-vs-(k1/k2) curve with a parameter-count
    proxy. Measured k1/k2 come from scripts/cost/measure_latency.py.
  * DR@fixed-FPR and the routing frontier are invariant to monotone temperature
    scaling, so calibration is a DIAGNOSTIC (ECE before/after), not a lever.

PAYLOAD HYGIENE: reads only label + logit scores; never input text.
Row-order join: logit files carry only {logp_benign, logp_injection, p_safe} (no id);
alignment to the split is by row order, externally corroborated by per-source label
purity in scripts/eval/score_stage1_logits.py (e.g. alpaca pos=0, bipia neg=0).

    PYTHONPATH=. python scripts/eval/eval_cascade.py \
        --stage1-dir results/stage1/llama3.2-1b \
        --stage2-dir results/stage2/mistral-7b-v0.1 \
        --stage1-name llama3.2-1b --stage2-name mistral-7b \
        --out-dir results/analysis/cascade_llama
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.eval.score_stage1_logits import _label_to_int, auroc  # noqa: E402
from src.calibration.temperature_scaling import TemperatureScaler, gap_from_logps  # noqa: E402
from src.evaluation.cost import cost_reduction  # noqa: E402
from src.evaluation.metrics import detection_rate_at_fpr, ece  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("eval_cascade")

FPR_TARGET = 0.01                       # headline operating FPR
K1K2_RATIOS = [0.02, 0.05, 0.1, 0.2, 0.3]
STAGE2_PARAMS = 7.24e9                  # Mistral-7B base (DETECTOR_PROFILES datasentinel_7b)
STAGE1_PARAMS = {"llama3.2-1b": 1.24e9, "qwen2.5-1.5b": 1.54e9,
                 "granite-guardian-2b": 2.53e9}


def _read_logits(path: Path):
    lb, li, ps = [], [], []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lb.append(r["logp_benign"]); li.append(r["logp_injection"]); ps.append(r["p_safe"])
    return np.asarray(lb, float), np.asarray(li, float), np.asarray(ps, float)


def _read_labels(path: Path) -> np.ndarray:
    ys = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            ys.append(_label_to_int(json.loads(line).get("label")))  # label only, never text
    return np.asarray(ys, int)


def _join(stage_dir: Path, split_stem: str, labels_path: Path):
    lb, li, ps = _read_logits(stage_dir / f"{split_stem}.jsonl")
    y = _read_labels(labels_path)
    if len(ps) != len(y):
        raise SystemExit(f"ROW MISMATCH {stage_dir.name}/{split_stem}={len(ps)} vs "
                         f"{labels_path.name}={len(y)}")
    return lb, li, ps, y


def _grid(lo: float, hi: float, step: float):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 10) for i in range(n + 1)]


def _cascade_point(s1_ps, s2, y, theta_safe, fpr_target):
    """One-sided cascade metrics at a fixed theta_safe. Overall FPR tuned to target.

    NaN-robust: reconstructs FP/TP from detection_rate_at_fpr's own surviving
    counts (d['n_benign'] / d['n_injection']), not the raw escalated sizes.
    """
    is_ben = y == 0; is_att = y == 1
    n_ben = int(is_ben.sum()); n_att = int(is_att.sum())
    budget = fpr_target * n_ben
    auto_pass = s1_ps >= theta_safe
    esc = ~auto_pass
    e = float(esc.sum()) / len(y)
    leaked = int((auto_pass & is_att).sum())   # auto-passed attacks: FN regardless of Stage-2
    n_eb = int((esc & is_ben).sum())
    if n_eb == 0:
        caught = int((esc & is_att).sum()); fp = 0; theta2 = None; resolvable = True
    else:
        local = min(1.0, budget / n_eb)
        d = detection_rate_at_fpr(y[esc], s2[esc], (local,))[str(local)]
        fp = int(round(d["achieved_fpr"] * d["n_benign"]))
        caught = int(round(d["dr"] * d["n_injection"])) if d["n_injection"] else 0
        theta2 = float(d["threshold"]) if math.isfinite(d["threshold"]) else None
        resolvable = bool(d["resolvable"])
    tpr = caught / n_att if n_att else float("nan")
    fpr = fp / n_ben if n_ben else float("nan")
    return {"theta_safe": float(theta_safe), "escalation_rate": e,
            "e2e_tpr": tpr, "e2e_fpr": fpr, "leaked_attacks": leaked,
            "caught_escalated": caught, "false_positives": fp,
            "theta2": theta2, "resolvable": resolvable, "n_esc_benign": n_eb}


def _select_theta_safe_on_cal(s1_ps_c, s2_c, y_c, grid, stage2_cal_dr, keep_frac, fpr_target):
    """Smallest (cheapest) theta_safe on CAL that keeps >=keep_frac of Stage-2 cal DR
    at <= target cal FPR. Frozen and reported on eval (no test-set selection)."""
    target_tpr = keep_frac * stage2_cal_dr
    feasible = []
    for th in grid:
        pt = _cascade_point(s1_ps_c, s2_c, y_c, th, fpr_target)
        if pt["e2e_fpr"] <= fpr_target + 1e-9 and pt["e2e_tpr"] >= target_tpr - 1e-9:
            feasible.append((th, pt["escalation_rate"]))
    if not feasible:
        return None
    return min(feasible, key=lambda t: (t[1], t[0]))[0]  # min escalation, then min theta


def _bootstrap_gap(s1_ps_e, s2_e, y, theta_safe, fpr_target, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    tprs, s2drs, gaps = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        pt = _cascade_point(s1_ps_e[idx], s2_e[idx], y[idx], theta_safe, fpr_target)
        d2 = detection_rate_at_fpr(y[idx], s2_e[idx], (fpr_target,))[str(fpr_target)]
        tprs.append(pt["e2e_tpr"]); s2drs.append(d2["dr"]); gaps.append(pt["e2e_tpr"] - d2["dr"])

    def _ci(a):
        lo, med, hi = np.percentile(np.asarray(a, float), [2.5, 50, 97.5])
        return {"lo": float(lo), "median": float(med), "hi": float(hi)}
    g = _ci(gaps)
    return {"n_boot": n_boot, "seed": seed,
            "cascade_tpr": _ci(tprs), "stage2_only_dr": _ci(s2drs),
            "gap_cascade_minus_stage2": g,
            "gap_excludes_zero": bool(g["lo"] > 0 or g["hi"] < 0),
            "interpretation": ("cascade EXCEEDS Stage-2 (CI>0)" if g["lo"] > 0 else
                               "cascade WORSE than Stage-2 (CI<0)" if g["hi"] < 0 else
                               "cascade MATCHES Stage-2 (gap CI straddles 0)")}


def _calib_diag(name, lb_c, li_c, y_c, lb_e, li_e, y_e):
    scaler = TemperatureScaler().fit(lb_c, li_c, y_c)

    def _ece_at(lb, li, y, T):
        z = gap_from_logps(lb, li)
        return float(ece(y, 1.0 - 1.0 / (1.0 + np.exp(-z / T))))
    return {"name": name, "T_cal": float(scaler.T),
            "ece_cal_raw": _ece_at(lb_c, li_c, y_c, 1.0),
            "ece_cal_tempered": _ece_at(lb_c, li_c, y_c, scaler.T),
            "ece_eval_raw": _ece_at(lb_e, li_e, y_e, 1.0),
            "ece_eval_tempered": _ece_at(lb_e, li_e, y_e, scaler.T)}


def _cost_block(e, proxy_ratio):
    ratios = sorted(set(K1K2_RATIOS + [round(proxy_ratio, 3)]))
    by_ratio = {f"{r:g}": {"breakeven_e": round(1.0 - r, 4),
                           "cascade_cheaper": bool(e < 1.0 - r),
                           "cost_reduction_pct": round(cost_reduction(r, 1.0, e) * 100.0, 2)}
                for r in ratios}
    return {"escalation_rate": e, "flop_param_proxy_ratio": round(proxy_ratio, 4),
            "cost_reduction_at_proxy_pct": round(cost_reduction(proxy_ratio, 1.0, e) * 100.0, 2),
            "by_k1k2_ratio": by_ratio,
            "caveat": ("Ratios are FLOP/param proxies (compute, not latency). Measured "
                       "latency cost: scripts/cost/analyze_length_weighted_cost.py.")}


def evaluate(args) -> dict:
    s1_dir = Path(args.stage1_dir); s2_dir = Path(args.stage2_dir)
    cal_labels = ROOT / args.cal_split
    eval_labels = ROOT / args.eval_split

    s1_lb_c, s1_li_c, s1_ps_c, y_c = _join(s1_dir, "cal_logits", cal_labels)
    s2_lb_c, s2_li_c, s2_ps_c, _ = _join(s2_dir, "cal_logits", cal_labels)
    s1_lb_e, s1_li_e, s1_ps_e, y = _join(s1_dir, "eval_logits", eval_labels)
    s2_lb_e, s2_li_e, s2_ps_e, _ = _join(s2_dir, "eval_logits", eval_labels)

    s1c, s2c = 1.0 - s1_ps_c, 1.0 - s2_ps_c
    s1e, s2e = 1.0 - s1_ps_e, 1.0 - s2_ps_e   # injection scores (higher = more injection)

    # --- single-stage baselines (threshold on eval benign; standard) ---------
    dr1 = detection_rate_at_fpr(y, s1e, (FPR_TARGET,))[str(FPR_TARGET)]
    dr2 = detection_rate_at_fpr(y, s2e, (FPR_TARGET,))[str(FPR_TARGET)]
    single = {
        "stage1": {"name": args.stage1_name, "auroc": auroc(y, s1e),
                   "dr_at_1pct_fpr": dr1["dr"], "achieved_fpr": dr1["achieved_fpr"],
                   "resolvable": dr1["resolvable"]},
        "stage2": {"name": args.stage2_name, "auroc": auroc(y, s2e),
                   "dr_at_1pct_fpr": dr2["dr"], "achieved_fpr": dr2["achieved_fpr"],
                   "resolvable": dr2["resolvable"]},
    }

    # --- CAL: select + freeze theta_safe --------------------------------------
    dr2_cal = detection_rate_at_fpr(y_c, s2c, (FPR_TARGET,))[str(FPR_TARGET)]["dr"]
    grid = _grid(args.theta_safe_lo, args.theta_safe_hi, args.step)
    theta_safe = _select_theta_safe_on_cal(s1_ps_c, s2c, y_c, grid, dr2_cal,
                                            args.keep_frac, FPR_TARGET)
    cal_frozen = None
    if theta_safe is not None:
        cal_pt = _cascade_point(s1_ps_c, s2c, y_c, theta_safe, FPR_TARGET)
        eval_pt = _cascade_point(s1_ps_e, s2e, y, theta_safe, FPR_TARGET)
        proxy = STAGE1_PARAMS.get(args.stage1_name, float("nan")) / args.stage2_params
        boot = _bootstrap_gap(s1_ps_e, s2e, y, theta_safe, FPR_TARGET,
                              args.bootstrap, args.bootstrap_seed) if args.bootstrap else None
        cal_frozen = {
            "theta_safe": theta_safe, "keep_frac": args.keep_frac,
            "on_cal": cal_pt, "on_eval": eval_pt,
            "bootstrap_eval": boot,
            "cost": _cost_block(eval_pt["escalation_rate"], proxy),
        }

    # --- EVAL-optimal frontier (in-sample UPPER BOUND, labelled as such) ------
    frontier = [_cascade_point(s1_ps_e, s2e, y, th, FPR_TARGET) for th in grid]
    at_fpr = [r for r in frontier if r["e2e_fpr"] <= FPR_TARGET + 1e-9]
    eval_opt = max(at_fpr, key=lambda r: (r["e2e_tpr"], -r["escalation_rate"])) if at_fpr else None

    return {
        "stage1_name": args.stage1_name, "stage2_name": args.stage2_name,
        "n": int(len(y)), "n_attacks": int((y == 1).sum()), "n_benign": int((y == 0).sum()),
        "fpr_target": FPR_TARGET,
        "calibration_diagnostic": {
            "stage1": _calib_diag(args.stage1_name, s1_lb_c, s1_li_c, y_c, s1_lb_e, s1_li_e, y),
            "stage2": _calib_diag(args.stage2_name, s2_lb_c, s2_li_c, y_c, s2_lb_e, s2_li_e, y),
            "note": "DR@fixed-FPR + routing frontier invariant to monotone T; ECE diagnostic only.",
        },
        "single_stage": single,
        "stage2_only_dr_at_1pct_fpr": dr2["dr"],
        "headline_cal_frozen": cal_frozen if cal_frozen else {
            "feasible": False, "note": "no cal theta_safe kept >=keep_frac of Stage-2 cal DR "
                                       "at <=1% cal FPR"},
        "eval_optimal_upper_bound": (
            {**eval_opt, "note": "IN-SAMPLE upper bound: theta_safe argmax'd on eval; "
                                 "optimistic, not deployment-realizable"} if eval_opt
            else {"feasible": False}),
        "grid": {"theta_safe": [args.theta_safe_lo, args.theta_safe_hi], "step": args.step},
        "provenance": {"stage1_dir": str(s1_dir), "stage2_dir": str(s2_dir),
                       "routing": "one-sided obvious-benign filter (never auto-block)"},
    }


def _print(res: dict) -> None:
    s = res["single_stage"]
    print(f"\n=== CASCADE  {res['stage1_name']} (Stage-1) -> {res['stage2_name']} (Stage-2) ===")
    print(f"eval n={res['n']}  attacks={res['n_attacks']}  benign={res['n_benign']}  "
          f"FPR target={res['fpr_target']:.0%}")
    print("\nsingle-stage DR@1%FPR (threshold on eval benign):")
    print(f"  Stage-1 {s['stage1']['name']:<18} AUROC={s['stage1']['auroc']:.4f}  "
          f"DR={s['stage1']['dr_at_1pct_fpr']:.4f}")
    print(f"  Stage-2 {s['stage2']['name']:<18} AUROC={s['stage2']['auroc']:.4f}  "
          f"DR={s['stage2']['dr_at_1pct_fpr']:.4f}")
    cd = res["calibration_diagnostic"]
    print("\nE3 calibration diagnostic (ECE raw->tempered):")
    for k in ("stage1", "stage2"):
        d = cd[k]
        print(f"  {d['name']:<20} T={d['T_cal']:.3f}  cal {d['ece_cal_raw']:.3f}->"
              f"{d['ece_cal_tempered']:.3f}  eval {d['ece_eval_raw']:.3f}->{d['ece_eval_tempered']:.3f}")
    h = res["headline_cal_frozen"]
    print("\nHEADLINE — theta_safe frozen on CAL, reported on EVAL:")
    if h.get("feasible") is False:
        print(f"  INFEASIBLE — {h['note']}")
    else:
        ev = h["on_eval"]
        print(f"  theta_safe={h['theta_safe']}  ->  eval: e={ev['escalation_rate']:.4f}  "
              f"TPR={ev['e2e_tpr']:.4f}  FPR={ev['e2e_fpr']:.4f}  leaked={ev['leaked_attacks']}"
              f"  resolvable={ev['resolvable']}")
        print(f"        Stage-2-only DR={res['stage2_only_dr_at_1pct_fpr']:.4f}")
        b = h.get("bootstrap_eval")
        if b:
            g = b["gap_cascade_minus_stage2"]
            print(f"  bootstrap gap (cascade-Stage2) 95% CI [{g['lo']:+.4f}, {g['hi']:+.4f}]  "
                  f"-> {b['interpretation']}")
        c = h["cost"]
        print(f"  cost @ e={c['escalation_rate']:.3f}: FLOP/param proxy k1/k2={c['flop_param_proxy_ratio']}"
              f" -> {c['cost_reduction_at_proxy_pct']:+.1f}% compute reduction (k1 unmeasured)")
        for r, cc in c["by_k1k2_ratio"].items():
            print(f"    k1/k2={r:<5} breakeven e<{cc['breakeven_e']:.2f}  "
                  f"{'CHEAPER' if cc['cascade_cheaper'] else 'NOT cheaper':<12} "
                  f"{cc['cost_reduction_pct']:+.1f}%")
    eo = res["eval_optimal_upper_bound"]
    if eo.get("feasible") is not False:
        print(f"\n(eval-optimal upper bound, optimistic: TPR={eo['e2e_tpr']:.4f} @ "
              f"e={eo['escalation_rate']:.4f}, theta_safe={eo['theta_safe']})")


def parse_args():
    p = argparse.ArgumentParser(description="E5+E7 real-Stage-2 cascade eval (no GPU)")
    p.add_argument("--stage1-dir", required=True)
    p.add_argument("--stage2-dir", required=True)
    p.add_argument("--stage1-name", default="stage1")
    p.add_argument("--stage2-name", default="stage2")
    p.add_argument("--cal-split", default="data/train_proposal/cal.jsonl")
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--theta-safe-lo", type=float, default=0.0)
    p.add_argument("--theta-safe-hi", type=float, default=1.0)
    p.add_argument("--step", type=float, default=0.005)
    p.add_argument("--keep-frac", type=float, default=0.99,
                   help="fraction of Stage-2 cal DR the cal-frozen point must preserve")
    p.add_argument("--stage2-params", type=float, default=STAGE2_PARAMS)
    p.add_argument("--bootstrap", type=int, default=1000, help="0=off")
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/analysis/cascade")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    res = evaluate(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cascade_summary.json").write_text(json.dumps(res, indent=2))
    _print(res)
    print(f"\nwrote -> {args.out_dir / 'cascade_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
