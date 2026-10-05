#!/usr/bin/env python3
"""Cascade cost reduction as a function of uniform request length.

The published headline fixes every request at 512 tokens; `analyze_length_weighted_cost.py`
re-prices the same routing on the benchmark's own empirical length mix. Neither answers
"how does the saving vary with request length?", which is what the paper claims when it
says the saving grows with length.

No GPU work is needed at the anchors themselves: reduction(L) = 1 - k1(L)/k2(L) - e, and
k1, k2 are measured at every anchor in results/analysis/latency_curve_*.json. The escalation
rate e is a property of the Stage-1 scores, not of request length, so it is held fixed per
arm at the value measured for that quantization regime.

The base curves have no anchor between 128 and 512, which is exactly where the saving turns
over. `--extra-curve` merges in a later session's mid-grid measurement (see
scripts/cost/benchmark_latency_curve.py). Targets present in both files are treated as a drift
control: the merged point comes from the extra file, and the relative gap against the base
file is recorded under "drift_check" so a session that moved can be spotted rather than
silently averaged in.

Usage:
    python scripts/cost/analyze_uniform_length_cost.py \
        --out results/analysis/cost_uniform_length_sweep.json

    # paper run: 8-anchor curves merged over the 5-anchor base
    python scripts/cost/analyze_uniform_length_cost.py \
        --extra-curve nf4=results/analysis/latency_curve_full_nf4.json \
        --extra-curve bf16=results/analysis/latency_curve_full_bf16.json \
        --out results/analysis/cost_uniform_length_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE2 = "mistral-7b-v0.1"
SECONDS_PER_HOUR = 3600.0
DEFAULT_GPU_HOURLY_USD = 1.20

# Escalation rate measured per quantization regime. bf16 Stage-1 scores differ from nf4
# (~5.2% mean relative difference on eval), so the same theta_safe routes a different
# fraction of traffic and e must be taken from the matching regime.
ESCALATION = {
    ("nf4", "llama3.2-1b"): 0.5160989629859789,
    ("nf4", "qwen2.5-1.5b"): 0.3968617703033363,
    ("bf16", "llama3.2-1b"): 0.44420709208839865,
    ("bf16", "qwen2.5-1.5b"): 0.37717015574630053,
}
CURVES = {
    "nf4": "results/analysis/latency_curve_nf4.json",
    "bf16": "results/analysis/latency_curve_bf16stage1.json",
}


def _by_target(path: Path) -> dict:
    """Median latency and realised prompt length, keyed on the benchmark's target length.

    Anchors are keyed on target rather than actual tokens because each tokeniser renders
    the same target to a slightly different prompt length (llama 87, qwen 86, mistral 83
    at the template floor), so keying on actual tokens gives Stage 1 and Stage 2 disjoint
    anchor sets and nothing to pair.
    """
    doc = json.loads(path.read_text())
    return {
        model: {
            int(k): {"median_ms": float(r["median_ms"]),
                     "actual_prompt_tokens": int(r["actual_prompt_tokens"])}
            for k, r in (spec.get("by_length") or {}).items()
        }
        for model, spec in doc["per_model"].items()
    }


def _merge(base: dict, extra: dict) -> tuple[dict, list[dict]]:
    """Base curve overlaid with a later session's points; overlaps become drift checks.

    Returns the merged curve and one drift record per (model, target) that both files
    contain. The extra file wins, because its points were taken in the same session as
    the new mid-grid anchors and the curve must stay internally consistent.
    """
    merged = {m: dict(pts) for m, pts in base.items()}
    drift = []
    for model, pts in extra.items():
        merged.setdefault(model, {})
        for target, rec in pts.items():
            old = base.get(model, {}).get(target)
            if old is not None:
                gap = rec["median_ms"] / old["median_ms"] - 1.0
                drift.append({"model": model, "target_tokens": target,
                              "base_ms": round(old["median_ms"], 3),
                              "extra_ms": round(rec["median_ms"], 3),
                              "relative_gap": round(gap, 4)})
            merged[model][target] = rec
    return merged, drift


def sweep(usd_per_gpu_hour: float, extra_curves: dict[str, str] | None = None) -> dict:
    arms = {}
    drift_by_regime: dict[str, list[dict]] = {}
    extra_curves = extra_curves or {}
    for regime, rel in CURVES.items():
        # The base curve is optional when an --extra-curve covers every target.
        curve = _by_target(ROOT / rel) if (ROOT / rel).exists() or regime not in extra_curves else {}
        if regime in extra_curves:
            curve, drift_by_regime[regime] = _merge(
                curve, _by_target(ROOT / extra_curves[regime]))
        k2_curve = curve[STAGE2]
        for stage1 in ("llama3.2-1b", "qwen2.5-1.5b"):
            e = ESCALATION[(regime, stage1)]
            points = []
            for target in sorted(set(curve[stage1]) & set(k2_curve)):
                k1 = curve[stage1][target]["median_ms"]
                k2 = k2_curve[target]["median_ms"]
                r = k1 / k2
                per_m = lambda ms: ms / 1000.0 / SECONDS_PER_HOUR * usd_per_gpu_hour * 1e6
                points.append({
                    "target_tokens": target,
                    "stage1_prompt_tokens": curve[stage1][target]["actual_prompt_tokens"],
                    "stage2_prompt_tokens": k2_curve[target]["actual_prompt_tokens"],
                    "k1_ms": round(k1, 3),
                    "k2_ms": round(k2, 3),
                    "r": round(r, 4),
                    "break_even_e": round(1 - r, 4),
                    "reduction": round(1 - r - e, 4),
                    "usd_per_million_baseline": round(per_m(k2), 2),
                    "usd_per_million_cascade": round(per_m(k1 + e * k2), 2),
                    "usd_per_million_saved": round(per_m(k2) - per_m(k1 + e * k2), 2),
                })
            best = max(points, key=lambda p: p["reduction"])
            arms[f"{regime}/{stage1}"] = {
                "regime": regime,
                "stage1": stage1,
                "escalation_rate": e,
                "peak_measured_at_tokens": best["target_tokens"],
                "peak_reduction": best["reduction"],
                "asymptotic_r_at_2048": points[-1]["r"],
                "points": points,
            }
    measured = sorted({p["target_tokens"] for arm in arms.values() for p in arm["points"]})
    gap = ("No 256-token anchor exists; the peak is bracketed by 128 and 1024 only."
           if 256 not in measured else
           f"Anchors measured at {measured}; the peak is bracketed by measured points.")
    return {
        "task": "uniform-length cost sweep (derived; no new GPU measurement)",
        "gpu": "NVIDIA A10G (g5)",
        "protocol": "batch=1, median of 40 reps after 8 warm-ups; Stage-2 always nf4",
        "formula": "reduction(L) = 1 - k1(L)/k2(L) - e",
        "gpu_hourly_usd_assumption": usd_per_gpu_hour,
        "gap": gap,
        "curves": {"base": CURVES, "extra": extra_curves},
        "drift_check": drift_by_regime,
        "arms": arms,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--usd-per-gpu-hour", type=float, default=DEFAULT_GPU_HOURLY_USD)
    p.add_argument("--extra-curve", action="append", default=[], metavar="REGIME=PATH",
                   help="merge a later session's curve into one regime (nf4 or bf16); "
                        "repeatable. Overlapping targets are reported under drift_check.")
    p.add_argument("--out", default="results/analysis/cost_uniform_length_sweep.json")
    args = p.parse_args()

    extra = {}
    for spec in args.extra_curve:
        regime, _, path = spec.partition("=")
        if regime not in CURVES or not path:
            raise SystemExit(f"--extra-curve expects one of {sorted(CURVES)}=PATH, got {spec!r}")
        extra[regime] = path

    res = sweep(args.usd_per_gpu_hour, extra)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")

    for regime, rows in res["drift_check"].items():
        if rows:
            worst = max(rows, key=lambda r: abs(r["relative_gap"]))
            print(f"drift check {regime}: {len(rows)} overlapping targets, "
                  f"largest gap {worst['relative_gap']:+.1%} "
                  f"({worst['model']} @ {worst['target_tokens']} tok)")

    for name, arm in res["arms"].items():
        print(f"\n{name}  e={arm['escalation_rate']:.4f}  "
              f"peak {arm['peak_reduction']:+.1%} at {arm['peak_measured_at_tokens']} tokens")
        for pt in arm["points"]:
            print(f"   {pt['target_tokens']:>5} tok  r={pt['r']:.3f}  "
                  f"reduction={pt['reduction']:+7.1%}  saved=${pt['usd_per_million_saved']:>7.2f}/M")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
