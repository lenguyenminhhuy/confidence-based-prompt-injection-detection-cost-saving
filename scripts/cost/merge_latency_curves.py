"""Merge a mid-grid latency session into a base curve, on target-length keys.

The base curves (results/analysis/latency_curve_{nf4,bf16stage1}.json) anchor at the
template floor, 128, 512, 1024 and 2048 tokens. The mid-grid session
adds 192, 256 and 384 and re-measures 128 and 512 as a drift
control. Downstream cost scripts take a single latency file, so this produces one.

On an overlapping target the mid-grid point wins: it was taken in the same session
as the new anchors, and a curve has to be internally consistent to be interpolated
over. The relative gap on every overlap is recorded under "merge_drift" so a session
that moved is visible rather than silently absorbed.

    python scripts/cost/merge_latency_curves.py \
        --base results/analysis/latency_curve_nf4.json \
        --extra results/analysis/latency_curve_midgrid_nf4.json \
        --out results/analysis/latency_curve_merged_nf4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def merge(base: dict, extra: dict) -> dict:
    out = json.loads(json.dumps(base))
    drift = []
    for model, spec in extra["per_model"].items():
        if model not in out["per_model"]:
            out["per_model"][model] = spec
            continue
        dest = out["per_model"][model]["by_length"]
        for target, rec in spec["by_length"].items():
            if target in dest:
                gap = rec["median_ms"] / dest[target]["median_ms"] - 1.0
                drift.append({"model": model, "target_tokens": int(target),
                              "base_ms": dest[target]["median_ms"],
                              "extra_ms": rec["median_ms"],
                              "relative_gap": round(gap, 4)})
            dest[target] = rec
        out["per_model"][model]["by_length"] = dict(
            sorted(dest.items(), key=lambda kv: int(kv[0])))
    out["merged_from"] = {"base": base.get("regime"), "extra": extra.get("regime"),
                          "extra_measured_utc": extra.get("measured_utc")}
    out["merge_drift"] = drift
    out["note"] = ("Merged curve: base anchors plus the mid-grid session. "
                   "Overlapping targets take the mid-grid value; see merge_drift. "
                   + str(base.get("note", "")))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--extra", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    res = merge(json.loads((ROOT / a.base).read_text()),
                json.loads((ROOT / a.extra).read_text()))
    (ROOT / a.out).write_text(json.dumps(res, indent=2) + "\n")
    worst = max(res["merge_drift"], key=lambda r: abs(r["relative_gap"]), default=None)
    print(f"wrote {a.out}  ({len(res['merge_drift'])} overlaps"
          + (f", largest {worst['relative_gap']:+.1%} on {worst['model']} @ "
             f"{worst['target_tokens']} tok)" if worst else ")"))


if __name__ == "__main__":
    main()
