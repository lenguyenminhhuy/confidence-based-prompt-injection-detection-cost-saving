"""Stage-1 selection table + single-stage eval rows — NO GPU,
payload-safe. Computes, for each candidate with {val,eval}_logits.jsonl:

  * val:  DR@1%FPR (primary selection metric), ECE, F1  -> stage1_selection.json
  * eval: AUROC, DR@1%FPR, ECE                          -> tab:stage1-single row

Selection rule: primary val DR@1%FPR; tie-break pre-calibration ECE.
Reuses the tested load_rows + compute_metrics, so llama/qwen reproduce their
published rows exactly and Granite slots in on the identical code path.

    PYTHONPATH=. python scripts/eval/fill_stage1_selection.py \
        --results-dir results/stage1 \
        --out results/metrics/stage1_selection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.score_stage1_logits import compute_metrics, load_rows  # noqa: E402

PARAMS_B = {"llama3.2-1b": 1.24, "qwen2.5-1.5b": 1.54, "granite-guardian-2b": 2.53}
SPLITS = {"val": ROOT / "data/train_proposal/val.jsonl",
          "eval": ROOT / "data/eval_proposal/eval.jsonl"}
FPR = "0.01"


def _metrics(cand_dir: Path, split_stem: str, split_path: Path):
    lp = cand_dir / f"{split_stem}_logits.jsonl"
    if not lp.exists():
        return None
    labels, scores, srcs, chans = load_rows(lp, split_path)
    return compute_metrics(labels, scores, srcs, chans)


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 selection + single-stage rows")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results/stage1")
    p.add_argument("--out", type=Path, default=ROOT / "results/metrics/stage1_selection.json")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    cands = sorted(d.name for d in a.results_dir.iterdir() if d.is_dir())
    rows = {}
    for name in cands:
        d = a.results_dir / name
        val = _metrics(d, "val", SPLITS["val"])
        ev = _metrics(d, "eval", SPLITS["eval"])
        if val is None and ev is None:
            continue
        rows[name] = {"params_b": PARAMS_B.get(name)}
        if val is not None:
            rows[name]["val"] = {"dr_at_1pct_fpr": round(val["detection_rate_at_fpr"][FPR]["dr"], 4),
                                 "ece": round(val["ece"], 4), "f1": round(val["f1"], 4),
                                 "auroc": round(val["auroc"], 4)}
        if ev is not None:
            rows[name]["eval"] = {"auroc": round(ev["auroc"], 4),
                                  "dr_at_1pct_fpr": round(ev["detection_rate_at_fpr"][FPR]["dr"], 4),
                                  "ece": round(ev["ece"], 4)}

    # selection: primary val DR@1%FPR, tie-break lower val ECE
    ranked = sorted((n for n in rows if "val" in rows[n]),
                    key=lambda n: (-rows[n]["val"]["dr_at_1pct_fpr"], rows[n]["val"]["ece"]))
    winner = ranked[0] if ranked else None
    result = {"selection_rule": "primary val DR@1%FPR, tie-break pre-calibration val ECE",
              "winner": winner, "ranked": ranked, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))

    print(f"\n=== Stage-1 selection (winner: {winner}) ===")
    print(f"{'candidate':22s} {'val DR@1%':>10s} {'val ECE':>8s} {'val F1':>8s} | "
          f"{'eval AUROC':>10s} {'eval DR@1%':>10s} {'eval ECE':>9s}")
    for n in sorted(rows, key=lambda x: -(rows[x].get('val',{}).get('dr_at_1pct_fpr') or -1)):
        v, e = rows[n].get("val", {}), rows[n].get("eval", {})
        print(f"{n:22s} {v.get('dr_at_1pct_fpr','   -'):>10} {v.get('ece','  -'):>8} "
              f"{v.get('f1','  -'):>8} | {e.get('auroc','  -'):>10} "
              f"{e.get('dr_at_1pct_fpr','  -'):>10} {e.get('ece','  -'):>9}")
    print(f"\nwrote -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
