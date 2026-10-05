"""Per-channel cascade detection (paper Table "cascade-channel").

For each Stage-1 model, at a fixed theta_safe and an overall 1% FPR, reports per
attack channel: number of attacks, M2-alone DR, cascade DR, and attacks cleared
(auto-passed) by Stage 1. No GPU; reads saved logits.

    python scripts/eval/analyze_by_channel.py \
        --stage1 qwen=results/stage1/qwen2.5-1.5b \
        --stage1 llama=results/stage1/llama3.2-1b \
        --stage1 granite=results/stage1/granite-guardian-2b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.evaluation.metrics import detection_rate_at_fpr  # noqa: E402
from eval_cascade import _cascade_point, _read_logits  # noqa: E402

FPR_TARGET = 0.01


def _read_eval(path: Path):
    ys, ch, src = [], [], []
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ys.append(int(r["label"])); ch.append(r.get("channel")); src.append(r["source"])
    return np.asarray(ys, int), np.asarray(ch, object), np.asarray(src, object)


def analyze(s1_ps, s2, y, ch, theta_safe, ties_flagged=False):
    pt = _cascade_point(s1_ps, s2, y, theta_safe, FPR_TARGET)
    t_global = detection_rate_at_fpr(y, s2, (FPR_TARGET,))[str(FPR_TARGET)]["threshold"]
    esc = s1_ps < theta_safe
    flagged_m2 = s2 > t_global  # same strict rule as detection_rate_at_fpr
    # Strict ">" matches _cascade_point's overall DR/FPR. The published table
    # counted Stage-2 scores equal to theta2 as flagged (--ties-flagged); that
    # flags 20 extra escalated rows for Qwen and pushes overall FPR just past 1%.
    above = (s2 >= pt["theta2"]) if ties_flagged else (s2 > pt["theta2"])
    flagged_cascade = esc & above if pt["theta2"] is not None else esc
    by = {}
    for c in sorted({c for c in ch[y == 1]}):
        m = (y == 1) & (ch == c)
        by[c] = {"n": int(m.sum()),
                 "m2_dr": float(flagged_m2[m].mean()),
                 "cascade_dr": float(flagged_cascade[m].mean()),
                 "cleared_by_stage1": int((m & ~esc).sum())}
    return {"theta_safe": theta_safe, "theta2_escalated": pt["theta2"],
            "m2_global_threshold": float(t_global),
            "escalation_rate": pt["escalation_rate"],
            "overall_dr": pt["e2e_tpr"], "overall_fpr": pt["e2e_fpr"], "by_channel": by}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage1", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--stage2-dir", type=Path, default=ROOT / "results/stage2/mistral-7b-v0.1")
    ap.add_argument("--eval-split", type=Path, default=ROOT / "data/eval_proposal/eval.jsonl")
    ap.add_argument("--theta-safe", type=float, default=0.005)
    ap.add_argument("--ties-flagged", action="store_true",
                    help="count Stage-2 scores equal to theta2 as flagged (as in the paper table)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/analysis/cascade_by_channel.json")
    args = ap.parse_args()

    y, ch, src = _read_eval(args.eval_split)
    s2 = 1.0 - _read_logits(args.stage2_dir / "eval_logits.jsonl")[2]
    assert len(s2) == len(y), "Stage-2 logits / eval rows mismatch"
    # M2 DR at 1% FPR with each injection source left out (benign rows all kept).
    out = {"m2_dr_without_source": {
        s: detection_rate_at_fpr(y[src != s], s2[src != s], (FPR_TARGET,))[str(FPR_TARGET)]["dr"]
        for s in sorted(set(src[y == 1]))}}
    print("M2 DR without source:", {k: round(v, 3) for k, v in out["m2_dr_without_source"].items()})
    for spec in args.stage1:
        name, d = spec.split("=", 1)
        s1_ps = _read_logits(Path(d) / "eval_logits.jsonl")[2]
        assert len(s1_ps) == len(y), f"{name}: logits / eval rows mismatch"
        out[name] = analyze(s1_ps, s2, y, ch, args.theta_safe, args.ties_flagged)
        for c, v in out[name]["by_channel"].items():
            print(f"{name:10s} {c:9s} n={v['n']:5d} m2={v['m2_dr']:.3f} "
                  f"cascade={v['cascade_dr']:.3f} cleared={v['cleared_by_stage1']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
