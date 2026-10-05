"""Length-band analysis (paper Tables "length-auroc" and "benignlen").

Per detector: whole-set and per-band AUROC (bands by character length of the
evaluation text), Spearman rho between suspicion (1 - p_safe) and length on
benign rows, global DR at 1% FPR, and DR at 1% FPR with a threshold set inside
each band (oracle, eval-tuned). No GPU; reads saved logits.

    python scripts/eval/analyze_length_bands.py \
        --detector qwen=results/stage1/qwen2.5-1.5b \
        --detector qwen-lb=results/stage1_length_balanced/qwen2.5-1.5b \
        --detector m2=results/stage2/mistral-7b-v0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.evaluation.metrics import detection_rate_at_fpr  # noqa: E402
from eval_cascade import _read_logits  # noqa: E402
from score_stage1_logits import auroc  # noqa: E402

FPR_TARGET = 0.01
BANDS = {"le_512": (0, 512), "513_2048": (513, 2048), "gt_2048": (2049, None)}


def _read_eval(path: Path):
    ys, lens = [], []
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ys.append(int(r["label"])); lens.append(len(r["text"]))
    return np.asarray(ys, int), np.asarray(lens, int)


def _dr(y, s):
    return detection_rate_at_fpr(y, s, (FPR_TARGET,))[str(FPR_TARGET)]["dr"]


def analyze(s, y, lens):
    ben = y == 0
    out = {"auroc_whole": auroc(y, s), "dr_at_1pct_fpr": _dr(y, s),
           "spearman_benign": float(spearmanr(s[ben], lens[ben]).statistic),
           "n_whole": [int(len(y)), int(y.sum())], "bands": {}}
    for name, (lo, hi) in BANDS.items():
        m = (lens >= lo) & ((lens <= hi) if hi is not None else True)
        out["bands"][name] = {"n": [int(m.sum()), int(y[m].sum())],
                              "auroc": auroc(y[m], s[m]),
                              "dr_at_1pct_fpr_band_threshold": _dr(y[m], s[m])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detector", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--eval-split", type=Path, default=ROOT / "data/eval_proposal/eval.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results/analysis/length_bands.json")
    args = ap.parse_args()

    y, lens = _read_eval(args.eval_split)
    out = {}
    for spec in args.detector:
        name, d = spec.split("=", 1)
        s = 1.0 - _read_logits(Path(d) / "eval_logits.jsonl")[2]
        assert len(s) == len(y), f"{name}: logits / eval rows mismatch"
        r = out[name] = analyze(s, y, lens)
        b = r["bands"]
        print(f"{name:12s} rho={r['spearman_benign']:+.3f} auroc={r['auroc_whole']:.3f} "
              f"dr={r['dr_at_1pct_fpr']:.3f} | band auroc "
              + " ".join(f"{b[k]['auroc']:.3f}" for k in BANDS)
              + " | band dr " + " ".join(f"{b[k]['dr_at_1pct_fpr_band_threshold']:.3f}" for k in BANDS))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
