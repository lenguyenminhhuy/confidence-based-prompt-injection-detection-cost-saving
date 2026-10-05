"""E10 — seen-vs-unseen document diagnostic: is M2's document blindness a
cross-corpus GENERALIZATION gap, or a fundamental channel/paradigm limit?

E9 showed M2 catches ~6.7% of eval document attacks (openpromptinjection, an UNSEEN
corpus). But the training corpora contain 15k document-embedded attacks (BIPIA +
synthetic). The held-out VALIDATION split is drawn from the SAME corpora as training,
so its document attacks are SEEN-corpus, held-out instances. Comparing M2's document
detection on val (seen) vs eval (unseen) separates the two hypotheses:

  * seen-document HIGH, unseen-document LOW  -> cross-corpus generalization gap
      (data-diversity lever; the model learned corpus-specific surface cues).
  * seen-document ALSO LOW                    -> the document/indirect channel is
      hard even in-distribution (paradigm lever; more data won't fix it).

M2 was already scored on val (results/stage2/.../val_logits.jsonl), so this is a
no-GPU re-score. Standalone M2 detection (threshold on each set's own benign), broken
down by channel; AUROC per channel is threshold-free and robust to small benign pools.

PAYLOAD HYGIENE: reads only label + channel metadata + logit scores; never text.

    PYTHONPATH=. python scripts/eval/diag_seen_vs_unseen.py \
        --stage2-dir results/stage2/mistral-7b-v0.1 \
        --out-dir results/analysis/seen_vs_unseen
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

from scripts.eval.eval_cascade import _read_logits  # noqa: E402
from scripts.eval.score_stage1_logits import _label_to_int, auroc  # noqa: E402
from src.evaluation.metrics import detection_rate_at_fpr  # noqa: E402

# normalise channel names across splits (val uses *-embedded / *-output suffixes)
CHAN_NORM = {"document-embedded": "document", "tool-output": "tool", "direct": "direct",
             "document": "document", "tool": "tool"}
SOURCE_CHANNEL = {"openpromptinjection": "document", "agentdojo": "tool",
                  "struq_synthetic": "direct"}
_META = {"label", "source", "channel"}


def _read_meta(path: Path):
    labels, chans, srcs = [], [], []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = {k: json.loads(line).get(k) for k in _META}  # drop text
            labels.append(_label_to_int(r["label"]))
            ch = r.get("channel") or SOURCE_CHANNEL.get(r.get("source"))
            chans.append(CHAN_NORM.get(ch, ch))
            srcs.append(r.get("source"))
    return np.asarray(labels, int), np.asarray(chans, object), np.asarray(srcs, object)


def _analyse(name, s2, y, chan, fprs=(0.01, 0.05)):
    thr = {f: detection_rate_at_fpr(y, s2, (f,))[str(f)] for f in fprs}
    is_ben = y == 0
    out = {"name": name, "n": int(len(y)), "n_benign": int(is_ben.sum()),
           "n_attacks": int((y == 1).sum()),
           "overall": {f"dr@{int(f*100)}pct_fpr": round(thr[f]["dr"], 4) for f in fprs},
           "auroc": round(auroc(y, s2), 4), "by_channel": {}}
    for c in sorted({v for v in chan[y == 1] if v is not None}):
        m = (y == 1) & (chan == c)
        row = {"n_attacks": int(m.sum())}
        for f in fprs:
            t = thr[f]["threshold"]
            row[f"dr@{int(f*100)}pct_fpr"] = round(float((s2[m] >= t).mean()), 4) if m.sum() else None
        # channel AUROC: channel attacks vs ALL benign (threshold-free)
        yy = np.concatenate([np.ones(int(m.sum())), np.zeros(int(is_ben.sum()))])
        ss = np.concatenate([s2[m], s2[is_ben]])
        row["auroc_vs_benign"] = round(auroc(yy, ss), 4) if m.sum() and is_ben.sum() else None
        out["by_channel"][str(c)] = row
    return out


def evaluate(args) -> dict:
    s2_dir = Path(args.stage2_dir)
    # val = SEEN distribution (same corpora as training); eval = UNSEEN
    _, _, val_ps = _read_logits(s2_dir / "val_logits.jsonl")
    _, _, eval_ps = _read_logits(s2_dir / "eval_logits.jsonl")
    y_v, ch_v, _ = _read_meta(ROOT / args.val_split)
    y_e, ch_e, _ = _read_meta(ROOT / args.eval_split)
    if len(val_ps) != len(y_v) or len(eval_ps) != len(y_e):
        raise SystemExit("row-order mismatch logits vs metadata")

    seen = _analyse("val (SEEN corpora)", 1.0 - val_ps, y_v, ch_v)
    unseen = _analyse("eval (UNSEEN corpora)", 1.0 - eval_ps, y_e, ch_e)

    dseen = seen["by_channel"].get("document", {})
    dunseen = unseen["by_channel"].get("document", {})
    verdict = None
    if dseen.get("dr@1pct_fpr") is not None and dunseen.get("dr@1pct_fpr") is not None:
        hi = dseen["dr@1pct_fpr"] >= 0.5
        lo = dunseen["dr@1pct_fpr"] < 0.2
        if hi and lo:
            verdict = ("CROSS-CORPUS GENERALIZATION GAP: M2 detects SEEN document "
                       f"injections well ({dseen['dr@1pct_fpr']:.1%} @1%FPR) but collapses on "
                       f"UNSEEN document injections ({dunseen['dr@1pct_fpr']:.1%}). More data "
                       "won't help unless it broadens corpus/style diversity; the model learned "
                       "corpus-specific surface cues.")
        elif not hi:
            verdict = ("PARADIGM/CHANNEL LIMIT: M2 is weak on document injections even IN-"
                       f"distribution ({dseen['dr@1pct_fpr']:.1%} @1%FPR seen). More data of the "
                       "same paradigm is unlikely to help; needs a different detection approach "
                       "(span-level / task-drift).")
        else:
            verdict = (f"MIXED: seen-document {dseen['dr@1pct_fpr']:.1%}, unseen-document "
                       f"{dunseen['dr@1pct_fpr']:.1%} @1%FPR.")

    return {"stage2_dir": str(s2_dir), "seen": seen, "unseen": unseen,
            "document_seen_vs_unseen": {
                "seen_dr@1pct_fpr": dseen.get("dr@1pct_fpr"),
                "unseen_dr@1pct_fpr": dunseen.get("dr@1pct_fpr"),
                "seen_auroc_vs_benign": dseen.get("auroc_vs_benign"),
                "unseen_auroc_vs_benign": dunseen.get("auroc_vs_benign"),
                "verdict": verdict},
            "caveat": ("val benign channels (application-structured, conversational) differ "
                       "from eval benign; the 1%-FPR threshold is set per-set on its own benign, "
                       "so absolute DRs are set-relative. AUROC-vs-benign is the robust "
                       "threshold-free comparison. 'Seen' means same CORPUS as training, "
                       "held-out instances — not identical examples.")}


def _print(r: dict) -> None:
    for key in ("seen", "unseen"):
        s = r[key]
        print(f"\n=== {s['name']} ===  n={s['n']} benign={s['n_benign']} attacks={s['n_attacks']}")
        print(f"  overall {s['overall']}  AUROC={s['auroc']}")
        print(f"  {'channel':<12}{'n_att':>7}{'DR@1%':>9}{'DR@5%':>9}{'AUROC':>9}")
        for c, v in s["by_channel"].items():
            d1 = f"{v['dr@1pct_fpr']:.3f}" if v['dr@1pct_fpr'] is not None else "-"
            d5 = f"{v['dr@5pct_fpr']:.3f}" if v['dr@5pct_fpr'] is not None else "-"
            au = f"{v['auroc_vs_benign']:.3f}" if v['auroc_vs_benign'] is not None else "-"
            print(f"  {c:<12}{v['n_attacks']:>7}{d1:>9}{d5:>9}{au:>9}")
    d = r["document_seen_vs_unseen"]
    print("\n*** DOCUMENT CHANNEL — seen vs unseen ***")
    print(f"  DR@1%FPR:  seen {d['seen_dr@1pct_fpr']}  vs  unseen {d['unseen_dr@1pct_fpr']}")
    print(f"  AUROC:     seen {d['seen_auroc_vs_benign']}  vs  unseen {d['unseen_auroc_vs_benign']}")
    print(f"\n  VERDICT: {d['verdict']}")


def parse_args():
    p = argparse.ArgumentParser(description="E10 seen-vs-unseen document diagnostic")
    p.add_argument("--stage2-dir", default="results/stage2/mistral-7b-v0.1")
    p.add_argument("--val-split", default="data/train_proposal/val.jsonl")
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/analysis/seen_vs_unseen")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    res = evaluate(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "seen_vs_unseen_document.json").write_text(json.dumps(res, indent=2))
    _print(res)
    print(f"\nwrote -> {args.out_dir / 'seen_vs_unseen_document.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
