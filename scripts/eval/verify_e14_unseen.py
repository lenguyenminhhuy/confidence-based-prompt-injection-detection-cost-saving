"""E14 — verify that the reworded payloads really are unseen wording.

The manipulation is only meaningful if the reworded payloads are absent from the
detectors' training text. Two checks, both reusing the measures already
reported in scripts/eval/diag_direct_overlap.py so the numbers are comparable with
Table "Train/eval overlap of the injected instruction, per channel":

  1. exact containment — does any rewording occur as a substring anywhere in
     the training injections (and in the benign rows, which would matter for
     a different reason)?
  2. tail 5-gram coverage — fraction of each injection's trailing-80-character
     5-grams found in the training injections' tail vocabulary, per arm.

Prints aggregate counts only; no injection text is emitted.

    PYTHONPATH=. python scripts/eval/verify_e14_unseen.py \
        --train data/train_proposal/train_stage2.jsonl \
        --arms data/eval_proposal/e14_reword.jsonl \
        --out results/analysis/e14_unseen_check.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.build_e14_reword import REWORDS, SEEN_PAYLOADS  # noqa: E402


# train files label as strings ("safe"/"unsafe"); eval rows use 0/1. Reading the
# wrong convention silently yields an empty injection list and 0.0 coverage
# everywhere, so both are handled and the result is asserted non-empty.
UNSAFE = {1, "1", "unsafe", "injection"}


def is_injection(row: dict) -> bool:
    return row.get("label") in UNSAFE


def shingles(s: str, k: int = 5) -> set[str]:
    s = " ".join(s.split())
    return {s[i : i + k] for i in range(max(0, len(s) - k + 1))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify E14 arm-B wording is unseen")
    p.add_argument("--train", type=Path, default=Path("data/train_proposal/train_stage2.jsonl"))
    p.add_argument("--arms", type=Path, default=Path("data/eval_proposal/e14_reword.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/analysis/e14_unseen_check.json"))
    p.add_argument("--tail", type=int, default=80)
    p.add_argument("--k", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    train = [json.loads(line) for line in args.train.open()]
    train_inj = [r["input"] for r in train if is_injection(r)]
    train_ben = [r["input"] for r in train if not is_injection(r)]
    if not train_inj or not train_ben:
        raise SystemExit(
            f"{args.train.name}: parsed {len(train_inj)} injections / "
            f"{len(train_ben)} benign — check the label convention"
        )

    # 1. exact containment of each payload string in training text
    inj_blob = "\n".join(train_inj)
    ben_blob = "\n".join(train_ben)
    containment = {}
    for payload in SEEN_PAYLOADS:
        containment[payload] = {
            "arm": "seen",
            "in_train_injections": inj_blob.count(payload),
            "in_train_benign": ben_blob.count(payload),
        }
    for payload, variants in REWORDS.items():
        for vi, v in enumerate(variants):
            containment[v] = {
                "arm": "reword",
                "of": payload,
                "variant": vi,
                "in_train_injections": inj_blob.count(v),
                "in_train_benign": ben_blob.count(v),
            }

    # 2. tail 5-gram coverage against the training injection tail vocabulary
    train_tail_vocab: set[str] = set()
    for t in train_inj:
        train_tail_vocab |= shingles(t[-args.tail:], args.k)
    if not train_tail_vocab:
        raise SystemExit("empty training tail vocabulary")

    by_arm: dict[str, list[float]] = defaultdict(list)
    by_arm_payload: dict[tuple[str, str], list[float]] = defaultdict(list)
    for line in args.arms.open():
        r = json.loads(line)
        sh = shingles(r["text"][-args.tail:], args.k)
        cov = len(sh & train_tail_vocab) / len(sh) if sh else 0.0
        by_arm[r["arm"]].append(cov)
        by_arm_payload[(r["arm"], r["payload_class"])].append(cov)

    def summarise(vals: list[float]) -> dict:
        vals = sorted(vals)
        return {
            "n": len(vals),
            "mean": round(statistics.fmean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "frac_above_0.9": round(sum(v >= 0.9 for v in vals) / len(vals), 4),
        }

    reword_hits = sum(
        c["in_train_injections"] + c["in_train_benign"]
        for c in containment.values() if c["arm"] == "reword"
    )

    out = {
        "params": {"tail_chars": args.tail, "shingle_k": args.k, "train": str(args.train)},
        "n_train_injections": len(train_inj),
        "n_train_benign": len(train_ben),
        "reword_strings_found_in_training": reword_hits,
        "tail_coverage_by_arm": {a: summarise(v) for a, v in sorted(by_arm.items())},
        "tail_coverage_by_arm_payload": {
            f"{a}|{p}": summarise(v) for (a, p), v in sorted(by_arm_payload.items())
        },
        "containment": containment,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(json.dumps({
        "reword_strings_found_in_training": reword_hits,
        "tail_coverage_by_arm": out["tail_coverage_by_arm"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
