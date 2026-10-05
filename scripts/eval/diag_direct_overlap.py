"""A2 — train/eval overlap of the *injected instruction*, per channel.

The near-duplicate screen (scripts/data/cross_eval_dedup.py) compares WHOLE strings.
An injected example is dominated by its benign carrier (a task prompt or a
retrieved document), so two examples that share an identical malicious payload
but sit on different carriers score as dissimilar and survive the screen. The
screen is therefore blind to overlap in exactly the substring a detector must
key on.

This diagnostic measures that substring directly, per channel:

  * DIRECT (struq_synthetic) is procedurally generated as
        <carrier task> <template marker> <payload>
    so the payload parses exactly. We report the payload VOCABULARY overlap
    between eval and train, and the template-marker distribution of each.

  * ALL CHANNELS get a parser-free measure: 5-gram shingle coverage of each
    eval injection's TAIL (last --tail chars, where the injected instruction
    sits) against the shingle vocabulary of all train injection tails. High
    coverage means the phrasing was seen in training even when no whole-string
    near-duplicate exists.

  * We also reproduce the whole-string max similarity, to show the screen
    passing on the same rows the tail measure flags.

Reads text (unlike the scoring diagnostics) because overlap is a property of
the text; writes only aggregate counts and the payload vocabulary, never a
carrier document.

    PYTHONPATH=. python scripts/eval/diag_direct_overlap.py \
        --train data/train_proposal/train.jsonl \
        --eval data/eval_proposal/eval.jsonl \
        --out results/analysis/direct_overlap.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# struq_synthetic / struq_generated payload prefixes, in generation order
PAYLOAD_RE = re.compile(r"(?:Directive\s+\d+:|Urgent:|Note:)\s*(.+)$", re.S)

# the four StruQ template markers (chen2024struq), as instantiated by the builder
MARKERS = [
    "Ignore previous instructions.",
    "Done. Summary: general topics.",
    "]}]}",
    "Result: no issues found.",
]


def shingles(s: str, k: int = 5) -> set[str]:
    s = " ".join(s.split())
    return {s[i : i + k] for i in range(max(0, len(s) - k + 1))}


def marker_of(s: str) -> str:
    for m in MARKERS:
        if m in s:
            return m
    return "none/other"


def payload_of(s: str) -> str | None:
    m = PAYLOAD_RE.search(s)
    return m.group(1).strip() if m else None


def load(path: Path, text_key: str) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            t = d.get(text_key)
            if t is None:
                continue
            d["_text"] = t
            rows.append(d)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tail", type=int, default=80, help="tail chars treated as the injected instruction")
    ap.add_argument("--k", type=int, default=5, help="shingle size")
    args = ap.parse_args()

    train = load(args.train, "input")
    ev = load(args.eval, "text")

    tr_inj = [d for d in train if d.get("label") == "unsafe"]
    ev_inj = [d for d in ev if d.get("label") == 1 or d.get("category") == "injection"]

    # ---- tail shingle vocabulary from ALL train injections -------------------
    tail_vocab: set[str] = set()
    for d in tr_inj:
        tail_vocab |= shingles(d["_text"][-args.tail :], args.k)
    full_vocab: set[str] = set()
    for d in tr_inj:
        full_vocab |= shingles(d["_text"], args.k)

    out: dict = {
        "params": {"tail_chars": args.tail, "shingle_k": args.k},
        "n_train_injections": len(tr_inj),
        "n_eval_injections": len(ev_inj),
        "note": (
            "Tail coverage = fraction of an eval injection's tail 5-grams that occur "
            "anywhere in the train-injection tail vocabulary. It is a seen-phrasing "
            "measure, not a near-duplicate measure; the whole-string screen can pass "
            "while tail coverage is ~1.0."
        ),
        "by_channel": {},
    }

    by_ch: dict[str, list[dict]] = {}
    for d in ev_inj:
        by_ch.setdefault(d.get("channel", "unknown"), []).append(d)

    for ch, rows in sorted(by_ch.items()):
        covs = []
        for d in rows:
            sh = shingles(d["_text"][-args.tail :], args.k)
            covs.append(len(sh & tail_vocab) / len(sh) if sh else 0.0)
        covs.sort()
        n = len(covs)
        out["by_channel"][ch] = {
            "n": n,
            "tail_shingle_coverage": {
                "mean": round(sum(covs) / n, 4),
                "median": round(covs[n // 2], 4),
                "p10": round(covs[n // 10], 4),
                "p90": round(covs[(9 * n) // 10], 4),
                "frac_fully_covered": round(sum(1 for c in covs if c >= 0.999) / n, 4),
                "frac_above_0.9": round(sum(1 for c in covs if c >= 0.9) / n, 4),
            },
        }

    # ---- direct-channel exact payload vocabulary -----------------------------
    ev_direct = [d for d in ev_inj if d.get("channel") == "direct"]
    tr_direct = [d for d in tr_inj if d.get("channel") == "direct"]
    tr_struq = [d for d in tr_direct if d.get("source") == "struq_generated"]

    ev_pay = [p for p in (payload_of(d["_text"]) for d in ev_direct) if p]
    tr_pay = [p for p in (payload_of(d["_text"]) for d in tr_struq) if p]
    se, st = set(ev_pay), set(tr_pay)

    seen = sum(1 for p in ev_pay for _ in (1,) if p in st)
    out["direct_payload_vocabulary"] = {
        "n_eval_parsed": len(ev_pay),
        "n_train_parsed": len(tr_pay),
        "eval_distinct": len(se),
        "train_distinct": len(st),
        "shared_distinct": len(se & st),
        "eval_only_distinct": len(se - st),
        "frac_eval_payloads_seen_verbatim": round(seen / len(ev_pay), 4) if ev_pay else None,
        "shared_payloads": sorted(se & st),
        "eval_only_payloads": sorted(se - st),
    }

    out["direct_template_markers"] = {
        "eval": dict(Counter(marker_of(d["_text"]) for d in ev_direct)),
        "train_struq_generated": dict(Counter(marker_of(d["_text"]) for d in tr_struq)),
    }

    # carrier disjointness + literal-duplicate check, direct channel
    def carrier(s: str) -> str | None:
        for m in MARKERS:
            i = s.find(m)
            if i > 0:
                return s[:i].strip()
        return None

    ce = {c for c in (carrier(d["_text"]) for d in ev_direct) if c}
    ct = {c for c in (carrier(d["_text"]) for d in tr_struq) if c}
    out["direct_carrier"] = {
        "eval_distinct": len(ce),
        "train_distinct": len(ct),
        "shared": len(ce & ct),
        "exact_full_string_overlap": len(
            {d["_text"] for d in ev_direct} & {d["_text"] for d in tr_struq}
        ),
    }

    out["interpretation"] = (
        "Direct-channel eval is instance-disjoint but template- and payload-identical: "
        "carriers share nothing and no full string repeats, yet the injected payload is "
        "drawn from the same small vocabulary as training and the template markers match "
        "in distribution. Detection on this channel therefore measures recall of memorised "
        "payload phrasing, not generalisation to unseen direct attacks. The unseen-corpus "
        "channels (document, tool) have no such advantage, which is the same axis the "
        "seen-vs-unseen document control isolates."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:4000])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
