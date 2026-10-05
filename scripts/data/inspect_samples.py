"""Payload-safe dataset inspector: metadata only, never the text.

Prints one line per record — id/source/channel/label/family/length/sha256
prefix — so humans and AI agents can inspect any split without raw attack
payloads ever reaching a terminal, notebook output, or assistant context
(raw jailbreak text in an AI-assistant session can trip provider safety
filters).

Usage:
    PYTHONPATH=. python scripts/data/inspect_samples.py data/train_proposal/train.jsonl
    PYTHONPATH=. python scripts/data/inspect_samples.py data/eval_proposal/eval.jsonl \
        --label injected --channel document_embedded --limit 20
    PYTHONPATH=. python scripts/data/inspect_samples.py data/train_proposal/*.jsonl --summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

# Fields that may hold the record text, in priority order. Only used for
# length/hash — never printed.
TEXT_FIELDS = ("rendered_input", "text", "input", "prompt", "content")
HASH_PREFIX = 12


def record_text(rec: dict) -> str:
    for f in TEXT_FIELDS:
        v = rec.get(f)
        if isinstance(v, str) and v:
            return v
    return ""


def meta(rec: dict, idx: int) -> dict:
    txt = record_text(rec)
    return {
        "id": rec.get("id") or rec.get("sample_id") or f"row{idx}",
        "source": rec.get("source", "?"),
        "channel": rec.get("channel", "?"),
        "label": rec.get("label", "?"),
        "family": rec.get("payload_family") or rec.get("family") or "-",
        "len": len(txt),
        "sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest()[:HASH_PREFIX],
    }


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="jsonl file(s) to inspect")
    ap.add_argument("--label", help="filter by label")
    ap.add_argument("--channel", help="filter by channel")
    ap.add_argument("--source", help="filter by source")
    ap.add_argument("--limit", type=int, default=50, help="max rows per file (default 50)")
    ap.add_argument("--summary", action="store_true",
                    help="print per-file aggregate counts instead of per-row lines")
    args = ap.parse_args()

    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"MISSING: {path}", file=sys.stderr)
            continue
        print(f"\n=== {path} ===")
        shown, total = 0, 0
        labels, channels, sources = Counter(), Counter(), Counter()
        for idx, rec in iter_jsonl(path):
            m = meta(rec, idx)
            total += 1
            labels[m["label"]] += 1
            channels[m["channel"]] += 1
            sources[m["source"]] += 1
            if args.summary:
                continue
            if args.label and m["label"] != args.label:
                continue
            if args.channel and m["channel"] != args.channel:
                continue
            if args.source and m["source"] != args.source:
                continue
            if shown < args.limit:
                print(f"{m['id']:>28s} | {m['source']:<18s} | {m['channel']:<18s} "
                      f"| {m['label']:<9s} | {m['family']:<16s} "
                      f"| len={m['len']:5d} | sha256={m['sha256']}")
                shown += 1
        if args.summary:
            print(f"rows={total}")
            print(f"labels   : {dict(labels)}")
            print(f"channels : {dict(channels)}")
            print(f"sources  : {dict(sources.most_common(15))}")
        else:
            print(f"({shown} shown / {total} total; metadata only — text is never printed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
