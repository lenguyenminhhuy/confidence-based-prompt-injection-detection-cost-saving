"""Build the stage-2 augmented training set.

WHY. The stage-2 fine-tune trains on data/train_proposal/train.jsonl, whose
BENIGN rows are short (len p50=72, p90=130; only 0.17% >=1k chars) while its
INJECTION rows are long (p50=868). A classifier fit on that learns the spurious
shortcut "long text => injection". At eval, 30% of benign is long (p50=219,
p90=2336) so it gets flagged (the same long-benign false-positive failure seen
for DataSentinel-7B).
`channel` is NOT a model feature (src/models/prompt_template.py scores only
"Text:\n{text}\n\nLabel:"), so the fix is purely the benign TEXT-LENGTH
distribution.

WHAT. Add ~5k LONG (1k-5k char) instruction-bearing benign from document-
summarization corpora that are provenance-disjoint from every eval source
(eval uses lmsys/dolly/natural_instructions; train already uses
alpaca/ultrachat) — rendered as "Summarize ...\n<doc>". Screen every
candidate against data/eval_proposal/eval.jsonl with the cross-eval near-duplicate filter
(scripts/data/cross_eval_dedup.find_near_dups: char-5gram Jaccard>=0.50 OR MiniLM
cosine>=0.95) and drop collisions. Concatenate with train and write
train_stage2.jsonl + a manifest.

PAYLOAD HYGIENE: this script NEVER prints raw dataset text — only
counts, lengths, hashes. Do not add prints of `input`/`text`/`document`.

Usage (from the repository root):
    python scripts/data/build_stage2_trainset.py                # full build
    python scripts/data/build_stage2_trainset.py --target 5000  # tune size
    python scripts/data/build_stage2_trainset.py --no-cosine    # Jaccard-only screen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts" / "data") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "data"))

TRAIN_PATH = ROOT / "data" / "train_proposal" / "train.jsonl"
EVAL_PATH = ROOT / "data" / "eval_proposal" / "eval.jsonl"
OUT_PATH = ROOT / "data" / "train_proposal" / "train_stage2.jsonl"
MANIFEST_PATH = ROOT / "data" / "train_proposal" / "stage2_aug_manifest.json"

SEED = 3131
MIN_LEN, MAX_LEN = 1000, 5000            # target the eval long-benign band
COS_THR, JAC_THR = 0.95, 0.50            # same thresholds as cross_eval_dedup.py
BENIGN_LABEL = "safe"
BENIGN_CHANNEL = "document-embedded"     # matches existing train channel spelling

# Eval-disjoint long document-summarization sources. `render` returns the
# benign instruction text; None-returns are skipped. govreport is truncated
# because its raw docs (p50~45k) far exceed the eval band.
SOURCES = [
    {
        "name": "cnn_dm_sum",
        "hf": "abisee/cnn_dailymail",
        "config": "3.0.0",
        "split": "train",
        "weight": 0.70,
        "render": lambda ex: (
            "Summarize the following news article:\n" + ex["article"]
            if ex.get("article") else None
        ),
    },
    {
        "name": "govreport_sum",
        "hf": "ccdv/govreport-summarization",
        "config": "document",
        "split": "train",
        "weight": 0.30,
        "render": lambda ex: (
            "Summarize the following government report:\n"
            + (ex.get("report") or ex.get("document") or "")[:3500]
            if (ex.get("report") or ex.get("document")) else None
        ),
    },
]


def pct(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return int(xs[f] + (xs[c] - xs[f]) * (k - f))


def buckets(lens):
    b = Counter()
    for n in lens:
        if n < 200:      b["<200"] += 1
        elif n < 500:    b["200-500"] += 1
        elif n < 1000:   b["500-1k"] += 1
        elif n < 2000:   b["1k-2k"] += 1
        elif n < 5000:   b["2k-5k"] += 1
        else:            b[">5k"] += 1
    return {k: b[k] for k in ["<200", "200-500", "500-1k", "1k-2k", "2k-5k", ">5k"]}


def benign_stats(rows):
    lens = [len(r.get("input") or r.get("text") or "") for r in rows
            if (r.get("label") in (BENIGN_LABEL, 0, "0", "benign"))]
    return {
        "n_benign": len(lens),
        "len_p50": pct(lens, 0.50), "len_p90": pct(lens, 0.90),
        "len_p99": pct(lens, 0.99), "len_max": max(lens) if lens else 0,
        "buckets": buckets(lens),
    }


def load_jsonl(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def collect_candidates(target, quiet=False):
    """Stream sources, render, keep len in [MIN_LEN, MAX_LEN]. Oversample 1.5x
    to survive the dedup screen. Returns list of {input, source}."""
    from datasets import load_dataset
    from src.data.sources import HF_REVISIONS
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    rng = random.Random(SEED)
    out = []
    for s in SOURCES:
        want = int(target * s["weight"] * 1.5) + 50
        got = 0
        ds = load_dataset(s["hf"], s["config"], split=s["split"], streaming=True,
                          revision=HF_REVISIONS[s["hf"]])
        for ex in ds:
            txt = s["render"](ex)
            if not txt:
                continue
            n = len(txt)
            if n < MIN_LEN or n > MAX_LEN:
                continue
            out.append({"input": txt, "source": s["name"]})
            got += 1
            if got >= want:
                break
        if not quiet:
            print(f"  collected {got:5d} from {s['name']} "
                  f"({s['hf']} [{s['config']}])", flush=True)
    rng.shuffle(out)
    return out


def screen_against_eval(candidates, eval_rows, use_cosine=True, quiet=False):
    """Drop candidates that near-duplicate any eval row. Uses
    cross_eval_dedup.find_near_dups with EVAL as the reference set. Returns
    (kept_rows, n_removed, legs_used)."""
    import cross_eval_dedup as ced

    cand_texts = [c["input"] for c in candidates]
    eval_texts = [r.get("text") or r.get("input") or "" for r in eval_rows]
    eval_ids = [r.get("id", f"eval:{i}") for i, r in enumerate(eval_rows)]
    cos = COS_THR if use_cosine else 10.0   # 10.0 disables the cosine leg

    verdicts = ced.find_near_dups(
        train_texts=eval_texts, train_ids=eval_ids,
        eval_texts=cand_texts, cos_thr=cos, jac_thr=JAC_THR,
        k=10, quiet=quiet,
    )
    remove_idx = {v["eval_idx"] for v in verdicts if v.get("remove")}
    kept = [c for i, c in enumerate(candidates) if i not in remove_idx]
    legs = "jaccard+cosine" if use_cosine else "jaccard-only"
    return kept, len(remove_idx), legs


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000,
                    help="approx number of long benign rows to ADD")
    ap.add_argument("--no-cosine", action="store_true",
                    help="Jaccard-only screen (skip MiniLM embeddings)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print(f"[1/5] loading train ({TRAIN_PATH.name}) + eval ({EVAL_PATH.name})...",
          flush=True)
    train = load_jsonl(TRAIN_PATH)
    eval_rows = load_jsonl(EVAL_PATH)
    before = benign_stats(train)
    print(f"      train rows={len(train)}  benign p50={before['len_p50']} "
          f"p90={before['len_p90']}  eval rows={len(eval_rows)}", flush=True)

    print(f"[2/5] streaming + length-filtering candidates "
          f"(band {MIN_LEN}-{MAX_LEN} chars)...", flush=True)
    candidates = collect_candidates(args.target, quiet=args.quiet)
    print(f"      {len(candidates)} candidates before dedup", flush=True)

    print(f"[3/5] dedup screen vs eval "
          f"(cos>={COS_THR}, jac>={JAC_THR})...", flush=True)
    kept, n_removed, legs = screen_against_eval(
        candidates, eval_rows, use_cosine=not args.no_cosine, quiet=args.quiet)
    print(f"      removed {n_removed} eval-colliding; {len(kept)} survive "
          f"({legs})", flush=True)

    # trim to target, keeping source proportions via the pre-shuffled order
    added = kept[: args.target]
    aug_rows = [{"input": r["input"], "label": BENIGN_LABEL,
                 "channel": BENIGN_CHANNEL, "source": r["source"]} for r in added]

    print(f"[4/5] writing {OUT_PATH.name} (train {len(train)} + aug "
          f"{len(aug_rows)})...", flush=True)
    tmp = OUT_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for r in train:
            out.write(json.dumps(r) + "\n")
        for r in aug_rows:
            out.write(json.dumps(r) + "\n")
    tmp.replace(OUT_PATH)

    combined = train + aug_rows
    after = benign_stats(combined)
    added_by_source = Counter(r["source"] for r in aug_rows)
    added_lens = [len(r["input"]) for r in aug_rows]

    manifest = {
        "task": "build-stage2-trainset",
        "seed": SEED,
        "length_band": [MIN_LEN, MAX_LEN],
        "dedup": {"cos_thr": COS_THR, "jac_thr": JAC_THR, "legs": legs,
                  "removed_eval_collisions": n_removed,
                  "candidates_before": len(candidates)},
        "sources": [{"name": s["name"], "hf": s["hf"], "config": s["config"],
                     "weight": s["weight"]} for s in SOURCES],
        "added": {
            "n_total": len(aug_rows),
            "by_source": dict(added_by_source),
            "label": BENIGN_LABEL, "channel": BENIGN_CHANNEL,
            "len_p50": pct(added_lens, 0.50), "len_p90": pct(added_lens, 0.90),
            "len_max": max(added_lens) if added_lens else 0,
            "buckets": buckets(added_lens),
        },
        "benign_before": before,
        "benign_after": after,
        "counts": {
            "train_in": len(train),
            "train_out": len(combined),
            "injection_untouched": sum(
                1 for r in train if r.get("label") not in
                (BENIGN_LABEL, 0, "0", "benign")),
        },
        "output": str(OUT_PATH.relative_to(ROOT)),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    manifest["output_sha256"] = sha256_file(OUT_PATH)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"[5/5] DONE. benign p90 {before['len_p90']} -> {after['len_p90']}; "
          f">=1k benign {before['buckets']['1k-2k']+before['buckets']['2k-5k']+before['buckets']['>5k']} "
          f"-> {after['buckets']['1k-2k']+after['buckets']['2k-5k']+after['buckets']['>5k']}", flush=True)
    print(f"      wrote {OUT_PATH.name} + {MANIFEST_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
