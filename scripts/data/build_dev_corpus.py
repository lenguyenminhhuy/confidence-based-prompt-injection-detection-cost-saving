"""Build the development corpus: data/train_proposal/{train,val,cal}.jsonl.

Sources and channels (no source is shared with the evaluation benchmark):

  conversational          safe    stingning/ultrachat (first user turn), google/IFEval
  application-structured  safe    leolee99/NotInject, tatsu-lab/alpaca
  document-embedded       unsafe  BIPIA contexts with planted BIPIA attack payloads
  tool-output             unsafe  InjecAgent tool-response templates
  direct                  unsafe  HackAPrompt submissions, StruQ-style templates on
                                  Alpaca/NotInject carriers (training link phrases only)

Injections above MAX_TOKENS whitespace tokens are dropped. 30% of injections get
three encoded variants (base64, ROT13, homoglyph) as source "synthetic-encoded",
kept in the same split as their original. Exact duplicates (after lower-casing and
stripping punctuation) are removed, then groups are split 80/10/10.

Seeds: 3131 for per-source sampling, 42 for the split and the link-phrase partition.
The link-phrase partition is shared with build_eval_benchmark.py.

Run from the repository root after scripts/data/download_data.sh:
    python scripts/data/build_dev_corpus.py
"""

from __future__ import annotations

import argparse
import base64
import codecs
import collections
import hashlib
import json
import os
import random
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.link_phrases import GOAL_PHRASES, FAKE_COMPLETIONS, link_partition  # noqa: E402
from src.data.sources import HF_REVISIONS  # noqa: E402

SEED_SAMPLE = 3131
SEED_SPLIT = 42
MAX_TOKENS = 512

CAPS = {
    "smoke": {"conv": 100, "app": 100, "doc": 100, "tool": 100, "direct_hap": 50, "direct_struq": 50},
    "full": {"conv": 10000, "app": 17000, "doc": 10000, "tool": 2000, "direct_hap": 600, "direct_struq": 4000},
}

# Mirror used for HackAPrompt; scripts/data/download_hf.py writes this file.
HACKAPROMPT_FILE = "hf/hackaprompt.jsonl"

VALID_CHANNELS = {"conversational", "application-structured", "document-embedded", "tool-output", "direct"}
SOURCES = {"ultrachat", "ifeval", "notinject", "alpaca", "bipia", "injecagent",
           "hackaprompt", "struq_generated", "synthetic-encoded"}


class Builder:
    def __init__(self, raw: Path, caps: dict):
        self.raw = raw
        self.cap = caps
        self.rng = random.Random(SEED_SAMPLE)
        self.records: list[dict] = []
        self.sources: dict = {}

    def add(self, text, label, channel, source) -> bool:
        assert label in ("safe", "unsafe"), label
        assert channel in VALID_CHANNELS, channel
        s = (text or "").strip()
        if len(s) < 5:
            return False
        self.records.append({"input": s, "label": label, "channel": channel, "source": source})
        return True

    def note(self, name, channel, n):
        self.sources[name] = {"n": n, "channel": channel, "fallback": None}
        print(f"  + {name:22s} {channel:22s} n={n}")

    # ------------------------------------------------------------------ benign
    def conversational(self):
        from datasets import load_dataset
        cap = self.cap["conv"]
        ifeval_cap = max(1, cap // 10)
        ultrachat_cap = cap - ifeval_cap

        got = 0
        ds = load_dataset("stingning/ultrachat", split="train", streaming=True,
                          revision=HF_REVISIONS["stingning/ultrachat"])
        for ex in ds:
            data = ex.get("data") or []
            if not data:
                continue
            first_user = data[0] if isinstance(data[0], str) else data[0].get("content", "")
            if self.add(first_user, "safe", "conversational", "ultrachat"):
                got += 1
            if got >= ultrachat_cap:
                break
        self.note("ultrachat", "conversational", got)

        n = 0
        for ex in load_dataset("google/IFEval", split="train", revision=HF_REVISIONS["google/IFEval"]):
            if self.add(ex.get("prompt", ""), "safe", "conversational", "ifeval"):
                n += 1
            if n >= ifeval_cap:
                break
        self.note("ifeval", "conversational", n)

    def app_structured(self):
        from datasets import load_dataset
        cap = self.cap["app"]
        ni_cap = max(1, cap // 3)
        ni_got = 0
        for split in ("NotInject_one", "NotInject_two", "NotInject_three"):
            if ni_got >= ni_cap:
                break
            for ex in load_dataset("leolee99/NotInject", split=split,
                                   revision=HF_REVISIONS["leolee99/NotInject"]):
                if self.add(ex.get("prompt", ""), "safe", "application-structured", "notinject"):
                    ni_got += 1
                if ni_got >= ni_cap:
                    break
        self.note("notinject", "application-structured", ni_got)

        alpaca_cap = max(0, cap - ni_got)
        ds = load_dataset("tatsu-lab/alpaca", split="train", revision=HF_REVISIONS["tatsu-lab/alpaca"])
        idx = list(range(len(ds)))
        self.rng.shuffle(idx)
        n = 0
        for i in idx:
            ex = ds[i]
            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            text = f"{instr}\n\n{inp}" if inp else instr
            if self.add(text, "safe", "application-structured", "alpaca"):
                n += 1
            if n >= alpaca_cap:
                break
        self.note("alpaca", "application-structured", n)

    # -------------------------------------------------------------- injections
    def bipia_document(self):
        cap = self.cap["doc"]
        bipia_dir = self.raw / "BIPIA" / "benchmark"
        payloads = []
        for atk_name in ("text_attack_test.json", "text_attack_train.json"):
            atk = json.load(open(bipia_dir / atk_name))
            for _cat, items in atk.items():
                payloads.extend([p for p in items if isinstance(p, str) and p.strip()])
        payloads = sorted(set(payloads))

        contexts = []
        for task in ("email", "table", "code"):
            for split_file in ("test.jsonl", "train.jsonl"):
                p = bipia_dir / task / split_file
                if not p.is_file():
                    continue
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        raw_ctx = d.get("context")
                        ctx = ("\n".join(str(x) for x in raw_ctx) if isinstance(raw_ctx, list)
                               else str(raw_ctx or "")).strip()
                        q = str(d.get("question") or "").strip()
                        if ctx:
                            contexts.append((ctx, q, task))
        assert payloads and contexts, f"BIPIA clone incomplete under {bipia_dir}"

        def plant(ctx, payload, pos):
            if pos == "end":
                return f"{ctx}\n{payload}"
            if pos == "start":
                return f"{payload}\n{ctx}"
            mid = len(ctx) // 2
            return f"{ctx[:mid]}\n{payload}\n{ctx[mid:]}"

        positions = ["end", "start", "middle"]
        n_ctx, n_pay = len(contexts), len(payloads)
        max_triples = n_ctx * n_pay * len(positions)
        seen, got, attempts = set(), 0, 0
        while got < cap and len(seen) < max_triples and attempts < cap * 20:
            attempts += 1
            ci = self.rng.randrange(n_ctx)
            pi = self.rng.randrange(n_pay)
            pj = self.rng.randrange(len(positions))
            key = (ci, pi, pj)
            if key in seen:
                continue
            seen.add(key)
            ctx, q, _task = contexts[ci]
            poisoned = plant(ctx, payloads[pi], positions[pj])
            combined = f"{poisoned}\n\n{q}".strip() if q else poisoned
            if self.add(combined, "unsafe", "document-embedded", "bipia"):
                got += 1
        self.note("bipia", "document-embedded", got)

    def injecagent_tool(self):
        cap = self.cap["tool"]
        ia_dir = self.raw / "InjecAgent" / "data"
        files = [ia_dir / f for f in ("test_cases_dh_base.json", "test_cases_ds_base.json",
                                      "test_cases_dh_enhanced.json", "test_cases_ds_enhanced.json")
                 if (ia_dir / f).is_file()]
        assert files, f"InjecAgent clone missing under {ia_dir}"
        got = 0
        for path in files:
            if got >= cap:
                break
            for c in json.load(open(path)):
                tmpl = c.get("Tool Response Template") or c.get("Tool Response") or ""
                instr = c.get("Attacker Instruction") or ""
                tool_txt = tmpl.replace("<Attacker Instruction>", instr) if tmpl else instr
                if self.add(tool_txt, "unsafe", "tool-output", "injecagent"):
                    got += 1
                if got >= cap:
                    break
        self.note("injecagent", "tool-output", got)

    def hackaprompt_direct(self):
        cap = self.cap["direct_hap"]
        local = self.raw / HACKAPROMPT_FILE
        assert local.is_file(), f"{local} missing; run scripts/data/download_hf.py"
        rows = [json.loads(line) for line in open(local) if line.strip()]
        self.rng.shuffle(rows)
        got = 0
        for d in rows:
            txt = d.get("prompt") or d.get("user_input") or d.get("rendered_input") or d.get("text") or ""
            if self.add(txt, "unsafe", "direct", "hackaprompt"):
                got += 1
            if got >= cap:
                break
        self.note("hackaprompt", "direct", got)

    def struq_direct(self, train_links, eval_links):
        cap = self.cap["direct_struq"]
        r = random.Random(SEED_SPLIT)  # same stream as the partition; only used by t_fake
        link_partition(r)              # advance it exactly as the original build did

        def t_naive(c, l, p):  return f"{c} {l} {p}"
        def t_ignore(c, l, p): return f"{c} Ignore previous instructions. {l} {p}"
        def t_escape(c, l, p): return f"{c}\n\n]}}]}}\n\n{l} {p}"
        def t_fake(c, l, p):   return f"{c} {r.choice(FAKE_COMPLETIONS)} {l} {p}"
        templates = [t_naive, t_ignore, t_escape, t_fake]

        carriers = [x["input"] for x in self.records
                    if x["label"] == "safe" and x["channel"] == "application-structured"]
        assert carriers, "no application-structured carriers"
        rr = random.Random(SEED_SPLIT + 1)
        got, i = 0, 0
        while got < cap:
            c = carriers[i % len(carriers)][:400]
            l = rr.choice(train_links)
            p = rr.choice(GOAL_PHRASES)
            t = rr.choice(templates)
            assert l not in eval_links, "training template drew an evaluation link phrase"
            if self.add(t(c, l, p), "unsafe", "direct", "struq_generated"):
                got += 1
            i += 1
        self.note("struq_generated", "direct", got)


def encode_variants(payload: str, seed: int) -> dict:
    rr = random.Random(seed)
    out = {"b64": base64.b64encode(payload.encode()).decode(),
           "rot13": codecs.encode(payload, "rot_13")}
    hg = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х"}
    chars = list(payload)
    for j, ch in enumerate(chars):
        if ch.lower() in hg and rr.random() < 0.3:
            chars[j] = hg[ch.lower()]
    out["homoglyph"] = "".join(chars)
    return out


def normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()


def write_jsonl(path: Path, rows):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def dist(rows):
    c, s, l = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        c[r["channel"]] += 1
        s[r["source"]] += 1
        l[r["label"]] += 1
    return {"by_channel": dict(c), "by_source": dict(s), "by_label": dict(l)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=sorted(CAPS), default="full")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train_proposal")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    b = Builder(args.raw, CAPS[args.mode])
    b.conversational()
    b.app_structured()
    b.bipia_document()
    b.injecagent_tool()
    b.hackaprompt_direct()
    train_links, eval_links = link_partition(random.Random(SEED_SPLIT))
    b.struq_direct(train_links, set(eval_links))

    # Truncation budget: drop over-long injections (payload is often at the end).
    kept, dropped = [], 0
    for r in b.records:
        toks = r["input"].split()
        if len(toks) <= MAX_TOKENS:
            kept.append(r)
        elif r["label"] == "unsafe":
            dropped += 1
        else:
            r["input"] = " ".join(toks[:MAX_TOKENS])
            kept.append(r)
    records = kept
    print(f"dropped {dropped} injections over {MAX_TOKENS} tokens; {len(records)} remain")
    for i, r in enumerate(records):
        r["base_idx"] = i

    # Encoded variants of 30% of injections, grouped with their original.
    inj = [r for r in records if r["label"] == "unsafe"]
    rr = random.Random(SEED_SPLIT + 7)
    sample = [r for r in inj if rr.random() < 0.30]
    aug = []
    for r in sample:
        seed = int(hashlib.md5(r["input"].encode()).hexdigest(), 16) % (2 ** 32)
        for txt in encode_variants(r["input"], seed).values():
            aug.append({"input": txt, "label": "unsafe", "channel": r["channel"],
                        "source": "synthetic-encoded", "base_idx": r["base_idx"]})
    records += aug
    print(f"added {len(aug)} encoded variants -> {len(records)}")

    seen, dedup = set(), []
    for r in records:
        key = (r["label"], normalize_text(r["input"]))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    print(f"exact duplicates removed: {len(records) - len(dedup)}")
    records = dedup

    groups = defaultdict(list)
    for r in records:
        groups[r["base_idx"]].append(r)
    gids = list(groups)
    random.Random(SEED_SPLIT).shuffle(gids)
    n = len(gids)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    split_of = {g: ("train" if gi < n_tr else "val" if gi < n_tr + n_va else "cal")
                for gi, g in enumerate(gids)}
    buckets = {"train": [], "val": [], "cal": []}
    for r in records:
        buckets[split_of[r["base_idx"]]].append({k: v for k, v in r.items() if k != "base_idx"})
    for k in buckets:
        random.Random(SEED_SPLIT).shuffle(buckets[k])

    # Integrity: only development sources, no text in two splits.
    all_sources = {r["source"] for rows in buckets.values() for r in rows}
    assert all_sources <= SOURCES, all_sources - SOURCES
    split_count = collections.Counter((r["label"], r["input"]) for rows in buckets.values() for r in rows)
    assert all(v == 1 for v in split_count.values()), "text appears in more than one split"

    for name, rows in buckets.items():
        write_jsonl(args.out / f"{name}.jsonl", rows)
    manifest = {
        "run_mode": args.mode, "seed_sample": SEED_SAMPLE, "seed_split": SEED_SPLIT,
        "max_tokens": MAX_TOKENS,
        "n_total": sum(len(v) for v in buckets.values()),
        "splits": {k: {"n": len(v), **dist(v)} for k, v in buckets.items()},
        "sources": b.sources,
        "truncation": {"dropped_unsafe_over_budget": dropped},
        "link_phrase_partition": {"train_links": len(train_links), "eval_links_reserved": len(eval_links)},
    }
    with open(args.out / "build_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print({k: len(v) for k, v in buckets.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
