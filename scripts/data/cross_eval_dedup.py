"""Cross-eval near-duplicate filter (development corpus -> evaluation benchmark).

Removes eval records that are near-duplicates of any TRAINING record, so the
train/eval split is disjoint by content as well as by source. A record is
removed if, against its nearest training neighbours, EITHER:

  * character 5-gram Jaccard >= 0.5, OR
  * MiniLM (all-MiniLM-L6-v2) cosine similarity >= 0.95.

Text is normalised with ``src/data/preprocessing/dedup.py``.

Embeddings use HuggingFace ``transformers`` directly (mean pooling + L2
normalisation) so no ``sentence-transformers``/``faiss`` dependency is needed;
nearest neighbours are found by batched matrix multiplication.

Usage (from the repository root):
    python scripts/data/cross_eval_dedup.py
    python scripts/data/cross_eval_dedup.py --dry-run          # report only, no rewrite
    python scripts/data/cross_eval_dedup.py --cos 0.95 --jac 0.5 --neighbours 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Repository root on path so `src...` imports resolve.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing.dedup import _normalize  # noqa: E402

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COS_THRESHOLD = 0.95
JAC_THRESHOLD = 0.50
CHAR_K = 5
N_NEIGHBOURS = 10  # shortlist size for the char-gram check

TRAIN_FILES = ["train.jsonl", "val.jsonl", "cal.jsonl"]
TRAIN_DIR = ROOT / "data" / "train_proposal"
EVAL_PATH = ROOT / "data" / "eval_proposal" / "eval.jsonl"
REMOVED_PATH = ROOT / "data" / "eval_proposal" / "dedup_removed.jsonl"
MANIFEST_PATH = ROOT / "data" / "eval_proposal" / "build_manifest.json"


# --------------------------------------------------------------------------- #
# Text similarity primitives
# --------------------------------------------------------------------------- #
def char_shingles(norm: str, k: int = CHAR_K) -> frozenset[str]:
    """Character k-gram set over the normalised string."""
    if len(norm) < k:
        return frozenset([norm]) if norm else frozenset()
    return frozenset(norm[i : i + k] for i in range(len(norm) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


# --------------------------------------------------------------------------- #
# MiniLM embeddings via transformers (mean pooling + L2 normalise)
# --------------------------------------------------------------------------- #
def _load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    from src.data.sources import HF_REVISIONS
    rev = HF_REVISIONS[MODEL_NAME]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=rev)
    model = AutoModel.from_pretrained(MODEL_NAME, revision=rev)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device).eval()
    return tok, model, device


def embed_texts(texts: list[str], batch_size: int = 64, quiet: bool = False) -> np.ndarray:
    """Return an (N, D) float32 array of L2-normalised MiniLM embeddings."""
    import torch

    tok, model, device = _load_model()
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tok(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(device)
            hidden = model(**enc).last_hidden_state  # (B, T, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            emb = summed / counts
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out[start : start + len(batch)] = emb.cpu().numpy()
            if not quiet and (start // batch_size) % 20 == 0:
                print(f"    embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    return out


# --------------------------------------------------------------------------- #
# Char-gram MinHash-LSH candidate generation
# --------------------------------------------------------------------------- #
# The char-5gram Jaccard criterion must find ANY train record with Jaccard >= 0.5,
# not just those that happen to be embedding neighbours. We approximate an all-pairs
# Jaccard join with MinHash-LSH: sign each doc's char-shingle set, band the signatures,
# and treat docs sharing a band bucket as candidates (exact Jaccard verifies them).
_MH_PERM = 256
_MH_BANDS = 64
_MH_ROWS = 4            # perm = bands * rows; detection at J=0.5 ~= 0.98
_MH_PRIME = 4294967291  # largest prime < 2**32; keeps a*h+b < 2**64 (uint64-safe)


def _shingle_hashes(shingles: frozenset[str]) -> np.ndarray:
    import zlib

    if not shingles:
        return np.empty(0, dtype=np.uint64)
    return np.array([zlib.crc32(s.encode("utf-8")) for s in shingles], dtype=np.uint64)


def _minhash_signatures(shingle_sets: list[frozenset[str]], seed: int = 3131) -> np.ndarray:
    """(N, _MH_PERM) uint64 MinHash signatures over char-shingle hashes."""
    rs = np.random.RandomState(seed)
    a = rs.randint(1, _MH_PRIME, size=_MH_PERM).astype(np.uint64)
    b = rs.randint(0, _MH_PRIME, size=_MH_PERM).astype(np.uint64)
    sigs = np.full((len(shingle_sets), _MH_PERM), np.uint64(_MH_PRIME), dtype=np.uint64)
    for i, sh in enumerate(shingle_sets):
        h = _shingle_hashes(sh)
        if h.size == 0:
            continue
        # (perm, m) = (a[:,None]*h[None,:] + b[:,None]) % prime  -> min over shingles
        hashed = (a[:, None] * h[None, :] + b[:, None]) % np.uint64(_MH_PRIME)
        sigs[i] = hashed.min(axis=1)
    return sigs


def _lsh_candidates(
    eval_sigs: np.ndarray, train_sigs: np.ndarray
) -> dict[int, set[int]]:
    """Map each eval index to the set of train indices sharing >=1 LSH band bucket."""
    cand: dict[int, set[int]] = {}
    for band in range(_MH_BANDS):
        c0 = band * _MH_ROWS
        c1 = c0 + _MH_ROWS
        # bucket train sigs by their band tuple
        buckets: dict[bytes, list[int]] = {}
        for ti in range(train_sigs.shape[0]):
            key = train_sigs[ti, c0:c1].tobytes()
            buckets.setdefault(key, []).append(ti)
        for ei in range(eval_sigs.shape[0]):
            key = eval_sigs[ei, c0:c1].tobytes()
            hit = buckets.get(key)
            if hit:
                cand.setdefault(ei, set()).update(hit)
    return cand


# --------------------------------------------------------------------------- #
# Core screen
# --------------------------------------------------------------------------- #
def find_near_dups(
    train_texts: list[str],
    train_ids: list[str],
    eval_texts: list[str],
    cos_thr: float = COS_THRESHOLD,
    jac_thr: float = JAC_THRESHOLD,
    k: int = N_NEIGHBOURS,
    quiet: bool = False,
) -> list[dict]:
    """For each eval text, return a verdict dict:
       {eval_idx, remove, cos, jac, reason, match_train_id, match_train_idx}."""
    if not quiet:
        print(f"  Embedding {len(train_texts)} train + {len(eval_texts)} eval texts...")
    train_emb = embed_texts(train_texts, quiet=quiet)
    eval_emb = embed_texts(eval_texts, quiet=quiet)

    train_shingles = [char_shingles(_normalize(t)) for t in train_texts]
    eval_shingles = [char_shingles(_normalize(t)) for t in eval_texts]

    # Char-gram LSH: candidate train indices per eval index whose Jaccard may be >= 0.5,
    # independent of embedding rank (so lexically-similar/semantically-distant pairs are caught).
    if not quiet:
        print("  Building char-gram MinHash-LSH candidates...")
    train_sigs = _minhash_signatures(train_shingles)
    eval_sigs = _minhash_signatures(eval_shingles)
    lsh_cand = _lsh_candidates(eval_sigs, train_sigs)

    verdicts: list[dict] = []
    chunk = 256
    for start in range(0, len(eval_texts), chunk):
        block = eval_emb[start : start + chunk]  # (b, D)
        sims = block @ train_emb.T  # (b, N_train) cosine (rows already normalised)
        for r in range(block.shape[0]):
            ei = start + r
            row = sims[r]
            top1 = int(row.argmax())
            cos = float(row[top1])
            # char-gram Jaccard candidates = k nearest embedding neighbours U LSH candidates
            k_eff = min(k, row.shape[0])
            nn = np.argpartition(-row, k_eff - 1)[:k_eff]
            candidates = set(int(x) for x in nn) | lsh_cand.get(ei, set())
            e_sh = eval_shingles[ei]
            best_jac, best_jac_idx = 0.0, top1
            for ti in candidates:
                j = jaccard(e_sh, train_shingles[ti])
                if j > best_jac:
                    best_jac, best_jac_idx = j, ti
            remove = cos >= cos_thr or best_jac >= jac_thr
            reason = []
            if cos >= cos_thr:
                reason.append("cosine")
            if best_jac >= jac_thr:
                reason.append("char5gram_jaccard")
            match_idx = top1 if cos >= best_jac else best_jac_idx
            verdicts.append(
                {
                    "eval_idx": ei,
                    "remove": remove,
                    "cos": round(cos, 4),
                    "jac": round(best_jac, 4),
                    "reason": "+".join(reason) or None,
                    "match_train_id": train_ids[match_idx],
                    "match_train_idx": match_idx,
                }
            )
        if not quiet:
            print(f"    screened {min(start + chunk, len(eval_texts))}/{len(eval_texts)}")
    return verdicts


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="cross-eval near-duplicate filter")
    ap.add_argument("--cos", type=float, default=COS_THRESHOLD)
    ap.add_argument("--jac", type=float, default=JAC_THRESHOLD)
    ap.add_argument("--neighbours", type=int, default=N_NEIGHBOURS)
    ap.add_argument("--dry-run", action="store_true", help="report only; do not rewrite eval.jsonl")
    args = ap.parse_args()

    # Load train corpus (field `input`) with stable ids for provenance.
    train_texts: list[str] = []
    train_ids: list[str] = []
    for fname in TRAIN_FILES:
        p = TRAIN_DIR / fname
        if not p.exists():
            print(f"  WARNING: {p} missing, skipping")
            continue
        for i, row in enumerate(_read_jsonl(p)):
            t = row.get("input", "")
            if t:
                train_texts.append(t)
                train_ids.append(f"{fname.split('.')[0]}:{i}")
    print(f"Loaded {len(train_texts)} train records from {TRAIN_DIR}")

    eval_records = _read_jsonl(EVAL_PATH)
    eval_texts = [r["text"] for r in eval_records]
    print(f"Loaded {len(eval_records)} eval records from {EVAL_PATH}")

    verdicts = find_near_dups(
        train_texts, train_ids, eval_texts,
        cos_thr=args.cos, jac_thr=args.jac, k=args.neighbours,
    )

    removed = [v for v in verdicts if v["remove"]]
    kept_records = [eval_records[i] for i, v in enumerate(verdicts) if not v["remove"]]

    print(f"\n=== cross-eval dedup ===")
    print(f"  eval before : {len(eval_records)}")
    print(f"  removed     : {len(removed)}")
    print(f"  eval after  : {len(kept_records)}")
    by_reason: dict[str, int] = {}
    for v in removed:
        by_reason[v["reason"]] = by_reason.get(v["reason"], 0) + 1
    print(f"  by reason   : {by_reason}")

    if args.dry_run:
        print("\n  --dry-run: no files written.")
        return

    # Persist removed records with match provenance.
    removed_rows = []
    for v in removed:
        rec = dict(eval_records[v["eval_idx"]])
        rec["_dedup"] = {
            "cos": v["cos"], "jac": v["jac"], "reason": v["reason"],
            "match_train_id": v["match_train_id"],
        }
        removed_rows.append(rec)
    _write_jsonl(REMOVED_PATH, removed_rows)
    _write_jsonl(EVAL_PATH, kept_records)
    print(f"\n  Wrote {len(removed_rows)} removed -> {REMOVED_PATH}")
    print(f"  Rewrote {len(kept_records)} kept    -> {EVAL_PATH}")

    # Update manifest.
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest.pop("TODO_cross_eval_dedup", None)
    manifest["cross_eval_dedup"] = {
        "cos_threshold": args.cos,
        "jac_threshold": args.jac,
        "char_k": CHAR_K,
        "neighbours": args.neighbours,
        "model": MODEL_NAME,
        "n_before": len(eval_records),
        "n_removed": len(removed),
        "n_after": len(kept_records),
        "by_reason": by_reason,
    }
    manifest["total_records"] = len(kept_records)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Updated manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
