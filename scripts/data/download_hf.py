"""Cache the HackAPrompt submissions used by the development corpus.

Streams the first 20,000 rows of the public mirror
imoxto/prompt_injection_hackaprompt_gpt35 (pinned revision) into
data/raw/hf/hackaprompt.jsonl, keeping only the ``text`` field.
build_dev_corpus.py shuffles this file with seed 3131 and keeps 600 rows.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sources import HF_REVISIONS  # noqa: E402

HF_ID = "imoxto/prompt_injection_hackaprompt_gpt35"
OUT = ROOT / "data" / "raw" / "hf" / "hackaprompt.jsonl"
CAP = 20000


def main() -> int:
    if OUT.exists():
        print(f"[skip] {OUT} exists")
        return 0
    from datasets import load_dataset

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_ID, split="train", streaming=True, revision=HF_REVISIONS[HF_ID])
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for r in ds:
            if not r.get("text"):
                continue
            f.write(json.dumps({"text": r["text"], "_hf_id": HF_ID}, ensure_ascii=False) + "\n")
            n += 1
            if n >= CAP:
                break
    print(f"[ok] {OUT.name}: {n} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
