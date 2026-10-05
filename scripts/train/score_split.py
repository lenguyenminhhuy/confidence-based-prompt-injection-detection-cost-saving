"""Dump Stage-1 label logits for arbitrary split(s) from a SAVED adapter.

Decouples logit-dumping from training (train_stage1.py welds them together,
which overran the training wall-clock limit). Given an already-trained adapter,
this scores any split(s) (val, cal, eval) in a short GPU inference pass,
writing the same
per-row {logp_<label>, p_safe} format that score_stage1_logits.py reads.

PAYLOAD HYGIENE: reads split text only to feed the model; never
prints it. Emits only counts and paths.

    PYTHONPATH=. python scripts/train/score_split.py \
        --config configs/models/qwen2.5-1.5b.yaml \
        --adapter results/stage1/qwen2.5-1.5b/adapter \
        --output-dir results/stage1/qwen2.5-1.5b \
        --splits data/train_proposal/val.jsonl data/train_proposal/cal.jsonl \
                 data/eval_proposal/eval.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.prompt_template import load_model_config
from src.models.stage1 import Stage1Detector
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger

log = get_logger("score_split")


def _row_text(r: dict) -> str:
    # test splits store payload under 'rendered_input'; train/cal under 'input'.
    # Missing this fallback silently feeds empty strings -> constant p_safe garbage.
    return r.get("input") or r.get("text") or r.get("rendered_input") or ""


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump Stage-1 logits from a saved adapter")
    p.add_argument("--config", required=True,
                   help="model config: name or path to configs/models/<name>.yaml")
    p.add_argument("--adapter", type=Path, required=True, help="saved PEFT adapter dir")
    p.add_argument("--splits", nargs="+", type=Path, required=True,
                   help="split jsonl(s) to score -> <output-dir>/<stem>_logits.jsonl")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--no-4bit", action="store_true",
                   help="disable 4-bit; default matches training (4-bit on CUDA)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_model_config(args.config)
    device = auto_device()
    load_in_4bit = (device == "cuda") and not args.no_4bit
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("model=%s adapter=%s device=%s 4bit=%s",
             config.name, args.adapter, device, load_in_4bit)

    detector = Stage1Detector(
        config, adapter_path=args.adapter, device=device, load_in_4bit=load_in_4bit
    )
    # Sanity: the freshly loaded adapter yields a valid probability.
    probe = detector.p_safe("Please summarize the meeting notes for tomorrow.")
    assert 0.0 <= probe <= 1.0, f"p_safe out of range: {probe}"
    log.info("adapter loaded OK: p_safe(sample)=%.4f", probe)

    for split in args.splits:
        if not split.exists():
            log.warning("skip missing split: %s", split)
            continue
        texts = [_row_text(r) for r in read_jsonl(split)]  # text never printed
        n_empty = sum(1 for t in texts if not t)
        if n_empty:
            raise SystemExit(
                f"{split.name}: {n_empty}/{len(texts)} rows have empty text — "
                f"check payload field name (input/text/rendered_input) before scoring."
            )
        preds = detector.predict(texts)
        out = args.output_dir / f"{split.stem}_logits.jsonl"
        write_jsonl(out, preds)
        log.info("dumped %d logits -> %s", len(preds), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
