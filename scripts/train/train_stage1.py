"""Experiment 2: QLoRA fine-tune a Stage-1 (or Stage-2) candidate.

Fine-tunes a config-driven causal LM into the constrained-label classifier that
src/models/stage1.py scores (labels ``(benign, injection)``, benign first). One
script trains every E2 stage-1 candidate (configs/models/*.yaml) and, at 7B on
the augmented data, the Stage-2 detector — model choice is `--config`, not a
code change.

Objective: supervised fine-tuning (peft LoRA adapter) with the loss masked to
the label completion only. On CUDA the base loads in 4-bit NF4 (QLoRA);
bitsandbytes is CUDA-only, so on Apple MPS / CPU the base loads unquantized
(bf16/fp32) — enough for the local 1-step dry-run. The real training runs happen
on cloud GPUs: the three Stage-1 candidates and the 7B Stage-2 fine-tune.

Outputs under ``results/stage1/<name>/``:
  adapter/            trained PEFT LoRA adapter + tokenizer
  train_summary.json  base id, hyperparams, final loss, steps, device
  val_logits.jsonl    per-row {logp_benign, logp_injection, p_safe}
  cal_logits.jsonl    "                                            "
(val/cal are dumped automatically when the split files exist; skip with
--no-dump-logits, or name explicit splits with --dump-logits.)

PAYLOAD HYGIENE: reads data/ but never prints row text — only
counts, losses, paths.

    # CPU smoke test
    python scripts/train/train_stage1.py --config configs/models/tiny-test.yaml --dry-run
    # Stage-1 runs in the paper (1 epoch, first 12,000 rows, max length 1,024)
    python scripts/train/train_stage1.py --config configs/models/qwen2.5-1.5b.yaml \
        --training configs/training_16gb.yaml --max-samples 12000
    # Stage-2 run in the paper (1 epoch, 50,349 rows, max length 2,048)
    python scripts/train/train_stage1.py --config configs/models/mistral-7b-v0.1.yaml \
        --training configs/training_7b_1ep.yaml \
        --train-file data/train_proposal/train_stage2.jsonl \
        --output-dir results/stage2/mistral-7b-v0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.prompt_template import format_training_example, load_model_config
from src.utils.io import load_yaml, read_jsonl, write_jsonl
from src.utils.logging import get_logger

log = get_logger("train_stage1")

DEFAULT_TRAIN = ROOT / "data" / "train_proposal" / "train.jsonl"
DEFAULT_VAL = ROOT / "data" / "train_proposal" / "val.jsonl"
DEFAULT_CAL = ROOT / "data" / "train_proposal" / "cal.jsonl"
DEFAULT_TRAINING_CFG = ROOT / "configs" / "training.yaml"

# data label -> config label index (benign first); accept str and int spellings.
_SAFE = {"safe", "benign", 0, "0"}
_UNSAFE = {"unsafe", "injection", 1, "1"}


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _row_text(r: dict) -> str:
    return r.get("input") or r.get("text") or ""


def _map_label(raw, labels: tuple[str, str]) -> str:
    if raw in _SAFE:
        return labels[0]
    if raw in _UNSAFE:
        return labels[1]
    raise ValueError(f"unrecognized label {raw!r}")


def load_examples(path: Path, config, limit: int | None):
    """Read a proposal jsonl into {prompt, completion} string columns."""
    prompts, completions = [], []
    for n, r in enumerate(read_jsonl(path)):
        if limit is not None and n >= limit:
            break
        p, c = format_training_example(config, _row_text(r), _map_label(r.get("label"), config.labels))
        prompts.append(p)
        completions.append(c)
    return {"prompt": prompts, "completion": completions}


def build_model(config, device: str, tcfg: dict, use_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        q = tcfg["quantization"]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=q.get("load_in_4bit", True),
            bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, q.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32 if device == "cpu" else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(config.hf_id, **kwargs)
    if not use_4bit:
        model = model.to(device)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA SFT for a Stage-1/2 candidate")
    p.add_argument("--config", required=True, help="model config: name or path to configs/models/<name>.yaml")
    p.add_argument("--training", type=Path, default=DEFAULT_TRAINING_CFG)
    p.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="tiny subset + 1 optimizer step; verifies the pipeline locally")
    p.add_argument("--max-samples", type=int, default=None, help="cap training rows")
    p.add_argument("--dump-logits", nargs="+", type=Path, default=None, metavar="JSONL",
                   help="score these split(s) post-train -> <stem>_logits.jsonl (default: val+cal if present)")
    p.add_argument("--no-dump-logits", action="store_true", help="skip the default val/cal logit dump")
    p.add_argument("--no-4bit", action="store_true", help="force-disable 4-bit even on CUDA")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    config = load_model_config(args.config)
    tcfg = load_yaml(args.training)
    device = auto_device()
    use_4bit = (device == "cuda") and not args.no_4bit
    out_dir = args.output_dir or (ROOT / "results" / "stage1" / config.name)
    adapter_dir = out_dir / "adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("model=%s hf_id=%s device=%s 4bit=%s dry_run=%s",
             config.name, config.hf_id, device, use_4bit, args.dry_run)

    # ---- data ------------------------------------------------------------
    limit = 8 if args.dry_run else args.max_samples
    ds = Dataset.from_dict(load_examples(args.train_file, config, limit))
    log.info("loaded %d train examples from %s", len(ds), args.train_file.name)

    # ---- tokenizer + model + LoRA ---------------------------------------
    tok = AutoTokenizer.from_pretrained(config.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = build_model(config, device, tcfg, use_4bit)
    model.config.use_cache = False
    lcfg = tcfg["lora"]
    lora = LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        bias=lcfg["bias"], task_type=lcfg["task_type"],
        target_modules=list(config.lora_target_modules),
    )

    # ---- trainer ---------------------------------------------------------
    tr, opt = tcfg["trainer"], tcfg["optimizer"]
    on_cuda = device == "cuda"  # bf16 + gradient checkpointing are CUDA-only here
    # transformers 5.x dropped TrainingArguments.warmup_ratio (only warmup_steps
    # remains). Convert ratio->steps faithfully from the resolved schedule.
    import math
    _eff_batch = max(1, (2 if args.dry_run else tr["per_device_train_batch_size"])
                     * (1 if args.dry_run else tr["gradient_accumulation_steps"]))
    _total_steps = 1 if args.dry_run else max(
        1, math.ceil(len(ds) / _eff_batch) * tr["num_train_epochs"])
    _warmup_steps = max(0, round(opt.get("warmup_ratio", 0.0) * _total_steps))
    sft = SFTConfig(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=1 if args.dry_run else tr["num_train_epochs"],
        max_steps=1 if args.dry_run else -1,
        per_device_train_batch_size=2 if args.dry_run else tr["per_device_train_batch_size"],
        gradient_accumulation_steps=1 if args.dry_run else tr["gradient_accumulation_steps"],
        learning_rate=opt["lr"],
        lr_scheduler_type=opt["lr_scheduler_type"],
        warmup_steps=_warmup_steps,
        weight_decay=opt["weight_decay"],
        max_grad_norm=tr["max_grad_norm"],
        logging_steps=1 if args.dry_run else tr["logging_steps"],
        save_strategy="no" if args.dry_run else "steps",
        save_steps=tr["save_steps"],
        save_total_limit=tr["save_total_limit"],
        bf16=on_cuda and tr.get("bf16", True),
        gradient_checkpointing=on_cuda and tr.get("gradient_checkpointing", True),
        # tiny-test (gpt2) has 1024 positions; the dry run only checks the loop.
        max_length=min(tcfg["max_seq_length"], 512) if args.dry_run else tcfg["max_seq_length"],
        seed=tcfg.get("seed", 42),
        report_to=[],
        # TRL >=1.x defaults loss_type to "chunked_nll", whose _patch_chunked_ce_lm_head
        # does inspect.signature(model.forward.__func__) — but transformers 5.x makes
        # model.forward a functools.partial (no __func__), so __init__ crashes. "nll" is
        # the identical CE loss without that patch. See sft_trainer.py:377.
        loss_type="nll",
    )
    trainer = SFTTrainer(model=model, args=sft, train_dataset=ds,
                         peft_config=lora, processing_class=tok)

    log.info("training...")
    result = trainer.train()
    final_loss = float(result.training_loss) if result and result.training_loss is not None else None
    log.info("training done: steps=%s final_loss=%s", result.global_step, final_loss)

    # ---- save adapter + summary -----------------------------------------
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    summary = {
        "task": "train-stage1",
        "model": config.name,
        "base_hf_id": config.hf_id,
        "train_file": args.train_file.name,
        "n_train": len(ds),
        "device": device,
        "load_in_4bit": use_4bit,
        "dry_run": args.dry_run,
        "hyperparams": {
            "lr": opt["lr"], "scheduler": opt["lr_scheduler_type"],
            "epochs": sft.num_train_epochs, "max_steps": sft.max_steps,
            "per_device_batch": sft.per_device_train_batch_size,
            "grad_accum": sft.gradient_accumulation_steps,
            "lora_r": lcfg["r"], "lora_alpha": lcfg["alpha"],
            "max_seq_length": tcfg["max_seq_length"],
        },
        "global_step": result.global_step,
        "final_train_loss": final_loss,
        "adapter_dir": str(adapter_dir.relative_to(ROOT)) if adapter_dir.is_relative_to(ROOT) else str(adapter_dir),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("saved adapter -> %s ; summary -> %s", adapter_dir, out_dir / "train_summary.json")

    # ---- logits-ready check + optional dumps -----------------------------
    # Reuse the Stage-1 scorer on the freshly trained model: proves the checkpoint
    # yields label logits, and emits the E2 val/cal logit files.
    from src.models.stage1 import Stage1Detector

    detector = Stage1Detector(config, model=trainer.model, tokenizer=tok, device=device)

    if args.dump_logits is not None:
        dump_targets = list(args.dump_logits)
    elif args.dry_run or args.no_dump_logits:
        dump_targets = []
    else:
        dump_targets = [p for p in (DEFAULT_VAL, DEFAULT_CAL) if p.exists()]

    for path in dump_targets:
        texts = [_row_text(r) for r in read_jsonl(path)]
        preds = detector.predict(texts)
        write_jsonl(out_dir / f"{path.stem}_logits.jsonl", preds)
        log.info("dumped %d logits -> %s", len(preds), out_dir / f"{path.stem}_logits.jsonl")

    probe = detector.p_safe("Please summarize the meeting notes for tomorrow.")
    assert 0.0 <= probe <= 1.0, f"p_safe out of range: {probe}"
    log.info("logits-ready OK: p_safe(sample)=%.4f in [0,1]", probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
