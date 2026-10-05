"""Stage-1 detector: p_safe from constrained label-token logits.

No free-form generation. For an input text we build the classification prompt
(src/models/prompt_template.py) and score the two label continuations
``config.labels = (benign, injection)`` by summing their token log-probs;
``p_safe = softmax([logp_benign, logp_injection])[0]`` is the confidence the
cascade's ``route()`` consumes and calibration rescales.

``model``/``tokenizer`` are injectable so the scoring logic is unit-testable
with fakes; when omitted they are loaded from ``config.hf_id`` (optionally
with a PEFT adapter and/or 4-bit NF4 quantization for CUDA).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from src.models.prompt_template import ModelConfig, format_prompt, load_model_config


def _auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Stage1Detector:
    def __init__(
        self,
        config: ModelConfig,
        *,
        model=None,
        tokenizer=None,
        adapter_path: str | Path | None = None,
        device: str | None = None,
        load_in_4bit: bool = False,
    ) -> None:
        self.config = config
        self.device = device or _auto_device()
        self.tokenizer = tokenizer if tokenizer is not None else self._load_tokenizer()
        self.model = (
            model if model is not None else self._load_model(adapter_path, load_in_4bit)
        )
        self.model.eval()
        # Multi-token labels are supported: log-probs are summed over the pieces.
        self.label_token_ids: list[list[int]] = [
            self.tokenizer(label, add_special_tokens=False)["input_ids"]
            for label in config.labels
        ]
        for label, ids in zip(config.labels, self.label_token_ids):
            if not ids:
                raise ValueError(f"label {label!r} tokenized to no tokens")

    @classmethod
    def from_config_name(cls, name: str, **kwargs) -> "Stage1Detector":
        return cls(load_model_config(name), **kwargs)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.config.tokenizer)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _load_model(self, adapter_path: str | Path | None, load_in_4bit: bool):
        from transformers import AutoModelForCausalLM

        kwargs: dict = {"torch_dtype": torch.bfloat16 if self.device != "cpu" else torch.float32}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(self.config.hf_id, **kwargs)
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
        if not load_in_4bit:  # quantized models are already placed by device_map
            model = model.to(self.device)
        return model

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_labels(self, text: str) -> dict[str, float]:
        """Summed log-prob of each label continuation after the prompt."""
        prompt = format_prompt(self.config, text)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        out: dict[str, float] = {}
        for label, label_ids in zip(self.config.labels, self.label_token_ids):
            full = torch.tensor([prompt_ids + label_ids], device=self.device)
            logits = self.model(full).logits.float()
            # label token k is predicted from position len(prompt_ids) - 1 + k
            log_probs = logits[0, len(prompt_ids) - 1 : -1, :].log_softmax(dim=-1)
            targets = torch.tensor(label_ids, device=self.device).unsqueeze(1)
            out[label] = log_probs.gather(1, targets).sum().item()
        return out

    def p_safe(self, text: str) -> float:
        """P(labels[0] | prompt) over the two constrained labels — in [0, 1]."""
        logps = self.score_labels(text)
        pair = torch.tensor([logps[label] for label in self.config.labels])
        return pair.softmax(dim=0)[0].item()

    def predict(self, texts: Iterable[str]) -> list[dict[str, float]]:
        """Per-text record with both raw log-probs and p_safe.

        Keys: logp_<label> for each label, plus p_safe. This is the row format
        that E2 persists as val_logits.jsonl / cal_logits.jsonl.
        """
        rows = []
        for text in texts:
            logps = self.score_labels(text)
            pair = torch.tensor([logps[label] for label in self.config.labels])
            row = {f"logp_{label}": lp for label, lp in logps.items()}
            row["p_safe"] = pair.softmax(dim=0)[0].item()
            rows.append(row)
        return rows
