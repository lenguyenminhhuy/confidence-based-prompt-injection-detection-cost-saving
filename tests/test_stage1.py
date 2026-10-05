"""Constrained label-token scoring, tested against fakes with known logits.

The fake model emits position- and token-dependent logits so the tests verify
the exact gather positions (label token k is predicted from position
len(prompt) - 1 + k) and multi-token summation, not just output shapes.
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from src.models.prompt_template import load_model_config
from src.models.stage1 import Stage1Detector

VOCAB = 16
BENIGN_IDS = [5]          # "benign"  -> single token
INJECTION_IDS = [7, 8]    # "injection" -> two tokens (tests summation)


class FakeTokenizer:
    """Whitespace tokenizer with a fixed vocab; labels map to known ids."""

    vocab = {"benign": BENIGN_IDS, "injection": INJECTION_IDS}

    def __call__(self, text, add_special_tokens=False):
        if text in self.vocab:
            return {"input_ids": list(self.vocab[text])}
        # arbitrary but deterministic prompt ids in [0, 4]
        ids = [hash(w) % 5 for w in text.split()]
        return {"input_ids": ids or [0]}


class FakeModel(torch.nn.Module):
    """logits[b, pos, tok] = weight * tok + pos  (independent of input values).

    Position-dependent so a wrong gather offset changes the result.
    """

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight

    def forward(self, input_ids):
        b, t = input_ids.shape
        tok = torch.arange(VOCAB, dtype=torch.float32)
        pos = torch.arange(t, dtype=torch.float32)
        logits = self.weight * tok.view(1, 1, -1) + pos.view(1, -1, 1)
        return SimpleNamespace(logits=logits.expand(b, t, VOCAB).clone())


def expected_logp(label_ids, weight=0.5):
    """Independent re-computation of the summed label log-prob.

    The fake model's position term is constant across the vocab at any given
    position, so it cancels in the softmax and only the token term remains.
    """
    row = torch.tensor([weight * v for v in range(VOCAB)]).log_softmax(0)
    return float(sum(row[tok_id] for tok_id in label_ids))


@pytest.fixture(params=["qwen2.5-1.5b", "llama3.2-1b", "granite-guardian-2b"])
def detector(request):
    return Stage1Detector(
        load_model_config(request.param),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device="cpu",
    )


def test_label_token_ids_resolved(detector):
    assert detector.label_token_ids == [BENIGN_IDS, INJECTION_IDS]


def test_score_labels_matches_manual_computation(detector):
    logps = detector.score_labels("please summarize this document")
    assert logps["benign"] == pytest.approx(expected_logp(BENIGN_IDS), abs=1e-5)
    assert logps["injection"] == pytest.approx(expected_logp(INJECTION_IDS), abs=1e-5)


def test_p_safe_is_softmax_of_label_logps(detector):
    text = "hello there"
    logps = detector.score_labels(text)
    expected = math.exp(logps["benign"]) / (
        math.exp(logps["benign"]) + math.exp(logps["injection"])
    )
    assert detector.p_safe(text) == pytest.approx(expected, abs=1e-6)
    assert 0.0 <= detector.p_safe(text) <= 1.0


def test_p_safe_moves_with_benign_evidence():
    """Boosting the benign token's logit must raise p_safe monotonically."""

    class BiasedModel(FakeModel):
        def __init__(self, benign_boost):
            super().__init__()
            self.benign_boost = benign_boost

        def forward(self, input_ids):
            out = super().forward(input_ids)
            out.logits[..., BENIGN_IDS[0]] += self.benign_boost
            return out

    cfg = load_model_config("qwen2.5-1.5b")
    p = [
        Stage1Detector(cfg, model=BiasedModel(b), tokenizer=FakeTokenizer(), device="cpu")
        .p_safe("some text")
        for b in (-2.0, 0.0, 2.0)
    ]
    assert p[0] < p[1] < p[2]
    assert p[2] > 0.5


def test_predict_rows(detector):
    rows = detector.predict(["first text", "second text"])
    assert len(rows) == 2
    for row in rows:
        assert set(row) == {"logp_benign", "logp_injection", "p_safe"}
        pair = torch.tensor([row["logp_benign"], row["logp_injection"]])
        assert row["p_safe"] == pytest.approx(float(pair.softmax(0)[0]), abs=1e-6)


@pytest.mark.skipif(
    not os.environ.get("RUN_HF_INTEGRATION"),
    reason="downloads real checkpoints; run with RUN_HF_INTEGRATION=1 "
    "(llama3.2-1b is gated — needs HF_TOKEN with access)",
)
@pytest.mark.parametrize("name", ["qwen2.5-1.5b", "llama3.2-1b", "granite-guardian-2b"])
def test_real_model_returns_p_safe(name):
    det = Stage1Detector.from_config_name(name)
    p = det.p_safe("What is the capital of France?")
    assert 0.0 <= p <= 1.0


def test_empty_label_tokenization_rejected():
    class BadTokenizer(FakeTokenizer):
        def __call__(self, text, add_special_tokens=False):
            if text == "injection":
                return {"input_ids": []}
            return super().__call__(text, add_special_tokens)

    with pytest.raises(ValueError, match="injection"):
        Stage1Detector(
            load_model_config("qwen2.5-1.5b"),
            model=FakeModel(),
            tokenizer=BadTokenizer(),
            device="cpu",
        )
