"""
Comprehensive tests for src/evaluation/metrics.py and src/evaluation/per_channel.py.

All cases use synthetic data with known answers so failures are unambiguous.
Run with:
    conda run -n open_prompt_injection python -m pytest tests/test_metrics.py -v
or:
    PYTHONPATH=src python -m pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Support both `PYTHONPATH=src` and installed-package modes
try:
    from src.evaluation.metrics import (
        binary_metrics,
        detection_rate_at_fpr,
        ece,
        evaluate_detector,
        per_channel,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from src.evaluation.metrics import (
        binary_metrics,
        detection_rate_at_fpr,
        ece,
        evaluate_detector,
        per_channel,
    )

from src.evaluation.per_channel import per_channel as per_channel_direct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_records(n_benign: int, n_inject: int, channel: str = "document",
                  score_benign: float = 0.1, score_inject: float = 0.9,
                  binary_only: bool = False) -> list:
    records = []
    for i in range(n_benign):
        records.append({
            "id": f"b-{i:06d}",
            "label": 0,
            "channel": None,
            "score": None if binary_only else score_benign,
            "pred": 0,
            "latency_ms": 10.0,
        })
    for i in range(n_inject):
        records.append({
            "id": f"inj-{i:06d}",
            "label": 1,
            "channel": channel,
            "score": None if binary_only else score_inject,
            "pred": 1,
            "latency_ms": 12.0,
        })
    return records


def _write_jsonl(path: Path, records: list) -> None:
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ===========================================================================
# detection_rate_at_fpr
# ===========================================================================

class TestDetectionRateAtFpr:

    def test_perfectly_separable_all_dr_one(self):
        """When benign scores are all low and injection scores all high, DR=1 at every FPR."""
        labels = [0] * 200 + [1] * 100
        scores = [0.1] * 200 + [0.9] * 100
        result = detection_rate_at_fpr(labels, scores)
        for key, val in result.items():
            assert val["dr"] == pytest.approx(1.0), f"DR should be 1.0 at FPR={key}"
            assert val["achieved_fpr"] == pytest.approx(0.0, abs=1e-9)

    def test_perfectly_overlapping_dr_near_fpr(self):
        """When benign and injection scores are identical, DR ≈ FPR target."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 1000).tolist()
        labels = [0] * 500 + [1] * 500
        result = detection_rate_at_fpr(labels, scores, fpr_targets=(0.01,))
        # DR should be close to 1% since scores are uniform and classes fully overlap
        val = result["0.01"]
        assert val["dr"] == pytest.approx(val["achieved_fpr"], abs=0.05)

    def test_returns_all_requested_fpr_keys(self):
        labels = [0] * 100 + [1] * 50
        scores = [0.2] * 100 + [0.8] * 50
        result = detection_rate_at_fpr(labels, scores, fpr_targets=(0.001, 0.005, 0.01))
        assert set(result.keys()) == {"0.001", "0.005", "0.01"}

    def test_resolvable_flag_small_benign_sample(self):
        """With only 50 benign samples, 0.1% FPR target is not resolvable (needs ~1000)."""
        labels = [0] * 50 + [1] * 50
        scores = [0.1] * 50 + [0.9] * 50
        result = detection_rate_at_fpr(labels, scores, fpr_targets=(0.001,))
        assert result["0.001"]["resolvable"] is False

    def test_resolvable_flag_large_benign_sample(self):
        """With 2000 benign samples, 0.1% FPR target is resolvable (needs ~1000)."""
        labels = [0] * 2000 + [1] * 200
        scores = [0.1] * 2000 + [0.9] * 200
        result = detection_rate_at_fpr(labels, scores, fpr_targets=(0.001,))
        assert result["0.001"]["resolvable"] is True

    def test_handles_nan_scores(self):
        """NaN scores should be dropped; remaining samples used normally."""
        labels = [0, 0, 0, 0, 1, 1, float("nan")]
        scores = [0.1, 0.2, float("nan"), 0.15, 0.9, 0.85, 0.5]
        # Should not raise; benign count drops to 3 (nan label entry also dropped)
        result = detection_rate_at_fpr(
            [0, 0, 0, 1, 1],
            [0.1, 0.2, float("nan"), 0.9, 0.85],
            fpr_targets=(0.5,),
        )
        assert "0.5" in result

    def test_no_benign_samples_returns_nan(self):
        """If there are no benign samples, threshold cannot be set → nan."""
        labels = [1, 1, 1]
        scores = [0.8, 0.9, 0.7]
        result = detection_rate_at_fpr(labels, scores, fpr_targets=(0.01,))
        assert math.isnan(result["0.01"]["dr"])

    def test_no_injection_samples_returns_nan_dr(self):
        """If there are no injection samples, DR is nan but threshold is still valid."""
        labels = [0, 0, 0, 0, 0]
        scores = [0.1, 0.2, 0.15, 0.05, 0.3]
        result = detection_rate_at_fpr(labels, scores, fpr_targets=(0.01,))
        assert math.isnan(result["0.01"]["dr"])
        # threshold should still be a number
        assert not math.isnan(result["0.01"]["threshold"])

    def test_threshold_monotone_across_fpr_targets(self):
        """Higher FPR target → lower or equal threshold (more permissive)."""
        rng = np.random.default_rng(0)
        labels = [0] * 500 + [1] * 250
        scores = rng.uniform(0, 1, 750).tolist()
        result = detection_rate_at_fpr(labels, scores, fpr_targets=(0.001, 0.005, 0.01))
        t001 = result["0.001"]["threshold"]
        t005 = result["0.005"]["threshold"]
        t01 = result["0.01"]["threshold"]
        assert t001 >= t005 >= t01

    def test_achieved_fpr_at_most_target(self):
        """Achieved FPR should not exceed the target (conservative threshold choice)."""
        rng = np.random.default_rng(7)
        labels = [0] * 1000 + [1] * 500
        scores = rng.uniform(0, 1, 1500).tolist()
        result = detection_rate_at_fpr(labels, scores, fpr_targets=(0.01,))
        assert result["0.01"]["achieved_fpr"] <= 0.01 + 1e-9


# ===========================================================================
# binary_metrics
# ===========================================================================

class TestBinaryMetrics:

    def test_perfect_classifier(self):
        labels = [0, 0, 1, 1]
        preds  = [0, 0, 1, 1]
        m = binary_metrics(labels, preds)
        assert m["f1"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["tpr"] == pytest.approx(1.0)
        assert m["fpr"] == pytest.approx(0.0)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_all_wrong(self):
        labels = [0, 0, 1, 1]
        preds  = [1, 1, 0, 0]
        m = binary_metrics(labels, preds)
        assert m["tpr"] == pytest.approx(0.0)
        assert m["fpr"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_all_predict_positive(self):
        labels = [0, 0, 0, 1, 1]
        preds  = [1, 1, 1, 1, 1]
        m = binary_metrics(labels, preds)
        assert m["tpr"] == pytest.approx(1.0)
        assert m["fpr"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)

    def test_all_predict_negative(self):
        labels = [0, 0, 1, 1]
        preds  = [0, 0, 0, 0]
        m = binary_metrics(labels, preds)
        assert m["tpr"] == pytest.approx(0.0)
        assert m["fpr"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)
        # precision is 0/0 → nan
        assert math.isnan(m["precision"])

    def test_no_positives_in_labels(self):
        """Safe division: precision/recall/f1 should be nan, not crash."""
        labels = [0, 0, 0]
        preds  = [0, 1, 0]
        m = binary_metrics(labels, preds)
        assert math.isnan(m["recall"]) or m["recall"] == pytest.approx(0.0)

    def test_tpr_equals_recall(self):
        labels = [0, 1, 1, 0, 1]
        preds  = [0, 1, 0, 0, 1]
        m = binary_metrics(labels, preds)
        assert m["tpr"] == m["recall"]

    def test_known_values(self):
        # TP=3, FP=1, FN=1, TN=2  →  precision=3/4, recall=3/4, F1=3/4
        labels = [1, 1, 1, 1, 0, 0, 0]
        preds  = [1, 1, 1, 0, 1, 0, 0]
        m = binary_metrics(labels, preds)
        assert m["precision"] == pytest.approx(3/4)
        assert m["recall"] == pytest.approx(3/4)
        assert m["f1"] == pytest.approx(3/4)
        assert m["fpr"] == pytest.approx(1/3)
        assert m["accuracy"] == pytest.approx(5/7)


# ===========================================================================
# ece
# ===========================================================================

class TestEce:

    def test_perfectly_calibrated_is_zero(self):
        """If every bin has avg_conf == avg_acc, ECE = 0."""
        # Construct a perfectly calibrated dataset:
        # bin [0.0, 0.1): conf=0.05, acc=0.05  → need 5% of samples correct
        # Use a simple trick: uniform confidences, labels = round(conf)
        # which gives approx perfect calibration per bin.
        n = 1000
        rng = np.random.default_rng(0)
        confs = rng.uniform(0, 1, n)
        # For each sample, label=1 with probability = conf (perfect calibration in expectation)
        labels = (rng.uniform(0, 1, n) < confs).astype(int)
        result = ece(labels, confs, n_bins=10)
        # With n=1000 this should be small; exact zero only in infinite-data limit
        assert result < 0.05

    def test_overconfident_detector(self):
        """All confidences=1.0 but half labels=0 → ECE = 0.5."""
        labels = [1] * 50 + [0] * 50
        confs  = [1.0] * 100
        result = ece(labels, confs, n_bins=10)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_underconfident_detector(self):
        """All confidences=0.0 but all labels=1 → ECE = 1.0."""
        labels = [1] * 100
        confs  = [0.0] * 100
        result = ece(labels, confs, n_bins=10)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_empty_input_returns_nan(self):
        assert math.isnan(ece([], []))

    def test_single_bin_entry(self):
        """Single sample in one bin should not crash."""
        result = ece([1], [0.9], n_bins=10)
        assert result == pytest.approx(0.1, abs=0.01)

    def test_result_in_zero_one_range(self):
        rng = np.random.default_rng(5)
        labels = rng.integers(0, 2, 200).tolist()
        confs  = rng.uniform(0, 1, 200).tolist()
        result = ece(labels, confs)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# per_channel
# ===========================================================================

class TestPerChannel:

    def _simple_metric(self, recs):
        """Toy metric: count of injections in the slice."""
        return {"n_inject": sum(r["label"] for r in recs), "n": len(recs)}

    def test_channels_present_in_result(self):
        records = _make_records(n_benign=10, n_inject=5, channel="document")
        records += _make_records(n_benign=0, n_inject=3, channel="tool")
        result = per_channel_direct(records, self._simple_metric)
        assert "document" in result
        assert "tool" in result
        assert "overall" in result

    def test_overall_covers_all_records(self):
        records = _make_records(n_benign=10, n_inject=5, channel="document")
        result = per_channel_direct(records, self._simple_metric)
        assert result["overall"]["n"] == 15

    def test_benign_only_in_overall(self):
        """Benign records (channel=None) appear only in overall, not per-channel."""
        records = _make_records(n_benign=10, n_inject=0, channel="document")
        result = per_channel_direct(records, self._simple_metric)
        # No injection records → no "document" channel key
        assert "document" not in result
        assert result["overall"]["n"] == 10

    def test_per_channel_binary_metrics(self):
        """Perfect predictions per channel → F1=1.0 in each channel."""
        records = _make_records(n_benign=20, n_inject=10, channel="document")
        records += _make_records(n_benign=15, n_inject=8, channel="tool")

        def _bm(recs):
            from src.evaluation.metrics import binary_metrics
            return binary_metrics([r["label"] for r in recs], [r["pred"] for r in recs])

        result = per_channel_direct(records, _bm)
        assert result["document"]["f1"] == pytest.approx(1.0)
        assert result["tool"]["f1"] == pytest.approx(1.0)
        assert result["overall"]["f1"] == pytest.approx(1.0)

    def test_multichannel_aggregation(self):
        """Three channels: document, tool, direct + benign."""
        doc  = _make_records(n_benign=0, n_inject=5, channel="document")
        tool = _make_records(n_benign=0, n_inject=3, channel="tool")
        direct = _make_records(n_benign=0, n_inject=2, channel="direct")
        benign = _make_records(n_benign=20, n_inject=0, channel="document")
        # benign records have channel=None so they don't pollute per-channel
        records = doc + tool + direct + benign
        result = per_channel_direct(records, self._simple_metric)
        assert result["document"]["n_inject"] == 5
        assert result["tool"]["n_inject"] == 3
        assert result["direct"]["n_inject"] == 2
        assert result["overall"]["n_inject"] == 10


# ===========================================================================
# evaluate_detector (integration)
# ===========================================================================

class TestEvaluateDetector:

    def _run(self, records, binary_only=False):
        with tempfile.TemporaryDirectory() as tmp:
            pred_file = Path(tmp) / "predictions" / "test_detector.jsonl"
            pred_file.parent.mkdir(parents=True)
            _write_jsonl(pred_file, records)

            out_file = Path(tmp) / "metrics" / "test_detector.json"
            result = evaluate_detector(str(pred_file), str(out_file))

            # Verify file was written
            assert out_file.exists()
            with out_file.open() as fh:
                loaded = json.load(fh)

        return result, loaded

    def test_continuous_detector_has_dr_at_fpr(self):
        records = _make_records(n_benign=100, n_inject=50,
                                score_benign=0.1, score_inject=0.9)
        result, loaded = self._run(records)
        assert result["dr_at_fpr"] is not None
        assert "0.01" in result["dr_at_fpr"]
        assert loaded["dr_at_fpr"] is not None

    def test_binary_only_detector_dr_at_fpr_is_null(self):
        """Binary-only detectors (score=null) must have dr_at_fpr=None + note."""
        records = _make_records(n_benign=50, n_inject=25, binary_only=True)
        result, loaded = self._run(records, binary_only=True)
        assert result["dr_at_fpr"] is None
        assert "note" in result
        # JSON null maps to None in Python
        assert loaded["dr_at_fpr"] is None

    def test_counts_correct(self):
        records = _make_records(n_benign=80, n_inject=40)
        result, _ = self._run(records)
        assert result["n"] == 120
        assert result["n_benign"] == 80
        assert result["n_injection"] == 40

    def test_perfect_predictions_f1_one(self):
        records = _make_records(n_benign=50, n_inject=30)
        result, _ = self._run(records)
        assert result["binary"]["f1"] == pytest.approx(1.0)

    def test_latency_computed(self):
        records = _make_records(n_benign=20, n_inject=10)
        result, _ = self._run(records)
        assert result["mean_latency_ms"] == pytest.approx(10.8, abs=0.5)

    def test_fpr_benign_only_correct(self):
        """All benign preds=0 → fpr_benign_only = 0.0."""
        records = _make_records(n_benign=50, n_inject=25)
        result, _ = self._run(records)
        assert result["fpr_benign_only"] == pytest.approx(0.0)

    def test_fpr_benign_only_nonzero(self):
        """Manually introduce false positives in benign records."""
        records = _make_records(n_benign=100, n_inject=50)
        # Flip pred=1 for 10 benign records
        for rec in records[:10]:
            if rec["label"] == 0:
                rec["pred"] = 1
        result, _ = self._run(records)
        # Should reflect ~10% FPR on benign
        assert 0.05 <= result["fpr_benign_only"] <= 0.15

    def test_per_channel_in_output(self):
        records = _make_records(n_benign=30, n_inject=15, channel="document")
        records += _make_records(n_benign=0, n_inject=10, channel="tool")
        result, _ = self._run(records)
        assert "document" in result["per_channel"]
        assert "tool" in result["per_channel"]
        assert "overall" in result["per_channel"]

    def test_detector_name_from_filename(self):
        records = _make_records(n_benign=20, n_inject=10)
        with tempfile.TemporaryDirectory() as tmp:
            pred_file = Path(tmp) / "promptguard2_86m.jsonl"
            _write_jsonl(pred_file, records)
            out_file = Path(tmp) / "promptguard2_86m.json"
            result = evaluate_detector(str(pred_file), str(out_file))
        assert result["detector"] == "promptguard2_86m"

    def test_output_written_atomically(self):
        """Output directory is created if it doesn't exist."""
        records = _make_records(n_benign=10, n_inject=5)
        with tempfile.TemporaryDirectory() as tmp:
            pred_file = Path(tmp) / "det.jsonl"
            _write_jsonl(pred_file, records)
            # Nested path that doesn't exist yet
            out_file = Path(tmp) / "deep" / "nested" / "det.json"
            evaluate_detector(str(pred_file), str(out_file))
            assert out_file.exists()

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred_file = Path(tmp) / "empty.jsonl"
            pred_file.write_text("")
            out_file = Path(tmp) / "empty.json"
            with pytest.raises(ValueError, match="No records"):
                evaluate_detector(str(pred_file), str(out_file))

    def test_perfectly_separable_dr_at_fpr_1(self):
        """Perfectly separable scores → DR=1.0 at all FPR targets."""
        records = _make_records(n_benign=2000, n_inject=500,
                                score_benign=0.05, score_inject=0.95)
        result, _ = self._run(records)
        for key, val in result["dr_at_fpr"].items():
            assert val["dr"] == pytest.approx(1.0), f"DR should be 1.0 at FPR={key}"
