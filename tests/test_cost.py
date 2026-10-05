"""
Tests for src/evaluation/cost.py — cost accounting and cascade cost model.

All cases use hand-computed answers so failures are unambiguous.
Run with:
    PYTHONPATH=src python -m pytest tests/test_cost.py -v
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.cost import (
    DEFAULT_GPU_HOURLY_USD,
    DETECTOR_PROFILES,
    breakeven_escalation_rate,
    cascade_cost_summary,
    cost_reduction,
    detector_cost_block,
    expected_cost,
    flops_per_request,
    gpu_seconds_per_1k_requests,
    usd_per_1m_requests,
    vram_estimate_gb,
)


class TestGpuSecondsPer1k:
    def test_numeric_identity_with_latency_ms(self):
        # 1k requests x 50 ms = 50,000 ms = 50 s
        assert gpu_seconds_per_1k_requests(50.0) == pytest.approx(50.0)

    def test_zero_latency(self):
        assert gpu_seconds_per_1k_requests(0.0) == 0.0

    def test_nan_propagates(self):
        assert math.isnan(gpu_seconds_per_1k_requests(float("nan")))

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            gpu_seconds_per_1k_requests(-1.0)


class TestUsdPer1m:
    def test_hand_computed(self):
        # 100 ms/request -> 1e6 requests = 1e5 s = 27.777... GPU-h; at $1.20/h = $33.33...
        assert usd_per_1m_requests(100.0, gpu_hourly_usd=1.20) == pytest.approx(
            100.0 / 1000.0 * 1e6 / 3600.0 * 1.20
        )

    def test_scales_linearly_in_rate(self):
        base = usd_per_1m_requests(42.0, gpu_hourly_usd=1.0)
        assert usd_per_1m_requests(42.0, gpu_hourly_usd=2.0) == pytest.approx(2 * base)

    def test_default_rate_used(self):
        assert usd_per_1m_requests(10.0) == pytest.approx(
            usd_per_1m_requests(10.0, gpu_hourly_usd=DEFAULT_GPU_HOURLY_USD)
        )

    def test_nan_propagates(self):
        assert math.isnan(usd_per_1m_requests(float("nan")))

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError):
            usd_per_1m_requests(-5.0)


class TestHardwareIndependentAxes:
    def test_vram_estimate_bf16_7b(self):
        # 7.24B params x 2 bytes x 1.2 overhead = 17.376 GB
        assert vram_estimate_gb(7_240_000_000, 2.0) == pytest.approx(17.376)

    def test_vram_overhead_below_one_raises(self):
        with pytest.raises(ValueError):
            vram_estimate_gb(1_000_000, 2.0, overhead_factor=0.9)

    def test_flops_per_request(self):
        # 2 x 1B x 512 tokens = 1.024e12
        assert flops_per_request(1_000_000_000, 512) == pytest.approx(1.024e12)

    def test_profiles_have_required_fields(self):
        for name, p in DETECTOR_PROFILES.items():
            assert p["param_count"] > 0, name
            assert p["load_dtype_bytes"] > 0, name
            assert p["load_dtype"], name


class TestCascadeCostModel:
    def test_expected_cost(self):
        # k1=2, k2=100, e=0.2 -> 2 + 20 = 22
        assert expected_cost(2.0, 100.0, 0.2) == pytest.approx(22.0)

    def test_expected_cost_zero_escalation_is_k1(self):
        assert expected_cost(3.5, 200.0, 0.0) == pytest.approx(3.5)

    def test_expected_cost_full_escalation_is_k1_plus_k2(self):
        assert expected_cost(3.5, 200.0, 1.0) == pytest.approx(203.5)

    def test_breakeven(self):
        # k1=2, k2=100 -> 1 - 0.02 = 0.98
        assert breakeven_escalation_rate(2.0, 100.0) == pytest.approx(0.98)

    def test_breakeven_equal_costs_is_zero(self):
        assert breakeven_escalation_rate(50.0, 50.0) == pytest.approx(0.0)

    def test_breakeven_k2_zero_is_nan(self):
        assert math.isnan(breakeven_escalation_rate(1.0, 0.0))

    def test_cost_reduction_hand_computed(self):
        # k1=2, k2=100, e=0.2 -> 1 - 22/100 = 0.78
        assert cost_reduction(2.0, 100.0, 0.2) == pytest.approx(0.78)

    def test_cost_reduction_negative_when_above_breakeven(self):
        # e=0.99 > breakeven 0.98 -> reduction < 0
        assert cost_reduction(2.0, 100.0, 0.99) < 0

    def test_cost_reduction_at_breakeven_is_zero(self):
        e_star = breakeven_escalation_rate(2.0, 100.0)
        assert cost_reduction(2.0, 100.0, e_star) == pytest.approx(0.0)

    def test_escalation_out_of_range_raises(self):
        with pytest.raises(ValueError):
            expected_cost(1.0, 10.0, 1.5)
        with pytest.raises(ValueError):
            cost_reduction(1.0, 10.0, -0.1)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            expected_cost(-1.0, 10.0, 0.5)

    def test_nan_inputs_propagate(self):
        assert math.isnan(expected_cost(float("nan"), 10.0, 0.5))
        assert math.isnan(cost_reduction(1.0, float("nan"), 0.5))


class TestDetectorCostBlock:
    def test_known_detector_gets_profile_fields(self):
        # nf4 per 02_datasentinel.ipynb: 7.24B x 0.55 bytes x 1.2 overhead
        block = detector_cost_block(2.0, detector="datasentinel_7b")
        assert block["param_count"] == 7_240_000_000
        assert block["load_dtype"] == "nf4"
        assert block["vram_estimate_gb"] == pytest.approx(4.7784)
        assert block["flops_per_request_estimate"] == pytest.approx(
            2 * 7_240_000_000 * 512
        )
        assert block["gpu_seconds_per_1k_requests"] == pytest.approx(2.0)

    def test_unknown_detector_gets_null_profile_fields(self):
        block = detector_cost_block(2.0, detector="mystery_detector")
        assert block["param_count"] is None
        assert block["vram_estimate_gb"] is None
        # Latency-derived fields still populated
        assert block["usd_per_1m_requests"] == pytest.approx(
            usd_per_1m_requests(2.0)
        )

    def test_nan_latency_yields_nan_cost_fields(self):
        block = detector_cost_block(float("nan"), detector="promptguard2_86m")
        assert math.isnan(block["gpu_seconds_per_1k_requests"])
        assert math.isnan(block["usd_per_1m_requests"])
        # Hardware-independent axes still available
        assert block["param_count"] == 86_000_000

    def test_rate_assumption_recorded(self):
        block = detector_cost_block(1.0, gpu_hourly_usd=2.5)
        assert block["gpu_hourly_usd_assumption"] == 2.5


class TestCascadeCostSummary:
    def test_hand_computed_summary(self):
        s = cascade_cost_summary(2.0, 100.0, 0.2, gpu_hourly_usd=1.0)
        assert s["k1_over_k2"] == pytest.approx(0.02)
        assert s["breakeven_escalation_rate"] == pytest.approx(0.98)
        assert s["cascade_is_cheaper"] is True
        assert s["expected_cost_latency_ms"] == pytest.approx(22.0)
        assert s["expected_cost_gpu_seconds_per_1k"] == pytest.approx(22.0)
        assert s["expected_cost_usd_per_1m"] == pytest.approx(
            22.0 / 1000.0 * 1e6 / 3600.0
        )
        assert s["cost_reduction_vs_stage2_only"] == pytest.approx(0.78)
        assert s["cost_reduction_pct"] == pytest.approx(78.0)

    def test_not_cheaper_above_breakeven(self):
        s = cascade_cost_summary(2.0, 100.0, 0.99)
        assert s["cascade_is_cheaper"] is False
        assert s["cost_reduction_vs_stage2_only"] < 0

    def test_nan_latency_gives_nan_and_none(self):
        s = cascade_cost_summary(float("nan"), 100.0, 0.2)
        assert math.isnan(s["k1_over_k2"])
        assert s["cascade_is_cheaper"] is None


class TestEvaluateDetectorIntegration:
    def test_metrics_json_gains_cost_block(self, tmp_path):
        import json

        from src.evaluation.metrics import evaluate_detector

        pred_path = tmp_path / "promptguard2_86m.jsonl"
        records = [
            {"id": "b-1", "label": 0, "channel": None, "score": 0.1, "pred": 0, "latency_ms": 10.0},
            {"id": "i-1", "label": 1, "channel": "document", "score": 0.9, "pred": 1, "latency_ms": 30.0},
        ]
        pred_path.write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )
        out_path = tmp_path / "promptguard2_86m.json"

        m = evaluate_detector(str(pred_path), str(out_path))

        assert "cost" in m
        assert m["cost"]["gpu_seconds_per_1k_requests"] == pytest.approx(20.0)
        assert m["cost"]["param_count"] == 86_000_000  # profile matched by stem
        on_disk = json.loads(out_path.read_text(encoding="utf-8"))
        assert on_disk["cost"]["gpu_seconds_per_1k_requests"] == pytest.approx(20.0)
