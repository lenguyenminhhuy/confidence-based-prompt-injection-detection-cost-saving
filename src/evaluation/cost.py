"""
Cost accounting for cascade-PID detectors.

Implements the cost measurement protocol (docs/cost_protocol.md):
per-detector cost axes (GPU-time, $-per-1M-requests, params, VRAM, FLOPs) and
the cascade cost model from the proposal:

    expected cost      C(e) = k1 + e * k2          (per request)
    break-even         cascade cheaper than Stage-2-only  iff  e < 1 - k1/k2
    cost reduction     x = 1 - (k1 + e * k2) / k2

k1 = Stage-1 per-request cost, k2 = Stage-2 per-request cost, e = escalation
rate. k1/k2 may be expressed in any common per-request unit (ms, GPU-seconds,
$) — the ratio-based quantities are unit-invariant as long as both stages use
the same unit measured on the same hardware.

Dependencies: stdlib only. Python 3.9-compatible syntax.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Pricing assumption
# ---------------------------------------------------------------------------

# Working assumption for a 24 GB-class cloud GPU (NVIDIA L4 / A10 on-demand
# tier). All $ figures scale linearly in this rate; the protocol requires
# stating the rate used alongside any reported $ number.
DEFAULT_GPU_HOURLY_USD = 1.20


# ---------------------------------------------------------------------------
# Detector hardware profiles (hardware-independent cost axes)
# ---------------------------------------------------------------------------
# param_count: total parameters of the loaded model (adapter params are
#   negligible: LoRA r16 adds <1% and is merged/loaded alongside the base).
# load_dtype_bytes: bytes/param as loaded in our runs (4.0 fp32, 2.0 bf16,
#   0.55 for 4-bit NF4 incl. quantization constants).
DETECTOR_PROFILES: Dict[str, dict] = {
    "promptguard2_86m": {
        "model": "meta-llama/Llama-Prompt-Guard-2-86M",
        "param_count": 86_000_000,
        "load_dtype": "fp32",
        "load_dtype_bytes": 4.0,
    },
    "protectai_deberta_v2": {
        "model": "protectai/deberta-v3-base-prompt-injection-v2",
        "param_count": 184_000_000,
        "load_dtype": "fp32",
        "load_dtype_bytes": 4.0,
    },
    # 4-bit NF4 per notebooks/baselines/02_datasentinel.ipynb (BitsAndBytesConfig)
    "datasentinel_7b": {
        "model": "mistralai/Mistral-7B-v0.1 + DataSentinel QLoRA adapter",
        "param_count": 7_240_000_000,
        "load_dtype": "nf4",
        "load_dtype_bytes": 0.55,
    },
    # Stage-1 inference dtype per src/models/stage1.py: load_in_4bit defaults to
    # False → bf16 on GPU. Update these if E5 opts into 4-bit loading.
    "qwen2.5-1.5b": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct (QLoRA adapter)",
        "param_count": 1_540_000_000,
        "load_dtype": "bf16",
        "load_dtype_bytes": 2.0,
    },
    "llama3.2-1b": {
        "model": "meta-llama/Llama-3.2-1B-Instruct (QLoRA adapter)",
        "param_count": 1_240_000_000,
        "load_dtype": "bf16",
        "load_dtype_bytes": 2.0,
    },
    "granite-guardian-2b": {
        "model": "ibm-granite/granite-guardian-3.0-2b (QLoRA adapter)",
        "param_count": 2_530_000_000,
        "load_dtype": "bf16",
        "load_dtype_bytes": 2.0,
    },
}


# ---------------------------------------------------------------------------
# Latency → GPU-time → $
# ---------------------------------------------------------------------------

def gpu_seconds_per_1k_requests(mean_latency_ms: float) -> float:
    """GPU-seconds consumed by 1,000 requests at the given mean latency.

    Numerically equal to ``mean_latency_ms`` (1k requests × latency_ms / 1000),
    but named explicitly so reported numbers carry their unit.
    """
    if mean_latency_ms is None or math.isnan(mean_latency_ms):
        return float("nan")
    if mean_latency_ms < 0:
        raise ValueError(f"mean_latency_ms must be >= 0, got {mean_latency_ms}")
    return float(mean_latency_ms * 1000.0 / 1000.0)


def usd_per_1m_requests(
    mean_latency_ms: float,
    gpu_hourly_usd: float = DEFAULT_GPU_HOURLY_USD,
) -> float:
    """Dollar cost of serving 1M requests sequentially at the given latency.

    cost = (latency_s x 1e6 requests / 3600 s/hr) x $/GPU-hr

    This is a *sequential, batch-dependent* figure: latency measured at batch=1
    is an upper bound; batched throughput lowers it proportionally. The
    measurement basis must be stated per the cost protocol.
    """
    if mean_latency_ms is None or math.isnan(mean_latency_ms):
        return float("nan")
    if mean_latency_ms < 0:
        raise ValueError(f"mean_latency_ms must be >= 0, got {mean_latency_ms}")
    if gpu_hourly_usd < 0:
        raise ValueError(f"gpu_hourly_usd must be >= 0, got {gpu_hourly_usd}")
    gpu_hours = (mean_latency_ms / 1000.0) * 1_000_000.0 / 3600.0
    return float(gpu_hours * gpu_hourly_usd)


# ---------------------------------------------------------------------------
# Hardware-independent axes
# ---------------------------------------------------------------------------

def vram_estimate_gb(
    param_count: int,
    bytes_per_param: float,
    overhead_factor: float = 1.2,
) -> float:
    """Estimated inference VRAM footprint in GB (weights x overhead).

    ``overhead_factor`` covers activations, KV cache at modest sequence
    lengths, and framework buffers; 1.2 is a batch=1 rule of thumb, not a
    measurement.
    """
    if param_count < 0 or bytes_per_param < 0 or overhead_factor < 1.0:
        raise ValueError("param_count/bytes_per_param must be >= 0, overhead_factor >= 1")
    return float(param_count * bytes_per_param * overhead_factor / 1e9)


def flops_per_request(param_count: int, tokens_per_request: int) -> float:
    """Forward-pass FLOPs estimate: 2 x params x tokens (dense transformer)."""
    if param_count < 0 or tokens_per_request < 0:
        raise ValueError("param_count and tokens_per_request must be >= 0")
    return float(2.0 * param_count * tokens_per_request)


# ---------------------------------------------------------------------------
# Cascade cost model (proposal Sec. Architecture)
# ---------------------------------------------------------------------------

def expected_cost(k1: float, k2: float, escalation_rate: float) -> float:
    """Per-request expected cascade cost: C(e) = k1 + e * k2."""
    _validate_cascade_args(k1, k2, escalation_rate)
    return float(k1 + escalation_rate * k2)


def breakeven_escalation_rate(k1: float, k2: float) -> float:
    """Escalation rate below which the cascade beats Stage-2-only: 1 - k1/k2."""
    _validate_cascade_args(k1, k2, 0.0)
    if k2 == 0:
        return float("nan")
    return float(1.0 - k1 / k2)


def cost_reduction(k1: float, k2: float, escalation_rate: float) -> float:
    """Fractional cost reduction vs Stage-2-only: x = 1 - (k1 + e*k2)/k2.

    Positive when the cascade is cheaper; negative when e exceeds break-even.
    """
    _validate_cascade_args(k1, k2, escalation_rate)
    if k2 == 0:
        return float("nan")
    return float(1.0 - (k1 + escalation_rate * k2) / k2)


def _validate_cascade_args(k1: float, k2: float, escalation_rate: float) -> None:
    if math.isnan(k1) or math.isnan(k2) or math.isnan(escalation_rate):
        return  # NaN propagates naturally through the arithmetic
    if k1 < 0 or k2 < 0:
        raise ValueError(f"k1 and k2 must be >= 0, got k1={k1}, k2={k2}")
    if not (0.0 <= escalation_rate <= 1.0):
        raise ValueError(f"escalation_rate must be in [0, 1], got {escalation_rate}")


# ---------------------------------------------------------------------------
# Report blocks
# ---------------------------------------------------------------------------

def detector_cost_block(
    mean_latency_ms: float,
    detector: Optional[str] = None,
    gpu_hourly_usd: float = DEFAULT_GPU_HOURLY_USD,
    tokens_per_request: int = 512,
) -> dict:
    """Cost fields for a detector's metrics JSON (consumed by evaluate_detector).

    Hardware-dependent axes (GPU-time, $) derive from ``mean_latency_ms``;
    hardware-independent axes (params, VRAM, FLOPs) come from
    ``DETECTOR_PROFILES`` when the detector name is known, else null.
    """
    block: dict = {
        "gpu_seconds_per_1k_requests": gpu_seconds_per_1k_requests(mean_latency_ms),
        "usd_per_1m_requests": usd_per_1m_requests(mean_latency_ms, gpu_hourly_usd),
        "gpu_hourly_usd_assumption": float(gpu_hourly_usd),
        "latency_basis": (
            "mean of per-record latency_ms; measurement conditions per "
            "docs/cost_protocol.md"
        ),
    }

    profile = DETECTOR_PROFILES.get(detector) if detector else None
    if profile is not None:
        block["param_count"] = profile["param_count"]
        block["load_dtype"] = profile["load_dtype"]
        block["vram_estimate_gb"] = vram_estimate_gb(
            profile["param_count"], profile["load_dtype_bytes"]
        )
        block["flops_per_request_estimate"] = flops_per_request(
            profile["param_count"], tokens_per_request
        )
        block["flops_tokens_per_request_assumption"] = tokens_per_request
    else:
        block["param_count"] = None
        block["load_dtype"] = None
        block["vram_estimate_gb"] = None
        block["flops_per_request_estimate"] = None
        block["flops_tokens_per_request_assumption"] = None

    return block


def cascade_cost_summary(
    k1_latency_ms: float,
    k2_latency_ms: float,
    escalation_rate: float,
    gpu_hourly_usd: float = DEFAULT_GPU_HOURLY_USD,
) -> dict:
    """Full cascade cost report from measured k1/k2 latencies and escalation rate.

    Expresses the expected cost and the headline reduction in all three
    protocol units (latency ms, GPU-seconds/1k requests, $/1M requests). The
    ratio quantities (break-even, reduction %) are unit-invariant.
    """
    ratio = (
        float(k1_latency_ms / k2_latency_ms)
        if k2_latency_ms and not math.isnan(k2_latency_ms) and not math.isnan(k1_latency_ms)
        else float("nan")
    )
    expected_ms = expected_cost(k1_latency_ms, k2_latency_ms, escalation_rate)
    breakeven = breakeven_escalation_rate(k1_latency_ms, k2_latency_ms)
    reduction = cost_reduction(k1_latency_ms, k2_latency_ms, escalation_rate)

    return {
        "k1_latency_ms": float(k1_latency_ms),
        "k2_latency_ms": float(k2_latency_ms),
        "escalation_rate": float(escalation_rate),
        "k1_over_k2": ratio,
        "breakeven_escalation_rate": breakeven,
        "cascade_is_cheaper": (
            bool(escalation_rate < breakeven) if not math.isnan(breakeven) else None
        ),
        "expected_cost_latency_ms": expected_ms,
        "expected_cost_gpu_seconds_per_1k": gpu_seconds_per_1k_requests(expected_ms),
        "expected_cost_usd_per_1m": usd_per_1m_requests(expected_ms, gpu_hourly_usd),
        "stage2_only_usd_per_1m": usd_per_1m_requests(k2_latency_ms, gpu_hourly_usd),
        "cost_reduction_vs_stage2_only": reduction,
        "cost_reduction_pct": (
            float(reduction * 100.0) if not math.isnan(reduction) else float("nan")
        ),
        "gpu_hourly_usd_assumption": float(gpu_hourly_usd),
    }
