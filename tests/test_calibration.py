"""Unit tests for the E3 calibration module.

Covers: temperature fit reduces NLL + ECE on synthetic overconfident logits;
TemperatureScaler JSON round-trip and calibrated_logps consistency; isotonic
monotonicity; and that choose_operating_point's vectorized routing matches the
canonical route() and fails loudly when infeasible.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):  # metrics.py uses bare `from evaluation...`
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.calibration.temperature_scaling import (
    IsotonicScaler,
    TemperatureScaler,
    _nll_injection,
    fit_temperature,
    gap_from_logps,
)
from src.calibration.threshold_search import _route_counts, choose_operating_point
from src.evaluation.metrics import ece
from src.pipeline.routing import Thresholds, route


def _synthetic_overconfident(n=4000, scale=3.0, seed=0):
    """Labels from a true gap; observed gap is `scale`x too sharp (overconfident)."""
    rng = np.random.default_rng(seed)
    z_true = rng.normal(0.0, 1.6, n)
    is_safe = rng.random(n) < expit(z_true)          # P(safe) = sigmoid(z_true)
    y = (~is_safe).astype(int)                        # 1 = injection
    z_obs = z_true * scale                            # overconfident logit gap
    logp_benign, logp_injection = z_obs, np.zeros_like(z_obs)
    return logp_benign, logp_injection, y, z_true


def test_fit_temperature_reduces_nll_and_ece():
    lb, li, y, _ = _synthetic_overconfident(scale=3.0)
    z = gap_from_logps(lb, li)
    T = fit_temperature(z, y)
    # Overconfident -> optimal temperature should be > 1 (softens the scores).
    assert T > 1.3
    # NLL strictly lower than the uncalibrated T=1 baseline.
    assert _nll_injection(T, z, y) < _nll_injection(1.0, z, y)
    # ECE (confidence axis = p_injection = 1 - p_safe) drops after scaling.
    ece_before = ece(y, 1.0 - expit(z))
    ece_after = ece(y, 1.0 - expit(z / T))
    assert ece_after < ece_before


def test_temperature_scaler_roundtrip_and_calibrated_logps(tmp_path):
    lb, li, y, _ = _synthetic_overconfident()
    s = TemperatureScaler().fit(lb, li, y)
    p_direct = s.transform(lb, li)
    # calibrated_logps must reproduce the same p_safe (gap/T invariance).
    clb, cli = s.calibrated_logps(lb, li)
    p_via_logps = expit(gap_from_logps(clb, cli))
    np.testing.assert_allclose(p_direct, p_via_logps, rtol=0, atol=1e-9)
    # JSON freeze/reload preserves T and reproduces transforms.
    path = tmp_path / "temperature.json"
    s.to_json(path)
    s2 = TemperatureScaler.from_json(path)
    assert s2.T == pytest.approx(s.T)
    np.testing.assert_allclose(s2.transform(lb, li), p_direct, atol=1e-12)


def test_isotonic_is_monotone():
    lb, li, y, _ = _synthetic_overconfident()
    p_safe = expit(gap_from_logps(lb, li))
    iso = IsotonicScaler().fit(p_safe, y)
    grid = np.linspace(0.0, 1.0, 200)
    out = iso.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)             # non-decreasing


def test_route_counts_matches_canonical_route():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < 0.4).astype(int)
    th = Thresholds(theta_safe=0.8, theta_unsafe=0.2)
    c = _route_counts(p, y == 1, y == 0, th.theta_safe, th.theta_unsafe)
    decisions = [route(float(pi), th) for pi in p]
    esc = sum(d == "escalate" for d in decisions)
    leaked = sum(d == "pass" and yi == 1 for d, yi in zip(decisions, y))
    fp = sum(d == "block" and yi == 0 for d, yi in zip(decisions, y))
    assert (c["escalated"], c["leaked_attacks"], c["auto_block_FP"]) == (esc, leaked, fp)


def test_choose_operating_point_selects_min_escalation():
    # Well-separated: benign p_safe ~0.9, attacks ~0.1 -> a clean point exists.
    rng = np.random.default_rng(2)
    p = np.concatenate([rng.uniform(0.85, 0.99, 300),      # benign
                        rng.uniform(0.01, 0.15, 200)])     # attacks
    y = np.array([0] * 300 + [1] * 200)
    safe_grid = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]
    unsafe_grid = [round(x, 2) for x in np.arange(0.00, 0.50, 0.05)]
    th, diag = choose_operating_point(
        p, y, fpr_target=0.01, safe_grid=safe_grid, unsafe_grid=unsafe_grid
    )
    assert diag["leaked_attacks"] == 0
    assert diag["e2e_fpr_autoblock"] <= 0.01
    assert 0.0 <= diag["escalation_rate"] <= 1.0
    assert isinstance(th, Thresholds)


def test_choose_operating_point_raises_when_infeasible():
    # One attack looks perfectly safe -> no zero-leak point in a bounded grid.
    p = np.array([0.99, 0.98, 0.02, 0.03, 1.00])          # last is a leaking attack
    y = np.array([0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="no in-grid operating point"):
        choose_operating_point(
            p, y, fpr_target=0.0,
            safe_grid=[0.5, 0.7, 0.9], unsafe_grid=[0.1, 0.2, 0.3],
        )
