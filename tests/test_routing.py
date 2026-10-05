import pytest

from src.pipeline.routing import Thresholds, route


def test_route_pass():
    # p_safe=0.95 >= theta_safe=0.80 → pass
    assert route(0.95, Thresholds(0.80, 0.30)) == "pass"


def test_route_block():
    # p_safe=0.05 <= theta_unsafe=0.30 → block
    assert route(0.05, Thresholds(0.80, 0.30)) == "block"


def test_route_escalate():
    # p_safe=0.50 is between theta_unsafe=0.30 and theta_safe=0.80 → escalate
    assert route(0.50, Thresholds(0.80, 0.30)) == "escalate"


def test_route_boundary_safe_is_pass():
    # p_safe exactly at theta_safe=0.80 → pass (>=)
    assert route(0.80, Thresholds(0.80, 0.30)) == "pass"


def test_route_boundary_unsafe_is_block():
    # p_safe exactly at theta_unsafe=0.30 → block (<=)
    assert route(0.30, Thresholds(0.80, 0.30)) == "block"


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        Thresholds(0.3, 0.8)  # theta_safe must be > theta_unsafe
