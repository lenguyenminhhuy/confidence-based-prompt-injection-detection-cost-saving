"""Stage-1 post-hoc calibration + deployment-threshold selection (E3/E4)."""

from src.calibration.temperature_scaling import (
    IsotonicScaler,
    TemperatureScaler,
    fit_temperature,
    gap_from_logps,
    nll_curvature,
)
from src.calibration.threshold_search import choose_operating_point

__all__ = [
    "TemperatureScaler",
    "IsotonicScaler",
    "fit_temperature",
    "gap_from_logps",
    "nll_curvature",
    "choose_operating_point",
]
