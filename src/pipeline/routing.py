from dataclasses import dataclass
from typing import Literal

RouteDecision = Literal["pass", "block", "escalate"]


@dataclass
class Thresholds:
    theta_safe: float
    theta_unsafe: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta_unsafe < self.theta_safe <= 1.0:
            raise ValueError(
                f"require 0 <= theta_unsafe ({self.theta_unsafe}) "
                f"< theta_safe ({self.theta_safe}) <= 1"
            )


def route(p_safe: float, thresholds: Thresholds) -> RouteDecision:
    if p_safe >= thresholds.theta_safe:
        return "pass"
    if p_safe <= thresholds.theta_unsafe:
        return "block"
    return "escalate"
