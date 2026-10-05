"""Post-hoc probability calibration for Stage-1 (E3) — NO GPU, payload-safe.

Stage-1 emits ``p_safe = sigmoid(logp_benign - logp_injection)``. Under
distribution shift this ranks well but is badly miscalibrated. This module fits *monotone* recalibrators
that leave the ranking untouched and only relocate the probability mass:

  - ``TemperatureScaler`` — one scalar ``T`` fit by NLL on a calibration split;
    ``p_safe_T = sigmoid(z / T)`` where ``z = logp_benign - logp_injection``.
  - ``IsotonicScaler`` — a free-form monotone map ``p_safe -> P(safe)``; included
    to demonstrate that even a *more flexible* monotone calibrator cannot beat
    the rank/discrimination ceiling.

Both are monotone, so on a wide+fine routing grid the cascade cost/quality
frontier is invariant to them (it is a rank-only quantity). They matter only for
placing a *fixed* deployment threshold — that is the practical E3 question.

Label convention (matches src/evaluation/metrics.py + score_stage1_logits.py):
``label == 1`` is injection; the ECE confidence axis is ``p_injection = 1 - p_safe``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # numerically stable logistic sigmoid


def gap_from_logps(logp_benign, logp_injection) -> np.ndarray:
    """Return the logit of ``p_safe``: ``z = logp_benign - logp_injection``."""
    return np.asarray(logp_benign, dtype=float) - np.asarray(logp_injection, dtype=float)


def _nll_injection(T: float, z: np.ndarray, y: np.ndarray) -> float:
    """Mean binary NLL of the injection class under temperature ``T``.

    With ``p_safe = sigmoid(z / T)`` the injection logit is ``-z / T`` and
    ``BCE(logit, y) = softplus(logit) - y * logit`` (stable via ``logaddexp``).
    """
    logit_inj = -z / T
    return float(np.mean(np.logaddexp(0.0, logit_inj) - y * logit_inj))


def fit_temperature(z: np.ndarray, y: np.ndarray, bounds=(1e-2, 1e2)) -> float:
    """Fit a scalar temperature ``T > 0`` minimizing injection NLL.

    ``z`` is the logit gap (``gap_from_logps``); ``y`` is 1=injection / 0=benign.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=int)
    res = minimize_scalar(_nll_injection, args=(z, y), bounds=bounds, method="bounded")
    return float(res.x)


def nll_curvature(z: np.ndarray, y: np.ndarray, T: float, eps: float = 1e-3) -> float:
    """Second derivative of NLL(T) at ``T`` (small = flat surface / weak identifiability).

    A near-flat NLL around the optimum (e.g. a saturated, already-separable cal
    split) means ``T`` is barely constrained and unlikely to transfer under shift.
    """
    f = lambda t: _nll_injection(t, z, y)  # noqa: E731
    return float((f(T + eps) - 2.0 * f(T) + f(T - eps)) / (eps * eps))


class TemperatureScaler:
    """Single-parameter temperature scaling of Stage-1 logits."""

    def __init__(self, T: float = 1.0) -> None:
        self.T = float(T)
        self.fit_nll: float | None = None
        self.n_fit: int | None = None

    def fit(self, logp_benign, logp_injection, labels) -> "TemperatureScaler":
        z = gap_from_logps(logp_benign, logp_injection)
        y = np.asarray(labels, dtype=int)
        self.T = fit_temperature(z, y)
        self.fit_nll = _nll_injection(self.T, z, y)
        self.n_fit = int(len(y))
        return self

    def transform_gap(self, z) -> np.ndarray:
        """Calibrated ``p_safe = sigmoid(z / T)`` from the logit gap."""
        return expit(np.asarray(z, dtype=float) / self.T)

    def transform(self, logp_benign, logp_injection) -> np.ndarray:
        """Calibrated ``p_safe`` from raw logps."""
        return self.transform_gap(gap_from_logps(logp_benign, logp_injection))

    def calibrated_logps(self, logp_benign, logp_injection):
        """Return ``(logp_benign / T, logp_injection / T)``.

        Dividing both logps by ``T`` reproduces ``p_safe = sigmoid(z / T)`` while
        preserving the ``{logp_benign, logp_injection, p_safe}`` file schema so the
        frontier engine consumes calibrated dumps unmodified. NB: the individual
        logps are no longer true log-probabilities — only their difference (and
        thus ``p_safe``) is meaningful downstream.
        """
        lb = np.asarray(logp_benign, dtype=float) / self.T
        li = np.asarray(logp_injection, dtype=float) / self.T
        return lb, li

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "method": "temperature",
            "T": self.T,
            "fit_nll": self.fit_nll,
            "n_fit": self.n_fit,
        }, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "TemperatureScaler":
        d = json.loads(Path(path).read_text())
        s = cls(T=d["T"])
        s.fit_nll = d.get("fit_nll")
        s.n_fit = d.get("n_fit")
        return s


class IsotonicScaler:
    """Free-form monotone calibration ``p_safe -> calibrated p_safe`` (isotonic).

    Fit on ``(p_safe, is_safe)`` where ``is_safe = 1 - injection_label``; increasing
    so higher raw ``p_safe`` maps to higher calibrated ``P(safe)``. Monotone, hence
    still bounded by the rank ceiling — used as a stress test of the calibrator form.
    """

    def __init__(self) -> None:
        from sklearn.isotonic import IsotonicRegression

        self._iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        self.fitted = False

    def fit(self, p_safe, labels) -> "IsotonicScaler":
        p = np.asarray(p_safe, dtype=float)
        is_safe = (np.asarray(labels, dtype=int) == 0).astype(float)
        self._iso.fit(p, is_safe)
        self.fitted = True
        return self

    def transform(self, p_safe) -> np.ndarray:
        return np.asarray(self._iso.predict(np.asarray(p_safe, dtype=float)), dtype=float)
