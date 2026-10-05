"""Deployment-point selection — pick one fixed (theta_safe, theta_unsafe).

This module picks the SINGLE operating point a deployed router
would freeze — the min-escalation point on the calibration split that leaks no
attacks and holds the false-positive rate under a target. That fixed point is
then applied to shifted test data (does a threshold chosen on cal transfer?).

Payload-safe: operates on p_safe + labels only. The vectorized routing here
is validated against the canonical ``src.pipeline.routing.route`` in tests/test_calibration.py — kept
local so ``src/`` does not import from ``scripts/``.
"""

from __future__ import annotations

import numpy as np

from src.pipeline.routing import Thresholds


def _route_counts(p_safe: np.ndarray, is_attack: np.ndarray, is_benign: np.ndarray,
                  theta_safe: float, theta_unsafe: float) -> dict:
    """Vectorized pass/block/escalate counts."""
    pass_mask = p_safe >= theta_safe
    block_mask = p_safe <= theta_unsafe
    esc_mask = ~pass_mask & ~block_mask
    return {
        "escalated": int(esc_mask.sum()),
        "leaked_attacks": int((pass_mask & is_attack).sum()),
        "auto_block_FP": int((block_mask & is_benign).sum()),
    }


def choose_operating_point(
    p_safe,
    labels,
    *,
    fpr_target: float,
    safe_grid,
    unsafe_grid,
    require_zero_leak: bool = True,
) -> tuple[Thresholds, dict]:
    """Min-escalation fixed point s.t. (optionally) zero attack-leak and FPR <= target.

    Parameters
    ----------
    p_safe, labels : arrays; ``label == 1`` is injection.
    fpr_target : max tolerated auto-block false-positive rate on benign.
    safe_grid, unsafe_grid : iterables of candidate thresholds (only valid pairs
        ``theta_unsafe < theta_safe`` are considered).
    require_zero_leak : if True, reject any point with a leaked attack.

    Returns
    -------
    (Thresholds, diag) for the feasible point with the lowest escalation rate.

    Raises
    ------
    ValueError : if no in-grid point satisfies the constraints — reports the
        closest miss rather than silently clipping.
    """
    p = np.asarray(p_safe, dtype=float)
    y = np.asarray(labels, dtype=int)
    is_attack = y == 1
    is_benign = y == 0
    n = int(len(p))
    n_benign = int(is_benign.sum())

    best = None  # (escalation_rate, Thresholds, diag)
    # Track the closest infeasible point for a useful error message.
    min_leak = None
    min_fpr_zero_leak = None

    for theta_safe in safe_grid:
        for theta_unsafe in unsafe_grid:
            if theta_unsafe >= theta_safe:
                continue
            c = _route_counts(p, is_attack, is_benign, theta_safe, theta_unsafe)
            leaked = c["leaked_attacks"]
            fpr = c["auto_block_FP"] / n_benign if n_benign else float("nan")
            e = c["escalated"] / n if n else float("nan")
            min_leak = leaked if min_leak is None else min(min_leak, leaked)
            if leaked == 0:
                min_fpr_zero_leak = (
                    fpr if min_fpr_zero_leak is None else min(min_fpr_zero_leak, fpr)
                )
            if require_zero_leak and leaked > 0:
                continue
            if fpr > fpr_target:
                continue
            diag = {
                "theta_safe": float(theta_safe),
                "theta_unsafe": float(theta_unsafe),
                "escalation_rate": e,
                "e2e_fpr_autoblock": fpr,
                "leaked_attacks": leaked,
                "auto_block_FP": c["auto_block_FP"],
                "n": n,
                "n_benign": n_benign,
            }
            if best is None or e < best[0]:
                best = (e, Thresholds(theta_safe, theta_unsafe), diag)

    if best is None:
        raise ValueError(
            "no in-grid operating point met the constraints "
            f"(require_zero_leak={require_zero_leak}, fpr_target={fpr_target:g}); "
            f"closest miss: min leaked_attacks={min_leak}, "
            f"min FPR at zero-leak={min_fpr_zero_leak}. Widen the grid or relax the target."
        )
    return best[1], best[2]
