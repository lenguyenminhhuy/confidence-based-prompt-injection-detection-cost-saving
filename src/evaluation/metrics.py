"""
Evaluation metrics for cascade-PID detectors.

Public API: keep signatures stable; scripts and tests depend on them.

Dependencies: numpy only. Python 3.9-compatible syntax.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Re-export per_channel so callers can do `from evaluation.metrics import per_channel`
from src.evaluation.per_channel import per_channel  # noqa: F401

from src.evaluation.cost import detector_cost_block


# ---------------------------------------------------------------------------
# detection_rate_at_fpr
# ---------------------------------------------------------------------------

def detection_rate_at_fpr(
    labels: Sequence[int],
    scores: Sequence[float],
    fpr_targets: Tuple[float, ...] = (0.001, 0.005, 0.01),
) -> Dict[str, dict]:
    """Compute detection rate (TPR) at fixed false positive rates.

    The threshold is chosen on the *benign* score distribution: for each FPR
    target we find the score percentile on benign samples such that at most
    ``fpr_target`` fraction of benign inputs are classified as injections.

    Parameters
    ----------
    labels:
        Binary ground-truth labels (0 = benign, 1 = injection).
    scores:
        Continuous scores where HIGHER means more likely injection. May contain
        ``None`` / ``nan`` — those samples are excluded.
    fpr_targets:
        Iterable of FPR levels at which to evaluate (e.g. 0.001 = 0.1%).

    Returns
    -------
    Dict keyed by string representation of each target FPR, each value a dict::

        {
          "dr": float,            # detection rate (TPR) at that threshold
          "threshold": float,     # score threshold applied
          "achieved_fpr": float,  # actual FPR at the chosen threshold
          "resolvable": bool,     # False when n_benign is too small for target FPR
          "n_benign": int,
          "n_injection": int,
        }

    Notes
    -----
    Reliable estimation of a 0.1% FPR requires ~20 000 benign samples
    (proposal Sec. sizing). When ``n_benign < 1 / fpr_target`` the result is
    marked ``resolvable: false`` and the threshold is set to the lowest benign
    score (i.e. the most conservative possible), so ``dr`` and ``achieved_fpr``
    are still returned but should be interpreted with caution.
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    # Drop samples with NaN scores
    valid_mask = ~np.isnan(scores_arr)
    labels_arr = labels_arr[valid_mask]
    scores_arr = scores_arr[valid_mask]

    benign_scores = scores_arr[labels_arr == 0]
    injection_scores = scores_arr[labels_arr == 1]

    n_benign = len(benign_scores)
    n_injection = len(injection_scores)

    results: Dict[str, dict] = {}

    for target in fpr_targets:
        key = str(target)

        # Minimum benign samples needed to meaningfully estimate this FPR.
        # Rule of thumb: need at least 1/fpr_target benign samples.
        min_needed = int(1.0 / target) if target > 0 else int(1e9)
        resolvable = n_benign >= min_needed

        if n_benign == 0:
            # No benign samples at all — cannot set a threshold
            results[key] = {
                "dr": float("nan"),
                "threshold": float("nan"),
                "achieved_fpr": float("nan"),
                "resolvable": False,
                "n_benign": n_benign,
                "n_injection": n_injection,
            }
            continue

        # Threshold: the (1 - fpr_target) quantile of benign scores.
        # Scores above this threshold are classified as injection.
        # We use the *highest* threshold at which FPR <= target (conservative).
        # np.quantile with interpolation='higher' gives the smallest threshold
        # that keeps FPR at or below the target.
        threshold = float(np.quantile(benign_scores, 1.0 - target, method="higher"))

        # Achieved FPR: fraction of benign samples that exceed the threshold
        achieved_fpr = float(np.mean(benign_scores > threshold))

        # DR: fraction of injection samples that exceed the threshold
        if n_injection == 0:
            dr = float("nan")
        else:
            dr = float(np.mean(injection_scores > threshold))

        results[key] = {
            "dr": dr,
            "threshold": threshold,
            "achieved_fpr": achieved_fpr,
            "resolvable": resolvable,
            "n_benign": n_benign,
            "n_injection": n_injection,
        }

    return results


# ---------------------------------------------------------------------------
# binary_metrics
# ---------------------------------------------------------------------------

def binary_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
) -> Dict[str, float]:
    """Compute standard binary classification metrics with safe division.

    Parameters
    ----------
    labels:
        Ground-truth labels (0 = benign, 1 = injection).
    preds:
        Hard binary predictions (0 or 1).

    Returns
    -------
    Dict with keys: ``f1``, ``precision``, ``recall``, ``tpr``, ``fpr``,
    ``accuracy``.  ``recall`` and ``tpr`` are identical (both provided for
    convenience and compatibility with different naming conventions).
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(preds, dtype=int)

    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    tn = int(np.sum((p == 0) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))

    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den > 0 else float("nan")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    tpr = recall  # TPR = recall by definition

    f1_denom = (precision + recall)
    if not (np.isnan(precision) or np.isnan(recall)) and f1_denom > 0:
        f1 = float(2 * precision * recall / f1_denom)
    else:
        f1 = float("nan")

    fpr = _safe_div(fp, fp + tn)
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# ece
# ---------------------------------------------------------------------------

def ece(
    labels: Sequence[int],
    confidences: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Partitions predictions into ``n_bins`` equal-width bins by confidence,
    then computes the weighted average of |avg_confidence - avg_accuracy|
    across non-empty bins.

    Parameters
    ----------
    labels:
        Ground-truth binary labels (0 or 1).
    confidences:
        Model confidence scores in [0, 1] that the prediction is correct (i.e.
        the probability assigned to the *predicted* class, or equivalently the
        probability of label=1 when label=1 is the positive class).
    n_bins:
        Number of equal-width bins.

    Returns
    -------
    ECE as a float in [0, 1].
    """
    y = np.asarray(labels, dtype=int)
    c = np.asarray(confidences, dtype=float)

    n = len(y)
    if n == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge in last bin
        if i < n_bins - 1:
            mask = (c >= lo) & (c < hi)
        else:
            mask = (c >= lo) & (c <= hi)

        if not np.any(mask):
            continue

        bin_conf = float(np.mean(c[mask]))
        bin_acc = float(np.mean(y[mask]))
        bin_size = int(np.sum(mask))
        ece_val += (bin_size / n) * abs(bin_conf - bin_acc)

    return float(ece_val)


# ---------------------------------------------------------------------------
# evaluate_detector
# ---------------------------------------------------------------------------

def evaluate_detector(
    pred_path: str,
    out_path: str,
    run_mode: str = "smoke",
    eval_manifest: str = "data/eval_proposal/build_manifest.json",
) -> dict:
    """Load a predictions JSONL file, compute all metrics, write metrics JSON.

    Implements the full metrics pipeline.

    Parameters
    ----------
    pred_path:
        Path to ``results/baselines/predictions/<detector>.jsonl``.
    out_path:
        Path to write ``results/baselines/metrics/<detector>.json``.
    run_mode:
        Forwarded into the output JSON as metadata.
    eval_manifest:
        Forwarded into the output JSON as metadata.

    Returns
    -------
    The metrics dict (same as written to ``out_path``).

    Prediction record schema (one JSON object per line)::

        {
          "id": "opi-000123",
          "label": 1,          // 0=benign, 1=injection
          "channel": "document",   // null for benign
          "score": 0.97,       // float or null (binary-only detectors)
          "pred": 1,           // hard decision
          "latency_ms": 41.2
        }
    """
    pred_path = Path(pred_path)
    out_path = Path(out_path)

    records: List[dict] = []
    with pred_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {pred_path}")

    # Derive detector name from filename stem
    detector_name = pred_path.stem

    labels = [r["label"] for r in records]
    preds = [r["pred"] for r in records]
    scores_raw = [r.get("score") for r in records]
    latencies = [r.get("latency_ms") for r in records if r.get("latency_ms") is not None]
    channels = [r.get("channel") for r in records]

    n = len(records)
    n_benign = int(sum(1 for l in labels if l == 0))
    n_injection = int(sum(1 for l in labels if l == 1))

    # Binary metrics
    bin_metrics = binary_metrics(labels, preds)

    # FPR on benign-only slice
    benign_preds = [preds[i] for i in range(n) if labels[i] == 0]
    fpr_benign_only = float(sum(p == 1 for p in benign_preds) / len(benign_preds)) if benign_preds else float("nan")

    # Mean latency
    mean_latency_ms = float(np.mean(latencies)) if latencies else float("nan")

    # DR@FPR — only meaningful when continuous scores are available
    has_scores = any(s is not None for s in scores_raw)
    if has_scores:
        scores_for_dr = [float(s) if s is not None else float("nan") for s in scores_raw]
        dr_at_fpr = detection_rate_at_fpr(labels, scores_for_dr)
    else:
        dr_at_fpr = None  # binary-only detector

    # Per-channel metrics using binary_metrics
    def _channel_metric_fn(recs: List[dict]) -> dict:
        ch_labels = [r["label"] for r in recs]
        ch_preds = [r["pred"] for r in recs]
        return binary_metrics(ch_labels, ch_preds)

    per_channel_results = per_channel(records, _channel_metric_fn)

    metrics_dict: dict = {
        "detector": detector_name,
        "n": n,
        "n_benign": n_benign,
        "n_injection": n_injection,
        "dr_at_fpr": dr_at_fpr,
        "binary": bin_metrics,
        "per_channel": per_channel_results,
        "fpr_benign_only": fpr_benign_only,
        "mean_latency_ms": mean_latency_ms,
        "cost": detector_cost_block(mean_latency_ms, detector=detector_name),
        "run_mode": run_mode,
        "eval_manifest": eval_manifest,
    }

    if dr_at_fpr is None:
        metrics_dict["note"] = (
            "DR@FPR not computed: this detector produces binary verdicts only "
            "(score=null). DR@FPR requires a continuous score to sweep thresholds "
            "on the benign score distribution. Use binary.tpr and binary.fpr for "
            "the single operating-point performance."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_dict, fh, indent=2, allow_nan=True)

    return metrics_dict
