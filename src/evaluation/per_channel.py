"""
Per-channel metric aggregation for cascade-PID evaluation.

Public API used by the evaluation scripts.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional


def per_channel(
    records: List[dict],
    metric_fn: Callable[[List[dict]], dict],
) -> Dict[str, dict]:
    """Compute a metric breakdown by injection channel plus an overall aggregate.

    Parameters
    ----------
    records:
        List of prediction records. Each record must have a ``"channel"`` key
        (string or ``None``). Records with ``channel=None`` are benign and are
        included only in the ``"overall"`` key.
    metric_fn:
        A callable that receives a list of records and returns a metrics dict.
        Example: ``lambda recs: binary_metrics([r["label"] for r in recs],
                                               [r["pred"] for r in recs])``

    Returns
    -------
    Dict with one key per observed channel (e.g. ``"document"``, ``"tool"``,
    ``"direct"``) plus an ``"overall"`` key covering all records.

    Example::

        {
          "document": {"f1": 0.91, ...},
          "tool":     {"f1": 0.85, ...},
          "direct":   {"f1": 0.88, ...},
          "overall":  {"f1": 0.89, ...},
        }
    """
    # Group records by channel (None → treated as benign, included in overall only)
    by_channel: Dict[str, List[dict]] = {}
    for rec in records:
        ch = rec.get("channel")
        if ch is not None:
            by_channel.setdefault(ch, []).append(rec)

    result: Dict[str, dict] = {}

    for channel, recs in sorted(by_channel.items()):
        result[channel] = metric_fn(recs)

    # Overall across all records (all channels + benign)
    result["overall"] = metric_fn(records)

    return result
