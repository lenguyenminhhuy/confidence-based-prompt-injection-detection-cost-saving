"""Length-weighted cascade cost on the real evaluation traffic.

The headline cost table fixes one request length (512 tokens) and applies the
closed form 1 - (k1 + e*k2)/k2. Scoring is per-input with no padding to a fixed
length (see src/models/stage1.py::score_labels), so each request actually costs
what its own token count costs. This script re-does the accounting per input:

  baseline = sum_i k2(len_i)
  cascade  = sum_i k1(len_i) + sum_{i escalated} k2(len_i)

k(len) interpolates the measured latency curve (see --latency) and is held flat
outside it. Routing uses the frozen theta_safe and the saved Stage-1 p_safe, so
escalation is each input's real decision, not an average.

`len_i` is the **rendered prompt** length: the request text put through
`format_prompt` (system prompt + template + payload) and tokenized with that
stage's own tokenizer, `add_special_tokens=False` -- exactly what
`benchmark_latency_curve.py` records as `actual_prompt_tokens`. Both sides of
the interpolation therefore speak the same units. An earlier version measured
the raw payload only, which priced every request ~74 tokens too far left on the
curve (the template's own length) and, because Stage 2's curve is much steeper
than Stage 1's, understated k2 more than k1 and so understated the reduction.
Stage 1 and Stage 2 get separate counts: their templates and tokenizers differ,
so the same request renders to different lengths for each.

Logits only + token counts; never prints text. Payload-safe.

Usage:
    PYTHONPATH=. python scripts/cost/analyze_length_weighted_cost.py \
        --stage1-dir results/stage1_prec/nf4/qwen2.5-1.5b \
        --stage1-name qwen2.5-1.5b --stage1-e 0.397 \
        --latency results/analysis/latency_curve_full_nf4.json \
        --out results/analysis/cost_lwfull_qwen2.5-1.5b_nf4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from src.models.prompt_template import format_prompt, load_model_config  # noqa: E402
# Anchors are derived from the latency file itself (see _curve) so a curve
# measured at extra lengths -- e.g. the 64-token point added by the short-input session --
# enters the interpolation instead of being silently ignored.
LENGTH_KEYS = None  # None => use every length present in the latency JSON
STAGE2 = "mistral-7b-v0.1"
THETA_SAFE = 0.005          # frozen on calibration; see leak_anatomy.json
SECONDS_PER_HOUR = 3600.0


def _curve(per_model: dict, name: str):
    """Measured (tokens, ms) anchors, sorted and deduplicated on token count.

    Targets below the prompt template's own length collapse to the same
    actual_prompt_tokens (a 32- and a 64-token target both render to ~86 tokens),
    which would put a duplicate x in np.interp. Such duplicates are averaged into
    one anchor so the curve stays single-valued.
    """
    by_len = per_model[name]["by_length"]
    keys = LENGTH_KEYS if LENGTH_KEYS is not None else sorted(by_len, key=int)
    agg: dict[int, list[float]] = {}
    for k in keys:
        rec = by_len[k]
        agg.setdefault(int(rec["actual_prompt_tokens"]), []).append(float(rec["median_ms"]))
    xs = np.array(sorted(agg), float)
    ys = np.array([sum(agg[int(x)]) / len(agg[int(x)]) for x in xs], float)
    return xs, ys


def _latency(per_model: dict, name: str, ntok: np.ndarray) -> np.ndarray:
    xs, ys = _curve(per_model, name)
    return np.interp(ntok, xs, ys)      # flat below xs[0] and above xs[-1]


def _read_eval(eval_path: Path) -> tuple[list[str], np.ndarray]:
    texts, sources = [], []
    with eval_path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                texts.append(r.get("text") or r.get("input") or "")
                sources.append(r.get("source"))
    return texts, np.asarray(sources, object)


def _rendered_token_counts(texts: list[str], model_name: str,
                           tokenizer_path: Path) -> np.ndarray:
    """Rendered-prompt token count per request, in the latency curve's own units.

    The curve's x-axis is `actual_prompt_tokens` = len(tokenizer(format_prompt(
    config, text), add_special_tokens=False)), so this reproduces that exactly.
    Counting the raw payload instead would omit the template's ~71-75 tokens.
    """
    config = load_model_config(model_name)
    tok = Tokenizer.from_file(str(tokenizer_path))
    prompts = [format_prompt(config, t) for t in texts]
    counts = np.array(
        [len(e.ids) for e in tok.encode_batch(prompts, add_special_tokens=False)],
        float)
    del prompts                                 # drop payloads immediately
    return counts


def _p_safe(path: Path) -> np.ndarray:
    vals = []
    with path.open() as f:
        for line in f:
            if line.strip():
                vals.append(json.loads(line)["p_safe"])
    return np.asarray(vals, float)


def _usd_per_million(total_ms: float, n: int, usd_per_gpu_hour: float) -> float:
    mean_s = total_ms / n / 1000.0
    return mean_s * 1e6 / SECONDS_PER_HOUR * usd_per_gpu_hour


def evaluate(args) -> dict:
    global LENGTH_KEYS
    if args.length_keys:
        LENGTH_KEYS = [k.strip() for k in args.length_keys.split(",")]
    lat = json.load(open(ROOT / args.latency))
    per_model = lat["per_model"]
    usd = args.usd_per_gpu_hour or lat.get("gpu_hourly_usd_assumption", 1.20)

    texts, sources = _read_eval(ROOT / args.eval_split)
    # Each stage renders and tokenizes the same request differently, so each gets
    # its own length array on its own curve.
    ntok1 = _rendered_token_counts(
        texts, args.stage1_name, ROOT / args.stage1_dir / "adapter/tokenizer.json")
    ntok2 = _rendered_token_counts(
        texts, STAGE2, ROOT / args.stage2_dir / "adapter/tokenizer.json")
    del texts                                   # drop payloads immediately
    p_safe = _p_safe(ROOT / args.stage1_dir / "eval_logits.jsonl")
    if len(p_safe) != len(ntok1):
        raise SystemExit(f"row mismatch: {len(p_safe)} logits vs {len(ntok1)} eval rows")

    escalated = p_safe < args.theta_safe
    k1 = _latency(per_model, args.stage1_name, ntok1)
    k2 = _latency(per_model, STAGE2, ntok2)
    n = len(ntok1)

    # Each curve's measured span. Outside it latency is held flat. The lower end is
    # not an approximation: the curve's floor is the rendered template's own length,
    # so no request can be shorter than it.
    xs1, xs2 = _curve(per_model, args.stage1_name)[0], _curve(per_model, STAGE2)[0]
    floor1, top1 = float(xs1[0]), float(xs1[-1])
    floor2, top2 = float(xs2[0]), float(xs2[-1])
    off_curve = float(((ntok1 > top1) | (ntok2 > top2)).mean())

    baseline_ms = float(k2.sum())
    cascade_ms = float(k1.sum() + k2[escalated].sum())
    avoided_ms = float(k2[~escalated].sum())
    router_ms = float(k1.sum())

    # uniform-512 accounting, as reported in the headline cost table
    k1_512 = per_model[args.stage1_name]["by_length"]["512"]["median_ms"]
    k2_512 = per_model[STAGE2]["by_length"]["512"]["median_ms"]
    uniform_reduction = 1.0 - (k1_512 + args.stage1_e * k2_512) / k2_512

    by_source = {}
    for src in sorted(set(sources.tolist())):
        m = sources == src
        by_source[src] = {"n": int(m.sum()),
                          "median_tokens": float(np.median(ntok1[m])),
                          "p90_tokens": float(np.percentile(ntok1[m], 90))}

    return {
        "stage1_name": args.stage1_name,
        "stage2_name": STAGE2,
        "n_eval": n,
        "theta_safe": args.theta_safe,
        "gpu_hourly_usd_assumption": usd,
        "escalation_rate_measured": float(escalated.mean()),
        "escalation_rate_reported": args.stage1_e,
        "inputs": {
            "latency_file": args.latency,
            "latency_measured_utc": lat.get("measured_utc"),
            "length_keys": LENGTH_KEYS or "all present in latency file",
            "stage1_anchors_tokens": [int(x) for x in _curve(per_model, args.stage1_name)[0]],
            "stage2_anchors_tokens": [int(x) for x in _curve(per_model, STAGE2)[0]],
            "stage1_logits": f"{args.stage1_dir}/eval_logits.jsonl",
            "eval_split": args.eval_split,
            "token_count_basis": "rendered prompt (format_prompt + own tokenizer, "
                                 "add_special_tokens=False); matches the curve's "
                                 "actual_prompt_tokens",
        },
        "token_length": {
            "_basis": "rendered prompt tokens, Stage-1 template/tokenizer",
            "median": float(np.median(ntok1)), "mean": float(ntok1.mean()),
            "p10": float(np.percentile(ntok1, 10)),
            "p90": float(np.percentile(ntok1, 90)),
            "min": float(ntok1.min()), "max": float(ntok1.max()),
            "stage1_curve_span_tokens": [floor1, top1],
            "stage2_curve_span_tokens": [floor2, top2],
            "frac_above_curve_top": off_curve,
            "frac_at_or_above_512": float((ntok1 >= 512).mean()),
            "stage2_median": float(np.median(ntok2)),
            "stage2_mean": float(ntok2.mean()),
            "by_source": by_source,
        },
        "uniform_512_accounting": {
            "k1_ms": k1_512, "k2_ms": k2_512,
            "r": k1_512 / k2_512,
            "break_even_e": 1.0 - k1_512 / k2_512,
            "reduction": float(uniform_reduction),
            "usd_per_million_baseline": _usd_per_million(k2_512 * n, n, usd),
            "usd_per_million_cascade": _usd_per_million(
                (k1_512 + args.stage1_e * k2_512) * n, n, usd),
        },
        "length_weighted_accounting": {
            "mean_k1_ms": float(k1.mean()),
            "mean_k2_ms": float(k2.mean()),
            "r_effective": float(k1.mean() / k2.mean()),
            "break_even_e": float(1.0 - k1.mean() / k2.mean()),
            "reduction": float(1.0 - cascade_ms / baseline_ms),
            "mean_k2_on_escalated_ms": float(k2[escalated].mean()),
            "mean_k2_on_autopassed_ms": float(k2[~escalated].mean()),
            "share_of_stage2_work_on_escalated": float(
                k2[escalated].sum() / k2.sum()),
            "usd_per_million_baseline": _usd_per_million(baseline_ms, n, usd),
            "usd_per_million_cascade": _usd_per_million(cascade_ms, n, usd),
            "usd_per_million_saved": _usd_per_million(baseline_ms - cascade_ms, n, usd),
            "usd_per_million_stage2_avoided": _usd_per_million(avoided_ms, n, usd),
            "usd_per_million_router_cost": _usd_per_million(router_ms, n, usd),
        },
        "caveats": [
            f"Curves span {floor1:.0f}-{top1:.0f} tokens (Stage-1) and "
            f"{floor2:.0f}-{top2:.0f} (Stage-2); latency is held flat outside. "
            "No request can fall below the floor -- that is the rendered "
            f"template's own length -- and only {off_curve:.2%} of requests run "
            "past the top anchor, so essentially every request is priced on the "
            "measured part of the curve.",
            "Single-stream (batch=1) only. Batching amortises the fixed overhead "
            "that dominates at short lengths and should recover part of the gap.",
            "Each stage's length is rendered and tokenized with its own template "
            "and tokenizer, so the two curves are read in their own units.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-dir", required=True)
    p.add_argument("--stage1-name", required=True)
    p.add_argument("--stage1-e", type=float, required=True,
                   help="reported escalation rate, for the uniform-512 comparison")
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--stage2-dir", default="results/stage2/mistral-7b-v0.1",
                   help="Stage-2 dir; its adapter/tokenizer.json renders the k2 lengths")
    p.add_argument("--latency",
                   default="results/analysis/latency_curve_full_nf4.json")
    p.add_argument("--length-keys", default=None,
                   help="comma-separated subset of latency anchors to interpolate over "
                        "(default: every length present). Lets one session's curve be "
                        "re-read at another session's anchor set, isolating the effect of "
                        "adding an anchor from session-to-session drift.")
    p.add_argument("--theta-safe", type=float, default=THETA_SAFE)
    p.add_argument("--usd-per-gpu-hour", type=float, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    res = evaluate(args)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")

    lw, un = res["length_weighted_accounting"], res["uniform_512_accounting"]
    print(f"{res['stage1_name']}  n={res['n_eval']}  "
          f"escalation={res['escalation_rate_measured']:.4f}")
    tl = res["token_length"]
    print(f"  rendered request tokens: median {tl['median']:.0f}, "
          f"range {tl['min']:.0f}-{tl['max']:.0f}; "
          f"{tl['frac_at_or_above_512']:.1%} >= 512; "
          f"{tl['frac_above_curve_top']:.2%} past the curve's top anchor")
    print(f"  uniform-512     reduction = {un['reduction']:+.1%}  "
          f"(r={un['r']:.3f}, break-even e={un['break_even_e']:.3f})")
    print(f"  length-weighted reduction = {lw['reduction']:+.1%}  "
          f"(r={lw['r_effective']:.3f}, break-even e={lw['break_even_e']:.3f})")
    print(f"  $/M: baseline={lw['usd_per_million_baseline']:.2f}  "
          f"cascade={lw['usd_per_million_cascade']:.2f}  "
          f"saved={lw['usd_per_million_saved']:.2f}")
    print(f"       stage-2 avoided={lw['usd_per_million_stage2_avoided']:.2f}  "
          f"router cost={lw['usd_per_million_router_cost']:.2f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
