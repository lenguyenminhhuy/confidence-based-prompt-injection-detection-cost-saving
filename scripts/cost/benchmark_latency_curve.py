"""Measured latency vs. input length for Stage 1 and Stage 2.

Why this exists
---------------
The paper says 512-token inputs save more than shorter inputs. The saved
fraction is 1 - k1/k2 - e, where k1 and k2 are the two stages' per-request
latencies at that length. Those latencies were only ever measured at
64 / 128 / 512 / 1024 / 2048 tokens, so everything between 128 and 512 -- where
a quarter of the evaluation traffic actually lives -- was interpolated. This
script measures the missing points.

It reproduces the protocol the earlier curves were taken under, which lives in
docs/cost_protocol.md:

  * batch = 1 (per-request serving, the only regime the Stage-2 baseline supports)
  * one timed unit = Stage1Detector.predict([text]) on ONE text, which is two
    label forwards (one per label word) -- the same unit for both stages
  * 8 warm-up reps discarded, then 40 timed reps; the median is reported
  * torch.cuda.synchronize() around every rep, so the timer measures the GPU
  * Stage 2 is always NF4; Stage 1 is NF4 or bf16 depending on --regime
  * every model in one output file is measured on the same GPU in the same
    process, so k1 and k2 are comparable (the protocol forbids mixing sessions)

Prompt lengths
--------------
--targets are *target prompt token counts*. Padding text is synthetic filler
("word0 word1 ..."), never dataset text, so no attack payload enters any log.
The chat template has its own length, so short targets floor out: a 32- and a
64-token target both render to ~86 real tokens. The script records both the
target and the actual rendered token count, and downstream analysis keys on the
target (each tokenizer renders the same target slightly differently, so keying
on the actual count would give Stage 1 and Stage 2 disjoint anchors).

The word-count search here lands on the target almost exactly (a 512 target
renders to 512 tokens), whereas the earlier session undershot slightly (its 512
target rendered to 505). So on the re-measured control lengths expect the new
numbers to sit a little ABOVE the old ones -- roughly 1% at 512 -- before
concluding the session drifted.

Usage (on the GPU box, from the repository root):
    PYTHONPATH=. python scripts/cost/benchmark_latency_curve.py \
        --regime nf4 \
        --targets 128,192,256,384,512 \
        --stage1 llama3.2-1b:results/stage1/llama3.2-1b/adapter \
        --stage1 qwen2.5-1.5b:results/stage1/qwen2.5-1.5b/adapter \
        --stage2 mistral-7b-v0.1 \
        --out results/analysis/latency_curve_midgrid_nf4.json

Re-measure at least one length that the existing curves already contain (128
and 512 above). If those come back within ~1% of the old session, the new
points can be merged with the old ones; if they do not, the session drifted and
only the new file is self-consistent.

Adapters: pass the same LoRA adapters the original curves used, for both stages.
The Stage-2 adapter is not in this repository (only its scored logits are), so
supply its path on the box as `--stage2 mistral-7b-v0.1:<path>`. Running Stage 2
without its adapter measures the bare base model, which is slightly faster; that
biases k2 down and so understates every saving. The path actually used is
recorded per model in the output as "adapter".
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from src.models.prompt_template import format_prompt, load_model_config
from src.models.stage1 import Stage1Detector

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = (128, 192, 256, 384, 512)
DEFAULT_REPS = 40
DEFAULT_WARMUP = 8
GPU_HOURLY_USD = 1.20


def _filler(n_words: int) -> str:
    """Synthetic padding. Never dataset text (payload hygiene)."""
    return " ".join(f"word{i}" for i in range(n_words))


def _text_for_target(detector: Stage1Detector, target: int) -> tuple[str, int]:
    """Filler text whose *rendered prompt* is as close to `target` tokens as possible.

    Binary search on word count: the template contributes a fixed prefix, and
    "wordN" is roughly one token, but neither is exact across tokenizers. Returns
    the text and the actual rendered prompt-token count.
    """
    def rendered(n_words: int) -> int:
        prompt = format_prompt(detector.config, _filler(n_words))
        return len(detector.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    if rendered(0) >= target:          # template alone already exceeds the target
        return _filler(0), rendered(0)

    lo, hi = 0, 16
    while rendered(hi) < target:
        lo, hi = hi, hi * 2
    while lo < hi:                     # smallest word count reaching the target
        mid = (lo + hi) // 2
        if rendered(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    # the two candidates that straddle the target; keep the closer one
    best = min((lo - 1, lo), key=lambda n: abs(rendered(max(n, 0)) - target))
    best = max(best, 0)
    return _filler(best), rendered(best)


def _time_one(detector: Stage1Detector, text: str, reps: int, warmup: int) -> dict:
    """Median wall-clock ms of predict([text]) -- two label forwards, batch=1."""
    cuda = detector.device == "cuda"

    def one_rep() -> float:
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        detector.predict([text])
        if cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        one_rep()
    samples = sorted(one_rep() for _ in range(reps))
    return {
        "reps": reps,
        "mean_ms": round(statistics.fmean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "p90_ms": round(samples[min(int(0.9 * reps), reps - 1)], 3),
        "stdev_ms": round(statistics.stdev(samples), 3) if reps > 1 else 0.0,
        "min_ms": round(samples[0], 3),
    }


def _measure_model(name: str, adapter: str | None, *, load_in_4bit: bool,
                   targets: list[int], reps: int, warmup: int, role: str) -> dict:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    detector = Stage1Detector(
        load_model_config(name),
        adapter_path=(ROOT / adapter) if adapter else None,
        load_in_4bit=load_in_4bit,
    )

    by_length: dict[str, dict] = {}
    for target in targets:
        text, actual = _text_for_target(detector, target)
        rec = _time_one(detector, text, reps, warmup)
        rec["target_tokens"] = target
        rec["actual_prompt_tokens"] = actual
        by_length[str(target)] = rec
        print(f"  {name:<18} target={target:<5} actual={actual:<5} "
              f"median={rec['median_ms']:.1f} ms  (sd {rec['stdev_ms']:.2f})",
              flush=True)

    peak_gb = (round(torch.cuda.max_memory_allocated() / 1e9, 2)
               if torch.cuda.is_available() else None)
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "role": role,
        "dtype": "nf4" if load_in_4bit else "bf16",
        "adapter": adapter,
        "peak_vram_gb": peak_gb,
        "by_length": by_length,
    }


def _spec(arg: str) -> tuple[str, str | None]:
    """'name' or 'name:path/to/adapter'."""
    name, _, adapter = arg.partition(":")
    return name, (adapter or None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regime", choices=("nf4", "bf16"), required=True,
                   help="Stage-1 quantization. Stage 2 is NF4 in both regimes.")
    p.add_argument("--stage1", action="append", required=True, metavar="NAME[:ADAPTER]",
                   help="repeatable, e.g. llama3.2-1b:results/stage1/llama3.2-1b/adapter")
    p.add_argument("--stage2", default="mistral-7b-v0.1", metavar="NAME[:ADAPTER]")
    p.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGETS)))
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--usd-per-gpu-hour", type=float, default=GPU_HOURLY_USD)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    targets = [int(t) for t in args.targets.split(",") if t.strip()]
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. cost_protocol.md forbids mixing MPS/CPU latencies "
            "with the CUDA numbers this curve is merged into. Run on the A10G box."
        )
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}   regime: Stage-1 {args.regime} / Stage-2 nf4   "
          f"targets: {targets}   reps: {args.reps} (+{args.warmup} warm-up)\n", flush=True)

    per_model: dict[str, dict] = {}
    for spec in args.stage1:
        name, adapter = _spec(spec)
        per_model[name] = _measure_model(name, adapter,
                                         load_in_4bit=(args.regime == "nf4"),
                                         targets=targets, reps=args.reps,
                                         warmup=args.warmup, role="stage1")
    s2_name, s2_adapter = _spec(args.stage2)
    per_model[s2_name] = _measure_model(s2_name, s2_adapter, load_in_4bit=True,
                                        targets=targets, reps=args.reps,
                                        warmup=args.warmup, role="stage2")

    # k1/k2 at each length, for every Stage-1 candidate. e (escalation rate) is a
    # property of the Stage-1 scores, not of length, so it is not measured here;
    # the cost sweep applies it in scripts/cost/analyze_uniform_length_cost.py.
    k1k2 = {}
    for name, rec in per_model.items():
        if rec["role"] != "stage1":
            continue
        k1k2[name] = {
            str(t): {
                "k1_ms": rec["by_length"][str(t)]["median_ms"],
                "k2_ms": per_model[s2_name]["by_length"][str(t)]["median_ms"],
                "k1_over_k2": round(rec["by_length"][str(t)]["median_ms"]
                                    / per_model[s2_name]["by_length"][str(t)]["median_ms"], 4),
                "breakeven_e": round(1.0 - rec["by_length"][str(t)]["median_ms"]
                                     / per_model[s2_name]["by_length"][str(t)]["median_ms"], 4),
            }
            for t in targets
        }

    out = {
        "gpu": gpu,
        "regime": (f"batch=1, Stage-1 {args.regime} / Stage-2 nf4, +adapter, "
                   "per-input = predict() = 2 label forwards"),
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "gpu_hourly_usd_assumption": args.usd_per_gpu_hour,
        "reps": args.reps,
        "warmup": args.warmup,
        "targets": targets,
        "per_model": per_model,
        "cascade_k1k2": k1k2,
        "note": ("Mid-grid points: fills the 128-512 gap so the "
                 "cost-vs-length curve is measured rather than interpolated. Overlapping "
                 "targets (128, 512) are re-measured here as a session-drift control -- "
                 "compare them against latency_curve_{nf4,bf16stage1}.json before merging. "
                 "Ratio quantities (k1/k2, break-even, reduction) are robust to the "
                 "2-forward scheme since both stages use it; absolute ms would drop under "
                 "a batched/KV-cached server."),
    }

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}")
    for name, rows in k1k2.items():
        print(f"\n{name}:  target  k1_ms    k2_ms    r=k1/k2  break-even e")
        for t in targets:
            r = rows[str(t)]
            print(f"          {t:<6} {r['k1_ms']:<8.1f} {r['k2_ms']:<8.1f} "
                  f"{r['k1_over_k2']:<8.3f} {r['breakeven_e']:.3f}")


if __name__ == "__main__":
    main()
