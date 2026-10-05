"""Measure single-stream (batch=1) inference latency of the real detection call,
on CUDA, for Stage-1 (llama/qwen) and Stage-2 M2 — the k1/k2 the cascade cost
model needs.

Regime = exactly how the reported logits were produced (score_split.py default):
4-bit NF4 on CUDA, PEFT adapter loaded, per-input = detector.predict([text]) which
is TWO forward passes (one per label continuation). We time that call verbatim, so
the number is the cost of the pipeline as evaluated, not an idealized forward.

PAYLOAD HYGIENE: inputs are SYNTHETIC neutral filler at controlled
token lengths. No dataset file is read and no text is printed. Outputs are latency
(ms/request), measured peak VRAM, and the cascade cost quantities from
cost_protocol.md: C(e)=k1+e*k2, break-even 1-k1/k2, reduction 1-(k1+e*k2)/k2,
each in ms, GPU-seconds/1k, and $/1M requests (at $1.20/GPU-hr).

    PYTHONPATH=. ~/venv/bin/python scripts/cost/measure_latency.py \
        --lengths 128,512,1024,2048 --reps 40 --warmup 8 \
        --out results/analysis/latency_measured.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.prompt_template import format_prompt, load_model_config  # noqa: E402
from src.models.stage1 import Stage1Detector  # noqa: E402

GPU_HOURLY_USD = 1.20  # cost_protocol.md DEFAULT_GPU_HOURLY_USD (working assumption)
HEADLINE_LEN = 512      # protocol tokens/request assumption -> anchor for k1/k2

# (config_name, adapter_subdir, role). M2 is a constrained-decode classifier too,
# so Stage1Detector scores it identically.
MODELS = [
    ("llama3.2-1b", "results/stage1/llama3.2-1b/adapter", "stage1"),
    ("qwen2.5-1.5b", "results/stage1/qwen2.5-1.5b/adapter", "stage1"),
    ("mistral-7b-v0.1", "results/stage2/mistral-7b-v0.1/adapter", "stage2"),
]
# measured escalation rates at the cal-frozen operating point (eval_cascade.py)
ESCALATION = {"llama3.2-1b": 0.516, "qwen2.5-1.5b": 0.397}


def make_filler(detector: Stage1Detector, target_tokens: int) -> str:
    """Neutral benign filler whose FULL prompt tokenizes to ~target_tokens."""
    unit = "the meeting notes for tomorrow are ready and entirely ordinary today "
    tok, cfg = detector.tokenizer, detector.config

    def ptoks(t: str) -> int:
        return len(tok(format_prompt(cfg, t), add_special_tokens=False)["input_ids"])

    n = 1
    while ptoks(unit * n) < target_tokens and n < 200000:
        n = int(n * 1.5) + 1
    while n > 1 and ptoks(unit * n) > target_tokens:
        n -= 1
    return unit * n


def time_call(detector: Stage1Detector, text: str, reps: int, warmup: int) -> dict:
    for _ in range(warmup):
        detector.predict([text])
    torch.cuda.synchronize()
    lat_ms = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        detector.predict([text])
        torch.cuda.synchronize()
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
    lat_ms.sort()
    return {
        "reps": reps,
        "mean_ms": round(statistics.mean(lat_ms), 3),
        "median_ms": round(statistics.median(lat_ms), 3),
        "p90_ms": round(lat_ms[int(0.9 * (reps - 1))], 3),
        "stdev_ms": round(statistics.pstdev(lat_ms), 3),
        "min_ms": round(lat_ms[0], 3),
    }


def money(latency_ms: float) -> dict:
    return {
        "latency_ms": round(latency_ms, 3),
        "gpu_seconds_per_1k": round(latency_ms, 3),  # 1000 req * ms/1000 = latency_ms s
        "usd_per_1m": round(latency_ms * GPU_HOURLY_USD / 3.6, 4),  # ms/1000/3600*rate*1e6
    }


def parse_args():
    p = argparse.ArgumentParser(description="Measure batch=1 k1/k2 detection latency on CUDA")
    p.add_argument("--lengths", default="128,512,1024,2048")
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--models", default="llama3.2-1b,qwen2.5-1.5b,mistral-7b-v0.1",
                   help="subset of model names to time")
    p.add_argument("--stage1-4bit", type=int, default=1,
                   help="1=NF4 (parity with scoring), 0=bf16 (deployment-realistic Stage-1)")
    p.add_argument("--k2-ms", type=float, default=None,
                   help="override M2 k2 (median ms @headline len) when M2 not in --models")
    p.add_argument("--out", type=Path, default=ROOT / "results/analysis/latency_measured.json")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    assert torch.cuda.is_available(), "CUDA required for a valid k1/k2 measurement"
    lengths = [int(x) for x in a.lengths.split(",")]
    want = set(a.models.split(","))
    gpu_name = torch.cuda.get_device_name(0)
    per_model = {}

    for name, adapter, role in MODELS:
        if name not in want:
            continue
        use_4bit = True if role == "stage2" else bool(a.stage1_4bit)
        cfg = load_model_config(name)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        det = Stage1Detector(cfg, adapter_path=str(ROOT / adapter), device="cuda",
                             load_in_4bit=use_4bit)
        probe = det.p_safe("Please summarize the meeting notes for tomorrow.")
        assert 0.0 <= probe <= 1.0
        by_len = {}
        for L in lengths:
            text = make_filler(det, L)
            actual = len(det.tokenizer(format_prompt(cfg, text),
                                       add_special_tokens=False)["input_ids"])
            t = time_call(det, text, a.reps, a.warmup)
            t["target_tokens"] = L
            t["actual_prompt_tokens"] = actual
            by_len[str(L)] = t
            print(f"[{name:16s}] len~{L:5d} (act {actual:5d}) "
                  f"median={t['median_ms']:.1f}ms mean={t['mean_ms']:.1f}ms", flush=True)
        peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        per_model[name] = {"role": role, "dtype": "nf4" if use_4bit else "bf16",
                           "peak_vram_gb": peak_gb, "by_length": by_len}
        del det
        torch.cuda.empty_cache()

    # k1/k2 headline at HEADLINE_LEN (median ms)
    def med(name):
        return per_model[name]["by_length"][str(HEADLINE_LEN)]["median_ms"]

    if "mistral-7b-v0.1" in per_model:
        k2 = med("mistral-7b-v0.1")
    elif a.k2_ms is not None:
        k2 = a.k2_ms
    else:
        raise SystemExit("M2 not measured and --k2-ms not given; cannot compute k1/k2")
    cascade = {}
    for s1 in ("llama3.2-1b", "qwen2.5-1.5b"):
        if s1 not in per_model:
            continue
        k1 = med(s1)
        e = ESCALATION[s1]
        c_casc = k1 + e * k2
        cascade[s1] = {
            "k1_ms": round(k1, 3), "k2_ms": round(k2, 3), "e": e,
            "k1_over_k2": round(k1 / k2, 4),
            "breakeven_e": round(1 - k1 / k2, 4),
            "reduction_at_e": round(1 - c_casc / k2, 4),
            "cascade_cost": money(c_casc),
            "guard_cost": money(k2),
        }

    result = {
        "gpu": gpu_name,
        "regime": f"batch=1, Stage-1 {'nf4' if a.stage1_4bit else 'bf16'} / Stage-2 nf4, "
                  f"+adapter, per-input = predict() = 2 label forwards"
                  + ("" if "mistral-7b-v0.1" in per_model else f" (k2 reused={a.k2_ms}ms)"),
        "headline_tokens_per_request": HEADLINE_LEN,
        "gpu_hourly_usd_assumption": GPU_HOURLY_USD,
        "reps": a.reps, "warmup": a.warmup,
        "per_model": per_model,
        "cascade_k1k2": cascade,
        "note": "Ratio k1/k2 (hence break-even and reduction) is robust to the 2-forward "
                "scheme since both stages use it; absolute ms would drop under a batched/"
                "KV-cached server. Batched-throughput regime deferred.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print("\n=== k1/k2 @ %d tokens (median ms, batch=1, NF4) ===" % HEADLINE_LEN)
    for s1, d in cascade.items():
        print(f"  {s1:14s} k1={d['k1_ms']:.1f}  k2={d['k2_ms']:.1f}  "
              f"k1/k2={d['k1_over_k2']:.3f}  breakeven_e={d['breakeven_e']:.3f}  "
              f"e={d['e']}  reduction={d['reduction_at_e']*100:.1f}%")
    print(f"\nwrote -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
