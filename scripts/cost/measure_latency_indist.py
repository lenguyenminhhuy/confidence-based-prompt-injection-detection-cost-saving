"""Directly measured per-request latency on a real split (in-distribution check).

Why this exists
---------------
Table tab:indist prices the in-distribution cost rows by looking each row's
rendered length up on the synthetic-filler latency curve (linear interpolation
between 8 measured lengths). This script measures the real detection call on
every row of the split instead, so the cascade / M2-alone totals can be
reported as measured GPU-seconds and the interpolation can be validated.

Protocol (same unit as benchmark_latency_curve.py / cost_protocol.md):
  * batch = 1, one timed unit = Stage1Detector.predict([text]) = two label forwards
  * torch.cuda.synchronize() around every rep, so the timer measures the GPU
  * warm-up reps on synthetic filler are discarded; then each row is timed
    `--reps` times and the per-row MEDIAN is recorded
  * Stage 2 is always NF4; Stage 1 is NF4 or bf16 via --regime
  * one model per invocation, so a crash loses at most one model's run;
    output is a per-row JSONL written incrementally and resumable (--resume)

Payload hygiene: this departs from the filler protocol in that REAL split texts
are fed to the model. Nothing but row index, label, rendered token count,
timings and p_safe is written; no text is printed or stored.

Usage (on the GPU box, from ~/cascade-pid):
    PYTHONPATH=. ~/venv/bin/python scripts/cost/measure_latency_indist.py \
        --split data/train_proposal/val.jsonl --regime nf4 --reps 3 --warmup 8 \
        --model mistral-7b-v0.1:results/stage2/mistral-7b-v0.1/adapter --role stage2 \
        --out results/analysis/indist_latency/val_mistral-7b-v0.1_nf4
Writes <out>.jsonl (per row) and <out>.summary.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.models.prompt_template import format_prompt, load_model_config  # noqa: E402
from src.models.stage1 import Stage1Detector                             # noqa: E402
from scripts.eval.eval_cascade import _read_labels                           # noqa: E402


def _texts(path: str) -> list[str]:
    out = []
    with open(ROOT / path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append(r.get("text") or r.get("input") or "")
    return out


def _filler(n_words: int) -> str:
    return " ".join(f"word{i}" for i in range(n_words))


def _spec(arg: str) -> tuple[str, str | None]:
    name, _, adapter = arg.partition(":")
    return name, (adapter or None)


def _one_rep(det: Stage1Detector, text: str) -> tuple[float, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    row = det.predict([text])[0]
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, row["p_safe"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, help="jsonl with text/input + label fields")
    p.add_argument("--model", required=True, metavar="NAME[:ADAPTER]")
    p.add_argument("--role", choices=("stage1", "stage2"), required=True)
    p.add_argument("--regime", choices=("nf4", "bf16"), default="nf4",
                   help="Stage-1 precision; Stage 2 is forced to NF4")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="time only the first N rows (smoke test)")
    p.add_argument("--resume", action="store_true", help="skip rows already in <out>.jsonl")
    p.add_argument("--out", required=True, help="output stem (no extension)")
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; cost_protocol.md requires the A10G box.")

    name, adapter = _spec(a.model)
    load_in_4bit = True if a.role == "stage2" else (a.regime == "nf4")
    dtype = "nf4" if load_in_4bit else "bf16"

    texts = _texts(a.split)
    y = _read_labels(ROOT / a.split)
    assert len(texts) == len(y), f"{len(texts)} texts vs {len(y)} labels"
    n_total = len(texts)
    idxs = list(range(n_total if not a.limit else min(a.limit, n_total)))

    out_jsonl = Path(str(ROOT / a.out) + ".jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if a.resume and out_jsonl.exists():
        with out_jsonl.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["idx"])
        print(f"resume: {len(done)} rows already timed", flush=True)
    elif out_jsonl.exists() and not a.resume:
        raise SystemExit(f"{out_jsonl} exists; pass --resume or choose another --out")
    todo = [i for i in idxs if i not in done]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    det = Stage1Detector(load_model_config(name),
                         adapter_path=(ROOT / adapter) if adapter else None,
                         load_in_4bit=load_in_4bit)
    print(f"loaded {name} ({dtype}, adapter={adapter}) in {time.perf_counter() - t_load:.1f}s "
          f"on {torch.cuda.get_device_name(0)}; rows to time: {len(todo)}/{len(idxs)}", flush=True)

    # warm-up on synthetic filler at a mid length, discarded
    for _ in range(a.warmup):
        _one_rep(det, _filler(200))

    t_start = time.perf_counter()
    with out_jsonl.open("a") as f:
        for k, i in enumerate(todo, 1):
            text = texts[i]
            ntok = len(det.tokenizer(format_prompt(det.config, text), add_special_tokens=False)["input_ids"])
            samples, p_safe = [], None
            for _ in range(a.reps):
                ms, ps = _one_rep(det, text)
                samples.append(ms)
                p_safe = ps if p_safe is None else p_safe
            samples.sort()
            rec = {"idx": i, "label": int(y[i]), "ntok": ntok,
                   "ms_median": round(statistics.median(samples), 3),
                   "ms_min": round(samples[0], 3), "ms_max": round(samples[-1], 3),
                   "reps": a.reps, "p_safe": p_safe}
            f.write(json.dumps(rec) + "\n")
            if k % 200 == 0 or k == len(todo):
                f.flush()
                el = time.perf_counter() - t_start
                eta = el / k * (len(todo) - k)
                print(f"  {k}/{len(todo)}  elapsed {el/60:.1f} min  eta {eta/60:.1f} min  "
                      f"last median {rec['ms_median']:.1f} ms @ {ntok} tok", flush=True)

    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    # summary over the full file (including resumed rows)
    rows = [json.loads(l) for l in out_jsonl.open() if l.strip()]
    ms = np.array([r["ms_median"] for r in rows], float)
    lab = np.array([r["label"] for r in rows], int)
    nt = np.array([r["ntok"] for r in rows], float)
    spread = np.array([(r["ms_max"] - r["ms_min"]) / r["ms_median"] for r in rows], float)

    def _stats(v: np.ndarray) -> dict:
        return {"n": int(v.size), "mean_ms": float(v.mean()), "median_ms": float(np.median(v)),
                "p90_ms": float(np.percentile(v, 90)), "p95_ms": float(np.percentile(v, 95)),
                "p99_ms": float(np.percentile(v, 99)), "max_ms": float(v.max()),
                "total_sec": float(v.sum() / 1000.0)}

    summary = {
        "model": name, "role": a.role, "dtype": dtype, "adapter": adapter,
        "split": a.split, "n_rows_timed": len(rows), "n_split": n_total,
        "reps_per_row": a.reps, "warmup": a.warmup,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "platform": platform.platform(),
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "peak_vram_gb": peak_gb,
        "regime": f"batch=1, {dtype}, +adapter, per-input = predict() = 2 label forwards, "
                  f"per-row median of {a.reps} reps after {a.warmup} filler warm-ups",
        "all": _stats(ms), "benign": _stats(ms[lab == 0]), "attack": _stats(ms[lab == 1]),
        "tokens": {"median": float(np.median(nt)), "mean": float(nt.mean()),
                   "median_benign": float(np.median(nt[lab == 0])),
                   "median_attack": float(np.median(nt[lab == 1]))},
        "rep_spread_rel": {"median": float(np.median(spread)), "p95": float(np.percentile(spread, 95))},
    }
    out_sum = Path(str(ROOT / a.out) + ".summary.json")
    out_sum.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("model", "dtype", "n_rows_timed", "peak_vram_gb", "all")}, indent=1),
          flush=True)
    print(f"wrote {out_jsonl} and {out_sum}", flush=True)


if __name__ == "__main__":
    main()
