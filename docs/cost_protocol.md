# Cost measurement protocol

Implementation: `src/evaluation/cost.py` (tested in `tests/test_cost.py`).

Cascade cost: **C(e) = k1 + e·k2**, break-even **e < 1 − k1/k2**, cost reduction
**1 − (k1 + e·k2)/k2**. k1 and k2 are measured Stage-1 and Stage-2 latencies.

Measurement conditions:

1. Batch 1, one request at a time, median of 40 timed repetitions after 8 warm-up
   passes (`scripts/cost/benchmark_latency_curve.py`).
2. k1 and k2 in one formula come from the same GPU in the same session. CPU or
   Apple MPS timings are never mixed with CUDA timings.
3. Latency covers the detector forward pass only; model load and data I/O excluded.
4. Dollar figures, where printed, use an assumed $1.20 per GPU-hour
   (`DEFAULT_GPU_HOURLY_USD`). They scale linearly with that rate. The paper
   reports latency and relative reductions only.
