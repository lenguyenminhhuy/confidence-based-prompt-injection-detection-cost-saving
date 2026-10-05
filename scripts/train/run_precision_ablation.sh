#!/usr/bin/env bash
# Remote-side driver: Stage-1 precision ablation (bf16 vs NF4) on one GPU box.
#
# WHY: Table 6's bf16 Stage-1 block reuses the escalation rate e measured under
# NF4 (see results/analysis/latency_measured_bf16stage1.json: "e": 0.397 /
# 0.516, "k2 reused"). k1 IS properly measured on A10G for both precisions, so
# the ONLY missing quantity is the Stage-1 bf16 *score* distribution, which sets
# both e and DR. Stage 2 never runs here: M2's per-input eval/cal logits are
# already saved and k2 is already measured.
#
# Produces results/stage1_prec/{bf16,nf4}/<model>/{eval,cal}_logits.jsonl.
# Idempotent: an existing non-empty pair is skipped, so it is safe to re-run
# after a disconnect.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python}"
MODELS="${MODELS:-qwen2.5-1.5b llama3.2-1b}"
PRECISIONS="${PRECISIONS:-bf16 nf4}"
cd "$REPO" || { echo "FATAL: no repo at $REPO"; exit 1; }

echo "=== host/GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "FATAL: no nvidia-smi -- this box has no GPU"; exit 1; }
$PY - <<'PYCHK' || exit 1
import torch, sys
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: torch cannot see CUDA"); sys.exit(1)
print("device", torch.cuda.get_device_name(0))
try:
    import bitsandbytes; print("bitsandbytes", bitsandbytes.__version__)
except Exception as e:
    print("WARN: bitsandbytes unavailable ->", e, "(nf4 arm will fail)")
PYCHK

SPLITS="data/eval_proposal/eval.jsonl data/train_proposal/cal.jsonl"
for f in $SPLITS; do [ -s "$f" ] || { echo "FATAL: missing $f"; exit 1; }; done

mkdir -p logs
for prec in $PRECISIONS; do
  case "$prec" in
    bf16) FLAG="--no-4bit" ;;
    nf4)  FLAG="" ;;
    *) echo "skip unknown precision $prec"; continue ;;
  esac
  for m in $MODELS; do
    OUT="results/stage1_prec/$prec/$m"
    ADAPTER=""
    for cand in "results/stage1/$m/adapter"; do
      [ -d "$cand" ] && { ADAPTER="$cand"; break; }
    done
    [ -n "$ADAPTER" ] || { echo "SKIP $prec/$m: no adapter found"; continue; }
    if [ -s "$OUT/eval_logits.jsonl" ] && [ -s "$OUT/cal_logits.jsonl" ]; then
      echo "SKIP $prec/$m: already scored ($(wc -l < "$OUT/eval_logits.jsonl") eval rows)"
      continue
    fi
    mkdir -p "$OUT"
    echo "=== RUN $prec / $m -> $OUT ($(date -u +%FT%TZ)) ==="
    PYTHONPATH=. $PY scripts/train/score_split.py \
      --config "configs/models/$m.yaml" \
      --adapter "$ADAPTER" \
      --splits $SPLITS \
      --output-dir "$OUT" \
      $FLAG 2>&1 | tee "logs/score_${prec}_${m}.log"
    # score_split.py writes {split.stem}_logits.jsonl, i.e. eval_logits.jsonl /
    # cal_logits.jsonl -- already the names eval_cascade.py expects.
    echo "--- $prec/$m rows: eval=$(wc -l < "$OUT/eval_logits.jsonl" 2>/dev/null || echo 0) cal=$(wc -l < "$OUT/cal_logits.jsonl" 2>/dev/null || echo 0)"
  done
done

echo "=== SUMMARY ==="
find results/stage1_prec -name '*_logits.jsonl' -exec sh -c 'echo "$(wc -l < "$1") $1"' _ {} \; | sort -k2
echo "DONE $(date -u +%FT%TZ)"
