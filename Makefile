# Reproduction targets. REPRODUCE.md maps each output to a paper table or figure.
# GPU targets need CUDA (bitsandbytes NF4).
#
#   make setup test          install + unit tests (CPU)
#   make data                download + build every dataset, then check-data (CPU, needs HF_TOKEN)
#   make train score         fine-tune and dump logits (GPU)
#   make latency             latency profiles (A10G GPU)
#   make analysis figures    every paper number from saved logits (CPU)

PY      ?= python
STAGE1  := qwen2.5-1.5b llama3.2-1b granite-guardian-2b
S2      := results/stage2/mistral-7b-v0.1
EVAL    := data/eval_proposal/eval.jsonl
CAL     := data/train_proposal/cal.jsonl
VAL     := data/train_proposal/val.jsonl
A       := results/analysis
TARGETS := 64,128,192,256,384,512,1024,2048
# model:precision:escalation rate:latency curve  (rates from the cascade runs)
LWCOST  := llama3.2-1b:nf4:0.516:full qwen2.5-1.5b:nf4:0.397:full \
           granite-guardian-2b:nf4:0.47582242591369867:full3 \
           llama3.2-1b:bf16:0.444:full qwen2.5-1.5b:bf16:0.377:full \
           granite-guardian-2b:bf16:0.47174428088709364:full3

.PHONY: setup test data check-data train score latency analysis figures all clean

setup:
	pip install -e ".[train,test]"
	hf auth login

test:
	$(PY) -m pytest -q

# ---- data (CPU) -------------------------------------------------------------
data:
	bash scripts/data/download_data.sh
	$(PY) scripts/data/download_hf.py
	$(PY) scripts/data/build_dev_corpus.py
	$(PY) scripts/data/build_eval_benchmark.py
	$(PY) scripts/data/cross_eval_dedup.py
	$(PY) scripts/data/build_stage2_trainset.py
	$(PY) scripts/data/build_length_balanced_trainset.py
	$(PY) scripts/eval/build_e14_reword.py
	$(MAKE) check-data

# The built files must match the ones used in the paper byte for byte.
check-data:
	cd data && shasum -a 256 -c ../scripts/data/SHA256SUMS

# ---- training + scoring (GPU) ----------------------------------------------
train:
	for m in $(STAGE1); do \
	  $(PY) scripts/train/train_stage1.py --config configs/models/$$m.yaml \
	    --training configs/training_16gb.yaml --max-samples 12000 || exit 1; done
	$(PY) scripts/train/train_stage1.py --config configs/models/mistral-7b-v0.1.yaml \
	  --training configs/training_7b_1ep.yaml \
	  --train-file data/train_proposal/train_stage2.jsonl --output-dir $(S2)
	for m in qwen2.5-1.5b llama3.2-1b; do \
	  $(PY) scripts/train/train_stage1.py --config configs/models/$$m.yaml \
	    --training configs/training_lb.yaml \
	    --train-file data/train_proposal/train_stage1_length_balanced.jsonl \
	    --output-dir results/stage1_length_balanced/$$m || exit 1; done

score:
	for m in $(STAGE1); do \
	  $(PY) scripts/train/score_split.py --config configs/models/$$m.yaml \
	    --adapter results/stage1/$$m/adapter --splits $(EVAL) \
	    --output-dir results/stage1/$$m || exit 1; \
	  $(PY) scripts/train/score_split.py --config configs/models/$$m.yaml --no-4bit \
	    --adapter results/stage1/$$m/adapter --splits $(EVAL) $(CAL) \
	    --output-dir results/stage1_prec/bf16/$$m || exit 1; done
	$(PY) scripts/train/score_split.py --config configs/models/mistral-7b-v0.1.yaml \
	  --adapter $(S2)/adapter --splits $(EVAL) data/eval_proposal/e14_reword.jsonl \
	  --output-dir $(S2)
	for m in qwen2.5-1.5b llama3.2-1b; do \
	  $(PY) scripts/train/score_split.py --config configs/models/$$m.yaml \
	    --adapter results/stage1_length_balanced/$$m/adapter --splits $(EVAL) \
	    --output-dir results/stage1_length_balanced/$$m || exit 1; done

latency:
	$(PY) scripts/cost/benchmark_latency_curve.py --regime nf4 --targets $(TARGETS) \
	  --stage1 llama3.2-1b --stage1 qwen2.5-1.5b --out $(A)/latency_curve_full_nf4.json
	$(PY) scripts/cost/benchmark_latency_curve.py --regime bf16 --targets $(TARGETS) \
	  --stage1 llama3.2-1b --stage1 qwen2.5-1.5b --out $(A)/latency_curve_full_bf16.json
	$(PY) scripts/cost/benchmark_latency_curve.py --regime nf4 --targets $(TARGETS) \
	  --stage1 llama3.2-1b --stage1 qwen2.5-1.5b --stage1 granite-guardian-2b \
	  --out $(A)/latency_curve_full3_nf4.json
	$(PY) scripts/cost/benchmark_latency_curve.py --regime bf16 --targets $(TARGETS) \
	  --stage1 llama3.2-1b --stage1 qwen2.5-1.5b --stage1 granite-guardian-2b \
	  --out $(A)/latency_curve_full3_bf16.json
	for m in $(STAGE1); do \
	  $(PY) scripts/cost/measure_latency_indist.py --split $(VAL) --model $$m --role stage1 \
	    --out $(A)/indist_latency/val_$${m}_nf4 || exit 1; done
	$(PY) scripts/cost/measure_latency_indist.py --split $(VAL) --model mistral-7b-v0.1 \
	  --role stage2 --out $(A)/indist_latency/val_mistral-7b-v0.1_nf4

# ---- analysis (CPU, from saved logits) --------------------------------------
analysis:
	$(PY) scripts/eval/fill_stage1_selection.py
	for m in $(STAGE1); do \
	  $(PY) scripts/train/calibrate.py --logits-dir results/stage1/$$m --no-plots \
	    --out-dir $(A)/calibration/$$m || exit 1; \
	  $(PY) scripts/eval/eval_cascade.py --stage1-dir results/stage1/$$m --stage2-dir $(S2) \
	    --stage1-name $$m --stage2-name mistral-7b --out-dir $(A)/cascade_nf4_$$m || exit 1; \
	  $(PY) scripts/eval/eval_cascade.py --stage1-dir results/stage1_prec/bf16/$$m \
	    --stage2-dir $(S2) --stage1-name $$m --stage2-name mistral-7b \
	    --out-dir $(A)/cascade_bf16_$$m || exit 1; \
	  for p in nf4 bf16; do d=results/stage1/$$m; [ $$p = bf16 ] && d=results/stage1_prec/bf16/$$m; \
	    $(PY) scripts/eval/analyze_failure_overlap.py --stage1-dir $$d --stage1-name $$m-$$p \
	      --out $(A)/failure_overlap_$${m}_$$p.json || exit 1; done; done
	$(PY) scripts/eval/eval_cascade_indist.py
	$(PY) scripts/eval/sweep_theta_safe.py
	$(PY) scripts/eval/analyze_by_channel.py --ties-flagged \
	  --stage1 qwen=results/stage1/qwen2.5-1.5b --stage1 llama=results/stage1/llama3.2-1b \
	  --stage1 granite=results/stage1/granite-guardian-2b
	$(PY) scripts/eval/analyze_length_bands.py \
	  --detector qwen=results/stage1/qwen2.5-1.5b \
	  --detector llama=results/stage1/llama3.2-1b \
	  --detector granite=results/stage1/granite-guardian-2b \
	  --detector qwen-lb=results/stage1_length_balanced/qwen2.5-1.5b \
	  --detector llama-lb=results/stage1_length_balanced/llama3.2-1b \
	  --detector m2=$(S2)
	$(PY) scripts/eval/diag_direct_overlap.py --train data/train_proposal/train_stage2.jsonl \
	  --eval $(EVAL) --out $(A)/direct_overlap.json
	$(PY) scripts/eval/verify_e14_unseen.py
	$(PY) scripts/eval/analyze_e14.py --arm-logits $(S2)/e14_reword_logits.jsonl \
	  --eval-logits $(S2)/eval_logits.jsonl --out $(A)/e14_reword_m2.json
	$(PY) scripts/eval/diag_seen_vs_unseen.py --out-dir $(A)/seen_vs_unseen
	for x in $(LWCOST); do \
	  m=$${x%%:*}; r=$${x#*:}; p=$${r%%:*}; r=$${r#*:}; e=$${r%%:*}; c=$${r#*:}; \
	  d=results/stage1/$$m; [ $$p = bf16 ] && d=results/stage1_prec/bf16/$$m; \
	  $(PY) scripts/cost/analyze_length_weighted_cost.py --stage1-dir $$d --stage1-name $$m \
	    --stage1-e $$e --latency $(A)/latency_curve_$${c}_$$p.json --usd-per-gpu-hour 1.2 \
	    --out $(A)/cost_lwfull_$${m}_$$p.json || exit 1; done
	$(PY) scripts/cost/analyze_uniform_length_cost.py \
	  --extra-curve nf4=$(A)/latency_curve_full_nf4.json \
	  --extra-curve bf16=$(A)/latency_curve_full_bf16.json
	$(PY) scripts/cost/analyze_indist_latency_measured.py

figures:
	$(PY) scripts/figures/make_figure_failure_overlap.py
	$(PY) scripts/figures/make_figures.py --only cost_frontier,generalization_gap --formats pdf
	$(PY) scripts/figures/make_figure_theta_sweep.py

all: data train score latency analysis figures

# Removes CPU-derived outputs only; logits and GPU latency files are kept.
clean:
	rm -rf results/figures results/metrics $(A)/cascade_* $(A)/calibration $(A)/failure_overlap_* \
	  $(A)/cost_lwfull_* $(A)/indist_cascade $(A)/seen_vs_unseen
