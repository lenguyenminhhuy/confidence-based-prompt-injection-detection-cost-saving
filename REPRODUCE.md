# Reproducing the paper

Run from the repository root. Outputs go to `data/` and `results/` (not tracked).
`A` below means `results/analysis`.

## 1. Data (CPU)

```bash
HF_TOKEN=... make data      # ends with `make check-data`
```

Steps, in order: `download_data.sh` (GitHub sources at pinned commits),
`download_hf.py` (HackAPrompt mirror), `build_dev_corpus.py`,
`build_eval_benchmark.py`, `cross_eval_dedup.py`, `build_stage2_trainset.py`,
`build_length_balanced_trainset.py`, `scripts/eval/build_e14_reword.py`.
`make check-data` compares every file against `scripts/data/SHA256SUMS`, the
checksums of the files used in the paper. All seven were rebuilt from scratch
and matched.

## 2. Training and scoring (GPU)

```bash
make train    # Stage 1 x3 (configs/training_16gb.yaml, first 12,000 rows),
              # Stage 2 (configs/training_7b_1ep.yaml, train_stage2.jsonl),
              # length-balanced Qwen and Llama (configs/training_lb.yaml)
make score    # eval logits for every model; bf16 Stage-1 eval+cal logits;
              # Stage-2 eval + payload-reword logits
```

Training dumps validation and calibration logits automatically. Every analysis
step below reads only these logit files and the labels, so it runs on CPU.

## 3. Latency (A10G GPU)

```bash
make latency
```

Writes `A/latency_curve_full_{nf4,bf16}.json` (Llama, Qwen, Stage 2; one
session), `A/latency_curve_full3_{nf4,bf16}.json` (second session including
Granite) and `A/indist_latency/val_*_nf4.jsonl` (every validation row timed).

## 4. Analysis and figures (CPU)

```bash
make analysis figures
```

| Paper item | Output | Field |
|---|---|---|
| Table dataset | `data/eval_proposal/eval.jsonl` | rows per `source` |
| Table latency-length | `A/latency_curve_full_*.json`, `A/latency_curve_full3_*.json` | `per_model` medians |
| Table stage1-single | `results/metrics/stage1_selection.json` | `rows.<model>.eval` (AUROC, ECE) |
| Table cascade-headline | `A/cascade_nf4_<model>/cascade_summary.json` | `single_stage.stage2` (M2 row); `headline_cal_frozen.on_eval`, `.bootstrap_eval` |
| Figure overlap | `results/figures/failure_overlap.pdf` | |
| Overlap text (union, rescue rates) | `A/failure_overlap_<model>_{nf4,bf16}.json` | `cascade_counterfactual`, `matched_threshold_overlap` |
| Table indist | `A/indist_cascade/summary.json` | `stage1.<model>.points.frozen` |
| M2 threshold transfer (57.9% FPR) | `A/indist_cascade/summary.json` | `m2_threshold_transfer` |
| Table cost | `A/cost_lwfull_<model>_{nf4,bf16}.json` | `uniform_512_accounting` |
| Figure frontier | `results/figures/cost_frontier_double.pdf` | |
| Length-weighted cost text | `A/cost_lwfull_<model>_{nf4,bf16}.json` | `length_weighted_accounting.reduction` |
| Cost vs uniform length text | `A/cost_uniform_length_sweep.json` | `arms.<regime>/<model>.points` |
| Table indist-cost | `A/indist_cascade/summary.json`, `A/indist_latency/measured_cost.json` | |
| Figure gengap | `results/figures/generalization_gap_double.pdf` | |
| Table cascade-channel | `A/cascade_by_channel.json` | `<model>.by_channel` |
| Leave-one-source-out M2 DR | `A/cascade_by_channel.json` | `m2_dr_without_source` |
| Seen vs unseen document sources | `A/seen_vs_unseen/seen_vs_unseen_document.json` | |
| Tables length-auroc, benignlen | `A/length_bands.json` | |
| Table overlap | `A/direct_overlap.json` | |
| Table reword | `A/e14_reword_m2.json`; tail coverage of rewordings in `A/e14_unseen_check.json` | |
| theta_safe sensitivity (Discussion) | `A/theta_safe_sweep.json` | |
| Table training-runs | `results/stage1/<model>/train_summary.json`, `results/stage2/mistral-7b-v0.1/train_summary.json` | `n_train`, `global_step` |

Run against the saved logits and latency files of the paper run, every output
above matched the saved results of that run (labels and file paths aside).

## Not produced by a script

- Figure pipeline (`pipeline.pdf`): a hand-drawn diagram.
- Table trace: example rows picked by hand. The scores come from
  `results/stage1/qwen2.5-1.5b/eval_logits.jsonl` and the Stage-2 logits; the
  M2 threshold (0.194) is `theta2` in `A/cascade_nf4_qwen2.5-1.5b`.
- "Benchmark share" column of Table latency-length, and the "median request is
  70 tokens / 52.1% at the floor" figures: these count raw input tokens. The
  current cost script counts rendered-prompt tokens (median 144).
- The 3,000-row near-duplicate audit (max cosine 0.899, max Jaccard 0.487).
- The drift check between latency sessions needs the earlier 5-anchor curves
  (`A/latency_curve_nf4.json`, `A/latency_curve_bf16stage1.json`, from
  `scripts/cost/benchmark_latency_curve.py` with its default targets).

## Optional baselines

`make_figures.py` can also draw ROC/PR/channel figures that include
off-the-shelf encoder baselines and DataSentinel. They need prediction files
under `results/baselines/predictions/` and `results/predictions/` that no
script here produces. They are not in the paper and are skipped when absent.
