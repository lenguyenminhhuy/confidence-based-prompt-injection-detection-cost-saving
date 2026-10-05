# Cost-Efficient Prompt Injection Detection through Confidence-Based Two-Stage Cascading

Research code for a two-stage prompt-injection detector. A small fine-tuned
language model (Stage 1) screens every request and clears only the inputs it
judges safe with high confidence. Everything else is deferred to a larger
detector (Stage 2), which makes every blocking decision.

The design is one-sided: Stage 1 can allow, never deny. All blocking stays with
the stronger model, while confidently benign traffic skips the expensive path.

## What is here

| Path | Contents |
|---|---|
| `scripts/data/` | Download sources; build the development corpus, evaluation benchmark and augmented training sets |
| `scripts/train/` | Fine-tune Stage 1 / Stage 2 (QLoRA), dump logits, temperature calibration |
| `scripts/eval/` | Cascade evaluation, per-channel and length-band analyses, payload-reword test |
| `scripts/cost/` | Latency measurement and the cost model |
| `scripts/figures/` | Paper figures |
| `src/` | Shared code: prompt template, Stage-1 scoring, metrics, cost model, data sources |
| `configs/` | Model and training configuration |
| `docs/cost_protocol.md` | How latency and cost are measured |
| `tests/` | Unit tests (CPU) |
| `datasheet.md` | Sources, counts and construction of every split |
| `REPRODUCE.md` | Which command produces which table and figure |

## Requirements

- Python 3.10 or 3.11.
- CPU only: data build, all analysis from saved logits, figures, tests.
- NVIDIA GPU with CUDA for training, scoring and latency. Stage-2 fine-tuning
  needs a 24 GB-class GPU; latency was measured on one A10G at batch size 1.
- A Hugging Face token (`HF_TOKEN`, see `.env.example`) with access to the gated
  `lmsys/lmsys-chat-1m` dataset and the Llama 3.2 and Mistral base models.

```bash
make setup     # pip install -e ".[train,test]", then hf auth login
make test
```

## Data

All splits are rebuilt from public datasets at pinned revisions. Each source
keeps its own licence, so the built splits are not redistributed here.

```bash
HF_TOKEN=... make data
```

`datasheet.md` lists every source and count. The builders are deterministic;
`REPRODUCE.md` gives the checksums of the files used in the paper.

## Pipeline

```bash
make data       # CPU
make train      # GPU: 3 Stage-1 models, Stage 2, 2 length-balanced ablation runs
make score      # GPU: per-row logits for every split the analysis reads
make latency    # GPU (A10G): latency profiles
make analysis   # CPU: every table, from saved logits and latency files
make figures    # CPU
```

## Models

Stage 1: Llama 3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, Granite Guardian 3.0-2B.
Stage 2 (M2): Mistral-7B-v0.1. All are fine-tuned with QLoRA (rank-16 adapters
over 4-bit NF4 base weights) as constrained-label classifiers.

## Licence

MIT. See `LICENSE`. The source datasets remain under their own licences.
