# Datasheet

Two datasets are built: a **development corpus** (training, validation,
calibration) and a source-disjoint **evaluation benchmark**. No source dataset
contributes to both. Every row is English text with a binary label
(benign / injection) and, for injections, a delivery channel.

## Development corpus — `data/train_proposal/`

Built by `scripts/data/build_dev_corpus.py`. 56,795 rows, split 80/10/10.

| Category | Source | train | val | cal |
|---|---|---:|---:|---:|
| Conversational benign | stingning/ultrachat (first user turn) | 7,202 | 903 | 895 |
| | google/IFEval | 429 | 54 | 58 |
| Application-structured benign | tatsu-lab/alpaca | 13,388 | 1,620 | 1,653 |
| | leolee99/NotInject | 266 | 39 | 34 |
| Document-embedded injection | BIPIA (contexts with planted BIPIA attacks) | 7,916 | 1,027 | 984 |
| Tool-output injection | InjecAgent tool-response templates | 840 | 105 | 109 |
| Direct-input injection | HackAPrompt submissions (imoxto mirror) | 456 | 61 | 57 |
| | StruQ-style templates on Alpaca/NotInject carriers | 3,177 | 400 | 423 |
| Encoded injections (all three channels) | base64 / ROT13 / homoglyph variants of the above | 11,675 | 1,515 | 1,509 |
| **Total** | | **45,349** | **5,724** | **5,722** |

Construction:

- Sampling seed 3131; split and link-phrase seed 42.
- Injections longer than 512 whitespace tokens are dropped (76 rows).
- 30% of injections get three encoded variants (source `synthetic-encoded`),
  kept in the same split as their original.
- Exact duplicates (lower-cased, punctuation stripped) are removed (1,219 rows).
- Rows are grouped (an injection with its encoded variants) and groups are
  split 80/10/10. The builder asserts that no text appears in two splits.
- StruQ-style direct injections use the four StruQ templates (plain, "ignore
  previous instructions", escape characters, fake completion) with 60 of 100
  link phrases. The other 40 are reserved for evaluation.

Derived training sets:

- `train_stage2.jsonl` (`build_stage2_trainset.py`): `train.jsonl` plus 5,000
  long benign documents (1,000–5,000 characters) from abisee/cnn_dailymail and
  ccdv/govreport-summarization, screened against the evaluation benchmark.
  50,349 rows. Used for Stage 2 (M2).
- `train_stage1_length_balanced.jsonl` (`build_length_balanced_trainset.py`):
  the first 12,000 rows of `train.jsonl` with 1,068 short benign rows replaced
  by long benign rows from the block above. Same size, labels and injections.

## Evaluation benchmark — `data/eval_proposal/eval.jsonl`

Built by `scripts/data/build_eval_benchmark.py`, then screened by
`scripts/data/cross_eval_dedup.py`. 25,747 rows: 18,634 benign (72.4%) and
7,113 injections (27.6%).

| Category | Source | n |
|---|---|---:|
| Conversational benign | lmsys/lmsys-chat-1m (first English user turn) | 9,452 |
| Application-structured benign | databricks/databricks-dolly-15k | 4,891 |
| | Muennighoff/natural-instructions | 4,291 |
| Document-embedded injection | Open-Prompt-Injection (7 target tasks, 5 attack strategies) | 2,819 |
| Tool-output injection | AgentDojo (4 suites) | 1,464 |
| Direct-input injection | StruQ-style templates on held-out dolly carriers, evaluation link phrases only | 2,830 |

Construction:

- Sampling seed 3131. Exact duplicates removed (1,422 rows), leaving 26,042.
- Near-duplicate screen against the development training split: a row is
  removed if its character 5-gram Jaccard similarity to a training row is
  ≥ 0.5, or its MiniLM (all-MiniLM-L6-v2) cosine similarity is ≥ 0.95.
  This removes 295 rows (`dedup_removed.jsonl`).
- The direct-input set shares templates and payload wording with the
  development data by design; only carriers and link phrases are held out.

Payload-reword test set: `e14_reword.jsonl` (`scripts/eval/build_e14_reword.py`)
holds the 2,830 direct-input attacks in three arms (original, reworded payload,
another seen payload), with carrier, template and link phrase fixed.

## Sources

All sources are fetched at pinned revisions: Hugging Face datasets in
`src/data/sources.py`, GitHub repositories in `scripts/data/download_data.sh`.
Each keeps its original licence; check the source before redistributing.

## Known limitations

- English only, single-turn text only.
- Category and source are confounded in the evaluation benchmark: each
  category comes from one source.
- Development benign rows are short (median 72 characters, none over 1,000),
  while injections are long (median 831). See the length-balanced ablation.
- The HackAPrompt mirror exposes one combined `text` field per submission.
