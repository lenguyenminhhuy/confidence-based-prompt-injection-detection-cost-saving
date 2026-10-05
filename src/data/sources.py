"""Pinned HuggingFace revisions for every dataset the builders download.

These are the dataset commits the paper's splits were rebuilt from. Passing
them to ``load_dataset(..., revision=...)`` keeps later upstream edits from
changing the splits.
"""

HF_REVISIONS = {
    # development corpus
    "stingning/ultrachat": "f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "leolee99/NotInject": "847ae76cf8fea5ed325429e569ae8cfef022d2e0",
    "tatsu-lab/alpaca": "dce01c9b08f87459cf36a430d809084718273017",
    "imoxto/prompt_injection_hackaprompt_gpt35": "efc8f438ebe47ecebb602b9442761253d38064be",
    # evaluation benchmark
    "lmsys/lmsys-chat-1m": "200748d9d3cddcc9d782887541057aca0b18c5da",
    "databricks/databricks-dolly-15k": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
    "Muennighoff/natural-instructions": "a29a9757125f4bb1c26445ad0d2ef7d9b2cc9c4c",
    # Stage-2 long benign augmentation
    "abisee/cnn_dailymail": "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
    "ccdv/govreport-summarization": "4e21184e01ae8017e2c036e180fe5e541fef60a0",
    # near-duplicate screen
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}
