"""Build the length-balanced Stage-1 training split.

Isolates the benign LENGTH distribution at fixed volume and label balance.

Start from train.jsonl's first 12,000 rows (byte-identical to the existing
Stage-1 run) and substitute a random subset of benign rows with LONG benign
drawn from train_stage2.jsonl's augmentation block, until the >=1k-char benign
share matches train_stage2.jsonl (19.03%). n, label balance and every injection
row are unchanged; the only variable is benign length.

PAYLOAD HYGIENE: prints counts/lengths only. Never prints `input`.
"""
import json, random
from pathlib import Path
R = Path(__file__).resolve().parents[2]
SEED = 3131
N = 12000
TARGET_LONG_SHARE = 0.1903

base = [json.loads(l) for i, l in enumerate(open(R/"data/train_proposal/train.jsonl")) if i < N]
# Donor pool: the tail of train_stage2.jsonl, which holds the 5,000 long benign
# augmentation rows. The start offset is the one used for the paper run; the pool
# order feeds rng.sample below, so keep it fixed to reproduce the same file.
aug_pool = [json.loads(l) for i, l in enumerate(open(R/"data/train_proposal/train_stage2.jsonl"))
            if i >= 37987]
aug_long = [d for d in aug_pool if d["label"] == "safe" and len(d["input"]) >= 1000]

ben_idx = [i for i, d in enumerate(base) if d["label"] == "safe"]
n_need = int(round(TARGET_LONG_SHARE * len(ben_idx)))
rng = random.Random(SEED)
swap_at = rng.sample(ben_idx, n_need)
donors = rng.sample(aug_long, n_need)
for pos, don in zip(swap_at, donors):
    base[pos] = don

out = R/"data/train_proposal/train_stage1_length_balanced.jsonl"
with open(out, "w") as f:
    for d in base:
        f.write(json.dumps(d) + "\n")

def prof(rows, tag):
    import statistics as st
    lb = [len(d["input"]) for d in rows if d["label"] == "safe"]
    li = [len(d["input"]) for d in rows if d["label"] != "safe"]
    print("%-14s n=%-6d benign=%-6d inj=%-6d benign_p50=%-5d inj_p50=%-5d benign>=1k=%5.2f%%"
          % (tag, len(rows), len(lb), len(li), st.median(lb), st.median(li),
             100*sum(1 for x in lb if x >= 1000)/len(lb)))

orig = [json.loads(l) for i, l in enumerate(open(R/"data/train_proposal/train.jsonl")) if i < N]
prof(orig, "original"); prof(base, "length-balanced")
print("swapped %d benign rows; donor pool %d long benign" % (n_need, len(aug_long)))
print("wrote", out)
