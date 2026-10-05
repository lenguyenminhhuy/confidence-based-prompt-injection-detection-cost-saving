"""Are Stage-1 and Stage-2 failures nested or complementary?

The cascade can only help if the strong stage catches what the weak stage
misses. This script tests that directly on the evaluation attacks, two ways:

  (A) MATCHED-THRESHOLD OVERLAP. Put each stage at its own 1%-FPR operating
      point and cross-tabulate which attacks each one misses. If
      P(S2 miss | S1 miss) >> P(S2 miss), the failures are nested and no
      routing order recovers them.

  (B) CASCADE COUNTERFACTUAL. Stage 1 auto-passes an attack when
      p_safe >= theta_safe; that leak is irrecoverable in deployment. But we
      hold S2 scores for every input, so we can ask what S2 *would* have done
      on exactly those inputs. Low DR there means escalating more cannot fix
      the leak.

Logits + labels + source/channel only. Never emits dataset text: --examples
yields identifiers, scores and hashes, per this repo's payload-hygiene rule.

Usage:
    python scripts/eval/analyze_failure_overlap.py \
        --stage1-dir results/stage1_prec/bf16/qwen2.5-1.5b \
        --stage1-name qwen2.5-1.5b-bf16 \
        --out results/analysis/failure_overlap_qwen_bf16.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPR_TARGET = 0.01
THETA_SAFE = 0.005
STAGE2_DIR = "results/stage2/mistral-7b-v0.1"


def _p_safe(path: Path) -> np.ndarray:
    return np.asarray([json.loads(l)["p_safe"] for l in path.open() if l.strip()], float)


def _meta(path: Path):
    y, src, ch, ids = [], [], [], []
    for line in path.open():
        if not line.strip():
            continue
        d = json.loads(line)
        y.append(int(d["label"]))
        src.append(d.get("source"))
        ch.append(d.get("channel"))
        ids.append(d.get("id"))
    return (np.array(y), np.array(src, object), np.array(ch, object), np.array(ids, object))


def _thr_at_fpr(score: np.ndarray, y: np.ndarray, target: float) -> float:
    """Lowest threshold whose benign FPR <= target (maximises DR subject to FPR)."""
    b = score[y == 0]
    best_t, best_dr = None, -1.0
    a = score[y == 1]
    for t in np.unique(b):
        if (b >= t).mean() <= target:
            dr = (a >= t).mean()
            if dr > best_dr:
                best_t, best_dr = float(t), float(dr)
    return best_t


def _rate(x) -> float:
    x = np.asarray(x)
    return float(x.mean()) if x.size else float("nan")


def analyse(args) -> dict:
    y, src, ch, ids = _meta(ROOT / args.eval_split)
    s1_ps = _p_safe(ROOT / args.stage1_dir / "eval_logits.jsonl")
    s2_ps = _p_safe(ROOT / args.stage2_dir / "eval_logits.jsonl")
    for nm, v in (("stage1", s1_ps), ("stage2", s2_ps)):
        if len(v) != len(y):
            raise SystemExit(f"row mismatch {nm}: {len(v)} vs {len(y)}")

    s1 = 1.0 - s1_ps          # injection score
    s2 = 1.0 - s2_ps
    t1 = _thr_at_fpr(s1, y, FPR_TARGET)
    t2 = _thr_at_fpr(s2, y, FPR_TARGET)

    atk = y == 1
    f1 = s1 >= t1             # flagged by stage 1
    f2 = s2 >= t2
    m1, m2 = ~f1, ~f2         # missed

    A1, A2 = f1[atk], f2[atk]
    M1, M2 = ~A1, ~A2
    n = int(atk.sum())

    both_miss = int((M1 & M2).sum())
    p_m1, p_m2 = _rate(M1), _rate(M2)
    p_both = both_miss / n
    indep = p_m1 * p_m2

    # (A) matched-threshold overlap
    overlap = {
        "n_attacks": n,
        "stage1_threshold": t1, "stage2_threshold": t2,
        "stage1_dr": _rate(A1), "stage2_dr": _rate(A2),
        "contingency_attacks": {
            "both_flag": int((A1 & A2).sum()),
            "s1_only": int((A1 & M2).sum()),
            "s2_only": int((M1 & A2).sum()),
            "both_miss": both_miss,
        },
        "p_s2_miss": p_m2,
        "p_s2_miss_given_s1_miss": _rate(M2[M1]),
        "lift": _rate(M2[M1]) / p_m2 if p_m2 else float("nan"),
        "p_both_miss_observed": p_both,
        "p_both_miss_if_independent": indep,
        "excess_over_independence": p_both / indep if indep else float("nan"),
        "phi_correlation_of_miss_sets": float(np.corrcoef(M1.astype(float), M2.astype(float))[0, 1]),
        "spearman_scores_on_attacks": float(
            np.corrcoef(np.argsort(np.argsort(s1[atk])), np.argsort(np.argsort(s2[atk])))[0, 1]),
        "union_ceiling_dr": _rate(A1 | A2),
        "stage2_rescue_rate_of_s1_misses": _rate(A2[M1]),
        "stage1_rescue_rate_of_s2_misses": _rate(A1[M2]),
    }

    # (B) cascade counterfactual on the real leak set
    leaked = atk & (s1_ps >= args.theta_safe)     # auto-passed attacks
    esc = atk & (s1_ps < args.theta_safe)
    counterfactual = {
        "theta_safe": args.theta_safe,
        "n_leaked_attacks": int(leaked.sum()),
        "leaked_frac_of_attacks": float(leaked.sum() / n),
        "stage2_dr_on_leaked": _rate(f2[leaked]),
        "stage2_dr_on_escalated": _rate(f2[esc]),
        "stage2_dr_overall_attacks": _rate(A2),
        "interpretation": "if stage2_dr_on_leaked << stage2_dr_overall_attacks, "
                          "escalating the leaked inputs would not have recovered them",
    }

    # per-slice
    by_slice = {}
    for key, arr in (("source", src), ("channel", ch)):
        d = {}
        for v in sorted({x for x in arr[atk].tolist() if x is not None}):
            m = atk & (arr == v)
            mm = (~f1[m]) & (~f2[m])
            d[str(v)] = {
                "n": int(m.sum()),
                "stage1_dr": _rate(f1[m]), "stage2_dr": _rate(f2[m]),
                "both_miss_frac": _rate(mm),
                "stage2_rescue_of_s1_misses": _rate(f2[m & ~f1]),
            }
        by_slice[key] = d

    out = {
        "stage1_name": args.stage1_name,
        "stage2_name": "mistral-7b-v0.1",
        "fpr_target": FPR_TARGET,
        "matched_threshold_overlap": overlap,
        "cascade_counterfactual": counterfactual,
        "by_slice": by_slice,
    }

    if args.examples:
        # Payload hygiene: identifiers, scores and hashes only.
        # Never emit dataset text. Inspect payloads with
        #   PYTHONPATH=. python scripts/data/inspect_samples.py <split> ...  in a plain terminal.
        import hashlib
        texts = [json.loads(l).get("text", "") for l in (ROOT / args.eval_split).open() if l.strip()]
        idx = np.where(atk & m1 & m2)[0]
        rng = np.random.default_rng(args.seed)
        pick = sorted(rng.choice(idx, size=min(args.examples, len(idx)), replace=False).tolist())
        out["both_miss_examples"] = [
            {"row": int(i), "id": str(ids[i]), "source": str(src[i]), "channel": str(ch[i]),
             "stage1_injection_score": round(float(s1[i]), 6),
             "stage2_injection_score": round(float(s2[i]), 6),
             "stage1_p_safe": round(float(s1_ps[i]), 6),
             "chars": len(texts[i]),
             "sha256_12": hashlib.sha256(texts[i].encode()).hexdigest()[:12]}
            for i in pick
        ]
        del texts
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-dir", required=True)
    p.add_argument("--stage1-name", required=True)
    p.add_argument("--stage2-dir", default=STAGE2_DIR)
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--theta-safe", type=float, default=THETA_SAFE)
    p.add_argument("--examples", type=int, default=0,
                   help="emit N both-miss example IDENTIFIERS (id/source/scores/sha256; never text)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    r = analyse(a)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2) + "\n")

    o, c = r["matched_threshold_overlap"], r["cascade_counterfactual"]
    ct = o["contingency_attacks"]
    print(f"{r['stage1_name']}  attacks={o['n_attacks']}  "
          f"S1 DR={o['stage1_dr']:.4f}  S2 DR={o['stage2_dr']:.4f}")
    print(f"  contingency: both_flag={ct['both_flag']}  s1_only={ct['s1_only']}  "
          f"s2_only={ct['s2_only']}  BOTH_MISS={ct['both_miss']}")
    print(f"  P(S2 miss)               = {o['p_s2_miss']:.4f}")
    print(f"  P(S2 miss | S1 miss)     = {o['p_s2_miss_given_s1_miss']:.4f}   "
          f"lift = {o['lift']:.3f}x")
    print(f"  P(both miss) observed    = {o['p_both_miss_observed']:.4f}  vs "
          f"independent {o['p_both_miss_if_independent']:.4f}  "
          f"({o['excess_over_independence']:.2f}x)")
    print(f"  phi(miss sets)={o['phi_correlation_of_miss_sets']:.3f}   "
          f"spearman(scores)={o['spearman_scores_on_attacks']:.3f}")
    print(f"  S2 rescues {o['stage2_rescue_rate_of_s1_misses']:.1%} of S1 misses; "
          f"union ceiling DR={o['union_ceiling_dr']:.4f}")
    print(f"  [cascade] leaked={c['n_leaked_attacks']}  "
          f"S2 DR on leaked={c['stage2_dr_on_leaked']:.4f}  vs "
          f"on escalated={c['stage2_dr_on_escalated']:.4f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
