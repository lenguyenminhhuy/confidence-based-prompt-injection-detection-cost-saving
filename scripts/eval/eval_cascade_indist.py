"""In-distribution counterpart of the cascade results: validation split, no GPU.

Reads only labels and stored per-row scores (val_logits / cal_logits) for the
three NF4 Stage-1 candidates and M2, plus rendered-token counts for the
length-weighted cost (texts are tokenised and dropped, never printed).

Reports, per Stage-1 model, at the frozen theta_safe (0.005, chosen on cal) and
at the cal-selected log-grid theta_safe where one exists:
  * M2 alone on val: DR@1%FPR, AUROC
  * cascade on val: e, benign skipped, attacks auto-passed, end-to-end DR/FPR,
    paired-bootstrap gap vs M2
  * e re-weighted to the OOD benchmark's benign share (72.4%)
  * length-weighted latency per request on val (own token counts, measured
    NF4 curves) at val prevalence and at the OOD benign share; uniform-512 figure
  * threshold transfer: M2's 1%-FPR threshold frozen on cal -> FPR on val (and
    on eval, to reproduce the OOD figure)

    PYTHONPATH=. python scripts/eval/eval_cascade_indist.py --out results/analysis/indist_cascade
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
from tokenizers import Tokenizer                                   # noqa: E402
from src.models.prompt_template import load_model_config, format_prompt  # noqa: E402
from src.evaluation.metrics import detection_rate_at_fpr           # noqa: E402
from scripts.eval.eval_cascade import _read_logits, _read_labels, _cascade_point, _bootstrap_gap  # noqa: E402
from scripts.eval.score_stage1_logits import auroc                      # noqa: E402

FPR = 0.01
THETA_FROZEN = 0.005
OOD_N_BENIGN, OOD_N = 18634, 25747
OOD_BENIGN_SHARE = OOD_N_BENIGN / OOD_N
STAGE2, STAGE2_DIR = "mistral-7b-v0.1", "results/stage2/mistral-7b-v0.1"
STAGE1 = {
    "qwen2.5-1.5b":        dict(dir="results/stage1/qwen2.5-1.5b", lat="results/analysis/latency_curve_full_nf4.json",  cal_theta=3.16e-4),
    "llama3.2-1b":         dict(dir="results/stage1/llama3.2-1b",  lat="results/analysis/latency_curve_full_nf4.json",  cal_theta=1e-4),
    "granite-guardian-2b": dict(dir="results/stage1/granite-guardian-2b", lat="results/analysis/latency_curve_full3_nf4.json", cal_theta=None),
}
VAL, CAL, EVAL = "data/train_proposal/val.jsonl", "data/train_proposal/cal.jsonl", "data/eval_proposal/eval.jsonl"


def _texts(path):
    out = []
    with open(ROOT / path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line); out.append(r.get("text") or r.get("input") or "")
    return out


def _ntok(texts, model_name, tok_path):
    cfg = load_model_config(model_name); tok = Tokenizer.from_file(str(ROOT / tok_path))
    return np.array([len(e.ids) for e in tok.encode_batch([format_prompt(cfg, t) for t in texts], add_special_tokens=False)], float)


def _curve(per_model, name):
    pts = sorted((v["actual_prompt_tokens"], v["median_ms"]) for v in per_model[name]["by_length"].values())
    xs, ys = {}, {}
    for x, y in pts: xs.setdefault(x, []).append(y)
    x = np.array(sorted(xs)); y = np.array([np.mean(xs[k]) for k in x])
    return x, y


def _lat(per_model, name, ntok):
    x, y = _curve(per_model, name); return np.interp(ntok, x, y)


def _weights(y, benign_share):
    """Per-row weights so that benign rows carry `benign_share` of the mass."""
    w = np.empty(len(y)); b = y == 0
    w[b] = benign_share / b.sum(); w[~b] = (1 - benign_share) / (~b).sum(); return w


def _cost(k1, k2, esc, w):
    base = float((w * k2).sum()); casc = float((w * (k1 + esc * k2)).sum())
    return {"baseline_ms_per_req": base, "cascade_ms_per_req": casc, "reduction": 1 - casc / base,
            "r_effective": float((w * k1).sum() / base)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/analysis/indist_cascade")
    ap.add_argument("--bootstrap", type=int, default=1000); a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)

    y_v = _read_labels(ROOT / VAL); y_c = _read_labels(ROOT / CAL); y_e = _read_labels(ROOT / EVAL)
    _, _, m2_v = _read_logits(ROOT / STAGE2_DIR / "val_logits.jsonl")
    _, _, m2_c = _read_logits(ROOT / STAGE2_DIR / "cal_logits.jsonl")
    _, _, m2_e = _read_logits(ROOT / STAGE2_DIR / "eval_logits.jsonl")
    for arr, yy, nm in ((m2_v, y_v, "val"), (m2_c, y_c, "cal"), (m2_e, y_e, "eval")):
        assert len(arr) == len(yy), f"M2 {nm} row mismatch {len(arr)} vs {len(yy)}"
    s2_v, s2_c, s2_e = 1 - m2_v, 1 - m2_c, 1 - m2_e          # suspicion
    n_v, nb_v, na_v = len(y_v), int((y_v == 0).sum()), int((y_v == 1).sum())

    res = {"split": VAL, "n": n_v, "n_benign": nb_v, "n_attacks": na_v, "benign_share": nb_v / n_v,
           "ood_benign_share": OOD_BENIGN_SHARE, "fpr_target": FPR, "precision": "NF4 both stages"}

    d = detection_rate_at_fpr(y_v, s2_v, (FPR, 0.001))
    res["m2_alone_val"] = {"dr_at_1pct": d[str(FPR)]["dr"], "dr_at_0.1pct": d["0.001"]["dr"], "auroc": float(auroc(y_v, s2_v)),
                           "theta2_val": d[str(FPR)]["threshold"],
                           "max_benign_suspicion": float(s2_v[y_v == 0].max()), "min_attack_suspicion": float(s2_v[y_v == 1].min())}
    # threshold transfer: M2's 1%-FPR cut frozen on cal
    thr_c = detection_rate_at_fpr(y_c, s2_c, (FPR,))[str(FPR)]["threshold"]
    res["m2_threshold_transfer"] = {"theta2_frozen_on_cal": float(thr_c),
                                    "fpr_on_val": float((s2_v[y_v == 0] >= thr_c).mean()), "dr_on_val": float((s2_v[y_v == 1] >= thr_c).mean()),
                                    "fpr_on_eval": float((s2_e[y_e == 0] >= thr_c).mean()), "dr_on_eval": float((s2_e[y_e == 1] >= thr_c).mean())}

    texts = _texts(VAL)
    ntok2 = _ntok(texts, STAGE2, f"{STAGE2_DIR}/adapter/tokenizer.json")
    res["stage1"] = {}
    for name, spec in STAGE1.items():
        _, _, ps_v = _read_logits(ROOT / spec["dir"] / "val_logits.jsonl"); assert len(ps_v) == n_v
        _, _, ps_c = _read_logits(ROOT / spec["dir"] / "cal_logits.jsonl"); assert len(ps_c) == len(y_c)
        lat = json.load(open(ROOT / spec["lat"])); pm = lat["per_model"]
        ntok1 = _ntok(texts, name, f"{spec['dir']}/adapter/tokenizer.json")
        k1 = _lat(pm, name, ntok1); k2 = _lat(pm, STAGE2, ntok2)
        k1_512 = pm[name]["by_length"]["512"]["median_ms"]; k2_512 = pm[STAGE2]["by_length"]["512"]["median_ms"]
        floor1 = float(_curve(pm, name)[0][0])
        row = {"logits_dir": spec["dir"], "latency_file": spec["lat"],
               "stage1_alone_val": {"auroc": float(auroc(y_v, 1 - ps_v)), "dr_at_1pct": detection_rate_at_fpr(y_v, 1 - ps_v, (FPR,))[str(FPR)]["dr"]},
               "tokens_val_stage1": {"median": float(np.median(ntok1)), "mean": float(ntok1.mean()), "p90": float(np.percentile(ntok1, 90)),
                                     "share_at_floor": float((ntok1 <= floor1).mean()), "share_ge_512": float((ntok1 >= 512).mean()),
                                     "median_benign": float(np.median(ntok1[y_v == 0])), "median_attack": float(np.median(ntok1[y_v == 1]))},
               "k_512": {"k1_ms": k1_512, "k2_ms": k2_512, "break_even_e": 1 - k1_512 / k2_512},
               "points": {}}
        thetas = {"frozen": THETA_FROZEN}
        if spec["cal_theta"]: thetas["cal_selected_loggrid"] = spec["cal_theta"]
        for tag, th in thetas.items():
            pt = _cascade_point(ps_v, s2_v, y_v, th, FPR)
            esc = ps_v < th
            skip_b = float((~esc & (y_v == 0)).sum() / nb_v); skip_a = float((~esc & (y_v == 1)).sum() / na_v)
            e_ood = OOD_BENIGN_SHARE * (1 - skip_b) + (1 - OOD_BENIGN_SHARE) * (1 - skip_a)
            w_val = np.full(n_v, 1 / n_v); w_ood = _weights(y_v, OOD_BENIGN_SHARE)
            pt.update({
                "benign_skipped_frac": skip_b, "attacks_autopassed_frac": skip_a,
                "missed_total": int(pt["leaked_attacks"] + (na_v - pt["leaked_attacks"] - pt["caught_escalated"])),
                "e_at_ood_benign_share": e_ood,
                "uniform512_reduction_val_e": 1 - (k1_512 + pt["escalation_rate"] * k2_512) / k2_512,
                "uniform512_reduction_ood_e": 1 - (k1_512 + e_ood * k2_512) / k2_512,
                "length_weighted_val_prevalence": _cost(k1, k2, esc, w_val),
                "length_weighted_ood_prevalence": _cost(k1, k2, esc, w_ood),
                "bootstrap_gap": _bootstrap_gap(ps_v, s2_v, y_v, th, FPR, a.bootstrap, 0) if a.bootstrap else None,
                "cal_point": _cascade_point(ps_c, s2_c, y_c, th, FPR),
            })
            row["points"][tag] = pt
        res["stage1"][name] = row
        print(f"[{name}] val: e={row['points']['frozen']['escalation_rate']:.3f} skip_benign={row['points']['frozen']['benign_skipped_frac']:.3f} "
              f"autopass_attacks={row['points']['frozen']['leaked_attacks']} DR={row['points']['frozen']['e2e_tpr']:.4f} FPR={row['points']['frozen']['e2e_fpr']:.4f} "
              f"e@OOD-share={row['points']['frozen']['e_at_ood_benign_share']:.3f} LW-red(val)={row['points']['frozen']['length_weighted_val_prevalence']['reduction']:+.3f} "
              f"LW-red(OOD-share)={row['points']['frozen']['length_weighted_ood_prevalence']['reduction']:+.3f} med_tok={row['tokens_val_stage1']['median']:.0f}")
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print("M2 alone val:", json.dumps(res["m2_alone_val"])); print("transfer:", json.dumps(res["m2_threshold_transfer"]))
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
