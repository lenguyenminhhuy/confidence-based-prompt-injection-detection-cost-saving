"""Escalation-threshold (theta_safe) sweep table -- one row per candidate threshold.

Answers "which threshold gives which result" for the cascade: at each theta_safe,
report the escalation rate e (cost axis), end-to-end DR/FPR at a fixed FPR budget,
the attacks Stage 1 auto-passes (leaked -- Stage 2 never sees them), per-channel
recall, and the latency reduction vs Stage-2-on-every-input.

Reuses scripts.eval.eval_cascade._cascade_point so this table cannot drift from
make_figures.py / eval_cascade.py. Payload-safe: reads labels, channel, and
logit scores only -- never input text.

    PYTHONPATH=. python scripts/eval/sweep_theta_safe.py
    PYTHONPATH=. python scripts/eval/sweep_theta_safe.py --log-grid --n-grid 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.eval_cascade import _cascade_point, _select_theta_safe_on_cal  # noqa: E402
from src.evaluation.metrics import detection_rate_at_fpr  # noqa: E402

CHANNELS = ("direct", "document", "tool")
STAGE1 = {"Qwen-1.5B": ("results/stage1/qwen2.5-1.5b", "qwen2.5-1.5b"),
          "Llama-1B": ("results/stage1/llama3.2-1b", "llama3.2-1b")}
STAGE2 = "results/stage2/mistral-7b-v0.1"
DEFAULT_GRID = [1e-5, 3.16e-5, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 5e-3,
                1e-2, 3.16e-2, 1e-1, 3.16e-1, 1.0]


def _psafe(path: Path) -> np.ndarray:
    return np.asarray([json.loads(l)["p_safe"] for l in path.open() if l.strip()], float)


def _meta(path: Path, str_labels: bool):
    y, ch = [], []
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        y.append((0 if r["label"] == "safe" else 1) if str_labels else int(r["label"]))
        ch.append(r.get("channel") or "")
    return np.asarray(y, int), np.asarray(ch, object)


def _k1k2(doc: dict, tokens: int) -> dict:
    """{stage1_name: {k1_ms, k2_ms}} at one request length, from either curve schema.

    The original curves carry a flat ``cascade_k1k2[model] = {k1_ms, k2_ms}`` fixed at
    512 tokens; scripts/cost/benchmark_latency_curve.py nests one entry per measured length.
    Both are read here so a sweep can be re-run against either without editing the paper's
    numbers by hand.
    """
    k = str(tokens)
    block = doc.get("cascade_k1k2") or {}
    first = next(iter(block.values()), None)
    if isinstance(first, dict) and "k1_ms" in first:      # flat, single length
        return block
    if isinstance(first, dict) and k in first:            # nested by target length
        return {m: rows[k] for m, rows in block.items()}
    # fall back to the raw per-model profile
    per = doc["per_model"]
    s2 = next(m for m, spec in per.items() if spec.get("role") == "stage2")
    return {m: {"k1_ms": spec["by_length"][k]["median_ms"],
                "k2_ms": per[s2]["by_length"][k]["median_ms"]}
            for m, spec in per.items() if spec.get("role") == "stage1"}


def sweep(args) -> dict:
    y, ch = _meta(ROOT / args.eval_split, str_labels=False)
    s2 = 1.0 - _psafe(ROOT / STAGE2 / "eval_logits.jsonl")
    lat = _k1k2(json.loads((ROOT / args.latency).read_text()), args.latency_tokens)
    s2_only = float(detection_rate_at_fpr(y, s2, (args.fpr,))[str(args.fpr)]["dr"])

    grid = list(DEFAULT_GRID)
    if args.log_grid:
        grid = sorted(set([float(f"{v:.3g}") for v in np.logspace(-6, 0, args.n_grid)]))
    if args.frozen not in grid:
        grid = sorted(grid + [args.frozen])

    out = {"fpr_target": args.fpr, "stage2_only_dr": s2_only,
           "n": int(len(y)), "n_benign": int((y == 0).sum()), "n_attacks": int((y == 1).sum()),
           "frozen_theta_safe": args.frozen, "sweep": {}}
    print(f"Stage-2-on-every-input: DR={s2_only:.3f} at e=1.00 "
          f"(n={len(y)}, benign={(y == 0).sum()}, attacks={(y == 1).sum()})")

    for name, (sd, latkey) in STAGE1.items():
        s1 = _psafe(ROOT / sd / "eval_logits.jsonl")
        k1, k2 = lat[latkey]["k1_ms"], lat[latkey]["k2_ms"]

        cal_pick = None
        if args.select_on_cal:
            yc, _ = _meta(ROOT / args.cal_split, str_labels=True)
            s1c = _psafe(ROOT / sd / "cal_logits.jsonl")
            s2c = 1.0 - _psafe(ROOT / STAGE2 / "cal_logits.jsonl")
            dr2c = detection_rate_at_fpr(yc, s2c, (args.fpr,))[str(args.fpr)]["dr"]
            cal_pick = _select_theta_safe_on_cal(s1c, s2c, yc, grid, dr2c, args.keep_frac, args.fpr)

        print(f"\n### {name} -> Stage 2   (cal-selected theta_safe = {cal_pick})")
        print(f"{'theta_safe':>11} {'e':>6} {'DR':>7} {'FPR':>7} {'leak':>6} "
              + " ".join(f"{c[:6]:>7}" for c in CHANNELS) + f" {'lat_red':>8}  note")
        rows = []
        for th in grid:
            pt = _cascade_point(s1, s2, y, th, args.fpr)
            pred = np.zeros(len(y), int)
            if pt["theta2"] is not None:
                pred[(s1 < th) & (s2 >= pt["theta2"])] = 1
            pc = {c: float(pred[(y == 1) & (ch == c)].mean()) for c in CHANNELS}
            red = 1.0 - (k1 + pt["escalation_rate"] * k2) / k2
            note = " ".join(t for t in (
                "FROZEN" if abs(th - args.frozen) < 1e-12 else "",
                "CAL-SELECTED" if cal_pick is not None and abs(th - cal_pick) < 1e-12 else "") if t)
            print(f"{th:>11.3g} {pt['escalation_rate']:6.3f} {pt['e2e_tpr']:7.3f} "
                  f"{pt['e2e_fpr']:7.4f} {pt['leaked_attacks']:6d} "
                  + " ".join(f"{pc[c]:7.3f}" for c in CHANNELS) + f" {red * 100:7.1f}%  {note}")
            rows.append({"theta_safe": th, "e": pt["escalation_rate"], "e2e_dr": pt["e2e_tpr"],
                         "e2e_fpr": pt["e2e_fpr"], "leaked_attacks": pt["leaked_attacks"],
                         "theta2": pt["theta2"], "per_channel": pc,
                         "latency_reduction": red, "note": note})
        out["sweep"][name] = {"cal_selected_theta_safe": cal_pick, "rows": rows}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    _emit_tex(out, args.out.with_suffix(".tex"))
    return out



TEX_CAPTION = (
    r"Escalation-threshold ($\theta_{\text{safe}}$) sweep on the evaluation split at a "
    r"fixed $1\%$ end-to-end FPR budget. $e$ is the fraction of inputs routed to "
    r"Stage~2; \emph{leak} counts attacks auto-passed by Stage~1, which Stage~2 never "
    r"sees; \emph{red.} is the measured latency reduction against M2-on-every-input. "
    r"The shipped operating point (\underline{underlined}) was selected on the "
    r"calibration split over a \emph{linear} grid of step $5\times10^{-3}$, so no "
    r"smaller candidate was ever considered. Re-selecting on the same calibration "
    r"split over a log-spaced grid (\textbf{bold}) reaches higher detection at lower "
    r"cost for both Stage~1 options, with no test-set tuning. The $\theta_{\text{safe}}=1$ "
    r"row is escalate-everything, i.e. M2-on-every-input. Generated by "
    r"\texttt{scripts/sweep\_theta\_safe.py}."
)


def _emit_tex(out: dict, path: Path) -> None:
    """Write the sweep table from the same run that produced the JSON.

    The table used to be maintained by hand and had drifted from the JSON (a
    different grid), so figure and table disagreed on which threshold the
    calibration split selects. Deriving both from one run removes that failure mode.
    """
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + TEX_CAPTION + "}",
         r"\label{tab:theta-safe-sweep}", r"\footnotesize",
         r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
         r"\textbf{Stage 1} & $\theta_{\text{safe}}$ & $e$ & \textbf{DR} & \textbf{FPR} "
         r"& \textbf{leak} & \textbf{direct} & \textbf{doc.} & \textbf{tool} "
         r"& \textbf{red.} \\", r"\midrule"]

    for i, (name, arm) in enumerate(out["sweep"].items()):
        if i:
            L.append(r"\midrule")
        for j, r in enumerate(arm["rows"]):
            note = r.get("note") or ""
            cells = [f"{r['theta_safe']:g}", f"{r['e']:.3f}", f"{r['e2e_dr']:.3f}",
                     f"{r['e2e_fpr']:.4f}", f"{r['leaked_attacks']:d}",
                     *(f"{r['per_channel'][c]:.3f}" for c in CHANNELS),
                     f"{r['latency_reduction'] * 100:.1f}\\%"]
            if "CAL-SELECTED" in note:
                cells = [r"\textbf{" + c + "}" for c in cells]
            elif "FROZEN" in note:
                cells = [r"\underline{" + c + "}" for c in cells]
            L.append(f"{name if j == 0 else ''} & " + " & ".join(cells) + r" \\")

    L += [r"\botrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(L))
    print(f"wrote {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--cal-split", default="data/train_proposal/cal.jsonl")
    p.add_argument("--fpr", type=float, default=0.01, help="end-to-end FPR budget")
    p.add_argument("--frozen", type=float, default=0.005, help="shipped theta_safe, marked in output")
    p.add_argument("--log-grid", action="store_true", help="log-spaced grid instead of the default list")
    p.add_argument("--n-grid", type=int, default=73, help="points for --log-grid over 1e-6..1")
    p.add_argument("--keep-frac", type=float, default=0.99)
    p.add_argument("--select-on-cal", action="store_true", default=True,
                   help="also report the cal-frozen pick over this grid")
    p.add_argument("--out", type=Path, default=ROOT / "results/analysis/theta_safe_sweep.json")
    p.add_argument("--latency-tokens", type=int, default=512,
                   help="request length at which k1/k2 are taken for the reduction column")
    p.add_argument("--latency", default="results/analysis/latency_curve_full_nf4.json",
                   help="latency JSON supplying k1/k2 (written by "
                        "scripts/cost/benchmark_latency_curve.py). Take k1 and k2 from the "
                        "same measurement session.")
    return p.parse_args()


if __name__ == "__main__":
    sweep(parse_args())
