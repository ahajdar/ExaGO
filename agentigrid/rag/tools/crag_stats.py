#!/usr/bin/env python3
"""Paired analysis for the CRAG experiment (paper 2).

Pairs runs by (case, goal, model, repetition), as experiment_stats.py does, and
compares conditions:

  * continuous outcomes (valid_proposal_rate, validator_rejection_rate, and
    cost_improvement_pct on the cost goals) by the two-sided Wilcoxon
    signed-rank test with a bootstrap 95% CI of the median paired difference;
  * per-run binary outcomes (goal_attained; band_narrowed on the voltage goal)
    by the exact McNemar test on the discordant pairs.

Each test is reported pooled over models and per model. Which tests are
confirmatory is fixed by --primary (contrast:metric pairs, Holm-corrected as one
family); everything else is reported with raw p and labelled exploratory. Until
the hypotheses are frozen, leave --primary empty: all tests are then
exploratory, and a Holm correction over all of them is shown as a conservative
reference.

    python rag/tools/crag_stats.py --per-run experiments/crag_v1/analysis/per_run.csv
    python rag/tools/crag_stats.py --per-run ... --primary C2r:C1e:band_narrowed,C2r:C1e:valid_proposal_rate
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("experiment_stats", _HERE / "experiment_stats.py")
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)

SEED = 20261105
COST_GOALS = ("cost10", "n1cost10")
DEFAULT_CONTRASTS = [("C2r", "C1e"), ("C2c", "C1e"), ("C2r", "C2c"), ("C1e", "C1g"), ("C1g", "C0")]
# metric -> (kind, goal filter)
METRICS = {
    "valid_proposal_rate": ("wilcoxon", None),
    "validator_rejection_rate": ("wilcoxon", None),
    "cost_improvement_pct": ("wilcoxon", COST_GOALS),
    "goal_attained": ("mcnemar", None),
    "band_narrowed": ("mcnemar", ("voltage",)),
}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from discordant counts b (treat 1, ctrl 0) and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def test(units, treat, ctrl, metric, model, rng) -> dict:
    kind, goals = METRICS[metric]
    stats = {}
    diffs, _ = es.paired(units, treat, ctrl, metric, goals=goals, model=model, stats=stats)
    out = {"contrast": f"{treat} vs {ctrl}", "treat": treat, "ctrl": ctrl, "metric": metric,
           "model": model or "pooled", "kind": kind, "n_pairs": len(diffs),
           "n_dropped_undefined": stats.get("n_dropped_undefined", 0)}
    if not diffs:
        return {**out, "p": None, "note": "no pairs"}
    if kind == "wilcoxon":
        w = es.wilcoxon(diffs)
        lo, hi = es.boot_median_ci(diffs, rng)
        return {**out, "median_diff": statistics.median(diffs), "mean_diff": statistics.mean(diffs),
                "ci_low": lo, "ci_high": hi, "p": w["p"], "test": w["method"], "n_nonzero": w["n"]}
    b = sum(1 for d in diffs if d > 0)       # treat 1, ctrl 0
    c = sum(1 for d in diffs if d < 0)       # treat 0, ctrl 1
    return {**out, "discordant_treat_only": b, "discordant_ctrl_only": c,
            "diff_in_proportion": (b - c) / len(diffs), "p": mcnemar_exact(b, c), "test": "exact McNemar"}


def analyse(rows, contrasts, primary: set, models=None) -> dict:
    units, excluded = es.index(rows)
    rng = random.Random(SEED)
    models = models or sorted({k[2] for k in units})
    results = []
    for treat, ctrl in contrasts:
        for metric in METRICS:
            for model in [None] + models:
                r = test(units, treat, ctrl, metric, model, rng)
                r["primary"] = (r["model"] == "pooled" and f"{treat}:{ctrl}:{metric}" in primary)
                results.append(r)
    with_p = [r for r in results if r.get("p") is not None]
    prim = [r for r in with_p if r["primary"]]
    for r, a in zip(prim, es.holm([r["p"] for r in prim]) if prim else []):
        r["p_holm_primary"] = a
    for r, a in zip(with_p, es.holm([r["p"] for r in with_p]) if with_p else []):
        r["p_holm_all"] = a
    present = sorted({k for per in units.values() for k in per})
    return {"created": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
            "conditions_present": present, "excluded_cells_baseline_attained": [list(e) for e in excluded],
            "primary": sorted(primary), "status": "confirmatory" if primary else "all exploratory (no --primary)",
            "results": results}


def report(res) -> str:
    lines = [f"Conditions present: {', '.join(res['conditions_present'])}; {res['status']}"]
    for r in res["results"]:
        if r.get("p") is None:
            continue
        tag = "PRIMARY" if r["primary"] else "expl."
        if r["kind"] == "wilcoxon":
            eff = f"median {r['median_diff']:+.3f} [{_f(r['ci_low'])}, {_f(r['ci_high'])}]"
        else:
            eff = (f"discordant {r['discordant_treat_only']}/{r['discordant_ctrl_only']} "
                   f"(Δp {r['diff_in_proportion']:+.3f})")
        holm = r.get("p_holm_primary", r.get("p_holm_all"))
        lines.append(f"  {tag:7s} {r['contrast']:11s} {r['metric']:25s} {r['model']:18s} n={r['n_pairs']:<4d} "
                     f"{eff:36s} p={r['p']:.3g} p_holm={holm:.3g}")
    return "\n".join(lines)


def _f(v):
    return "—" if v is None else f"{v:+.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-run", required=True)
    ap.add_argument("--out", help="output JSON (default: crag_stats.json next to the per-run file)")
    ap.add_argument("--contrasts", help="comma list treat:ctrl (default: C2r:C1e,C2c:C1e,C2r:C2c,C1e:C1g,C1g:C0)")
    ap.add_argument("--primary", default="", help="comma list treat:ctrl:metric forming the confirmatory family")
    a = ap.parse_args(argv)
    rows = es.load_rows(a.per_run)
    contrasts = ([tuple(x.split(":")) for x in a.contrasts.split(",")] if a.contrasts else DEFAULT_CONTRASTS)
    primary = {x.strip() for x in a.primary.split(",") if x.strip()}
    bad = [p for p in primary if len(p.split(":")) != 3 or p.split(":")[2] not in METRICS]
    if bad:
        ap.error(f"bad --primary entries {bad}; metrics: {list(METRICS)}")
    present = {es.cond_key(r["condition"]) for r in rows}
    contrasts = [c for c in contrasts if c[0] in present and c[1] in present]
    if not contrasts:
        ap.error(f"none of the contrasts has both conditions in the data (present: {sorted(present)})")
    res = analyse(rows, contrasts, primary)
    out = Path(a.out) if a.out else Path(a.per_run).with_name("crag_stats.json")
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(report(res))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
