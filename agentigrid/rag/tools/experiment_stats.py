#!/usr/bin/env python3
"""Pre-registered hypothesis tests for the RAG ablation (paper Section IV-G, Table VI).

Reads the evaluator's per-run table (experiment_eval.py -> analysis/per_run.csv)
and runs exactly the contrasts fixed before the of-record runs. Nothing here is
chosen after seeing results; exploratory analyses belong elsewhere.

Pairing. Runs are matched across conditions by (case, goal, model, rep): the
same task, the same model and the same repetition index. Only runs with status
"ok" are used. A cell whose baseline already attains the goal
(baseline_attained = 1) measures nothing and is excluded (and listed).

Tests.
  * H1 (primary family, pooled over models; Holm-corrected together):
      C1 vs C0  on valid_proposal_rate, goal_attained (goals with a predicate)
                and cost_improvement_pct (cost goals: the primary cost outcome);
      C2a vs C1 on valid_proposal_rate (two-sided; C2a acts through refinement only).
    Each: paired Wilcoxon signed-rank test (zero differences dropped; exact null
    for n <= 25 without ties, else normal approximation with tie and continuity
    correction), two-sided, Holm step-down at alpha = 0.05; effect = median of
    the paired differences with a 95% percentile-bootstrap CI (fixed seed); for the
    0/1 goal_attained the mean difference (difference in attainment proportions)
    is reported alongside, since its median difference is usually 0.
  * Same contrasts per model (secondary family, Holm within it): the capability
    floor of H2 is read from which models show an effect.
  * H2 interaction: for each retrieval condition vs C0, do the paired
    differences in valid_proposal_rate differ between models? Kruskal-Wallis
    across models (tie-corrected chi-square approximation).
  * H3 (per model, each retrieval condition vs C0; amended before the of-record
    runs). Primary: ABSOLUTE reduction AR = mean(C0) - mean(condition), in rate
    points, for validator_rejection_rate and for intent_violation_rate over the
    same matched units (goals with intent checks); delta = AR_val - AR_int with a
    95% paired-bootstrap CI (resampling matched units). Verdict: "supported" if
    the CI lies above 0; "rejected" if AR_int >= AR_val (delta <= 0);
    otherwise "inconclusive". Always defined. Secondary: the relative reduction
    RR = 1 - mean(condition) / mean(C0), reported where C0's mean is not 0.

Dependency-free (standard library only), deterministic (fixed bootstrap seed).

    python rag/tools/experiment_stats.py --per-run experiments/ofrecord_v2/analysis/per_run.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

ALPHA = 0.05
BOOT = 5000
SEED = 20260929
COST_GOALS = ("cost10", "n1cost10")

# (hypothesis, treatment, control, metric, goal filter or None)
PRIMARY = [
    ("H1", "C1", "C0", "valid_proposal_rate", None),
    ("H1", "C1", "C0", "goal_attained", None),
    ("H1", "C1", "C0", "cost_improvement_pct", COST_GOALS),
]
RETRIEVAL = ("C1",)   # C2a/C2b were assessed by the retrieval preview and not run


# ----------------------------------------------------------------------------- statistics

def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _ranks(values):
    """Average ranks (1-based) with ties sharing the mean rank; also tie-group sizes."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, ties = [0.0] * len(values), []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank test. Returns dict(n, w_plus, p, method)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "p": 1.0, "method": "all differences zero"}
    ranks, ties = _ranks([abs(x) for x in d])
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    if n <= 25 and not ties:
        # exact null distribution of W+ over the integer ranks 1..n
        counts = [1] + [0] * (n * (n + 1) // 2)
        for k in range(1, n + 1):
            for s in range(len(counts) - 1, k - 1, -1):
                counts[s] += counts[s - k]
        total = 2 ** n
        w = int(round(w_plus))
        lo = sum(counts[: w + 1]) / total
        hi = sum(counts[w:]) / total
        return {"n": n, "w_plus": w_plus, "p": min(1.0, 2 * min(lo, hi)), "method": "exact"}
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t ** 3 - t for t in ties) / 48.0
    if var <= 0:
        return {"n": n, "w_plus": w_plus, "p": 1.0, "method": "degenerate"}
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var)
    return {"n": n, "w_plus": w_plus, "p": min(1.0, 2 * _norm_sf(max(z, 0.0))), "method": "normal"}


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def _regularized_gamma_q(a: float, x: float) -> float:
    """Q(a, x) = upper regularized incomplete gamma (Numerical Recipes gser/gcf)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, delta = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1
            delta *= x / ap
            s += delta
            if abs(delta) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def kruskal(groups):
    """Kruskal-Wallis H test across groups (lists). Returns dict(h, df, p)."""
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return {"h": None, "df": None, "p": None}
    allv = [v for g in groups for v in g]
    n = len(allv)
    ranks, ties = _ranks(allv)
    h, pos = 0.0, 0
    for g in groups:
        rs = ranks[pos: pos + len(g)]
        pos += len(g)
        h += sum(rs) ** 2 / len(g)
    h = 12.0 / (n * (n + 1)) * h - 3 * (n + 1)
    corr = 1 - sum(t ** 3 - t for t in ties) / (n ** 3 - n) if n > 1 else 1
    if corr <= 0:
        return {"h": 0.0, "df": len(groups) - 1, "p": 1.0}
    h /= corr
    df = len(groups) - 1
    return {"h": h, "df": df, "p": _regularized_gamma_q(df / 2.0, h / 2.0)}


def boot_median_ci(diffs, rng, reps=BOOT):
    if not diffs:
        return None, None
    n = len(diffs)
    meds = sorted(statistics.median(rng.choices(diffs, k=n)) for _ in range(reps))
    return meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1]


# ----------------------------------------------------------------------------- data

def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = str(v).strip()
    if s in ("", "None", "nan", "NaN"):
        return None
    if s in ("True", "False"):
        return 1.0 if s == "True" else 0.0
    try:
        return float(s)
    except ValueError:
        return None


def cond_key(c: str) -> str:
    """'C1-basic' -> 'C1' (spec ids carry a descriptive suffix)."""
    return str(c).split("-")[0]


def load_rows(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"]


def index(rows):
    """{(case, goal, model, rep): {cond: row}} and the set of excluded cells."""
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["case"], r["goal"], r["model"]), []).append(r)
    excluded = sorted(k for k, rs in by_cell.items()
                      if any(_num(r.get("baseline_attained")) == 1.0 for r in rs))
    units = {}
    for r in rows:
        cell = (r["case"], r["goal"], r["model"])
        if cell in excluded:
            continue
        units.setdefault(cell + (str(r["rep"]),), {})[cond_key(r["condition"])] = r
    return units, excluded


def paired(units, treat, ctrl, metric, goals=None, model=None):
    diffs, keys = [], []
    for k, per in sorted(units.items()):
        if goals and k[1] not in goals:
            continue
        if model and k[2] != model:
            continue
        if treat in per and ctrl in per:
            a, b = _num(per[treat].get(metric)), _num(per[ctrl].get(metric))
            if a is not None and b is not None:
                diffs.append(a - b)
                keys.append(k)
    return diffs, keys


def contrast(units, spec, rng, model=None):
    hyp, treat, ctrl, metric, goals = spec
    diffs, _ = paired(units, treat, ctrl, metric, goals, model)
    w = wilcoxon(diffs)
    lo, hi = boot_median_ci(diffs, rng)
    return {"hypothesis": hyp, "contrast": f"{treat} vs {ctrl}", "metric": metric,
            "goals": ",".join(goals) if goals else "all", "model": model or "pooled",
            "n_pairs": len(diffs), "n_nonzero": w["n"],
            "median_diff": statistics.median(diffs) if diffs else None,
            "mean_diff": statistics.mean(diffs) if diffs else None,   # for 0/1 outcomes: difference in proportions
            "ci_low": lo, "ci_high": hi, "p": w["p"], "test": w["method"]}


def h3(units, treat, model, rng):
    vals = []
    for k, per in sorted(units.items()):
        if k[2] != model or treat not in per or "C0" not in per:
            continue
        v = [_num(per[c].get(m)) for c in (treat, "C0")
             for m in ("validator_rejection_rate", "intent_violation_rate")]
        if None not in v:
            vals.append(v)          # [val_t, int_t, val_0, int_0]

    def ar(sample):
        av = statistics.mean(x[2] for x in sample) - statistics.mean(x[0] for x in sample)
        ai = statistics.mean(x[3] for x in sample) - statistics.mean(x[1] for x in sample)
        return av, ai, av - ai

    def rr(sample):
        mv0 = statistics.mean(x[2] for x in sample)
        mi0 = statistics.mean(x[3] for x in sample)
        rv = 1 - statistics.mean(x[0] for x in sample) / mv0 if mv0 else None
        ri = 1 - statistics.mean(x[1] for x in sample) / mi0 if mi0 else None
        return rv, ri

    out = {"hypothesis": "H3", "contrast": f"{treat} vs C0", "model": model, "n_pairs": len(vals)}
    if not vals:
        return {**out, "verdict": "no data"}
    av, ai, d = ar(vals)
    boots = sorted(ar(rng.choices(vals, k=len(vals)))[2] for _ in range(BOOT))
    lo, hi = boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT) - 1]
    verdict = ("rejected" if d <= 0 else "supported" if lo > 0 else "inconclusive")
    rv, ri = rr(vals)
    return {**out, "ar_validator": av, "ar_intent": ai, "delta": d, "ci_low": lo, "ci_high": hi,
            "rr_validator": rv, "rr_intent": ri, "verdict": verdict}


def analyse(rows) -> dict:
    rng = random.Random(SEED)
    units, excluded = index(rows)
    models = sorted({k[2] for k in units})
    primary = [contrast(units, s, rng) for s in PRIMARY]
    for r, p in zip(primary, holm([x["p"] for x in primary])):
        r["p_holm"] = p
        r["significant"] = p < ALPHA
    per_model = [contrast(units, s, rng, m) for m in models for s in PRIMARY]
    for r, p in zip(per_model, holm([x["p"] for x in per_model])):
        r["p_holm"] = p
        r["significant"] = p < ALPHA
    h2 = []
    for t in RETRIEVAL:
        groups = [paired(units, t, "C0", "valid_proposal_rate", None, m)[0] for m in models]
        kw = kruskal(groups)
        h2.append({"hypothesis": "H2", "contrast": f"{t} vs C0 differs by model", "metric": "valid_proposal_rate",
                   "models": models, "n_per_model": [len(g) for g in groups], **kw})
    h3r = [h3(units, t, m, rng) for m in models for t in RETRIEVAL]
    return {"created": datetime.now().isoformat(), "alpha": ALPHA, "bootstrap": BOOT, "seed": SEED,
            "n_runs_ok": len(rows), "n_units": len(units), "models": models,
            "excluded_cells_baseline_attained": [list(c) for c in excluded],
            "primary": primary, "per_model": per_model, "h2_interaction": h2, "h3": h3r}


def _f(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def report(res) -> str:
    L = [f"{res['n_runs_ok']} ok runs, {res['n_units']} matched units, models: {', '.join(res['models'])}"]
    if res["excluded_cells_baseline_attained"]:
        L.append("Excluded (baseline already attains the goal): " +
                 "; ".join("/".join(c) for c in res["excluded_cells_baseline_attained"]))
    L.append("\nH1 primary (pooled over models, Holm across these tests):")
    for r in res["primary"]:
        extra = f" (diff. in proportion {_f(r['mean_diff'])})" if r["metric"] == "goal_attained" else ""
        L.append(f"  {r['contrast']:<10} {r['metric']:<22} n={r['n_pairs']:<4} median diff={_f(r['median_diff'])}{extra} "
                 f"[{_f(r['ci_low'])}, {_f(r['ci_high'])}]  p={_f(r['p'], 4)}  p_holm={_f(r['p_holm'], 4)}"
                 f"  {'*' if r['significant'] else ''}")
    L.append("\nPer model (Holm across these tests; capability floor, H2):")
    for r in res["per_model"]:
        L.append(f"  {r['model']:<20} {r['contrast']:<10} {r['metric']:<22} n={r['n_pairs']:<4} "
                 f"median={_f(r['median_diff'])}  p_holm={_f(r['p_holm'], 4)}  {'*' if r['significant'] else ''}")
    L.append("\nH2 interaction (Kruskal-Wallis on paired differences across models):")
    for r in res["h2_interaction"]:
        L.append(f"  {r['contrast']:<28} H={_f(r['h'])} df={r['df']} p={_f(r['p'], 4)}")
    L.append("\nH3 (absolute reduction vs C0, rate points: validator rejections minus intent violations):")
    for r in res["h3"]:
        L.append(f"  {r['model']:<20} {r['contrast']:<10} n={r['n_pairs']:<4} AR_val={_f(r.get('ar_validator'))} "
                 f"AR_int={_f(r.get('ar_intent'))} delta={_f(r.get('delta'))} "
                 f"[{_f(r.get('ci_low'))}, {_f(r.get('ci_high'))}]  {r['verdict']}"
                 f"   (relative: val={_f(r.get('rr_validator'))}, int={_f(r.get('rr_intent'))})")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-run", required=True, help="per_run.csv written by experiment_eval.py")
    ap.add_argument("--out", help="output JSON (default: stats.json next to the per-run file)")
    a = ap.parse_args(argv)
    rows = load_rows(a.per_run)
    if not rows:
        print("no ok runs in " + a.per_run, file=sys.stderr)
        return 1
    res = analyse(rows)
    res["per_run"] = a.per_run
    out = Path(a.out) if a.out else Path(a.per_run).with_name("stats.json")
    out.write_text(json.dumps(res, indent=2, default=str) + "\n")
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        cols = ["hypothesis", "contrast", "metric", "goals", "model", "n_pairs", "median_diff", "mean_diff",
                "ci_low", "ci_high", "p", "p_holm", "significant", "test"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(res["primary"] + res["per_model"])
    print(report(res))
    print(f"\n-> {out} and {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
