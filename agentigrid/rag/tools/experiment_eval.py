#!/usr/bin/env python3
"""Phase-0 experiment EVALUATOR for the AgentiGrid RAG comparison.

Reads the runs produced by experiment_runner.py (each run = manifest.json +
journal.json) and aggregates them into two CSVs:

  * per_run.csv   — one row per run, with the objective (oracle) metrics.
  * summary.csv   — one row per (case, goal, condition, model) cell, with
                    mean and 95% CI over repetitions.

Dependency-free (stdlib only). Metric definitions are documented inline and are
deliberately conservative; refine them as the study matures.

Usage
-----
    python rag/tools/experiment_eval.py --runs experiments/run1
    python rag/tools/experiment_eval.py --runs experiments/run1 --out experiments/run1/analysis

You can also point it at a bare folder of journals (no manifests) with
    --journals-glob 'workdir/journal_*.json'
in which case condition/model are taken as "unknown".
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from pathlib import Path

# Mirror journal.py: statuses that never denote a real scalar solve.
NON_SOLVE = frozenset({"ANALYSIS", "COMPLETE", "EXPLORE", "SWEEP", "CONTINGENCY"})
TERMINAL = NON_SOLVE | {"FAILED"}

# Two-sided 95% Student-t critical values t_{0.975, df} (exact for df 1-30).
# Between table points the NEXT-LOWER df is used, which is conservative (wider CI).
_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
      8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
      15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
      22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
      29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T:
        return _T[df]
    lower = [k for k in _T if k <= df]
    return _T[max(lower)]


def _num(x):
    return x if isinstance(x, (int, float)) else None


def solve_entries(entries):
    """Real scalar-solve iterations (exclude base i==0 and non-solve markers)."""
    out = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("convergence_status") in TERMINAL:
            continue
        out.append(e)
    return out


def base_cost(entries):
    for e in entries:
        if isinstance(e, dict) and e.get("iteration") == 0 and _num(e.get("objective_value")) is not None:
            return e["objective_value"]
    # fallback: first feasible solve
    for e in solve_entries(entries):
        if e.get("feasible") and _num(e.get("objective_value")) is not None:
            return e["objective_value"]
    return None


# ---------------------------------------------------------------------------
# Goal attainment
# ---------------------------------------------------------------------------
# A goal declares how success is judged:
#   * "target_pct": N         -> cost reduction vs. base of at least N percent;
#   * "success": "n1_secure"  -> a SCOPFLOW solve that converged feasibly with a
#                                contingency file AND all contingencies selected
#                                (-scopflow_Nc -1; the ExaGO default of 0 would
#                                silently solve the base case only);
#   * "success": "no_violations" -> a solve that converged with no voltage, thermal
#                                or power-balance violation (journal ``feasible`` is
#                                True only when violations_count == 0).
# Goals with neither (e.g. loadmax) have no attainment predicate -> None.
#
# Attainment is judged on post-baseline solve entries only: the agent must reach
# the goal. ``baseline_attained`` reports whether iteration 0 already satisfied
# it; such a cell measures nothing and must be fixed (stressed variant) rather
# than analysed. Sweep/explore variants are not counted (single-solve entries only).

SUCCESS_PREDICATES = ("n1_secure", "no_violations")


def _argv(e):
    cmd = e.get("exago_command") or {}
    argv = cmd.get("argv") if isinstance(cmd, dict) else None
    return argv if isinstance(argv, list) else []


def _application(e):
    cmd = e.get("exago_command") or {}
    app = cmd.get("application") if isinstance(cmd, dict) else None
    if app:
        return str(app).lower()
    argv = _argv(e)
    return Path(argv[0]).name.lower() if argv else ""


def all_contingencies_selected(e) -> bool:
    argv = _argv(e)
    if "-ctgcfile" not in argv or "-scopflow_Nc" not in argv:
        return False
    i = argv.index("-scopflow_Nc")
    return i + 1 < len(argv) and str(argv[i + 1]).strip() == "-1"


def _single_solve(e) -> bool:
    """A scalar solve entry (not analysis/complete/sweep/explore/failed). Journals
    written before invocation records existed have no ``exago_command``; they still
    count here, but cannot satisfy n1_secure, which needs the recorded argv."""
    if not isinstance(e, dict) or e.get("convergence_status") in TERMINAL:
        return False
    cmd = e.get("exago_command")
    return not (isinstance(cmd, dict) and cmd.get("mode", "single") != "single")


def entry_attains(e, goal: dict, base_cost) -> bool:
    """Does this single journal entry satisfy the goal's success predicate?"""
    if not _single_solve(e) or not e.get("feasible"):
        return False
    success = goal.get("success")
    if success == "n1_secure":
        return "scopflow" in _application(e) and all_contingencies_selected(e)
    if success == "no_violations":
        return (e.get("violations_count") or 0) == 0
    target = goal.get("target_pct")
    if target is not None:
        obj = _num(e.get("objective_value"))
        if obj is None or base_cost in (None, 0):
            return False
        return (base_cost - obj) / base_cost * 100.0 >= float(target)
    return False


def attainment_for(entries, goal: dict, base_cost) -> dict:
    """goal_attained / baseline_attained / iterations_to_goal for one run."""
    if goal.get("success") is None and goal.get("target_pct") is None:
        return {"goal_attained": None, "baseline_attained": None, "iterations_to_goal": None}
    if goal.get("success") is not None and goal["success"] not in SUCCESS_PREDICATES:
        raise ValueError(f"unknown success predicate {goal['success']!r}; "
                         f"expected one of {SUCCESS_PREDICATES}")
    dict_entries = [e for e in entries if isinstance(e, dict)]
    base = [e for e in dict_entries if e.get("iteration", 0) == 0]
    post = [e for e in dict_entries if e.get("iteration", 0) != 0]
    hits = sorted(e.get("iteration", 0) for e in post if entry_attains(e, goal, base_cost))
    return {
        "goal_attained": int(bool(hits)),
        "baseline_attained": int(any(entry_attains(e, goal, base_cost) for e in base)),
        "iterations_to_goal": hits[0] if hits else None,
    }


def has_modification(e):
    cmds = e.get("commands")
    return bool(cmds) or (e.get("exago_command") is not None)


def metrics_for(journal: dict, manifest: dict) -> dict:
    entries = journal.get("entries", []) if isinstance(journal, dict) else []
    solves = solve_entries(entries)
    non_base = [e for e in solves if e.get("iteration", 0) != 0]

    bc = base_cost(entries)
    feasible_costs = [e["objective_value"] for e in solves
                      if e.get("feasible") and _num(e.get("objective_value")) is not None]
    best = min(feasible_costs) if feasible_costs else None
    if best is None:
        sb = journal.get("session_best") if isinstance(journal, dict) else None
        if isinstance(sb, dict) and _num(sb.get("cost")) is not None:
            best = sb["cost"]

    improvement = None
    if bc not in (None, 0) and best is not None:
        improvement = round((bc - best) / bc * 100.0, 3)

    attempts = manifest.get("max_iter") or (max((e.get("iteration", 0) for e in entries), default=0))
    n_mods = sum(1 for e in non_base if has_modification(e))
    valid_rate = round(n_mods / attempts, 3) if attempts else None

    any_feasible = any(e.get("feasible") for e in solves)
    goal = {"target_pct": manifest.get("target_pct"), "success": manifest.get("success")}
    attain = attainment_for(entries, goal, bc)

    elapsed = sum(_num(e.get("elapsed_seconds")) or 0.0 for e in entries)

    return {
        "base_cost": bc,
        "best_cost": best,
        "cost_improvement_pct": improvement,
        "n_solve_iters": len(non_base),
        "n_modifications": n_mods,
        "attempts": attempts,
        "valid_proposal_rate": valid_rate,
        "any_feasible": int(bool(any_feasible)),
        **attain,
        "iterations_recorded": len(entries),
        "solve_elapsed_s": round(elapsed, 1),
        "rag_enabled": (journal.get("rag_enabled")
                        if isinstance(journal, dict) and "rag_enabled" in journal
                        else manifest.get("condition_env", {}).get("AGENTIGRID_RAG")),
        **usage_metrics(journal),
    }


USAGE_KEYS = ("calls", "prompt_tokens", "completion_tokens",
              "cache_creation_tokens", "cache_read_tokens")


def usage_metrics(journal: dict) -> dict:
    """Token usage over ALL LLM calls of the run (journal ``llm_usage``).

    ``llm_prompt_tokens`` is the total input (uncached + cache writes + cache
    reads). Cost in USD is deliberately NOT computed here: prices change, so it
    is derived offline from these counts and the rates in effect at run time.
    Journals written before usage metering carry no ``llm_usage`` -> None.
    """
    u = journal.get("llm_usage") if isinstance(journal, dict) else None
    if not isinstance(u, dict):
        return {f"llm_{k}": None for k in USAGE_KEYS}
    return {f"llm_{k}": u.get(k) for k in USAGE_KEYS}


def collect_runs(runs_dir: Path):
    rows = []
    for mpath in sorted(runs_dir.glob("*/manifest.json")):
        try:
            manifest = json.loads(mpath.read_text())
        except Exception:
            continue
        jpath = mpath.parent / "journal.json"
        journal = {}
        if jpath.exists():
            try:
                journal = json.loads(jpath.read_text())
            except Exception:
                journal = {}
        row = {
            "run_id": manifest.get("run_id"),
            "case": manifest.get("case"),
            "network": manifest.get("network", manifest.get("case")),
            "app": manifest.get("app"),
            "goal": manifest.get("goal_id"),
            "condition": manifest.get("condition"),
            "backend": manifest.get("backend"),
            "model": manifest.get("model"),
            "rep": manifest.get("rep"),
            "status": manifest.get("status"),
            "wall_s": manifest.get("wall_s"),
            "exit_code": manifest.get("exit_code"),
        }
        row.update(metrics_for(journal, manifest))
        rows.append(row)
    return rows


def collect_journals(pattern: str):
    rows = []
    for jp in sorted(glob.glob(pattern)):
        try:
            journal = json.loads(Path(jp).read_text())
        except Exception:
            continue
        row = {"run_id": Path(jp).stem, "case": "unknown", "goal": "unknown",
               "condition": "unknown", "backend": "unknown", "model": "unknown",
               "rep": None, "status": "ok", "wall_s": None, "exit_code": None}
        row.update(metrics_for(journal, {}))
        rows.append(row)
    return rows


NUMERIC = ["cost_improvement_pct", "valid_proposal_rate", "solve_elapsed_s",
           "llm_prompt_tokens", "llm_completion_tokens",
           "llm_cache_creation_tokens", "llm_cache_read_tokens",
           "n_solve_iters", "wall_s",
           "iterations_to_goal"]  # mean over runs that attained the goal only
RATE = ["any_feasible", "goal_attained", "baseline_attained"]  # averaged as proportions


def summarize(rows):
    cells = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["case"], r["goal"], r["condition"], r["model"])
        cells.setdefault(key, []).append(r)

    out = []
    for (case, goal, cond, model), rs in sorted(cells.items()):
        rec = {"case": case, "goal": goal, "condition": cond, "model": model, "n_runs": len(rs)}
        for m in NUMERIC:
            vals = [r[m] for r in rs if isinstance(r.get(m), (int, float))]
            if vals:
                mean = statistics.mean(vals)
                rec[m + "_mean"] = round(mean, 3)
                if len(vals) >= 2:
                    sd = statistics.stdev(vals)
                    ci = t975(len(vals) - 1) * sd / math.sqrt(len(vals))
                    rec[m + "_ci95"] = round(ci, 3)
                else:
                    rec[m + "_ci95"] = None
            else:
                rec[m + "_mean"] = None
                rec[m + "_ci95"] = None
        for m in RATE:
            vals = [r[m] for r in rs if isinstance(r.get(m), (int, float))]
            rec[m + "_rate"] = round(statistics.mean(vals), 3) if vals else None
        out.append(rec)
    return out


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("")
        return
    cols = list({k: None for row in rows for k in row}.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="runner out_dir (containing <run_id>/manifest.json)")
    ap.add_argument("--journals-glob", help="alternatively, a glob of bare journal_*.json")
    ap.add_argument("--out", help="output dir (default: <runs>/analysis)")
    args = ap.parse_args()

    if args.runs:
        runs_dir = Path(args.runs)
        rows = collect_runs(runs_dir)
        out_dir = Path(args.out) if args.out else runs_dir / "analysis"
    elif args.journals_glob:
        rows = collect_journals(args.journals_glob)
        out_dir = Path(args.out) if args.out else Path("analysis")
    else:
        ap.error("provide --runs DIR or --journals-glob PATTERN")

    if not rows:
        print("No runs/journals found.", file=__import__("sys").stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_run.csv", rows)
    summ = summarize(rows)
    write_csv(out_dir / "summary.csv", summ)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"Parsed {len(rows)} runs ({ok} ok). Wrote:")
    print(f"  {out_dir/'per_run.csv'}")
    print(f"  {out_dir/'summary.csv'}")
    print("\nSummary (valid-proposal rate / cost-improvement % / goal attainment, mean over reps):")
    for r in summ:
        vp = r.get("valid_proposal_rate_mean")
        ci = r.get("cost_improvement_pct_mean")
        print(f"  {r['case']:>10} | {r['goal']:<10} | {r['condition']:<10} | {r['model']:<14} "
              f"| n={r['n_runs']} | valid={vp} | costΔ%={ci} | feas={r.get('any_feasible_rate')} "
              f"| attained={r.get('goal_attained_rate')} | baseline_attained={r.get('baseline_attained_rate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
