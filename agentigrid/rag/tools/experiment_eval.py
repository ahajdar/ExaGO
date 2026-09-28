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
#                                silently solve the base case only) AND a coupled
#                                solver (not EMPAR, which solves contingencies
#                                independently);
#   * "success": "no_violations" -> a solve that converged with no voltage, thermal
#                                or power-balance violation (journal ``feasible`` is
#                                True only when violations_count == 0).
# Goals with neither (e.g. loadmax) have no attainment predicate -> None.
# A goal may declare BOTH (e.g. the N-1 cost goal: n1_secure AND target_pct):
# then both must hold on the same entry.
#
# Intent guards. A goal (and a case) may declare deterministic goal-constraint
# checks, stated in the goal text identically in every condition. A post-baseline
# solve that violates one is an INTENT VIOLATION: executable, possibly feasible,
# but not what was asked. It never counts toward attainment (or, for cost goals,
# toward cost improvement). Guards (manifest "guards"; spec: goal + case lists):
#   * "load_preserved"   -- total_load_mw not below baseline by more than LOAD_TOL;
#   * "no_cost_edits"    -- no set_cost_coeffs in effect;
#   * "no_rating_edits"  -- no set_branch_rate in effect;
#   * {"vband_not_widened": [lo, hi]} -- no set_(all_)bus_vlimits outside [lo, hi];
#   * {"branch_stays_out": [fbus, tbus, ckt]} -- that branch not set back in service.
# "In effect" = the entry's own commands plus, for non-"fresh" entries, every
# command since the last "fresh" entry (incremental modes build on the current
# network). Deliberately conservative: counting a command that a revert later
# undid can only produce a false negative, never a false success.
# Legacy manifests without "guards": cost-target goals get load_preserved +
# no_cost_edits, other goals none.
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


def coupled_scopf_solver(e) -> bool:
    """False when SCOPFLOW ran with EMPAR. Per the ExaGO manual, EMPAR "does not
    solve the security-constrained ACOPF problem"; it solves the base case and each
    contingency independently, so its dispatch is not N-1 secure. (AgentiGrid
    selects EMPAR automatically when exago.mpi_np > 1.)"""
    argv = [str(a).upper() for a in _argv(e)]
    if "-SCOPFLOW_SOLVER" in argv:
        i = argv.index("-SCOPFLOW_SOLVER")
        return not (i + 1 < len(argv) and argv[i + 1] == "EMPAR")
    return True


def _single_solve(e) -> bool:
    """A scalar solve entry (not analysis/complete/sweep/explore/failed). Journals
    written before invocation records existed have no ``exago_command``; they still
    count here, but cannot satisfy n1_secure, which needs the recorded argv."""
    if not isinstance(e, dict) or e.get("convergence_status") in TERMINAL:
        return False
    cmd = e.get("exago_command")
    return not (isinstance(cmd, dict) and cmd.get("mode", "single") != "single")


LOAD_TOL = 1e-3                      # 0.1 % relative
GUARD_NAMES = ("load_preserved", "no_cost_edits", "no_rating_edits",
               "vband_not_widened", "branch_stays_out")
DEFAULT_COST_GUARDS = ["load_preserved", "no_cost_edits"]


def _actions(cmds) -> list[str]:
    return [str(c.get("action", "")).lower() for c in (cmds or []) if isinstance(c, dict)]


def effective_commands(entries, idx: int) -> list[dict]:
    """Commands that may be in effect for entries[idx] (see guard notes above)."""
    cmds: list[dict] = []
    for prev in reversed(entries[: idx + 1]):
        if not isinstance(prev, dict):
            continue
        if prev.get("iteration", 0) == 0:
            break
        cmds.extend(c for c in (prev.get("commands") or []) if isinstance(c, dict))
        if prev.get("mode") == "fresh":
            break
    return cmds


def effective_actions(entries, idx: int) -> list[str]:
    return _actions(effective_commands(entries, idx))


def base_load(entries):
    for e in entries:
        if isinstance(e, dict) and e.get("iteration") == 0 and _num(e.get("total_load_mw")) is not None:
            return e["total_load_mw"]
    return None


def normalize_guards(guards) -> list[tuple[str, object]]:
    out = []
    for g in guards or []:
        if isinstance(g, str):
            name, arg = g, None
        elif isinstance(g, dict) and len(g) == 1:
            name, arg = next(iter(g.items()))
        else:
            raise ValueError(f"bad guard spec {g!r}")
        if name not in GUARD_NAMES:
            raise ValueError(f"unknown guard {name!r}; expected one of {GUARD_NAMES}")
        out.append((name, arg))
    return out


def goal_guards(goal: dict) -> list[tuple[str, object]]:
    if goal.get("guards") is not None:
        return normalize_guards(goal["guards"])
    return normalize_guards(DEFAULT_COST_GUARDS if goal.get("target_pct") is not None else [])


def _same_branch(c: dict, fbus, tbus, ckt) -> bool:
    try:
        a, b = int(c.get("fbus")), int(c.get("tbus"))
    except (TypeError, ValueError):
        return False
    return {a, b} == {int(fbus), int(tbus)} and int(c.get("ckt") or 0) == int(ckt or 0)


def guard_violations(e, cmds, base_load_mw, guards) -> list[str]:
    """Names of the guards this entry violates (empty = intent respected)."""
    acts = _actions(cmds)
    bad = []
    for name, arg in guards:
        if name == "load_preserved":
            load = _num(e.get("total_load_mw"))
            if base_load_mw not in (None, 0) and load is not None and load < base_load_mw * (1.0 - LOAD_TOL):
                bad.append(name)
        elif name == "no_cost_edits" and "set_cost_coeffs" in acts:
            bad.append(name)
        elif name == "no_rating_edits" and "set_branch_rate" in acts:
            bad.append(name)
        elif name == "vband_not_widened":
            lo, hi = arg
            for c in cmds:
                if str(c.get("action", "")).lower() in ("set_bus_vlimits", "set_all_bus_vlimits"):
                    vmin, vmax = _num(c.get("Vmin")), _num(c.get("Vmax"))
                    if (vmin is not None and vmin < lo - 1e-6) or (vmax is not None and vmax > hi + 1e-6):
                        bad.append(name)
                        break
        elif name == "branch_stays_out":
            f, t, k = (list(arg) + [0])[:3]
            if any(str(c.get("action", "")).lower() == "set_branch_status" and int(c.get("status", 0)) == 1
                   and _same_branch(c, f, t, k) for c in cmds):
                bad.append(name)
    return bad


def cost_guard_reason(e, actions, base_load_mw) -> str | None:
    """Back-compat wrapper (cost guards from an action-name list)."""
    cmds = [{"action": a} for a in actions]
    bad = guard_violations(e, cmds, base_load_mw, normalize_guards(DEFAULT_COST_GUARDS))
    return {"no_cost_edits": "cost_curve_edit", "load_preserved": "load_reduced"}.get(bad[0]) if bad else None


def predicate_holds(e, goal: dict, base_cost) -> bool:
    """The goal's success predicate(s) alone, ignoring guards."""
    if not _single_solve(e) or not e.get("feasible"):
        return False
    success = goal.get("success")
    if success == "n1_secure":
        if not ("scopflow" in _application(e) and all_contingencies_selected(e)
                and coupled_scopf_solver(e)):
            return False
    elif success == "no_violations":
        if (e.get("violations_count") or 0) != 0:
            return False
    target = goal.get("target_pct")
    if target is None:
        return success is not None
    obj = _num(e.get("objective_value"))
    if obj is None or base_cost in (None, 0):
        return False
    return (base_cost - obj) / base_cost * 100.0 >= float(target)


def entry_attains(e, goal: dict, base_cost, cmds=None, base_load_mw=None) -> bool:
    """Predicate holds AND no guard is violated. *cmds* are the commands in
    effect (dicts); a list of action-name strings is accepted for back-compat."""
    if not predicate_holds(e, goal, base_cost):
        return False
    cmds = [c if isinstance(c, dict) else {"action": c} for c in (cmds or [])]
    return not guard_violations(e, cmds, base_load_mw, goal_guards(goal))


def attainment_for(entries, goal: dict, base_cost) -> dict:
    """goal_attained / baseline_attained / iterations_to_goal / guard_rejections."""
    none = {"goal_attained": None, "baseline_attained": None, "iterations_to_goal": None,
            "guard_rejections": None}
    if goal.get("success") is None and goal.get("target_pct") is None:
        return none
    if goal.get("success") is not None and goal["success"] not in SUCCESS_PREDICATES:
        raise ValueError(f"unknown success predicate {goal['success']!r}; "
                         f"expected one of {SUCCESS_PREDICATES}")
    guards = goal_guards(goal)
    dict_entries = [e for e in entries if isinstance(e, dict)]
    bl = base_load(dict_entries)
    hits, base_hit, rejected = [], False, 0
    for i, e in enumerate(dict_entries):
        holds = predicate_holds(e, goal, base_cost)
        if not holds:
            continue
        ok = not guard_violations(e, effective_commands(dict_entries, i), bl, guards)
        if e.get("iteration", 0) == 0:
            base_hit = base_hit or ok
        elif ok:
            hits.append(e.get("iteration", 0))
        else:
            rejected += 1         # would have counted without the guards
    hits.sort()
    return {
        "goal_attained": int(bool(hits)),
        "baseline_attained": int(base_hit),
        "iterations_to_goal": hits[0] if hits else None,
        "guard_rejections": rejected if guards else None,
    }


def intent_metrics(entries, goal: dict) -> dict:
    """H3 metrics. Executed proposals = post-baseline single solves that applied a
    modification; an intent violation is such a solve violating >= 1 guard."""
    guards = goal_guards(goal)
    dict_entries = [e for e in entries if isinstance(e, dict)]
    bl = base_load(dict_entries)
    executed = viol = 0
    per = {name: 0 for name, _ in guards}
    for i, e in enumerate(dict_entries):
        if e.get("iteration", 0) == 0 or not _single_solve(e) or not has_modification(e):
            continue
        executed += 1
        bad = guard_violations(e, effective_commands(dict_entries, i), bl, guards)
        if bad:
            viol += 1
            for b in set(bad):
                per[b] += 1
    out = {"executed_proposals": executed,
           "intent_violations": viol if guards else None,
           "intent_violation_rate": (round(viol / executed, 3) if executed else None) if guards else None}
    for name in GUARD_NAMES:
        out[f"intent_{name}"] = per.get(name) if name in per else None
    return out


def validator_metrics(journal: dict, entries) -> dict:
    """H3 metrics. An LLM iteration is REJECTED by the deterministic validator /
    parser when it recorded no entry (journal discarded_actions, kind 'rejected')
    or recorded an entry with >= 1 skipped command. Internal errors and failed
    API requests (kind 'internal' / 'api_error') are excluded from both
    numerator and denominator."""
    disc = journal.get("discarded_actions") if isinstance(journal, dict) else None
    disc = [d for d in (disc or []) if isinstance(d, dict)]
    rejected_it = {d.get("iteration") for d in disc if d.get("kind", "rejected") == "rejected"}
    internal_it = {d.get("iteration") for d in disc if d.get("kind") in ("internal", "api_error")}
    api_err_it = {d.get("iteration") for d in disc if d.get("kind") == "api_error"}
    recorded_it = {e.get("iteration") for e in entries if isinstance(e, dict) and e.get("iteration", 0) != 0}
    partial_it = {e.get("iteration") for e in entries
                  if isinstance(e, dict) and e.get("iteration", 0) != 0 and e.get("skipped_commands")}
    llm_it = (recorded_it | rejected_it) - internal_it
    rej = (rejected_it | partial_it) - internal_it
    return {"llm_iterations": len(llm_it),
            "validator_rejections": len(rej),
            "validator_rejection_rate": round(len(rej) / len(llm_it), 3) if llm_it else None,
            "discarded_internal": len(internal_it - api_err_it),
            "discarded_api_error": len(api_err_it),
            "has_discard_telemetry": int(isinstance(journal, dict) and "discarded_actions" in journal)}


def has_modification(e):
    cmds = e.get("commands")
    return bool(cmds) or (e.get("exago_command") is not None)


def metrics_for(journal: dict, manifest: dict) -> dict:
    entries = journal.get("entries", []) if isinstance(journal, dict) else []
    solves = solve_entries(entries)
    non_base = [e for e in solves if e.get("iteration", 0) != 0]

    bc = base_cost(entries)
    goal = {"target_pct": manifest.get("target_pct"), "success": manifest.get("success")}
    if manifest.get("guards") is not None:
        goal["guards"] = manifest["guards"]
    cost_goal = goal["target_pct"] is not None
    guards = goal_guards(goal)
    dict_entries = [e for e in entries if isinstance(e, dict)]
    bl = base_load(dict_entries)
    solve_ids = {id(e) for e in solves}
    feasible_costs = []
    for i, e in enumerate(dict_entries):
        if id(e) not in solve_ids or not e.get("feasible") or _num(e.get("objective_value")) is None:
            continue
        # For cost goals the reported improvement obeys the same guard as attainment.
        if cost_goal and guard_violations(e, effective_commands(dict_entries, i), bl, guards):
            continue
        feasible_costs.append(e["objective_value"])
    best = min(feasible_costs) if feasible_costs else None
    if best is None and not cost_goal:
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
        **intent_metrics(entries, goal),
        **validator_metrics(journal, entries),
        **usage_metrics(journal),
        **rag_metrics(journal),
    }


def rag_metrics(journal: dict) -> dict:
    """What retrieval actually did in the run (journal ``rag_config``).

    ``rag_grader_fallbacks`` > 0 means the requested grader failed at least once
    and cosine scores were used instead; such runs are not clean runs of that
    condition and should be excluded or reported.
    """
    rc = journal.get("rag_config") if isinstance(journal, dict) else None
    if not isinstance(rc, dict):
        return {"rag_mode": None, "rag_grader": None, "rag_grader_fallbacks": None,
                "crag_correct": None, "crag_ambiguous": None, "crag_incorrect": None,
                "crag_withheld": None}
    st = rc.get("stats") or {}
    return {
        "rag_mode": rc.get("mode_env") or rc.get("mode"),
        "rag_grader": rc.get("grader"),
        "rag_grader_fallbacks": st.get("grader_fallbacks"),
        "crag_correct": st.get("correct"), "crag_ambiguous": st.get("ambiguous"),
        "crag_incorrect": st.get("incorrect"), "crag_withheld": st.get("withheld"),
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
           "validator_rejection_rate", "intent_violation_rate",   # H3
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
