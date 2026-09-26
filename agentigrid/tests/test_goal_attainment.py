"""Evaluator goal-attainment predicates (cost target, N-1 security, no violations),
baseline attainment, iterations-to-goal, and the t critical-value table."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("experiment_eval", "rag/tools/experiment_eval.py")


def _solve(it, *, feasible=True, obj=100.0, viol=0, status="CONVERGED", argv=None, app=None, mode="single"):
    cmd = None
    if argv is not None:
        cmd = {"mode": mode, "application": app, "argv": argv}
    return {"iteration": it, "feasible": feasible, "objective_value": obj,
            "violations_count": viol, "convergence_status": status, "exago_command": cmd,
            "commands": [{"action": "x"}] if it else []}


SCOPF_ALL = ["./applications/scopflow", "-netfile", "n.m", "-ctgcfile", "n.cont", "-scopflow_Nc", "-1"]
SCOPF_DEFAULT = ["./applications/scopflow", "-netfile", "n.m", "-ctgcfile", "n.cont"]
SCOPF_FIRST4 = ["./applications/scopflow", "-netfile", "n.m", "-ctgcfile", "n.cont", "-scopflow_Nc", "4"]


# --- cost target ----------------------------------------------------------

def test_cost_target_attained_and_iteration():
    entries = [_solve(0, obj=100.0), _solve(1, obj=95.0), _solve(2, obj=89.0), _solve(3, obj=85.0)]
    a = ev.attainment_for(entries, {"target_pct": 10}, 100.0)
    assert a == {"goal_attained": 1, "baseline_attained": 0, "iterations_to_goal": 2}


def test_cost_target_ignores_infeasible_and_non_solve_entries():
    entries = [_solve(0, obj=100.0), _solve(1, obj=50.0, feasible=False),
               _solve(2, obj=50.0, status="SWEEP"), _solve(3, obj=97.0)]
    assert ev.attainment_for(entries, {"target_pct": 10}, 100.0)["goal_attained"] == 0


# --- N-1 security -----------------------------------------------------------

def test_n1_requires_all_contingencies_selected():
    g = {"success": "n1_secure"}
    ok = [_solve(0, feasible=False, argv=SCOPF_ALL, app="scopflow"),
          _solve(1, argv=SCOPF_ALL, app="scopflow")]
    assert ev.attainment_for(ok, g, 100.0)["goal_attained"] == 1
    # ExaGO default (-scopflow_Nc 0) silently solves the base case only -> not N-1
    default = [_solve(0, feasible=False, argv=SCOPF_ALL, app="scopflow"),
               _solve(1, argv=SCOPF_DEFAULT, app="scopflow")]
    assert ev.attainment_for(default, g, 100.0)["goal_attained"] == 0
    subset = [_solve(0, feasible=False, argv=SCOPF_ALL, app="scopflow"),
              _solve(1, argv=SCOPF_FIRST4, app="scopflow")]
    assert ev.attainment_for(subset, g, 100.0)["goal_attained"] == 0


def test_n1_needs_scopflow_and_recorded_argv():
    g = {"success": "n1_secure"}
    opf = [_solve(1, argv=["./applications/opflow", "-netfile", "n.m"], app="opflow")]
    assert ev.attainment_for(opf, g, 100.0)["goal_attained"] == 0
    no_record = [_solve(1)]  # old journal without exago_command
    assert ev.attainment_for(no_record, g, 100.0)["goal_attained"] == 0
    # application inferred from argv[0] when not recorded
    inferred = [_solve(1, argv=SCOPF_ALL, app=None)]
    assert ev.attainment_for(inferred, g, 100.0)["goal_attained"] == 1


def test_n1_baseline_already_secure_is_flagged():
    g = {"success": "n1_secure"}
    entries = [_solve(0, argv=SCOPF_ALL, app="scopflow"), _solve(1, argv=SCOPF_ALL, app="scopflow")]
    a = ev.attainment_for(entries, g, 100.0)
    assert a["baseline_attained"] == 1 and a["goal_attained"] == 1


# --- no violations (PFLOW relieve / voltage) --------------------------------

def test_no_violations():
    g = {"success": "no_violations"}
    entries = [_solve(0, feasible=False, viol=3), _solve(1, feasible=False, viol=1),
               _solve(2, feasible=True, viol=0), _solve(3, feasible=False, viol=0, status="DID NOT CONVERGE")]
    assert ev.attainment_for(entries, g, None) == {
        "goal_attained": 1, "baseline_attained": 0, "iterations_to_goal": 2}


def test_multi_call_records_do_not_count():
    g = {"success": "no_violations"}
    entries = [_solve(0, feasible=False, viol=2), _solve(1, argv=["pflow"], app="pflow", mode="explore")]
    assert ev.attainment_for(entries, g, None)["goal_attained"] == 0


# --- no predicate / bad predicate --------------------------------------------

def test_goal_without_predicate_is_none():
    assert ev.attainment_for([_solve(1)], {}, 100.0) == {
        "goal_attained": None, "baseline_attained": None, "iterations_to_goal": None}


def test_unknown_predicate_raises():
    with pytest.raises(ValueError):
        ev.attainment_for([_solve(1)], {"success": "n2_secure"}, 100.0)


# --- integration: metrics_for picks the predicate from the manifest -----------

def test_metrics_for_uses_manifest_success():
    journal = {"entries": [_solve(0, feasible=False, viol=2), _solve(1, feasible=True, viol=0)]}
    m = ev.metrics_for(journal, {"max_iter": 4, "success": "no_violations"})
    assert m["goal_attained"] == 1 and m["iterations_to_goal"] == 1 and m["baseline_attained"] == 0


def test_repo_spec_goals_have_valid_predicates():
    spec = json.loads((ROOT / "grader_ablation_spec.json").read_text())
    by_id = {g["id"]: g for g in spec["goals"]}
    assert by_id["cost10"]["target_pct"] == 10
    assert by_id["n1"]["success"] == "n1_secure"
    assert by_id["relieve"]["success"] == by_id["voltage"]["success"] == "no_violations"
    assert "success" not in by_id["loadmax"] and "target_pct" not in by_id["loadmax"]
    for g in spec["goals"]:
        assert g.get("success") in (None, *ev.SUCCESS_PREDICATES)


# --- t table ------------------------------------------------------------------

@pytest.mark.parametrize("df, t", [(1, 12.706), (4, 2.776), (11, 2.201), (19, 2.093),
                                   (29, 2.045), (35, 2.042), (50, 2.021), (1000, 1.980)])
def test_t975_exact_or_conservative(df, t):
    assert ev.t975(df) == t


def test_n1_rejects_empar_uncoupled_solve():
    g = {"success": "n1_secure"}
    empar = SCOPF_ALL + ["-scopflow_solver", "EMPAR"]
    entries = [_solve(0, feasible=False, argv=SCOPF_ALL, app="scopflow"),
               _solve(1, argv=empar, app="scopflow")]
    assert ev.attainment_for(entries, g, 100.0)["goal_attained"] == 0
    ipopt = SCOPF_ALL + ["-scopflow_solver", "IPOPT"]
    entries[1] = _solve(1, argv=ipopt, app="scopflow")
    assert ev.attainment_for(entries, g, 100.0)["goal_attained"] == 1
