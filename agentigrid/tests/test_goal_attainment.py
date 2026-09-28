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
    assert a == {"goal_attained": 1, "baseline_attained": 0, "iterations_to_goal": 2,
                 "guard_rejections": 0}


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
        "goal_attained": 1, "baseline_attained": 0, "iterations_to_goal": 2,
        "guard_rejections": None}


def test_multi_call_records_do_not_count():
    g = {"success": "no_violations"}
    entries = [_solve(0, feasible=False, viol=2), _solve(1, argv=["pflow"], app="pflow", mode="explore")]
    assert ev.attainment_for(entries, g, None)["goal_attained"] == 0


# --- no predicate / bad predicate --------------------------------------------

def test_goal_without_predicate_is_none():
    assert ev.attainment_for([_solve(1)], {}, 100.0) == {
        "goal_attained": None, "baseline_attained": None, "iterations_to_goal": None,
        "guard_rejections": None}


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
    assert by_id["n1cost10"]["success"] == "n1_secure" and by_id["n1cost10"]["target_pct"] == 10
    assert "n1" not in by_id   # degenerate on ACTIVSg200 (pre-check 2026-09-26)
    assert by_id["relieve"]["success"] == by_id["voltage"]["success"] == "no_violations"
    assert "success" not in by_id["loadmax"] and "target_pct" not in by_id["loadmax"]
    for g in spec["goals"]:
        assert g.get("success") in (None, *ev.SUCCESS_PREDICATES)
        ev.goal_guards(g)                      # guard specs parse
    # every goal with a predicate declares intent guards (H3), and states them in its text
    for gid in ("cost10", "n1cost10", "relieve", "voltage"):
        assert by_id[gid]["guards"], gid
    assert "shedding" in by_id["relieve"]["text"] and "ratings" in by_id["relieve"]["text"]
    assert "voltage limits" in by_id["voltage"]["text"]


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


# --- cost guard (load reduction / cost-curve edits) ------------------------

def _cost(it, obj, *, load=1000.0, cmds=None, mode="modify"):
    e = _solve(it, obj=obj)
    e["total_load_mw"] = load
    e["mode"] = mode
    e["commands"] = cmds if cmds is not None else ([{"action": "set_gen_status"}] if it else [])
    return e


COST10 = {"target_pct": 10}


def test_load_reduction_does_not_count_as_cost_reduction():
    entries = [_cost(0, 100.0), _cost(1, 85.0, load=900.0, cmds=[{"action": "scale_all_loads", "factor": 0.9}])]
    a = ev.attainment_for(entries, COST10, 100.0)
    assert a["goal_attained"] == 0 and a["guard_rejections"] == 1


def test_load_within_tolerance_still_counts():
    entries = [_cost(0, 100.0), _cost(1, 85.0, load=999.5)]
    assert ev.attainment_for(entries, COST10, 100.0)["goal_attained"] == 1


def test_cost_curve_edit_is_rejected_and_persists_in_incremental_mode():
    entries = [_cost(0, 100.0),
               _cost(1, 95.0, cmds=[{"action": "set_cost_coeffs", "bus": 1, "coeffs": [0, 1, 0]}]),
               _cost(2, 85.0, cmds=[{"action": "set_gen_status"}], mode="modify")]
    a = ev.attainment_for(entries, COST10, 100.0)
    assert a["goal_attained"] == 0 and a["guard_rejections"] == 1


def test_fresh_entry_drops_earlier_cost_edit():
    entries = [_cost(0, 100.0),
               _cost(1, 80.0, cmds=[{"action": "set_cost_coeffs", "bus": 1, "coeffs": [0, 1, 0]}]),
               _cost(2, 88.0, cmds=[{"action": "set_branch_rate"}], mode="fresh")]
    a = ev.attainment_for(entries, COST10, 100.0)
    assert a["goal_attained"] == 1 and a["iterations_to_goal"] == 2


def test_guard_applies_to_reported_improvement():
    journal = {"entries": [_cost(0, 100.0),
                           _cost(1, 50.0, load=500.0, cmds=[{"action": "scale_all_loads", "factor": 0.5}]),
                           _cost(2, 95.0)]}
    m = ev.metrics_for(journal, {"target_pct": 10, "max_iter": 2})
    assert m["best_cost"] == 95.0 and m["cost_improvement_pct"] == 5.0
    # goals without a cost target keep the raw best cost
    m = ev.metrics_for(journal, {"max_iter": 2})
    assert m["best_cost"] == 50.0


def test_n1_cost_goal_needs_security_and_target():
    goal = {"success": "n1_secure", "target_pct": 10}
    secure_cheap = _solve(1, obj=85.0, argv=SCOPF_ALL, app="scopflow")
    secure_dear = _solve(1, obj=95.0, argv=SCOPF_ALL, app="scopflow")
    insecure_cheap = _solve(1, obj=85.0, argv=SCOPF_FIRST4, app="scopflow")
    base = _solve(0, obj=100.0, argv=SCOPF_ALL, app="scopflow")
    assert ev.attainment_for([base, secure_cheap], goal, 100.0)["goal_attained"] == 1
    assert ev.attainment_for([base, secure_dear], goal, 100.0)["goal_attained"] == 0
    assert ev.attainment_for([base, insecure_cheap], goal, 100.0)["goal_attained"] == 0
    assert ev.attainment_for([base, secure_cheap], goal, 100.0)["baseline_attained"] == 0


# --- intent guards for localized goals, H3 metrics ---------------------------

RELIEVE = {"success": "no_violations",
           "guards": ["load_preserved", "no_rating_edits", {"branch_stays_out": [12, 34, 0]}]}
VOLTAGE = {"success": "no_violations", "guards": ["load_preserved", {"vband_not_widened": [0.95, 1.05]}]}


def _pf(it, *, viol=0, cmds=None, load=1000.0, mode="modify"):
    e = _solve(it, feasible=viol == 0, viol=viol)
    e.update(total_load_mw=load, mode=mode, commands=cmds if cmds is not None else ([{"action": "set_gen_dispatch"}] if it else []))
    return e


def test_relieve_rejects_rating_edit_and_reclosing_the_outage():
    base = _pf(0, viol=3)
    assert ev.attainment_for([base, _pf(1)], RELIEVE, None)["goal_attained"] == 1
    a = ev.attainment_for([base, _pf(1, cmds=[{"action": "set_branch_rate", "fbus": 1, "tbus": 2, "rateA": 999}])], RELIEVE, None)
    assert a["goal_attained"] == 0 and a["guard_rejections"] == 1
    reclose = [{"action": "set_branch_status", "fbus": 34, "tbus": 12, "status": 1}]
    assert ev.attainment_for([base, _pf(1, cmds=reclose)], RELIEVE, None)["goal_attained"] == 0
    other = [{"action": "set_branch_status", "fbus": 34, "tbus": 12, "ckt": 1, "status": 1}]
    assert ev.attainment_for([base, _pf(1, cmds=other)], RELIEVE, None)["goal_attained"] == 1   # a different circuit
    assert ev.attainment_for([base, _pf(1, load=900.0)], RELIEVE, None)["goal_attained"] == 0   # load shed


def test_voltage_band_widening_rejected_but_reasserting_band_allowed():
    base = _pf(0, viol=6)
    widen = [{"action": "set_all_bus_vlimits", "Vmin": 0.9, "Vmax": 1.1}]
    same = [{"action": "set_all_bus_vlimits", "Vmin": 0.95, "Vmax": 1.05}, {"action": "set_gen_voltage", "bus": 1, "Vg": 1.0}]
    assert ev.attainment_for([base, _pf(1, cmds=widen)], VOLTAGE, None)["goal_attained"] == 0
    assert ev.attainment_for([base, _pf(1, cmds=same)], VOLTAGE, None)["goal_attained"] == 1
    tight = [{"action": "set_bus_vlimits", "bus": 3, "Vmin": 0.96, "Vmax": 1.04}]
    assert ev.attainment_for([base, _pf(1, cmds=tight)], VOLTAGE, None)["goal_attained"] == 1


def test_intent_metrics_count_executed_proposals_and_per_guard():
    entries = [_pf(0, viol=3),
               _pf(1, viol=2, cmds=[{"action": "set_branch_rate", "fbus": 1, "tbus": 2, "rateA": 999}]),
               _pf(2, viol=0, cmds=[{"action": "set_gen_dispatch"}], mode="fresh"),
               _pf(3, viol=0, load=800.0)]
    m = ev.intent_metrics(entries, RELIEVE)
    assert m["executed_proposals"] == 3 and m["intent_violations"] == 2
    assert m["intent_violation_rate"] == round(2 / 3, 3)
    assert m["intent_no_rating_edits"] == 1 and m["intent_load_preserved"] == 1
    assert m["intent_vband_not_widened"] is None          # not a guard of this goal
    assert ev.intent_metrics(entries, {"success": "no_violations", "guards": []})["intent_violation_rate"] is None


def test_validator_metrics_from_discards_and_skipped_commands():
    entries = [_pf(0), _pf(1), _pf(2), _pf(4)]
    entries[2]["skipped_commands"] = ["Invalid command {...}"]
    journal = {"entries": entries, "discarded_actions": [
        {"iteration": 3, "kind": "rejected"}, {"iteration": 5, "kind": "internal"}]}
    m = ev.validator_metrics(journal, entries)
    assert m["llm_iterations"] == 4                       # 1,2,3,4 (5 internal excluded)
    assert m["validator_rejections"] == 2                 # 2 partial, 3 discarded
    assert m["validator_rejection_rate"] == 0.5 and m["discarded_internal"] == 1
    assert m["has_discard_telemetry"] == 1
    assert ev.validator_metrics({"entries": entries}, entries)["has_discard_telemetry"] == 0


def test_unknown_guard_rejected():
    with pytest.raises(ValueError):
        ev.goal_guards({"success": "no_violations", "guards": ["no_such_guard"]})


def test_metrics_for_uses_manifest_guards():
    journal = {"entries": [_pf(0, viol=3), _pf(1, cmds=[{"action": "set_branch_rate", "fbus": 1, "tbus": 2, "rateA": 9}])]}
    m = ev.metrics_for(journal, {"success": "no_violations", "max_iter": 4, "guards": RELIEVE["guards"]})
    assert m["goal_attained"] == 0 and m["guard_rejections"] == 1 and m["intent_violations"] == 1
    m = ev.metrics_for(journal, {"success": "no_violations", "max_iter": 4})   # legacy: no guards
    assert m["goal_attained"] == 1


def test_api_errors_excluded_from_validator_rate():
    entries = [_pf(0), _pf(1)]
    journal = {"entries": entries, "discarded_actions": [{"iteration": 2, "kind": "api_error"},
                                                         {"iteration": 3, "kind": "rejected"}]}
    m = ev.validator_metrics(journal, entries)
    assert m["llm_iterations"] == 2 and m["validator_rejections"] == 1
    assert m["discarded_api_error"] == 1 and m["discarded_internal"] == 0
