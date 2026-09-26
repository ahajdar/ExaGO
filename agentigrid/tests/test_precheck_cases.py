"""Pre-check verdict logic (no ExaGO needed) and the stressed-variant writer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("precheck_cases", ROOT / "rag" / "tools" / "precheck_cases.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def _rec(feasible=True, converged=True, volt=0, therm=0):
    return {"parsed": True, "converged": converged,
            "feasibility": "feasible" if feasible else "infeasible",
            "num_violations": volt + therm, "voltage_violations": volt, "thermal_violations": therm}


def test_split_violations():
    d = ["Bus 3: Vm=0.930 pu < 0.95 (undervoltage)", "Bus 9: Vm=1.07 pu > 1.05 (overvoltage)",
         "Branch 1-2: Sf=120.00 > Slim=100.00 MVA", "Power balance violation: generation ..."]
    assert pc.split_violations(d) == {"voltage_violations": 2, "thermal_violations": 1, "balance_violations": 1}


def test_all_goals_ok_at_base():
    r = {1.0: {"opflow": _rec(), "scopflow": _rec(feasible=False),
               "pflow": _rec(feasible=False, volt=2, therm=1)}}
    v = pc.classify(r)
    assert all(v[g]["meaningful_at_base"] for g in ("n1", "relieve", "voltage"))


def test_n1_degenerate_when_scopflow_already_feasible():
    r = {1.0: {"opflow": _rec(), "scopflow": _rec(), "pflow": _rec()},
         1.2: {"opflow": _rec(), "scopflow": _rec(feasible=False), "pflow": _rec(feasible=False, therm=3)},
         1.4: {"opflow": _rec(feasible=False), "scopflow": _rec(feasible=False), "pflow": _rec(converged=False)}}
    v = pc.classify(r)
    assert v["n1"]["meaningful_at_base"] is False
    assert v["n1"]["first_meaningful_scale"] == 1.2
    assert "already feasible" in v["n1"]["note"]
    # at 1.4 OPFLOW itself is infeasible -> not an N-1-specific problem
    assert 1.4 not in v["n1"]["meaningful_scales"]
    assert v["relieve"]["first_meaningful_scale"] == 1.2
    # non-converged PFLOW never counts
    assert 1.4 not in v["relieve"]["meaningful_scales"]
    assert v["voltage"]["first_meaningful_scale"] is None
    assert "redesign" in v["voltage"]["verdict"]


def test_scopflow_args_match_agentigrid():
    assert pc.scopflow_args(Path("c.cont"), 1) == ["-ctgcfile", "c.cont", "-scopflow_Nc", "-1"]
    assert pc.scopflow_args(Path("c.cont"), 4)[-2:] == ["-scopflow_solver", "EMPAR"]


def test_write_stressed_scales_loads(tmp_path):
    from agentigrid.parsers.matpower_parser import parse_matpower
    case = ROOT.parent / "datafiles" / "case9" / "case9mod.m"
    out = pc.write_stressed(case, 1.2, tmp_path)
    base, scaled = parse_matpower(case), parse_matpower(out)
    pd0 = sum(b.Pd for b in base.buses)
    pd1 = sum(b.Pd for b in scaled.buses)
    assert abs(pd1 - 1.2 * pd0) < 1e-6 * max(1.0, pd0)
    assert out.name == "case9mod_load1.2.m"
