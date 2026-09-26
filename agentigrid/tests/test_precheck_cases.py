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
    a = pc.scopflow_args(Path("c.cont"), 1)
    assert a[0] == "-ctgcfile" and a[1].endswith("c.cont") and a[2:] == ["-scopflow_Nc", "-1"]
    assert "EMPAR" not in pc.scopflow_args(Path("c.cont"), 4)   # coupled solve only


def test_write_stressed_scales_loads(tmp_path):
    from agentigrid.parsers.matpower_parser import parse_matpower
    case = ROOT.parent / "datafiles" / "case9" / "case9mod.m"
    out = pc.write_stressed(case, 1.2, tmp_path)
    base, scaled = parse_matpower(case), parse_matpower(out)
    pd0 = sum(b.Pd for b in base.buses)
    pd1 = sum(b.Pd for b in scaled.buses)
    assert abs(pd1 - 1.2 * pd0) < 1e-6 * max(1.0, pd0)
    assert out.name == "case9mod_load1.2.m"


def test_failed_scopflow_is_undetermined_not_ok():
    """Regression: a SCOPFLOW run that never executed (e.g. contingency file not
    found) must not be read as 'SCOPFLOW infeasible' -> 'n1 OK'."""
    failed = {"parsed": False, "converged": False, "status": "FAILED", "error": "Cannot open file"}
    r = {1.0: {"opflow": _rec(), "scopflow": failed, "pflow": _rec()}}
    v = pc.classify(r)
    assert v["n1"]["meaningful_at_base"] is False
    assert v["n1"]["verdict"].startswith("UNDETERMINED")
    assert v["n1"]["undetermined_scales"] == [1.0]


def test_scopflow_args_use_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = pc.scopflow_args(Path("../x/c.cont"), 1)
    assert Path(args[1]).is_absolute()


def test_write_stressed_with_voltage_band(tmp_path):
    from agentigrid.parsers.matpower_parser import parse_matpower
    case = ROOT.parent / "datafiles" / "case9" / "case9mod.m"
    out = pc.write_stressed(case, 1.1, tmp_path, (0.95, 1.05))
    assert out.name == "case9mod_load1.1_v0.95-1.05.m"
    assert {(b.Vmin, b.Vmax) for b in parse_matpower(out).buses} == {(0.95, 1.05)}


def test_marginal_scopflow_is_undetermined():
    """Regression: SCOPFLOW that stopped at IPOPT's iteration limit ('marginal')
    is not evidence of N-1 insecurity and must not yield 'n1 OK'."""
    marginal = {"parsed": True, "converged": False, "feasibility": "marginal", "num_violations": 0}
    r = {1.0: {"opflow": _rec(), "scopflow": marginal, "pflow": _rec()}}
    v = pc.classify(r)
    assert v["n1"]["verdict"].startswith("UNDETERMINED")
    proven = {"parsed": True, "converged": False, "feasibility": "infeasible", "num_violations": 0}
    r = {1.0: {"opflow": _rec(), "scopflow": proven, "pflow": _rec()}}
    assert pc.classify(r)["n1"]["meaningful_at_base"] is True
