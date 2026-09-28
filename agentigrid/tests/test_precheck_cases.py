"""Pre-check verdict logic (no ExaGO needed) and the stressed-variant writer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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


# --- physical operating point: slack limits and AGC redispatch -------------

G200 = ROOT.parent / "datafiles" / "case_ACTIVSg200.m"
needs_200 = pytest.mark.skipif(not G200.exists(), reason="ExaGO datafiles not present")


@needs_200
def test_slack_check_flags_unphysical_scales():
    from agentigrid.parsers.matpower_parser import parse_matpower
    base = parse_matpower(G200)
    assert pc.slack_check(pc.stressed_network(base, 0.9), base)["within_limits"] is True
    for f in (0.7, 1.3):   # slack would go below Pmin / above Pmax
        assert pc.slack_check(pc.stressed_network(base, f), base)["within_limits"] is False


@needs_200
@pytest.mark.parametrize("factor", [0.7, 1.3, 1.5])
def test_agc_redispatch_balances_within_limits(factor):
    from agentigrid.parsers.matpower_parser import parse_matpower
    base = parse_matpower(G200)
    net = pc.stressed_network(base, factor, agc=True)
    d_load = sum(b.Pd for b in net.buses) - sum(b.Pd for b in base.buses)
    d_gen = sum(g.Pg for g in net.generators) - sum(g.Pg for g in base.generators)
    assert abs(d_load - d_gen) < 1e-6 * abs(d_load)
    fuels = pc._gen_fuels(net)
    ref = {b.bus_i for b in net.buses if b.type == 3}
    for g0, g, f in zip(base.generators, net.generators, fuels):
        if g.status != 1:
            assert g.Pg == g0.Pg                       # offline units untouched
        elif g.bus in ref:
            assert g.Pg == g0.Pg                       # slack only covers losses
        elif f in pc.RENEWABLE_FUELS:
            assert -1e-9 <= g.Pg <= g0.Pg + 1e-9       # curtail only, never above forecast
        else:
            assert g.Pmin - 1e-9 <= g.Pg <= g.Pmax + 1e-9
    assert pc.stressed_network(base, 1.0, agc=True).generators[0].Pg == base.generators[0].Pg


def test_agc_raises_when_fleet_cannot_absorb(tmp_path):
    from agentigrid.parsers.matpower_parser import parse_matpower
    base = parse_matpower(ROOT.parent / "datafiles" / "case9" / "case9mod.m")
    with pytest.raises(ValueError):
        pc.agc_redispatch(base, 1e6)


def test_unphysical_slack_makes_pflow_goal_unusable():
    r = {"opflow": _rec(), "scopflow": _rec(), "pflow": _rec(feasible=False, therm=3),
         "_slack_check": {"within_limits": False}}
    assert pc.check_goal("relieve", r) is False
    r["_slack_check"] = {"within_limits": True}
    assert pc.check_goal("relieve", r) is True


# --- single-branch outages (relieve stress) ----------------------------------

def test_parse_outage():
    assert pc.parse_outage("12-34") == (12, 34, 0)
    assert pc.parse_outage("12-34-1") == (12, 34, 1)
    with pytest.raises(ValueError):
        pc.parse_outage("12")


@needs_200
def test_branch_keys_islanding_and_outage_variant(tmp_path):
    from agentigrid.parsers.matpower_parser import parse_matpower
    base = parse_matpower(G200)
    keys = pc.branch_keys(base)
    assert len(keys) == len(base.branches) and len(set(keys)) == len(keys)
    # a radial branch (one endpoint of degree 1) islands; count that some do and some don't
    flags = [pc.islands_without(base, i) for i in range(len(base.branches))]
    assert any(flags) and not all(flags)
    i = flags.index(False)
    out = pc.write_stressed(G200, 1.0, tmp_path, outage=keys[i])
    net = parse_matpower(out)
    assert net.branches[i].status == 0
    assert sum(br.status == 0 for br in net.branches) == sum(br.status == 0 for br in base.branches) + 1
    assert out.name == f"case_ACTIVSg200_load1_out{keys[i][0]}-{keys[i][1]}" + (f"-{keys[i][2]}" if keys[i][2] else "") + ".m"


# --- cost headroom (allowed levers for cost goals) ----------------------------

def test_greedy_commitment_keeps_only_improving_units():
    singles = {("a", 0): 95.0, ("b", 0): 97.0, ("c", 0): 101.0, ("d", 0): None}
    combos = {(("a", 0),): 95.0, (("a", 0), ("b", 0)): 96.0}

    def combo(ks):
        return combos.get(tuple(ks))
    g = pc.greedy_commitment(100.0, singles, combo)
    assert g["committed"] == [("a", 0)] and g["cost"] == 95.0 and g["improvement_pct"] == 5.0


@needs_200
def test_gen_keys_address_units_like_set_gen_status():
    from agentigrid.parsers.matpower_parser import parse_matpower
    net = parse_matpower(G200)
    keys = pc.gen_keys(net)
    assert len(keys) == len(net.generators) and len(set(keys)) == len(keys)
    assert all(k >= 0 for _b, k in keys)
