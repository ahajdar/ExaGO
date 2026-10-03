#!/usr/bin/env python3
"""Base-case PRE-CHECK for the of-record experiment goals.

Answers, with ExaGO itself, whether each goal is meaningful on a case BEFORE any
LLM run is spent on it:

  * n1       (SCOPFLOW, all contingencies): if the unmodified case already has a
             feasible N-1-secure dispatch, SCOPFLOW solves the goal at iteration 0
             in every condition, so the goal cannot tell conditions apart.
  * relieve  (PFLOW): needs at least one branch overload at the base point.
  * voltage  (PFLOW): needs at least one bus-voltage violation at the base point.
  * OPFLOW is solved too, for the reference cost and feasibility.

If a goal is degenerate at scale 1.0, the script scans uniform load-scaling
factors and reports the lowest factor at which the goal becomes meaningful —
the candidate "stressed variant". `--write-stressed F` then freezes the case
scaled by F to a new .m file for the spec to point at.

It reuses AgentiGrid's own executor, command set and parsers, so what it reports
is exactly what the agent would see. No LLM, no network.

Usage (from the agentigrid project root, venv active):

    python rag/tools/precheck_cases.py --config configs/local_config.yaml
    python rag/tools/precheck_cases.py --scales 1.0,1.1,1.2,1.3 --json precheck.json
    python rag/tools/precheck_cases.py --write-stressed 1.15   # freeze a variant
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_CASE = "../datafiles/case_ACTIVSg200.m"
DEFAULT_CTGC = "../datafiles/case_ACTIVSg200.cont"
DEFAULT_SCALES = "1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.4,1.5"
APPS = ("opflow", "pflow", "scopflow")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without ExaGO)
# ---------------------------------------------------------------------------

def split_violations(details: list[str]) -> dict:
    """Count violation kinds from the parser's violation_details strings."""
    volt = sum(1 for d in details if "undervoltage" in d or "overvoltage" in d)
    therm = sum(1 for d in details if d.startswith("Branch "))
    balance = sum(1 for d in details if d.startswith("Power balance"))
    return {"voltage_violations": volt, "thermal_violations": therm,
            "balance_violations": balance}


def summarize_result(res, sim) -> dict:
    """Flatten an OPFLOWResult (+ SimulationResult) into a JSON-able record."""
    if res is None:
        return {"parsed": False, "converged": False, "status": "FAILED",
                "error": getattr(sim, "error_message", None) or (getattr(sim, "stderr", "") or "")[-400:],
                "argv": list(getattr(sim, "argv", []) or []),
                "elapsed_s": round(getattr(sim, "elapsed_seconds", 0.0) or 0.0, 2)}
    rec = {
        "parsed": True,
        "converged": bool(res.converged),
        "status": res.convergence_status,
        "feasibility": res.feasibility_detail,
        "ipopt_exit": getattr(res, "ipopt_exit_status", "") or "",
        "objective": res.objective_value,
        "num_violations": res.num_violations,
        "max_line_loading_pct": round(res.max_line_loading_pct, 2),
        "voltage_min": round(res.voltage_min, 4),
        "voltage_max": round(res.voltage_max, 4),
        "argv": list(getattr(sim, "argv", []) or []),
        "elapsed_s": round(getattr(sim, "elapsed_seconds", 0.0) or 0.0, 2),
    }
    rec.update(split_violations(list(res.violation_details or [])))
    return rec


def _feasible(rec: dict | None) -> bool:
    return bool(rec and rec.get("parsed") and rec.get("converged")
                and rec.get("feasibility") == "feasible" and rec.get("num_violations", 0) == 0)


def _pflow_usable(rec: dict | None) -> bool:
    """PFLOW must converge for its violations to mean anything."""
    return bool(rec and rec.get("parsed") and rec.get("converged"))


def _ran(rec: dict | None) -> bool:
    """The solver actually ran and its output was parsed (converged or not)."""
    return bool(rec and rec.get("parsed"))


def check_goal(goal: str, r: dict[str, dict]):
    """True / False = goal meaningful / degenerate at this scale;
    None = undetermined because a required solve did not run or was not parsed
    (e.g. a missing file). Undetermined is never reported as usable."""
    if goal == "n1":
        o, sc = r.get("opflow"), r.get("scopflow")
        if not (_ran(o) and _ran(sc)):
            return None
        if not _feasible(o):
            return False            # base infeasibility, not an N-1 problem
        if _feasible(sc):
            return False            # SCOPFLOW already solves the goal
        if sc.get("feasibility") == "marginal":
            # IPOPT stopped without a verdict (iteration limit, tiny steps, ...):
            # that is not evidence that no N-1-secure dispatch exists.
            return None
        return True
    p = r.get("pflow")
    if not _ran(p):
        return None
    sc = r.get("_slack_check")
    if isinstance(sc, dict) and not sc.get("within_limits", True):
        return False                # slack outside its limits: not a physical operating point
    if goal == "relieve":
        return _pflow_usable(p) and p.get("thermal_violations", 0) > 0
    if goal == "voltage":
        return _pflow_usable(p) and p.get("voltage_violations", 0) > 0
    raise ValueError(goal)


def classify(results: dict[float, dict[str, dict]]) -> dict:
    """Verdict per goal from {scale: {app: record}}.

    A goal is meaningful at a scale when the case at that scale poses the
    problem the goal asks the agent to solve:
      n1      -> SCOPFLOW (all contingencies) is NOT feasible, while OPFLOW is
                 (so N-1 security is the binding issue, not base infeasibility);
      relieve -> PFLOW converges and shows >= 1 thermal overload;
      voltage -> PFLOW converges and shows >= 1 bus-voltage violation.
    For PFLOW goals the operating point must also be physical: with plain load
    scaling the slack unit absorbs the whole change, and a scale that pushes it
    outside [Pmin, Pmax] is rejected (use --agc to share the change instead).
    A scale where a required solve failed to run is UNDETERMINED, not usable.
    """
    scales = sorted(results)
    base = 1.0 if 1.0 in results else scales[0]
    out = {}
    for goal in ("n1", "relieve", "voltage"):
        checks = {s: check_goal(goal, results[s]) for s in scales}
        ok_scales = [s for s in scales if checks[s] is True]
        undetermined = [s for s in scales if checks[s] is None]
        first = ok_scales[0] if ok_scales else None
        at_base = checks[base] is True
        if checks[base] is None:
            verdict = "UNDETERMINED at base — a required solve did not run; fix the error and re-run"
        elif at_base:
            verdict = "OK at base case — use the unmodified case"
        elif first is not None:
            verdict = f"degenerate at base; meaningful from load scale x{first} — freeze a stressed variant"
        elif undetermined:
            verdict = "degenerate where determinable; some scales undetermined — fix errors and re-run"
        else:
            verdict = "degenerate at every scanned scale — widen --scales or redesign the goal"
        out[goal] = {"meaningful_at_base": at_base, "first_meaningful_scale": first,
                     "meaningful_scales": ok_scales, "undetermined_scales": undetermined,
                     "verdict": verdict}
    b = results[base]
    if _ran(b.get("scopflow")) and _feasible(b.get("scopflow")):
        out["n1"]["note"] = ("SCOPFLOW is already feasible with all contingencies at the base case, "
                             "so it returns the minimum-cost N-1-secure dispatch by itself.")
    elif _ran(b.get("opflow")) and not _feasible(b.get("opflow")):
        out["n1"]["note"] = "OPFLOW itself is infeasible at the base case — the problem is not N-1 specific."
    return out


# ---------------------------------------------------------------------------
# ExaGO-backed run
# ---------------------------------------------------------------------------

def scopflow_args(ctgc: Path, mpi_np: int) -> list[str]:
    """Exactly what AgentiGrid passes for SCOPFLOW (see agent_loop._build_extra_args)."""
    # Absolute: ExaGO runs with cwd = its per-iteration run dir, so a relative path
    # would not resolve (AgentiGrid's config loader makes ctgc_file absolute too).
    args = ["-ctgcfile", str(Path(ctgc).resolve()), "-scopflow_Nc", "-1"]
    # Deliberately NOT switching to EMPAR when mpi_np > 1 (AgentiGrid does):
    # EMPAR solves the base case and contingencies independently, so it cannot
    # tell whether an N-1-secure dispatch exists.
    return args


def run_scan(cfg, case: Path, ctgc: Path, scales: list[float], apps: tuple[str, ...],
             vband: tuple[float, float] | None = None, agc: bool = False):
    from agentigrid.engine.commands import ScaleAllLoads
    from agentigrid.engine.executor import SimulationExecutor
    from agentigrid.engine.modifier import apply_modifications
    from agentigrid.parsers import parse_simulation_result_for_app
    from agentigrid.parsers.matpower_parser import parse_matpower

    base_net = parse_matpower(case)
    limits = {b.bus_i: (b.Vmin, b.Vmax) for b in base_net.buses}
    if vband is not None:  # judge voltage violations against a tighter/looser band
        limits = {bus: vband for bus in limits}
    executor = SimulationExecutor(cfg.exago, cfg.output)
    results: dict[float, dict[str, dict]] = {}
    for i, s in enumerate(scales):
        net = base_net
        if abs(s - 1.0) > 1e-12:
            try:
                net = stressed_network(base_net, s, agc)
            except ValueError as exc:
                print(f"  x{s:<5} skipped: {exc}")
                results[s] = {"_skipped": str(exc)}
                continue
        results[s] = {}
        sc = slack_check(net, base_net) if not agc else None
        if sc is not None and not sc["within_limits"]:
            sl = sc["slack"][0]
            print(f"  x{s:<5} WARNING: slack bus {sl['bus']} would carry ~{sl['Pg_est']} MW, outside "
                  f"[{sl['Pmin']}, {sl['Pmax']}] (losses ignored) -- PFLOW point is not physical; use --agc")
        results[s]["_slack_check"] = sc
        for app in apps:
            extra = scopflow_args(ctgc, cfg.exago.mpi_np) if app == "scopflow" else None
            sim = executor.run(net, application=app, iteration=900 + i, extra_args=extra)
            res = parse_simulation_result_for_app(sim, app, bus_limits=limits) if sim.success else None
            rec = summarize_result(res, sim)
            results[s][app] = rec
            print(f"  x{s:<5} {app:<9} status={rec.get('status')!s:<22} "
                  f"feas={rec.get('feasibility')!s:<10} viol(V/T)={rec.get('voltage_violations','-')}/"
                  f"{rec.get('thermal_violations','-')}  maxload={rec.get('max_line_loading_pct','-')}%  "
                  f"V=[{rec.get('voltage_min','-')},{rec.get('voltage_max','-')}]  "
                  f"obj={rec.get('objective')}  ({rec.get('elapsed_s')}s)"
                  + (f"  ipopt={rec['ipopt_exit']!r}" if rec.get("ipopt_exit") and not rec.get("converged") else ""))
            if not rec.get("parsed"):
                print(f"           error: {str(rec.get('error') or 'no output').strip()[:300]}")
            if rec.get("ipopt_exit"):
                print(f"           ipopt: {rec['ipopt_exit']}")
    return results


RENEWABLE_FUELS = ("wind", "solar")


def _gen_fuels(net) -> list[str]:
    """Fuel label per generator from mpc.genfuel ('' when absent)."""
    raw = net.extra_sections.get("genfuel", "")
    fuels = []
    for line in raw.split("\n"):
        t = line.strip().strip("';").strip()
        if t and not t.startswith("%") and not t.startswith("mpc.") and t not in ("{", "}"):
            fuels.append(t.lower())
    return fuels + [""] * (len(net.generators) - len(fuels))


def slack_check(net, base_net) -> dict:
    """Where the slack unit lands if it alone picks up the load change of *net*
    relative to *base_net* (losses ignored), and whether that is inside its limits.
    Plain load scaling leaves every other set-point unchanged, so in PFLOW the
    slack absorbs the whole change; outside [Pmin, Pmax] the operating point is
    not physical."""
    ref = {b.bus_i for b in net.buses if b.type == 3}
    delta = sum(b.Pd for b in net.buses) - sum(b.Pd for b in base_net.buses)
    out = []
    for g in net.generators:
        if g.status == 1 and g.bus in ref:
            est = g.Pg + delta
            out.append({"bus": g.bus, "Pg_est": round(est, 1), "Pmin": g.Pmin, "Pmax": g.Pmax,
                        "within_limits": g.Pmin - 1e-6 <= est <= g.Pmax + 1e-6})
            delta = 0.0     # first slack unit takes it all
    return {"load_change_mw": round(sum(b.Pd for b in net.buses) - sum(b.Pd for b in base_net.buses), 1),
            "slack": out, "within_limits": all(x["within_limits"] for x in out)}


def agc_redispatch(net, delta_mw: float):
    """Share a load change of *delta_mw* among the online non-slack units, in
    proportion to their Pmax (a common AGC participation convention), clipped to
    unit limits and re-shared until absorbed. Thermal units move within
    [Pmin, Pmax]; wind/solar can be curtailed down to 0 but not raised above
    their forecast (the Pg in the file). The slack unit is left to cover losses.
    Returns a modified deep copy; raises ValueError if the fleet cannot absorb it."""
    import copy

    net = copy.deepcopy(net)
    fuels = _gen_fuels(net)
    ref = {b.bus_i for b in net.buses if b.type == 3}
    lo, hi = {}, {}
    for i, g in enumerate(net.generators):
        if g.status != 1 or g.bus in ref or g.Pmax <= 0:
            continue
        ren = fuels[i] in RENEWABLE_FUELS
        lo[i] = 0.0 if ren else g.Pmin
        hi[i] = g.Pg if ren else g.Pmax
    remaining = float(delta_mw)
    for _ in range(len(lo) + 1):
        up = remaining > 0
        active = [i for i in lo
                  if (net.generators[i].Pg < hi[i] - 1e-9 if up else net.generators[i].Pg > lo[i] + 1e-9)]
        weight = sum(net.generators[i].Pmax for i in active)
        if abs(remaining) < 1e-6 or weight <= 0:
            break
        moved = 0.0
        for i in active:
            g = net.generators[i]
            new = min(hi[i], max(lo[i], g.Pg + remaining * g.Pmax / weight))
            moved += new - g.Pg
            g.Pg = new
        remaining -= moved
    if abs(remaining) > 1e-6:
        raise ValueError(f"AGC redispatch cannot absorb {delta_mw:.1f} MW "
                         f"({remaining:.1f} MW left at unit limits)")
    return net


def branch_keys(net) -> list[tuple[int, int, int]]:
    """(fbus, tbus, ckt) per branch, ckt = 0-based ordinal among parallel branches
    of the same bus pair -- the addressing AgentiGrid's set_branch_status uses."""
    seen: dict[tuple[int, int], int] = {}
    keys = []
    for br in net.branches:
        pair = (min(br.fbus, br.tbus), max(br.fbus, br.tbus))
        k = seen.get(pair, 0)
        seen[pair] = k + 1
        keys.append((br.fbus, br.tbus, k))
    return keys


def parse_outage(text: str) -> tuple[int, int, int]:
    """'FBUS-TBUS' or 'FBUS-TBUS-CKT' -> (fbus, tbus, ckt)."""
    parts = [int(x) for x in text.split("-")]
    if len(parts) not in (2, 3):
        raise ValueError(f"outage must be FBUS-TBUS[-CKT], got {text!r}")
    return (parts[0], parts[1], parts[2] if len(parts) == 3 else 0)


def islands_without(net, skip_index: int) -> bool:
    """True if taking branch *skip_index* out splits the in-service network."""
    buses = [b.bus_i for b in net.buses if b.type != 4]
    adj: dict[int, list[int]] = {b: [] for b in buses}
    for i, br in enumerate(net.branches):
        if i == skip_index or br.status != 1 or br.fbus not in adj or br.tbus not in adj:
            continue
        adj[br.fbus].append(br.tbus)
        adj[br.tbus].append(br.fbus)
    if not buses:
        return False
    seen, stack = {buses[0]}, [buses[0]]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) != len(buses)


def with_outage(net, outage: tuple[int, int, int]):
    from agentigrid.engine.commands import SetBranchStatus
    from agentigrid.engine.modifier import apply_modifications

    fbus, tbus, ckt = outage
    out, _ = apply_modifications(net, [SetBranchStatus(fbus=fbus, tbus=tbus, status=0, ckt=ckt)])
    return out


def stressed_network(base_net, factor: float, agc: bool = False, outage=None):
    from agentigrid.engine.commands import ScaleAllLoads
    from agentigrid.engine.modifier import apply_modifications

    net, _ = apply_modifications(base_net, [ScaleAllLoads(factor=factor)])
    if agc:
        delta = sum(b.Pd for b in net.buses) - sum(b.Pd for b in base_net.buses)
        net = agc_redispatch(net, delta)
    if outage is not None:
        net = with_outage(net, outage)
    return net


def write_stressed(case: Path, factor: float, out_dir: Path,
                   vband: tuple[float, float] | None = None, agc: bool = False,
                   outage: tuple[int, int, int] | None = None) -> Path:
    from agentigrid.parsers.matpower_parser import parse_matpower
    from agentigrid.parsers.matpower_writer import write_matpower

    net = stressed_network(parse_matpower(case), factor, agc, outage)
    suffix = f"_load{factor:g}" + ("_agc" if agc else "")
    if outage is not None:
        suffix += f"_out{outage[0]}-{outage[1]}" + (f"-{outage[2]}" if outage[2] else "")
    if vband is not None:
        for b in net.buses:
            b.Vmin, b.Vmax = vband
        suffix += f"_v{vband[0]:g}-{vband[1]:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{case.stem}{suffix}.m"
    write_matpower(net, out)
    return out


def gen_keys(net) -> list[tuple[int, int]]:
    """(bus, gen_id) per generator, gen_id = 0-based ordinal among the bus's units
    (the addressing AgentiGrid's set_gen_status uses)."""
    seen: dict[int, int] = {}
    out = []
    for g in net.generators:
        k = seen.get(g.bus, 0)
        seen[g.bus] = k + 1
        out.append((g.bus, k))
    return out


def greedy_commitment(base_cost: float, singles: dict, combo_cost) -> dict:
    """Greedy unit commitment over offline units: try units in order of their
    single-unit saving, keep each one that lowers the cost further.
    singles: {key: cost with only that unit committed (None = failed/infeasible)};
    combo_cost(keys) -> cost (None = failed)."""
    order = sorted((k for k, c in singles.items() if c is not None and c < base_cost),
                   key=lambda k: singles[k])
    chosen, best = [], base_cost
    for k in order:
        c = combo_cost(chosen + [k])
        if c is not None and c < best - 1e-6:
            chosen, best = chosen + [k], c
    return {"committed": chosen, "cost": best,
            "improvement_pct": round((base_cost - best) / base_cost * 100.0, 3) if base_cost else None}


def cost_headroom(cfg, case: Path, app: str, ctgc: Path, scale: float, agc: bool) -> dict:
    """How much of a cost goal is reachable with levers ALLOWED under the intent
    guards? The baseline is already an optimal (SC)OPF, so with load, cost
    curves, voltage band and ratings fixed, the main remaining lever is unit
    commitment, which OPF itself does not decide: bringing offline units online,
    and taking online units offline (saving their fixed cost term, which OPF pays
    even at Pmin). Solves the baseline, every single commitment change, and a
    greedy combination. Taps, shunts and phase shifters are not searched, so the
    result is a lower bound on the achievable saving."""
    from agentigrid.engine.commands import SetGenStatus
    from agentigrid.engine.executor import SimulationExecutor
    from agentigrid.engine.modifier import apply_modifications
    from agentigrid.parsers import parse_simulation_result_for_app
    from agentigrid.parsers.matpower_parser import parse_matpower

    base_file = parse_matpower(case)
    base = base_file if abs(scale - 1.0) < 1e-12 else stressed_network(base_file, scale, agc)
    executor = SimulationExecutor(cfg.exago, cfg.output)
    extra = scopflow_args(ctgc, cfg.exago.mpi_np) if app == "scopflow" else None

    def solve(net, it):
        sim = executor.run(net, application=app, iteration=it, extra_args=extra)
        res = parse_simulation_result_for_app(sim, app) if sim.success else None
        rec = summarize_result(res, sim)
        return rec["objective"] if _feasible(rec) else None

    base_cost = solve(base, 970)
    if base_cost is None:
        return {"error": "baseline not feasible", "base_cost": None}
    keys = gen_keys(base)
    ref = {b.bus_i for b in base.buses if b.type == 3}
    actions = []                         # (bus, gen_id, new_status)
    for i, g in enumerate(base.generators):
        if g.Pmax <= 0:
            continue
        if g.status != 1:
            actions.append((*keys[i], 1))
        elif g.bus not in ref:
            actions.append((*keys[i], 0))

    def with_changes(acts):
        net, _ = apply_modifications(base, [SetGenStatus(bus=b, status=st, gen_id=k) for b, k, st in acts])
        return net

    singles = {}
    for act in actions:
        singles[act] = c = solve(with_changes([act]), 971)
        verb = "commit" if act[2] == 1 else "decommit"
        print(f"  {verb:<8} bus {act[0]} (unit {act[1]}): "
              + (f"cost {c:,.2f} ({(base_cost - c) / base_cost * 100:+.3f}%)" if c is not None else "not feasible"))
    greedy = greedy_commitment(base_cost, singles, lambda acts: solve(with_changes(acts), 972))
    return {"case": str(case), "app": app, "scale": scale, "agc": agc, "base_cost": base_cost,
            "candidates": len(actions),
            "singles": {f"{b}-{k}->{st}": c for (b, k, st), c in singles.items()}, "greedy": greedy}


def outage_scan(cfg, case: Path, scale: float, agc: bool, vband, top: int = 15) -> list[dict]:
    """Single-branch outages at one load level: PFLOW for every non-islanding
    outage, then OPFLOW for the *top* outages by overload count / max loading.
    A usable relieve case: PFLOW converges with >= 1 overload, the slack stays
    inside its limits, and OPFLOW with the outage is feasible (a fix exists)."""
    from agentigrid.engine.executor import SimulationExecutor
    from agentigrid.parsers import parse_simulation_result_for_app
    from agentigrid.parsers.matpower_parser import parse_matpower

    base_file = parse_matpower(case)
    base = base_file if abs(scale - 1.0) < 1e-12 else stressed_network(base_file, scale, agc)
    limits = {b.bus_i: (vband if vband else (b.Vmin, b.Vmax)) for b in base.buses}
    sc = slack_check(base, base_file)
    if not agc and not sc["within_limits"]:
        print(f"WARNING: at x{scale} the slack leaves its limits without --agc; results are not physical.")
    executor = SimulationExecutor(cfg.exago, cfg.output)
    keys = branch_keys(base)
    rows: list[dict] = []
    n_island = 0
    for i, (br, key) in enumerate(zip(base.branches, keys)):
        if br.status != 1:
            continue
        if islands_without(base, i):
            n_island += 1
            continue
        net = with_outage(base, key)
        sim = executor.run(net, application="pflow", iteration=960)
        res = parse_simulation_result_for_app(sim, "pflow", bus_limits=limits) if sim.success else None
        rec = summarize_result(res, sim)
        rows.append({"outage": key, "rateA": br.rateA, "pflow": rec})
    usable = [r for r in rows if _pflow_usable(r["pflow"]) and r["pflow"].get("thermal_violations", 0) > 0]
    usable.sort(key=lambda r: (-r["pflow"].get("thermal_violations", 0), -r["pflow"].get("max_line_loading_pct", 0)))
    for r in usable[:top]:
        net = with_outage(base, r["outage"])
        sim = executor.run(net, application="opflow", iteration=961)
        res = parse_simulation_result_for_app(sim, "opflow", bus_limits=limits) if sim.success else None
        r["opflow"] = summarize_result(res, sim)
    print(f"x{scale}{' (agc)' if agc else ''}: {len(rows)} outages solved, {n_island} skipped (would island), "
          f"{len(usable)} with PFLOW overloads; top {min(top, len(usable))}:")
    print("  outage (f-t-ckt)   overloads  maxload%  V-viol  OPFLOW       cost")
    for r in usable[:top]:
        p, o = r["pflow"], r.get("opflow", {})
        f, t, c = r["outage"]
        print(f"  {f:>5}-{t:<5}-{c:<3}   {p.get('thermal_violations', 0):>6}    {p.get('max_line_loading_pct', '-'):>7}  "
              f"{p.get('voltage_violations', 0):>5}   {('feasible' if _feasible(o) else o.get('feasibility') or o.get('status')):<11} {o.get('objective')}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="AgentiGrid config YAML (default: built-in defaults)")
    ap.add_argument("--case", default=DEFAULT_CASE)
    ap.add_argument("--ctgc", default=DEFAULT_CTGC)
    ap.add_argument("--scales", default=DEFAULT_SCALES, help="comma-separated load-scaling factors")
    ap.add_argument("--apps", default=",".join(APPS))
    ap.add_argument("--json", help="write full results + verdicts to this file")
    ap.add_argument("--write-stressed", type=float, metavar="F",
                    help="write the case scaled by F to --stressed-dir and exit")
    ap.add_argument("--stressed-dir", default="data/exago/datafiles")
    ap.add_argument("--vband", metavar="LO,HI",
                    help="judge bus-voltage violations against this band (pu) instead of the "
                         "case's Vmin/Vmax; with --write-stressed, also write it into the variant")
    ap.add_argument("--agc", action="store_true",
                    help="share the load change among online non-slack units in proportion to "
                         "Pmax (AGC-style) instead of leaving it all to the slack unit")
    ap.add_argument("--outage", metavar="FBUS-TBUS[-CKT]",
                    help="with --write-stressed: also take this branch out of service")
    ap.add_argument("--outage-scan", type=float, metavar="F",
                    help="scan single-branch outages at load scale F (PFLOW, then OPFLOW for the "
                         "top candidates) and exit")
    ap.add_argument("--top", type=int, default=15, help="outage-scan: candidates to check with OPFLOW")
    ap.add_argument("--cost-headroom", metavar="APP", choices=("opflow", "scopflow"),
                    help="estimate the cost saving reachable with allowed levers (unit commitment: "
                         "committing offline and decommitting online units) for a cost goal solved "
                         "with APP, at --scales' first value; exit")
    args = ap.parse_args(argv)

    case, ctgc = Path(args.case), Path(args.ctgc)
    if not case.exists():
        print(f"case not found: {case}", file=sys.stderr)
        return 2

    vband = None
    if args.vband:
        lo, hi = (float(x) for x in args.vband.split(","))
        vband = (lo, hi)

    if args.write_stressed is not None:
        outage = parse_outage(args.outage) if args.outage else None
        out = write_stressed(case, args.write_stressed, Path(args.stressed_dir), vband, args.agc, outage)
        print(f"Wrote stressed variant: {out}\nPoint the spec's pflow (and/or scopflow) case entry at it.")
        return 0

    if args.cost_headroom:
        from agentigrid.config import load_config
        cfg = load_config(args.config) if args.config else load_config(None)
        scale = float(args.scales.split(",")[0])
        rep = cost_headroom(cfg, case, args.cost_headroom, ctgc, scale, args.agc)
        g = rep.get("greedy") or {}
        print(f"\nbase cost {rep.get('base_cost')}; {rep.get('candidates')} commitment changes tried; greedy "
              f"combination of {len(g.get('committed', []))} change(s) -> cost {g.get('cost')} "
              f"({g.get('improvement_pct')}% saving)")
        if args.json:
            Path(args.json).write_text(json.dumps(rep, indent=2, default=str))
        return 0

    if args.outage_scan is not None:
        from agentigrid.config import load_config
        cfg = load_config(args.config) if args.config else load_config(None)
        rows = outage_scan(cfg, case, args.outage_scan, args.agc, vband, args.top)
        if args.json:
            Path(args.json).write_text(json.dumps({
                "case": str(case), "scale": args.outage_scan, "agc": args.agc,
                "timestamp": datetime.now().isoformat(),
                "rows": [{**r, "outage": list(r["outage"])} for r in rows]}, indent=2))
        return 0

    apps = tuple(a.strip() for a in args.apps.split(",") if a.strip())
    if "scopflow" in apps and not ctgc.exists():
        print(f"contingency file not found: {ctgc}", file=sys.stderr)
        return 2
    scales = sorted({float(x) for x in args.scales.split(",") if x.strip()})

    from agentigrid.config import load_config
    cfg = load_config(args.config) if args.config else load_config(None)

    print(f"Pre-check {case} (contingencies: {ctgc}) at scales {scales}\n")
    if vband:
        print(f"Voltage band for violation checks: {vband[0]}-{vband[1]} pu (overrides case limits)\n")
    results = run_scan(cfg, case, ctgc, scales, apps, vband, args.agc)
    verdicts = classify(results) if {"opflow", "pflow", "scopflow"} <= set(apps) else {}

    print("\nVerdicts")
    for goal, v in verdicts.items():
        print(f"  {goal:<8} {v['verdict']}")
        if v.get("note"):
            print(f"           note: {v['note']}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "case": str(case), "ctgc": str(ctgc), "scales": scales,
            "timestamp": datetime.now().isoformat(),
            "results": {str(s): r for s, r in results.items()},
            "verdicts": verdicts,
        }, indent=2))
        print(f"\nFull results: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
