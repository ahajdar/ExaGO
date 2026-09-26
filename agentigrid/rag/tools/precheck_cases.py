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
             vband: tuple[float, float] | None = None):
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
            net, _report = apply_modifications(base_net, [ScaleAllLoads(factor=s)])
        results[s] = {}
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


def write_stressed(case: Path, factor: float, out_dir: Path,
                   vband: tuple[float, float] | None = None) -> Path:
    from agentigrid.engine.commands import ScaleAllLoads
    from agentigrid.engine.modifier import apply_modifications
    from agentigrid.parsers.matpower_parser import parse_matpower
    from agentigrid.parsers.matpower_writer import write_matpower

    net, _ = apply_modifications(parse_matpower(case), [ScaleAllLoads(factor=factor)])
    suffix = f"_load{factor:g}"
    if vband is not None:
        for b in net.buses:
            b.Vmin, b.Vmax = vband
        suffix += f"_v{vband[0]:g}-{vband[1]:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{case.stem}{suffix}.m"
    write_matpower(net, out)
    return out


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
    ap.add_argument("--stressed-dir", default="data")
    ap.add_argument("--vband", metavar="LO,HI",
                    help="judge bus-voltage violations against this band (pu) instead of the "
                         "case's Vmin/Vmax; with --write-stressed, also write it into the variant")
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
        out = write_stressed(case, args.write_stressed, Path(args.stressed_dir), vband)
        print(f"Wrote stressed variant: {out}\nPoint the spec's pflow (and/or scopflow) case entry at it.")
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
    results = run_scan(cfg, case, ctgc, scales, apps, vband)
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
