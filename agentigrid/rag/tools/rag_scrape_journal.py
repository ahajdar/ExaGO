#!/usr/bin/env python3
"""Exemplar harvester — turns SOLVER-CERTIFIED AgentiGrid iterations into RAG corpus.

AgentiGrid already contains a correctness oracle: the deterministic command
validator plus the ExaGO solve. An iteration is kept as an exemplar only if

  * it is a modify iteration after the baseline (iteration > 0) whose `commands`
    list is non-empty and every command parses with AgentiGrid's own parser;
  * every proposed command was APPLIED (journal `skipped_commands == []`; journals
    written before that field existed are refused unless --allow-legacy);
  * the solve converged, is feasible, has zero violations and an objective value;
  * it is a real single solve (not ANALYSIS/COMPLETE/EXPLORE/SWEEP/CONTINGENCY);
  * its description is real (not empty / "No description" / "no commands applied").

The exemplar teaches what the model must EMIT: the full `modify` action with its
nested `commands` — never the ExaGO command line (which carried absolute paths
and is not what the model writes).

Hold-out: journals from the of-record experiment's networks or goals are
excluded (default: grader_ablation_spec.json if present), so the corpus never
contains worked solutions to the evaluated tasks.

Run from the project root:
    python rag/tools/rag_scrape_journal.py --runs-dir experiments/corpus_bootstrap_v1 --inspect
    python rag/tools/rag_scrape_journal.py --runs-dir experiments/corpus_bootstrap_v1
    rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

# Mirror journal.py: statuses that never denote a real scalar solve.
NON_SOLVE_STATUSES = frozenset({"ANALYSIS", "COMPLETE", "EXPLORE", "SWEEP", "CONTINGENCY"})
_BAD_DESC = {"", "no description", "n/a", "none"}
_TODAY = _dt.date.today().isoformat()
DEFAULT_EXCLUDE_SPEC = "grader_ablation_spec.json"
OUT_NAME = "exago_exemplars_from_runs.txt"


def _looks_like_journal(data: object) -> bool:
    if not isinstance(data, dict) or "entries" not in data:
        return False
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        return False
    first = entries[0]
    return isinstance(first, dict) and "iteration" in first and "convergence_status" in first


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def app_of(entry: dict) -> str:
    ec = entry.get("exago_command")
    if isinstance(ec, dict):
        app = ec.get("application")
        if isinstance(app, str) and app:
            return app.lower()
        argv = ec.get("argv")
        if isinstance(argv, list) and argv:
            return Path(str(argv[0])).name.lower()
    return "unknown"


def network_of(entry: dict) -> str | None:
    """Network stem from the recorded -netfile (basename only; never the path)."""
    ec = entry.get("exago_command")
    argv = ec.get("argv") if isinstance(ec, dict) else None
    if isinstance(argv, list) and "-netfile" in argv:
        i = argv.index("-netfile")
        if i + 1 < len(argv):
            return Path(str(argv[i + 1])).stem
    return None


def _commands_parse(cmds) -> bool:
    try:
        from agentigrid.engine.commands import parse_command
    except Exception:
        return False
    try:
        for c in cmds:
            parse_command(c)
    except Exception:
        return False
    return True


def reject_reason(entry: dict, allow_legacy: bool = False) -> str | None:
    """None if the entry is a certified exemplar, else a short reason."""
    if not isinstance(entry, dict):
        return "not an entry"
    if entry.get("iteration", 0) == 0:
        return "baseline"
    if entry.get("convergence_status") in NON_SOLVE_STATUSES or entry.get("convergence_status") == "FAILED":
        return "not a solve"
    if not entry.get("feasible") or (entry.get("violations_count") or 0) != 0:
        return "infeasible or violations"
    if entry.get("objective_value") is None:
        return "no objective"
    cmds = entry.get("commands")
    if not isinstance(cmds, list) or not cmds or not all(isinstance(c, dict) for c in cmds):
        return "no proposal"
    desc = _norm(entry.get("description"))
    if desc in _BAD_DESC or "no commands applied" in desc:
        return "no real description"
    skipped = entry.get("skipped_commands")
    if skipped is None and not allow_legacy:
        return "legacy journal (no skip record)"
    if skipped:
        return "some commands were skipped"
    if not _commands_parse(cmds):
        return "command does not parse"
    return None


def load_holdout(spec_paths: list[Path]) -> dict:
    goals, networks = set(), set()
    for sp in spec_paths:
        spec = json.loads(Path(sp).read_text())
        goals |= {_norm(g["text"]) for g in spec.get("goals", []) if g.get("text")}
        for c in spec.get("cases", []):
            if c.get("path"):
                networks.add(Path(c["path"]).stem.lower())
    return {"goals": goals, "networks": networks}


def held_out(entry: dict, manifest: dict | None, holdout: dict) -> str | None:
    net = network_of(entry)
    if net and net.lower() in holdout["networks"]:
        return f"evaluated network {net}"
    if manifest:
        if _norm(manifest.get("goal_text")) in holdout["goals"]:
            return "evaluated goal"
        mnet = Path(str(manifest.get("case_path", ""))).stem.lower()
        if mnet and mnet in holdout["networks"]:
            return f"evaluated network {mnet}"
    if _norm(entry.get("description")) in holdout["goals"]:
        return "evaluated goal"
    return None


def exemplar_text(entry: dict, manifest: dict | None) -> str:
    app = app_of(entry)
    net = network_of(entry) or (Path(str(manifest.get("case_path", ""))).stem if manifest else "") or "unknown"
    desc = (entry.get("description") or "").strip()
    reasoning = re.sub(r"\s+", " ", (entry.get("llm_reasoning") or "").strip())
    if len(reasoning) > 400:
        reasoning = reasoning[:400].rstrip() + "…"
    action = {"action": "modify", "mode": entry.get("mode") or "accumulative",
              "description": desc, "commands": entry["commands"]}
    model = (manifest or {}).get("model", "unknown model")
    tag = (f"[source: solver-certified AgentiGrid run | app: {app} | network: {net} | "
           f"generated by: {model} | harvested: {_TODAY}]")
    lines = [tag]
    if manifest and manifest.get("goal_text"):
        lines.append(f"Goal: {manifest['goal_text']}")
    lines.append(f"Step: {desc}")
    if reasoning:
        lines.append(f"Reasoning: {reasoning}")
    lines.append(f"Correct response (JSON): {json.dumps(action, separators=(', ', ': '))}")
    lines.append(f"Result: {app.upper()} converged, feasible, 0 violations, "
                 f"objective {entry['objective_value']:,.2f}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="workdir",
                    help="directory scanned recursively for journal JSON (runner out_dir or workdir)")
    ap.add_argument("--out", default="rag/corpus")
    ap.add_argument("--max", type=int, default=120, help="max exemplars to keep")
    ap.add_argument("--exclude-spec", action="append", default=None,
                    help=f"experiment spec whose networks/goals are held out (repeatable; "
                         f"default: {DEFAULT_EXCLUDE_SPEC} if present)")
    ap.add_argument("--no-exclude", action="store_true", help="disable the hold-out filter")
    ap.add_argument("--allow-legacy", action="store_true",
                    help="accept journals without a skip record (parse-checked only; weaker)")
    ap.add_argument("--inspect", action="store_true", help="report only; write nothing")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"runs-dir {runs_dir} does not exist.", file=sys.stderr)
        return 1

    specs = [] if args.no_exclude else [Path(p) for p in (args.exclude_spec or
             ([DEFAULT_EXCLUDE_SPEC] if Path(DEFAULT_EXCLUDE_SPEC).exists() else []))]
    holdout = load_holdout(specs) if specs else {"goals": set(), "networks": set()}
    if not specs and not args.no_exclude:
        print("warning: no hold-out spec found; evaluated tasks are NOT filtered", file=sys.stderr)

    journals = []
    for jf in sorted(runs_dir.rglob("*.json")):
        if jf.name == "manifest.json":
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if _looks_like_journal(data):
            mpath = jf.parent / "manifest.json"
            manifest = None
            if mpath.exists():
                try:
                    manifest = json.loads(mpath.read_text())
                except Exception:
                    manifest = None
            journals.append((jf, data, manifest))
    if not journals:
        print(f"No journal JSON found under {runs_dir}/.", file=sys.stderr)
        return 1

    seen, exemplars, rejected = set(), [], {}
    for _jf, data, manifest in journals:
        for entry in data["entries"]:
            why = reject_reason(entry, args.allow_legacy) or held_out(entry, manifest, holdout)
            if why:
                key = "held out: " + why if why.startswith("evaluated") else why
                rejected[key] = rejected.get(key, 0) + 1
                continue
            key = (_norm(entry.get("description")), json.dumps(entry["commands"], sort_keys=True))
            if key in seen:
                rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                continue
            seen.add(key)
            exemplars.append(exemplar_text(entry, manifest))

    print(f"Scanned {len(journals)} journal(s); {len(exemplars)} certified exemplar(s).")
    for why, n in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  rejected {n:>4}  {why}")
    if args.inspect:
        for i, ex in enumerate(exemplars[: args.max], 1):
            print(f"\n--- exemplar {i} ---\n{ex}")
        return 0
    if not exemplars:
        print("No certified exemplars. Run the bootstrap spec with a capable model first.", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / OUT_NAME).write_text("\n\n".join(exemplars[: args.max]) + "\n", encoding="utf-8")
    print(f"Wrote {min(len(exemplars), args.max)} exemplar(s) -> {out / OUT_NAME}")
    print("Then: python rag/tools/corpus_guard.py --spec grader_ablation_spec.json rag/corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
