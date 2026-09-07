#!/usr/bin/env python3
"""Exemplar harvester — turns SUCCESSFUL AgentiGrid runs into RAG corpus.

Self-bootstrapping grounding: your good runs (e.g. under claude-sonnet) already
contain valid goal -> proposal -> result trajectories in their exported journal
JSON. This scrapes the converged, feasible iterations that carried a real
proposal and writes them as provenance-tagged exemplar chunks. Retrieving these
few-shot examples is the highest-leverage fix for the "flat iterations" failure,
because that is a format/grounding gap, not a knowledge gap.

Only entries that represent a real, successful ExaGO solve are kept:
  * feasible is True
  * objective_value is not None
  * convergence_status is a SOLVE status (not ANALYSIS/COMPLETE/EXPLORE/SWEEP/
    CONTINGENCY, mirroring the journal's own NON_SOLVE_STATUSES)
  * a proposal was actually made (commands non-empty OR exago_command present)
  * description is real (not empty / "No description")

Run from the project root:
    python rag_tools/rag_scrape_journal.py --runs-dir workdir
    rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus

Review the generated file before trusting it — a run being "successful" does
not guarantee every proposal in it is a good teaching example.
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

OUT_NAME = "exago_exemplars_from_runs.txt"


def _looks_like_journal(data: object) -> bool:
    if not isinstance(data, dict) or "entries" not in data:
        return False
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        return False
    first = entries[0]
    return isinstance(first, dict) and "iteration" in first and "convergence_status" in first


def _app_of(entry: dict) -> str:
    """Best-effort app name from the reproducible invocation record."""
    ec = entry.get("exago_command")
    if isinstance(ec, dict):
        for key in ("app", "application", "binary", "cmd"):
            v = ec.get(key)
            if isinstance(v, str) and v:
                # take the basename token if it's a path/command
                return re.split(r"[\\/ ]", v.strip())[-1]
    return "unknown"


def _compact_commands(entry: dict) -> str:
    """A compact, single-line representation of what was actually run."""
    ec = entry.get("exago_command")
    if isinstance(ec, dict) and ec:
        return json.dumps(ec, separators=(",", ":"))
    cmds = entry.get("commands")
    if isinstance(cmds, list) and cmds:
        return json.dumps(cmds, separators=(",", ":"))
    return ""


def _is_good_exemplar(entry: dict) -> bool:
    if not entry.get("feasible"):
        return False
    if entry.get("objective_value") is None:
        return False
    if entry.get("convergence_status") in NON_SOLVE_STATUSES:
        return False
    desc = (entry.get("description") or "").strip()
    if desc.lower() in _BAD_DESC:
        return False
    has_proposal = bool(entry.get("commands")) or bool(entry.get("exago_command"))
    return has_proposal


def _exemplar_text(entry: dict) -> str:
    app = _app_of(entry)
    desc = (entry.get("description") or "").strip()
    reasoning = (entry.get("llm_reasoning") or "").strip()
    reasoning = re.sub(r"\s+", " ", reasoning)
    if len(reasoning) > 400:
        reasoning = reasoning[:400].rstrip() + "…"
    cmds = _compact_commands(entry)
    obj = entry.get("objective_value")
    viol = entry.get("violations_count", 0)
    status = entry.get("convergence_status", "")

    tag = (f"[source: successful AgentiGrid run | app: {app} | status: {status} | "
           f"harvested: {_TODAY} | auto-generated exemplar, review before trusting]")
    lines = [tag, f"Goal: {desc}"]
    if reasoning:
        lines.append(f"Reasoning: {reasoning}")
    if cmds:
        lines.append(f"Proposal (commands actually run): {cmds}")
    lines.append(
        f"Result: converged & feasible, objective_value={obj:,.2f}, "
        f"{viol} violation(s)."
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="workdir",
                    help="Directory to scan recursively for exported journal *.json (default: workdir)")
    ap.add_argument("--out", default="rag/corpus", help="Corpus output directory (default: rag/corpus)")
    ap.add_argument("--max", type=int, default=40, help="Max exemplars to keep (default: 40)")
    ap.add_argument("--inspect", action="store_true",
                    help="Only report what would be harvested; write nothing")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"runs-dir {runs_dir} does not exist. Point --runs-dir at where "
              f"journal JSONs are exported (often workdir/ or a runs/ folder).",
              file=sys.stderr)
        return 1

    journals = []
    for jf in sorted(runs_dir.rglob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if _looks_like_journal(data):
            journals.append((jf, data))

    if not journals:
        print(f"No journal JSON files found under {runs_dir}/. "
              f"Run a search first, or check the path.", file=sys.stderr)
        return 1

    seen = set()
    exemplars = []
    for jf, data in journals:
        for entry in data["entries"]:
            if not isinstance(entry, dict) or not _is_good_exemplar(entry):
                continue
            key = ((entry.get("description") or "").strip().lower(), _compact_commands(entry))
            if key in seen:
                continue
            seen.add(key)
            exemplars.append(_exemplar_text(entry))

    print(f"Scanned {len(journals)} journal file(s); found {len(exemplars)} unique good exemplar(s).")
    if args.inspect:
        for i, ex in enumerate(exemplars[: args.max], 1):
            print(f"\n--- exemplar {i} ---\n{ex}")
        return 0
    if not exemplars:
        print("No qualifying exemplars. Need converged+feasible iterations that "
              "carried a real proposal (use a capable model like claude-sonnet-4-6).",
              file=sys.stderr)
        return 1

    exemplars = exemplars[: args.max]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME
    out_path.write_text("\n\n".join(exemplars) + "\n", encoding="utf-8")
    print(f"Wrote {len(exemplars)} exemplar(s) -> {out_path}")
    print("Re-ingest with:  rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
