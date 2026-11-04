#!/usr/bin/env python3
"""Replay preview: what each retrieval query mode and grader would inject at every
state the agent actually reached in recorded runs -- no LLM, no solver.

Paper 1's preview could be computed per goal because the query was the goal text.
With state-dependent queries (AGENTIGRID_RAG_QUERY = state | error) the query
depends on the state of the loop, so this tool rebuilds, from each run's journal,
the state the agent saw before every LLM iteration (the latest solve and the
previous iteration's command errors), builds the query exactly as the agent loop
would (agentigrid.rag.query.build_query), and runs it through basic retrieval and
through corrective retrieval with each grader against the frozen store.

It answers, before any new agent run (gate 1 of the CRAG experiment):
  * does the retrieved chunk set change between states (does the query steer)?
  * how often is each grader's verdict not CORRECT (does correction ever act)?
  * how often do the graders disagree, and inject different text?

Fidelity notes (reported in summary.json):
  * Journals from before the violation_details field (e.g. ofrecord_v3) do not
    store the solver's violation lines, so replayed state/error queries omit the
    "Violations:" line that live runs include; status, voltage range, line
    loading, violation count and command errors are reproduced.
  * Command errors are rebuilt from the previous iteration's skipped commands and
    discarded-action feedback; other feedback the live loop may send (e.g. after
    an analysis action) is not reproduced.
  * Corrective retrieval runs with refine='whole' and the thresholds given on the
    command line (defaults are the untuned 0.30 / 0.50).

Outputs (in --out):
  states.csv     one row per (run, iteration, query mode)
  queries.jsonl  each distinct query text once (sha -> text)
  chunks.jsonl   each distinct retrieved chunk once (sha -> source, text)
  pairs.csv      distinct (query, chunk) pairs with cosine score and grades --
                 the sampling pool for human relevance labels
  summary.json   gate metrics per goal x mode and overall

Run from the agentigrid project root with the frozen store ingested and
OLLAMA_HOST set (the reranker also needs sentence-transformers):
    python rag/tools/replay_preview.py --runs experiments/ofrecord_v3 --out experiments/replay_preview_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentigrid.rag.query import VALID_QUERY_MODES, build_query  # noqa: E402

NON_STATE = {"COMPLETE", "ANALYSIS", "EXPLORE", "SWEEP", "CONTINGENCY", "FAILED"}
VERDICTS = ("correct", "ambiguous", "incorrect")


def sha(text: str, n: int = 16) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:n]


# -- state reconstruction ------------------------------------------------------
def _state_of(entry: dict) -> SimpleNamespace:
    return SimpleNamespace(
        convergence_status=entry.get("convergence_status") or "",
        feasibility_detail=entry.get("feasibility_detail") or "",
        num_violations=entry.get("violations_count"),
        voltage_min=entry.get("voltage_min"),
        voltage_max=entry.get("voltage_max"),
        max_line_loading_pct=entry.get("max_line_loading_pct"),
        violation_details=entry.get("violation_details") or [],
    )


def _is_state_entry(e: dict) -> bool:
    return (isinstance(e, dict) and str(e.get("convergence_status", "")).upper() not in NON_STATE
            and (e.get("voltage_max") or 0) > 0)


def iteration_states(journal: dict) -> list[dict]:
    """The state before each LLM iteration that happened in a run.

    Returns [{iteration, state (SimpleNamespace|None), error_feedback (str|None),
    has_violation_details (bool)}], in iteration order. An LLM iteration happened
    if it recorded a journal entry or a discarded action.
    """
    entries = [e for e in journal.get("entries", []) if isinstance(e, dict)]
    disc = [d for d in (journal.get("discarded_actions") or []) if isinstance(d, dict)]
    its = sorted({e.get("iteration") for e in entries if (e.get("iteration") or 0) > 0}
                 | {d.get("iteration") for d in disc if (d.get("iteration") or 0) > 0})
    out = []
    for i in its:
        prior = [e for e in entries if (e.get("iteration") or 0) < i and _is_state_entry(e)]
        last = max(prior, key=lambda e: e.get("iteration") or 0) if prior else None
        errs: list[str] = []
        for e in entries:
            if e.get("iteration") == i - 1 and e.get("skipped_commands"):
                errs.extend(str(s) for s in e["skipped_commands"])
        for d in disc:
            if d.get("iteration") == i - 1 and d.get("feedback"):
                errs.append(str(d["feedback"]))
        out.append({
            "iteration": i,
            "state": _state_of(last) if last else None,
            "error_feedback": ("Command errors:\n" + "\n".join(errs)) if errs else None,
            "has_violation_details": bool(last and last.get("violation_details")),
        })
    return out


# -- retrieval -----------------------------------------------------------------
def _refset(trace: dict, crag) -> str:
    """The chunks a corrective retrieval actually injected (whole refinement),
    plus a hedge marker for AMBIGUOUS. Compared across graders instead of the
    injected text, which embeds each grader's own scores and so always differs."""
    src = trace.get("hits", [])
    if trace.get("rewrite") and trace.get("rewrite_verdict") not in ("", None, "incorrect"):
        src = trace.get("rewrite_hits", [])
    kept = [h.get("sha", "") for h in src
            if h.get("grade") is not None and h["grade"] >= crag.tau_lower][: crag.max_refine_docs]
    hedge = ["hedge"] if trace.get("verdict") == "ambiguous" or trace.get("rewrite_verdict") == "ambiguous" else []
    return "|".join(kept + hedge)


class Replayer:
    """Runs one query through basic retrieval and corrective retrieval per grader,
    caching by query text (identical queries give identical results)."""

    def __init__(self, base, correctives: dict):
        self.base = base
        self.correctives = correctives          # name -> CorrectiveRetriever
        self.cache: dict[str, dict] = {}
        self.chunks: dict[str, dict] = {}       # chunk sha -> {source, text}
        self.pairs: dict[tuple, dict] = {}      # (query sha, chunk sha) -> scores

    def _note_hits(self, qsha: str, hits, grades=None, grader=None):
        for i, (doc, meta, score) in enumerate(hits or []):
            csha = sha(doc)
            self.chunks.setdefault(csha, {"source": (meta or {}).get("source"), "text": doc})
            rec = self.pairs.setdefault((qsha, csha), {"query_sha": qsha, "chunk_sha": csha,
                                                       "source": (meta or {}).get("source"),
                                                       "cosine": round(float(score), 4)})
            if grader and grades is not None and i < len(grades):
                rec[f"grade_{grader}"] = round(float(grades[i]), 4)

    def run(self, query: str) -> dict:
        if query in self.cache:
            return self.cache[query]
        qsha = sha(query)
        res: dict = {"query_sha": qsha}
        injected = self.base.retrieve(query)
        res["basic_action"] = self.base.last_trace.get("action")
        hits = self.base.query_hits(query, self.base.k)
        self._note_hits(qsha, hits)
        res["basic_chunks"] = "|".join(sha(d) for d, _m, _s in hits)
        res["basic_injected"] = sha(injected) if injected else ""
        for name, crag in self.correctives.items():
            before = crag.stats.get("grader_fallbacks", 0)
            out = crag.retrieve(query)
            if crag.stats.get("grader_fallbacks", 0) > before:
                raise RuntimeError(f"grader {name!r} fell back to cosine; refusing to label it {name}")
            t = crag.last_trace
            grades = [h.get("grade") for h in t.get("hits", [])]
            if hits and grades and len(grades) == len(hits):
                self._note_hits(qsha, hits, grades, name)
            res[f"{name}_verdict"] = t.get("verdict", "")
            res[f"{name}_action"] = t.get("action", "")
            res[f"{name}_rewrite_verdict"] = t.get("rewrite_verdict", "")
            res[f"{name}_top_grade"] = max((g for g in grades if g is not None), default="")
            res[f"{name}_injected"] = sha(out) if out else ""
            res[f"{name}_refset"] = _refset(t, crag) if out else ""
            res[f"{name}_refs"] = t.get("injected_refs", 0)
        self.cache[query] = res
        return res


# -- summary -------------------------------------------------------------------
def _quantiles(vals: list[float]) -> dict:
    v = sorted(x for x in vals if isinstance(x, (int, float)))
    if not v:
        return {}
    pick = lambda q: round(v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))], 4)
    return {"p10": pick(0.10), "p50": pick(0.50), "p90": pick(0.90), "min": v[0], "max": v[-1]}


def summarize(rows: list[dict], graders: list[str]) -> dict:
    def block(rs: list[dict]) -> dict:
        n = len(rs)
        if not n:
            return {"n_states": 0}
        b = {"n_states": n,
             "unique_queries": len({r["query_sha"] for r in rs}),
             "unique_basic_chunksets": len({r["basic_chunks"] for r in rs}),
             "distinct_chunks_retrieved": len({c for r in rs for c in r["basic_chunks"].split("|") if c})}
        # steering: consecutive LLM iterations of the same run whose basic chunk set changed
        by_run = defaultdict(list)
        for r in rs:
            by_run[r["run_id"]].append(r)
        steps = changed = 0
        for lst in by_run.values():
            lst.sort(key=lambda r: r["iteration"])
            for a, c in zip(lst, lst[1:]):
                steps += 1
                changed += a["basic_chunks"] != c["basic_chunks"]
        b["chunkset_changed_between_iterations"] = round(changed / steps, 4) if steps else None
        for g in graders:
            cnt = Counter(r.get(f"{g}_verdict", "") for r in rs)
            b[g] = {"verdict_share": {v: round(cnt.get(v, 0) / n, 4) for v in VERDICTS},
                    "non_correct_share": round(1 - cnt.get("correct", 0) / n, 4),
                    "withheld_share": round(sum(r.get(f"{g}_action") == "withheld" for r in rs) / n, 4),
                    "top_grade": _quantiles([r.get(f"{g}_top_grade") for r in rs])}
        if len(graders) >= 2:
            g1, g2 = graders[0], graders[1]
            b[f"verdict_differs_{g1}_vs_{g2}"] = round(
                sum(r.get(f"{g1}_verdict") != r.get(f"{g2}_verdict") for r in rs) / n, 4)
            # same chunks + same hedge = same injected content (the text itself always
            # differs because each block shows its own grader's scores)
            b[f"injected_differs_{g1}_vs_{g2}"] = round(
                sum(r.get(f"{g1}_refset") != r.get(f"{g2}_refset") for r in rs) / n, 4)
        return b

    out: dict = {"by_mode": {}, "by_goal_mode": {}}
    for m in sorted({r["mode"] for r in rows}):
        out["by_mode"][m] = block([r for r in rows if r["mode"] == m])
        for g in sorted({r["goal"] for r in rows}):
            out["by_goal_mode"][f"{g} | {m}"] = block([r for r in rows if r["mode"] == m and r["goal"] == g])
    return out


# -- main ----------------------------------------------------------------------
def _parse_tau(s: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in s.split(","))
    if lo > hi:
        raise argparse.ArgumentTypeError("lower threshold exceeds upper")
    return lo, hi


def collect(runs_dir: Path, models=None, conditions=None, limit=None) -> list[tuple]:
    out = []
    for mf in sorted(runs_dir.glob("*/manifest.json")):
        m = json.loads(mf.read_text())
        if m.get("status") != "ok":
            continue
        if models and not any(x in str(m.get("model", "")) for x in models):
            continue
        if conditions and not any(str(m.get("condition", "")).startswith(c) for c in conditions):
            continue
        jp = mf.parent / "journal.json"
        if not jp.exists():
            continue
        out.append((mf.parent.name, m, json.loads(jp.read_text())))
        if limit and len(out) >= limit:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="runner out_dir with <run_id>/manifest.json + journal.json")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--modes", default="goal,state,error")
    ap.add_argument("--graders", default="cosine,reranker")
    ap.add_argument("--tau-cosine", type=_parse_tau, default=(0.30, 0.50))
    ap.add_argument("--tau-reranker", type=_parse_tau, default=(0.30, 0.50))
    ap.add_argument("--store", default=os.environ.get("AGENTIGRID_RAG_STORE", "rag/store"))
    ap.add_argument("--collection", default=os.environ.get("AGENTIGRID_RAG_COLLECTION", "agentigrid_kb"))
    ap.add_argument("--models", help="comma list of substrings to keep (default: all)")
    ap.add_argument("--conditions", help="comma list of condition prefixes to keep, e.g. C0,C1 (default: all)")
    ap.add_argument("--limit", type=int, help="first N runs only (smoke test)")
    a = ap.parse_args(argv)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in VALID_QUERY_MODES]
    if bad:
        ap.error(f"unknown query mode(s) {bad}; valid: {VALID_QUERY_MODES}")
    graders = [g.strip() for g in a.graders.split(",") if g.strip()]
    if any(g not in ("cosine", "reranker") for g in graders):
        ap.error("graders: cosine and/or reranker")

    from agentigrid.rag.corrective import CorrectiveRetriever
    from agentigrid.rag.retriever import Retriever
    host = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    base = Retriever(enabled=True, path=a.store, collection=a.collection, host=host)
    if not base.enabled:
        print(f"Cannot open the store at {a.store!r} (chromadb installed? store ingested?)", file=sys.stderr)
        return 2
    correctives = {}
    for g in graders:
        lo, hi = a.tau_cosine if g == "cosine" else a.tau_reranker
        grader = None
        if g == "reranker":
            from agentigrid.rag.grader_reranker import RerankerGrader, reranker_available
            if not reranker_available():
                print("The reranker needs sentence-transformers (pip install sentence-transformers).", file=sys.stderr)
                return 2
            grader = RerankerGrader()
        correctives[g] = CorrectiveRetriever(base, grader=grader, refine="whole", tau_lower=lo, tau_upper=hi)

    runs = collect(Path(a.runs), a.models and a.models.split(","), a.conditions and a.conditions.split(","), a.limit)
    if not runs:
        print("No ok runs with journal.json found.", file=sys.stderr)
        return 1

    rep = Replayer(base, correctives)
    rows, queries = [], {}
    n_states = n_detail = 0
    for k, (run_id, m, journal) in enumerate(runs, 1):
        goal = m.get("goal_text") or ""
        for st in iteration_states(journal):
            n_states += 1
            n_detail += st["has_violation_details"]
            for mode in modes:
                q = build_query(mode, goal, st["state"], st["error_feedback"])
                res = rep.run(q)
                queries.setdefault(res["query_sha"], {"query_sha": res["query_sha"], "mode": mode,
                                                      "goal": m.get("goal_id"), "query": q})
                rows.append({"run_id": run_id, "model": m.get("model"), "condition": m.get("condition"),
                             "case": m.get("case"), "goal": m.get("goal_id"), "rep": m.get("rep"),
                             "iteration": st["iteration"], "mode": mode,
                             "has_errors": int(bool(st["error_feedback"])), **res})
        if k % 25 == 0 or k == len(runs):
            print(f"  {k}/{len(runs)} runs, {len(rep.cache)} distinct queries", flush=True)
        if base.stats.get("errors"):
            print(f"Retrieval errors ({base.stats['errors']}); is OLLAMA_HOST reachable? Stopping.", file=sys.stderr)
            return 2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "states.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "queries.jsonl", "w") as fh:
        for q in queries.values():
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(out / "chunks.jsonl", "w") as fh:
        for csha, c in rep.chunks.items():
            fh.write(json.dumps({"chunk_sha": csha, **c}, ensure_ascii=False) + "\n")
    pair_rows = list(rep.pairs.values())
    pf = ["query_sha", "chunk_sha", "source", "cosine"] + [f"grade_{g}" for g in graders if g != "cosine"]
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pf, extrasaction="ignore")
        w.writeheader()
        w.writerows(pair_rows)

    summ = summarize(rows, graders)
    summ["meta"] = {
        "created": datetime.now().isoformat(timespec="seconds"), "runs_dir": str(a.runs),
        "n_runs": len(runs), "n_states": n_states, "modes": modes, "graders": graders,
        "tau": {g: (a.tau_cosine if g == "cosine" else a.tau_reranker) for g in graders},
        "refine": "whole", "store": a.store, "collection": a.collection,
        "distinct_queries": len(rep.cache), "distinct_chunks": len(rep.chunks), "pairs": len(pair_rows),
        "states_with_violation_details": n_detail,
        "fidelity_note": "states without violation_details: replayed state/error queries omit the Violations line",
        "reranker": (correctives["reranker"].describe().get("grader_detail") if "reranker" in correctives else None),
    }
    (out / "summary.json").write_text(json.dumps(summ, indent=2))

    print(f"\nReplayed {n_states} states from {len(runs)} runs; {len(rep.cache)} distinct queries, "
          f"{len(rep.chunks)} distinct chunks, {len(pair_rows)} query-chunk pairs.")
    if n_detail < n_states:
        print(f"Note: {n_states - n_detail} of {n_states} states have no journaled violation lines "
              "(older journals); their state/error queries omit the Violations line.")
    hdr = f"{'mode':6s} {'states':>6s} {'uniq q':>6s} {'chunks chg':>10s}"
    for g in graders:
        hdr += f" {g[:8]+' nonC':>13s} {g[:8]+' wh':>11s}"
    if len(graders) >= 2:
        hdr += f" {'verdict diff':>12s} {'inject diff':>11s}"
    print(hdr)
    for m, b in summ["by_mode"].items():
        line = f"{m:6s} {b['n_states']:6d} {b['unique_queries']:6d} {str(b['chunkset_changed_between_iterations']):>10s}"
        for g in graders:
            line += f" {b[g]['non_correct_share']:13.3f} {b[g]['withheld_share']:11.3f}"
        if len(graders) >= 2:
            line += (f" {b[f'verdict_differs_{graders[0]}_vs_{graders[1]}']:12.3f}"
                     f" {b[f'injected_differs_{graders[0]}_vs_{graders[1]}']:11.3f}")
        print(line)
    print(f"\nWrote {out}/states.csv, queries.jsonl, chunks.jsonl, pairs.csv, summary.json")
    print("Thresholds are the untuned defaults unless --tau-* was given; verdict shares change once they are fitted on labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
