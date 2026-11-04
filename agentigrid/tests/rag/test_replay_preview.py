"""Replay preview: state reconstruction from journals, cached replay, gate summary."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agentigrid.rag.corrective import CorrectiveRetriever
from agentigrid.rag.query import build_query
from agentigrid.rag.retriever import Retriever

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("replay_preview", ROOT / "rag" / "tools" / "replay_preview.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

GOAL = "Eliminate all bus voltage violations"


def _entry(it, **kw):
    e = dict(iteration=it, convergence_status="CONVERGED", feasibility_detail="feasible", violations_count=6,
             voltage_min=1.015, voltage_max=1.06, max_line_loading_pct=70.9, commands=[], skipped_commands=None)
    e.update(kw)
    return e


JOURNAL = {
    "entries": [
        _entry(0),
        _entry(1, violations_count=149, skipped_commands=[]),
        _entry(2, violations_count=149,
               skipped_commands=["Skipped SetGenDispatch(bus=189, Pg=350.0, gen_id=None): Bus 189 is the reference/slack bus"]),
        # iteration 3 discarded (unknown action); iteration 4 completes
        _entry(4, convergence_status="COMPLETE", voltage_max=0.0, violations_count=0),
    ],
    "discarded_actions": [{"iteration": 3, "kind": "rejected",
                           "feedback": "Unknown action 'set_gen_status'. Valid actions: modify, complete, analyze."}],
}


def test_iteration_states_rebuilds_state_and_errors():
    st = rp.iteration_states(JOURNAL)
    assert [s["iteration"] for s in st] == [1, 2, 3, 4]
    assert st[0]["state"].num_violations == 6 and st[0]["error_feedback"] is None      # base case
    assert st[1]["state"].num_violations == 149 and st[1]["error_feedback"] is None
    assert "slack bus" in st[2]["error_feedback"] and st[2]["error_feedback"].startswith("Command errors:\n")
    assert "Unknown action 'set_gen_status'" in st[3]["error_feedback"]               # from the discarded action
    assert st[3]["state"].num_violations == 149                                        # last real solve (iter 2)
    assert not any(s["has_violation_details"] for s in st)


def test_replayed_query_matches_the_agent_loop_builder():
    st = rp.iteration_states(JOURNAL)[2]
    q = build_query("error", GOAL, st["state"], st["error_feedback"])
    assert q.startswith("Command errors: Skipped SetGenDispatch(bus=189")
    assert "149 violations" in q and GOAL not in q


class _Store:
    """Keyword store: voltage queries get chunk A,B; error queries get chunk C."""
    def query(self, q, k):
        if "slack" in q or "Unknown action" in q:
            return [("C slack bus note", {"source": "c.md"}, 0.71)]
        return [("A voltage example", {"source": "a.md"}, 0.75), ("B voltage example", {"source": "b.md"}, 0.74)][:k]


def _base():
    r = Retriever(enabled=False)
    r.enabled, r._store = True, _Store()
    return r


def _strict_grader(q, hits):          # a "reranker" that dislikes voltage chunks
    return [0.1 if d.startswith(("A", "B")) else 0.8 for d, _m, _s in hits]


def test_replayer_runs_caches_and_builds_label_pool():
    base = _base()
    reps = rp.Replayer(base, {"cosine": CorrectiveRetriever(base, refine="whole"),
                              "reranker": CorrectiveRetriever(base, grader=_strict_grader, refine="whole")})
    r1 = reps.run("voltage violations")
    assert r1["cosine_verdict"] == "correct" and r1["reranker_verdict"] == "incorrect"
    assert r1["reranker_action"] == "withheld" and r1["reranker_injected"] == ""
    assert r1["basic_chunks"].count("|") == 1
    assert reps.run("voltage violations") is r1                       # cached
    r2 = reps.run("Command errors: slack bus")
    assert r2["reranker_verdict"] == "correct" and r2["basic_chunks"] != r1["basic_chunks"]
    pool = list(reps.pairs.values())
    assert len(pool) == 3 and all("cosine" in p for p in pool)
    assert any(p.get("grade_reranker") == 0.1 for p in pool)
    assert len(reps.chunks) == 3


def test_replayer_refuses_grader_fallback():
    def broken(q, hits):
        raise RuntimeError("model missing")
    base = _base()
    reps = rp.Replayer(base, {"reranker": CorrectiveRetriever(base, grader=broken, refine="whole")})
    with pytest.raises(RuntimeError, match="fell back"):
        reps.run("voltage violations")


def test_summary_gate_metrics():
    base = _base()
    reps = rp.Replayer(base, {"cosine": CorrectiveRetriever(base, refine="whole"),
                              "reranker": CorrectiveRetriever(base, grader=_strict_grader, refine="whole")})
    rows = []
    for it, q in ((1, "voltage violations"), (2, "voltage violations"), (3, "Command errors: slack bus")):
        rows.append({"run_id": "r1", "goal": "voltage", "iteration": it, "mode": "error", **reps.run(q)})
    s = rp.summarize(rows, ["cosine", "reranker"])["by_mode"]["error"]
    assert s["n_states"] == 3 and s["unique_queries"] == 2
    assert s["chunkset_changed_between_iterations"] == 0.5             # 1 of 2 steps changed
    assert s["cosine"]["non_correct_share"] == 0.0
    assert s["reranker"]["non_correct_share"] == round(2 / 3, 4)
    assert s["verdict_differs_cosine_vs_reranker"] == round(2 / 3, 4)


def test_journal_entries_now_record_violation_details():
    from types import SimpleNamespace
    from agentigrid.engine.journal import SearchJournal
    j = SearchJournal()
    res = SimpleNamespace(feasibility_detail="infeasible", num_violations=2, converged=True, losses_mw=1.0,
                          total_load_mw=100.0, objective_value=1.0, convergence_status="CONVERGED",
                          voltage_min=0.97, voltage_max=1.06, max_line_loading_pct=70.0, total_gen_mw=101.0,
                          solver="IPOPT", violation_details=[f"Bus {i}: Vm=1.06 pu > 1.05 (overvoltage)" for i in range(25)])
    e = j.add_from_results(iteration=1, description="d", commands=[], opflow_result=res, sim_elapsed=0.1,
                           llm_reasoning="", mode="modify")
    assert len(e.violation_details) == 20 and e.violation_details[0].startswith("Bus 0")


def test_violation_details_survive_csv_export_and_session_reload(tmp_path):
    import csv as _csv, json as _json
    from types import SimpleNamespace
    from agentigrid.engine.journal import SearchJournal
    j = SearchJournal()
    res = SimpleNamespace(feasibility_detail="infeasible", num_violations=1, converged=True, losses_mw=1.0,
                          total_load_mw=100.0, objective_value=1.0, convergence_status="CONVERGED",
                          voltage_min=0.97, voltage_max=1.06, max_line_loading_pct=70.0, total_gen_mw=101.0,
                          solver="IPOPT", violation_details=["Bus 1: Vm=1.06 pu > 1.05 (overvoltage)"])
    j.add_from_results(iteration=1, description="d", commands=[], opflow_result=res, sim_elapsed=0.1,
                       llm_reasoning="", mode="modify")
    p = tmp_path / "j.csv"
    j.export_csv(p)
    row = next(_csv.DictReader(open(p)))
    assert _json.loads(row["violation_details"]) == ["Bus 1: Vm=1.06 pu > 1.05 (overvoltage)"]
