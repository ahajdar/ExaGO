"""Experiment runner: goal allowlists, per-case args, and the guard that refuses
conditions whose RAG mode / grader is not implemented (which AgentiGrid would
otherwise silently degrade to basic / cosine, mislabeling the data)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = ROOT / "rag" / "tools" / "experiment_runner.py"
    spec = importlib.util.spec_from_file_location("experiment_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()

SPEC = {
    "goals": [{"id": "cost10", "text": "c"}, {"id": "loadmax", "text": "l"}, {"id": "n1", "text": "n"}],
    "max_iter": 4,
}


def test_goals_for_case_allowlist_and_default():
    assert [g["id"] for g in runner.goals_for_case(SPEC, {"name": "a", "goals": ["n1"]})] == ["n1"]
    assert [g["id"] for g in runner.goals_for_case(SPEC, {"name": "b"})] == ["cost10", "loadmax", "n1"]


def test_goals_for_case_rejects_unknown_goal():
    with pytest.raises(SystemExit):
        runner.goals_for_case(SPEC, {"name": "a", "goals": ["nope"]})


def test_case_extra_args_precede_model_args():
    case = {"name": "x", "path": "net.m", "app": "scopflow", "extra_args": ["--ctgc", "f.cont"]}
    model = {"backend": "ollama", "model": "m", "extra_args": ["--quiet"]}
    cmd = runner.build_cmd(SPEC, case, {"id": "n1", "text": "n"}, model)
    assert cmd[-3:] == ["--ctgc", "f.cont", "--quiet"]
    assert "--app" in cmd and cmd[cmd.index("--app") + 1] == "scopflow"


@pytest.mark.parametrize("env, ok", [
    ({"AGENTIGRID_RAG_MODE": "off"}, True),
    ({"AGENTIGRID_RAG_MODE": "basic"}, True),
    ({"AGENTIGRID_RAG_MODE": "corrective", "AGENTIGRID_CRAG_GRADER": "cosine"}, True),
    ({"AGENTIGRID_RAG_MODE": "graph"}, False),
    ({"AGENTIGRID_RAG_MODE": "corrective", "AGENTIGRID_CRAG_GRADER": "reranker_calibrated"}, False),
    ({"AGENTIGRID_RAG": "1"}, True),  # legacy switch, no mode/grader keys
])
def test_unimplemented_reason(env, ok):
    reason = runner.unimplemented_reason({"id": "c", "env": env})
    assert (reason is None) == ok, reason


def test_jev_blocked_when_unavailable(monkeypatch):
    import agentigrid.rag.grader_jev as gj
    monkeypatch.setattr(gj, "jev_available", lambda: False)
    reason = runner.unimplemented_reason(
        {"id": "c", "env": {"AGENTIGRID_RAG_MODE": "corrective", "AGENTIGRID_CRAG_GRADER": "jev"}})
    assert reason and "silently run cosine" in reason


def _run_main(monkeypatch, tmp_path, argv):
    monkeypatch.setattr("sys.argv", ["experiment_runner.py", *argv])
    monkeypatch.chdir(tmp_path)
    return runner.main()


def _write_spec(tmp_path, conditions):
    spec = {
        "project_root": str(tmp_path), "workdir": "workdir", "out_dir": "out", "reps": 2,
        "cases": [
            {"name": "c39", "path": "a.m", "app": "opflow", "goals": ["cost10", "loadmax"]},
            {"name": "g200-scopf", "path": "b.m", "app": "scopflow",
             "extra_args": ["--ctgc", "b.cont"], "goals": ["n1"]},
        ],
        "goals": SPEC["goals"],
        "conditions": conditions,
        "models": [{"backend": "ollama", "model": "m", "extra_args": []}],
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    return p


def test_main_refuses_unimplemented_conditions(monkeypatch, tmp_path, capsys):
    p = _write_spec(tmp_path, [{"id": "C0", "env": {"AGENTIGRID_RAG_MODE": "off"}},
                               {"id": "C3", "env": {"AGENTIGRID_RAG_MODE": "graph"}}])
    rc = _run_main(monkeypatch, tmp_path, ["--spec", str(p)])
    assert rc == 2
    assert "Refusing to start" in capsys.readouterr().out
    assert not (tmp_path / "out" / "runs_index.jsonl").exists()


def test_dry_run_skips_unimplemented_and_counts_allowlisted_goals(monkeypatch, tmp_path, capsys):
    p = _write_spec(tmp_path, [{"id": "C0", "env": {"AGENTIGRID_RAG_MODE": "off"}},
                               {"id": "C3", "env": {"AGENTIGRID_RAG_MODE": "graph"}}])
    rc = _run_main(monkeypatch, tmp_path, ["--spec", str(p), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    # (2 goals on c39 + 1 goal on g200) x 1 implemented condition x 1 model x 2 reps
    assert "6 runs" in out
    assert "C3" in out and "not implemented" in out
    assert "--ctgc b.cont" in out


def test_repo_spec_is_consistent():
    spec = json.loads((ROOT / "grader_ablation_spec.json").read_text())
    ids = {g["id"] for g in spec["goals"]}
    for case in spec["cases"]:
        assert set(case.get("goals", ids)) <= ids
    assert spec["reps"] == 20
    assert [c["id"].split("-")[0] for c in spec["conditions"]] == ["C0", "C1", "C2a"]  # C2b/C2c not run (retrieval preview); graph RAG out of scope
    scopf = [c for c in spec["cases"] if c["app"] == "scopflow"]
    assert scopf and all("--ctgc" in c["extra_args"] for c in scopf)


def test_reranker_gated_on_library(monkeypatch):
    import agentigrid.rag.grader_reranker as gr
    cond = {"id": "c", "env": {"AGENTIGRID_RAG_MODE": "corrective", "AGENTIGRID_CRAG_GRADER": "reranker"}}
    monkeypatch.setattr(gr, "reranker_available", lambda: False)
    assert "sentence-transformers" in runner.unimplemented_reason(cond)
    monkeypatch.setattr(gr, "reranker_available", lambda: True)
    assert runner.unimplemented_reason(cond) is None


def test_pending_case_refused_unless_skipped(monkeypatch, tmp_path, capsys):
    p = _write_spec(tmp_path, [{"id": "C0", "env": {"AGENTIGRID_RAG_MODE": "off"}}])
    spec = json.loads(p.read_text())
    spec["cases"][1]["pending"] = "voltage band not checked"
    p.write_text(json.dumps(spec))
    assert _run_main(monkeypatch, tmp_path, ["--spec", str(p)]) == 2
    assert "pending" in capsys.readouterr().out
    assert _run_main(monkeypatch, tmp_path, ["--spec", str(p), "--dry-run"]) == 0
    assert "4 runs" in capsys.readouterr().out   # only c39's 2 goals x 2 reps


def test_frozen_spec_cases_exist_and_keep_genfuel():
    """Frozen variants must be present, scaled as named, and readable by ExaGO
    (genfuel closed by '};' -- see test_matpower_cell_terminators)."""
    from agentigrid.parsers.matpower_parser import parse_matpower
    spec = json.loads((ROOT / "grader_ablation_spec.json").read_text())
    base = ROOT.parent / "datafiles" / "case_ACTIVSg200.m"
    if not base.exists():
        pytest.skip("ExaGO datafiles not present")
    pd0 = sum(b.Pd for b in parse_matpower(base).buses)
    frozen = [c for c in spec["cases"] if "specs/cases/" in c["path"]]
    assert frozen
    for c in frozen:
        path = (ROOT / c["path"]).resolve()
        assert path.exists(), path
        scale = float(path.stem.split("_load")[1].split("_")[0])
        assert abs(sum(b.Pd for b in parse_matpower(path).buses) - scale * pd0) < 1e-6 * pd0
        text = path.read_text()
        assert "mpc.genfuel" in text and "\n};" in text.split("mpc.genfuel", 1)[1]
        net = parse_matpower(path)
        if "_out" in path.stem:                  # forced-outage variant: branch out, guard declared
            f, t = (int(x) for x in path.stem.split("_out")[1].split("-")[:2])
            assert [b.status for b in net.branches if {b.fbus, b.tbus} == {f, t}] == [0]
            assert {"branch_stays_out": [f, t, 0]} in c.get("guards", [])
    assert not [c for c in spec["cases"] if c.get("pending")], "all case entries resolved"


def test_retrieval_conditions_need_frozen_corpus(monkeypatch, tmp_path, capsys):
    p = _write_spec(tmp_path, [{"id": "C0", "env": {"AGENTIGRID_RAG_MODE": "off"}},
                               {"id": "C1", "env": {"AGENTIGRID_RAG_MODE": "basic"}}])
    assert _run_main(monkeypatch, tmp_path, ["--spec", str(p)]) == 2
    assert "not frozen" in capsys.readouterr().out
    assert runner.corpus_check({"id": "C0", "env": {"AGENTIGRID_RAG_MODE": "off"}}, tmp_path) is None


def test_placeholder_model_refused(monkeypatch, tmp_path, capsys):
    p = _write_spec(tmp_path, [{"id": "C0", "env": {"AGENTIGRID_RAG_MODE": "off"}}])
    spec = json.loads(p.read_text())
    spec["models"] = [{"backend": "anthropic", "model": "<opus-model-id>", "extra_args": []}]
    p.write_text(json.dumps(spec))
    assert _run_main(monkeypatch, tmp_path, ["--spec", str(p)]) == 2
    assert "placeholder" in capsys.readouterr().out


def test_store_env_selects_corpus_store(monkeypatch):
    import agentigrid.rag as rag
    monkeypatch.setenv("AGENTIGRID_RAG_MODE", "basic")
    monkeypatch.setenv("AGENTIGRID_RAG_STORE", "rag/store_docs")
    monkeypatch.setenv("AGENTIGRID_RAG_COLLECTION", "docs_kb")
    r = rag.build_retriever(host="http://localhost:1")
    d = r.describe()
    assert d["store_path"] == "rag/store_docs" and d["collection"] == "docs_kb"


def test_llm_failure_detects_runs_with_no_successful_llm_call(tmp_path):
    j = tmp_path / "journal.json"
    j.write_text(json.dumps({"entries": [{"iteration": 0}], "llm_usage": {"calls": 4, "prompt_tokens": 0,
                             "completion_tokens": 0}, "discarded_actions": [{"iteration": 1, "kind": "rejected"}]}))
    assert "failed" in runner.llm_failure(j)
    j.write_text(json.dumps({"entries": [{"iteration": 0}, {"iteration": 1}],
                             "llm_usage": {"calls": 3, "prompt_tokens": 900, "completion_tokens": 50}}))
    assert runner.llm_failure(j) is None
    assert runner.llm_failure(None) is None


def test_case_voltage_band_guard_resolves_from_case_file():
    g = runner.resolve_guards(["load_preserved", {"vband_not_widened": "case"}],
                              ROOT.parent / "datafiles" / "case118.m", ROOT)
    assert g == ["load_preserved", {"vband_not_widened": [0.94, 1.06]}]
    assert runner.resolve_guards(None, "x.m", ROOT) is None
