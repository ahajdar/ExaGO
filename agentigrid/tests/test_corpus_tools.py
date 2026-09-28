"""Corpus tools: certified scraping, schema exemplars, hold-out leakage guard."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_SPEC = ROOT / "grader_ablation_spec.json"
BOOT_SPEC = ROOT / "rag" / "tools" / "specs" / "corpus_bootstrap_spec.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "rag" / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scrape = _load("rag_scrape_journal")
schema = _load("rag_schema_exemplars")
guard = _load("corpus_guard")


def _entry(it=1, cmds=None, skipped=None, feasible=True, viol=0, obj=100.0,
           desc="Tighten voltage band", net="/home/u/p/workdir/iter_001/case118.m", status="CONVERGED"):
    return {"iteration": it, "description": desc, "llm_reasoning": "because",
            "commands": [{"action": "set_all_bus_vlimits", "Vmin": 0.97, "Vmax": 1.03}] if cmds is None else cmds,
            "objective_value": obj, "feasible": feasible, "violations_count": viol,
            "convergence_status": status, "mode": "accumulative", "skipped_commands": skipped,
            "exago_command": {"mode": "single", "application": "opflow",
                              "argv": ["/home/u/p/applications/opflow", "-netfile", net, "-print_output"]}}


# --- scraper certification ---------------------------------------------------

@pytest.mark.parametrize("kwargs, reason", [
    (dict(it=0, skipped=[]), "baseline"),
    (dict(skipped=["Skipped x"]), "some commands were skipped"),
    (dict(skipped=None), "legacy journal (no skip record)"),
    (dict(cmds=[], skipped=[]), "no proposal"),
    (dict(feasible=False, skipped=[]), "infeasible or violations"),
    (dict(viol=2, skipped=[]), "infeasible or violations"),
    (dict(status="SWEEP", skipped=[]), "not a solve"),
    (dict(desc="Tighten voltage band — no commands applied", skipped=[]), "no real description"),
    (dict(cmds=[{"action": "set_all_bus_vlimitz"}], skipped=[]), "command does not parse"),
])
def test_reject_reasons(kwargs, reason):
    assert scrape.reject_reason(_entry(**kwargs)) == reason


def test_certified_entry_and_legacy_opt_in():
    assert scrape.reject_reason(_entry(skipped=[])) is None
    assert scrape.reject_reason(_entry(skipped=None), allow_legacy=True) is None


def test_exemplar_teaches_action_json_without_paths():
    text = scrape.exemplar_text(_entry(skipped=[]), {"model": "m", "goal_text": "Some held-out goal"})
    assert '"action": "modify"' in text and '"set_all_bus_vlimits"' in text
    assert "/home/" not in text and "network: case118" in text
    assert guard.audit_text(text, {"goals": set(), "networks": set()}) == []


def test_holdout_filters_eval_network_and_goal():
    hold = scrape.load_holdout([EVAL_SPEC])
    e200 = _entry(skipped=[], net="/x/iter_001/case_ACTIVSg200.m")
    assert scrape.held_out(e200, None, hold).startswith("evaluated network")
    goal = json.loads(EVAL_SPEC.read_text())["goals"][0]["text"]
    assert scrape.held_out(_entry(skipped=[]), {"goal_text": goal}, hold) == "evaluated goal"
    assert scrape.held_out(_entry(skipped=[]), {"goal_text": "other"}, hold) is None


def test_scraper_end_to_end(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    good = _entry(skipped=[])
    base = _entry(it=0, skipped=[])
    (run / "journal.json").write_text(json.dumps({"entries": [base, good, good]}))
    (run / "manifest.json").write_text(json.dumps({"model": "m", "goal_text": "g", "case_path": "x/case118.m"}))
    out = tmp_path / "corpus"
    rc = scrape.main(["--runs-dir", str(tmp_path / "runs"), "--out", str(out),
                      "--exclude-spec", str(EVAL_SPEC)])
    assert rc == 0
    text = (out / scrape.OUT_NAME).read_text()
    assert text.count("[source:") == 1          # baseline dropped, duplicate dropped
    assert "/home/" not in text


# --- schema exemplars -----------------------------------------------------------

def test_schema_exemplars_certify_on_case118():
    chunks = schema.build(ROOT.parent / "datafiles" / "case118.m")
    assert len(chunks) == len(schema.EXEMPLARS)
    applied = sum("applied on held-out case118" in c for c in chunks)
    assert applied >= 12
    # the one command without a host branch in any held-out case is labelled honestly
    assert any("set_phase_shift_angle" in c and "schema-validated only" in c for c in chunks)
    for ex in schema.EXEMPLARS:
        assert ex["response"]["action"] in schema.TOP_LEVEL_ACTIONS


def test_schema_rejects_top_level_command():
    bad = {"request": "x", "apps": ["opflow"],
           "response": {"action": "set_all_bus_vlimits", "Vmin": 0.9, "Vmax": 1.1}}
    with pytest.raises(ValueError):
        schema.certify(bad, None)


# --- leakage guard ---------------------------------------------------------------

def test_generated_corpus_does_not_leak_eval_tasks():
    hold = guard.load_holdout([EVAL_SPEC])
    text = "\n\n".join(schema.build(ROOT.parent / "datafiles" / "case118.m"))
    assert guard.audit_text(text, hold) == []


def test_bootstrap_spec_is_held_out():
    hold = guard.load_holdout([EVAL_SPEC])
    boot = json.loads(BOOT_SPEC.read_text())
    assert not {Path(c["path"]).stem.lower() for c in boot["cases"]} & hold["networks"]
    assert not {guard._norm(g["text"]) for g in boot["goals"]} & hold["goals"]
    assert [c["id"] for c in boot["conditions"]] == ["C0-norag"]


def test_guard_flags():
    hold = {"goals": {guard._norm("Reduce total generation cost by 10%")},
            "networks": {"case_activsg200", "case39"}}
    text = ("Request: Reduce total generation cost by 10%\n\n"
            "Correct specification: application scopflow on case_ACTIVSg200\n\n"
            "Case facts: case_ACTIVSg200 has 200 buses\n\n"
            "Proposal ran /home/ahajdar/x.m\n\n"
            "Correct response on case390 is fine")
    kinds = [f["kind"] for f in guard.audit_text(text, hold)]
    assert kinds.count("goal-leak") == 1
    assert kinds.count("eval-exemplar") == 1        # facts-only chunk and case390 are not flagged
    assert kinds.count("personal-path") == 1


def test_freeze_refuses_on_findings_and_hashes(tmp_path):
    (tmp_path / "a.txt").write_text("clean fact\n")
    assert guard.main([str(tmp_path), "--freeze"]) == 0
    m = json.loads((tmp_path / guard.MANIFEST_NAME).read_text())
    assert len(m["corpus_sha256"]) == 64 and list(m["files"]) == ["a.txt"]
    (tmp_path / "b.txt").write_text("path /home/someone/x\n")
    assert guard.main([str(tmp_path), "--freeze"]) == 1


def test_journal_roundtrips_skipped_commands(tmp_path):
    from agentigrid.engine.journal import JournalEntry, SearchJournal
    j = SearchJournal()
    e = JournalEntry(iteration=1, description="d", commands=[], objective_value=1.0, feasible=True,
                     convergence_status="CONVERGED", violations_count=0, voltage_min=1.0, voltage_max=1.0,
                     max_line_loading_pct=0.0, total_gen_mw=0.0, total_load_mw=0.0, llm_reasoning="",
                     mode="fresh", elapsed_seconds=0.0)
    assert e.skipped_commands is None
    e.skipped_commands = ["Skipped x"]
    j.add_entry(e)
    out = tmp_path / "j.json"
    j.export_json(out)
    assert json.loads(out.read_text())["entries"][0]["skipped_commands"] == ["Skipped x"]


# --- intent guards in the scraper, frozen-corpus identity ------------------------

def test_scraper_drops_intent_violations(tmp_path):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    base = _entry(it=0, skipped=[])
    base["total_load_mw"] = 1000.0
    widen = _entry(it=1, skipped=[], cmds=[{"action": "set_all_bus_vlimits", "Vmin": 0.9, "Vmax": 1.1}],
                   desc="Relax the voltage band")
    widen["total_load_mw"] = 1000.0
    shed = _entry(it=2, skipped=[], cmds=[{"action": "scale_all_loads", "factor": 0.9}], desc="Shed load", )
    shed["total_load_mw"] = 900.0
    shed["mode"] = "fresh"
    ok = _entry(it=3, skipped=[], cmds=[{"action": "set_gen_voltage", "bus": 10, "Vg": 1.0}], desc="Lower setpoint")
    ok["total_load_mw"] = 1000.0
    ok["mode"] = "fresh"
    (run / "journal.json").write_text(json.dumps({"entries": [base, widen, shed, ok]}))
    (run / "manifest.json").write_text(json.dumps({
        "model": "m", "goal_text": "g", "case_path": "x/case118.m", "success": "no_violations",
        "guards": ["load_preserved", {"vband_not_widened": [0.95, 1.05]}]}))
    out = tmp_path / "corpus"
    assert scrape.main(["--runs-dir", str(tmp_path / "runs"), "--out", str(out),
                        "--exclude-spec", str(EVAL_SPEC)]) == 0
    text = (out / scrape.OUT_NAME).read_text()
    assert text.count("[source:") == 1 and "Lower setpoint" in text


def test_bootstrap_goals_declare_guards_and_model_is_outside_eval_set():
    boot = json.loads(BOOT_SPEC.read_text())
    ev = json.loads(EVAL_SPEC.read_text())
    assert all("guards" in g for g in boot["goals"])
    eval_models = {m["model"] for m in ev["models"]}
    assert not {m["model"] for m in boot["models"]} & eval_models


def test_corpus_status_tracks_freeze_and_ingest(tmp_path):
    from agentigrid.rag.corpus_hash import INGEST_MANIFEST_NAME, corpus_hash, corpus_status
    corpus, store = tmp_path / "corpus", tmp_path / "store"
    corpus.mkdir(); store.mkdir()
    (corpus / "a.txt").write_text("alpha")
    assert "not frozen" in corpus_status(corpus, store)["reason"]
    assert guard.main([str(corpus), "--freeze"]) == 0
    assert "store not built" in corpus_status(corpus, store)["reason"]
    digest, _ = corpus_hash(corpus)
    (store / INGEST_MANIFEST_NAME).write_text(json.dumps({"corpus_sha256": digest}))
    st = corpus_status(corpus, store)
    assert st["ok"] and st["corpus_sha256"] == digest
    (corpus / "a.txt").write_text("alpha changed")
    assert "changed since it was frozen" in corpus_status(corpus, store)["reason"]
