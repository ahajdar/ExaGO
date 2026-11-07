"""CRAG campaign: specs, threshold placeholders, band and rag_log metrics, paired stats."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


er = _load("experiment_runner", "rag/tools/experiment_runner.py")
ev = _load("experiment_eval", "rag/tools/experiment_eval.py")
cs = _load("crag_stats", "rag/tools/crag_stats.py")


def test_specs_are_consistent_and_thresholds_are_refused_until_filled():
    spec = json.loads((ROOT / "rag/tools/specs/crag_spec.json").read_text())
    pilot = json.loads((ROOT / "rag/tools/specs/crag_pilot_spec.json").read_text())
    ids = [c["id"] for c in spec["conditions"]]
    assert ids == ["C0-norag", "C1g-basic", "C1e-basic", "C2c-cosine", "C2r-reranker"]
    assert len({i.split("-")[0] for i in ids}) == len(ids)          # distinct keys for the stats
    assert spec["reps"] == 20 and pilot["reps"] == 1 and spec["out_dir"] != pilot["out_dir"]
    by = {c["id"]: c for c in spec["conditions"]}
    for cid in ("C1e-basic", "C2c-cosine", "C2r-reranker"):
        env = by[cid]["env"]
        assert env["AGENTIGRID_RAG_QUERY"] == "error" and env["AGENTIGRID_RAG_STORE"] == "rag/store_core_docs"
        assert by[cid]["corpus"] == ["rag/corpus", "rag/corpus_docs"]
    assert by["C1g-basic"]["env"]["AGENTIGRID_RAG_STORE"] == "rag/store_core_docs"   # same store: query isolated
    assert er.unimplemented_reason(by["C1e-basic"]) is None
    assert er.unimplemented_reason(by["C2c-cosine"]).startswith("AGENTIGRID_CRAG_TAU_LOWER")
    filled = json.loads(json.dumps(by["C2c-cosine"]))
    filled["env"].update(AGENTIGRID_CRAG_TAU_LOWER="0.62", AGENTIGRID_CRAG_TAU_UPPER="0.71")
    assert er.unimplemented_reason(filled) is None
    pc = {c["id"]: c for c in pilot["conditions"]}
    assert set(pc) == {"C1e-basic", "C2r-reranker-default"}
    assert "AGENTIGRID_CRAG_TAU_LOWER" not in pc["C2r-reranker-default"]["env"]


def test_band_metrics_counts_and_directions():
    g = ev.normalize_guards(["load_preserved", {"vband_not_widened": [0.95, 1.05]}])
    entries = [{"iteration": 0, "commands": []},
               {"iteration": 1, "commands": [{"action": "set_all_bus_vlimits", "Vmin": 0.97, "Vmax": 1.03}]},
               {"iteration": 2, "commands": [{"action": "set_all_bus_vlimits", "Vmin": 0.95, "Vmax": 1.05},
                                             {"action": "set_gen_voltage", "bus": 3}]}]
    assert ev.band_metrics(entries, g) == {"band_cmds": 2, "band_narrowed": 1, "band_widened": 0}
    assert ev.band_metrics(entries, ev.normalize_guards(["load_preserved"]))["band_narrowed"] is None


def test_rag_log_metrics():
    j = {"rag_log": [
        {"iteration": 1, "query_mode": "error", "action": "used", "verdict": "correct", "injected_refs": 3},
        {"iteration": 2, "query_mode": "error", "action": "withheld", "verdict": "incorrect", "rewrite": True},
        {"iteration": 3, "query_mode": "error", "action": "used", "verdict": "ambiguous", "injected_refs": 1}]}
    m = ev.rag_log_metrics(j)
    assert (m["rag_log_iters"], m["rag_log_used"], m["rag_log_withheld"], m["rag_log_noncorrect"],
            m["rag_log_rewrites"], m["rag_log_mean_refs"], m["rag_log_query_mode"]) == (3, 2, 1, 2, 1, 2.0, "error")
    assert ev.rag_log_metrics({})["rag_log_iters"] is None


def test_mcnemar_exact():
    assert cs.mcnemar_exact(0, 0) == 1.0
    assert abs(cs.mcnemar_exact(12, 0) - 2 * 0.5 ** 12) < 1e-12
    assert abs(cs.mcnemar_exact(20, 0) - 2 * 0.5 ** 20) < 1e-12
    assert cs.mcnemar_exact(5, 5) == 1.0


def _rows(tmp_path):
    rows = []
    for model in ("m1", "m2"):
        for rep in range(1, 13):
            for cond, narrowed, valid in (("C1e-basic", 1, 0.8), ("C2r-reranker", 0, 0.9)):
                rows.append({"case": "v", "goal": "voltage", "model": model, "rep": rep, "condition": cond,
                             "status": "ok", "baseline_attained": 0, "band_narrowed": narrowed,
                             "valid_proposal_rate": valid + 0.001 * rep, "validator_rejection_rate": 0.1,
                             "goal_attained": 0, "cost_improvement_pct": "", "valid_proposal_rate_zero_by_rule": 0})
    p = tmp_path / "per_run.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


def test_crag_stats_primary_family_and_exploratory(tmp_path):
    p = _rows(tmp_path)
    assert cs.main(["--per-run", str(p), "--primary", "C2r:C1e:band_narrowed"]) == 0
    res = json.loads((tmp_path / "crag_stats.json").read_text())
    assert res["conditions_present"] == ["C1e", "C2r"]
    band = [r for r in res["results"] if r["metric"] == "band_narrowed" and r["model"] == "pooled"][0]
    assert band["primary"] and band["discordant_treat_only"] == 0 and band["discordant_ctrl_only"] == 24
    assert band["p"] < 1e-6 and "p_holm_primary" in band
    valid = [r for r in res["results"] if r["metric"] == "valid_proposal_rate" and r["model"] == "pooled"][0]
    assert not valid["primary"] and valid["median_diff"] > 0
    assert cs.main(["--per-run", str(p)]) == 0
    assert json.loads((tmp_path / "crag_stats.json").read_text())["status"].startswith("all exploratory")
