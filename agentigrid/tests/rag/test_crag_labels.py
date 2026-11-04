"""Blind CRAG label sheet: sampling with quotas, blindness, scoring."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("crag_labels", ROOT / "rag" / "tools" / "crag_labels.py")
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)

VOLT = "Eliminate all bus voltage violations without changing the bus voltage limits"
COST = "Reduce total generation cost by 10%"


def _replay(tmp: Path) -> Path:
    d = tmp / "replay"
    d.mkdir()
    qs = [
        {"query_sha": "qv", "mode": "goal", "goal": "voltage", "query": VOLT},
        {"query_sha": "qc", "mode": "goal", "goal": "cost10", "query": COST},
        {"query_sha": "qs", "mode": "state", "goal": "voltage", "query": VOLT + "\nCurrent state: CONVERGED, 149 violations."},
        {"query_sha": "qe", "mode": "error", "goal": "cost10", "query": "Command errors: Skipped X: slack bus"},
    ]
    for i in range(20):
        qs.append({"query_sha": f"qx{i}", "mode": "error", "goal": "cost10", "query": f"Command errors: e{i}"})
    (d / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in qs) + "\n")
    ch = [
        {"chunk_sha": "band", "source": "exago_exemplars_from_runs.txt", "text": "Step applied: set_all_bus_vlimits(Vmin=0.97, Vmax=1.03)"},
        {"chunk_sha": "doc", "source": "exago_docs_opflow.txt", "text": "[source: ExaGO documentation | file: opflow.md] options"},
        {"chunk_sha": "ok", "source": "exago_exemplars_from_runs.txt", "text": "Step applied: set_gen_status"},
    ]
    (d / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in ch) + "\n")
    rows = []
    for q in qs:
        for c, cos, rr in (("band", 0.70, 0.60), ("doc", 0.55, 0.05), ("ok", 0.68, 0.95)):
            rows.append({"query_sha": q["query_sha"], "chunk_sha": c, "source": "", "cosine": cos, "grade_reranker": rr})
    with open(d / "pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["query_sha", "chunk_sha", "source", "cosine", "grade_reranker"])
        w.writeheader()
        w.writerows(rows)
    return d


def test_situation_text():
    goals = {"voltage": VOLT}
    assert cl.situation({"mode": "goal", "goal": "voltage", "query": VOLT}, goals) == cl.START
    assert cl.situation({"mode": "state", "goal": "voltage", "query": VOLT + "\nCurrent state: X"}, goals) == "Current state: X"
    assert cl.situation({"mode": "error", "goal": "voltage", "query": "Command errors: e"}, goals) == "Command errors: e"


def test_make_quotas_blindness_and_key(tmp_path):
    from openpyxl import load_workbook
    d = _replay(tmp_path)
    out = tmp_path / "labels" / "s.xlsx"
    assert cl.main(["make", "--replay", str(d), "--out", str(out), "--n", "20", "--doc-quota", "3", "--band-quota", "2"]) == 0
    key = list(csv.DictReader(open(out.with_name("s_key.csv"))))
    assert len(key) == 20
    assert sum(k["is_doc"] == "True" for k in key) >= 3
    assert sum(k["is_band"] == "True" for k in key) >= 2       # band = voltage goal + band command
    ws = load_workbook(out)["Pairs"]
    header = [c.value for c in ws[1]]
    assert header[:5] == ["Item", "Goal", "Situation", "Retrieved reference", "Label (1/0)"]
    flat = " ".join(str(c.value) for row in ws.iter_rows() for c in row if c.value is not None)
    for leak in ("0.95", "grade", "cosine", "reranker", "C1-basic", "qwen"):
        assert leak not in flat
    assert ws.max_row == 21
    meta = json.loads(out.with_name("s_meta.json").read_text())
    assert meta["n"] == 20 and meta["label_definition"].startswith("B:")
    # deterministic
    out2 = tmp_path / "labels" / "s2.xlsx"
    cl.main(["make", "--replay", str(d), "--out", str(out2), "--n", "20", "--doc-quota", "3", "--band-quota", "2"])
    assert [k["chunk_sha"] for k in key] == [k["chunk_sha"] for k in csv.DictReader(open(out2.with_name("s2_key.csv")))]


def test_score_reports_auc_thresholds_and_subsets(tmp_path):
    from openpyxl import load_workbook
    d = _replay(tmp_path)
    out = tmp_path / "s.xlsx"
    cl.main(["make", "--replay", str(d), "--out", str(out), "--n", "30"])
    key = {k["item"]: k for k in csv.DictReader(open(tmp_path / "s_key.csv"))}
    wb = load_workbook(out)
    ws = wb["Pairs"]
    for row in ws.iter_rows(min_row=2):
        k = key[row[0].value]
        row[4].value = 1 if k["chunk_sha"] == "ok" else 0          # band and doc chunks are not useful
        row[5].value = "Y"
    filled = tmp_path / "s_filled.xlsx"
    wb.save(filled)
    assert cl.main(["score", "--sheet", str(filled), "--key", str(tmp_path / "s_key.csv"),
                    "--out", str(tmp_path / "scores.json")]) == 0
    rep = json.loads((tmp_path / "scores.json").read_text())
    assert rep["n_labelled"] == 30 and rep["unlabelled"] == 0
    assert rep["graders"]["reranker"]["auc"] == 1.0          # 0.95 for useful, 0.60/0.05 otherwise
    assert rep["graders"]["cosine"]["auc"] < 1.0             # band chunk has the higher cosine
    assert rep["graders"]["reranker"]["replay_arg"].startswith("--tau-reranker ")
    assert rep["subsets"]["band_change_pairs"]["positives"] == 0
    assert rep["subsets"]["confident_share"] == 1.0
    sr = rep["stopping_rule"]
    # 30 labels: ok-chunk items are 1 (about a third) -> fewer than 10 of one class is possible;
    # the check must be consistent with the counts either way
    pos = rep["positives"]
    assert sr["second_batch_needed"] == (pos < 10 or 30 - pos < 10 or
                                         any(g["tau_lower"] is None or g["tau_upper"] is None
                                             for g in rep["graders"].values()))


def test_defaults_are_50_with_6_plus_6_quotas(tmp_path):
    d = _replay(tmp_path)
    out = tmp_path / "d.xlsx"
    cl.main(["make", "--replay", str(d), "--out", str(out)])
    meta = json.loads(out.with_name("d_meta.json").read_text())
    assert meta["n"] == 50 and meta["doc_quota"] == 6 and meta["band_quota"] == 6
    assert "second batch" in meta["stopping_rule"]


def test_stopping_rule_function():
    base = {"positives": 25, "n_labelled": 50,
            "graders": {"cosine": {"tau_lower": 0.6, "tau_upper": 0.7}, "reranker": {"tau_lower": 0.3, "tau_upper": 0.9}}}
    assert not cl.second_batch_check(base)["second_batch_needed"]
    few = {**base, "positives": 8}
    assert cl.second_batch_check(few)["reasons"] == ["only 8 labels of 1 (< 10)"]
    nothr = {**base, "graders": {**base["graders"], "cosine": {"tau_lower": None, "tau_upper": 0.7}}}
    assert cl.second_batch_check(nothr)["reasons"] == ["no threshold found for cosine"]
