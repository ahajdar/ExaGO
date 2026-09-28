"""Relevance-label tooling: pair sampling, thresholds, Platt, AUC, ECE (no model needed)."""

from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("relevance_labels", ROOT / "rag" / "tools" / "relevance_labels.py")
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)


def test_auc_and_ece_basics():
    assert rl.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert rl.auc([0.5, 0.5], [1, 0]) == 0.5
    assert rl.auc([0.3], [1]) is None
    assert rl.ece([1.0, 0.0], [1, 0]) == 0.0
    assert abs(rl.ece([0.9] * 10, [1] * 5 + [0] * 5) - 0.4) < 1e-9


def test_platt_recovers_a_logistic_link():
    rng = random.Random(0)
    x = [rng.uniform(-4, 4) for _ in range(3000)]
    y = [1 if rng.random() < 1 / (1 + math.exp(-(1.5 * xi - 0.5))) else 0 for xi in x]
    a, b = rl.platt_fit(x, y)
    assert abs(a - 1.5) < 0.2 and abs(b + 0.5) < 0.2


def test_thresholds_meet_precision_and_npv():
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    lo, hi = rl.thresholds(s, y, precision=0.8, npv=0.8)
    assert hi == 0.5        # precision at >=0.5: 5/6 >= 0.8
    # below 0.6 the share of irrelevant chunks is 4/5 = 0.8, but tau_lower may not
    # exceed tau_upper (the AMBIGUOUS band cannot be negative), so it is clipped
    assert lo == 0.5
    lo2, hi2 = rl.thresholds([0.1, 0.9], [1, 0], 0.99, 0.99)
    assert (lo2, hi2) == (None, None)


def test_sample_pairs_topk_plus_random_and_rewrites(monkeypatch):
    docs = [(f"d{i}", f"chunk {i}", {"source": f"s{i}.txt"}) for i in range(20)]

    def qfn(text, k):
        return [(f"chunk {i}", {"source": f"s{i}.txt"}, 1 - i / 100) for i in range(k)]

    queries = [{"goal_id": "g1", "kind": "goal", "text": "reduce cost"},
               {"goal_id": "g1", "kind": "rewrite", "text": "reduce cost x"}]
    rows = rl.sample_pairs(queries, docs, qfn, top=5, n_random=3, seed=1)
    assert len(rows) == 2 * (5 + 3)
    top = [r for r in rows if r["rank"] != "random"]
    rnd = [r for r in rows if r["rank"] == "random"]
    assert {r["doc_id"] for r in top if r["query_kind"] == "goal"} == {f"d{i}" for i in range(5)}
    assert not {r["doc_id"] for r in rnd if r["query_kind"] == "goal"} & {f"d{i}" for i in range(5)}
    assert [r["pair_id"] for r in rows] == list(range(1, len(rows) + 1))


def test_build_queries_adds_crag_rewrites():
    spec = {"goals": [{"id": "v", "text": "Eliminate all bus voltage violations"},
                      {"id": "x", "text": "zzz"}]}
    q = rl.build_queries(spec)
    kinds = [(r["goal_id"], r["kind"]) for r in q]
    assert ("v", "goal") in kinds and ("v", "rewrite") in kinds and ("x", "rewrite") not in kinds


def test_fit_report_end_to_end_with_fake_reranker():
    rng = random.Random(3)
    rows = []
    for g in ("a", "b", "c", "d"):
        for i in range(40):
            y = 1 if i % 3 == 0 else 0
            rows.append({"goal_id": g, "query": f"q{g}", "chunk": f"{g}{i}", "label": str(y),
                         "cosine": str(0.5 + 0.2 * y + rng.uniform(-0.1, 0.1)), "true": y})
    rows.append({"goal_id": "a", "query": "qa", "chunk": "unlabelled", "label": "", "cosine": "0.5"})
    truth = {r["chunk"]: r.get("true", 0) for r in rows}

    def fake_rerank(pairs):
        return [min(0.99, max(0.01, 0.2 + 0.6 * truth[c] + rng.uniform(-0.15, 0.15))) for _q, c in pairs]

    rep = rl.fit_report(rows, {"c", "d"}, fake_rerank)
    assert rep["n_labelled"] == 160 and rep["n_heldout"] == 80
    assert rep["heldout"]["auc_reranker"] > 0.95
    assert rep["C2b"]["tau_upper"] is not None
    assert rep["heldout"]["ece_calibrated"] <= rep["heldout"]["ece_reranker_unit"] + 0.05
    with pytest.raises(SystemExit):
        rl.fit_report(rows, {"zz"}, fake_rerank)
