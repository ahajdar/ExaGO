"""experiment_stats: Wilcoxon, Holm, Kruskal-Wallis against scipy (when present), pairing, H3 rule."""

from __future__ import annotations

import csv
import importlib.util
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("experiment_stats", ROOT / "rag" / "tools" / "experiment_stats.py")
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)


def test_wilcoxon_matches_scipy():
    st = pytest.importorskip("scipy.stats")
    rng = random.Random(1)
    small = [rng.uniform(-1, 1.5) for _ in range(15)]                 # no ties -> exact
    assert es.wilcoxon(small)["method"] == "exact"
    assert es.wilcoxon(small)["p"] == pytest.approx(st.wilcoxon(small, method="exact").pvalue, abs=1e-9)
    tied = [rng.choice([-0.25, 0, 0.25, 0.5, 0.75]) for _ in range(60)]   # ties + zeros -> normal
    ref = st.wilcoxon(tied, zero_method="wilcox", correction=True, method="approx").pvalue
    assert es.wilcoxon(tied)["p"] == pytest.approx(ref, abs=1e-9)
    assert es.wilcoxon([0, 0])["p"] == 1.0


def test_holm():
    assert es.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_kruskal_matches_scipy():
    st = pytest.importorskip("scipy.stats")
    g = [[0.1, 0.2, 0.2, 0.5], [0.3, 0.4, 0.4, 0.9, 1.0], [0.0, 0.0, 0.1]]
    ours, ref = es.kruskal(g), st.kruskal(*g)
    assert ours["h"] == pytest.approx(ref.statistic, rel=1e-9)
    assert ours["p"] == pytest.approx(ref.pvalue, rel=1e-7)


def _rows(tmp_path, effect=0.3, intent_drop=0.0):
    rng = random.Random(7)
    rows = []
    for model in ("weak", "strong"):
        for goal in ("cost10", "relieve"):
            for rep in range(1, 13):
                base_v = rng.uniform(0.2, 0.5)
                for cond in ("C0-norag", "C1-basic", "C2a-crag-cosine"):
                    lift = 0 if cond.startswith("C0") else (effect if model == "strong" else 0.0)
                    val_rej = 0.5 if cond.startswith("C0") else 0.2
                    intent = 0.4 if cond.startswith("C0") else 0.4 - intent_drop
                    rows.append({"case": "c", "goal": goal, "model": model, "rep": rep, "condition": cond,
                                 "status": "ok", "valid_proposal_rate": round(base_v + lift + rng.uniform(-.02, .02), 3),
                                 "goal_attained": 1 if lift and rng.random() < .7 else 0,
                                 "cost_improvement_pct": 5 + 10 * lift if goal == "cost10" else "",
                                 "baseline_attained": 0,
                                 "validator_rejection_rate": val_rej + rng.uniform(-.05, .05),
                                 "intent_violation_rate": intent + rng.uniform(-.05, .05)})
    p = tmp_path / "per_run.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


def test_end_to_end_detects_strong_model_effect_and_h3(tmp_path):
    res = es.analyse(es.load_rows(_rows(tmp_path)))
    c1 = next(r for r in res["primary"] if r["contrast"] == "C1 vs C0" and r["metric"] == "valid_proposal_rate")
    assert c1["n_pairs"] == 48 and c1["significant"]
    strong = [r for r in res["per_model"] if r["model"] == "strong" and r["metric"] == "valid_proposal_rate"
              and r["contrast"] == "C1 vs C0"][0]
    weak = [r for r in res["per_model"] if r["model"] == "weak" and r["metric"] == "valid_proposal_rate"
            and r["contrast"] == "C1 vs C0"][0]
    assert strong["significant"] and not weak["significant"]          # capability floor
    assert res["h2_interaction"][0]["p"] < 0.05
    cost = next(r for r in res["primary"] if r["metric"] == "cost_improvement_pct")
    assert cost["n_pairs"] == 24                                       # cost goals only
    assert {r["verdict"] for r in res["h3"]} == {"supported"}          # validator rejections fall, intent flat


def test_h3_rejected_when_intent_falls_more(tmp_path):
    res = es.analyse(es.load_rows(_rows(tmp_path, intent_drop=0.38)))
    assert {r["verdict"] for r in res["h3"]} == {"rejected"}


def test_h3_defined_when_c0_intent_rate_is_zero(tmp_path):
    rows = es.load_rows(_rows(tmp_path))
    for r in rows:
        r["intent_violation_rate"] = "0"
    res = es.analyse(rows)
    assert {r["verdict"] for r in res["h3"]} == {"supported"}          # absolute reduction still defined
    assert all(r["rr_intent"] is None and r["ar_intent"] == 0 for r in res["h3"])


def test_baseline_attained_cells_are_excluded(tmp_path):
    p = _rows(tmp_path)
    rows = es.load_rows(p)
    for r in rows:
        if r["goal"] == "relieve" and r["model"] == "weak":
            r["baseline_attained"] = "1"
    res = es.analyse(rows)
    assert ["c", "relieve", "weak"] in res["excluded_cells_baseline_attained"]
    assert res["n_units"] == 36
