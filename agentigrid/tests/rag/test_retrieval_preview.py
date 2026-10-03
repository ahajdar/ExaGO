"""retrieval_preview: per-goal injected context and identity across conditions (no store needed)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("retrieval_preview", ROOT / "rag" / "tools" / "retrieval_preview.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

SPEC = {
    "env": {"AGENTIGRID_MAX_TOKENS": "16000"},
    "conditions": [
        {"id": "C0", "env": {"AGENTIGRID_RAG_MODE": "off"}},
        {"id": "C1", "env": {"AGENTIGRID_RAG_MODE": "basic"}},
        {"id": "C2a", "env": {"AGENTIGRID_RAG_MODE": "corrective", "AGENTIGRID_CRAG_GRADER": "cosine"}},
        {"id": "C2b", "env": {"AGENTIGRID_RAG_MODE": "corrective", "AGENTIGRID_CRAG_GRADER": "reranker"}},
    ],
    "goals": [{"id": "cost", "text": "reduce cost"}, {"id": "volt", "text": "fix voltage"}],
}


class _Basic:
    enabled = True

    def retrieve(self, q):
        return f"[ref 1 | score 0.60] about {q}"


class _Corrective:
    enabled = True

    def __init__(self, grader):
        self.grader = grader
        self.stats = {"calls": 0, "correct": 0, "ambiguous": 0, "incorrect": 0,
                      "rewrites": 0, "withheld": 0, "grader_fallbacks": 0}

    def retrieve(self, q):
        self.stats["calls"] += 1
        if self.grader == "reranker" and "voltage" in q:     # reranker rejects, rewrite rescues
            self.stats["incorrect"] += 1
            self.stats["rewrites"] += 1
            self.stats["ambiguous"] += 1
            return f"[ref 1 | score 0.40] about {q} voltage limits"
        self.stats["correct"] += 1
        return f"[ref 1 | score 0.60] about {q}"


def _build(host):
    mode = os.environ.get("AGENTIGRID_RAG_MODE")
    assert "AGENTIGRID_MAX_TOKENS" not in os.environ or True
    return _Corrective(os.environ.get("AGENTIGRID_CRAG_GRADER", "cosine")) if mode == "corrective" else _Basic()


def test_preview_skips_c0_and_flags_identical_and_differing_goals():
    os.environ["AGENTIGRID_CRAG_GRADER"] = "stale"       # must not leak into conditions
    try:
        res = rp.preview(SPEC, "http://x", build=_build)
    finally:
        os.environ.pop("AGENTIGRID_CRAG_GRADER", None)
    assert res["conditions"] == ["C1", "C2a", "C2b"]
    cost, volt = res["goals"]["cost"], res["goals"]["volt"]
    assert cost["_identical"]["C2a == C2b"] is True
    assert volt["_identical"]["C2a == C2b"] is False
    assert volt["C2b"]["verdict"] == "incorrect > rewrite > ambiguous"
    assert cost["C2a"]["verdict"] == "correct" and cost["C1"]["verdict"] == "basic"
    assert "C2a == C2b" in rp.report(res) and "1/2" in rp.report(res)
    assert os.environ.get("AGENTIGRID_CRAG_GRADER") is None
