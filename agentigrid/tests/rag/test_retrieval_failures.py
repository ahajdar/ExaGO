"""A retrieval run whose queries fail must not pass as a clean retrieval run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from agentigrid.rag.corrective import CorrectiveRetriever
from agentigrid.rag.retriever import Retriever

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("experiment_runner", ROOT / "rag" / "tools" / "experiment_runner.py")
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)


class _DownStore:
    def query(self, q, k):
        raise ConnectionError("embedding server unreachable")


def _retriever():
    r = Retriever(enabled=False)
    r.enabled, r._store = True, _DownStore()
    return r


def test_basic_retriever_counts_errors():
    r = _retriever()
    assert r.retrieve("reduce cost") == ""
    assert r.stats == {"calls": 1, "errors": 1, "empty": 0}
    assert r.describe()["stats"]["errors"] == 1


def test_corrective_exposes_base_errors():
    c = CorrectiveRetriever(_retriever())
    assert c.retrieve("reduce cost") == ""
    assert c.describe()["stats"]["retrieval_errors"] >= 1


def _journal(tmp_path, stats, enabled=True):
    p = tmp_path / "journal.json"
    p.write_text(json.dumps({"rag_config": {"enabled": enabled, "stats": stats}}))
    return p


def test_runner_flags_rag_errors_only_for_retrieval_conditions(tmp_path):
    c1 = {"id": "C1", "env": {"AGENTIGRID_RAG_MODE": "basic"}}
    c0 = {"id": "C0", "env": {"AGENTIGRID_RAG_MODE": "off"}}
    bad = _journal(tmp_path, {"calls": 4, "errors": 4, "empty": 0})
    assert "failed" in er.rag_failure(bad, c1)
    assert er.rag_failure(bad, c0) is None
    good = _journal(tmp_path, {"calls": 4, "errors": 0, "empty": 0})
    assert er.rag_failure(good, c1) is None
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"rag_config": {"enabled": True}}))        # pre-telemetry journal
    assert er.rag_failure(old, c1) is None


def test_embed_preflight_reports_unreachable_host():
    assert er.embed_preflight("http://127.0.0.1:9") is not None


def test_chunking_keeps_worked_examples_whole():
    from agentigrid.rag.ingest import chunk
    example = "Goal: lower cost\nStep applied: x\nCorrect response (JSON): " + '{"a": 1}, ' * 180 + "\nResult: ok"
    assert 800 < len(example) < 3000
    assert chunk(example + "\n\nsecond paragraph") == [example, "second paragraph"]
    long = "y" * 7000
    parts = chunk(long)
    assert len(parts) == 3 and all(len(p) <= 3000 for p in parts)


def test_status_refuses_store_with_other_chunking(tmp_path):
    from agentigrid.rag.corpus_hash import CHUNKING, INGEST_MANIFEST_NAME, MANIFEST_NAME, corpus_hash, corpus_status
    corpus, store = tmp_path / "corpus", tmp_path / "store"
    corpus.mkdir(); store.mkdir()
    (corpus / "a.txt").write_text("hello")
    digest, _ = corpus_hash(corpus)
    (corpus / MANIFEST_NAME).write_text(json.dumps({"corpus_sha256": digest}))
    (store / INGEST_MANIFEST_NAME).write_text(json.dumps({"corpus_sha256": digest}))          # v1 store
    st = corpus_status(corpus, store)
    assert not st["ok"] and "chunking" in st["reason"]
    (store / INGEST_MANIFEST_NAME).write_text(json.dumps({"corpus_sha256": digest, "chunking": dict(CHUNKING)}))
    assert corpus_status(corpus, store)["ok"]
