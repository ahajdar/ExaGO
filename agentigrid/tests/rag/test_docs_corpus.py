"""ExaGO documentation corpus builder; one store from several frozen corpora."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import agentigrid.rag.ingest as ing
from agentigrid.rag.corpus_hash import (MANIFEST_NAME, combined_hash, corpus_files, corpus_hash,
                                        corpus_status)

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bd = _load("build_docs_corpus", "rag/tools/build_docs_corpus.py")
er = _load("experiment_runner", "rag/tools/experiment_runner.py")

MD = """<!-- comment -->
## SCOPFLOW
SCOPFLOW solves a contingency-constrained OPF. See [OPFLOW](./opflow.md).
![diagram](img.png)

### Usage

#### Options
|-netfile| Name of network file|
|-ctgcfile| contingency list|

```
# not a heading inside code
./scopflow -netfile x.m
```
"""


def test_sections_fold_heading_only_and_ignore_code_hashes():
    secs = bd.sections(bd.clean(MD))
    paths = [p for p, _ in secs]
    assert paths == ["SCOPFLOW", "SCOPFLOW > Usage > Options"]
    body = "\n".join(secs[1][1])
    assert "# not a heading inside code" in body


def test_blocks_carry_provenance_and_clean_markdown():
    blocks = bd.blocks_for("scopflow.md", MD, 2800)
    assert blocks[0].startswith("[source: ExaGO documentation | file: scopflow.md | section: SCOPFLOW]\n")
    assert "See OPFLOW." in blocks[0] and "](" not in blocks[0] and "img.png" not in blocks[0]
    assert all("\n\n" not in b for b in blocks)          # ingest keeps each block one chunk


def test_long_sections_are_split_under_the_cap():
    md = "## Big\n" + "\n\n".join("p" * 900 for _ in range(6))
    blocks = bd.blocks_for("big.md", md, 2800)
    assert len(blocks) >= 2 and all(len(b) <= 2800 for b in blocks)
    assert all(b.startswith("[source: ExaGO documentation | file: big.md | section: Big]") for b in blocks)


def test_build_is_deterministic_and_notice_is_not_indexed(tmp_path):
    src = tmp_path / "repo" / "docs" / "web"
    src.mkdir(parents=True)
    (src / "scopflow.md").write_text(MD)
    (tmp_path / "repo" / "LICENSE").write_text("licence text")
    out = tmp_path / "corpus_docs"
    bd.build(src, out)
    h1 = corpus_hash(out)[0]
    bd.build(src, out)
    assert corpus_hash(out)[0] == h1
    names = sorted(p.name for p in corpus_files(out))
    assert names == ["exago_docs_scopflow.txt"]           # SOURCE and LICENSE.ExaGO are not indexed
    assert (out / "LICENSE.ExaGO").read_text() == "licence text"


# -- several corpora in one store ----------------------------------------------
class _FakeStore:
    def __init__(self, *a, **k):
        self.n = 0

    def reset(self):
        self.n = 0

    def add(self, *a, **k):
        self.n += 1

    def count(self):
        return self.n


def _frozen(tmp_path, name, text):
    d = tmp_path / name
    d.mkdir()
    (d / "a.txt").write_text(text)
    (d / MANIFEST_NAME).write_text(json.dumps({"corpus_sha256": corpus_hash(d)[0]}))
    return d


def test_multi_corpus_ingest_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "VectorStore", _FakeStore)
    core, docs = _frozen(tmp_path, "core", "x\n\ny"), _frozen(tmp_path, "docs", "z")
    store = tmp_path / "store"
    store.mkdir()
    ing.ingest([str(core), str(docs)], str(store))
    m = json.loads((store / "ingest_manifest.json").read_text())
    assert m["chunks"] == 3 and m["corpus_dir"] == [str(core), str(docs)]
    assert m["corpus_sha256s"] == [corpus_hash(core)[0], corpus_hash(docs)[0]]
    assert m["corpus_sha256"] == combined_hash(m["corpus_sha256s"])
    assert corpus_status([core, docs], store)["ok"]
    assert not corpus_status([docs, core], store)["ok"]       # order is part of the identity
    assert not corpus_status(core, store)["ok"]               # single-corpus check against a merged store
    (docs / "a.txt").write_text("changed")
    st = corpus_status([core, docs], store)
    assert not st["ok"] and "changed since it was frozen" in st["reason"]


def test_single_corpus_ingest_manifest_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "VectorStore", _FakeStore)
    core = _frozen(tmp_path, "core", "x")
    store = tmp_path / "store"
    store.mkdir()
    ing.ingest(str(core), str(store))
    m = json.loads((store / "ingest_manifest.json").read_text())
    assert m["corpus_dir"] == str(core) and "corpus_sha256s" not in m
    assert corpus_status(core, store)["ok"]


def test_runner_corpus_check_accepts_a_list(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "VectorStore", _FakeStore)
    _frozen(tmp_path, "core", "x")
    _frozen(tmp_path, "docs", "z")
    (tmp_path / "store_cd").mkdir()
    ing.ingest([str(tmp_path / "core"), str(tmp_path / "docs")], str(tmp_path / "store_cd"))
    cond = {"id": "C2", "corpus": ["core", "docs"],
            "env": {"AGENTIGRID_RAG_MODE": "corrective", "AGENTIGRID_RAG_STORE": "store_cd"}}
    assert er.corpus_check(cond, tmp_path)["ok"]
