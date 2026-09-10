# RAG Usage — AgentiGrid generation-stage grounding

## What it is (short)

Generation-stage RAG grounds the LLM's proposal generation in a curated corpus
(ExaGO tool facts, methodology notes, and `request → correct-spec` exemplars) to
cut executable-but-misaligned proposals **at the source**. **Design rule: RAG
touches only the generation path — never the deterministic validator**
(`engine/validation.py`), so verification stays reproducible and the
anti-circularity argument holds. A single environment variable, `AGENTIGRID_RAG`,
is the on/off ablation switch. Retrieval is **defensive**: any failure (Ollama
down, empty index, `chromadb` missing) returns nothing, so the run behaves exactly
like the no-RAG baseline — a broken index can never silently corrupt a result.

## Architecture

- **Importable package** `agentigrid/agentigrid/rag/` (imported as `agentigrid.rag`):
  `embed.py` (Ollama `nomic-embed-text`), `store.py` (persistent Chroma, cosine
  distance), `retriever.py` (`top_k`, `min_score` threshold), `ingest.py` (build the
  index), `__init__.py`.
- **Data** lives at the app level under `agentigrid/rag/`: `corpus/` (source
  `.txt`/`.md`), `store/` (Chroma index), `tools/` (corpus builders).
- **Defaults**: collection `agentigrid_kb`, embed model `nomic-embed-text`, `k=3`,
  `min_score=0.35` (cosine similarity, `1.0` = identical meaning).

All commands below run from the **app root** with the venv active:

```bash
cd ~/projects/ExaGO/agentigrid
source .venv/bin/activate
```

## Prerequisites

- Ollama running with the embedding model pulled: `ollama pull nomic-embed-text`.
- `chromadb` in the venv: `pip install chromadb`.
- `OLLAMA_HOST` exported (WSL → Windows host if Ollama runs on Windows):
  `export OLLAMA_HOST="http://<WIN_IP>:11434"` — verify with `echo $OLLAMA_HOST`
  (it must **not** read `http://:11434`). If you use `env.sh`, this is already set.

## Step 1 — Build the corpus

Populate `rag/corpus` in either (or both) of two ways.

**(a) By hand** — drop curated `.txt`/`.md` files into `rag/corpus` (tool docs, N-1
methodology, `request → spec` exemplars).

**(b) With the harvest tools** (recommended starting point):

```bash
# Tier-1 ground truth: each app's --help/--version + MATPOWER case bus/gen/branch
# counts -> rag/corpus/exago_*.txt. No LLM, no network. Idempotent.
python rag/tools/rag_harvest.py

# Mine exemplars from past run journals (converged + feasible iterations with real
# proposals). --inspect previews and writes nothing; drop it to write to the corpus.
python rag/tools/rag_scrape_journal.py --runs-dir workdir --inspect
python rag/tools/rag_scrape_journal.py --runs-dir workdir
```

## Step 2 — Build the index (ingest)

```bash
rm -rf rag/store                              # only after a corpus OR store.py change
python -m agentigrid.rag.ingest rag/corpus    # prints "Ingested N chunks; ... holds M."
```

`rm -rf rag/store` is required after any corpus or `store.py` change because the
cosine space needs a fresh index. Chunking is paragraph-first, 800 chars with 100
overlap.

## Step 3 — Enable and run

**CLI (the ablation switch is the env var):**

```bash
GOAL="Reduce total generation cost by 10%"
AGENTIGRID_RAG=0 agentigrid ./data/case39.m "$GOAL" --backend ollama --model qwen2.5:7b --max-iter 4 < /dev/null   # baseline
AGENTIGRID_RAG=1 agentigrid ./data/case39.m "$GOAL" --backend ollama --model qwen2.5:7b --max-iter 4 < /dev/null   # RAG on
```

**UI:** launch the Streamlit app (`./launcher/run.sh`). The "reference knowledge"
toggle sets `AGENTIGRID_RAG` for the run, and the live monitor shows a grounding
indicator (references used + top similarity score) during the search.

## Step 4 — Verify grounding

**Standalone** (embeds a query, prints retrieved refs + scores):

```bash
python -c "from agentigrid.rag import Retriever; print(Retriever(enabled=True, host='$OLLAMA_HOST').retrieve('Assess security under single-element outages'))"
```

Non-empty output with `[ref N | score 0.xx]` lines = working. Empty = see
Troubleshooting.

**In the UI:** the grounding caption during a search shows the reference count and
top score.

## Using it as an experiment ablation

`AGENTIGRID_RAG=0` vs `=1` is the **only** difference between the two conditions
(`C0-norag` / `C1-basic`) in the experiment harness. Because the retriever degrades
gracefully, `C1` collapses to `C0` behavior if retrieval fails — so the ablation is
clean: RAG either adds grounded references or nothing at all.

## Maintenance

- Re-ingest after any corpus edit: `rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus`.
- Tune relevance in `retriever.py`: `k` (how many chunks) and `min_score` (drop weak
  matches; lower it to admit more).

## Troubleshooting

- **`RAGDEBUG enabled=True len=0`** → retrieval ran but returned nothing: (a) embedder
  host is `localhost` not the WIN_IP → `export OLLAMA_HOST=...`; (b) index empty or old
  format → `rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus`; (c) every
  hit is below `min_score` → lower it in `retriever.py`.
- **`RAGDEBUG enabled=False`** → the retriever disabled itself at construction: missing
  `chromadb` (`pip install chromadb`) or a bad `rag/store` path.
- **Standalone `retrieve()` returns `""` but the UI shows references** → the shell's
  `OLLAMA_HOST` was empty. `export OLLAMA_HOST="http://<WIN_IP>:11434"`.
- **`model '...' not found (404)`** for embeddings → `ollama pull nomic-embed-text`.

> `rag/store/` is a build artifact and is gitignored — rebuild it locally with
> `ingest`. If `rag/corpus/` comes up empty on a fresh clone, run the harvest tools in
> Step 1, then ingest.
