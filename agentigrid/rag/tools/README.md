# RAG corpus tools

Two scripts that auto-build the AgentiGrid RAG knowledge base from **ground
truth** (the binaries, the case files, your own successful runs) instead of
hand-authored prose. Every chunk is provenance-tagged, so retrieved references
trace back to a source — which keeps the corpus auditable for the thesis.

## Install (in WSL)

Drop this folder at the project root so it sits next to `rag/`:

    ~/projects/ExaGO/agentigrid/
      rag/
        corpus/          <- output lands here
        tools/           <- put rag_harvest.py + rag_scrape_journal.py here
      applications/      <- ExaGO binaries (symlinks)
      data/  datafiles/  <- .m case files
      workdir/           <- where journal JSONs are exported

Run everything **from the project root** (the same place you launch AgentiGrid),
because the defaults are relative (`applications`, `data`, `workdir`, `rag/corpus`).

## 1. Harvest tool + case metadata (Tier 1)

    python rag/tools/rag_harvest.py

Writes `rag/corpus/exago_<app>_help.txt` (one per app, from `--help`/`--version`)
and `rag/corpus/exago_cases.txt` (bus/gen/branch counts per `.m` file).

Options:
    --bin-dir applications      # where the binaries are (or build/applications)
    --data-dir data --data-dir datafiles
    --apps opflow scopflow ...  # subset
    --no-cases / --no-help

## 2. Scrape exemplars from successful runs (self-bootstrapping)

    python rag/tools/rag_scrape_journal.py --runs-dir workdir --inspect   # preview
    python rag/tools/rag_scrape_journal.py --runs-dir workdir             # write

Writes `rag/corpus/exago_exemplars_from_runs.txt` — the converged, feasible
iterations that carried a real proposal, as few-shot `goal -> reasoning ->
commands -> result` examples. Only keeps genuine solves (drops FAILED,
ANALYSIS/EXPLORE/SWEEP/CONTINGENCY/COMPLETE, and anything without a proposal or
with an empty/"No description" goal).

> Point `--runs-dir` at wherever your runs export `journal.json`. If it's not
> `workdir/`, find one with: `find . -name 'journal*.json'` and pass its parent.

Generate exemplars from a **capable** model run (`claude-sonnet-4-6` or
`mixtral:latest`) — a weak model produces no valid proposals, so there's nothing
good to harvest (that's the same flat-iterations failure documented in the setup
guide).

## 3. Re-ingest after harvesting

Both scripts print this; run it once when you're done:

    rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus

## Notes

- **Idempotent.** Re-running overwrites only `exago_*.txt` (the generated files);
  your hand-written corpus files are left alone.
- **Offline.** Neither script calls the LLM or the network. `ingest` does (Ollama
  embeddings), the harvesters don't.
- **Review before trusting.** Every chunk carries an "auto-generated, review
  before trusting" tag. A successful run doesn't guarantee every proposal in it
  is a good teaching example — skim `exago_exemplars_from_runs.txt` and delete
  weak ones.
- **Provenance granularity.** Each chunk is prefixed with its `[source: ...]`
  tag inline, so the retriever's `[ref N]` output is self-identifying. The tag
  costs a few tokens per chunk; that's deliberate.
