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
      applications/exago/ <- ExaGO binaries (symlinks)
      data/exago/         <- .m case files (datafiles/ yours, examples/ ExaGO links)
      workdir/           <- where journal JSONs are exported

Run everything **from the project root** (the same place you launch AgentiGrid),
because the defaults are relative (`applications/exago`, `data/exago`, `workdir`, `rag/corpus`).

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

## 2b. Schema exemplars (no domain knowledge needed)

    python rag/tools/rag_schema_exemplars.py

Writes `rag/corpus/agentigrid_schema_exemplars.txt`: complete, correct responses
(e.g. `set_all_bus_vlimits` nested inside a `modify` action, never as the
top-level action). Each exemplar is certified by AgentiGrid's own parser,
validator and modifier on the held-out case118; the one that no held-out case can
host (phase shifter) is labelled "schema-validated only".

## 2c. Corpus bootstrap (knowledge distillation)

A capable model works HELD-OUT tasks; the scraper keeps only solver-certified
iterations (applied commands, converged, feasible, zero violations):

    python rag/tools/experiment_runner.py --spec rag/tools/specs/corpus_bootstrap_spec.json
    python rag/tools/rag_scrape_journal.py --runs-dir experiments/corpus_bootstrap_v1 --inspect
    python rag/tools/rag_scrape_journal.py --runs-dir experiments/corpus_bootstrap_v1

The scraper refuses journals without a skip record (`--allow-legacy` to override),
teaches the `modify` action JSON rather than the ExaGO command line, and drops
anything from the evaluated networks/goals (`grader_ablation_spec.json`).

## 2d. Audit and freeze

    python rag/tools/corpus_guard.py --spec grader_ablation_spec.json rag/corpus
    python rag/tools/corpus_guard.py --spec grader_ablation_spec.json rag/corpus --freeze

Flags goal leaks, worked examples on evaluated networks, and personal paths;
`--freeze` (refused while findings remain) writes `corpus_manifest.json` with the
corpus SHA-256 that runs should record.

Copyright: do not paste paper or textbook text into the corpus. State facts in
your own words and cite, or keep such notes in a local, gitignored corpus.

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
