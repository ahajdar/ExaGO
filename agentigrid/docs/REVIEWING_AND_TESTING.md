# Reviewing & Testing AgentiGrid (RAG branch)

A short guide for a reviewer to (A) read the code and (B) build and run it.

## What this is

AgentiGrid is an LLM-driven optimal-power-flow (OPF) search loop built on ExaGO:
an LLM proposes network modifications, ExaGO solves the OPF, and a **deterministic
verifier** checks each proposal — the verifier is the core research contribution.
This branch adds a **generation-stage RAG** component (subordinate to the verifier),
supporting tooling, and an experiment harness. It is a fork of ORNL/ExaGO; all new
work is additive under `agentigrid/`.

## Where the code is

- **Fork (public):** https://github.com/ahajdar/ExaGO
- **Branch:** `amir/agentigrid-rag` (base: `develop`)
- **Diff on the web:** https://github.com/ahajdar/ExaGO/compare/develop...amir/agentigrid-rag

Everything new is under `agentigrid/`:

| Path | What it is |
|------|------------|
| `agentigrid/agentigrid/engine/validation.py` | Deterministic verifier — the thesis core (note: RAG is deliberately absent here) |
| `agentigrid/agentigrid/engine/agent_loop.py` | The search loop; RAG is injected only at the generation step |
| `agentigrid/agentigrid/rag/` | RAG package: `embed` / `store` / `retriever` / `ingest` |
| `agentigrid/rag/tools/` | Corpus builders + experiment harness (`rag_harvest.py`, `rag_scrape_journal.py`, `experiment_runner.py`, `experiment_eval.py`) |
| `agentigrid/launcher/` | Streamlit UI |
| `agentigrid/docs/` | This guide, `RAG_USAGE.md`, `AgentiGrid_Setup.md`, experiment plan, ADR |

## Path A — Review the code only (no build)

1. Open the **compare URL** above to see the full diff vs `develop`.
2. Or locally:

   ```bash
   git clone https://github.com/ahajdar/ExaGO.git
   cd ExaGO
   git checkout amir/agentigrid-rag
   git diff develop...amir/agentigrid-rag -- agentigrid/     # the change set
   ```

3. Suggested reading order:
   - `agentigrid/docs/AgentiGrid_Setup.md` — how the whole system builds and runs.
   - `agentigrid/docs/RAG_USAGE.md` — the RAG feature this branch adds.
   - `agentigrid/agentigrid/engine/validation.py` — the deterministic verifier.
   - `agentigrid/agentigrid/rag/retriever.py`, `store.py`, `ingest.py` — the RAG path.
   - `agentigrid/agentigrid/engine/agent_loop.py` — the loop; search for `_retriever`
     and `AGENTIGRID_RAG` to see where (and only where) RAG is used.
   - `agentigrid/docs/AgentiGrid_RAG_Experiment_Plan.md` — how it is evaluated.

   **Key point to verify:** RAG is generation-stage only — the retriever is never
   called from `validation.py`. `AGENTIGRID_RAG` is the single on/off ablation switch.

## Path B — Build and run it

The full build (ExaGO + PETSc + Ipopt + CoinHSL via Spack, then the app venv) is
documented step by step in `agentigrid/docs/AgentiGrid_Setup.md`. Summary:

1. **Build ExaGO** (Setup.md, phases 1–6). Reserve roughly an hour — it compiles the
   solver stack.
2. **Install the app:**

   ```bash
   cd ExaGO/agentigrid
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .                 # installs the `agentigrid` CLI + dependencies
   ```

3. **Pick a backend:**
   - **Local Ollama** (no API key, free): install Ollama, then
     `ollama pull qwen2.5:7b nomic-embed-text`, and set `OLLAMA_HOST`. Small models
     have lower JSON reliability.
   - **Anthropic Claude** (needs the reviewer's **own** key): copy `.env.example` to
     `agentigrid/.env` and set `ANTHROPIC_API_KEY=...`. `.env` is gitignored — never
     commit a key. Claude gives the most reliable proposals.

4. **Smoke test** (dry run first, then 2 iterations):

   ```bash
   cd ExaGO/agentigrid && source .venv/bin/activate
   agentigrid ./data/case39.m "Reduce total generation cost by 10%" --dry-run
   agentigrid ./data/case39.m "Reduce total generation cost by 10%" --backend ollama --model qwen2.5:7b --max-iter 2 < /dev/null
   ```

5. **See RAG on vs off** (the point of the branch) — build the index first (see
   `RAG_USAGE.md`, Steps 1–2), then:

   ```bash
   GOAL="Reduce total generation cost by 10%"
   AGENTIGRID_RAG=0 agentigrid ./data/case39.m "$GOAL" --backend ollama --model qwen2.5:7b --max-iter 4 < /dev/null
   AGENTIGRID_RAG=1 agentigrid ./data/case39.m "$GOAL" --backend ollama --model qwen2.5:7b --max-iter 4 < /dev/null
   ```

6. **Or use the UI:**

   ```bash
   cd ExaGO/agentigrid && source .venv/bin/activate
   pip install -r launcher/requirements.txt
   ./launcher/run.sh                # or: streamlit run launcher/app.py (from the agentigrid root)
   ```

## What to expect

- Weak local models (e.g. `llama3`) often emit schema-invalid proposals; the UI
  labels these honestly as **"LLM output rejected — …"**. That is expected model
  behavior, not a code fault. `qwen2.5:7b` or Claude produce valid proposals far more
  often.
- If a Claude run returns `HTTP 404 model not found`, check the model tag — the repo
  was pinned off a since-retired tag; the current default is `claude-sonnet-4-6`
  (Setup.md covers the repo-wide rename).
- **No secrets are in the repo.** Each reviewer supplies their own key (Claude) or
  runs Ollama (no key).

## Feedback

Open an issue or PR on the fork, or comment directly on the compare view linked above.
