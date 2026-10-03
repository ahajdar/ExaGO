# AgentiGrid Tests

Tests are grouped by the part of AgentiGrid they cover:

| Folder | Covers |
|---|---|
| `llm/` | LLM backends and JSON extraction (`agentigrid/backends/`) |
| `grid_files/` | MATPOWER reader, writer and network summary (`parsers/matpower_*`, `network_*`) |
| `commands/` | Grid-change commands, validation and the modifier (`engine/commands.py`, `modifier.py`) |
| `runner/` | Running simulation binaries and logging their invocations (`engine/executor.py`) |
| `results/` | Result readers and summaries for the LLM (`parsers/*_parser.py`, `*_summary.py`) |
| `exago/` | Support for individual ExaGO applications (DCOPFLOW, PFLOW, SCOPFLOW, TCOPFLOW, SOPFLOW) |
| `agent_loop/` | Agent loop controller and interactive steering (`engine/agent_loop.py`) |
| `journal/` | Search journal, objectives, Pareto front, session save and report regeneration |
| `analyses/` | Explore, sweeps, boundary search, contingency, relief, reserve and topology |
| `e2e/` | End-to-end runs through the agent loop |
| `rag/` | Retrieval, corrective-RAG graders, corpus tools, retrieval preview, precision@k and relevance labels (`agentigrid/rag/`, `rag/tools/`) |
| `evaluation/` | Experiment runner, evaluator, statistics, goal attainment and base-case pre-checks (`rag/tools/experiment_*`, `precheck_cases.py`) |
| `fixtures/` | Shared sample files (e.g. `sample_opflow_output.txt`) |

Run the fast tests from the AgentiGrid root:

```bash
python -m pytest -m "not slow"
```

Tests marked `slow` call a real LLM API and need an API key.
