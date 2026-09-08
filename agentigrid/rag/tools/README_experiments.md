# Phase-0 experiment harness

Two dependency-free scripts (stdlib only) that run and measure the RAG comparison.
They change nothing about AgentiGrid — they orchestrate repeated CLI runs and
aggregate the journal JSONs each run already exports.

Put both in `agentigrid/rag/tools/` and run **from the project root** (venv active,
`OLLAMA_HOST` set), so `./data`, `./applications`, `./workdir` resolve.

## 1. Runner — `experiment_runner.py`

Executes the matrix `case × goal × condition × model × repetition`, overlaying each
condition's environment (that's how `AGENTIGRID_RAG` is toggled), and captures per run:
`manifest.json` (what was run) + `journal.json` (the copied export) + `run.log`.

```bash
# scaffold an editable spec
python rag/tools/experiment_runner.py --init-spec experiment_spec.json

# preview (no execution)
python rag/tools/experiment_runner.py --spec experiment_spec.json --dry-run

# run for real
python rag/tools/experiment_runner.py --spec experiment_spec.json
```

**Spec fields:** `cases` (name/path/app), `goals` (id/text/optional `target_pct`),
`conditions` (id + `env` dict, e.g. `{"AGENTIGRID_RAG":"1"}`), `models`
(backend/model/optional `extra_args`), plus `reps`, `max_iter`, `timeout_s`,
`out_dir`, `project_root`, `skip_existing`.

Add a new RAG variant later by adding a condition whose `env` sets whatever switch
that variant reads (e.g. a future `AGENTIGRID_RAG_MODE=corrective`) — no runner change.

> AgentiGrid's CLI exposes no `--seed`/`--temperature`, so repetitions capture
> stochasticity. If seed/temp control is added, put the flags in a model's
> `extra_args` and bump `reps` accordingly.

## 2. Evaluator — `experiment_eval.py`

Reads the runner's output and writes `analysis/per_run.csv` (one row per run) and
`analysis/summary.csv` (one row per cell, mean + 95% CI over reps).

```bash
python rag/tools/experiment_eval.py --runs experiments/run1
# or aggregate bare journals with no manifests:
python rag/tools/experiment_eval.py --journals-glob 'workdir/journal_*.json'
```

**Metrics (Tier-A, from the ExaGO oracle — no verifier needed):**
- `cost_improvement_pct` = (base − best feasible) / base × 100.
- `valid_proposal_rate` = iterations (past base) that applied a real modification ÷ attempts (`max_iter`). Directly quantifies the flat-iteration failure.
- `any_feasible`, `goal_attained` (if the goal has `target_pct`), `n_solve_iters`, `solve_elapsed_s`, `wall_s`.

Metric definitions are conservative and documented inline — refine as the study
matures. The `verifier verdict` (Tier-B, your thesis metric) is **not** computed
here yet; when the verifier lands, add its per-proposal verdict to the journal and
extend `metrics_for()` — the same runs are reused, no re-runs.

## Suggested first experiment
Start tiny to validate the loop, then scale:
- 1 case (`case39`), 2 goals (one cost, one topology), conditions `C0-norag` + `C1-basic`, model `gemma4:26b`, `reps: 3`.
- Confirm `C1 > C0` on `valid_proposal_rate` before implementing CRAG / Graph RAG.
