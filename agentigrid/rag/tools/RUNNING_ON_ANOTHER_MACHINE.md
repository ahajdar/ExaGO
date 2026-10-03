# Running the experiment on another computer

> **Layout.** This version uses the per-tool layout from ORNL #133 (`applications/exago/`, `data/exago/`). The paper-1 of-record runs use the tag `paper1-of-record`, which still has the old layout (`applications/`, `data/`); follow that tag's copy of this file for those runs.

This covers setting up a second machine to run, or to continue, the of-record campaign so that its runs are equivalent to those from the first machine. It complements `EVALUATION_RUNBOOK.md` (setup details, gotchas) and `docs/AgentiGrid_Setup.md` (building ExaGO).

## Setup (once per machine)

1. **Linux or WSL Ubuntu with enough memory.** One SCOPFLOW solve reached about 7.3 GB and was killed by the out-of-memory killer. Give WSL at least 16 GB plus swap in `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   memory=24GB
   swap=16GB
   ```
   Leave Windows 6–8 GB for itself and Ollama. Apply with `wsl --shutdown`, then `wsl`, and check with `free -h`.
2. **Build ExaGO** as in `docs/AgentiGrid_Setup.md`: Spack builds PETSc and Ipopt with CoinHSL, CMake builds the apps, and you symlink the binaries into `agentigrid/applications/exago/`. PETSc takes hours. The CoinHSL tarball is licensed, so copy it from the first machine.
3. **Get the code at the same commit as the first machine:**
   ```bash
   git clone -b amir/agentigrid-rag <fork URL> ExaGO
   cd ExaGO && git remote rename origin myfork
   git log --oneline -1          # must match the first machine's top commit
   ```
4. **Python environment:**
   ```bash
   cd agentigrid && python3 -m venv .venv && . .venv/bin/activate
   pip install -e . && pip install -r requirements.txt chromadb
   ```
5. **Local settings.** Copy `configs/default_config.yaml` and `env.sh` from the first machine. Adjust `exago.binary_dir` and `llm.ollama_host`, and keep `exago.mpi_np: 1`. Then:
   ```bash
   # link ExaGO's example data into data/exago/examples/ (commands in data/exago/README.md)
   ```
6. **Ollama.** Install it, then pull the models:
   ```bash
   ollama pull nomic-embed-text && ollama pull llama3 && ollama pull qwen2.5:7b
   ```
7. **Rebuild the vector store from the frozen corpus** (runbook section 2):
   ```bash
   export OLLAMA_HOST=http://<ollama address>:11434
   python -m agentigrid.rag.ingest rag/corpus
   python -c "from agentigrid.rag.corpus_hash import corpus_status as s; print(s('rag/corpus','rag/store'))"
   ```
   The last command must print `'ok': True`, with the same `corpus_sha256` as `rag/corpus/corpus_manifest.json`.
8. **Check that retrieval matches the first machine:**
   ```bash
   python rag/tools/retrieval_preview.py --spec grader_ablation_spec.json --out experiments/retrieval_preview_machine2.json
   ```
   Each goal's context hash must equal the one in the first machine's `experiments/retrieval_preview.json`.
9. **Set the API key and launch** from a plain terminal, not VS Code:
   ```bash
   export ANTHROPIC_API_KEY=...
   nohup python -u rag/tools/experiment_runner.py --spec grader_ablation_spec.json > ofrecord.log 2>&1 &
   ```

## Validity: make sure the other machine gives the same results

- **Same local models.** A tag such as `llama3:latest` can point to different weights if it was pulled on a different day. Run `ollama list` on both machines and compare the IDs. If they differ, copy the models across (Ollama's models directory) or pin an explicit tag on both machines. Otherwise the "same" model is not the same model.
- **Don't split one model's comparisons across machines.** The tests pair C0, C1 and C2a runs of the same model, goal and repetition. Keep every run of a given model on one machine. The clean split is by model: for example, Sonnet on one computer and the two local models on another. Machine differences then affect only time and cost, which are reported per machine, and not the comparisons.
- **Record the machine.** Note which machine ran which models, with each machine's `ollama list` output, in the experiment notes.

## Continuing a campaign that's already started

Copy `experiments/ofrecord_v2/` from the first machine. The runner skips every run already marked `ok` and continues with the rest.

## Splitting by model

Make one spec per machine that keeps only that machine's models and the same `out_dir`:

```bash
python - <<'EOF'
import json; s = json.load(open("grader_ablation_spec.json"))
s["models"] = [m for m in s["models"] if m["backend"] == "anthropic"]   # machine 2: != "anthropic"
json.dump(s, open("ofrecord_hosted.json", "w"), indent=2)
EOF
nohup python -u rag/tools/experiment_runner.py --spec ofrecord_hosted.json > ofrecord_hosted.log 2>&1 &
```

Afterwards, copy each machine's run folders into one `experiments/ofrecord_v2/`. The folder names don't collide, because they include the model. Then run `experiment_eval.py` and `experiment_stats.py` on the merged folder, exactly as for a single-machine campaign.
