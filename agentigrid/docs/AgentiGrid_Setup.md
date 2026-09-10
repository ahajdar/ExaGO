# AgentiGrid + ExaGO — Fresh Setup Guide (WSL Ubuntu)

**Goal:** Set up ExaGO in a new directory and run **AgentiGrid** (the LLM-driven simulation/analysis tool that lives in `ExaGO/agentigrid/`) on your Windows laptop, via WSL Ubuntu.

**How the pieces fit** (verified against the repo):
- `agentigrid/` is a **folder inside the ExaGO repo** — not a separate repo. So there is **one checkout** of ExaGO at one branch.
- AgentiGrid is a **Python package** that shells out to ExaGO's compiled **app binaries** (`opflow`, `scopflow`, `tcopflow`, `sopflow`, `dcopflow`, `pflow`) as **CLI subprocesses** (confirmed in its architecture doc). You build ExaGO's apps once, then symlink those binaries into `agentigrid/applications/`.
- AgentiGrid's UI is a **Streamlit launcher** (`agentigrid/launcher/`) that invokes the LLM (Anthropic/OpenAI/Ollama) itself. It does **not** use ExaGO's `pyexago` bindings — so we build ExaGO with `-DEXAGO_ENABLE_PYTHON=OFF` (avoids the `mpi4py` dependency) and lose nothing.
- Dependencies (PETSc, Ipopt, MPI, …) come from **Spack**, which ExaGO bundles as a submodule.

> **BRANCH = `develop`.** The `samimk/sandbox` work is merged into ExaGO's default branch `develop` (which is what "development branch" meant). AgentiGrid is confirmed present on `develop`. Do **not** check out `samimk/sandbox`.

---

## Phase 0. Fresh WSL Ubuntu (clean slate)

You said the old environment can go and you don't need any previous settings. This wipes the old Ubuntu distro and installs a clean one. **All commands in this section run in Windows PowerShell** (not inside Ubuntu).

> ⚠️ **`wsl --unregister` is irreversible.** It deletes *everything* in that Ubuntu distro — all files, packages, and settings, not just ExaGO. You confirmed that's fine.

1. See what distros you currently have:
   ```powershell
   wsl --list --verbose
   ```
2. Remove the old one (replace `Ubuntu` with the exact name from the list above, e.g. `Ubuntu-22.04`):
   ```powershell
   wsl --unregister Ubuntu
   ```
3. Make sure WSL itself is current, then install a fresh Ubuntu 24.04 LTS:
   ```powershell
   wsl --update
   wsl --install -d Ubuntu-24.04
   ```
   This opens the new Ubuntu and asks you to create a **username and password** — these are your Linux account, unrelated to Windows.
4. You're now at an Ubuntu shell. Confirm:
   ```bash
   whoami
   lsb_release -a
   ```

From here on, **everything runs inside this Ubuntu shell**, and all files live in the Linux home (`$HOME`), never on the Windows `/mnt/c/` path.

---

## 0. Variables (set these once per terminal)

```bash
# Where everything goes (WSL path — NOT the Windows Documents folder)
export PROJECT_ROOT="$HOME/projects"          # change if you prefer
export EXAGO_DIR="$PROJECT_ROOT/ExaGO"
export BRANCH="develop"                        # samimk/sandbox is merged into develop
```

---

## 1. Prerequisites (WSL Ubuntu)

```bash
sudo apt update
sudo apt install -y build-essential gfortran git cmake python3 python3-venv python3-pip \
                    curl pkg-config libblas-dev liblapack-dev
```

Check versions (ExaGO needs CMake ≥ 3.18; a recent Python 3):
```bash
cmake --version
python3 --version
gcc --version
```

**GPU note (only if you do the optional GPU section later):** CUDA in WSL2 requires the **NVIDIA driver installed on Windows** (with WSL support) — *not* a driver inside Ubuntu. Verify it's visible from WSL:
```bash
nvidia-smi        # should list your Razer's GPU. If "command not found", GPU build is not ready yet.
```

---

## 2. Clone ExaGO (correct branch) + submodules

```bash
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
git clone https://github.com/ORNL/ExaGO.git
cd "$EXAGO_DIR"

git fetch --all
git checkout "$BRANCH"          # develop

# Confirm AgentiGrid is present on this branch:
ls agentigrid

# Pulls toml11, spdlog, pybind11, and the bundled Spack:
git submodule update --init --recursive
```

---

## 3. Build the dependencies with Spack (PETSc + Ipopt/CoinHSL)

Use the **bundled** Spack (`tpl/spack`) to build the dependency libraries, then (Phase 4) build ExaGO from source with CMake so the app binaries match the `develop` branch.

```bash
cd "$EXAGO_DIR"
source tpl/spack/share/spack/setup-env.sh
spack compiler find
spack compilers        # confirm gcc is listed (e.g. gcc@13.3.0)
```

> **Use CoinHSL, not MUMPS — learned the hard way.** The obvious license-free path (`ipopt+mumps`) builds fine but **ExaGO fails to link**: `libdmumps.so: undefined reference to omp_get_num_threads`. ExaGO does not put OpenMP on its executable link line, and injecting `-fopenmp`/`-lgomp` via CMake flags does **not** stick (ExaGO overrides them). Rebuilding MUMPS `~openmp` didn't clear it either. **CoinHSL is the solver ExaGO's Spack recipe actually requires and CI-tests**, so it links cleanly. It's free for academics (register at https://licences.stfc.ac.uk/product/coin-hsl) — get `coinhsl-2023.11.17.tar.gz`.

> **Compiler note.** Don't append `%gcc@13.3.0` after the spec — in this Spack version the trailing variants bind to `gcc` and it errors (`No such variant ... for spec 'gcc@...'`). gcc is the only compiler; omit it.

### 3a. PETSc (the long build — start it and walk away)
```bash
spack install petsc+mpi+shared~hypre~superlu-dist~mumps ^openmpi
spack find petsc          # confirm ≥ 3.24 (we got 3.24.1)
```
~30–60 min on 12 cores. If PETSc concretizes below 3.24, pin `petsc@3.24:`.

### 3b. Register your CoinHSL tarball in the bundled Spack package
The bundled `coinhsl` package lists `2023.11.17` only as a *remote, un-checksummed* version, so Spack won't concretize it until you add the checksum. Put the tarball in a working dir and patch the package non-interactively:

```bash
mkdir -p ~/coinhsl
cp "/mnt/c/Users/amirh/Documents/Projects/ExaGO/coinhsl-2023.11.17.tar.gz" ~/coinhsl/
cd ~/coinhsl
sha256sum coinhsl-2023.11.17.tar.gz     # note the hash (ours: 43438fb9…9065)

PKG_DIR="$(spack location --package-dir coinhsl)"
python3 - "$PKG_DIR/package.py" <<'PY'
import sys
p = sys.argv[1]
lines = open(p).readlines()
block = [
    '    version(\n',
    '        "2023.11.17",\n',
    '        sha256="43438fb9317dd4648625a6f5dd46ffedf1d33bd47d05885805b651fe93729065",\n',
    '    )\n',
]
for i, ln in enumerate(lines):
    if ln.lstrip().startswith('version(') and len(ln) - len(ln.lstrip()) == 4:
        lines[i:i] = block; break
else:
    raise SystemExit("ERROR: no 4-space version( found")
open(p, 'w').writelines(lines)
PY
grep -n "2023.11.17" "$PKG_DIR/package.py"   # confirm version + sha256 present
```
> The package's download URL is hardcoded to `file://{os.getcwd()}/coinhsl-2023.11.17.tar.gz`, so you **must run the CoinHSL install from `~/coinhsl`** (the dir holding the tarball).

### 3c. Build Ipopt with CoinHSL
```bash
cd ~/coinhsl
spack install ipopt+coinhsl~mumps ^coinhsl@2023.11.17~metis ^openblas
```
Each flag matters:
- `~mumps` — no MUMPS (avoids the OpenMP link failure above).
- `~metis` — CoinHSL 2023.11.17's `meson.build` uses an old Meson API for its Metis header check that Meson 1.8.5 rejects (`compiler.has_header ... include_directories was of type array[str]`). Metis is optional for MA57; skipping it dodges the error. (Fallback if it still trips: add `^meson@1.4`.)
- `^openblas` — forces OpenBLAS. Without it Spack hands CoinHSL Intel MKL and mangles the BLAS arg (`-Dlibblas=mkl_scalapack_lp64`, which isn't BLAS).
- **No `+blas`** — that variant only exists under the *autotools* build (older versions) and requesting it forces autotools, conflicting with `@2023.11.17`. Meson builds link BLAS automatically.
- **No `^openmpi`** — with `~mumps` nothing here needs MPI, so `^openmpi` errors ("not a dependency of any root").

### Capture the dependency locations
```bash
export PETSC_DIR="$(spack location -i petsc)"
export IPOPT_DIR="$(spack location -i "ipopt+coinhsl")"
echo "PETSC_DIR=$PETSC_DIR"
echo "IPOPT_DIR=$IPOPT_DIR"     # note this hash — it's the CoinHSL Ipopt
```

---

## 4. Build the ExaGO apps

> **Critical: clear stray Spack loads first.** ExaGO's CMake prefers what's on `CMAKE_PREFIX_PATH` (i.e. whatever you've `spack load`-ed) **over `IPOPT_DIR`**. If an earlier `spack load ipopt` (e.g. a MUMPS build from a failed attempt) is still active in the shell, ExaGO links *that* Ipopt and you get `libdmumps.so: undefined reference to omp_get_num_threads` even though `IPOPT_DIR` points at the CoinHSL build. So unload everything and load only the correct stack.

```bash
export EXAGO_DIR="$HOME/projects/ExaGO"
cd "$EXAGO_DIR"
source tpl/spack/share/spack/setup-env.sh

spack unload --all                          # drop any stray (MUMPS) ipopt
spack load petsc openmpi
spack load "ipopt ^coinhsl"                 # load the CoinHSL ipopt specifically
                                            # (if ambiguous, use its hash: spack load /<hash>)

echo "--- must show the CoinHSL ipopt and NO mumps ---"
spack find --loaded | grep -iE "ipopt|mumps|coinhsl|openblas"

export PETSC_DIR="$(spack location -i petsc)"
export IPOPT_DIR="$(spack location -i "ipopt+coinhsl")"
echo "IPOPT_DIR=$IPOPT_DIR"

rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXAGO_ENABLE_PETSC=ON \
  -DEXAGO_ENABLE_MPI=ON \
  -DEXAGO_ENABLE_IPOPT=ON \
  -DEXAGO_ENABLE_HIOP=OFF \
  -DEXAGO_ENABLE_GPU=OFF \
  -DEXAGO_ENABLE_PYTHON=OFF \
  -DPETSC_DIR="$PETSC_DIR" \
  -DIPOPT_DIR="$IPOPT_DIR" \
  -DCMAKE_INSTALL_PREFIX="$EXAGO_DIR/install"

make -j12
```

`-DEXAGO_ENABLE_PYTHON=OFF` avoids ExaGO's own `pyexago` bindings (which need `mpi4py`); AgentiGrid doesn't use them.

Locate the built app binaries (path is confirmed during setup — see Phase 5):
```bash
find "$EXAGO_DIR/build" -type f -executable -name "opflow"
```

---

## 5. Wire the ExaGO binaries into AgentiGrid

AgentiGrid looks for the apps in `agentigrid/applications/`. **First find where the build put them** — on this branch it is *not* `build/bin`:

```bash
export BIN_DIR="$(dirname "$(find "$EXAGO_DIR/build" -type f -executable -name opflow | head -1)")"
echo "BIN_DIR=$BIN_DIR"     # must be non-empty and contain the apps
ls "$BIN_DIR"
```

Then symlink from that real location (source of truth stays in the build tree):
```bash
cd "$EXAGO_DIR/agentigrid"
for app in opflow scopflow tcopflow sopflow dcopflow pflow; do
  ln -sf "$BIN_DIR/$app" "applications/$app"
done
ls -l applications/          # arrows must be white/valid, not red/broken
```

*Alternative:* instead of symlinks, set `exago.binary_dir` in the config (Step 6) to the `BIN_DIR` path.

**Populate `data/` with case files.** `agentigrid/data/` ships with only a `README.md`; ExaGO's own `datafiles/` has plenty. Copy a few (note: under `datafiles/test_validation/`, some `.m` names are *directories* — use the single-file ones in `datafiles/` root):
```bash
cd "$EXAGO_DIR/agentigrid"
cp "$EXAGO_DIR/datafiles/case_ACTIVSg200.m"    data/   # 200-bus synthetic case
cp "$EXAGO_DIR/datafiles/case_ACTIVSg200.cont" data/   # its contingency file (SCOPFLOW)
cp "$EXAGO_DIR/datafiles/case39.m"             data/   # small 39-bus case
ls -l data/*.m
```

---

## 6. Install and configure AgentiGrid (Python)

```bash
cd "$EXAGO_DIR/agentigrid"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .          # installs the `agentigrid` CLI + its pinned dependencies
```

### Create the real configs (the repo ships only `.template` files)
`configs/` contains `*.template` files — copy to real names. The default template is already set to `backend: anthropic` and `binary_dir: ./applications`:
```bash
cd "$EXAGO_DIR/agentigrid"
cp configs/default_config.yaml.template configs/default_config.yaml
```

### Generate `env_setup.sh` with the real Spack lib paths
AgentiGrid sources `configs/env_setup.sh` before each ExaGO subprocess. Point it at the actual Spack library dirs so the binaries find CoinHSL/PETSc/OpenBLAS at runtime:
```bash
source "$EXAGO_DIR/tpl/spack/share/spack/setup-env.sh"
COINHSL=$(spack location -i coinhsl); IPOPTD=$(spack location -i "ipopt+coinhsl")
PETSCD=$(spack location -i petsc);   OPENBLASD=$(spack location -i openblas)
OPENMPID=$(spack location -i openmpi)
cat > configs/env_setup.sh <<EOF
#!/bin/bash
export LD_LIBRARY_PATH=$COINHSL/lib:$COINHSL/lib64:$IPOPTD/lib:$PETSCD/lib:$OPENBLASD/lib:$OPENMPID/lib:\$LD_LIBRARY_PATH
EOF
cat configs/env_setup.sh
```

### Set the model and API key
**The default model (`claude-sonnet-4-20250514`) is retired and returns HTTP 404.** It's hardcoded in *several* places — `default_config.yaml`, the LLM backend defaults, **and the Streamlit launcher's auto-populated default** — so fixing only the config leaves the UI broken. Replace it **repo-wide** (skipping the venv):
```bash
grep -rl "claude-sonnet-4-20250514" . 2>/dev/null | grep -v "/.venv/" \
  | xargs sed -i 's/claude-sonnet-4-20250514/claude-sonnet-4-6/g'
grep -rn "claude-sonnet-4-6" . 2>/dev/null | grep -v "/.venv/"   # confirm where it lived
```
`claude-sonnet-4-6` is the model AgentiGrid's docs validated for near-perfect JSON reliability. List what your account actually has if 4-6 isn't available:
```bash
curl -s https://api.anthropic.com/v1/models -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" | python3 -m json.tool | grep '"id"'
```

> **API key ≠ Claude Pro/desktop subscription.** AgentiGrid calls the **Anthropic API**, billed separately with its own credits at console.anthropic.com (Billing). Your Claude.ai/desktop usage does not apply. The key (`sk-ant-api03-…`) is **per-shell** — export it each session or bake it into `env.sh`/`~/.bashrc`:
> ```bash
> export ANTHROPIC_API_KEY="sk-ant-api03-…your real key…"
> echo "${ANTHROPIC_API_KEY:0:14}"    # should print sk-ant-api03-
> ```

> **Free alternative to the API:** AgentiGrid also supports local **Ollama** models — `--backend ollama --model qwen2.5:7b` after `ollama` is installed, no API cost (lower JSON reliability).

(The Streamlit GUI is set up separately in Phase 7¾.)

### 6a. Providing credentials (for you and collaborators)

The real interface is the **environment variable `ANTHROPIC_API_KEY`** — the app reads it from the environment, and nothing more. The *file* you keep it in is just a convenience; a collaborator does **not** need to recreate your exact file, only to get that variable set by whatever mechanism they prefer (a `.env` file, their shell rc, or a secrets manager). Never commit a real key.

The portable convention: a project-local `.env` (gitignored), with a committed `.env.example` showing what to provide.

```bash
cd "$EXAGO_DIR/agentigrid"
cp .env.example .env            # then edit .env and paste your own key
```

`.env` is loaded automatically by `env.sh` (Phase 8) — the `set -a; source .env; set +a` line exports everything in it. So each collaborator does the copy-and-edit once and never touches `env.sh`.

Make sure the real file is ignored and only the example is tracked. Add the rule (the example is force-kept so it stays committable):

```bash
printf '\n# local secrets — never commit\n.env\n!.env.example\n' >> .gitignore
git check-ignore .env          # should print ".env" (i.e. ignored)
git add .env.example .gitignore # the placeholder IS committed
```

Two things to know: the key is **per-user** (each collaborator uses their own; Anthropic billing is per-key), and it's **only needed for the Anthropic backend** — anyone running the local Ollama backend needs no key at all. For a shared or hosted deployment, the same `ANTHROPIC_API_KEY` contract holds; the host just injects it from a secrets manager (Vault, AWS Secrets Manager, a systemd `EnvironmentFile`, or a Docker/K8s secret) instead of a person editing `.env`.

> Already using `~/.agentigrid_secrets`? Keep it — no migration needed. The Phase 8 `env.sh` sources **both** the project `.env` and `~/.agentigrid_secrets` (each `[ -f … ]`-guarded), with your home-dir file sourced **last** so your real key always wins. Your setup is untouched; `.env` + `.env.example` exist purely so collaborators (who won't have your home file) have a documented way in.

---

## 7. Run AgentiGrid

Always activate the venv and have the key set first (`source .venv/bin/activate`; `export ANTHROPIC_API_KEY=…`).

**Step 1 — dry-run** (validates config, `env_setup.sh`, binary resolution, base-case parse; no LLM call):
```bash
agentigrid ./data/case_ACTIVSg200.m "test" --dry-run
```
Ends with `Dry-run complete — exiting.` and prints the resolved config.

**Step 2 — real run** (calls the LLM and runs `opflow` each iteration):
```bash
agentigrid ./data/case_ACTIVSg200.m \
  "Find the maximum uniform load scaling factor before the system becomes infeasible" \
  --max-iter 3
```
A healthy run shows the base case solving, e.g.:
```
[Iter 0] Running: .../applications/opflow -netfile .../case_ACTIVSg200.m -print_output
Simulation succeeded in 0.0s (exit 0)
[Iter 0] Base case: CONVERGED, cost=$27,557.57
```
then the LLM proposing changes and `opflow` re-running each iteration.

Interactive wrapper (prompts for your goal) and positional args also work:
```bash
./run_agentigrid.sh                                                  # prompts interactively
./run_agentigrid.sh configs/default_config.yaml ./data/case39.m 10   # config, case, max-iter
```

Optional test suite:
```bash
python -m pytest tests/ -v
```

> **If every iteration errors `404 not_found_error: model: …`** — the model name is retired. Fix the `model:` line in `configs/default_config.yaml` (Phase 6) or pass `--model claude-sonnet-4-6`. Note the ExaGO side still runs fine in this case (you'll see the base case CONVERGED) — it's purely the LLM call failing.

---

## 7½. Verification & steering (human-in-the-loop)

**What the loop checks for you — and what it doesn't:**

- **Physics / feasibility (auto).** ExaGO *solves* every proposed change and reports `converged` / `infeasible` plus cost and binding limits. The LLM only proposes; the solver decides — so the LLM cannot produce a feasible-looking but false result.
- **Structure (auto).** AgentiGrid validates the LLM's response against its JSON action schema (malformed → `Failed to parse JSON … aborting`) and bounds every run via the config: `timeout`, `max_iterations`, `max_variants`, `boundary_max_mw`, `relief_max_solves`, `contingency_max_count`.
- **Intent (NOT auto).** ExaGO validates that a change is *feasible*, not that it's the *right experiment for your question*. The LLM can misinterpret the goal and solve the wrong question perfectly. For research, review the run rather than trusting it open-loop.

**Audit trail (so you can double-check after the fact):**
```bash
# every iteration's cost / feasibility / actions:
ls workdir/journal_*.json          # save_journal=true, journal_format json|csv
# each iteration's actual modified case (save_modified_files=true):
ls workdir/iter_*/                  # re-run any step yourself:
"$BIN_DIR/opflow" -netfile workdir/iter_000_*/case_ACTIVSg200.m -print_output
```

**Live steering** — type these at the prompt *while a run is active* (from the `[Steering]` banner):
```
<text>            inject a directive — augment mode (adds to current guidance)
replace: <text>   inject a directive — replace mode (overrides current guidance)
pause             pause at the next iteration boundary
resume            resume a paused search
stop              request a graceful stop
save              save the session to disk
status            show current steering state
```
Example — rein in the search mid-run:
```
pause
replace: only scale loads at buses 1–50, keep all generation fixed
resume
```

**Save / resume across runs (CLI flags):**
```bash
agentigrid ./data/case_ACTIVSg200.m "…goal…" --save-on-stop     # write a resumable session on stop
agentigrid --resume workdir/<session_dir>                        # continue a saved session
# verbosity: --verbose  |  --quiet
```

---

## 7¾. Streamlit GUI launcher (web UI)

The launcher (`agentigrid/launcher/`) is a Streamlit web app: pick a case, choose backend/model, run, watch a live monitor, browse results, and export a PDF report.

**Install its dependencies** (it has its own requirements — `streamlit`, `plotly`, `kaleido`, `reportlab`, `pyyaml`):
```bash
cd "$EXAGO_DIR/agentigrid"
source .venv/bin/activate
export ANTHROPIC_API_KEY="sk-ant-api03-…"      # the launching shell must have it
pip install -r launcher/requirements.txt
```

**Run it from the project root** (not from inside `launcher/`, or the `./applications`/`./data`/`./workdir` paths won't resolve):
```bash
./launcher/run.sh                 # or: streamlit run launcher/app.py
```
It prints `Local URL: http://localhost:8501`. Open that in your **Windows** browser — WSL2 forwards `localhost`. (A `gio: … Operation not supported` line just means Streamlit couldn't auto-open a browser inside WSL — harmless. A first-run email prompt → press Enter, or add `--server.headless true`.)

In the sidebar: base case from `data/`, **Backend = Anthropic**, **Model = `claude-sonnet-4-6`** (the repo-wide fix in Phase 6 makes this the default), iterations, then run.

**PDF report needs a headless Chrome** (Kaleido renders the charts into the PDF). The bundled Chrome crashes on WSL for lack of system libraries; installing Google Chrome from the `.deb` pulls all of them in and fixes it:
```bash
cd /tmp
wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install -y ./google-chrome-stable_current_amd64.deb
google-chrome --version
```
Then regenerate the PDF (restart Streamlit if it was already running). The search and the on-screen interactive charts work **without** Chrome — only the PDF's embedded chart images need it.

---

## 7⅞. Local LLM via Ollama on Windows (optional, free)

AgentiGrid supports a local Ollama backend — no API key, no cost. If you already run **Ollama on Windows**, you can reuse it (and its pulled models) from WSL; the only wrinkle is networking, because WSL's `localhost` is *not* the Windows host under default (NAT) networking.

> **Do NOT use `networkingMode=mirrored` to "fix" this** unless you're on Windows 11 22H2+ and it actually works. On unsupported setups it fails with `ConfigureNetworking … 0x8007054f` and **falls back to `networkingMode None`, which leaves WSL with no internet at all**. If that happens: remove the `networkingMode` line from `C:\Users\<you>\.wslconfig`, run `wsl --shutdown`, reopen. The NAT method below needs no `.wslconfig` change.

**1. Expose Windows Ollama and find the host IP.** On **Windows**, set a system env var `OLLAMA_HOST=0.0.0.0` and restart Ollama from the tray (so it listens on all interfaces, not just `127.0.0.1`); allow port `11434` through Windows Firewall. Then from **WSL**:
```bash
WIN_IP=$(ip route show default | awk '{print $3}')
curl -s http://$WIN_IP:11434/api/tags | python3 -m json.tool | grep '"name"'   # your models
```

**2. The host IP changes on reboot — auto-refresh it, don't hardcode.** The `env.sh` below rewrites `ollama_host` in the config each session, so it's always current.

**3. Point AgentiGrid at Ollama and pick a real model tag.** Backend `ollama`; the model must be an **exact** tag you have (a mismatch is `model '…' not found (404)`). Use a **chat** model — not an embedding model (`nomic-embed-text`) or a narrow code model (`deepseek-coder`). Good general choices if present: `mixtral:latest` (strong JSON), `llama3:latest` (lighter, and the natural open-source arm for comparisons). Set it in the sidebar, or:
```bash
cd "$EXAGO_DIR/agentigrid"
sed -i 's#^\( *model:\).*#\1 "mixtral:latest"#' configs/default_config.yaml
# CLI: agentigrid ./data/case39.m "…goal…" --backend ollama --model mixtral:latest --max-iter 3
```
Local models are slower per call and produce more `Failed to parse JSON` retries than Sonnet — keep first runs small.

---

## 8. Every new terminal — re-establish the environment

The Spack loads, `PETSC_DIR/IPOPT_DIR`, venv, and API key are per-shell. Save this as `agentigrid/env.sh` and `source` it each session:

```bash
# env.sh
export EXAGO_DIR="$HOME/projects/ExaGO"
source "$EXAGO_DIR/tpl/spack/share/spack/setup-env.sh"
spack unload --all
spack load petsc openmpi
spack load "ipopt ^coinhsl"        # the CoinHSL ipopt, not a stray MUMPS one
export PETSC_DIR="$(spack location -i petsc)"
export IPOPT_DIR="$(spack location -i "ipopt+coinhsl")"
# keep Windows-Ollama host IP current (NAT networking changes it each session):
WIN_IP=$(ip route show default | awk '{print $3}')
sed -i "s#ollama_host:.*#ollama_host: \"http://$WIN_IP:11434\"#" "$EXAGO_DIR/agentigrid/configs/default_config.yaml"
export OLLAMA_HOST="http://$WIN_IP:11434"   # so the RAG embedder & any CLI shell agree with the config
source "$EXAGO_DIR/agentigrid/.venv/bin/activate"
# API key — sourced from an untracked file, never hardcoded here (see Phase 6a).
# Collaborators use the project .env; a personal ~/.agentigrid_secrets wins if present
# (sourced last, so it overrides a stray placeholder in .env):
[ -f "$EXAGO_DIR/agentigrid/.env" ] && set -a && source "$EXAGO_DIR/agentigrid/.env" && set +a
[ -f ~/.agentigrid_secrets ] && source ~/.agentigrid_secrets
```
```bash
source "$EXAGO_DIR/agentigrid/env.sh"
```

---

## 9. Optional — GPU / CUDA upgrade (do this *after* CPU works)

Only worth it once AgentiGrid runs on the CPU build and `nvidia-smi` works in WSL.

1. Find your GPU's CUDA compute capability (e.g. `86` for Ampere) — check NVIDIA's list for your Razer's GPU.
2. Add the GPU stack in Spack directly (long build; `<XX>` = your compute capability, e.g. `86`):
   ```bash
   spack install hiop+cuda+raja+sparse cuda_arch=<XX> ^openmpi
   spack install magma+cuda cuda_arch=<XX>
   spack load hiop raja umpire magma cuda
   ```
3. Reconfigure ExaGO with GPU on, then rebuild:
   ```bash
   cd "$EXAGO_DIR/build"
   cmake .. \
     -DEXAGO_ENABLE_HIOP=ON -DEXAGO_ENABLE_GPU=ON -DEXAGO_ENABLE_CUDA=ON \
     -DEXAGO_ENABLE_RAJA=ON \
     -DHIOP_DIR="$(spack location -i hiop)" \
     -DRAJA_DIR="$(spack location -i raja)" \
     -Dumpire_DIR="$(spack location -i umpire)" \
     -DMAGMA_DIR="$(spack location -i magma)"
   make -j"$(nproc)"
   ```
   The symlinks in `agentigrid/applications/` still point at `build/bin/`, so no re-wiring needed.

---

## 11. RAG — generation-stage grounding (optional, for the thesis ablation)

Retrieval-augmented generation grounds the LLM's spec/proposal generation in a curated corpus (tool docs, methodology notes, and `request → correct-spec` exemplars), to reduce executable-but-misaligned specs *at the source*. **Design rule: RAG touches only the generation path — never the deterministic validator** (`engine/validation.py`), so verification stays reproducible and the anti-circularity argument holds. A single env var, `AGENTIGRID_RAG`, is the on/off ablation switch.

**Module** (`agentigrid/agentigrid/rag/`, net-new): `embed.py` (Ollama `nomic-embed-text`), `store.py` (persistent Chroma, cosine distance), `retriever.py` (`min_score` threshold, `k`), `ingest.py` (build index from a corpus folder), `__init__.py`.

### 11a. Install + build the index
```bash
AG=~/projects/ExaGO/agentigrid
source "$AG/.venv/bin/activate"
pip install chromadb

mkdir -p "$AG/rag/corpus"
# put curated .txt/.md in rag/corpus (tool help, N-1 methodology, request→spec exemplars). e.g.:
"$AG/applications/opflow" --help > "$AG/rag/corpus/opflow_help.txt" 2>&1 || true

cd "$AG"
export OLLAMA_HOST="http://$(ip route show default | awk '{print $3}'):11434"
rm -rf rag/store                              # cosine space ⇒ rebuild after any store.py/corpus change
python -m agentigrid.rag.ingest rag/corpus    # prints "Ingested N chunks…"
```

### 11b. The two hooks in `engine/agent_loop.py`
- Imports: `from agentigrid.rag import Retriever` (`import os` already present).
- In `__init__`, after `self._backend = create_backend(config.llm)`:
  ```python
          self._retriever = Retriever(
              enabled=os.environ.get("AGENTIGRID_RAG", "0") == "1",
              host=os.environ.get("OLLAMA_HOST") or getattr(config.llm, "ollama_host", None) or "http://localhost:11434",
          )
  ```
  > **Host gotcha (this cost real time):** the embedder must use the **`OLLAMA_HOST` env var**, same as the chat backend. If it falls back to `config.llm.ollama_host` (often still `localhost`), the embed call fails silently and `retrieve()` returns `""` — RAG appears off. Prefer the env var as above.
- Right after the `_assemble_prompt(...)` call (before `complete()`), inject the retrieved block:
  ```python
          _retrieved = self._retriever.retrieve(goal)
          if _retrieved:
              user_prompt = (
                  "=== Section B: Reference Material (retrieved) ===\n"
                  f"{_retrieved}\n\n{user_prompt}"
              )
  ```

### 11c. Run the ablation
```bash
cd "$AG"; export OLLAMA_HOST="http://$(ip route show default | awk '{print $3}'):11434"
GOAL="Assess the security of the system under single-element (N-1) outages"
AGENTIGRID_RAG=0 agentigrid ./data/case39.m "$GOAL" --backend ollama --model llama3:latest --max-iter 1 < /dev/null   # baseline
AGENTIGRID_RAG=1 agentigrid ./data/case39.m "$GOAL" --backend ollama --model llama3:latest --max-iter 1 < /dev/null   # RAG on
```

### 11d. Verifying it's live
- Standalone: `python -c "from agentigrid.rag import Retriever; print(Retriever(enabled=True, host='$OLLAMA_HOST').retrieve('Assess security under single-element outages'))"` → prints your SCOPFLOW/N-1 refs with scores.
- **Do not judge by the `Tokens: N prompt` line** — Ollama caches the (large) system-prompt prefix, so `prompt_eval_count` stays flat even when the injected user content changes. It is *not* evidence RAG is off. Confirm instead with a temporary debug (`'Section B' in user_prompt`) or a corpus-only question whose answer only RAG can supply.

### 11e. Tuning
`retriever.py`: `k` (candidate cap) and `min_score` (cosine floor, default 0.35) together decide how many refs land in the prompt. The retriever asks the store for `k` candidates, then drops any below `min_score`, so it returns **at most `k`**. With a small `k` (e.g. 3) the top matches almost always clear the floor, so you get a **constant `k` every run** — informative of nothing. Raise `k` (≈6) and let `min_score` be the real gate: the count then **varies per query** (a narrow goal retrieves fewer, a broad one more), which is both more honest and lets genuinely relevant refs through. `k` is *not* a magic number — the principled way to set it is an ablation (`k ∈ {1,2,3,5,8}`, measure proposal validity/alignment), which doubles as a thesis experiment. Caveat: more refs = more prompt tokens; a weak local model (llama3 8B) can degrade past ~4–5 refs ("lost in the middle"), so if quality drops, lower `k` or raise `min_score` to ~0.4. Grow/curate the corpus (especially `request → spec` exemplars, and the auto-harvested content from Phase 11g) for sharper hits. `rag/store/` is generated — gitignored; re-`ingest` after corpus or `store.py` changes.

### 11f. Streamlit UI integration (toggle + grounding indicator)
Two user-facing pieces, wired without new plumbing (they reuse `AGENTIGRID_RAG` and the existing `on_phase` callback):

**Toggle** — in `launcher/app.py` `render_sidebar()`, an **🔧 Advanced** expander with a plain-language checkbox (default on; label "Use curated reference knowledge", *not* "RAG"), then set the env the controller reads:
```python
        with st.expander("🔧 Advanced"):
            use_reference_knowledge = st.checkbox("Use curated reference knowledge", value=True, disabled=disabled, help="Ground proposals in a curated knowledge base via retrieval.")
        os.environ["AGENTIGRID_RAG"] = "1" if use_reference_knowledge else "0"
```
Streamlit reruns top-to-bottom, so this sets the flag before Start triggers `start_search()` → the controller reads it at construction. Default-on means every user benefits; the `min_score` threshold keeps it safe when nothing relevant is found. The CLI env var is unaffected (still the ablation control).

**Grounding indicator** — count **and top similarity score**, three edits:
1. `agent_loop.py`, after the injection block, emit count + top score via the existing phase channel (the score is parsed out of the retrieved block's `score X` tokens):
   ```python
           if self._on_phase:
               import re as _re
               _scores = [float(x) for x in _re.findall(r"score (\d+\.\d+)", _retrieved)]
               _top = max(_scores) if _scores else 0.0
               self._on_phase(iteration, f"rag_retrieved:{_retrieved.count('[ref ')}:{_top:.2f}")
   ```
2. `app.py` monitor phase handler — add a branch that parses both fields without overwriting `current_phase`. **Keep the existing `else:` `phase_labels` mapping** — it handles all the normal phases:
   ```python
                   if raw_phase.startswith("rag_retrieved:"):
                       _parts = raw_phase.split(":")
                       try:
                           st.session_state.rag_last_refs = int(_parts[1])
                           st.session_state.rag_top_score = float(_parts[2]) if len(_parts) > 2 else None
                       except (ValueError, IndexError):
                           pass
                   else:
                       # …existing phase_labels mapping (Sending prompt to LLM…, Running simulation…, etc.)…
   ```
3. `app.py`, after the "Search in Progress" header, show it. Guard on `AGENTIGRID_RAG` so it reads "enabled" from the start (before the first retrieval sets a count), then upgrades to count + score:
   ```python
       _rag_on = os.environ.get("AGENTIGRID_RAG") == "1"
       _refs = st.session_state.get("rag_last_refs")
       _top = st.session_state.get("rag_top_score")
       if _rag_on:
           if _refs and _refs > 0:
               _s = f" · top score {_top:.2f}" if _top else ""
               st.caption(f"🔎 Grounding: {_refs} reference(s) retrieved from the knowledge base{_s}")
           else:
               st.caption("🔎 Grounding: enabled (curated reference knowledge)")
   ```
   Reset `st.session_state.rag_last_refs = None` (and `rag_top_score`) in the new-search handler so a prior run's numbers don't linger.

Renders as **"🔎 Grounding: 4 reference(s) retrieved from the knowledge base · top score 0.71"** under the live monitor — both the count and the quality move per query. `on_phase` is `None` on the CLI path, so this is UI-only. Changes to `agent_loop.py`/`app.py` need a full Streamlit restart (Ctrl+C + relaunch), not just a browser refresh — imported modules are cached in `sys.modules`.

> **UI default model:** `launcher/config_builder.py` → `DEFAULT_MODELS["ollama"]` was `qwen2.5:7b` (not pulled locally); set it to a tag you have (`llama3:latest`). Restart Streamlit to pick up the change.

### 11g. Building the corpus from ground truth (`rag/tools/`)

Don't hand-author the knowledge base — harvest it from sources that can't be faked, so every chunk is provenance-tagged and auditable (which matters for the verification argument). Two scripts live in `agentigrid/rag/tools/`:

- **`rag_harvest.py`** — Tier 1. Runs `<app> --help`/`--version` for all six apps and parses the MATPOWER `.m` case files (bus/gen/branch counts), writing `rag/corpus/exago_<app>_help.txt` and `rag/corpus/exago_cases.txt`. Pure ground truth; no LLM, no network. Idempotent — a re-run overwrites only its own `exago_*.txt` files and leaves hand-written corpus files alone.
- **`rag_scrape_journal.py`** — self-bootstrapping exemplars. Reads exported `journal.json` files and pulls out the converged, feasible iterations that carried a real proposal, writing them as few-shot `goal → reasoning → commands → result` chunks (`rag/corpus/exago_exemplars_from_runs.txt`). This is the highest-leverage content for the flat-iterations failure, because that's a format/grounding gap, not a knowledge gap. It mirrors the journal's own `NON_SOLVE_STATUSES`, so it keeps only genuine solves (drops FAILED / ANALYSIS / EXPLORE / SWEEP / CONTINGENCY / COMPLETE and anything with an empty or "No description" goal).

```bash
cd "$AG"                                    # run from the project root (defaults are relative)

# Tier-1: tool + case metadata
python rag/tools/rag_harvest.py

# Exemplars — generate from a CAPABLE model run (claude-sonnet-4-6 or mixtral),
# a weak model produces no valid proposals so there's nothing good to harvest.
python rag/tools/rag_scrape_journal.py --runs-dir workdir --inspect   # preview, writes nothing
python rag/tools/rag_scrape_journal.py --runs-dir workdir             # write

# Rebuild the index (cosine space ⇒ full rebuild)
rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus
```

> If journals aren't under `workdir/`, locate them with `find . -name 'journal*.json'` and pass the parent to `--runs-dir`. **Review `exago_exemplars_from_runs.txt` before ingesting** — a "successful" run doesn't make every proposal in it worth teaching from; delete weak ones. Each chunk is prefixed with a compact `[source: …]` tag so the retriever's `[ref N]` output is self-identifying (a few tokens per chunk, deliberate). Harvested `exago_*.txt` files are regenerable — commit them only if you want the exact corpus reproducible without re-harvesting; otherwise gitignore `rag/corpus/` and keep just the scripts.

---

## Troubleshooting / gotchas

- **`nvidia-smi` not found in WSL** → GPU path not ready; stay on the CPU build. Fix the Windows NVIDIA driver first.
- **Spack concretization fails on PETSc version** → pin `petsc@3.24:` in the spec.
- **`No such variant ... for spec 'gcc@...'`** → you appended `%gcc@13.3.0` and the variants bound to the compiler. Remove `%gcc...`; Spack uses the only compiler automatically.
- **`libdmumps.so: undefined reference to omp_get_num_threads`** (ExaGO link) → the classic one. Two independent causes, both covered by using CoinHSL + clean loads: (a) you're on a MUMPS Ipopt — switch to CoinHSL (Phase 3c); (b) a stray `spack load`-ed MUMPS Ipopt is overriding `IPOPT_DIR` — `spack unload --all` and reload per Phase 4. Confirm with `make test_opflow_functionality VERBOSE=1 2>&1 | tr ' ' '\n' | grep -iE "ipopt|mumps" | sort -u` — the path shown is the Ipopt actually being linked.
- **CoinHSL `No version exists that satisfies coinhsl@2023.11.17`** → the version is un-checksummed in the package; run the Phase 3b patch to add it (or `spack install --no-checksum …`).
- **CoinHSL `meson.build:60 … include_directories was of type array[str]`** → Meson too new for CoinHSL 2023.11.17's Metis check. Build with `~metis` (Phase 3c), or pin `^meson@1.4`.
- **CoinHSL `ipopt requires conflicting variant values '~mumps' and '+mumps'`** → you went through the `exago` recipe, which forces `ipopt~mumps`. Install `ipopt`/`coinhsl` **directly** (Phase 3c).
- **CoinHSL `-Dlibblas=mkl_scalapack_lp64` / BLAS weirdness** → Spack picked Intel MKL. Add `^openblas` to the CoinHSL install (Phase 3c).
- **`openmpi is not a dependency of any root`** → with `~mumps`, Ipopt+CoinHSL needs no MPI; drop `^openmpi` from that install.
- **AgentiGrid can't find an app** → confirm the symlinks in `applications/` resolve (`ls -l`) and aren't broken, or set `exago.binary_dir` in the config.
- **Data file not found** → AgentiGrid expects `.m` case files under `data/`; locate them via `find "$EXAGO_DIR" -name "case_*.m"`.
- **`agentigrid` command not found** → the venv isn't active; `source .venv/bin/activate`.
- **CMake: `Could NOT find mpi4py`** → that's ExaGO's own Python bindings (`interfaces/python`), which AgentiGrid does not use. Add `-DEXAGO_ENABLE_PYTHON=OFF`.
- **Streamlit UI won't start (`streamlit: command not found`)** → `pip install -r launcher/requirements.txt` inside the venv.
- **UI still calls the retired model** (`404 … claude-sonnet-4-20250514`) even after fixing `default_config.yaml` → the launcher has its own hardcoded default; do the **repo-wide** `sed` in Phase 6, or set the model in the sidebar.
- **UI errors trying Ollama / `Connection refused … 11434`** → the sidebar backend defaulted to Ollama; switch it to **Anthropic**.
- **PDF report: `Kaleido requires … Chrome` / `browser seemed to close immediately`** → install Google Chrome via the `.deb` (Phase 7¾) to supply the missing system libs. Non-blocking: the search and on-screen charts work without it.
- **Run the launcher from the project root**, never from inside `launcher/` — otherwise `./applications`, `./data`, `./workdir` don't resolve.
- **WSL lost internet after editing `.wslconfig`** (`ConfigureNetworking … 0x8007054f`, `falling back to networkingMode None`) → mirrored networking isn't supported here; remove the `networkingMode` line, `wsl --shutdown`, reopen.
- **Ollama: `model '…' not found (404)`** → the tag isn't pulled or is misspelled. `curl http://$WIN_IP:11434/api/tags` lists exact tags; use a chat model (`mixtral:latest`, `llama3:latest`), not `nomic-embed-text` (embeddings) or `deepseek-coder` (code-only).
- **Ollama connection works but IP broke after reboot** → NAT gateway IP changed; the `env.sh` auto-refresh line (Phase 8) rewrites `ollama_host` each session.
- **AgentiGrid: `404 not_found_error: model: claude-sonnet-4-20250514`** → the template's default model is retired. Set `claude-sonnet-4-6` (or list yours via the `/v1/models` curl in Phase 6). ExaGO still runs; only the LLM call fails.
- **AgentiGrid: `Anthropic API error 401 / invalid x-api-key`** → placeholder or unset key. `export ANTHROPIC_API_KEY=sk-ant-api03-…` (per shell). The API is billed separately from Claude Pro/desktop.
- **`cp: -r not specified; omitting directory` when copying a case** → under `datafiles/test_validation/`, some `.m` entries are directories. Use the single-file cases in `datafiles/` root (`case_ACTIVSg200.m`, `case39.m`, `case9modalt.m`).
- **`configs/default_config.yaml: No such file`** → the repo ships `.template` files; `cp configs/default_config.yaml.template configs/default_config.yaml` (Phase 6).
- **Don't build in the Windows `Documents\Projects\ExaGO` folder** — build inside the WSL filesystem (`$HOME/...`). Building on the `/mnt/c/...` path is very slow and can break symlinks/permissions.
- **RAG seems off / `RAGDEBUG enabled=True len=0`** → retrieval ran but returned nothing: (a) embedder host is `localhost` not the WIN_IP (use the `OLLAMA_HOST` env in the retriever init, Phase 11b); (b) index empty or in old format — `rm -rf rag/store && python -m agentigrid.rag.ingest rag/corpus`; (c) everything below `min_score` — lower it in `retriever.py`.
- **RAG: `RAGDEBUG enabled=False`** → the retriever disabled itself at construction (missing `chromadb`, or a store path error). `pip install chromadb`; check `rag/store` path.
- **Standalone `Retriever(...).retrieve()` returns `""` but the UI shows references** → the shell's `OLLAMA_HOST` was empty, so the embedder had no host and the query failed silently (`retrieve()` returns `""` on any failure). The UI works because `agent_loop` falls back to `config.llm.ollama_host`. Fix: `export OLLAMA_HOST="http://$WIN_IP:11434"` before testing (now baked into `env.sh`, Phase 8). Verify with `echo $OLLAMA_HOST` — it must not be `http://:11434`.
- **Grounding indicator always shows exactly "3 reference(s)"** → not a bug: that's the retriever's `k=3` cap. It returns *at most* `k` chunks above `min_score`, and a decent corpus almost always has ≥3 that clear the threshold, so you hit the cap every run. The count shows *how many*, not *which* — confirm retrieval is query-sensitive by comparing the `[ref …]` **content/scores** across different goals (Phase 11d), not the count. To make the indicator informative, show the top score or distinct-source count instead of the raw number.
- **RAG on/off shows identical `Tokens: N prompt`** → expected — Ollama caches the system-prompt prefix, so that count doesn't reflect the injected user content. Confirm RAG via the standalone retrieve or a `'Section B' in user_prompt` debug (Phase 11d), not token counts.
- **Every iteration is identical (same cost, e.g. `$41,864.18` each, "No description")** → **not** an infinite loop. The run is bounded by `--max-iterations` and *is* progressing (it reaches "Iteration N: Parsing results…"); the problem is that the LLM never emits a valid modification proposal, so each iteration re-runs the *unchanged* base case. Cause: a weak local model with low structured-output (JSON) reliability — `llama3:latest` (8B) and other small local models routinely fail here. Fix: use `claude-sonnet-4-6` (Anthropic) or `mixtral:latest` for real iteration progress; keep small local models only for the RAG-embedding path, not for the planning LLM. Confirm the parse failures from the CLI: `agentigrid ... --model llama3:latest ... 2>&1 | grep -iE 'parse|json|proposal'`. This is a model-quality limit, unrelated to the RAG/UI changes.

---

### Decisions (settled)
1. **Branch** — `develop` (samimk/sandbox merged in). ✓
2. **LLM provider** — Anthropic (`ANTHROPIC_API_KEY`) as primary; local **Ollama on Windows** (`mixtral:latest`/`llama3:latest`) available free via NAT + IP auto-refresh (Phase 7⅞). ✓
3. **Linear solver** — Ipopt + **CoinHSL** (academic license, `coinhsl-2023.11.17`), built Meson/`~metis`/`^openblas`. MUMPS was tried first and abandoned — ExaGO won't link its OpenMP symbols. ✓
4. **Build target** — CPU/Ipopt. GPU/CUDA is an optional add-on (Phase 9) and not attempted yet.
5. **Machine** — clean WSL Ubuntu 24.04, user `ahajdar`, 12 cores (`make -j12`). ✓
6. **Verified versions** — PETSc 3.24.1, Ipopt 3.14.14, CoinHSL 2023.11.17, OpenBLAS 0.3.30, gcc 13.3.0, CMake 3.28.3. ✓
7. **LLM model** — `claude-sonnet-4-6` (template default `claude-sonnet-4-20250514` is retired → 404). ✓
8. **Status** — ExaGO builds; all 6 apps run; `opflow` base case CONVERGES via AgentiGrid. **CLI and Streamlit UI both operational** (model fixed repo-wide; PDF export via Google Chrome for Kaleido). ✓
9. **RAG (generation-stage)** — `agentigrid/rag/` (Chroma + Ollama `nomic-embed-text`), hooked into `agent_loop.py`, `AGENTIGRID_RAG` ablation switch, verifier untouched. **UI**: Advanced "Use curated reference knowledge" toggle (default on) + live "🔎 Grounding: N references" indicator. **Corpus auto-built from ground truth** via `rag/tools/` (`rag_harvest.py` = app `--help` + case metadata; `rag_scrape_journal.py` = self-bootstrapping exemplars from successful runs) — Phase 11g. Live and confirmed (Phase 11). ✓ Work is on branch `amir/agentigrid-rag`.
