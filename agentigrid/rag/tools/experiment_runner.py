#!/usr/bin/env python3
"""Phase-0 experiment RUNNER for the AgentiGrid RAG comparison.

Executes the experiment matrix (case × goal × condition × model × repetition),
toggling the RAG ablation via environment variables, and captures each run's
exported journal JSON + a manifest. It changes nothing about AgentiGrid; it only
orchestrates repeated CLI invocations and collects their outputs.

Pairing runs to journals: AgentiGrid writes workdir/journal_<timestamp>.json.
This script snapshots workdir before each run and grabs the newly created file
after — so runs MUST be sequential (they are here).

Usage
-----
    # 1. write an example spec you can edit:
    python rag/tools/experiment_runner.py --init-spec experiment_spec.json

    # 2. run it (from the AgentiGrid project root, venv active, OLLAMA_HOST set):
    python rag/tools/experiment_runner.py --spec experiment_spec.json

    # dry-run (print the commands, execute nothing):
    python rag/tools/experiment_runner.py --spec experiment_spec.json --dry-run

Notes
-----
* Run from the project root so ./data, ./applications and ./workdir resolve.
* The current environment is inherited (venv PATH, OLLAMA_HOST, Spack libs),
  then each condition's `env` is overlaid — that's how C0/C1 set AGENTIGRID_RAG.
* AgentiGrid's CLI (as of this branch) exposes no --seed/--temperature, so
  stochasticity is captured by repetitions, not seed control. If seed/temp
  control is added later, put it in each model entry's `extra_args`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EXAMPLE_SPEC = {
    "agentigrid_cmd": "agentigrid",
    "project_root": ".",
    "workdir": "workdir",
    "out_dir": "experiments/run1",
    "reps": 3,
    "max_iter": 4,
    "timeout_s": 1200,
    "skip_existing": True,
    "cases": [
        {"name": "case39", "path": "./data/case39.m", "app": "opflow"},
    ],
    "goals": [
        {"id": "cost10", "text": "Reduce total generation cost by 10%", "target_pct": 10},
        {"id": "loadmax", "text": "Find the maximum uniform load scaling factor before infeasible"},
    ],
    "conditions": [
        {"id": "C0-norag", "env": {"AGENTIGRID_RAG": "0"}},
        {"id": "C1-basic", "env": {"AGENTIGRID_RAG": "1"}},
    ],
    "models": [
        {"backend": "ollama", "model": "gemma4:26b", "extra_args": []},
    ],
}


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def build_cmd(spec, case, goal, model):
    cmd = [spec.get("agentigrid_cmd", "agentigrid"), case["path"], goal["text"]]
    cmd += ["--backend", model["backend"], "--model", model["model"]]
    if case.get("app"):
        cmd += ["--app", case["app"]]
    cmd += ["--max-iter", str(spec.get("max_iter", 4))]
    cmd += list(model.get("extra_args", []))
    return cmd


def newest_new_journal(workdir: Path, before: set[str]) -> Path | None:
    after = set(glob.glob(str(workdir / "journal_*.json")))
    fresh = [Path(p) for p in (after - before)]
    if not fresh:
        return None
    return max(fresh, key=lambda p: p.stat().st_mtime)


def run_one(spec, case, goal, cond, model, rep, out_dir, project_root, workdir, dry):
    run_id = "__".join([
        sanitize(case["name"]), sanitize(goal["id"]), sanitize(cond["id"]),
        sanitize(model["model"]), f"r{rep}",
    ])
    run_dir = out_dir / run_id
    manifest_path = run_dir / "manifest.json"

    if spec.get("skip_existing", True) and manifest_path.exists() and not dry:
        try:
            m = json.loads(manifest_path.read_text())
            if m.get("status") == "ok":
                print(f"  [skip] {run_id} (already ok)")
                return m
        except Exception:
            pass

    cmd = build_cmd(spec, case, goal, model)
    env = dict(os.environ)
    for k, v in cond.get("env", {}).items():
        env[k] = str(v)

    manifest = {
        "run_id": run_id,
        "case": case["name"], "case_path": case["path"], "app": case.get("app"),
        "goal_id": goal["id"], "goal_text": goal["text"], "target_pct": goal.get("target_pct"),
        "condition": cond["id"], "condition_env": cond.get("env", {}),
        "backend": model["backend"], "model": model["model"],
        "max_iter": spec.get("max_iter", 4), "rep": rep,
        "cmd": cmd, "cwd": str(project_root),
    }

    if dry:
        print(f"  [dry] {run_id}: {' '.join(cmd)}   (env: {cond.get('env', {})})")
        return manifest

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    before = set(glob.glob(str(workdir / "journal_*.json")))
    start = time.time()
    timed_out = False
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(
                cmd, cwd=str(project_root), env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                timeout=spec.get("timeout_s", 1200), check=False,
            )
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        exit_code, timed_out = 124, True
    wall = round(time.time() - start, 1)

    journal_src = newest_new_journal(workdir, before)
    journal_dst = None
    if journal_src is not None:
        journal_dst = run_dir / "journal.json"
        shutil.copy2(journal_src, journal_dst)

    manifest.update({
        "start_ts": datetime.fromtimestamp(start).isoformat(),
        "wall_s": wall, "exit_code": exit_code, "timed_out": timed_out,
        "journal_src": str(journal_src) if journal_src else None,
        "journal_file": str(journal_dst) if journal_dst else None,
        "log_file": str(log_path),
        "status": "ok" if (exit_code == 0 and journal_dst) else "failed",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    flag = "ok " if manifest["status"] == "ok" else "FAIL"
    print(f"  [{flag}] {run_id}  ({wall}s, exit={exit_code}, journal={'yes' if journal_dst else 'NONE'})")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", help="experiment spec JSON")
    ap.add_argument("--init-spec", metavar="PATH", help="write an example spec and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands, execute nothing")
    args = ap.parse_args()

    if args.init_spec:
        Path(args.init_spec).write_text(json.dumps(EXAMPLE_SPEC, indent=2))
        print(f"Example spec written to {args.init_spec} — edit cases/goals/conditions/models, then --spec it.")
        return 0
    if not args.spec:
        ap.error("provide --spec SPEC.json (or --init-spec PATH to scaffold one)")

    spec = json.loads(Path(args.spec).read_text())
    project_root = Path(spec.get("project_root", ".")).resolve()
    workdir = (project_root / spec.get("workdir", "workdir"))
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = (project_root / spec.get("out_dir", "experiments/run1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for case in spec["cases"]:
        for goal in spec["goals"]:
            for cond in spec["conditions"]:
                for model in spec["models"]:
                    for rep in range(1, int(spec.get("reps", 1)) + 1):
                        tasks.append((case, goal, cond, model, rep))

    print(f"{'DRY-RUN: ' if args.dry_run else ''}{len(tasks)} runs "
          f"({len(spec['cases'])} cases × {len(spec['goals'])} goals × "
          f"{len(spec['conditions'])} conditions × {len(spec['models'])} models × {spec.get('reps',1)} reps)")
    print(f"project_root={project_root}\nout_dir={out_dir}\n")

    index_path = out_dir / "runs_index.jsonl"
    n_ok = 0
    idx = open(index_path, "a") if not args.dry_run else None
    try:
        for i, (case, goal, cond, model, rep) in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}]", end=" ")
            m = run_one(spec, case, goal, cond, model, rep, out_dir, project_root, workdir, args.dry_run)
            if idx is not None:
                idx.write(json.dumps(m) + "\n")
                idx.flush()
                if m.get("status") == "ok":
                    n_ok += 1
    finally:
        if idx is not None:
            idx.close()

    if not args.dry_run:
        print(f"\nDone. {n_ok}/{len(tasks)} runs ok. Results under {out_dir}/")
        print(f"Next: python rag/tools/experiment_eval.py --runs {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
