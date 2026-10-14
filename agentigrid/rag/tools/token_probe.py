#!/usr/bin/env python3
"""Replay one finished run and measure, for every LLM call, how many prompt
tokens Ollama evaluates with the request's own options (as the run sent them)
and with a large context (the true prompt length in that model's tokenizer).

If default_eval < full_eval, the model did not see the whole prompt: Ollama
keeps the first few tokens and the last half-window of an over-long prompt.

    python rag/tools/token_probe.py <run_dir> [--ctx 32768] [--out experiments/token_probe]

Run from the agentigrid root with the campaign venv and OLLAMA_HOST set. The
probe calls use num_predict=1; the run itself proceeds unchanged. Each call
reloads the model (the context size changes), so a probed run is slow.
"""
import argparse
import json
import os
from pathlib import Path

import ollama

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir")
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--out", default="experiments/token_probe")
a = ap.parse_args()
m = json.load(open(Path(a.run_dir) / "manifest.json"))
for k, v in {**(m.get("spec_env") or {}), **(m.get("condition_env") or {})}.items():
    os.environ[k] = str(v)
rows, orig = [], ollama.Client.chat


def chat(self, *args, **kw):
    msgs, opts = kw.get("messages"), dict(kw.get("options") or {})
    probe = {k: v for k, v in kw.items() if k != "format"}
    r0 = orig(self, *args, **{**probe, "options": {**opts, "num_predict": 1}})
    r1 = orig(self, *args, **{**probe, "options": {**opts, "num_predict": 1, "num_ctx": a.ctx}})
    row = {"call": len(rows) + 1, "num_ctx_sent": opts.get("num_ctx"),
           "default_eval": r0.get("prompt_eval_count"), "full_eval": r1.get("prompt_eval_count"),
           "system_chars": len(msgs[0]["content"]), "user_chars": len(msgs[1]["content"])}
    rows.append(row)
    print("[token_probe]", row, flush=True)
    return orig(self, *args, **kw)          # the real call, unchanged


ollama.Client.chat = chat
from agentigrid.cli import main  # noqa: E402

try:
    main(list(m["cmd"][1:]))
finally:
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = Path(a.out) / f"{m['run_id']}.json"
    out.write_text(json.dumps({"run_id": m["run_id"], "model": m["model"], "ctx": a.ctx, "calls": rows}, indent=2))
    print("wrote", out)
