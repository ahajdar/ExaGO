#!/usr/bin/env python3
"""Retrieval preview: exactly what each RAG condition injects, per goal, with no
LLM and no solver.

The agent loop retrieves with the goal text as the query (agent_loop:
``self._retriever.retrieve(goal)``), and every mode is deterministic for a fixed
store. So for a given (goal, condition) the injected reference block is the same
in every iteration and every repetition. That makes the effect of the grader
decidable before any agent run:

  * a corrective grader acts only through its verdict on the top-k hits
    (correct / ambiguous -> refined context, ambiguous adds a hedge line;
    incorrect -> one keyword rewrite, else context withheld); strip refinement
    is cosine-based whatever the grader;
  * if two conditions inject byte-identical context for a goal, their agent runs
    on that goal differ only by sampling noise.

For every evaluated goal and every retrieval condition in the spec this prints
the verdict path and whether the context equals that of the other conditions,
and writes the full context blocks to a JSON file (paper appendix / audit).

Run from the agentigrid project root, with the frozen store ingested and the
same OLLAMA_HOST the runs use (C2b also needs sentence-transformers):
    python rag/tools/retrieval_preview.py --spec grader_ablation_spec.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

RAG_ENV_KEYS = ("AGENTIGRID_RAG", "AGENTIGRID_RAG_MODE", "AGENTIGRID_CRAG_GRADER",
                "AGENTIGRID_CRAG_TAU_LOWER", "AGENTIGRID_CRAG_TAU_UPPER",
                "AGENTIGRID_RAG_STORE", "AGENTIGRID_RAG_COLLECTION",
                "AGENTIGRID_RERANKER_MODEL", "AGENTIGRID_RERANKER_DEVICE")


def _retrieval_active(env: dict) -> bool:
    mode = str(env.get("AGENTIGRID_RAG_MODE", "")).strip().lower()
    if mode:
        return mode != "off"
    return str(env.get("AGENTIGRID_RAG", "0")) == "1"


def _with_env(env: dict):
    """Context manager: the process env holds exactly this condition's RAG vars."""
    class _Ctx:
        def __enter__(self):
            self.saved = {k: os.environ.get(k) for k in RAG_ENV_KEYS}
            for k in RAG_ENV_KEYS:
                os.environ.pop(k, None)
            for k, v in env.items():
                if k in RAG_ENV_KEYS:
                    os.environ[k] = str(v)
            return self

        def __exit__(self, *exc):
            for k, v in self.saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _Ctx()


def _verdict(before: dict, after: dict) -> str:
    """Verdict path of one corrective retrieve() call from its stats delta."""
    d = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    if d.get("rewrites"):          # incorrect, then the verdict on the rewritten query
        second = next((v for v in ("correct", "ambiguous") if d.get(v)), "incorrect")
        s = f"incorrect > rewrite > {second}"
    else:
        s = next((v for v in ("correct", "ambiguous", "incorrect") if d.get(v)), "-")
    if d.get("withheld"):
        s += " (withheld)"
    if d.get("grader_fallbacks"):
        s += " (GRADER FELL BACK TO COSINE)"
    return s


def preview(spec: dict, host: str, build=None) -> dict:
    if build is None:
        from agentigrid.rag import build_retriever as build
    conds = [dict(c, env={**spec.get("env", {}), **c.get("env", {})}) for c in spec.get("conditions", [])]
    conds = [c for c in conds if _retrieval_active(c["env"])]
    goals = spec.get("goals", [])
    out = {"created": datetime.now().isoformat(), "host": host,
           "conditions": [c["id"] for c in conds], "goals": {}}
    for c in conds:
        with _with_env(c["env"]):
            r = build(host=host)
            for g in goals:
                stats = getattr(r, "stats", None)
                before = dict(stats) if isinstance(stats, dict) else None
                ctx = r.retrieve(g["text"]) if getattr(r, "enabled", False) else ""
                entry = {"context": ctx,
                         "sha256": hashlib.sha256(ctx.encode("utf-8")).hexdigest()[:12] if ctx else None,
                         "refs": ctx.count("[ref "),
                         "verdict": _verdict(before, dict(r.stats)) if before is not None else "basic"}
                out["goals"].setdefault(g["id"], {})[c["id"]] = entry
    # pairwise identity per goal
    ids = out["conditions"]
    for gid, per in out["goals"].items():
        same = {}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                same[f"{a} == {b}"] = per[a]["context"] == per[b]["context"]
        per["_identical"] = same
    return out


def report(res: dict) -> str:
    ids = res["conditions"]
    lines = []
    for gid, per in res["goals"].items():
        lines.append(f"== {gid}")
        for cid in ids:
            e = per[cid]
            lines.append(f"   {cid:<20} refs={e['refs']}  context={e['sha256'] or 'EMPTY':<12}  verdict={e['verdict']}")
        diff = [k for k, v in per["_identical"].items() if not v]
        same = [k for k, v in per["_identical"].items() if v]
        if same:
            lines.append("   identical: " + "; ".join(same))
        if diff:
            lines.append("   differ:    " + "; ".join(diff))
    pairs = {}
    for per in res["goals"].values():
        for k, v in per["_identical"].items():
            pairs.setdefault(k, []).append(v)
    lines.append("")
    lines.append("Summary (goals with byte-identical injected context):")
    for k, vs in pairs.items():
        lines.append(f"   {k:<45} {sum(vs)}/{len(vs)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="grader_ablation_spec.json")
    ap.add_argument("--out", default="experiments/retrieval_preview.json")
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    res = preview(spec, host)
    res["spec"] = args.spec
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(report(res))
    print(f"\nFull context blocks -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
