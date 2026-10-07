#!/usr/bin/env python3
"""Rule-based curation of run exemplars (final corpus, 2026-10-08).

Two fixed rules, decided before the of-record runs and applied without looking at
the evaluated goals' retrieval results:

  R1  Drop a worked example whose commands change bus voltage limits
      (set_all_bus_vlimits / set_bus_vlimits) while its goal forbids changing or
      relaxing voltage limits. Such an example teaches the model to touch the
      voltage band on tasks where the band must stay as it is.
  R2  Keep one example per (source network, app, step). Copies that differ only in
      the wording of the goal are dropped; the copy with the longest goal text
      (the stricter wording) is kept, the first one on a tie.

Deterministic and idempotent:

    python rag/tools/curate_exemplars.py rag/corpus/exago_exemplars_from_runs.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_VLIMIT_STEP = re.compile(r"\bset_(?:all_)?bus_vlimits\b")
_FORBIDS_VLIMIT_CHANGE = re.compile(
    r"without[^.]*\b(?:relaxing|changing)\b[^.]*\bvoltage limits\b", re.I)
_TAG_FIELD = r"\| {}: ([^|\]]+)"


def _field(tag: str, name: str) -> str:
    m = re.search(_TAG_FIELD.format(name), tag)
    return m.group(1).strip() if m else ""


def _line(lines: list[str], prefix: str) -> str:
    return next((l[len(prefix):] for l in lines if l.startswith(prefix)), "")


def parse(block: str) -> dict:
    lines = block.strip("\n").split("\n")
    return {"text": "\n".join(lines), "network": _field(lines[0], "network"),
            "app": _field(lines[0], "app"), "goal": _line(lines, "Goal: "),
            "step": _line(lines, "Step applied: ")}


def r1_violates(ex: dict) -> bool:
    return bool(_VLIMIT_STEP.search(ex["step"]) and _FORBIDS_VLIMIT_CHANGE.search(ex["goal"]))


def curate(blocks: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (kept blocks in original order, [(rule, first line of dropped block)])."""
    exs = [parse(b) for b in blocks]
    dropped: list[tuple[str, str]] = []
    keep = []
    for ex in exs:
        if r1_violates(ex):
            dropped.append(("R1", ex["goal"][:70]))
        else:
            keep.append(ex)
    best: dict[tuple, dict] = {}
    for ex in keep:
        key = (ex["network"], ex["app"], ex["step"])
        if key not in best or len(ex["goal"]) > len(best[key]["goal"]):
            best[key] = ex
    out = []
    for ex in keep:
        if best[(ex["network"], ex["app"], ex["step"])] is ex:
            out.append(ex["text"])
        else:
            dropped.append(("R2", f'{ex["network"]} {ex["app"]}: {ex["step"][:50]}'))
    return out, dropped


def curate_file(path: Path) -> tuple[int, int, list[tuple[str, str]]]:
    blocks = [b for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    kept, dropped = curate(blocks)
    path.write_text("\n\n".join(kept) + "\n", encoding="utf-8")
    return len(blocks), len(kept), dropped


def main(argv: list[str] | None = None) -> int:
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        n, k, dropped = curate_file(p)
        print(f"{p}: {n} -> {k} exemplar(s)")
        for rule, what in dropped:
            print(f"  dropped ({rule}): {what}")
    print("Then: python rag/tools/corpus_guard.py --spec grader_ablation_spec.json rag/corpus --freeze  and re-ingest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
