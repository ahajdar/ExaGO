#!/usr/bin/env python3
"""Precision@k of what basic retrieval (C1) actually injects, per goal.

A small, descriptive relevance check that needs no grader study: take the
reference block C1 injects for each evaluated goal (from retrieval_preview.py's
JSON -- the goal text is the query, so this block is fixed per goal), have one
person judge each injected chunk relevant (1) or not (0) to the goal, and report
precision@k = relevant chunks / k per goal and on average. With k = 3 and five
goals that is 15 judgments.

    python rag/tools/precision_at_k.py sheet --preview experiments/retrieval_preview.json \
        --out labels/precision_at3.xlsx
    # fill the yellow Label cells (1/0), save
    python rag/tools/precision_at_k.py score --labels labels/precision_at3.xlsx \
        --out experiments/precision_at3.json

The sheet hides similarity scores and ranks, so the judge is blind to the
retriever. A goal for which C1 returned fewer than k chunks counts the missing
ones as not relevant (standard precision@k); precision over returned chunks is
reported alongside.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REF = re.compile(r"^\[ref (\d+) \| score ([0-9.]+)\] ", re.M)
HEADERS = ["Goal id", "Goal", "Ref", "Chunk", "Label (1/0)", "Comment"]
CRITERIA = [
    "Question: if the agent read only this chunk, would its next proposal for this goal be more "
    "likely to be valid and on task?",
    "1 = the chunk shows a worked step of the same kind of task, explains a command or option the "
    "goal needs, or gives a tool fact the goal depends on.",
    "0 = the chunk is about a different kind of task, an application the goal does not use, a "
    "generic listing, or shows an action the goal forbids.",
    "Worked examples come from other networks; judge whether the method applies, not the bus numbers.",
    "Judge each chunk on its own. Leave Label empty only if you truly cannot decide.",
]


def parse_refs(context: str) -> list[dict]:
    """Split a basic-retriever block into its references (number, score, chunk text)."""
    marks = list(REF.finditer(context or ""))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(context)
        out.append({"ref": int(m.group(1)), "score": float(m.group(2)),
                    "chunk": context[m.end():end].rstrip("\n")})
    return out


def items_from_preview(preview: dict, spec: dict, condition: str) -> list[dict]:
    text = {g["id"]: g["text"] for g in spec.get("goals", [])}
    items = []
    for gid, per in preview["goals"].items():
        if condition not in per:
            raise SystemExit(f"condition {condition!r} not in the preview for goal {gid}")
        for r in parse_refs(per[condition]["context"]):
            items.append({"goal_id": gid, "goal": text.get(gid, ""), **r})
    return items


def write_sheet(items, out, meta):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    font, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    wrap = Alignment(wrap_text=True, vertical="top")
    yellow = PatternFill("solid", start_color="FFFF00")
    wb = Workbook()
    ws = wb.active
    ws.title = "Instructions"
    ws.column_dimensions["A"].width = 120
    ws["A1"] = "Relevance check: what basic retrieval shows the agent"
    ws["A1"].font = Font(name="Arial", size=13, bold=True)
    ws["A3"] = ("For each row of the Items sheet, read the Goal and the Chunk and enter 1 or 0 in the "
                "yellow Label cell. These are the chunks the retrieval step actually gives the agent for "
                "each goal.")
    for i, line in enumerate(CRITERIA, 5):
        ws.cell(i, 1, "• " + line)
    for row in ws.iter_rows(min_row=3, max_row=4 + len(CRITERIA)):
        for c in row:
            c.font, c.alignment = font, wrap
    ws = wb.create_sheet("Items")
    for j, (h, w) in enumerate(zip(HEADERS, [10, 45, 6, 100, 12, 30]), 1):
        c = ws.cell(1, j, h)
        c.font = bold
        ws.column_dimensions[c.column_letter].width = w
    for i, it in enumerate(items, 2):
        for j, v in enumerate((it["goal_id"], it["goal"], it["ref"], it["chunk"], None, None), 1):
            c = ws.cell(i, j, v)
            c.font, c.alignment = font, wrap
        ws.cell(i, 5).fill = yellow
        ws.cell(i, 6).fill = yellow
        lines = sum(max(1, len(l) // 95 + 1) for l in it["chunk"].split("\n"))
        ws.row_dimensions[i].height = min(409, 15 * lines + 4)
    dv = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"E2:E{len(items) + 1}")
    ws.freeze_panes = "C2"
    wm = wb.create_sheet("_meta")
    for i, (k, v) in enumerate(meta.items(), 1):
        wm.cell(i, 1, k)
        wm.cell(i, 2, v)
    wm.sheet_state = "hidden"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


def score(rows: list[dict], k: int) -> dict:
    """rows: goal_id, label in {'1','0',''}. precision@k per goal and macro average."""
    per = {}
    for r in rows:
        g = per.setdefault(r["goal_id"], {"returned": 0, "relevant": 0, "unlabelled": 0})
        g["returned"] += 1
        lab = str(r.get("label") if r.get("label") is not None else "").strip()
        if lab in ("1", "1.0"):
            g["relevant"] += 1
        elif lab not in ("0", "0.0"):
            g["unlabelled"] += 1
    for g in per.values():
        g["precision_at_k"] = round(g["relevant"] / k, 3)
        labelled = g["returned"] - g["unlabelled"]
        g["precision_returned"] = round(g["relevant"] / labelled, 3) if labelled else None
    vals = [g["precision_at_k"] for g in per.values()]
    return {"k": k, "goals": per, "macro_precision_at_k": round(sum(vals) / len(vals), 3) if vals else None,
            "unlabelled": sum(g["unlabelled"] for g in per.values())}


def read_labels(path) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        import csv
        with open(p, newline="", encoding="utf-8") as f:
            return [{"goal_id": r["goal_id"], "label": r.get("label", "")} for r in csv.DictReader(f)]
    from openpyxl import load_workbook
    ws = load_workbook(p, data_only=True)["Items"]
    head = [c.value for c in ws[1]]
    gi, li = head.index("Goal id"), head.index("Label (1/0)")
    return [{"goal_id": r[gi], "label": "" if r[li] is None else str(r[li])}
            for r in ws.iter_rows(min_row=2, values_only=True) if r[gi]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--preview", default="experiments/retrieval_preview.json")
    s.add_argument("--spec", default="grader_ablation_spec.json")
    s.add_argument("--condition", default="C1-basic")
    s.add_argument("--out", default="labels/precision_at3.xlsx")
    c = sub.add_parser("score")
    c.add_argument("--labels", default="labels/precision_at3.xlsx")
    c.add_argument("--k", type=int, default=3)
    c.add_argument("--out", default="experiments/precision_at3.json")
    a = ap.parse_args(argv)
    if a.cmd == "sheet":
        preview = json.loads(Path(a.preview).read_text())
        spec = json.loads(Path(a.spec).read_text())
        items = items_from_preview(preview, spec, a.condition)
        meta = {"preview": a.preview, "preview_sha256": hashlib.sha256(Path(a.preview).read_bytes()).hexdigest(),
                "condition": a.condition, "items": len(items), "created": datetime.now().isoformat()}
        write_sheet(items, a.out, meta)
        print(f"Wrote {len(items)} item(s) from {a.condition} -> {a.out}")
        return 0
    res = score(read_labels(a.labels), a.k)
    res.update({"labels": a.labels, "scored": datetime.now().isoformat()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    for gid, g in res["goals"].items():
        print(f"  {gid:<10} P@{a.k}={g['precision_at_k']:<6} relevant={g['relevant']}/{g['returned']}"
              + (f"  unlabelled={g['unlabelled']}" if g["unlabelled"] else ""))
    print(f"Macro P@{a.k} = {res['macro_precision_at_k']}  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
