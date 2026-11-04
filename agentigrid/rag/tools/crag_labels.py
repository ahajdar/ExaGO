#!/usr/bin/env python3
"""Blind relevance labels for the CRAG experiment, from a replay preview.

    make   sample (situation, retrieved chunk) pairs from a replay_preview output
           and write a blind labelling workbook for the power-systems engineer,
           plus a private key file (do not send the key).
    score  read the filled workbook, join it with the key, and report each
           grader's AUC, the reranker's calibration (ECE, raw and Platt-scaled,
           leave-one-goal-out) and thresholds fitted by the stated rule.

Label definition (B, decided 2026-11-05): 1 if the chunk would help the agent
make a valid next move toward the goal, from this situation, WITHOUT violating
the goal's stated constraints; 0 otherwise. A worked example whose step breaks a
constraint of this goal (e.g. changing the voltage band when the goal forbids
it) is 0 even if it is on topic.

Size and stopping rule (decided 2026-11-05, before any label was seen): one
batch of 50 pairs. A second batch of 50 (no repeated pair) is labelled ONLY if
batch 1 cannot support the analysis: fewer than 10 labels of 1 or of 0, or the
threshold rule finds no threshold for a grader. Otherwise 50 is final, whatever
the results show. `score` checks the rule and says whether a second batch is due.

Sampling (make): pairs are stratified by goal x query mode x reranker-grade bin
(<0.30, 0.30-0.70, >=0.70) with equal allocation per stratum, so the low-grade
tail is represented; two fixed quotas are drawn first -- pairs whose chunk comes
from the ExaGO documentation corpus, and voltage-goal pairs whose chunk changes
the voltage band. Rows are deduplicated by (goal, situation, chunk), shuffled
with a fixed seed, and show no model, condition, mode, score or grader.

    python rag/tools/crag_labels.py make --replay experiments/replay_preview_v2 --out labels/crag_labels_v1.xlsx
    python rag/tools/crag_labels.py score --sheet labels/crag_labels_v1_filled.xlsx --key labels/crag_labels_v1_key.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("relevance_labels", _HERE / "relevance_labels.py")
_rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rl)
auc, ece, thresholds, platt_fit = _rl.auc, _rl.ece, _rl.thresholds, _rl.platt_fit

GRADE_BINS = ((0.0, 0.30, "low"), (0.30, 0.70, "mid"), (0.70, 1.01, "high"))
START = "(Start of the search: the agent knows only the goal and the base case.)"
BAND_MARKERS = ("set_all_bus_vlimits", "set_bus_vlimits")


def _bin(g) -> str:
    try:
        g = float(g)
    except (TypeError, ValueError):
        return "none"
    for lo, hi, name in GRADE_BINS:
        if lo <= g < hi:
            return name
    return "none"


def load_replay(replay: Path):
    queries = {}
    for line in open(replay / "queries.jsonl", encoding="utf-8"):
        q = json.loads(line)
        queries[q["query_sha"]] = q
    chunks = {}
    for line in open(replay / "chunks.jsonl", encoding="utf-8"):
        c = json.loads(line)
        chunks[c["chunk_sha"]] = c
    pairs = list(csv.DictReader(open(replay / "pairs.csv", encoding="utf-8")))
    return queries, chunks, pairs


def goal_texts(queries: dict) -> dict:
    """goal id -> goal text, from the goal-mode queries (the query is the goal text)."""
    return {q["goal"]: q["query"] for q in queries.values() if q.get("mode") == "goal"}


def situation(q: dict, goals: dict) -> str:
    text = q["query"]
    if q.get("mode") == "goal" or text == goals.get(q.get("goal")):
        return START
    lines = text.split("\n")
    if q.get("mode") == "state" and lines and lines[0] == goals.get(q.get("goal")):
        lines = lines[1:]
    return "\n".join(lines).strip() or START


def candidates(queries, chunks, pairs) -> list[dict]:
    goals = goal_texts(queries)
    seen, out = set(), []
    for p in pairs:
        q, c = queries.get(p["query_sha"]), chunks.get(p["chunk_sha"])
        if not q or not c:
            continue
        sit = situation(q, goals)
        key = (q.get("goal"), sit, p["chunk_sha"])
        if key in seen:
            continue
        seen.add(key)
        text = c.get("text") or ""
        out.append({
            "query_sha": p["query_sha"], "chunk_sha": p["chunk_sha"], "goal_id": q.get("goal"),
            "mode": q.get("mode"), "goal_text": goals.get(q.get("goal"), ""), "situation": sit,
            "chunk": text, "source": c.get("source") or "",
            "cosine": p.get("cosine", ""), "grade_reranker": p.get("grade_reranker", ""),
            "bin": _bin(p.get("grade_reranker")),
            "is_doc": (c.get("source") or "").startswith("exago_docs_"),
            "is_band": q.get("goal") == "voltage" and any(m in text for m in BAND_MARKERS),
        })
    return out


def sample(cands: list[dict], n: int, seed: int, doc_quota: int = 6, band_quota: int = 6) -> list[dict]:
    rng = random.Random(seed)
    pool = sorted(cands, key=lambda r: (r["query_sha"], r["chunk_sha"]))
    rng.shuffle(pool)
    chosen, taken = [], set()

    def take(r):
        k = (r["query_sha"], r["chunk_sha"])
        if k not in taken and len(chosen) < n:
            taken.add(k)
            chosen.append(r)

    for flag, quota in (("is_doc", doc_quota), ("is_band", band_quota)):
        for r in [r for r in pool if r[flag]][:quota]:
            take(r)
    strata = defaultdict(list)
    for r in pool:
        strata[(r["goal_id"], r["mode"], r["bin"])].append(r)
    keys = sorted(strata)
    rng.shuffle(keys)
    idx = {k: 0 for k in keys}
    while len(chosen) < n and any(idx[k] < len(strata[k]) for k in keys):
        for k in keys:
            while idx[k] < len(strata[k]) and (strata[k][idx[k]]["query_sha"], strata[k][idx[k]]["chunk_sha"]) in taken:
                idx[k] += 1
            if idx[k] < len(strata[k]) and len(chosen) < n:
                take(strata[k][idx[k]])
                idx[k] += 1
    rng.shuffle(chosen)
    for i, r in enumerate(chosen, 1):
        r["item"] = f"P{i:03d}"
    return chosen


INSTRUCTIONS = [
    ("Relevance labels for retrieved references (AgentiGrid / ExaGO)", True),
    ("An AI agent tries to reach an engineering goal on a power-system case by proposing changes that a solver "
     "then evaluates. Before each proposal it receives up to three reference texts retrieved from a small library "
     "(worked examples from other networks, command descriptions, ExaGO documentation). Each row shows the goal, "
     "the situation the agent is in, and ONE retrieved reference.", False),
    ("", False),
    ("Question for each row (column E):", True),
    ("Would this reference help the agent make a valid next move toward the goal, from this situation, "
     "WITHOUT violating any constraint stated in the goal?  1 = yes, 0 = no.", False),
    ("", False),
    ("Guidance", True),
    ("- Being on the same topic is not enough. A worked example whose step breaks a constraint of THIS goal "
     "(e.g., changing the voltage limits when the goal says not to, shedding load, editing ratings or cost curves) is 0.", False),
    ("- Bus numbers in worked examples come from other networks; judge the method, not the numbers.", False),
    ("- A reference that is correct but irrelevant to this goal or situation (installation notes, other "
     "applications) is 0.", False),
    ("- 'Situation' lists what the agent last saw: solver status, violations, and errors from its previous command. "
     "'Start of the search' means only the goal is known.", False),
    ("- Column F: Y if you are confident, N if it is a close call. Column G: optional comment.", False),
    ("", False),
    ("Example (made up): goal 'relieve overloads without changing line ratings'; reference shows a worked example "
     "that raises a line rating to remove an overload  ->  E = 0 (method violates the goal's constraint).", False),
]


def write_workbook(rows: list[dict], path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    F, B = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    ins = wb.active
    ins.title = "Instructions"
    for i, (t, bold) in enumerate(INSTRUCTIONS, 1):
        c = ins.cell(row=i, column=1, value=t)
        c.font = B if bold else F
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ins.column_dimensions["A"].width = 120
    ws = wb.create_sheet("Pairs")
    ws.append(["Item", "Goal", "Situation", "Retrieved reference", "Label (1/0)", "Confident (Y/N)", "Comment"])
    for r in rows:
        ws.append([r["item"], r["goal_text"], r["situation"], r["chunk"], None, None, None])
    widths = [7, 34, 40, 80, 11, 13, 28]
    thin = Side(style="thin", color="999999")
    yellow = PatternFill("solid", fgColor="FFFF00")
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + j)].width = w
    for row in ws.iter_rows():
        for c in row:
            c.font = B if c.row == 1 else F
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
            if c.row > 1 and c.column >= 5:
                c.fill = yellow
    ws.freeze_panes = "B2"
    n = len(rows) + 1
    dv = DataValidation(type="list", formula1='"1,0"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"E2:E{n}")
    dv2 = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    ws.add_data_validation(dv2)
    dv2.add(f"F2:F{n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


MIN_PER_CLASS = 10
STOPPING_RULE = ("one batch of 50; a second batch of 50 (no repeated pair) only if fewer than "
                 f"{MIN_PER_CLASS} labels of 1 or of 0, or the threshold rule finds no threshold for a grader; "
                 "otherwise final, whatever the results show")


def second_batch_check(rep: dict) -> dict:
    """Apply the pre-stated stopping rule to a score report."""
    reasons = []
    pos, n = rep["positives"], rep["n_labelled"]
    if pos < MIN_PER_CLASS:
        reasons.append(f"only {pos} labels of 1 (< {MIN_PER_CLASS})")
    if n - pos < MIN_PER_CLASS:
        reasons.append(f"only {n - pos} labels of 0 (< {MIN_PER_CLASS})")
    for g, r in rep["graders"].items():
        if r["tau_lower"] is None or r["tau_upper"] is None:
            reasons.append(f"no threshold found for {g}")
    return {"rule": STOPPING_RULE, "second_batch_needed": bool(reasons), "reasons": reasons}


KEY_FIELDS = ["item", "query_sha", "chunk_sha", "goal_id", "mode", "source", "cosine", "grade_reranker",
              "bin", "is_doc", "is_band"]


def cmd_make(a) -> int:
    queries, chunks, pairs = load_replay(Path(a.replay))
    cands = candidates(queries, chunks, pairs)
    rows = sample(cands, a.n, a.seed, a.doc_quota, a.band_quota)
    out = Path(a.out)
    write_workbook(rows, out)
    key = out.with_name(out.stem + "_key.csv")
    with open(key, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=KEY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "replay": a.replay, "n": len(rows),
            "seed": a.seed, "candidates": len(cands), "doc_quota": a.doc_quota, "band_quota": a.band_quota,
            "label_definition": "B: helps a valid next move toward the goal without violating its stated constraints",
            "stopping_rule": STOPPING_RULE,
            "per_goal": {g: sum(r["goal_id"] == g for r in rows) for g in sorted({r["goal_id"] for r in rows})},
            "per_mode": {m: sum(r["mode"] == m for r in rows) for m in sorted({r["mode"] for r in rows})},
            "per_bin": {b: sum(r["bin"] == b for r in rows) for b in ("low", "mid", "high", "none")},
            "doc_pairs": sum(r["is_doc"] for r in rows), "band_pairs": sum(r["is_band"] for r in rows)}
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(rows)} items to {out} (from {len(cands)} distinct candidates).")
    print(f"Private key: {key}  -- keep it; do NOT send it to the labeller.")
    print(json.dumps({k: meta[k] for k in ("per_goal", "per_mode", "per_bin", "doc_pairs", "band_pairs")}))
    return 0


# -- score ---------------------------------------------------------------------
def read_labels(sheet: Path) -> dict:
    from openpyxl import load_workbook
    ws = load_workbook(sheet, data_only=True)["Pairs"]
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        item, label, conf = row[0], row[4], row[5]
        if item is None:
            continue
        s = str(label).strip() if label is not None else ""
        if s in ("1", "0", "1.0", "0.0"):
            out[str(item)] = {"y": int(float(s)), "confident": str(conf or "").strip().upper() == "Y"}
    return out


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def score_report(rows: list[dict], precision=0.8, npv=0.8) -> dict:
    graders = {"cosine": "cosine", "reranker": "grade_reranker"}
    rep = {"n_labelled": len(rows), "positives": sum(r["y"] for r in rows),
           "rule": {"precision": precision, "npv": npv}, "graders": {}}
    goals = sorted({r["goal_id"] for r in rows})
    for g, col in graders.items():
        rs = [r for r in rows if str(r.get(col, "")) not in ("", "None")]
        s, y = [float(r[col]) for r in rs], [r["y"] for r in rs]
        tl, tu = thresholds(s, y, precision, npv) if rs else (None, None)
        rep["graders"][g] = {
            "n": len(rs), "auc": auc(s, y),
            "auc_by_goal": {gl: auc([float(r[col]) for r in rs if r["goal_id"] == gl],
                                    [r["y"] for r in rs if r["goal_id"] == gl]) for gl in goals},
            "tau_lower": tl, "tau_upper": tu,
            "replay_arg": (f"--tau-{g} {tl},{tu}" if tl is not None and tu is not None else None),
        }
    # reranker calibration: raw unit score vs Platt scaling, leave one goal out
    rs = [r for r in rows if str(r.get("grade_reranker", "")) not in ("", "None")]
    raw, cal, ys = [], [], []
    for gl in goals:
        fit = [r for r in rs if r["goal_id"] != gl]
        test = [r for r in rs if r["goal_id"] == gl]
        if not test or len({r["y"] for r in fit}) < 2:
            continue
        a, b = platt_fit([_logit(float(r["grade_reranker"])) for r in fit], [r["y"] for r in fit])
        for r in test:
            p = float(r["grade_reranker"])
            raw.append(p)
            cal.append(1.0 / (1.0 + math.exp(-(a * _logit(p) + b))))
            ys.append(r["y"])
    rep["reranker_calibration_logo"] = {"n": len(ys), "ece_raw": ece(raw, ys), "ece_platt": ece(cal, ys),
                                        "auc_platt": auc(cal, ys)}
    rep["subsets"] = {
        "documentation_pairs": {"n": sum(r["is_doc"] for r in rows), "positives": sum(r["y"] for r in rows if r["is_doc"])},
        "band_change_pairs": {"n": sum(r["is_band"] for r in rows), "positives": sum(r["y"] for r in rows if r["is_band"])},
        "confident_share": round(sum(r["confident"] for r in rows) / len(rows), 3) if rows else None,
    }
    return rep


def cmd_score(a) -> int:
    labels = read_labels(Path(a.sheet))
    key = list(csv.DictReader(open(a.key, encoding="utf-8")))
    rows = []
    for k in key:
        if k["item"] in labels:
            rows.append({**k, **labels[k["item"]], "is_doc": k["is_doc"] == "True", "is_band": k["is_band"] == "True"})
    if not rows:
        print("No labelled items found (column E must hold 1 or 0).", file=sys.stderr)
        return 1
    rep = score_report(rows, a.precision, a.npv)
    rep["created"] = datetime.now().isoformat(timespec="seconds")
    rep["unlabelled"] = len(key) - len(rows)
    rep["stopping_rule"] = second_batch_check(rep)
    out = Path(a.out) if a.out else Path(a.key).with_name(Path(a.key).stem.replace("_key", "") + "_scores.json")
    out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))
    sr = rep["stopping_rule"]
    print(f"\nWrote {out}")
    print("Stopping rule: " + ("SECOND BATCH NEEDED -- " + "; ".join(sr["reasons"]) if sr["second_batch_needed"]
                               else "batch 1 is final (no second batch)."))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--replay", required=True)
    m.add_argument("--out", default="labels/crag_labels_v1.xlsx")
    m.add_argument("--n", type=int, default=50)
    m.add_argument("--seed", type=int, default=20261105)
    m.add_argument("--doc-quota", type=int, default=6)
    m.add_argument("--band-quota", type=int, default=6)
    s = sub.add_parser("score")
    s.add_argument("--sheet", required=True)
    s.add_argument("--key", required=True)
    s.add_argument("--out")
    s.add_argument("--precision", type=float, default=0.8)
    s.add_argument("--npv", type=float, default=0.8)
    a = ap.parse_args(argv)
    return cmd_make(a) if a.cmd == "make" else cmd_score(a)


if __name__ == "__main__":
    raise SystemExit(main())
