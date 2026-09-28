#!/usr/bin/env python3
"""Relevance labels for the grader study (C2b thresholds, C2c calibration, H4).

Two steps, both offline (no LLM, no API):

  sample  Build a CSV of (query, chunk) pairs from the FROZEN, ingested corpus for
          a human to label 1 (the chunk would help an agent pursue this goal) or
          0 (it would not). Queries are the evaluated goal texts -- exactly what
          retrieval uses -- plus their CRAG keyword reformulation where it differs.
          Per query: the top --top chunks by cosine similarity (where grading
          decisions are made) and --random further chunks drawn at random (so
          clearly irrelevant pairs are represented too).

  fit     Read the labelled CSV, split it BY GOAL into a fitting set and a
          held-out set, then
            * C2b: fix the corrective thresholds on the reranker's unit-interval
              score (fitting set): tau_upper = lowest score at which precision of
              "relevant" >= --precision; tau_lower = highest score below which the
              share of irrelevant chunks >= --npv;
            * C2c: fit Platt scaling (2-parameter logistic regression) from the
              reranker's logit to P(relevant) on the fitting set;
            * H4 (held-out set): AUC of cosine, reranker and calibrated scores;
              expected calibration error (10 equal-width bins) of the reranker's
              unit score and of the calibrated probability.
          Writes a JSON report and the env lines for the C2b condition.

Usage (agentigrid project root, venv active, store ingested from the frozen corpus):
    OLLAMA_HOST=... python rag/tools/relevance_labels.py sample --out labels/pairs.csv
    # label the "label" column (1/0) in any spreadsheet tool, keep the CSV format
    python rag/tools/relevance_labels.py fit --labels labels/pairs.csv --heldout-goals relieve,n1cost10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

FIELDS = ["pair_id", "goal_id", "query_kind", "query", "doc_id", "source", "cosine", "rank",
          "label", "chunk"]


# ----------------------------------------------------------------------------- sample

def build_queries(spec: dict) -> list[dict]:
    from agentigrid.rag.corrective import CorrectiveRetriever
    reform = CorrectiveRetriever._reformulate
    out = []
    for g in spec.get("goals", []):
        out.append({"goal_id": g["id"], "kind": "goal", "text": g["text"]})
        r = reform(None, g["text"])
        if r != g["text"]:
            out.append({"goal_id": g["id"], "kind": "rewrite", "text": r})
    return out


def sample_pairs(queries, docs, query_fn, top: int, n_random: int, seed: int) -> list[dict]:
    """docs: [(id, text, meta)]; query_fn(text, k) -> [(doc_text, meta, cosine)]."""
    rng = random.Random(seed)
    by_text = {}
    for did, text, meta in docs:
        by_text.setdefault(text, (did, meta))
    rows = []
    for q in queries:
        hits = query_fn(q["text"], top)
        seen = set()
        for rank, (text, meta, cos) in enumerate(hits, 1):
            did, m = by_text.get(text, ("?", meta or {}))
            seen.add(did)
            rows.append(_row(q, did, (m or {}).get("source", ""), text, cos, rank))
        pool = [d for d in docs if d[0] not in seen]
        for did, text, meta in rng.sample(pool, min(n_random, len(pool))):
            rows.append(_row(q, did, (meta or {}).get("source", ""), text, None, "random"))
    for i, r in enumerate(rows, 1):
        r["pair_id"] = i
    return rows


def _row(q, did, source, text, cos, rank):
    return {"goal_id": q["goal_id"], "query_kind": q["kind"], "query": q["text"], "doc_id": did,
            "source": source, "cosine": "" if cos is None else round(float(cos), 4), "rank": rank,
            "label": "", "chunk": text}


def cmd_sample(args) -> int:
    import os
    from agentigrid.rag.corpus_hash import INGEST_MANIFEST_NAME
    from agentigrid.rag.embed import embed
    from agentigrid.rag.store import VectorStore

    spec = json.loads(Path(args.spec).read_text())
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    store = VectorStore(args.store, args.collection, host, args.embed_model)
    docs = store.all_documents()
    if not docs:
        print(f"store {args.store} is empty -- ingest the frozen corpus first", file=sys.stderr)
        return 2
    queries = build_queries(spec)
    doc_emb = {}

    def cosine_all(text, k):
        # exact cosine over ALL chunks (not the ANN index), so ranks are reproducible
        qv = embed(text, host, args.embed_model)
        scored = []
        for did, d, meta in docs:
            if did not in doc_emb:
                doc_emb[did] = embed(d, host, args.embed_model)
            scored.append((d, meta, _cos(qv, doc_emb[did])))
        scored.sort(key=lambda x: -x[2])
        return scored[:k]

    rows = sample_pairs(queries, docs, cosine_all, args.top, args.random, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    ing = Path(args.store) / INGEST_MANIFEST_NAME
    meta = {"created": datetime.now().isoformat(), "spec": args.spec, "store": args.store,
            "corpus_sha256": json.loads(ing.read_text()).get("corpus_sha256") if ing.exists() else None,
            "queries": len(queries), "pairs": len(rows), "top": args.top, "random": args.random,
            "seed": args.seed}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(rows)} pairs for {len(queries)} queries -> {out}\n"
          f"Label the 'label' column 1 (would help an agent pursue this goal) or 0 (would not).")
    return 0


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


# ----------------------------------------------------------------------------- fit

def auc(scores, labels) -> float | None:
    """Mann-Whitney AUC; ties count 1/2. None if a class is missing."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def ece(probs, labels, bins: int = 10) -> float | None:
    if not probs:
        return None
    total, err = len(probs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(acc - conf)
    return err


def platt_fit(x, y, iters: int = 200):
    """Platt scaling: P(y=1|x) = 1 / (1 + exp(-(a*x + b))), with Platt's target
    smoothing, fitted by Newton's method. Returns (a, b)."""
    n1 = sum(y)
    n0 = len(y) - n1
    t1, t0 = (n1 + 1) / (n1 + 2), 1 / (n0 + 2)
    t = [t1 if yi == 1 else t0 for yi in y]
    a, b = 0.0, math.log((n0 + 1) / (n1 + 1))
    for _ in range(iters):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for xi, ti in zip(x, t):
            p = 1.0 / (1.0 + math.exp(-(a * xi + b)))
            d = p - ti
            w = p * (1 - p)
            g_a += d * xi; g_b += d
            h_aa += w * xi * xi; h_ab += w * xi; h_bb += w
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a, b = a - da, b - db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return a, b


def thresholds(scores, labels, precision: float, npv: float):
    """tau_upper: lowest threshold with precision(score >= tau) >= `precision`;
    tau_lower: highest threshold with share of 0-labels among score < tau >= `npv`."""
    cand = sorted(set(scores))
    tau_u = None
    for c in cand:
        sel = [y for s, y in zip(scores, labels) if s >= c]
        if sel and sum(sel) / len(sel) >= precision:
            tau_u = c
            break
    tau_l = None
    for c in reversed(cand):
        sel = [y for s, y in zip(scores, labels) if s < c]
        if sel and (len(sel) - sum(sel)) / len(sel) >= npv:
            tau_l = c
            break
    if tau_u is not None and tau_l is not None and tau_l > tau_u:
        tau_l = tau_u
    return tau_l, tau_u


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def fit_report(rows, heldout: set, reranker_fn, precision=0.8, npv=0.8) -> dict:
    lab = [r for r in rows if str(r.get("label", "")).strip() in ("0", "1")]
    if not lab:
        raise SystemExit("no labelled rows (label column must be 0 or 1)")
    for r in lab:
        r["y"] = int(str(r["label"]).strip())
    missing = heldout - {r["goal_id"] for r in lab}
    if missing:
        raise SystemExit(f"held-out goals without labels: {sorted(missing)}")
    scores = reranker_fn([(r["query"], r["chunk"]) for r in lab])   # unit interval
    for r, s in zip(lab, scores):
        r["rerank"] = float(s)
    fit = [r for r in lab if r["goal_id"] not in heldout]
    test = [r for r in lab if r["goal_id"] in heldout]
    if not fit or not test:
        raise SystemExit("need labelled rows in both the fitting and the held-out goals")
    tau_l, tau_u = thresholds([r["rerank"] for r in fit], [r["y"] for r in fit], precision, npv)
    a, b = platt_fit([_logit(r["rerank"]) for r in fit], [r["y"] for r in fit])

    def calib(p):
        return 1.0 / (1.0 + math.exp(-(a * _logit(p) + b)))

    ty = [r["y"] for r in test]
    cos_ok = [r for r in test if str(r.get("cosine", "")) not in ("", "None")]
    report = {
        "created": datetime.now().isoformat(),
        "n_labelled": len(lab), "n_fit": len(fit), "n_heldout": len(test),
        "heldout_goals": sorted(heldout), "positives_fit": sum(r["y"] for r in fit),
        "positives_heldout": sum(ty),
        "rule": {"precision": precision, "npv": npv},
        "C2b": {"tau_lower": tau_l, "tau_upper": tau_u,
                "env": {"AGENTIGRID_CRAG_TAU_LOWER": tau_l, "AGENTIGRID_CRAG_TAU_UPPER": tau_u}},
        "C2c": {"platt_a": a, "platt_b": b},
        "heldout": {
            "auc_cosine": auc([float(r["cosine"]) for r in cos_ok], [r["y"] for r in cos_ok]),
            "auc_cosine_note": "top-k pairs only (random pairs carry no cosine rank)",
            "auc_reranker": auc([r["rerank"] for r in test], ty),
            "auc_calibrated": auc([calib(r["rerank"]) for r in test], ty),
            "ece_reranker_unit": ece([r["rerank"] for r in test], ty),
            "ece_calibrated": ece([calib(r["rerank"]) for r in test], ty),
        },
    }
    return report


def cmd_fit(args) -> int:
    with open(args.labels, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    heldout = {g.strip() for g in args.heldout_goals.split(",") if g.strip()}
    from agentigrid.rag.grader_reranker import RerankerGrader
    grader = RerankerGrader(model_name=args.model) if args.model else RerankerGrader()

    def rerank(pairs):
        hits = [(chunk, {}, 0.0) for _q, chunk in pairs]
        out = []
        for (q, _c), h in zip(pairs, hits):        # one query per pair
            out.extend(grader(q, [h]))
        return out

    report = fit_report(rows, heldout, rerank, args.precision, args.npv)
    report["reranker"] = grader.describe()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    c = report["C2b"]
    print(json.dumps(report["heldout"], indent=2))
    print(f"\nC2b thresholds (fitting goals): tau_lower={c['tau_lower']}, tau_upper={c['tau_upper']}")
    print(f"C2c Platt: a={report['C2c']['platt_a']:.4f}, b={report['C2c']['platt_b']:.4f}")
    print(f"Report -> {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--spec", default="grader_ablation_spec.json")
    s.add_argument("--store", default="rag/store")
    s.add_argument("--collection", default="agentigrid_kb")
    s.add_argument("--embed-model", default="nomic-embed-text")
    s.add_argument("--top", type=int, default=25)
    s.add_argument("--random", type=int, default=10)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="labels/pairs.csv")
    f = sub.add_parser("fit")
    f.add_argument("--labels", required=True)
    f.add_argument("--heldout-goals", required=True, help="comma-separated goal ids held out for AUC/ECE")
    f.add_argument("--precision", type=float, default=0.8)
    f.add_argument("--npv", type=float, default=0.8)
    f.add_argument("--model", default=None, help="reranker model (default: grader default / env)")
    f.add_argument("--out", default="labels/fit_report.json")
    args = ap.parse_args(argv)
    return cmd_sample(args) if args.cmd == "sample" else cmd_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
