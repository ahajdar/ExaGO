"""Jev-backed relevance grader for Corrective RAG (optional, generation-stage only).

Drop-in for `CorrectiveRetriever(base, grader=JevGrader())`. It fills the grader
seam already exposed by CorrectiveRetriever:

    grader: Callable[[str, list[Hit]], list[float]]   # -> one relevance score per hit

Jev (TypeSafe AI) is a "System-1" decision model: instead of generating text it
returns typed, CALIBRATED decisions. We use its `Noul` primitive ("is this
reference relevant to the task?") to get a calibrated relevance probability in
[0, 1] per retrieved chunk, replacing the deterministic cosine buckets. The
calibrated probability is what makes it interesting: the corrective thresholds
(tau_lower/tau_upper) become probabilities with a real meaning instead of
hand-tuned cosine cutoffs.

DESIGN RULES (unchanged from the rest of the RAG package):
  - Generation-stage only. This NEVER touches the deterministic validator
    (engine/validation.py). Jev grades *retrieval*, not grid feasibility.
  - Defensive. Any failure (no key, no SDK, network error, shape mismatch)
    falls back to the chunks' own cosine similarity, so a run never breaks and
    degrades to exactly the basic-cosine behavior.

STATUS: adapter stub. Jev has no published SDK spec yet, so the ONLY code that
touches the vendor is `_make_client` and `_score_relevance` (clearly marked).
Everything else — the seam contract, batching, clamping, fallback — is stable
and testable without a key. Wire the two methods when you have API access.

Usage:
    from agentigrid.rag.grader_jev import JevGrader
    from agentigrid.rag.corrective import CorrectiveRetriever
    cr = CorrectiveRetriever(base_retriever, grader=JevGrader())

Or via env, if you extend build_retriever() (see the snippet at the bottom).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

# Same shape the retriever/corrective modules use.
Hit = tuple[str, dict, float]

DEFAULT_STATEMENT = (
    "This reference material is relevant and directly helpful for producing a "
    "correct action for the task described in the query."
)


class JevGrader:
    """Callable relevance grader backed by Jev's Noul primitive.

    `__call__(query, hits) -> [relevance in 0..1 per hit]`, aligned with `hits`.
    Falls back to cosine similarity (hits' own scores) on any problem, so it is
    always safe to pass in.
    """

    def __init__(
        self,
        *,
        api_key_env: str = "TYPESAFE_API_KEY",
        model: str = "jev-1",
        statement: str = DEFAULT_STATEMENT,
        timeout_s: float = 5.0,
    ):
        self._api_key = os.environ.get(api_key_env)
        self._model = model
        self._statement = statement
        self._timeout_s = timeout_s
        self._client = None
        if self._api_key:
            try:
                self._client = self._make_client()
            except Exception:
                self._client = None  # degrade to cosine fallback

    @property
    def available(self) -> bool:
        return self._client is not None

    def __call__(self, query: str, hits: list[Hit]) -> list[float]:
        if not hits:
            return []
        cosine = [s for (_d, _m, s) in hits]  # fallback = current basic behavior
        if self._client is None or not query:
            return cosine
        docs = [d for (d, _m, _s) in hits]
        try:
            scores = self._score_relevance(query, docs)
        except Exception:
            return cosine  # retrieval grading must never break a run
        if not scores or len(scores) != len(docs):
            return cosine
        # Clamp into [0, 1] so downstream thresholds behave.
        return [max(0.0, min(1.0, float(x))) for x in scores]

    def describe(self) -> dict:
        # Report what actually runs: if Jev isn't usable, scores are cosine.
        name = "jev" if self.available else "jev (fallback: cosine)"
        return {"grader": name, "model": self._model, "available": self.available}

    # ------------------------------------------------------------------
    # The ONLY vendor boundary. Wire these two to the real TypeSafe SDK.
    # ------------------------------------------------------------------
    def _make_client(self):
        """Construct the Jev client. Replace with the real SDK, e.g.:

            from typesafe import Jev
            return Jev(api_key=self._api_key, timeout=self._timeout_s)
        """
        raise NotImplementedError("Wire JevGrader._make_client to the TypeSafe SDK.")

    def _score_relevance(self, query: str, docs: Sequence[str]) -> list[float]:
        """Return one calibrated relevance probability (0..1) per doc, in order.

        Intended shape using Jev's Noul primitive (map-reduced over docs),
        pseudocode against a plausible SDK:

            results = self._client.noul(
                model=self._model,
                statement=self._statement,
                state=[{"query": query, "reference": d} for d in docs],
            )
            return [r.probability for r in results]   # 0..1 per doc

        Keep it batched (one call for all docs) — Jev is built for map-reduce and
        it's cheaper/faster than per-doc calls.
        """
        raise NotImplementedError("Wire JevGrader._score_relevance to Jev's Noul primitive.")


@lru_cache(maxsize=1)
def jev_available() -> bool:
    """True only when Jev is actually usable (SDK wired + TYPESAFE_API_KEY set).

    Used by the UI to enable/disable the grader selector. Cached for the life of
    the process — restart Streamlit after adding the key or wiring the SDK.
    """
    try:
        return JevGrader().available
    except Exception:
        return False


# ----------------------------------------------------------------------
# Optional: enable via env in build_retriever() WITHOUT importing this
# module (and its future SDK dependency) unless actually requested:
#
#   # in agentigrid/rag/__init__.py, inside build_retriever(), corrective branch:
#   grader = None
#   if os.environ.get("AGENTIGRID_CRAG_GRADER", "").lower() == "jev":
#       from .grader_jev import JevGrader
#       grader = JevGrader()
#   return CorrectiveRetriever(base, grader=grader)
#
# CorrectiveRetriever already treats grader=None as "use cosine", so this is
# backward-compatible and the default stays deterministic/reproducible.
# ----------------------------------------------------------------------
