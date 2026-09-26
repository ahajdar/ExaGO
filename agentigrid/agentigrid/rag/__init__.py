"""RAG (retrieval-augmented generation) support for AgentiGrid.

Generation-stage only: retrieval grounds the LLM's spec/proposal generation.
It is NEVER used by the deterministic validator (engine/validation.py).

Mode is selected by env var (the ablation axis):

    AGENTIGRID_RAG_MODE = off | basic | corrective

For backward compatibility the legacy switch still works:
    AGENTIGRID_RAG=1  ->  basic
    AGENTIGRID_RAG=0/unset (and no _MODE) -> off

Use `build_retriever(**kwargs)` to get the retriever for the active mode; every
mode exposes the same surface (`.enabled` and `.retrieve(query) -> str`).
"""
from __future__ import annotations

import os

from .retriever import Retriever
from .corrective import CorrectiveRetriever

VALID_MODES = ("off", "basic", "corrective")


def resolve_rag_mode() -> str:
    """Return the active RAG mode, honoring AGENTIGRID_RAG_MODE and falling back
    to the legacy AGENTIGRID_RAG switch. Unknown values degrade to 'basic'.
    """
    m = os.environ.get("AGENTIGRID_RAG_MODE")
    if m:
        m = m.strip().lower()
        return m if m in VALID_MODES else "basic"
    return "basic" if os.environ.get("AGENTIGRID_RAG", "0") == "1" else "off"


def resolve_crag_grader() -> str:
    """Return the corrective-RAG grader name from AGENTIGRID_CRAG_GRADER.
    'cosine' (default/unset) or 'jev'. Unknown values fall back to 'cosine'.
    """
    name = os.environ.get("AGENTIGRID_CRAG_GRADER", "").strip().lower()
    return name if name in ("cosine", "jev") else "cosine"


def _make_crag_grader():
    """Build the optional grader for corrective mode (selected by
    AGENTIGRID_CRAG_GRADER). 'cosine' -> None (deterministic cosine buckets, the
    reproducible default). 'jev' -> JevGrader (calibrated System-1 grader; it
    falls back to cosine internally if the TypeSafe SDK/key is missing). The Jev
    module is imported lazily so its SDK is only needed when actually requested.
    """
    name = resolve_crag_grader()
    if name == "jev":
        try:
            from .grader_jev import JevGrader
            return JevGrader()
        except Exception:
            return None  # degrade to cosine
    return None  # 'cosine' (and any unknown value already normalized)


def build_retriever(**base_kwargs):
    """Construct the retriever for the active mode.

    `base_kwargs` are passed to the basic Retriever (e.g. host=, path=, model=).
    Returns a Retriever (off/basic) or a CorrectiveRetriever (corrective); both
    share the `.enabled` / `.retrieve()` surface.
    """
    mode = resolve_rag_mode()
    base_kwargs.pop("enabled", None)
    if mode == "off":
        return Retriever(enabled=False)
    base = Retriever(enabled=True, **base_kwargs)
    if mode == "corrective":
        return CorrectiveRetriever(base, grader=_make_crag_grader())
    return base  # 'basic' (and any unknown value already normalized to basic)


__all__ = [
    "Retriever", "CorrectiveRetriever", "build_retriever",
    "resolve_rag_mode", "resolve_crag_grader",
]
