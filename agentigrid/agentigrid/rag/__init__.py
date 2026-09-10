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
        return CorrectiveRetriever(base)
    return base  # 'basic' (and any unknown value already normalized to basic)


__all__ = ["Retriever", "CorrectiveRetriever", "build_retriever", "resolve_rag_mode"]
