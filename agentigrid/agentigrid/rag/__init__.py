"""RAG (retrieval-augmented generation) support for AgentiGrid.

Generation-stage only: retrieval grounds the LLM's spec/proposal generation.
It is NEVER used by the deterministic validator (engine/validation.py).
"""
from .retriever import Retriever

__all__ = ["Retriever"]
