"""Build the RAG index from a corpus folder. Run ONCE (and whenever the corpus changes).

Usage (from the agentigrid project root, venv active):
    OLLAMA_HOST=http://<WIN_IP>:11434 python -m agentigrid.rag.ingest rag/corpus
Reads .txt/.md files under the corpus dir, chunks them, embeds, and stores them
in a persistent Chroma index at rag/store.
"""
from __future__ import annotations

import glob
import os
import sys

from .store import VectorStore


def chunk(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Paragraph-first chunking; long paragraphs are wrapped with overlap."""
    out: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            out.append(para)
        else:
            i = 0
            while i < len(para):
                out.append(para[i : i + size])
                i += size - overlap
    return out


def ingest(
    corpus_dir: str = "rag/corpus",
    store_path: str = "rag/store",
    collection: str = "agentigrid_kb",
    host: str = "http://localhost:11434",
    model: str = "nomic-embed-text",
) -> None:
    store = VectorStore(store_path, collection, host, model)
    n = 0
    for path in sorted(glob.glob(os.path.join(corpus_dir, "**", "*.*"), recursive=True)):
        if not path.endswith((".txt", ".md")):
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        for i, c in enumerate(chunk(text)):
            store.add(f"{path}#{i}", c, {"source": os.path.basename(path)})
            n += 1
    print(f"Ingested {n} chunks; collection '{collection}' now holds {store.count()}.")


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "rag/corpus"
    ingest(corpus_dir=corpus, host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
