"""Dựng vector store Day 7 bằng đúng embedding model và ChromaStore của Day 6 (đọc từ repo Day 6, không sửa)."""
from __future__ import annotations

import json
import shutil
import sys
import time

from .config import CHROMA_DIR, CHROMA_INDEX, EMBEDDING_MODEL
from .data import load_chunks
from .day6_loader import submodule


def build(reset: bool = False) -> dict:
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    models = submodule("models")
    embeddings = submodule("embeddings")
    store_mod = submodule("store")
    rows = load_chunks()
    chunks = [models.Chunk(chunk_id=r["id"], document_id=r["source_id"], source=r["source"], page=r["page"],
                           text=r["text"], metadata={**r["metadata"], "page_end": r["page_end"]}) for r in rows]
    embedder = embeddings.MyEmbedder(EMBEDDING_MODEL, device="cpu")
    start = time.perf_counter()
    vectors = embedder.embed_documents([c.text for c in chunks])
    embed_s = time.perf_counter() - start
    CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
    store = store_mod.ChromaStore(str(CHROMA_DIR), dimensions=embedder.dimensions, embedding_model=EMBEDDING_MODEL)
    stats = store.upsert(CHROMA_INDEX, chunks, vectors)
    stats.update({"chunks": len(chunks), "embed_seconds": round(embed_s, 2), "dimensions": embedder.dimensions})
    return stats


if __name__ == "__main__":
    print(json.dumps(build(reset="--reset" in sys.argv), ensure_ascii=False, indent=2))
