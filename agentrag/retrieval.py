"""Ba phương pháp retrieval: dense (vector store Day 6), BM25 (lexical), hybrid (RRF dense + BM25).

Mỗi retriever trả list dict: id, text, source_id, source, page, page_end, metadata, score, rank, method.
Filter chỉ cho phép `document_id`; giá trị đến từ `infer_filters` (suy ra từ text câu hỏi), không từ gold.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

from .config import CHROMA_DIR, CHROMA_INDEX, EMBEDDING_MODEL
from .data import load_chunks, norm
from .day6_loader import submodule

RRF_K = 60
DEPTH = 20  # số candidate mỗi nhánh trước khi hợp nhất


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def infer_filters(question: str) -> dict[str, str] | None:
    """Suy ra document_id khi câu hỏi nêu rõ số hiệu văn bản. Không suy luận khi câu hỏi mơ hồ."""
    q = norm(question).casefold()
    if re.search(r"nghị định\s*(số\s*)?13\s*/\s*2023|nghị định 13|13/2023", q):
        return {"document_id": "nd-13-2023"}
    if re.search(r"91/2025|luật\s*(số\s*)?91|luật bảo vệ dữ liệu", q):
        return {"document_id": "luat-91-2025"}
    return None


def _as_result(row: dict[str, Any], rank: int, score: float, method: str) -> dict[str, Any]:
    return {"id": row["id"], "text": row["text"], "source_id": row["source_id"], "source": row["source"],
            "page": row["page"], "page_end": row["page_end"], "metadata": row.get("metadata", {}),
            "score": round(float(score), 6), "rank": rank, "method": method}


class BM25:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_terms: list[str], index: int) -> float:
        d, length, s = self.docs[index], self.lengths[index], 0.0
        for t in query_terms:
            if t not in d:
                continue
            tf = d[t]
            s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
        return s


class LexicalRetriever:
    method = "bm25"

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.bm25 = BM25(chunks)

    def search(self, query: str, *, filters: dict[str, str] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        terms = tokenize(query)
        scored = []
        for i, c in enumerate(self.chunks):
            if filters and any(c["source_id"] != v for k, v in filters.items() if k == "document_id"):
                continue
            s = self.bm25.score(terms, i)
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [_as_result(c, r, s, self.method) for r, (s, c) in enumerate(scored[:top_k], start=1)]


class DenseRetriever:
    """Embedding query tương thích Day 6 (e5 + prefix 'query: ') và truy vấn ChromaStore của Day 6."""
    method = "dense"

    def __init__(self, chunks_by_id: dict[str, dict[str, Any]]):
        embeddings = submodule("embeddings")
        store_mod = submodule("store")
        self.embedder = embeddings.MyEmbedder(EMBEDDING_MODEL, device="cpu")
        self.store = store_mod.ChromaStore(str(CHROMA_DIR), dimensions=self.embedder.dimensions, embedding_model=EMBEDDING_MODEL)
        self.chunks_by_id = chunks_by_id

    def search(self, query: str, *, filters: dict[str, str] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        vector = self.embedder.embed_query(query)
        hits = self.store.search(CHROMA_INDEX, vector, top_k=top_k, filters=filters or None)
        out = []
        for h in hits:
            base = self.chunks_by_id[h["chunk_id"]]
            out.append(_as_result(base, h["rank"], h["score"], self.method))
        return out

    def search_depth(self, query: str, *, filters: dict[str, str] | None, depth: int) -> list[dict[str, Any]]:
        return self.search(query, filters=filters, top_k=depth)


class HybridRetriever:
    """Reciprocal Rank Fusion: score = Σ 1/(RRF_K + rank) qua hai nhánh, không cộng score thô."""
    method = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, lexical: LexicalRetriever):
        self.dense, self.lexical = dense, lexical

    def search(self, query: str, *, filters: dict[str, str] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        a = self.dense.search(query, filters=filters, top_k=DEPTH)
        b = self.lexical.search(query, filters=filters, top_k=DEPTH)
        fused: dict[str, float] = {}
        rows: dict[str, dict[str, Any]] = {}
        for lst in (a, b):
            for r in lst:
                fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (RRF_K + r["rank"])
                rows.setdefault(r["id"], r)
        order = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        return [_as_result(rows[i], rank, fused[i], self.method) for rank, i in enumerate(order, start=1)]


def pack_context(results: list[dict[str, Any]], budget: int) -> tuple[str, list[str]]:
    """Ghép chunk theo thứ tự, bỏ trùng; chunk không vừa budget (gồm nguồn) thì bỏ."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        block = f"[{c['id']} | {c['source_id']} | trang {c['page']}-{c['page_end']}]\n{c['text']}"
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids


RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # cross-encoder đa ngữ, cho điểm từng cặp (câu hỏi, đoạn)
RERANK_CANDIDATES = 20                      # số candidate lấy từ retriever gốc trước khi xếp lại


class _Reranker:
    """Nạp lười: chỉ tải model khi thật sự có truy vấn rerank (model ~2 GB)."""
    _model = None

    @classmethod
    def predict(cls, pairs: list[tuple[str, str]]) -> list[float]:
        if cls._model is None:
            from sentence_transformers import CrossEncoder
            cls._model = CrossEncoder(RERANK_MODEL, device="cpu", max_length=512)
        return [float(x) for x in cls._model.predict(pairs, batch_size=8, show_progress_bar=False)]


class RerankRetriever:
    """Lấy nhiều candidate từ retriever gốc rồi xếp lại bằng cross-encoder; trả top-k.

    Mỗi kết quả có `pre_rank` (hạng trước khi xếp lại) để so sánh trước/sau.
    """

    def __init__(self, base: Any, candidates: int = RERANK_CANDIDATES):
        self.base, self.candidates = base, candidates
        self.method = f"{base.method}+rerank"
        self.last_pool: list[dict[str, Any]] = []

    def search(self, query: str, *, filters: dict[str, str] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        pool = self.base.search(query, filters=filters, top_k=max(self.candidates, top_k))
        self.last_pool = pool
        if not pool:
            return []
        scores = _Reranker.predict([(query, c["text"]) for c in pool])
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i]["id"]))[:top_k]
        return [{**pool[i], "score": round(scores[i], 6), "rank": r, "method": self.method, "pre_rank": pool[i]["rank"]}
                for r, i in enumerate(order, start=1)]


def make_retrievers(with_rerank: bool = False) -> dict[str, Any]:
    chunks = load_chunks()
    by_id = {c["id"]: c for c in chunks}
    dense = DenseRetriever(by_id)
    lexical = LexicalRetriever(chunks)
    hybrid = HybridRetriever(dense, lexical)
    out: dict[str, Any] = {"dense": dense, "bm25": lexical, "hybrid_rrf": hybrid}
    if with_rerank:  # chỉ tạo vỏ bọc; model được nạp khi truy vấn đầu tiên
        out.update({"dense_rerank": RerankRetriever(dense), "bm25_rerank": RerankRetriever(lexical),
                    "hybrid_rerank": RerankRetriever(hybrid)})
    return out
