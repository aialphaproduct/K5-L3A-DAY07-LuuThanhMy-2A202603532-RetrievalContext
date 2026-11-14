"""Nạp chunk Day 6, câu hỏi benchmark và hàm đối chiếu evidence (chỉ dùng để chấm, không đưa vào agent)."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .config import BENCHMARK, DAY6_CHUNKS


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def read_jsonl(path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_chunks() -> list[dict[str, Any]]:
    """Chunk Day 6 (struct, tối đa 1000 ký tự) theo schema: id, source_id, page, page_end, text, metadata."""
    out = []
    for row in read_jsonl(DAY6_CHUNKS):
        meta = row.get("metadata", {})
        out.append({"id": row["chunk_id"], "source_id": row["document_id"], "source": row["source"],
                    "page": int(row["page"]), "page_end": int(meta.get("page_end", row["page"])),
                    "text": row["text"], "metadata": meta})
    return out


def load_questions(split: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(BENCHMARK)
    return [q for q in rows if split is None or q["split"] == split]


def chunk_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Chunk chứa evidence nguyên văn (sau chuẩn hóa khoảng trắng) và phủ trang gold."""
    text = norm(chunk["text"])
    for g in gold:
        if chunk["source_id"] == g["source_id"] and chunk["page"] <= g["page"] <= chunk["page_end"] \
                and norm(g["evidence"]) in text:
            return True
    return False


@dataclass
class Case:
    id: str
    split: str
    question: str
    answerable: bool
    gold: list[dict[str, Any]] = field(default_factory=list)
    expected_answer: str = ""


def cases(split: str | None = None) -> list[Case]:
    return [Case(q["id"], q["split"], q["question"], q["answerable"], q.get("gold", []), q.get("expected_answer", ""))
            for q in load_questions(split)]
