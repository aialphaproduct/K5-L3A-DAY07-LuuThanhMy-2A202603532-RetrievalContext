"""Đường dẫn và tham số dùng chung. Dữ liệu Day 6 được đọc trực tiếp từ repo Day 6 (không sửa)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAY6_DIR = Path(os.environ.get("DAY6_DIR", ROOT.parent / "K5-L3A-DAY06-LuuThanhMy-2A202603532-DataIngestion"))
DAY6_STARTER = DAY6_DIR / "starter"
DAY6_CHUNKS = DAY6_DIR / "results" / "chunks_struct.jsonl"   # cấu hình B/C của Day 6
BENCHMARK = ROOT / "benchmark" / "questions.jsonl"
CHROMA_DIR = ROOT / "store" / "chroma_db"                   # vector store Day 7 (dựng bằng code Day 6)
RESULTS = ROOT / "results"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"          # cùng model Day 6
CHROMA_INDEX = "struct"                                     # collection day06_struct
LLM_MODEL = "claude-haiku-5-5"
# File .env chứa ANTHROPIC_API_KEY (không commit). Mặc định: .env của repo Day 6 đang dùng key; có thể đổi bằng biến AGENT_ENV_FILE.
AGENT_ENV_FILE = Path(os.environ.get("AGENT_ENV_FILE", DAY6_DIR.parent / "K5-L3A-Day06-LuuThanhMy-2A202603532-DataIngestionEngineering" / "starter" / ".env"))
