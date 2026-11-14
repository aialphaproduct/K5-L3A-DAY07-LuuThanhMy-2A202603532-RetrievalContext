# Cách chạy Day 7

Python 3.13 (3.11+). Repo Day 6 phải nằm cạnh repo này: `../K5-L3A-DAY06-LuuThanhMy-2A202603532-DataIngestion` (đọc `results/chunks_struct.jsonl` và code `starter/src`). Đổi bằng biến `DAY6_DIR`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install chromadb sentence-transformers anthropic python-dotenv pypdf
```

Khóa API: đặt `ANTHROPIC_API_KEY` vào file `.env` (không commit; `.env` đã nằm trong `.gitignore`). Mặc định code đọc file ghi trong `AGENT_ENV_FILE` (xem `agentrag/config.py`).

```bash
# 1. Dựng vector store (e5-small + ChromaDB, dùng code Day 6)
python -m agentrag.index --reset

# 2. Retrieval không LLM (dense, bm25, hybrid RRF); chỉ chạy test sau khi chốt cấu hình
python -m agentrag.eval_retrieval --split dev
python -m agentrag.eval_retrieval --split dev --budget 2400 --tag dev-budget2400

# 3. Agent thật (cần API key)
python -m agentrag.run_agent_eval --split dev --configs base imp1 imp2 imp3
python -m agentrag.summarize --split dev --configs base imp1 imp2 imp3

# 4. Test — chạy một lần sau khi chốt (results/SELECTION_DECISION.md)
python -m agentrag.run_agent_eval --split test --configs base imp3
python -m agentrag.summarize --split test --configs base imp3

# 5. Metrics từ nhãn đã chấm
python starter/agent_metrics.py --cases results/audit/cases.json \
  --traces results/agent/test/imp3.json --reviews results/audit/test_reviews_imp3.json \
  --split test --output results/agent/test_metrics_imp3.json
```

Test của starter (mock, chỉ kiểm code): `python -m unittest discover -s tests -v`.

Vector store (`store/`) và `.venv/` không được commit; tái tạo bằng bước 1.

## Demo

```bash
.venv/Scripts/python demo/server.py   # http://127.0.0.1:8770, xem DEMO.md
```

Trong demo, mục **Chạy terminal** có sẵn từng lệnh với nút Chép để dán vào terminal.

## Biểu đồ so sánh trong terminal

```bash
.venv/Scripts/python -m agentrag.chart            # retrieval + agent, dev và test
```

`eval_retrieval` và `summarize` cũng tự vẽ biểu đồ sau khi in số (tắt bằng `--no-chart`).
