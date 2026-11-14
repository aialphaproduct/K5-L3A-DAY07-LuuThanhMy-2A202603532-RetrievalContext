# agentrag

Code Day 7 (agentic RAG trên dữ liệu Day 6). Cách chạy: xem [RUN.md](../RUN.md).

| File | Công dụng |
|---|---|
| `index.py` | Dựng vector store từ chunk Day 6 (embedding e5 + ChromaStore của Day 6) |
| `retrieval.py` | Dense, BM25, Hybrid RRF và rerank (cross-encoder) |
| `agent.py` | Agent dùng model thật, tool search_documents / final_answer |
| `eval_retrieval.py`, `run_agent_eval.py`, `summarize.py` | Đo retrieval và agent, tổng hợp |
| `chart.py` | Biểu đồ cột so sánh trong terminal |
