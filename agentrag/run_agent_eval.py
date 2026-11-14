"""Chạy agent thật trên một split với một hoặc nhiều cấu hình. Mỗi cấu hình ghi một file trace.

Ví dụ:
    python -m agentrag.run_agent_eval --split dev --configs base imp1 imp2 imp3
    python -m agentrag.run_agent_eval --split dev --configs base --ids q01 u01   # kiểm tra nhanh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import AgentConfig, run_agent
from .config import AGENT_ENV_FILE, RESULTS
from .data import cases
from .retrieval import make_retrievers

# Mỗi cấu hình đổi đúng một yếu tố so với cấu hình trước (xem report).
CONFIGS = {
    "base": AgentConfig("base", retriever="dense", budget=800, max_searches=2, use_filter=False),
    "imp1": AgentConfig("imp1", retriever="hybrid_rrf", budget=800, max_searches=2, use_filter=False),
    "imp2": AgentConfig("imp2", retriever="hybrid_rrf", budget=2400, max_searches=2, use_filter=False),
    "imp3": AgentConfig("imp3", retriever="hybrid_rrf", budget=2400, max_searches=2, use_filter=True),
    # Thử nghiệm sau chốt test: thêm rerank cross-encoder (chỉ đánh giá trên dev).
    "imp4": AgentConfig("imp4", retriever="hybrid_rerank", budget=2400, max_searches=2, use_filter=False),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--configs", nargs="+", required=True, choices=sorted(CONFIGS))
    p.add_argument("--ids", nargs="*", help="Chỉ chạy các ID này (kiểm tra nhanh)")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    from dotenv import load_dotenv
    from anthropic import Anthropic
    # Nạp .env của Day 6 (không commit, không in ra) nếu môi trường chưa có key; không ghi key vào repo.
    # override=True: biến rỗng trong môi trường sẽ được thay bằng giá trị trong .env.
    load_dotenv(AGENT_ENV_FILE, override=True)
    client = Anthropic()
    retrievers = make_retrievers(with_rerank=any("rerank" in CONFIGS[c].retriever for c in a.configs))
    selected = [c for c in cases(a.split) if not a.ids or c.id in a.ids]
    for name in a.configs:
        cfg = CONFIGS[name]
        retriever = retrievers[cfg.retriever]
        traces = []
        for case in selected:
            trace = run_agent(case.id, case.question, cfg, retriever, client)
            trace["split"] = case.split
            traces.append(trace)
            print(f"{name} {case.id} {trace['status']} calls={trace['usage']['llm_calls']} "
                  f"searches={sum(e['event'] == 'search' for e in trace['events'])}"
                  + (f" ERR={trace['error']}" if trace['error'] else ""), flush=True)
        out = RESULTS / "agent" / f"{a.split}{('-' + a.tag) if a.tag else ''}" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("->", out)


if __name__ == "__main__":
    main()
