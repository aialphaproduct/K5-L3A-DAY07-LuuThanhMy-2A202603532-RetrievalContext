"""Đánh giá retrieval (không có LLM). Metric: Hit@1, Evidence Hit@3, Context Evidence Rate.

Câu không có đáp án không nằm trong mẫu số. Chạy: python -m agentrag.eval_retrieval --split dev
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .config import RESULTS
from .data import cases, chunk_hit
from .retrieval import infer_filters, make_retrievers, pack_context

DEFAULT_BUDGET = 800  # ký tự, cùng mặc định starter


def run(split: str, methods: list[str] | None, use_filter: bool, budget: int = DEFAULT_BUDGET,
        rerank: bool = False) -> dict:
    retrievers = make_retrievers(with_rerank=rerank)
    names = methods or list(retrievers)
    report = {"split": split, "budget_chars": budget, "filter_inferred": use_filter, "methods": {}}
    answerable = [c for c in cases(split) if c.answerable]
    for name in names:
        r = retrievers[name]
        traces, times = [], []
        for case in answerable:
            filters = infer_filters(case.question) if use_filter else None
            t0 = time.perf_counter()
            res = r.search(case.question, filters=filters, top_k=3)
            pool = getattr(r, "last_pool", None)
            times.append((time.perf_counter() - t0) * 1000)
            context, ids = pack_context(res, budget)
            marked = [{**c, "hit": chunk_hit(c, case.gold)} for c in res]
            traces.append({"query_id": case.id, "question": case.question, "filter": filters,
                           "results": marked, "context_ids": ids, "context_chars": len(context),
                           "candidate_hit": (any(chunk_hit(c, case.gold) for c in pool) if pool is not None else None),
                           "hit1": bool(marked and marked[0]["hit"]),
                           "hit3": any(m["hit"] for m in marked),
                           "context_hit": any(m["hit"] and m["id"] in ids for m in marked)})
        n = len(answerable)
        summary = {"n_answerable": n,
                   "hit1": sum(t["hit1"] for t in traces), "evidence_hit3": sum(t["hit3"] for t in traces),
                   "context_evidence_rate": sum(t["context_hit"] for t in traces),
                   "median_ms": round(statistics.median(times), 1), "service_cost_usd": 0.0}
        if traces and traces[0]["candidate_hit"] is not None:
            summary["candidates"] = getattr(r, "candidates", None)
            summary["candidate_recall"] = sum(bool(t["candidate_hit"]) for t in traces)
        report["methods"][name] = {"summary": summary, "traces": traces}
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--methods", nargs="*")
    p.add_argument("--filter", action="store_true", help="Bật suy luận filter document_id từ câu hỏi")
    p.add_argument("--tag", default="")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--rerank", action="store_true", help="Thêm 3 phương pháp +rerank (cross-encoder) để so sánh trước/sau")
    p.add_argument("--no-chart", action="store_true", help="Không vẽ biểu đồ cột trong terminal")
    a = p.parse_args()
    report = run(a.split, a.methods, a.filter, a.budget, a.rerank)
    out = RESULTS / "retrieval" / f"{a.split}{('-' + a.tag) if a.tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in report["methods"].items()}, ensure_ascii=False, indent=2))
    print("->", out)
    if not a.no_chart:
        from .chart import color_default, emit, retrieval_report_chart
        emit(retrieval_report_chart(report, color_default()))


if __name__ == "__main__":
    main()
