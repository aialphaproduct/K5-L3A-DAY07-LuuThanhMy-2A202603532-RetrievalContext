"""Tổng hợp chỉ số tự động từ trace agent (không thay nhãn chấm thủ công).

Tự động: Context Evidence Rate, tool use, trạng thái, abstention, token, thời gian.
Thủ công (không tính ở đây): grounded answer 0-2 và citation hỗ trợ — xem results/audit/.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .config import RESULTS
from .data import cases, chunk_hit, load_chunks


def summarize(split: str, config: str, traces_dir: Path | None = None) -> dict:
    base = traces_dir or (RESULTS / "agent" / split)
    traces = json.loads((base / f"{config}.json").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in load_chunks()}
    case_by_id = {c.id: c for c in cases(split)}
    out = {"split": split, "config": config, "n": len(traces), "errors": 0, "answerable": 0, "unanswerable": 0,
           "context_evidence": 0, "answered_answerable": 0, "abstain_answerable": 0,
           "abstain_unanswerable": 0, "answered_unanswerable": 0, "tool_use": 0, "no_error_runs": 0,
           "searches_dist": {}, "citation_total": 0, "citation_in_context": 0,
           "citation_with_evidence_chunk": 0, "answers_with_citation": 0, "answered": 0,
           "input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
    times, per_query = [], []
    for t in traces:
        case = case_by_id[t["query_id"]]
        if t["status"] == "error":
            out["errors"] += 1
            continue
        out["no_error_runs"] += 1
        searches = [e for e in t["events"] if e["event"] == "search"]
        out["tool_use"] += bool(searches)
        out["searches_dist"][str(len(searches))] = out["searches_dist"].get(str(len(searches)), 0) + 1
        final_ids = searches[-1]["context_ids"] if searches else []
        ctx_hit = any(chunk_hit(by_id[i], case.gold) for i in final_ids) if case.answerable else None
        for k in ("input_tokens", "output_tokens", "llm_calls"):
            out[k] += t["usage"][k]
        times.append(t["elapsed_ms"])
        if t["status"] == "answered":
            out["answered"] += 1
            out["answers_with_citation"] += bool(t["citations"])
            out["citation_total"] += len(t["citations"])
            out["citation_in_context"] += sum(c in final_ids for c in t["citations"])
            if case.answerable:
                out["citation_with_evidence_chunk"] += sum(chunk_hit(by_id[c], case.gold) for c in t["citations"] if c in by_id)
        if case.answerable:
            out["answerable"] += 1
            out["context_evidence"] += bool(ctx_hit)
            out["answered_answerable" if t["status"] == "answered" else "abstain_answerable"] += 1
        else:
            out["unanswerable"] += 1
            out["abstain_unanswerable" if t["status"] == "abstain" else "answered_unanswerable"] += 1
        per_query.append({"query_id": t["query_id"], "status": t["status"], "context_evidence": ctx_hit,
                          "n_search": len(searches), "citations": t["citations"]})
    out["median_ms"] = round(statistics.median(times), 1) if times else None
    out["per_query"] = per_query
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--no-chart", action="store_true", help="Không vẽ biểu đồ cột trong terminal")
    a = p.parse_args()
    res = {c: summarize(a.split, c) for c in a.configs}
    out = RESULTS / "agent" / f"{a.split}_summary.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c, r in res.items():
        keys = ("answerable", "context_evidence", "answered_answerable", "abstain_answerable",
                "abstain_unanswerable", "answered_unanswerable", "tool_use", "no_error_runs", "errors",
                "citation_total", "citation_in_context", "citation_with_evidence_chunk",
                "input_tokens", "output_tokens", "llm_calls", "median_ms")
        print(c, {k: r[k] for k in keys})
    print("->", out)
    if not a.no_chart:
        from .chart import agent_chart, color_default, emit
        emit(agent_chart(color_default()))


if __name__ == "__main__":
    main()
