"""Biểu đồ cột vẽ ngay trong terminal để so sánh các phương pháp / cấu hình.

    python -m agentrag.chart                    # retrieval + agent, dev và test
    python -m agentrag.chart --what retrieval   # chỉ retrieval
    python -m agentrag.chart --what agent       # chỉ agent
    python -m agentrag.chart --no-color         # không dùng màu (khi dán ra văn bản)

Số liệu đọc từ results/ (không tính lại, không gọi model). ★ = tốt nhất trong nhóm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable

from .config import RESULTS

WIDTH = 26
NAMES = {"dense": "Dense", "bm25": "BM25", "hybrid_rrf": "Hybrid RRF"}
METHODS = ("dense", "bm25", "hybrid_rrf")
CONFIGS = ("base", "imp1", "imp2", "imp3", "imp4")
CFG_NOTE = {"base": "dense · 800", "imp1": "hybrid · 800", "imp2": "hybrid · 2400", "imp3": "hybrid · 2400 · filter", "imp4": "hybrid · 2400 · rerank"}

_CODES = {"green": "32", "red": "31", "cyan": "36", "dim": "90", "bold": "1", "yellow": "33"}


def paint(text: str, style: str, on: bool) -> str:
    return f"\033[{_CODES[style]}m{text}\033[0m" if on else text


def emit(text: str) -> None:
    """In UTF-8 an toàn trên terminal Windows."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print(text)


def color_default() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _bar(value: float, maximum: float, on: bool, style: str) -> str:
    cells = 0 if maximum <= 0 else max(1 if value > 0 else 0, round(WIDTH * value / maximum))
    return paint("█" * cells, style, on) + paint("░" * (WIDTH - cells), "dim", on)


def block(title: str, rows: list[tuple[str, float]], maximum: float, fmt: Callable[[float], str],
          on: bool, better: str = "max", hint: str = "") -> str:
    """Một nhóm cột. better='max' hoặc 'min' quyết định cột nào được đánh ★."""
    values = [v for _, v in rows]
    target = max(values) if better == "max" else min(values)
    lines = [paint(title, "bold", on) + (paint("  " + hint, "dim", on) if hint else "")]
    width = max(len(label) for label, _ in rows)
    for label, v in rows:
        best = v == target and len(set(values)) > 1
        style = "green" if best else "cyan"
        lines.append(f"  {label:<{width}}  {_bar(v, maximum, on, style)}  {fmt(v):<10}" + (paint(" ★", "yellow", on) if best else ""))
    return "\n".join(lines)


def _frac(n_total: int) -> Callable[[float], str]:
    return lambda v: f"{int(v)}/{n_total} {round(100 * v / n_total):>3d}%"


def _load(rel: str):
    path = RESULTS / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _title(text: str, on: bool) -> str:
    bar = "═" * 74
    return "\n" + paint(bar, "dim", on) + "\n" + paint(text, "bold", on) + "\n" + paint(bar, "dim", on)


# ------------------------------------------------------------------ retrieval
def retrieval_blocks(split: str, by_budget: dict[int, dict], on: bool) -> str:
    """by_budget: {budget: {method: summary}}. Metric Hit@k lấy từ budget đầu tiên (không phụ thuộc budget)."""
    first = next(iter(by_budget.values()))
    n = first["dense"]["n_answerable"]
    parts = []
    for key, title, hint in (("hit1", "Hit@1  — đoạn hạng 1 chứa evidence", "càng cao càng tốt"),
                             ("evidence_hit3", "Evidence Hit@3 — top-3 có evidence", "càng cao càng tốt")):
        parts.append(block(title, [(NAMES[m], first[m][key]) for m in METHODS], n, _frac(n), on, hint=hint))
    for budget, methods in by_budget.items():
        parts.append(block(f"Ngữ cảnh còn evidence — budget {budget} ký tự",
                           [(NAMES[m], methods[m]["context_evidence_rate"]) for m in METHODS], n, _frac(n), on,
                           hint="chunk ~1000 ký tự không vừa thì bị bỏ"))
    ms = [(NAMES[m], first[m]["median_ms"]) for m in METHODS]
    parts.append(block("Thời gian tìm trung vị (mili-giây)", ms, max(v for _, v in ms), lambda v: f"{v:.1f} ms", on,
                       better="min", hint="càng thấp càng nhanh"))
    return "\n\n".join(parts)


def retrieval_chart(on: bool) -> str:
    dev800, dev2400, test800 = _load("retrieval/dev.json"), _load("retrieval/dev-dev-budget2400.json"), _load("retrieval/test.json")
    if not dev800:
        return "Chưa có kết quả retrieval. Chạy: python -m agentrag.eval_retrieval --split dev"
    out = [_title("RETRIEVAL · DEV (câu có đáp án) — Dense vs BM25 vs Hybrid RRF", on)]
    budgets = {800: {k: v["summary"] for k, v in dev800["methods"].items()}}
    if dev2400:
        budgets[2400] = {k: v["summary"] for k, v in dev2400["methods"].items()}
    out.append(retrieval_blocks("dev", budgets, on))
    if test800:
        out.append(_title("RETRIEVAL · TEST (chạy sau khi chốt, chỉ 5 câu nên chênh 1 câu là chưa kết luận)", on))
        out.append(retrieval_blocks("test", {800: {k: v["summary"] for k, v in test800["methods"].items()}}, on))
    return "\n".join(out)


def retrieval_report_chart(report: dict, on: bool) -> str:
    """Biểu đồ cho đúng một lần chạy eval_retrieval (một split, một budget)."""
    summaries = {k: v["summary"] for k, v in report["methods"].items()}
    if not all(m in summaries for m in METHODS):
        return ""
    if "hybrid_rerank" in summaries:
        return _title(f"RERANK · {report['split'].upper()} — trước / sau · budget {report['budget_chars']} ký tự", on) + "\n" + rerank_blocks(summaries, report["budget_chars"], on)
    head = _title(f"BIỂU ĐỒ SO SÁNH · {report['split'].upper()} · budget {report['budget_chars']} ký tự", on)
    return head + "\n" + retrieval_blocks(report["split"], {report["budget_chars"]: summaries}, on)



# ------------------------------------------------------------------ rerank: trước / sau
RERANK_FILE = "retrieval/dev-rerank-budget2400.json"
RK = {"dense": "Dense", "bm25": "BM25", "hybrid_rrf": "Hybrid RRF"}


def rerank_blocks(methods: dict, budget: int, on: bool) -> str:
    """methods: {tên: summary}; cần cả m và m_rerank. Hàng 'trước' = retriever gốc, 'sau' = + rerank."""
    base = {k.replace("_rerank", "_rrf") if k == "hybrid_rerank" else k.replace("_rerank", ""): k for k in methods if k.endswith("_rerank")}
    rows = lambda key: [(f"{RK[b]} {tag}", methods[name][key]) for b, rk in base.items()
                        for tag, name in (("(trước)", b), ("+ rerank (sau)", rk))]
    n = methods["dense"]["n_answerable"]
    parts = [block("Hit@1 — đoạn hạng 1 chứa evidence", rows("hit1"), n, _frac(n), on, hint="trước = retriever gốc; sau = lấy 20 candidate rồi xếp lại"),
             block("Evidence Hit@3 — top-3 có evidence", rows("evidence_hit3"), n, _frac(n), on),
             block(f"Ngữ cảnh còn evidence — budget {budget} ký tự", rows("context_evidence_rate"), n, _frac(n), on)]
    cand = [(f"{RK[b]} (top-20)", methods[rk]["candidate_recall"]) for b, rk in base.items() if "candidate_recall" in methods[rk]]
    if cand:
        parts.append(block("Trần của rerank: evidence có trong 20 candidate", cand, n, _frac(n), on,
                           hint="rerank không cứu được câu mà candidate không chứa evidence"))
    ms = rows("median_ms")
    parts.append(block("Thời gian trung vị (ms) — chi phí của rerank", ms, max(v for _, v in ms), lambda v: f"{v:.0f} ms", on, better="min",
                       hint="cross-encoder chạy CPU"))
    return "\n\n".join(parts)


RERANK_TEST_FILE = "retrieval/test-rerank-budget2400.json"


def rerank_chart(on: bool) -> str:
    out = _rerank_dev_chart(on)
    test = _load(RERANK_TEST_FILE)
    if test:
        methods = {k: v["summary"] for k, v in test["methods"].items()}
        out += "\n" + _title("RERANK · TEST — xem sau khi chốt cấu hình (5 câu, không dùng để đổi lựa chọn)", on) + "\n" + rerank_blocks(methods, test["budget_chars"], on)
    return out


def _rerank_dev_chart(on: bool) -> str:
    data = _load(RERANK_FILE)
    if not data:
        return "Chưa có kết quả rerank. Chạy: python -m agentrag.eval_retrieval --split dev --rerank --budget 2400 --tag rerank-budget2400"
    methods = {k: v["summary"] for k, v in data["methods"].items()}
    return _title("RERANK · DEV — trước / sau khi xếp lại bằng cross-encoder (BAAI/bge-reranker-v2-m3)", on) + "\n" + rerank_blocks(methods, data["budget_chars"], on)

# ------------------------------------------------------------------ agent
def agent_chart(on: bool) -> str:
    out = []
    dev = _load("agent/dev_summary.json")
    if dev:
        cfgs = [c for c in CONFIGS if c in dev]
        n, u = dev[cfgs[0]]["answerable"], dev[cfgs[0]]["unanswerable"]
        out.append(_title("AGENT THẬT · DEV — 4 cấu hình (mỗi dòng đổi một yếu tố)", on))
        label = lambda c: f"{c} ({CFG_NOTE[c]})"
        out.append(block("Evidence còn trong ngữ cảnh gửi model", [(label(c), dev[c]["context_evidence"]) for c in cfgs], n, _frac(n), on,
                         hint="càng cao càng tốt"))
        out.append(block("Trả lời được câu có đáp án", [(label(c), dev[c]["answered_answerable"]) for c in cfgs], n, _frac(n), on))
        out.append(block("Từ chối nhầm (câu có đáp án mà agent từ chối)", [(label(c), dev[c]["abstain_answerable"]) for c in cfgs], n, _frac(n), on,
                         better="min", hint="càng thấp càng tốt"))
        out.append(block("Từ chối đúng (câu thiếu bằng chứng)", [(label(c), dev[c]["abstain_unanswerable"]) for c in cfgs], u, _frac(u), on))
        tok = [(label(c), dev[c]["input_tokens"] / 1000) for c in cfgs]
        out.append(block("Token đầu vào (nghìn) — chi phí", tok, max(v for _, v in tok), lambda v: f"{v:.0f}k", on, better="min",
                         hint="càng thấp càng rẻ"))
    test = _load("agent/test_summary.json")
    if test:
        cfgs = [c for c in ("base", "imp3", "imp4") if c in test]
        n, u = test[cfgs[0]]["answerable"], test[cfgs[0]]["unanswerable"]
        out.append(_title("AGENT THẬT · TEST — base, imp3 (đã chọn), imp4 (rerank, xem sau khi chốt)", on))
        out.append(block("Evidence còn trong ngữ cảnh", [(c, test[c]["context_evidence"]) for c in cfgs], n, _frac(n), on))
        out.append(block("Từ chối nhầm", [(c, test[c]["abstain_answerable"]) for c in cfgs], n, _frac(n), on, better="min"))
        out.append(block("Từ chối đúng (câu thiếu bằng chứng)", [(c, test[c]["abstain_unanswerable"]) for c in cfgs], u, _frac(u), on))
        means = []
        for c in cfgs:
            m = _load(f"agent/test_metrics_{c}.json")
            if m and m["grounded_answer"]["mean"] is not None:
                means.append((c, m["grounded_answer"]["mean"]))
        if means:
            out.append(block("Grounded answer trung bình (thang 0–2, nhãn nháp AI)", means, 2.0, lambda v: f"{v:.2f}/2", on,
                             hint="càng cao càng tốt"))
    return "\n".join(out) if out else "Chưa có kết quả agent. Chạy: python -m agentrag.run_agent_eval --split dev --configs base imp1 imp2 imp3"


def main() -> None:
    p = argparse.ArgumentParser(description="Vẽ biểu đồ cột so sánh ngay trong terminal")
    p.add_argument("--what", choices=("all", "retrieval", "rerank", "agent"), default="all")
    p.add_argument("--no-color", action="store_true")
    a = p.parse_args()
    on = color_default() and not a.no_color
    parts = []
    if a.what in ("all", "retrieval"):
        parts.append(retrieval_chart(on))
    if a.what in ("all", "rerank"):
        parts.append(rerank_chart(on))
    if a.what in ("all", "agent"):
        parts.append(agent_chart(on))
    emit("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
