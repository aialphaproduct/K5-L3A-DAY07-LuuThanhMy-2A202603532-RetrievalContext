"""Demo Day 7: chạy agent thật (có phát trực tiếp từng bước) và đọc số liệu thật từ results/.

    python demo/server.py            # http://127.0.0.1:8770
Chỉ lắng nghe 127.0.0.1. Khóa API đọc từ file .env (không bao giờ trả về trình duyệt).
Model embedding và retriever được nạp sẵn ở nền khi khởi động để lần bấm đầu không phải chờ lâu.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentrag.config import AGENT_ENV_FILE, RESULTS  # noqa: E402
from agentrag.data import load_chunks, load_questions  # noqa: E402

PORT = int(os.environ.get("PORT", "8770"))  # đổi cổng: đặt biến PORT
CONFIG_NAME = "imp3"  # cấu hình đã chốt (hybrid RRF · budget 2400)
CONFIGS_OK = ("base", "imp1", "imp2", "imp3", "imp4")
PRESETS = [
    ("q14", "ok", "Có bằng chứng"),
    ("u03", "abstain", "Thiếu bằng chứng"),
    ("q16", "error", "Có lỗi"),
]
CHUNKS = {c["id"]: c for c in load_chunks()}
QUESTIONS = {q["id"]: q for q in load_questions()}
_state = {"init_lock": threading.Lock(), "run_lock": threading.Lock(), "client": None, "retrievers": None,
          "loading": False, "error": None, "has_key": None}


def _init() -> None:
    """Nạp client Anthropic và retriever một lần (mất vài chục giây vì model embedding, reranker nạp lười)."""
    with _state["init_lock"]:
        if _state["client"] is not None:
            return
        _state["loading"], _state["error"] = True, None
        try:
            from dotenv import load_dotenv
            from anthropic import Anthropic
            from agentrag.retrieval import make_retrievers
            load_dotenv(AGENT_ENV_FILE, override=True)
            _state["has_key"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
            retrievers = make_retrievers(with_rerank=True)
            _state["retrievers"] = retrievers
            _state["client"] = Anthropic()
        except Exception as exc:  # báo lỗi thật lên giao diện, không thay bằng kết quả giả
            _state["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _state["loading"] = False


def status() -> dict:
    return {"ready": _state["client"] is not None, "loading": _state["loading"], "has_key": _state["has_key"],
            "error": _state["error"]}


def enrich_step(e: dict) -> dict:
    kept = set(e["context_ids"])
    return {"query": e["query"], "filter": e.get("filter"), "context_chars": e["context_chars"],
            "results": [{**r, "short": CHUNKS[r["id"]]["text"][:220].replace("\n", " "), "in_context": r["id"] in kept}
                        for r in e["results"]],
            "context_ids": e["context_ids"]}


def enrich(trace: dict, budget: int) -> dict:
    """Thêm văn bản chunk và cờ 'vào ngữ cảnh' để giao diện vẽ được từng bước."""
    steps = [enrich_step(e) for e in trace["events"] if e["event"] == "search"]
    last_ids = steps[-1]["context_ids"] if steps else []
    return {"query_id": trace.get("query_id"), "status": trace["status"], "answer": trace["answer"],
            "citations": trace["citations"], "steps": steps, "budget": budget,
            "context": [{"id": i, "page": CHUNKS[i]["page"], "source_id": CHUNKS[i]["source_id"]} for i in last_ids],
            "usage": trace["usage"], "elapsed_ms": trace["elapsed_ms"], "error": trace.get("error")}


def replay(case_id: str) -> dict:
    """Phát lại trace đã lưu (dự phòng khi không có khóa API / mất mạng)."""
    traces = json.loads((RESULTS / "agent" / "test" / f"{CONFIG_NAME}.json").read_text(encoding="utf-8"))
    trace = next(t for t in traces if t["query_id"] == case_id)
    out = enrich(trace, trace["settings"]["budget"])
    q = QUESTIONS[case_id]
    out["question"] = q["question"]
    out["expected"] = q["expected_answer"] or None  # chỉ để trình bày ca mẫu, không đưa vào agent
    out["mode"] = "replay"
    return out


def run_live(question: str, config: str, on_event=None) -> tuple[dict, object]:
    """Chạy agent thật. Trả (trace, cfg). Ném lỗi nếu chưa nạp được."""
    _init()
    from agentrag.agent import run_agent
    from agentrag.run_agent_eval import CONFIGS
    cfg = CONFIGS[config]
    with _state["run_lock"]:  # chạy lần lượt: retriever và reranker dùng chung bộ nhớ
        trace = run_agent("live", question, cfg, _state["retrievers"][cfg.retriever], _state["client"], on_event=on_event)
    return trace, cfg


def live(question: str, config: str = "imp3") -> dict:
    trace, cfg = run_live(question, config)
    if trace["status"] == "error":
        return {"error": "Không gọi được model. Kiểm tra khóa API trong .env.", "detail": trace["error"]}
    out = enrich(trace, cfg.budget)
    out["question"], out["expected"], out["mode"], out["config"] = question, None, "live", config
    out["settings"] = trace["settings"]
    return out


def overview() -> dict:
    """Số liệu thật đọc từ results/ để mọi tab trong demo đối chiếu được với báo cáo và rubric."""
    def rj(rel):
        return json.loads((RESULTS / rel).read_text(encoding="utf-8"))

    def retr(rel):
        data = rj(rel)
        return {"budget": data["budget_chars"], "filter": data["filter_inferred"],
                "methods": {k: v["summary"] for k, v in data["methods"].items()}}

    def strip(d):
        return {k: {x: y for x, y in v.items() if x != "per_query"} for k, v in d.items()}

    counts: dict = {}
    for q in QUESTIONS.values():
        c = counts.setdefault(q["split"], {"answerable": 0, "unanswerable": 0})
        c["answerable" if q["answerable"] else "unanswerable"] += 1
    has = lambda rel: (RESULTS / rel).exists()
    return {
        "chunks": len(CHUNKS), "counts": counts,
        "retrieval": {"dev800": retr("retrieval/dev.json"), "dev1200": retr("retrieval/dev-dev-budget1200.json"),
                      "dev2400": retr("retrieval/dev-dev-budget2400.json"), "test800": retr("retrieval/test.json"),
                      "dev_filter": retr("retrieval/dev-filter-budget2400.json"),
                      "test_filter": retr("retrieval/test-filter-budget2400.json")},
        "rerank": retr("retrieval/dev-rerank-budget2400.json") if has("retrieval/dev-rerank-budget2400.json") else None,
        "rerank_test": retr("retrieval/test-rerank-budget2400.json") if has("retrieval/test-rerank-budget2400.json") else None,
        "agent_dev": strip(rj("agent/dev_summary.json")), "agent_test": strip(rj("agent/test_summary.json")),
        "metrics_test": {c: rj(f"agent/test_metrics_{c}.json") for c in ("base", "imp3", "imp4")
                         if has(f"agent/test_metrics_{c}.json")},
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path == "/":
            self._send(200, (Path(__file__).with_name("index.html")).read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/assets/day7-end-to-end.svg":  # chỉ phục vụ đúng sơ đồ này
            self._send(200, (ROOT / "assets" / "day7-end-to-end.svg").read_bytes(), "image/svg+xml")
        elif url.path == "/api/overview":
            self._json(200, overview())
        elif url.path == "/api/status":
            self._json(200, status())
        elif url.path == "/api/presets":
            self._json(200, [{"id": i, "kind": k, "label": lb, "question": QUESTIONS[i]["question"],
                              "expected": QUESTIONS[i]["expected_answer"] or None} for i, k, lb in PRESETS])
        elif url.path == "/api/replay":
            cid = (qs.get("id") or [""])[0]
            if cid not in {p[0] for p in PRESETS}:
                self._json(404, {"error": "Không có ca này."})
            else:
                self._json(200, replay(cid))
        elif url.path == "/api/chunk":
            c = CHUNKS.get((qs.get("id") or [""])[0])
            self._json(200, {"id": c["id"], "page": c["page"], "source_id": c["source_id"], "text": c["text"]}) if c \
                else self._json(404, {"error": "Không có đoạn này."})
        else:
            self._json(404, {"error": "Không tìm thấy."})

    def do_HEAD(self) -> None:  # noqa: N802  (kiểm tra sống của công cụ xem trước)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _stream(self, question: str, config: str, preset: str | None) -> None:
        """Phát NDJSON: mỗi lượt tìm của agent thật được gửi ngay khi xảy ra, cuối cùng là kết quả."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(obj) -> None:
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        try:
            if _state["client"] is None:
                send({"type": "status", "message": "Đang nạp model embedding và kho vector (lần đầu mất vài chục giây)…"})
            send({"type": "start", "question": question, "config": config})

            def on_event(e: dict) -> None:
                if e["event"] == "search":
                    send({"type": "search", "step": enrich_step(e)})

            trace, cfg = run_live(question, config, on_event)
            if trace["status"] == "error":
                send({"type": "error", "message": "Không gọi được model. Kiểm tra khóa API trong .env.", "detail": trace["error"]})
                return
            out = enrich(trace, cfg.budget)
            out.update({"question": question, "mode": "live", "config": config, "settings": trace["settings"],
                        "expected": (QUESTIONS[preset]["expected_answer"] or None) if preset else None,
                        "preset": preset})
            send({"type": "final", "result": out})
        except (BrokenPipeError, ConnectionResetError):
            return  # người dùng đóng tab giữa chừng
        except Exception as exc:  # hiển thị lỗi thật, không thay bằng kết quả giả
            try:
                send({"type": "error", "message": "Agent gặp lỗi.", "detail": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in ("/api/ask", "/api/ask_stream"):
            return self._json(404, {"error": "Không tìm thấy."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(min(length, 4000)) or b"{}")
            question = str(body.get("question", "")).strip()
            config = str(body.get("config", "imp3"))
            preset = body.get("preset")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "Dữ liệu gửi lên không hợp lệ."})
        if preset is not None:
            if preset not in {p[0] for p in PRESETS}:
                return self._json(400, {"error": "Ca mẫu không hợp lệ."})
            question = QUESTIONS[preset]["question"]  # câu hỏi lấy từ benchmark, không do trình duyệt gửi
        if not question or len(question) > 500:
            return self._json(400, {"error": "Câu hỏi cần từ 1 đến 500 ký tự."})
        if config not in CONFIGS_OK:
            return self._json(400, {"error": "Cấu hình không hợp lệ."})
        if path == "/api/ask_stream":
            return self._stream(question, config, preset)
        try:
            self._json(200, live(question, config))
        except Exception as exc:
            self._json(500, {"error": "Agent gặp lỗi.", "detail": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args) -> None:  # giữ log gọn
        sys.stderr.write("[demo] " + fmt % args + "\n")


if __name__ == "__main__":
    print(f"Demo: http://127.0.0.1:{PORT}  (Ctrl+C để dừng)")

    def _warm() -> None:
        try:
            _init()
        except Exception:
            pass  # lỗi đã lưu trong _state['error'] và hiện trên giao diện

    threading.Thread(target=_warm, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
