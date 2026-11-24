from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from brain.agent import SleepingBrain


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class BrainServer:
    """Pure conversational & autonomous web-study HTTP server (no 2D world loop)."""

    def __init__(self, brain: SleepingBrain, static_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.brain = brain
        self.static_dir = static_dir
        self.host = host
        self.port = port

    def _make_handler(self):
        brain = self.brain
        static_dir = self.static_dir

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(
                self,
                payload: Any,
                status: int = 200,
                extra_headers: dict[str, str] | None = None,
            ) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                if extra_headers:
                    for k, v in extra_headers.items():
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                return json.loads(raw.decode("utf-8"))

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/api/state":
                    self._send_json(brain.snapshot())
                    return

                if parsed.path == "/api/export":
                    self._send_json(
                        brain.export_knowledge(),
                        extra_headers={
                            "Content-Disposition": 'attachment; filename="brain_knowledge_export.json"'
                        },
                    )
                    return

                if parsed.path in ("/", "/index.html"):
                    index_file = static_dir / "index.html"
                    if not index_file.exists():
                        self.send_error(HTTPStatus.NOT_FOUND, "index.html missing")
                        return
                    data = index_file.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                try:
                    body = self._read_json()

                    if parsed.path == "/api/chat":
                        res = brain.chat(body.get("message", ""))
                        self._send_json({"ok": True, **res})
                        return

                    if parsed.path == "/api/chat/stream":
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        try:
                            for ev in brain.chat_stream(body.get("message", "")):
                                frame = f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode("utf-8")
                                self.wfile.write(frame)
                                self.wfile.flush()
                        except Exception as stream_exc:
                            err_frame = f"data: {json.dumps({'type': 'error', 'error': str(stream_exc)}, ensure_ascii=False)}\n\n".encode("utf-8")
                            try:
                                self.wfile.write(err_frame)
                                self.wfile.flush()
                            except Exception:
                                pass
                        return

                    if parsed.path == "/api/study":
                        res = brain.start_study_async(
                            topic=body.get("topic", ""),
                            max_rounds=body.get("max_rounds"),
                            consolidate_after=bool(body.get("consolidate_after", True)),
                        )
                        self._send_json({"ok": True, **res})
                        return

                    if parsed.path == "/api/study/stop":
                        brain.stop_study()
                        self._send_json({"ok": True})
                        return

                    if parsed.path == "/api/curriculum":
                        deep_web = bool(body.get("deep_web", False))
                        max_topics = int(body.get("max_topics", 12))
                        res = brain.start_curriculum_async(deep_web=deep_web, max_topics=max_topics)
                        self._send_json({"ok": True, **res})
                        return

                    if parsed.path == "/api/study/continuous":
                        mins = int(body.get("duration_minutes", 60))
                        res = brain.start_continuous_learning_async(duration_minutes=mins)
                        self._send_json({"ok": True, **res})
                        return

                    if parsed.path == "/api/sleep":
                        report = brain.sleep_now(visual_pacing=True)
                        self._send_json({"ok": True, "report": report.to_dict()})
                        return

                    if parsed.path == "/api/memory/add":
                        mem = brain.add_manual_memory(
                            text=body.get("text", ""),
                            category=body.get("category", "wiedza"),
                            importance=int(body.get("importance", 4)),
                        )
                        self._send_json({"ok": True, "memory": mem})
                        return

                    if parsed.path == "/api/memory/delete":
                        if "id" in body and body["id"]:
                            deleted = brain.delete_memory(str(body["id"]))
                        elif "topic" in body and body["topic"]:
                            deleted = brain.delete_studied_topic(str(body["topic"]))
                        else:
                            raise ValueError("Podaj 'id' wspomnienia lub 'topic' do usunięcia.")
                        self._send_json({"ok": True, "deleted": deleted})
                        return

                    if parsed.path == "/api/control":
                        if "auto_sleep" in body:
                            brain.auto_sleep = bool(body["auto_sleep"])
                        if "auto_web" in body:
                            brain.auto_web = bool(body["auto_web"])
                        self._send_json({"ok": True, "auto_sleep": brain.auto_sleep, "auto_web": brain.auto_web})
                        return

                    if parsed.path == "/api/save":
                        path = brain.save_state()
                        self._send_json({"ok": True, "path": str(path)})
                        return

                    if parsed.path == "/api/reset":
                        brain.reset_brain()
                        self._send_json({"ok": True})
                        return

                    self._send_json({"ok": False, "error": "Unknown endpoint"}, status=404)
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=400)

        return Handler

    def create_server(self) -> _ReusableThreadingHTTPServer:
        return _ReusableThreadingHTTPServer((self.host, self.port), self._make_handler())

    def start(self) -> None:
        httpd = self.create_server()
        print("============================================================")
        print("  SLEEPING NEURAL BRAIN — CZAT + NAUKA Z SIECI (PyTorch GPU)")
        print(f"  GPU   : {self.brain.gpu_info()['device_name']}")
        print(f"  URL   : http://{self.host}:{self.port}")
        print("  CTRL+C aby zapisać stan mózgu i zakończyć.")
        print("============================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nZapisywanie stanu mózgu...")
            self.brain.stop_study()
            self.brain.save_state()
            self.brain.language_cortex.shutdown()
            httpd.server_close()
            print("Zapisano. Do zobaczenia!")
