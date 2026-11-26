from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from brain.agent import SleepingBrain
from brain.teacher_distill import load_teacher_dialogues


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

                if parsed.path == "/api/teacher_dialogues":
                    qs = parse_qs(parsed.query)
                    try:
                        limit = max(1, min(500, int(qs.get("limit", ["80"])[0])))
                    except Exception:
                        limit = 80
                    all_pairs = load_teacher_dialogues(brain.state_dir)
                    shown = all_pairs[-limit:]
                    self._send_json(
                        {
                            "ok": True,
                            "total": len(all_pairs),
                            "dialogues": [{"q": q, "a": a} for q, a in shown],
                        }
                    )
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

                    if parsed.path == "/api/train_language":
                        epochs = int(body.get("epochs", brain.cfg.language_train_epochs))
                        res = brain.train_language_now(epochs=epochs)
                        self._send_json({"ok": True, **res})
                        return

                    if parsed.path == "/api/train_parquet":
                        filename = str(body.get("filename", "__ALL__"))
                        max_steps = int(body.get("max_steps", 5000))
                        batch_size = body.get("batch_size")
                        lr = body.get("lr")
                        res = brain.start_parquet_polish_async(
                            filename=filename,
                            max_steps=max_steps,
                            batch_size=int(batch_size) if batch_size is not None else None,
                            lr=float(lr) if lr is not None else None,
                        )
                        self._send_json({"ok": True, **res})
                        return

                    if parsed.path == "/api/train_teacher":
                        num_dialogues = int(body.get("num_dialogues", 80))
                        train_epochs = int(body.get("train_epochs", 160))
                        skip_generation = bool(body.get("skip_generation", False))
                        res = brain.start_teacher_distill_async(
                            num_dialogues=num_dialogues,
                            train_epochs=train_epochs,
                            skip_generation=skip_generation,
                        )
                        self._send_json({"ok": True, **res})
                        return

                    if parsed.path == "/api/training_feed/clear":
                        brain.clear_training_dialogue_feed()
                        self._send_json({"ok": True})
                        return

                    if parsed.path == "/api/study/continuous":
                        mins = int(body.get("duration_minutes", 60))
                        res = brain.start_continuous_learning_async(duration_minutes=mins)
                        self._send_json({"ok": True, **res})
                        return

                    if parsed.path == "/api/study/turbo":
                        count = int(body.get("topics_count", 200))
                        res = brain.start_turbo_learn_async(topics_count=count)
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
                        if "creativity_level" in body:
                            lvl = max(0.0, min(1.0, float(body["creativity_level"])))
                            brain.creativity_level = lvl
                            brain.cfg.creativity_level = lvl
                        self._send_json({
                            "ok": True,
                            "auto_sleep": brain.auto_sleep,
                            "auto_web": brain.auto_web,
                            "creativity_level": brain.creativity_level,
                            "creativity_score": brain.creativity_score(),
                        })
                        return

                    if parsed.path == "/api/living/budget":
                        b_mb = body.get("memory_budget_mb")
                        t_dev = body.get("target_device")
                        res = brain.living_brain.set_memory_budget(
                            budget_mb=float(b_mb) if b_mb is not None else None,
                            target_device=str(t_dev) if t_dev is not None else None,
                        )
                        self._send_json({"ok": True, **res})
                        return

                    if parsed.path == "/api/living/stimulate":
                        steps = max(1, min(20, int(body.get("steps", 5))))
                        force_grow = bool(body.get("force_grow", False))
                        born_total = 0
                        for _ in range(steps):
                            stim = torch.randn(brain.living_brain.sensory_dim)
                            tel = brain.living_brain.cognitive_step(
                                sensory_vec=stim,
                                external_valence=0.30,
                                allow_growth=True,
                                spontaneous=False,
                            )
                            born_total += tel.neurons_added
                        if force_grow and born_total == 0:
                            born_total += brain.living_brain.grow_structure()
                        self._send_json({
                            "ok": True,
                            "neurons_added": born_total,
                            "living_brain": brain.living_brain.snapshot(),
                        })
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
        self.brain.living_brain.start_spontaneous_clock(interval_sec=0.65)
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
            self.brain.living_brain.stop_spontaneous_clock()
            self.brain.stop_study()
            self.brain.save_state()
            self.brain.language_cortex.shutdown()
            httpd.server_close()
            print("Zapisano. Do zobaczenia!")
