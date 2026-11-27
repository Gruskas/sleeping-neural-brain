from __future__ import annotations

import json
import sys
import tempfile
import threading
import types
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import torch

from brain.agent import SleepingBrain
from brain.config import BrainConfig
from brain.language_trainer import build_dialogue_qa_pairs, train_reasoning_with_bielik
from brain.teacher_distill import (
    build_reasoning_prompt_queue,
    count_reasoning_dialogues,
    generate_bielik_reasoning_dialogues,
    is_ai_corporate_reply,
    load_reasoning_dialogues,
    load_reasoning_records,
    reasoning_dialogues_path,
)
from server import BrainServer


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "left"
        self._replies: list[str] = []

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        sys_txt = msgs[0]["content"]
        usr_txt = msgs[1]["content"]
        return f"SYS:{sys_txt}\nUSR:{usr_txt}"

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=448):
        self._last_batch = texts

        class _Enc(dict):
            def to(self, _dev):
                return self

        bs = len(texts)
        return _Enc({"input_ids": torch.zeros((bs, 4), dtype=torch.long)})

    def decode(self, token_ids, skip_special_tokens=True):
        idx = int(token_ids[0].item())
        return self._replies[idx]


class _FakeModel:
    def __init__(self, tok: _FakeTokenizer, responder) -> None:
        self.tok = tok
        self.responder = responder
        self.call_count = 0

    def to(self, _dev):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.call_count += 1
        batch = self.tok._last_batch
        self.tok._replies = [self.responder(txt, self.call_count, i) for i, txt in enumerate(batch)]
        bs = len(batch)
        out = torch.zeros((bs, 6), dtype=torch.long)
        for i in range(bs):
            out[i, 4] = i
        return out


class TestBielikReasoning(unittest.TestCase):
    def test_anti_ai_corporate_filter(self) -> None:
        banned_samples = [
            "Jako AI nie mogę odpowiedzieć na to pytanie.",
            "Jestem modelem językowym i nie mam własnych doświadczeń.",
            "Jako model jezykowy nie posiadam uczuc ani swiadomosci.",
            "Nie mogę jako asystent udzielić takiej informacji.",
            "Zostałem zaprogramowany, aby pomagać użytkownikom.",
            "Nie mam dostępu do internetu ani bieżących danych.",
            "Nie jestem człowiekiem, jestem tylko programem komputerowym.",
        ]
        for s in banned_samples:
            self.assertTrue(is_ai_corporate_reply(s), f"Expected AI cliché rejection for: {s!r}")

        clean_samples = [
            "Nie wiem tego jeszcze — opowiesz mi o tym?",
            "Nie mam tego w pamięci, wytłumaczysz?",
            "Młotek ma teraz Marek.",
            "Prąd nie działa, więc router też się wyłączył i nie ma internetu.",
            "Sztuczna inteligencja pomaga w diagnostyce medycznej.",
            "Nie pamiętam tego jeszcze, ale chętnie się nauczę.",
        ]
        for s in clean_samples:
            self.assertFalse(is_ai_corporate_reply(s), f"Clean reply falsely rejected: {s!r}")

    def test_reasoning_prompt_queue_balanced_and_paired(self) -> None:
        queue = build_reasoning_prompt_queue(num_samples=90)
        self.assertEqual(len(queue), 90)
        counts = {"relational": 0, "causal": 0, "ignorance": 0}
        by_pid: dict[str, list[dict[str, str]]] = {}
        for item in queue:
            cat = item["category"]
            counts[cat] += 1
            if cat == "relational":
                by_pid.setdefault(item["pair_id"], []).append(item)

        self.assertEqual(counts["relational"], 30)
        self.assertEqual(counts["causal"], 30)
        self.assertEqual(counts["ignorance"], 30)
        self.assertEqual(len({it["q"].lower() for it in queue}), 90)

        for pid, pair_items in by_pid.items():
            self.assertEqual(len(pair_items), 2, f"Relational pair {pid} incomplete")
            self.assertNotEqual(pair_items[0]["q"], pair_items[1]["q"])
            self.assertEqual({pair_items[0]["role_order"], pair_items[1]["role_order"]}, {"ab", "ba"})

    def test_generate_filters_ai_cliches_and_retries(self) -> None:
        tok = _FakeTokenizer()

        def _responder(prompt_txt: str, call_idx: int, row_idx: int) -> str:
            # First batch emits corporate AI clichés on rows 0 & 1; retry fixes row 0, keeps cliché on row 1
            if call_idx == 1 and row_idx == 0:
                return "Jako AI nie posiadam dostępu do takich informacji."
            if call_idx == 1 and row_idx == 1:
                return "Jestem modelem językowym i nie mam uczuć."
            if call_idx > 1 and "UWAGA: poprzednia odpowiedź" in prompt_txt:
                if "śniadanie" in prompt_txt.lower() or "kto " not in prompt_txt.lower():
                    return "Nie wiem tego jeszcze — opowiesz mi o tym?"
            if "Kto " in prompt_txt:
                return "Ma go teraz druga osoba ze zdania."
            return "Z powodu braku zasilania urządzenie nie działa. Wniosek jest prosty."

        fake_model = _FakeModel(tok, _responder)
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: tok),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **kw: fake_model),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                res = generate_bielik_reasoning_dialogues(
                    state_dir=tmp,
                    num_samples=6,
                    device=torch.device("cpu"),
                )

            self.assertGreaterEqual(res["filtered_ai"], 1)
            saved_recs = load_reasoning_records(tmp)
            self.assertGreaterEqual(len(saved_recs), 4)
            for rec in saved_recs:
                self.assertFalse(is_ai_corporate_reply(rec["a"]))

            # Even if a banned phrase is manually appended to JSONL, loader must filter it out
            path = reasoning_dialogues_path(tmp)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"q": "Testowe pytanie?", "a": "Jako model językowy nie wiem.", "category": "ignorance"}) + "\n")
            for _q, a in load_reasoning_dialogues(tmp):
                self.assertFalse(is_ai_corporate_reply(a))

    def test_cortex_hippocampus_and_server_endpoint_integration(self) -> None:
        cfg = BrainConfig(
            language_bootstrap_epochs=0,
            num_neurons=512,
            sdr_k=24,
            num_columns=8,
            column_dim=64,
            wernicke_layers=1,
            wernicke_heads=2,
            wernicke_ff_dim=128,
            wernicke_max_patch_slots=16,
            broca_dim=64,
            broca_layers=1,
            broca_heads=2,
            broca_ff_dim=128,
            broca_use_moe=False,
            broca_max_seq_len=64,
            broca_max_new_bytes=24,
            max_input_bytes=64,
            language_train_epochs=3,
            dmn_enabled=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.auto_sleep = False
            brain.auto_web = False

            # 1) Block corporate AI phrases from adapt_qa_pairs_online & Hippocampus
            brain.language_cortex.adapt_qa_pairs_online(
                [("Kim jesteś?", "Jestem sztuczną inteligencją i modelem językowym.")],
                cortex=brain.cortex,
                steps=4,
            )
            self.assertFalse(
                any(is_ai_corporate_reply(a) for _q, a in brain.language_cortex._learned_qa_replay)
            )
            ingest_bad = brain._ingest_teacher_pairs_to_short_term(
                [("Kim jesteś?", "Jako AI nie posiadam uczuć ani świadomości, zostałem zaprogramowany.")]
            )
            self.assertEqual(ingest_bad["engrams_added"], 0)

            # Verify build_dialogue_qa_pairs excludes corporate AI extra pairs
            qa_built = build_dialogue_qa_pairs(
                include_curriculum=False,
                extra_pairs=[
                    ("Co jadłem?", "Jako model językowy nie mam dostępu do Twoich danych."),
                    ("Co jadłem?", "Nie wiem tego jeszcze — opowiesz mi o tym?"),
                ],
            )
            self.assertTrue(any("Nie wiem tego jeszcze" in a for _q, a in qa_built))
            self.assertFalse(any(is_ai_corporate_reply(a) for _q, a in qa_built))

            # 2) Seed valid reasoning records & run train_reasoning_with_bielik(skip_generation=True)
            r_path = reasoning_dialogues_path(brain.state_dir)
            seed_recs = [
                {
                    "q": "Janek pożyczył młotek Markowi. Kto ma teraz młotek?",
                    "a": "Młotek ma teraz Marek.",
                    "category": "relational",
                    "pair_id": "rel_0",
                },
                {
                    "q": "Marek pożyczył młotek Jankowi. Kto ma teraz młotek?",
                    "a": "Młotek ma teraz Janek.",
                    "category": "relational",
                    "pair_id": "rel_0",
                },
                {
                    "q": "Wyłączono prąd w domu, a router działa tylko z gniazdka. Czy działa Wi-Fi?",
                    "a": "Router stracił zasilanie przez brak prądu, więc Wi-Fi nie działa.",
                    "category": "causal",
                    "pair_id": "",
                },
                {
                    "q": "Co dzisiaj jadłem na śniadanie?",
                    "a": "Nie wiem tego jeszcze — opowiesz mi o tym?",
                    "category": "ignorance",
                    "pair_id": "",
                },
            ]
            with r_path.open("w", encoding="utf-8") as fh:
                for rec in seed_recs:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            self.assertEqual(count_reasoning_dialogues(brain.state_dir), 4)
            snap = brain.snapshot()
            self.assertEqual(snap["teacher_distill"]["saved_reasoning"], 4)

            res = train_reasoning_with_bielik(
                brain,
                num_samples=6,
                train_epochs=3,
                skip_generation=True,
            )
            self.assertEqual(res["total_reasoning_dialogues"], 4)
            self.assertTrue(res["train_result"]["trained"])
            self.assertGreaterEqual(len(brain.training_dialogue_feed), 1)

            # 3) Test HTTP endpoint /api/train_reasoning_bielik
            srv = BrainServer(brain=brain, static_dir=Path("static"), host="127.0.0.1", port=0).create_server()
            port = srv.server_address[1]
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/train_reasoning_bielik",
                    data=json.dumps({"num_samples": 6, "train_epochs": 2, "skip_generation": True}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(payload["ok"])
                self.assertTrue(payload["started"])
                self.assertEqual(payload["existing_reasoning"], 4)
                # Wait briefly for background training thread to finish before tempdir cleanup
                for _ in range(100):
                    if not brain.study_state.get("active"):
                        break
                    threading.Event().wait(0.05)
            finally:
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
