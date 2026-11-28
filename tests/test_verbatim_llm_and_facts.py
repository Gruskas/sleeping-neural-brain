from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from brain.agent import SleepingBrain, normalize_training_pipeline_plan
from brain.config import BrainConfig
from brain.fact_sources import (
    CURATED_FACT_ENCYCLOPEDIA,
    build_fact_training_pairs,
    count_fact_knowledge_records,
    fetch_curated_polish_facts,
    get_fact_engram_texts,
    load_fact_knowledge_records,
)
from brain.neural_language import (
    BrocaByteTransformer,
    encode_prompt_prefix_bytes,
    format_verbatim_broca_prompt,
)
from brain.sensory_wernicke import BOS_BYTE, EOS_BYTE, PAD_BYTE


def _small_cfg(use_moe: bool = False) -> BrainConfig:
    return BrainConfig(
        num_neurons=512,
        sdr_k=24,
        num_columns=8,
        column_dim=64,
        wernicke_layers=1,
        wernicke_heads=2,
        wernicke_ff_dim=128,
        wernicke_max_patch_slots=16,
        broca_dim=64,
        broca_layers=2,
        broca_heads=2,
        broca_ff_dim=128,
        broca_use_moe=use_moe,
        broca_ffn_mode="moe" if use_moe else "dense",
        broca_num_experts=4,
        broca_moe_top_k=2,
        broca_moe_expert_dim=64,
        broca_max_seq_len=512,
        broca_max_new_bytes=24,
        broca_max_prompt_bytes=256,
        broca_verbatim_min_trained_steps=3,
        max_input_bytes=128,
        language_bootstrap_epochs=0,
        language_train_epochs=3,
        nrem_sleep_steps=2,
        rem_2hop_steps=1,
        dmn_enabled=False,
    )


class TestVerbatimLLMAndFacts(unittest.TestCase):
    def test_01_verbatim_prompt_formatting_and_utf8_safety(self) -> None:
        raw_ctx = (
            "Kontekst rozmowy: Użytkownik: Cześć, mówię o żółwiu i żubrze | "
            "Ja: Żółw to gad, a żubr to ssak. || Odpowiedz na: Który z nich żyje w Puszczy Białowieskiej?"
        )
        formatted = format_verbatim_broca_prompt(
            raw_ctx,
            recalled_facts=["Żubr europejski występuje licznie w Puszczy Białowieskiej."],
            max_bytes=300,
        )
        self.assertIn("[Pamięć]", formatted)
        self.assertIn("U: Cześć", formatted)
        self.assertIn("A: Żółw", formatted)
        self.assertTrue(formatted.endswith("\nA: "))
        self.assertLessEqual(len(formatted.encode("utf-8")), 300)

        # Truncation on tricky 2-byte Polish characters must never fail UTF-8 round-trip
        pb = encode_prompt_prefix_bytes("Zażółć gęślą jaźń " * 20, max_bytes=37)
        self.assertLessEqual(len(pb), 37)
        decoded = bytes(pb).decode("utf-8", errors="strict")
        self.assertTrue(len(decoded) > 0)

    def test_02_broca_prompt_loss_masking_checkpoint_and_prefill(self) -> None:
        cfg = _small_cfg(use_moe=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        broca = BrocaByteTransformer(cfg).to(device)
        self.assertFalse(broca.supports_verbatim_prompt)

        prompt_txt = format_verbatim_broca_prompt("Co to jest fotosynteza?", max_bytes=128)
        pb = encode_prompt_prefix_bytes(prompt_txt, max_bytes=128)
        ans = b"Proces wytwarzania glukozy i tlenu ze swiatla."
        seq = torch.tensor([[BOS_BYTE] + pb + list(ans) + [EOS_BYTE]], dtype=torch.long, device=device)
        p_lens = torch.tensor([len(pb)], dtype=torch.long, device=device)
        z_mem = torch.randn((1, cfg.num_columns, cfg.column_dim), device=device)

        opt = torch.optim.AdamW(broca.parameters(), lr=2e-3)
        broca.train()
        losses: list[float] = []
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            loss = broca.compute_loss(seq, z_mem, decoder_byte_dropout=0.0, prompt_lens=p_lens)
            loss.backward()
            opt.step()
            broca.trained_steps += 1
            broca.verbatim_prompt_steps += 1
            losses.append(float(loss.item()))

        self.assertLess(losses[-1], losses[0])
        self.assertTrue(broca.supports_verbatim_prompt)

        # Export & import state preserves verbatim_prompt_steps; legacy state falls back cleanly
        exported = broca.export_state(include_peer_banks=False)
        broca2 = BrocaByteTransformer(cfg).to(device)
        broca2.import_state(exported, device=device)
        self.assertEqual(broca2.verbatim_prompt_steps, 5)
        self.assertTrue(broca2.supports_verbatim_prompt)

        legacy_state = dict(exported)
        legacy_state.pop("verbatim_prompt_steps", None)
        broca3 = BrocaByteTransformer(cfg).to(device)
        broca3.import_state(legacy_state, device=device)
        self.assertEqual(broca3.verbatim_prompt_steps, 0)
        self.assertFalse(broca3.supports_verbatim_prompt)

        # Streaming (CUDA Graphs or eager), batched generation (homogeneous & heterogeneous), and confidence scoring
        streamed = "".join(
            broca.generate_stream(z_mem, max_new_bytes=16, prompt_text=prompt_txt)
        )
        self.assertIsInstance(streamed, str)
        self.assertGreater(len(streamed), 0)

        z_batch = z_mem.expand(2, -1, -1).contiguous()
        homog = broca.generate_batch(
            z_batch, temperatures=[0.5, 0.7], max_new_bytes=16, prompt_texts=[prompt_txt, prompt_txt]
        )
        self.assertEqual(len(homog), 2)
        hetero = broca.generate_batch(
            z_batch, temperatures=[0.5, 0.7], max_new_bytes=16, prompt_texts=[prompt_txt, "U: Krótko?\nA: "]
        )
        self.assertEqual(len(hetero), 2)

        c_single = broca.score_text_confidence("Proces wytwarzania glukozy.", z_mem, prompt_text=prompt_txt)
        c_batch = broca.score_texts_confidence_batch(
            ["Proces wytwarzania glukozy.", "Inna odpowiedź."], z_batch, prompt_texts=[prompt_txt, prompt_txt]
        )
        self.assertIn("avg_nll", c_single)
        self.assertEqual(len(c_batch), 2)

    def test_03_fact_sources_offline_harvester_and_builders(self) -> None:
        self.assertGreaterEqual(len(CURATED_FACT_ENCYCLOPEDIA), 100)
        pairs = build_fact_training_pairs(
            state_dir=None,
            include_offline=True,
            include_grounded_memory=True,
            include_multiturn=True,
            max_pairs=500,
        )
        self.assertTrue(any(q.startswith("[Pamięć]") for q, _ in pairs))
        self.assertTrue(any("Kontekst rozmowy:" in q for q, _ in pairs))

        def _mock_wiki_api(params: dict[str, str], timeout: int = 10) -> dict:
            if params.get("list") == "categorymembers":
                return {
                    "query": {
                        "categorymembers": [
                            {"title": "Lista polskich laureatów"},
                            {"title": "Grawitacja kwantowa (ujednoznacznienie)"},
                            {"title": "Fale grawitacyjne"},
                            {"title": "Krótki szkic"},
                        ]
                    }
                }
            return {
                "query": {
                    "pages": {
                        "1": {
                            "title": "Fale grawitacyjne",
                            "extract": (
                                "Fale grawitacyjne to rozchodzące się z prędkością światła zaburzenia czasoprzestrzeni "
                                "przewidziane przez ogólną teorię względności Alberta Einsteina w 1916 roku. "
                                "Powstają podczas przyspieszania masywnych obiektów, na przykład zderzeń czarnych dziur. "
                                "Po raz pierwszy bezpośrednio wykryto je w obserwatorium LIGO w 2015 roku."
                            ),
                            "fullurl": "https://pl.wikipedia.org/wiki/Fale_grawitacyjne",
                        },
                        "2": {
                            "title": "Grawitacja kwantowa (ujednoznacznienie)",
                            "extract": "Może odnosić się do kilku pojęć w fizyce teoretycznej.",
                        },
                        "3": {
                            "title": "Krótki szkic",
                            "extract": "Za krótki tekst.",
                        },
                    }
                }
            }

        with tempfile.TemporaryDirectory() as tmp:
            with patch("brain.fact_sources._wiki_api_json", side_effect=_mock_wiki_api):
                res1 = fetch_curated_polish_facts(tmp, target_count=5)
                res2 = fetch_curated_polish_facts(tmp, target_count=5)
            self.assertEqual(res1["added"], 1)
            self.assertEqual(res2["added"], 0)  # deduplicated against existing JSONL
            recs = load_fact_knowledge_records(tmp)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["title"], "Fale grawitacyjne")
            self.assertGreater(count_fact_knowledge_records(tmp, include_offline=True), len(CURATED_FACT_ENCYCLOPEDIA))
            engrams = get_fact_engram_texts(tmp, include_offline=True, max_items=300)
            self.assertTrue(any("Fale grawitacyjne" in title for _, title, _ in engrams))

    def test_04_agent_llm_pro_pipeline_verbatim_context_and_ui_routes(self) -> None:
        cfg = _small_cfg(use_moe=False)
        norm = normalize_training_pipeline_plan(
            {
                "preset": "Tryb LLM-Pro",
                "dialogue_train": {"epochs": 4, "use_verbatim_prompt": True, "include_fact_qa": True},
                "knowledge": {"include_curated_facts": True, "fetch_wiki_featured": 15},
            },
            cfg,
        )
        self.assertTrue(norm["dialogue_train"]["use_verbatim_prompt"])
        self.assertTrue(norm["dialogue_train"]["include_fact_qa"])
        self.assertTrue(norm["knowledge"]["include_curated_facts"])
        self.assertEqual(norm["knowledge"]["fetch_wiki_featured"], 15)

        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            try:
                brain.auto_sleep = False
                brain.auto_web = False
                with brain.lock:
                    brain.working_chat.append({"role": "user", "content": "Mam na imię Marek i lubię fizykę."})
                    brain.working_chat.append({"role": "assistant", "content": "Miło Cię poznać, Marku!"})
                vp = brain._build_verbatim_broca_prompt(
                    "Co studiuję?",
                    recalled_facts=["Fizyka bada własności materii i energii."],
                )
                self.assertIn("[Pamięć] Fizyka", vp)
                self.assertIn("U: Mam na imię Marek", vp)
                self.assertIn("U: Co studiuję?", vp)

                snap = brain.snapshot()
                self.assertIn("fact_knowledge_count", snap)
                self.assertIn("broca_verbatim_ready", snap["language_cortex"])
                self.assertIn("broca_verbatim_steps", snap["language_cortex"])
                json.dumps(snap, ensure_ascii=False)
            finally:
                brain.living_brain.stop_spontaneous_clock()

        html = (ROOT_DIR / "static" / "index.html").read_text(encoding="utf-8")
        srv = (ROOT_DIR / "server.py").read_text(encoding="utf-8")
        for token in ("llm_pro", "pp_dt_verbatim", "pp_dt_factqa", "pp_kn_curated", "pp_kn_wiki_feat", "startFactKnowledgeStudy"):
            self.assertIn(token, html)
        self.assertIn("/api/study/facts", srv)


if __name__ == "__main__":
    unittest.main()
