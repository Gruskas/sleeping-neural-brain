from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import torch

import server
from brain.agent import SleepingBrain
from brain.config import BrainConfig
from brain.curriculum import CONCEPT_SYNTHESIS_COURSE, get_concept_synthesis_qa_pairs
from brain.hippocampus import EngramMeta


def _small_cfg() -> BrainConfig:
    return BrainConfig(
        num_neurons=512,
        sdr_k=24,
        column_dim=128,
        wernicke_layers=2,
        wernicke_heads=4,
        wernicke_ff_dim=256,
        broca_dim=128,
        broca_layers=2,
        broca_heads=4,
        broca_ff_dim=256,
        broca_use_moe=True,
        broca_ffn_mode="moe",
        broca_num_experts=4,
        broca_moe_top_k=2,
        broca_moe_expert_dim=128,
        broca_max_seq_len=160,
        broca_max_new_bytes=48,
        language_bootstrap_epochs=0,
        dmn_enabled=True,
    )


class TestDMNAndSynthesisCourse(unittest.TestCase):
    def test_concept_synthesis_course_integrity(self) -> None:
        self.assertGreaterEqual(len(CONCEPT_SYNTHESIS_COURSE), 55)
        qa_pairs = get_concept_synthesis_qa_pairs()
        self.assertEqual(len(qa_pairs), len(CONCEPT_SYNTHESIS_COURSE))
        for item in CONCEPT_SYNTHESIS_COURSE:
            self.assertTrue(str(item.get("module", "")).strip())
            self.assertTrue(str(item.get("topic", "")).strip())
            self.assertGreaterEqual(len(str(item.get("fact", ""))), 24)
        self.assertEqual(len({q.lower() for q, _ in qa_pairs}), len(qa_pairs))

    def test_overconfident_offtopic_safeguard_rejects_low_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=_small_cfg(), base_dir=tmp, load_language_mode="sync")
            try:
                brain.auto_sleep = False
                brain.auto_web = False
                brain.language_cortex._register_known_questions(
                    ["co to jest kot", "siema", "jak masz na imie"],
                    replace=True,
                )
                w_probe = brain.language_cortex.perceive_one("jestes w symulacji czy emulacji?")
                th_probe = brain.cortex.perceive_and_think(
                    w_probe.columns,
                    w_probe.sdr,
                    engram_bank=None,
                    sdr_to_columns_fn=brain.language_cortex._wernicke_raw.sdr_to_columns,
                )
                fake_delib = {
                    "w_out": w_probe,
                    "thought": th_probe,
                    "temperature": 0.41,
                    "draft": "Tak! W mojej fazie REM łączę odległe wspomnienia i podtrzymuje się masji.",
                    "score": 0.79,
                    "confidence": 0.87,
                    "avg_nll": 0.14,
                    "quality": 1.0,
                    "alignment": 0.21,
                    "critique_notes": ["Ścieżka Precyzyjna (T=0.41): ocena=79%"],
                }
                with patch.object(
                    brain.language_cortex,
                    "deliberate_and_critique",
                    return_value=fake_delib,
                ):
                    prep, _ = brain._prepare_tensor_thought("jestes w symulacji czy emulacji?")
                self.assertTrue(prep["is_ood"])
                self.assertTrue(any("Samokrytyka odrzuciła" in line for line in prep["thinking"]))
            finally:
                brain.living_brain.stop_spontaneous_clock()

    def test_run_concept_synthesis_course_with_parquet_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=_small_cfg(), base_dir=tmp, load_language_mode="sync")
            try:
                brain.auto_sleep = False
                brain.auto_web = False
                fake_replay = [
                    (f"Kontekst polski numer {i} o przyrodzie i technice.", f"Zdanie ciągłe opisujące zjawisko nr {i} w prosty sposób.")
                    for i in range(8)
                ]
                with patch.object(brain, "_sample_parquet_replay_pairs", return_value=fake_replay):
                    res = brain.run_concept_synthesis_course(consolidate=False, epochs=2)
                self.assertEqual(res["added_facts"], len(CONCEPT_SYNTHESIS_COURSE))
                self.assertEqual(res["parquet_replay"], 8)
                self.assertTrue(res["train_result"]["trained"])
                self.assertTrue(torch.isfinite(torch.tensor(res["train_result"]["loss_end"])).item())
            finally:
                brain.living_brain.stop_spontaneous_clock()

    def test_dmn_status_fallback_and_ui_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=_small_cfg(), base_dir=tmp, load_language_mode="sync")
            try:
                brain.auto_sleep = False
                for txt in (
                    "Woda paruje pod wpływem ciepła Słońca i wraca jako deszcz.",
                    "Kora mózgowa kojarzy odległe fakty przez mosty skojarzeniowe.",
                    "Termos wykorzystuje próżnię między ściankami do izolacji ciepła.",
                ):
                    w = brain.language_cortex.perceive_one(txt)
                    brain.hippocampus.semantic.add_batch(
                        w.sdr,
                        [EngramMeta(category="wiedza", preview_label=txt, sparsity_pct=4.0)],
                        strengths=[1.5],
                        importances=[5],
                        day=1,
                    )
                snap = brain.snapshot()
                self.assertIn("dmn_status", snap)
                self.assertIn("dmn_thoughts", snap)
                self.assertEqual(snap["dmn_status"]["semantic_count"], 3)
                self.assertTrue(snap["dmn_status"]["enabled"])

                # Verify relaxed fallback DMN bridge runs cleanly without exceptions
                with patch.object(
                    brain.language_cortex,
                    "synthesize_2hop_thought",
                    return_value={
                        "accepted": True,
                        "thought_text": "Ciepło Słońca i próżnia termosu opisują przepływ energii.",
                        "thought_sdr": brain.language_cortex.perceive_one("przepływ energii").sdr[0],
                        "bridge_columns": torch.zeros(16, dtype=torch.float32),
                        "joint_coherence": 0.42,
                    },
                ):
                    res = brain.run_dmn_2hop_step(non_blocking=False)
                self.assertIsNotNone(res)
                self.assertTrue(res["accepted"])
                self.assertEqual(len(brain.dmn_thoughts), 1)
                clean_res = {
                    k: v
                    for k, v in res.items()
                    if k not in ("thought_sdr", "bridge_columns") and not hasattr(v, "detach")
                }
                self.assertNotIn("thought_sdr", clean_res)
                self.assertNotIn("bridge_columns", clean_res)
                json.dumps(clean_res, ensure_ascii=False)
                json.dumps(res, ensure_ascii=False, default=server._json_default)
            finally:
                brain.living_brain.stop_spontaneous_clock()

        root = pathlib.Path(__file__).resolve().parent.parent
        html = (root / "static" / "index.html").read_text(encoding="utf-8")
        srv = (root / "server.py").read_text(encoding="utf-8")
        for token in ("dmnThoughtsBox", "triggerDmnThink", "startConceptCourse", "pp_kn_course"):
            self.assertIn(token, html)
        for route in ("/api/dmn/think", "/api/study/course"):
            self.assertIn(route, srv)


if __name__ == "__main__":
    unittest.main()
