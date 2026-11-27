from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import torch

from brain.agent import SleepingBrain
from brain.config import BrainConfig


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
        dmn_enabled=False,
    )


class TestChatContextAndGating(unittest.TestCase):
    def test_smalltalk_blocks_auto_web_hijack_and_preserves_factual_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=_small_cfg(), base_dir=tmp, load_language_mode="sync")
            try:
                brain.auto_sleep = False
                brain.auto_web = True
                self.assertTrue(brain._is_smalltalk_or_greeting("hej"))
                self.assertTrue(brain._is_smalltalk_or_greeting("chyba git a jak tobie?"))
                self.assertFalse(brain._is_followup_query("chyba git a jak tobie?"))

                brain.working_chat.append({"role": "user", "content": "hej"})
                brain.working_chat.append({"role": "assistant", "content": "Hej! Jak Ci mija dzień?"})
                self.assertEqual(brain._extract_recent_topic(), "")

                with patch("brain.agent.gather_web_sources") as mock_web:
                    brain.chat("chyba git a jak tobie?")
                    mock_web.assert_not_called()

                # Real factual topic + follow-up still resolves properly
                brain.working_chat.append({"role": "user", "content": "co to jest fotosynteza?"})
                brain.working_chat.append({"role": "assistant", "content": "Proces wytwarzania cukru przez rośliny."})
                self.assertEqual(brain._extract_recent_topic(), "fotosynteza")
                self.assertTrue(brain._is_followup_query("dlaczego?"))
                self.assertTrue(brain._is_factual_web_query("co to jest fotosynteza?"))
            finally:
                brain.living_brain.stop_spontaneous_clock()

    def test_hypothetical_and_complex_questions_reach_deliberation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=_small_cfg(), base_dir=tmp, load_language_mode="sync")
            try:
                brain.auto_sleep = False
                brain.auto_web = False
                brain.language_cortex._register_known_questions(
                    ["co to jest kot", "siema", "jak masz na imie"],
                    replace=True,
                )

                q_hyp = "co bys zjadl na sniadanie jakbys byl czlowiekiem"
                self.assertFalse(brain._is_factual_web_query(q_hyp))
                with patch.object(
                    brain.language_cortex,
                    "deliberate_and_critique",
                    wraps=brain.language_cortex.deliberate_and_critique,
                ) as spy_delib:
                    prep, _ = brain._prepare_tensor_thought(q_hyp)
                    self.assertGreaterEqual(spy_delib.call_count, 1)

                # Keyboard smash / non-alpha gibberish still hits upfront OOD gate
                prep_num, _ = brain._prepare_tensor_thought("12345!@#$% 67890")
                self.assertTrue(prep_num["is_ood"])
            finally:
                brain.living_brain.stop_spontaneous_clock()

    def test_conversation_context_builder_and_multiturn_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=_small_cfg(), base_dir=tmp, load_language_mode="sync")
            try:
                brain.auto_sleep = False
                brain.auto_web = False
                for idx in range(5):
                    brain.working_chat.append({"role": "user", "content": f"Pytanie testowe numer {idx} o kosmos i gwiazdy"})
                    brain.working_chat.append({"role": "assistant", "content": f"Odpowiedź testowa numer {idx} o planetach"})

                cur = "a dlaczego to powiedziałeś?"
                ctx = brain._build_conversation_context_text(cur, max_turns=3, max_chars=260)
                self.assertTrue(ctx.startswith("Kontekst rozmowy:"))
                self.assertTrue(ctx.endswith(f"|| Odpowiedz na: {cur}"))
                self.assertLessEqual(len(ctx), 260)

                mixed_pairs = [
                    ("siema", "Hej! Co tam u Ciebie?"),
                    (
                        "Kontekst rozmowy: Użytkownik: chyba git a jak tobie? | Ja: U mnie super! || Odpowiedz na: pamiętasz o czym rozmawialiśmy?",
                        "Tak, mówiłeś że u Ciebie jest git i pytałeś jak u mnie.",
                    ),
                    ("co byś zjadł na śniadanie jakbyś był człowiekiem?", "Chętnie spróbowałbym ciepłej jajecznicy i dobrej kawy."),
                ]
                res = brain.language_cortex.train_dialogue_finetune(
                    mixed_pairs,
                    cortex=brain.cortex,
                    epochs=4,
                    batch_size=2,
                    train_working_memory=True,
                )
                self.assertTrue(res["trained"])
                self.assertTrue(torch.isfinite(torch.tensor(res["loss_end"])).item())

                wm_slots = torch.randn(1, 2, brain.cfg.column_dim, device=brain.device)
                delib = brain.language_cortex.deliberate_and_critique(
                    user_message="pamiętasz o czym mówiliśmy?",
                    cortex=brain.cortex,
                    engram_bank=None,
                    working_memory_slots=wm_slots,
                    context_user_message=ctx,
                    max_new_bytes=32,
                    num_candidates=2,
                )
                self.assertIn("draft", delib)
                self.assertTrue(any("multi-turn" in n for n in delib["critique_notes"]))
            finally:
                brain.living_brain.stop_spontaneous_clock()


if __name__ == "__main__":
    unittest.main()
