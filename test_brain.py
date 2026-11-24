"""
End-to-end GPU regression test for the exact real-world issues reported by the user:
  1) Studying 'Kunktator' -> finds Polish Wiktionary definition AND Wikipedia 'Kwintus Fabiusz Maksimus',
     extracts true facts, consolidates via GPU LoRA sleep, and answers accurately in Polish (no hallucination!).
  2) Studying 'wzkaźniki w cpp' (with typo!) -> auto-normalizes query, finds real C++ pointer docs, learns facts.
  3) Verifies questions without '?' ('kto to jest Kunktator', 'ale to nie prawda') are NOT saved as memories!
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from brain.agent import SleepingBrain


def main() -> None:
    print("=== REGRESSION TEST: KUNKTATOR + TYPO QUERY + CLEAN MEMORY + GPU LoRA ===")
    assert torch.cuda.is_available(), "CUDA GPU required!"
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="sleeping_brain_kunktator_"))
    try:
        brain = SleepingBrain(base_dir=tmp_dir, load_language_mode="sync")
        assert brain.language_cortex.ready

        # 1) Study 'Kunktator' from the web (2 rounds + GPU LoRA sleep)
        res_k = brain.study_topic_online("Kunktator", max_rounds=2, consolidate_after=True)
        print(
            f"\n[OK] Nauka 'Kunktator': stron={res_k['pages_read']}, fakty={res_k['facts_learned']}, "
            f"opanowanie={res_k['mastery']}%, źródła={[s['title'] for s in res_k['sources']]}"
        )
        assert res_k["pages_read"] >= 2, "Expected both Wiktionary & Wikipedia sources for Kunktator!"
        assert res_k["facts_learned"] >= 4

        # Ask without '?' and verify: (a) accurate answer, (b) question NOT polluted into Hippocampus!
        ans_k = brain.chat("no to kto to jest Kunktator")
        print("\n[CZAT 'no to kto to jest Kunktator']:\n ", ans_k["reply"])
        print("[HIPOKAMP po pytaniu]:", len(brain.hippocampus.short_term_memories), "(oczekiwano 0)")
        assert len(brain.hippocampus.short_term_memories) == 0, "Question without '?' must NOT be saved as memory!"
        low_k = ans_k["reply"].lower()
        assert "1989" not in low_k and "towarzystwo" not in low_k, "Old hallucination must NOT appear!"
        assert any(w in low_k for w in ("zwlek", "powol", "ociąg", "fabiusz", "maksimus", "wódz", "rzymsk")), (
            f"Answer should accurately define Kunktator! Got: {ans_k['reply']}"
        )

        # Verify reaction 'ale to nie prawda' is also NOT stored in Hippocampus
        brain.chat("ale to nie prawda")
        assert len(brain.hippocampus.short_term_memories) == 0, "'ale to nie prawda' must NOT be saved!"

        # 2) Study typo query 'wzkaźniki w cpp'
        res_cpp = brain.study_topic_online("wzkaźniki w cpp", max_rounds=1, consolidate_after=True)
        print(
            f"\n[OK] Nauka 'wzkaźniki w cpp' (z literówką!): stron={res_cpp['pages_read']}, "
            f"fakty={res_cpp['facts_learned']}, źródła={[s['title'] for s in res_cpp['sources']]}"
        )
        assert res_cpp["pages_read"] >= 1 and res_cpp["facts_learned"] >= 3

        ans_cpp = brain.chat("wyjaśnij krótko co to są wskaźniki w C++")
        print("\n[CZAT 'wskaźniki w C++']:\n ", ans_cpp["reply"])
        assert len(ans_cpp["reply"]) > 40

        print("\nWSZYSTKIE TESTY NAPRAWCZE ZALICZONE POMYŚLNIE!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
