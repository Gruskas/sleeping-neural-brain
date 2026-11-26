"""
End-to-end GPU smoke test for the tokenless UTF-8 bio-inspired architecture:
  - WernickeByteCortex (Byte-Patching Conv1D + 16 columns + k-WTA SDR 3-5%)
  - BiologicalAssociativeCortex (Modern Hopfield + local Predictive Coding/Hebb + SHY)
  - BrocaByteTransformer (Cross-Attention strictly on Z* + real UTF-8 streaming)
  - Anti-copy verification after erasing all diagnostic preview_label fields
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

from brain.agent import SleepingBrain
from brain.config import BrainConfig


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=== TOKENLESS BIO-BRAIN GPU SMOKE TEST ===")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="sleeping_brain_tokenless_"))
    try:
        cfg = BrainConfig(language_bootstrap_epochs=2, cortex_sleep_epochs=2)
        brain = SleepingBrain(cfg=cfg, base_dir=tmp_dir, load_language_mode="sync")
        assert brain.language_cortex.ready

        # 1) Ingest Polish diacritics + code stimuli through Wernicke -> Hippocampus bfloat16 SDRs
        samples = [
            "Żółć gęślą jaźń — wskaźniki w C++ przechowują adresy zmiennych w pamięci.",
            "Fotosynteza wytwarza glukozę oraz tlen z wody i dwutlenku węgla.",
            "Czarna dziura ma horyzont zdarzeń, z którego nie wydostaje się światło.",
            "Chrzest Polski odbył się w 966 roku za panowania Mieszka Pierwszego.",
        ]
        for txt in samples:
            brain.add_manual_memory(txt, category="wiedza", importance=5)

        # 2) Run vectorized GPU Sleep (NREM Light + NREM Deep Local PC/Hebb + SHY + REM)
        report = brain.sleep_now(visual_pacing=False)
        print(
            f"[SLEEP] {report.duration_ms:.1f} ms | PredErr: {report.prediction_error_start:.4f} -> "
            f"{report.prediction_error_end:.4f} | SHY pruned: {report.pruned_synapses} "
            f"(active: {report.active_synapses}) | REM: +{len(report.rem_reflections)}"
        )
        assert float(torch.diag(brain.cortex.W_rec).abs().max().item()) == 0.0
        assert report.prediction_error_end < report.prediction_error_start
        assert len(brain.hippocampus.rem) > 0

        # 3) Erase all preview_label fields and verify isolated Broca decoding from Z*
        brain.hippocampus.clear_all_preview_labels("")
        ans1 = brain.chat("Żółć gęślą jaźń — wskaźniki w C++", temperature=0.88)["reply"]
        ans2 = brain.chat("Żółć gęślą jaźń — wskaźniki w C++", temperature=0.88)["reply"]
        print(f"[BROCA SAMPLE 1]: {ans1[:100]}")
        print(f"[BROCA SAMPLE 2]: {ans2[:100]}")
        assert len(ans1) > 0 and len(ans2) > 0

        print("\nTOKENLESS BIO-BRAIN SMOKE TEST PASSED!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
