from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from brain.agent import SleepingBrain
from brain.config import BrainConfig
from brain.hippocampus import EngramMeta, Hippocampus
from brain.language_cortex import parse_thought_and_answer


def _make_sparse_sdr(num_neurons: int, active_indices: list[int], device: torch.device) -> torch.Tensor:
    v = torch.zeros(num_neurons, device=device, dtype=torch.float32)
    v[active_indices] = 1.0
    return F.normalize(v, dim=-1, eps=1e-6)


class TestAlwaysOnDMN(unittest.TestCase):
    @staticmethod
    def _small_cfg(**overrides) -> BrainConfig:
        base = dict(
            language_bootstrap_epochs=0,
            num_neurons=512,
            sdr_k=32,
            num_columns=16,
            column_dim=64,
            wernicke_layers=2,
            wernicke_heads=4,
            wernicke_ff_dim=128,
            broca_dim=64,
            broca_layers=2,
            broca_heads=4,
            broca_ff_dim=128,
            broca_use_moe=True,
            broca_num_experts=8,
            broca_moe_top_k=2,
            broca_moe_expert_dim=64,
            broca_max_seq_len=96,
            broca_max_new_bytes=48,
            dmn_enabled=True,
            dmn_always_think=True,
            idle_think_delay_sec=15.0,
            dmn_think_interval_sec=12.0,
        )
        base.update(overrides)
        return BrainConfig(**base)

    def test_01_always_on_fallback_across_banks_and_lru_eviction(self) -> None:
        """When semantic < 3 or strict thresholds fail, fallback links across short_term+semantic and evicts 25% LRU on exhaustion."""
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hippo = Hippocampus(num_neurons=512, sdr_k=32, device=dev)

        # Orthogonal / distant SDRs that strict thresholds (>=0.45) would never match
        s1 = _make_sparse_sdr(512, list(range(0, 32)), dev)
        s2 = _make_sparse_sdr(512, list(range(32, 64)), dev)
        s3 = _make_sparse_sdr(512, list(range(64, 96)), dev)
        s4 = _make_sparse_sdr(512, list(range(96, 128)), dev)

        hippo.semantic.add_batch(
            sdrs=s1.unsqueeze(0),
            metas=[EngramMeta(id="sem1", preview_label="Kot jest ssakiem drapieżnym.")],
            strengths=[2.0],
            importances=[4.0],
            day=1,
        )
        hippo.short_term.add_batch(
            sdrs=torch.stack([s2, s3, s4], dim=0),
            metas=[
                EngramMeta(id="st1", preview_label="Wisła płynie przez Polskę."),
                EngramMeta(id="st2", preview_label="Tlen podtrzymuje spalanie."),
                EngramMeta(id="st3", preview_label="Ziemia krąży wokół Słońca."),
            ],
            strengths=[1.5, 1.5, 1.5],
            importances=[3.0, 2.5, 2.0],
            day=2,
        )

        # Strict semantic < 3 fails, but fallback across semantic + short_term succeeds!
        br = hippo.sample_2hop_bridge(
            min_hop_sim=0.45,
            max_direct_sim=0.25,
            include_short_term=True,
            allow_fallback=True,
        )
        self.assertIsNotNone(br)
        self.assertEqual(len({br["a"]["id"], br["b"]["id"], br["c"]["id"]}), 3)

        # Exhaust all 6 unordered pairs in explored_pairs -> 25% LRU eviction keeps DMN alive
        all_ids = ["sem1", "st1", "st2", "st3"]
        for i in range(len(all_ids)):
            for j in range(i + 1, len(all_ids)):
                hippo.semantic.mark_pair_explored(all_ids[i], all_ids[j])
        self.assertEqual(len(hippo.semantic.explored_pairs), 6)

        br2 = hippo.sample_2hop_bridge(
            min_hop_sim=0.45,
            max_direct_sim=0.25,
            include_short_term=True,
            allow_fallback=True,
        )
        self.assertIsNotNone(br2, "Fallback should evict oldest 25% of explored_pairs and keep discovering bridges")

    def test_02_thought_scratchpad_w_rec_consolidation_and_interrupt(self) -> None:
        """[MYŚL]...[ODP] is parsed, saved to Hippocampus + dmn_thoughts, updates W_rec (@no_grad, zero diag), and yields to interrupt."""
        s, a = parse_thought_and_answer("[MYŚL] łączę fotosyntezę z energią [ODP] Rośliny wykorzystują światło podobnie jak ogniwa.")
        self.assertEqual(s, "łączę fotosyntezę z energią")
        self.assertEqual(a, "Rośliny wykorzystują światło podobnie jak ogniwa.")

        cfg = self._small_cfg(dmn_enabled=True, dmn_always_think=True)
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            try:
                dev = brain.device
                sdr_a = _make_sparse_sdr(cfg.num_neurons, list(range(0, 32)), dev)
                sdr_b = _make_sparse_sdr(cfg.num_neurons, list(range(16, 48)), dev)
                sdr_c = _make_sparse_sdr(cfg.num_neurons, list(range(32, 64)), dev)

                brain.hippocampus.short_term.add_batch(
                    sdrs=torch.stack([sdr_a, sdr_b, sdr_c], dim=0),
                    metas=[
                        EngramMeta(id="fA", preview_label="Fotosynteza zachodzi w liściach roślin."),
                        EngramMeta(id="fB", preview_label="Światło słoneczne niesie energię."),
                        EngramMeta(id="fC", preview_label="Ogniwa słoneczne wytwarzają prąd."),
                    ],
                    strengths=[2.0, 2.0, 2.0],
                    importances=[4.0, 3.0, 3.5],
                    day=1,
                )

                good_sdr = F.normalize(0.42 * sdr_a + 0.16 * sdr_b + 0.42 * sdr_c, dim=-1, eps=1e-6)
                draft_with_tags = "[MYŚL] światło łączy liście i ogniwa [ODP] Zarówno rośliny, jak i ogniwa przetwarzają energię światła."
                updates_before = brain.cortex.total_local_updates

                with patch.object(
                    brain.language_cortex._broca_raw,
                    "generate_batch",
                    return_value=[{"text": draft_with_tags, "num_bytes": len(draft_with_tags.encode("utf-8"))}],
                ), patch.object(
                    brain.language_cortex,
                    "analyze_self_heard_draft",
                    return_value={
                        "total": 0.90,
                        "quality": 0.92,
                        "alignment": 0.45,
                        "confidence": 0.88,
                        "avg_nll": 0.45,
                        "issues": [],
                        "needs_correction": False,
                        "heard_sdr": good_sdr,
                    },
                ):
                    res = brain.run_dmn_2hop_step(non_blocking=True)

                self.assertIsNotNone(res)
                self.assertTrue(res["accepted"])
                self.assertEqual(res["scratchpad"], "światło łączy liście i ogniwa")
                self.assertEqual(res["thought_text"], "Zarówno rośliny, jak i ogniwa przetwarzają energię światła.")
                self.assertGreater(brain.cortex.total_local_updates, updates_before)
                self.assertEqual(float(brain.cortex.W_rec.diagonal().abs().max().item()), 0.0)
                self.assertTrue(brain.cortex.W_rec.is_all_finite())
                self.assertEqual(brain.dmn_thoughts[-1]["scratchpad"], "światło łączy liście i ogniwa")

                # Snapshot status keys
                snap = brain.snapshot()
                self.assertIn("dmn_active", snap)
                self.assertIn("dmn_total_thoughts", snap)
                self.assertIn("dmn_always_think", snap)
                self.assertGreaterEqual(snap["dmn_total_thoughts"], 1)

                # Immediate preemption when user sends a message
                brain._user_interrupt_event.set()
                self.assertIsNone(brain.run_dmn_2hop_step(non_blocking=True))
                brain._user_interrupt_event.clear()
            finally:
                brain.living_brain.stop_spontaneous_clock()


if __name__ == "__main__":
    unittest.main(verbosity=2)
