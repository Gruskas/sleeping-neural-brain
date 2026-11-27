from __future__ import annotations

import sys
import tempfile
import time
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


def _make_sparse_sdr(num_neurons: int, active_indices: list[int], device: torch.device) -> torch.Tensor:
    v = torch.zeros(num_neurons, device=device, dtype=torch.float32)
    v[active_indices] = 1.0
    return F.normalize(v, dim=-1, eps=1e-6)


class TestDMN2HopConceptGraph(unittest.TestCase):
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
            dmn_enabled=False,
            rem_2hop_steps=4,
        )
        base.update(overrides)
        return BrainConfig(**base)

    def _build_bridge_sdrs(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct exact sparse SDRs with:
          - sim(A, B) = 16/32 = 0.50 >= 0.45
          - sim(B, C_diff) = 16/32 = 0.50 >= 0.45
          - sim(A, C_diff) = 0.0 <= 0.25
          - sim(B, C_same) = 16/32 = 0.50 >= 0.45, sim(A, C_same) = 0.0 (same day as A)
        """
        a_idx = list(range(0, 32))
        b_idx = list(range(16, 48))
        c_diff_idx = list(range(32, 64))
        c_same_idx = list(range(32, 48)) + list(range(80, 96))
        return (
            _make_sparse_sdr(n, a_idx, device),
            _make_sparse_sdr(n, b_idx, device),
            _make_sparse_sdr(n, c_diff_idx, device),
            _make_sparse_sdr(n, c_same_idx, device),
        )

    def test_01_sample_2hop_bridge_thresholds_cross_day_and_lru(self) -> None:
        """GPU 2-hop discovery enforces sim(A,B)>=0.45, sim(B,C)>=0.45, sim(A,C)<=0.25, prefers cross-day, deduplicates via LRU."""
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hippo = Hippocampus(num_neurons=512, sdr_k=32, device=dev)
        sdr_a, sdr_b, sdr_c_diff, sdr_c_same = self._build_bridge_sdrs(512, dev)

        hippo.semantic.add_batch(
            sdrs=torch.stack([sdr_a, sdr_b, sdr_c_same], dim=0),
            metas=[
                EngramMeta(id="nodeA", preview_label="Fotosynteza zachodzi w liściach roślin.", topic="biologia"),
                EngramMeta(id="nodeB", preview_label="Energia słoneczna napędza przemiany chemiczne.", topic="fizyka"),
                EngramMeta(id="nodeCs", preview_label="Panele fotowoltaiczne zamieniają światło w prąd.", topic="biologia"),
            ],
            strengths=[2.0, 2.0, 2.0],
            importances=[5.0, 1.0, 1.0],
            day=5,
        )
        hippo.semantic.add_batch(
            sdrs=sdr_c_diff.unsqueeze(0),
            metas=[
                EngramMeta(id="nodeCd", preview_label="Ogniwa słoneczne zasilają satelity na orbicie.", topic="kosmos"),
            ],
            strengths=[2.0],
            importances=[1.0],
            day=1,
        )

        # Sample bridges until exhausted; verify constraints & no duplicate (A, C) pairs
        seen_pairs: set[tuple[str, str]] = set()
        found_cross_day = False
        for _ in range(10):
            br = hippo.sample_2hop_bridge(min_hop_sim=0.45, max_direct_sim=0.25, current_day=5)
            if br is None:
                break
            id_a, id_b, id_c = br["a"]["id"], br["b"]["id"], br["c"]["id"]
            self.assertEqual(len({id_a, id_b, id_c}), 3)
            self.assertGreaterEqual(br["sim_ab"], 0.45)
            self.assertGreaterEqual(br["sim_bc"], 0.45)
            self.assertLessEqual(br["sim_ac"], 0.25)
            canon = (id_a, id_c) if id_a <= id_c else (id_c, id_a)
            self.assertNotIn(canon, seen_pairs, f"Duplicate pair returned: {canon}")
            seen_pairs.add(canon)
            if br["cross_day"]:
                found_cross_day = True

        self.assertGreaterEqual(len(seen_pairs), 2)
        self.assertTrue(found_cross_day, "Expected at least one cross-day bridge")
        self.assertIsNone(hippo.sample_2hop_bridge(min_hop_sim=0.45, max_direct_sim=0.25, current_day=5))
        self.assertEqual(len(hippo.semantic.explored_pairs), len(seen_pairs))

    def test_02_synthesize_2hop_quality_gate_rejects_bad_and_accepts_good(self) -> None:
        """Multi-slot Z_ctx synthesis rejects loops/copies/low-quality and saves+strengthens+logs accepted insights."""
        cfg = self._small_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            try:
                sdr_a, sdr_b, sdr_c, _ = self._build_bridge_sdrs(cfg.num_neurons, brain.device)
                node_a = {"id": "enA", "sdr": sdr_a, "preview_label": "Fotosynteza zamienia światło w energię roślin."}
                node_b = {"id": "enB", "sdr": sdr_b, "preview_label": "Światło słoneczne jest źródłem energii."}
                node_c = {"id": "enC", "sdr": sdr_c, "preview_label": "Panele słoneczne produkują prąd elektryczny."}

                # A) Reject verbatim copy & word repetition loop even if audit were permissive
                good_sdr = F.normalize(0.45 * sdr_a + 0.15 * sdr_b + 0.40 * sdr_c, dim=-1, eps=1e-6)
                with patch.object(
                    brain.language_cortex._broca_raw,
                    "generate_batch",
                    return_value=[
                        {"text": "Światło światło światło światło energia.", "num_bytes": 40},
                        {"text": "Fotosynteza zamienia światło w energię roślin.", "num_bytes": 46},
                    ],
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
                    rej = brain.synthesize_2hop_thought(node_a, node_b, node_c, tag="dmn_2hop")
                    self.assertFalse(rej["accepted"])
                    self.assertEqual(len(brain.hippocampus.short_term), 0)

                # B) Accept novel coherent Polish insight -> saved to Hippocampus + Oja-Hebb A<->C + logged
                novel_text = "Zarówno rośliny, jak i panele przetwarzają energię światła słonecznego."
                with patch.object(
                    brain.language_cortex._broca_raw,
                    "generate_batch",
                    return_value=[{"text": novel_text, "num_bytes": len(novel_text.encode("utf-8"))}],
                ), patch.object(
                    brain.language_cortex,
                    "analyze_self_heard_draft",
                    return_value={
                        "total": 0.89,
                        "quality": 0.91,
                        "alignment": 0.42,
                        "confidence": 0.87,
                        "avg_nll": 0.50,
                        "issues": [],
                        "needs_correction": False,
                        "heard_sdr": good_sdr,
                    },
                ):
                    acc = brain.synthesize_2hop_thought(node_a, node_b, node_c, tag="dmn_2hop", importance=3.0)

                self.assertTrue(acc["accepted"], f"Expected acceptance, got: {acc.get('rejection_reasons')}")
                self.assertEqual(acc["z_ctx_shape"], (1, cfg.num_columns * 4, cfg.column_dim))
                self.assertEqual(len(brain.hippocampus.short_term), 1)
                saved_meta = brain.hippocampus.short_term.meta[0]
                self.assertFalse(saved_meta.pinned)
                self.assertEqual(saved_meta.tag, "dmn_2hop")
                self.assertEqual(saved_meta.parent_ids, ["enA", "enB", "enC"])
                self.assertAlmostEqual(float(brain.hippocampus.short_term.importances[0].item()), 3.0, places=2)

                # Oja-Hebb strengthened A <-> C in LivingBrainEngine
                self.assertGreater(acc["oja_stats"]["delta_norm"], 0.0)
                self.assertTrue(torch.isfinite(brain.living_brain.W_assoc_rec).all())
                self.assertEqual(float(brain.living_brain.W_assoc_rec.diagonal().abs().max().item()), 0.0)

                # Logged in event_log / brain_log and dmn_thoughts
                self.assertTrue(any(e["kind"] == "DMN" and "enA" in e["msg"] and "enC" in e["msg"] and "enB" in e["msg"] for e in brain.brain_log))
                self.assertEqual(len(brain.dmn_thoughts), 1)
            finally:
                brain.living_brain.stop_spontaneous_clock()

    def test_03_dmn_idle_clock_preemption_and_rem_sleep_2hop(self) -> None:
        """DMN fires autonomously on idle clock, yields instantly to user chat, and REM runs 2-hop steps during /sen."""
        cfg = self._small_cfg(
            dmn_enabled=True,
            idle_think_delay_sec=0.15,
            dmn_think_interval_sec=0.15,
            rem_2hop_steps=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            try:
                sdr_a, sdr_b, sdr_c, sdr_c2 = self._build_bridge_sdrs(cfg.num_neurons, brain.device)
                brain.hippocampus.semantic.add_batch(
                    sdrs=torch.stack([sdr_a, sdr_b, sdr_c2], dim=0),
                    metas=[
                        EngramMeta(id="idA", preview_label="Woda paruje pod wpływem ciepła.", topic="fizyka"),
                        EngramMeta(id="idB", preview_label="Para wodna unosi się i tworzy chmury.", topic="pogoda"),
                        EngramMeta(id="idC2", preview_label="Deszcz nawadnia glebę i rzeki.", topic="przyroda"),
                    ],
                    strengths=[2.0, 2.0, 2.0],
                    importances=[4.0, 3.0, 3.0],
                    day=2,
                )
                brain.hippocampus.semantic.add_batch(
                    sdrs=sdr_c.unsqueeze(0),
                    metas=[EngramMeta(id="idC1", preview_label="Rzeki płyną z gór aż do morza.", topic="geografia")],
                    strengths=[2.0],
                    importances=[3.0],
                    day=1,
                )

                good_sdr = F.normalize(0.45 * sdr_a + 0.15 * sdr_b + 0.40 * sdr_c, dim=-1, eps=1e-6)
                rem_insight = "Ciepło słoneczne napędza obieg wody od parowania po rzeki."

                with patch.object(
                    brain.language_cortex._broca_raw,
                    "generate_batch",
                    return_value=[{"text": rem_insight, "num_bytes": len(rem_insight.encode("utf-8"))}],
                ), patch.object(
                    brain.language_cortex,
                    "analyze_self_heard_draft",
                    return_value={
                        "total": 0.88,
                        "quality": 0.90,
                        "alignment": 0.40,
                        "confidence": 0.86,
                        "avg_nll": 0.52,
                        "issues": [],
                        "needs_correction": False,
                        "heard_sdr": good_sdr,
                    },
                ):
                    # 1) Wait briefly for autonomous background DMN clock to fire
                    brain._last_user_activity_ts = time.monotonic() - 10.0
                    deadline = time.monotonic() + 4.0
                    while time.monotonic() < deadline and len(brain.dmn_thoughts) == 0:
                        time.sleep(0.08)
                    self.assertGreaterEqual(len(brain.dmn_thoughts), 1, "DMN background clock should produce at least 1 idle 2-hop thought")
                    self.assertEqual(brain.dmn_thoughts[0]["tag"], "dmn_2hop")

                    # 2) Preemption check: while user_interrupt_event is set, DMN immediately yields
                    brain._user_interrupt_event.set()
                    self.assertIsNone(brain.run_dmn_2hop_step(non_blocking=True))
                    brain._user_interrupt_event.clear()

                    # 3) Reset explored_pairs and run full /sen (sleep_now) -> REM 2-hop synthesis
                    brain.hippocampus.semantic.explored_pairs.clear()
                    report = brain.sleep_now(visual_pacing=False)
                    rem_2hop_metas = [m for m in brain.hippocampus.rem.meta if m.tag == "rem_2hop"]
                    self.assertGreaterEqual(len(rem_2hop_metas), 1, "REM phase should record rem_2hop thoughts")
                    self.assertEqual(len(rem_2hop_metas[0].parent_ids), 3)
                    self.assertFalse(rem_2hop_metas[0].pinned)
                    self.assertTrue(any(lk.get("source") == "REM_2HOP" for lk in report.rem_new_links))
                    self.assertEqual(len(report.rem_reflections), len(brain.hippocampus.rem))

                    # 4) Checkpoint round-trip preserves parent_ids, tag, and dmn_thoughts
                    brain.save_state()
                    reloaded = SleepingBrain(cfg=self._small_cfg(dmn_enabled=False), base_dir=tmp, load_language_mode="sync")
                    try:
                        rel_rem = [m for m in reloaded.hippocampus.rem.meta if m.tag == "rem_2hop"]
                        self.assertEqual(len(rel_rem), len(rem_2hop_metas))
                        self.assertEqual(rel_rem[0].parent_ids, rem_2hop_metas[0].parent_ids)
                        self.assertGreaterEqual(len(reloaded.dmn_thoughts), 1)
                    finally:
                        reloaded.living_brain.stop_spontaneous_clock()
            finally:
                brain.living_brain.stop_spontaneous_clock()


if __name__ == "__main__":
    unittest.main(verbosity=2)
