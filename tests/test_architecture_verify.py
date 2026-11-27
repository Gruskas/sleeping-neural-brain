from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from brain.agent import SleepingBrain
from brain.config import BrainConfig
from brain.living_brain import LivingStepTelemetry, NeuromodulatorState
from brain.sensory_wernicke import BOS_BYTE, WernickeByteCortex, encode_utf8_bytes


class TestRebuiltArchitecture9Points(unittest.TestCase):
    @staticmethod
    def _small_cfg(**overrides) -> BrainConfig:
        base = dict(
            language_bootstrap_epochs=0,
            num_neurons=1024,
            sdr_k=40,
            num_columns=16,
            column_dim=64,
            wernicke_layers=2,
            wernicke_heads=4,
            wernicke_ff_dim=128,
            wernicke_max_patch_slots=64,
            wernicke_rf_anchor_weight=0.12,
            broca_dim=64,
            broca_layers=2,
            broca_heads=4,
            broca_ff_dim=128,
            broca_max_seq_len=512,
            broca_max_new_bytes=32,
            working_memory_slots=8,
            system2_reasoning_steps=3,
        )
        base.update(overrides)
        return BrainConfig(**base)

    def test_01_checkpoint_migration_and_startup(self) -> None:
        """1. Migracja checkpointu i start systemu bez RuntimeError: size mismatch."""
        cfg = self._small_cfg()

        # A) Niezgodne kształty tensorów i brak nowych kluczy Z_ctx / working_memory_slots
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            ckpt_path = brain.save_state()

            saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            saved["language_cortex"]["wernicke"]["weights"]["patch_pos"] = torch.randn(1, 12, cfg.column_dim)
            saved["language_cortex"]["broca"]["weights"]["z_summary_proj.weight"] = torch.randn(cfg.column_dim, 64)
            saved["cortex"].pop("segment_embeddings", None)
            saved.pop("working_memory_slots", None)
            torch.save(saved, ckpt_path)

            migrated = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            self.assertEqual(tuple(migrated.cortex.segment_embeddings.shape), (4, cfg.column_dim))
            self.assertEqual(migrated._wm_slot_count, 0)

            w_out = migrated.language_cortex.perceive_one("Test migracji Z_ctx.")
            th = migrated.cortex.perceive_and_think(
                w_out.columns,
                w_out.sdr,
                sdr_to_columns_fn=migrated.language_cortex.wernicke_module.sdr_to_columns,
                sensory_patch_columns=w_out.patch_columns,
                sensory_patch_mask=w_out.patch_mask,
            )
            self.assertIsNotNone(th.context_columns)
            assert th.context_columns is not None
            self.assertTrue(torch.isfinite(th.context_columns).all())

        # B) Stary tokenowy checkpoint -> automatyczny backup *.legacy_*.bak i czysty start
        with tempfile.TemporaryDirectory() as tmp_leg:
            state_dir = Path(tmp_leg) / "brain_state"
            state_dir.mkdir(parents=True, exist_ok=True)
            pt_path = state_dir / "brain_weights.pt"
            js_path = state_dir / "brain_memory.json"
            torch.save({"concept_cortex": {"id_to_concept": ["token"]}, "language_cortex": {"tokenizer": {}}}, pt_path)
            js_path.write_text(json.dumps({"embedding_version": "standalone_pytorch_v1", "day": 5}), encoding="utf-8")

            fresh = SleepingBrain(cfg=cfg, base_dir=tmp_leg, load_language_mode="sync")
            self.assertTrue(fresh.language_cortex.ready)
            self.assertEqual(len(list(state_dir.glob("brain_weights.pt.legacy_*.bak"))), 1)
            self.assertEqual(len(list(state_dir.glob("brain_memory.json.legacy_*.bak"))), 1)

    def test_02_forward_no_nan_inf_extreme_inputs(self) -> None:
        """2. Przebieg forward bez NaN/Inf na skrajnych wejściach (1 znak, krótkie, ~1500 znaków PL + emoji)."""
        cfg = self._small_cfg(max_input_bytes=4096)
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            long_text = (
                "Zażółć gęślą jaźń 🧠🚀✨🔥🌍 — test długiej sekwencji składniowej z polskimi znakami "
                "i wielobajtowymi znakami UTF-8 w korze Wernickego oraz dekoderze Broki. "
            ) * 10
            self.assertGreaterEqual(len(long_text), 1400)

            for sample in ("a", "siema", long_text):
                in_b, in_m = encode_utf8_bytes(
                    [sample],
                    max_bytes=cfg.max_input_bytes,
                    patch_size=cfg.patch_stride,
                    device=brain.device,
                )
                with torch.no_grad():
                    w_out = brain.language_cortex.wernicke_module(in_b, in_m)
                    th = brain.cortex.perceive_and_think(
                        w_out.columns,
                        w_out.sdr,
                        sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns,
                        sensory_patch_columns=w_out.patch_columns,
                        sensory_patch_mask=w_out.patch_mask,
                    )
                    assert th.context_columns is not None
                    bos = torch.tensor([[BOS_BYTE]], dtype=torch.long, device=brain.device)
                    logits = brain.language_cortex.broca_module(bos, th.context_columns, memory_mask=th.context_mask)
                    gen = brain.language_cortex.broca_module.generate(
                        th.context_columns,
                        max_new_bytes=24,
                        memory_mask=th.context_mask,
                    )

                for name, tensor in (
                    ("columns", w_out.columns),
                    ("sdr", w_out.sdr),
                    ("patch_columns", w_out.patch_columns),
                    ("z_star", th.z_star),
                    ("columns_star", th.columns_star),
                    ("context_columns", th.context_columns),
                    ("logits", logits),
                ):
                    self.assertIsNotNone(tensor, f"{name} is None for sample={sample[:20]!r}")
                    self.assertTrue(torch.isfinite(tensor).all(), f"Non-finite values in {name} for sample={sample[:20]!r}")
                self.assertNotIn("\ufffd", gen["text"])

    def test_03_variable_binding_reversed_roles(self) -> None:
        """3. Wiązanie relacji w zdaniu: 'Pies ugryzł kota.' vs 'Kot ugryzł psa.'."""
        cfg = self._small_cfg()
        wernicke = WernickeByteCortex(cfg)
        wernicke.eval()
        w_out = wernicke.encode_texts(["Pies ugryzł kota.", "Kot ugryzł psa."], device=torch.device("cpu"))
        assert w_out.patch_columns is not None and w_out.patch_mask is not None

        p_a = w_out.patch_columns[0]
        p_b = w_out.patch_columns[1]
        valid_a = int(w_out.patch_mask[0].sum().item())
        valid_b = int(w_out.patch_mask[1].sum().item())
        obj_idx = min(valid_a, valid_b) - 1
        self.assertGreaterEqual(obj_idx, 2)

        subj_cos = float(F.cosine_similarity(p_a[0], p_b[0], dim=0).item())
        obj_cos = float(F.cosine_similarity(p_a[obj_idx], p_b[obj_idx], dim=0).item())
        seq_cos = float(
            F.cosine_similarity(
                p_a[: obj_idx + 1].reshape(-1),
                p_b[: obj_idx + 1].reshape(-1),
                dim=0,
            ).item()
        )
        self.assertLess(subj_cos, 0.95, f"Subject patch collapsed: cos={subj_cos:.4f}")
        self.assertLess(obj_cos, 0.95, f"Object patch collapsed: cos={obj_cos:.4f}")
        self.assertLess(seq_cos, 0.95, f"Full patch sequence collapsed: cos={seq_cos:.4f}")

    def test_04_no_hard_hash_table_overwrite(self) -> None:
        """4. Brak twardego podmieniania wejścia na wektor z _known_q_sdrs."""
        cfg = self._small_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.language_cortex._register_known_questions(["co to jest kot"], replace=True)

            w_known = brain.language_cortex.perceive_one("co to jest kot")
            w_diff, _, canon_chat = brain.language_cortex.perceive_for_chat("czym kot różni się od kamienia?")
            sim_diff, canon_match = brain.language_cortex.match_known_attractor(w_diff.sdr[0])

            self.assertIsInstance(sim_diff, float)
            self.assertIsNone(canon_chat)
            self.assertIsNone(canon_match)

            th_known = brain.cortex.perceive_and_think(
                w_known.columns,
                w_known.sdr,
                sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns,
                sensory_patch_columns=w_known.patch_columns,
                sensory_patch_mask=w_known.patch_mask,
            )
            th_diff = brain.cortex.perceive_and_think(
                w_diff.columns,
                w_diff.sdr,
                sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns,
                sensory_patch_columns=w_diff.patch_columns,
                sensory_patch_mask=w_diff.patch_mask,
            )
            assert th_known.context_columns is not None and th_diff.context_columns is not None
            z_cos = float(F.cosine_similarity(th_known.columns_star.flatten(1), th_diff.columns_star.flatten(1)).item())
            self.assertLess(z_cos, 0.95, f"Different question collapsed onto known attractor: cos={z_cos:.4f}")

    def test_05_multislot_prefrontal_working_memory(self) -> None:
        """5. Wieloslotowa pamięć robocza GPU [K_wm, D] przechowuje osobne tury."""
        cfg = self._small_cfg(working_memory_slots=8)
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.auto_sleep = False
            brain.auto_web = False

            self.assertEqual(tuple(brain._wm_slot_buf.shape), (8, cfg.column_dim))
            brain.chat("Mam na imię Kuba.")
            c1 = brain._wm_slot_count
            self.assertGreaterEqual(c1, 2)
            slot0_after_t1 = brain._wm_slot_buf[0].detach().clone()

            brain.chat("Mój ulubiony język to C++.")
            c2 = brain._wm_slot_count
            self.assertGreater(c2, c1)
            # Wcześniejszy slot 0 pozostaje nienaruszony (nie jest nadpisywany ani zlewany)
            self.assertTrue(torch.allclose(brain._wm_slot_buf[0], slot0_after_t1, atol=1e-4))

            brain.chat("Jak mam na imię?")
            c3 = brain._wm_slot_count
            self.assertGreater(c3, c2)

            slots = brain._get_working_memory_slots()
            self.assertIsNotNone(slots)
            assert slots is not None
            self.assertEqual(tuple(slots.shape), (1, min(c3, 8), cfg.column_dim))
            slot_cos = float(F.cosine_similarity(slots[0, 0], slots[0, 2], dim=0).item())
            self.assertLess(slot_cos, 0.99, "Turn slots in PFC working memory should remain distinct")

    def test_06_pinned_core_memory_protection_compaction_and_sleep(self) -> None:
        """6. Ochrona rdzenia pamięci (pinned=True) podczas kompaktowania i pełnego snu."""
        cfg = self._small_cfg(cortex_sleep_epochs=1, nrem_sleep_steps=6)
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.auto_sleep = False
            brain.auto_web = False

            # Przypięty fakt tożsamościowy + ręczny engram pinned=True
            brain.chat("okej to nadaje ci imie: Orion")
            pinned_info = brain.add_manual_memory("Kluczowa zasada rdzenia: nazywam się Orion.", category="zasada", importance=5, pinned=True)
            pinned_eid = pinned_info["id"]

            # Nieprzypięte tło do przycinania
            for idx in range(12):
                brain.add_manual_memory(f"Tymczasowa notatka robocza numer {idx} o pogodzie.", category="wiedza", importance=1, pinned=False)

            pinned_replay_before = [p for p in brain.language_cortex._learned_qa_replay if p[0].lower() in brain.language_cortex._pinned_qa_keys]
            self.assertGreaterEqual(len(pinned_replay_before), 1)

            # Zasilenie M2 epizodami pinned=True oraz pinned=False i wymuszenie kompaktowania (68 MB)
            brain.living_brain.encode_external_engrams(
                torch.randn(2, 512),
                importances=[5.0, 5.0],
                pinned_flags=[True, True],
            )
            brain.living_brain.encode_external_engrams(
                torch.randn(30, 512),
                importances=[1.0] * 30,
                pinned_flags=[False] * 30,
            )
            pinned_m2_before = int(brain.living_brain.m2_pinned.sum().item())
            self.assertGreaterEqual(pinned_m2_before, 2)

            brain.living_brain.grow_structure(add_l1=64, add_l2=64, add_l3=48, add_assoc=48, add_m4=24)
            brain.living_brain.set_memory_budget(budget_mb=68.0)
            brain.living_brain.enforce_memory_budget()
            brain.sleep_now(visual_pacing=False)

            # Weryfikacja nienaruszalności wszystkich warstw pinned=True
            pinned_m2_after = int(brain.living_brain.m2_pinned.sum().item())
            self.assertGreaterEqual(pinned_m2_after, pinned_m2_before)

            all_hippo_ids = {m.id for m in brain.hippocampus.short_term.meta + brain.hippocampus.semantic.meta if m.pinned}
            self.assertIn(pinned_eid, all_hippo_ids)

            pinned_replay_after = [p for p in brain.language_cortex._learned_qa_replay if p[0].lower() in brain.language_cortex._pinned_qa_keys]
            self.assertEqual(len(pinned_replay_after), len(pinned_replay_before))

    def test_07_living_brain_neuromodulators_and_z_ctx(self) -> None:
        """7. Wpływ LivingBrainEngine i neuromodulatorów (NA, DA, ACh, ciekawość) na Z_ctx i dekoder."""
        cfg = self._small_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.auto_sleep = False
            brain.auto_web = False

            w_out = brain.language_cortex.perceive_one("Co to jest fotosynteza?")
            base_th = brain.cortex.perceive_and_think(
                w_out.columns,
                w_out.sdr,
                sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns,
                sensory_patch_columns=w_out.patch_columns,
                sensory_patch_mask=w_out.patch_mask,
                steps=3,
            )

            for _ in range(4):
                brain.living_brain.cognitive_step(sensory_vec=torch.randn(512), external_valence=0.4, allow_growth=False)
            assoc_vec = brain.living_brain.get_cortical_association_vector(top_k_m2=3)

            lb_th = brain.cortex.perceive_and_think(
                w_out.columns,
                w_out.sdr,
                sdr_to_columns_fn=brain.language_cortex.wernicke_module.sdr_to_columns,
                sensory_patch_columns=w_out.patch_columns,
                sensory_patch_mask=w_out.patch_mask,
                steps=3,
                living_assoc_vec=assoc_vec,
                living_blend=0.12,
            )
            assert base_th.context_columns is not None and lb_th.context_columns is not None
            ctx_diff = float((lb_th.context_columns - base_th.context_columns).abs().mean().item())
            self.assertGreater(ctx_diff, 1e-3, "LivingBrain state_assoc must visibly modulate Z_ctx")

            high_na = NeuromodulatorState(noradrenaline=2.2, dopamine=0.5, acetylcholine=0.8)
            high_pe = LivingStepTelemetry(surprise=0.80, intrinsic_curiosity=0.10, boredom_exploration=False)
            low_na = NeuromodulatorState(noradrenaline=0.30, dopamine=0.7, acetylcholine=0.4)
            high_cur = LivingStepTelemetry(surprise=0.15, intrinsic_curiosity=0.95, boredom_exploration=True)

            delib_na = brain.language_cortex.deliberate_and_critique(
                user_message="Co to jest fotosynteza?",
                cortex=brain.cortex,
                engram_bank=None,
                base_temperature=0.62,
                max_new_bytes=24,
                num_candidates=3,
                living_assoc_vec=assoc_vec,
                neuromod=high_na,
                living_telemetry=high_pe,
            )
            delib_cur = brain.language_cortex.deliberate_and_critique(
                user_message="Co to jest fotosynteza?",
                cortex=brain.cortex,
                engram_bank=None,
                base_temperature=0.62,
                max_new_bytes=24,
                num_candidates=3,
                living_assoc_vec=assoc_vec,
                neuromod=low_na,
                living_telemetry=high_cur,
            )

            self.assertEqual(delib_na["thought"].attractor_steps, 6)
            prec_na = next(c["temperature"] for c in delib_na["candidates"] if c["label"] == "Precyzyjna")
            prec_low = next(c["temperature"] for c in delib_cur["candidates"] if c["label"] == "Precyzyjna")
            cre_na = next(c["temperature"] for c in delib_na["candidates"] if c["label"] == "Kreatywna")
            cre_cur = next(c["temperature"] for c in delib_cur["candidates"] if c["label"] == "Kreatywna")
            self.assertLess(prec_na, prec_low)
            self.assertGreater(cre_cur, cre_na)

    def test_08_sequential_system2_and_auditory_self_critique(self) -> None:
        """8. Sekwencyjna pętla rozumowania System-2 (krok t -> slot kroku t+1) + Arcuate Fasciculus."""
        cfg = self._small_cfg(system2_reasoning_steps=3)
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.auto_sleep = False
            brain.auto_web = False
            brain.add_manual_memory("Kot jest ssakiem drapieżnym z rodziny kotowatych.", category="wiedza", importance=4)

            w_probe = brain.language_cortex.perceive_one("Opowiedz krótko o kocie.")
            assert w_probe.patch_columns is not None
            patch_len = int(w_probe.patch_columns.shape[1])

            delib = brain.language_cortex.deliberate_and_critique(
                user_message="Opowiedz krótko o kocie.",
                cortex=brain.cortex,
                engram_bank=brain.hippocampus.get_hopfield_bank(),
                working_memory_slots=torch.randn(1, 2, cfg.column_dim, device=brain.device),
                max_new_bytes=28,
                num_candidates=3,
            )

            self.assertEqual(delib["reasoning_steps"], 3)
            self.assertEqual(len(delib["candidates"]), 3)
            self.assertTrue(any("Łańcuch System-2" in note for note in delib["critique_notes"]))
            self.assertTrue(any("Słyszę" in note for note in delib["critique_notes"]))
            self.assertTrue(delib["score"] == delib["score"] and 0.0 <= delib["score"] <= 1.0)
            self.assertTrue(delib["avg_nll"] == delib["avg_nll"])
            # Weryfikacja sekwencyjnego dopisywania slotu myśli: krok 0 -> krok 1 -> krok 2
            cands = delib["candidates"]
            self.assertEqual([c["thought_slots"] for c in cands], [0, 1, 2])
            self.assertGreater(cands[1]["context_slots"], cands[0]["context_slots"])
            self.assertGreater(cands[2]["context_slots"], cands[1]["context_slots"])
            self.assertGreaterEqual(cands[2]["context_slots"], 16 + patch_len + 2 + 2)

    def test_09_gradient_flow_and_loss_decrease(self) -> None:
        """9. Przepływ gradientu do Wernickego i Broki oraz spadek Loss w krótkim treningu."""
        cfg = self._small_cfg(broca_dialogue_lr=1.5e-3, broca_dropout=0.05)
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            lc = brain.language_cortex
            lc._wernicke_raw.train()
            lc._broca_raw.train()

            # A) Sprawdzenie przepływu gradientu w pojedynczym kroku backward()
            in_b, in_m = encode_utf8_bytes(
                ["Co to jest kot?", "Kto napisał Pana Tadeusza?"],
                max_bytes=cfg.max_input_bytes,
                patch_size=cfg.patch_stride,
                device=brain.device,
            )
            tgt_b, _ = encode_utf8_bytes(
                ["Kot to zwierzę domowe.", "Adam Mickiewicz."],
                max_bytes=cfg.broca_max_seq_len,
                patch_size=1,
                add_bos_eos=True,
                device=brain.device,
            )
            lc._wernicke_raw.zero_grad(set_to_none=True)
            lc._broca_raw.zero_grad(set_to_none=True)
            w_out = lc._wernicke_raw(in_b, in_m)
            z_w = lc._normalize_z_columns(w_out.reconstructed_columns)
            z_ctx, z_mask = brain.cortex.build_context_columns(
                columns_star=z_w,
                sensory_patch_columns=w_out.patch_columns,
                sensory_patch_mask=w_out.patch_mask,
            )
            loss = lc._broca_raw.compute_loss(tgt_b, z_ctx, memory_mask=z_mask, clean_z=True) + lc._wernicke_bottleneck_loss(w_out, z_w)
            loss.backward()

            for mod_name, param in (
                ("wernicke.byte_emb", lc._wernicke_raw.byte_emb.weight),
                ("wernicke.patch_conv", lc._wernicke_raw.patch_conv.weight),
                ("wernicke.block0.qkv", lc._wernicke_raw.blocks[0].qkv.weight),
                ("wernicke.to_sdr", lc._wernicke_raw.to_sdr.weight),
                ("wernicke.sdr_expand_out", lc._wernicke_raw.sdr_expand_out.weight),
                ("broca.byte_emb", lc._broca_raw.byte_emb.weight),
                ("broca.cross_kv", lc._broca_raw.blocks[0].cross_kv.weight),
                ("broca.film_proj", lc._broca_raw.blocks[0].film_proj.weight),
                ("broca.lm_head", lc._broca_raw.lm_head.weight),
            ):
                self.assertIsNotNone(param.grad, f"Missing gradient on {mod_name}")
                grad_norm = float(param.grad.detach().float().norm().item())
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient on {mod_name}")
                self.assertGreater(grad_norm, 0.0, f"Zero gradient on {mod_name}")

            # B) Krótki trening 20 epok na 16 parach dialogowych -> wyraźny spadek Loss
            sample_pairs = [
                ("Cześć", "Cześć! Miło Cię widzieć."),
                ("Jak się masz?", "Mam się dobrze i chętnie pomogę."),
                ("Co to jest kot?", "Kot to ssak domowy z rodziny kotowatych."),
                ("Co to jest pies?", "Pies to wierny udomowiony ssak."),
                ("Co to jest słońce?", "Słońce to gwiazda w centrum Układu Słonecznego."),
                ("Co to jest woda?", "Woda to ciecz niezbędna do życia."),
                ("Co to jest procesor?", "Procesor wykonuje instrukcje programu."),
                ("Co to jest pamięć RAM?", "RAM to szybka pamięć operacyjna komputera."),
                ("Kto napisał Pana Tadeusza?", "Autorem Pana Tadeusza jest Adam Mickiewicz."),
                ("Jaka jest stolica Polski?", "Stolicą Polski jest Warszawa."),
                ("Co to jest fotosynteza?", "Fotosynteza przekształca światło w glukozę."),
                ("Co to jest algorytm?", "Algorytm to skończony przepis rozwiązania problemu."),
                ("Czy lubisz matematykę?", "Tak, matematyka opisuje porządek świata."),
                ("Dziękuję za pomoc", "Nie ma za co, zawsze chętnie pomogę."),
                ("Dobranoc", "Dobranoc, śpij dobrze!"),
                ("Pa", "Do zobaczenia wkrótce!"),
            ]
            stats = lc.train_dialogue_finetune(
                pairs=sample_pairs,
                cortex=brain.cortex,
                epochs=20,
                batch_size=8,
                early_stop_loss=0.10,
                register_questions=False,
            )
            self.assertTrue(stats["trained"])
            self.assertEqual(stats["loss_start"], stats["loss_start"])
            self.assertEqual(stats["loss_end"], stats["loss_end"])
            self.assertLess(
                stats["loss_end"],
                stats["loss_start"] * 0.75,
                f"Expected clear loss drop, got start={stats['loss_start']} -> end={stats['loss_end']}",
            )


if __name__ == "__main__":
    unittest.main()
