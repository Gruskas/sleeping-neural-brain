from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from brain.agent import PIPELINE_STAGE_ORDER, SleepingBrain, normalize_training_pipeline_plan
from brain.config import BrainConfig
from brain.teacher_distill import reasoning_dialogues_path, teacher_dialogues_path
from server import BrainServer


def _small_cfg(ffn_mode: str = "dense") -> BrainConfig:
    is_peer = ffn_mode == "peer"
    return BrainConfig(
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
        broca_use_moe=(ffn_mode == "moe"),
        broca_ffn_mode=ffn_mode,
        peer_grid_size=8 if is_peer else 256,
        peer_num_heads=2 if is_peer else 8,
        peer_top_k=4 if is_peer else 8,
        peer_cache_slots=64 if is_peer else 4096,
        broca_max_seq_len=64,
        broca_max_new_bytes=24,
        max_input_bytes=64,
        language_train_epochs=3,
        nrem_sleep_steps=2,
        rem_2hop_steps=1,
        dmn_enabled=False,
    )


def _write_temp_polish_parquet(base_dir: Path) -> Path:
    rows = [
        "Warszawa jest stolicą Polski i leży nad rzeką Wisłą od wielu stuleci.",
        "Sztuczna inteligencja przetwarza język polski bajt po bajcie bez tokenizera.",
        "Kora przedczołowa analizuje złożone zależności przyczynowo-skutkowe w zdaniach.",
        "Podczas głębokiego snu hipokamp konsoliduje świeże ślady pamięciowe do kory.",
        "Polskie znaki diakrytyczne takie jak ą, ć, ę, ł, ń, ó, ś, ź, ż są kodowane w UTF-8.",
        "Każdy neuron w sieci skojarzeniowej Hopfielda utrzymuje stabilny wzorzec aktywności.",
        "Nauka z korpusu tekstów poprawia płynność gramatyczną oraz poprawną pisownię.",
        "Wiosną w polskich lasach zakwitają pierwsze kwiaty oraz młode liście drzew.",
    ] * 4
    table = pa.table({"text": rows})
    pq_path = base_dir / "train-00000-of-00006.parquet"
    pq.write_table(table, pq_path)
    return pq_path


class TestTrainingPipeline(unittest.TestCase):
    def test_01_pipeline_validation_order_and_peer_split_save(self) -> None:
        """Validate plan normalization, all-6-stage ordering, no post-parquet dialogue repeat, and split peer_bank.pt."""
        cfg = _small_cfg(ffn_mode="peer")

        with self.assertRaises(ValueError):
            normalize_training_pipeline_plan(
                {k: {"enabled": False} for k, _ in PIPELINE_STAGE_ORDER},
                cfg,
            )

        norm = normalize_training_pipeline_plan(
            {"parquet": {"max_steps": 9999999}, "sleep": {"cycles": 99}},
            cfg,
        )
        self.assertEqual(norm["parquet"]["max_steps"], 500000)
        self.assertEqual(norm["sleep"]["cycles"], 5)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_temp_polish_parquet(tmp_path)
            brain = SleepingBrain(cfg=cfg, base_dir=tmp_path, load_language_mode="sync")
            brain2 = None
            try:
                brain.auto_sleep = False
                brain.auto_web = False

                call_order: list[str] = []
                orig_pq = brain.train_polish_from_parquet
                orig_dt = brain.language_cortex.train_language_cortex
                orig_sleep = brain.sleep_now

                def _wrap_pq(*args, **kwargs):
                    call_order.append(f"parquet:skip_align={kwargs.get('skip_post_dialogue_align')}")
                    return orig_pq(*args, **kwargs)

                def _wrap_dt(*args, **kwargs):
                    call_order.append("dialogue_train")
                    return orig_dt(*args, **kwargs)

                def _wrap_sleep(*args, **kwargs):
                    call_order.append("sleep")
                    return orig_sleep(*args, **kwargs)

                def _fake_teacher(state_dir, model_id, num_dialogues, extra_topics=None, stop_event=None, progress_callback=None):
                    call_order.append("bielik_dialogues")
                    p = teacher_dialogues_path(state_dir)
                    with p.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"q": "Czym jest słońce?", "a": "Słońce jest gwiazdą w centrum układu."}, ensure_ascii=False) + "\n")
                    if progress_callback:
                        progress_callback({"done": num_dialogues, "total": num_dialogues, "generated": 1, "new_pairs": [{"q": "Czym jest słońce?", "a": "Słońce jest gwiazdą w centrum układu."}], "message": "OK"})
                    return {"generated": 1, "total_saved": 1, "error": ""}

                def _fake_reasoning(state_dir, model_id, num_samples, stop_event=None, progress_callback=None):
                    call_order.append("bielik_reasoning")
                    p = reasoning_dialogues_path(state_dir)
                    with p.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"q": "Dlaczego lód topnieje w cieple?", "a": "Ciepło topi lód w wodę.", "category": "causal"}, ensure_ascii=False) + "\n")
                    if progress_callback:
                        progress_callback({"done": num_samples, "total": num_samples, "generated": 1, "new_pairs": [{"q": "Dlaczego lód topnieje w cieple?", "a": "Ciepło topi lód w wodę.", "category": "causal"}], "message": "OK"})
                    return {"generated": 1, "filtered_ai": 0, "total_saved": 1, "error": ""}

                plan = {
                    "preset": "Testowy",
                    "parquet": {"enabled": True, "filename": "train-00000-of-00006.parquet", "max_steps": 3, "target_loss": 0.5, "batch_size": 2},
                    "bielik_dialogues": {"enabled": True, "num_dialogues": 4},
                    "bielik_reasoning": {"enabled": True, "num_samples": 3},
                    "dialogue_train": {"enabled": True, "epochs": 2, "target_loss": 0.05, "freeze_wernicke": True, "train_recalled_columns": True},
                    "knowledge": {"enabled": True, "include_elementary": True, "include_bielik_facts": True, "turbo_topics": 0},
                    "sleep": {"enabled": True, "cycles": 1},
                }

                with (
                    patch.object(brain, "train_polish_from_parquet", side_effect=_wrap_pq),
                    patch.object(brain.language_cortex, "train_language_cortex", side_effect=_wrap_dt),
                    patch.object(brain, "sleep_now", side_effect=_wrap_sleep),
                    patch("brain.agent.run_bielik_teacher_subprocess", side_effect=_fake_teacher),
                    patch("brain.agent.run_bielik_reasoning_subprocess", side_effect=_fake_reasoning),
                ):
                    res = brain.run_training_pipeline(plan)

                self.assertTrue(res["completed"])
                self.assertFalse(res["stopped"])
                self.assertEqual(
                    call_order,
                    ["parquet:skip_align=True", "bielik_dialogues", "bielik_reasoning", "dialogue_train", "sleep"],
                )
                snap = brain.snapshot()
                ps = snap["pipeline_state"]
                self.assertFalse(ps["active"])
                self.assertEqual(ps["overall_pct"], 100.0)
                self.assertEqual([st["status"] for st in ps["stages"]], ["done"] * 6)
                self.assertGreater(len(brain.hippocampus.semantic), 0)

                # Split PEER bank verification
                weights_file = tmp_path / "brain_state" / "brain_weights.pt"
                peer_file = tmp_path / "brain_state" / "peer_bank.pt"
                self.assertTrue(weights_file.exists())
                self.assertTrue(peer_file.exists())
                ckpt_data = torch.load(weights_file, map_location="cpu", weights_only=False)
                self.assertNotIn("peer_banks", ckpt_data["language_cortex"]["broca"])

                # Fast chat-turn save (save_peer_bank=False) must NOT rewrite peer_bank.pt
                mtime_before = peer_file.stat().st_mtime_ns
                time.sleep(0.02)
                brain.save_state(save_peer_bank=False)
                self.assertEqual(peer_file.stat().st_mtime_ns, mtime_before)

                # Reloading into a fresh SleepingBrain restores PEER bank from peer_bank.pt
                expected_u = brain.language_cortex.broca_module.blocks[0].ff.ram_u_q.clone()
                brain2 = SleepingBrain(cfg=cfg, base_dir=tmp_path, load_language_mode="sync")
                self.assertTrue(torch.equal(expected_u, brain2.language_cortex.broca_module.blocks[0].ff.ram_u_q))
            finally:
                if brain2 is not None:
                    brain2.living_brain.stop_spontaneous_clock()
                brain.living_brain.stop_spontaneous_clock()

    def test_02_bielik_single_gpu_cpu_gpu_transfer_batching_and_no_premature_train(self) -> None:
        """Stages 2 & 3 batch >1000 / >300 samples, share ONE CPU offload/GPU restore, and skip premature training."""
        cfg = _small_cfg(ffn_mode="dense")
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            try:
                brain.auto_sleep = False
                brain.auto_web = False

                events: list[str] = []
                teacher_chunks: list[int] = []
                reasoning_chunks: list[int] = []

                def _fake_offload() -> bool:
                    events.append("offload_cpu")
                    return True

                def _fake_restore() -> bool:
                    events.append("restore_gpu")
                    return True

                def _fake_teacher(state_dir, model_id, num_dialogues, extra_topics=None, stop_event=None, progress_callback=None):
                    events.append(f"teacher_sub:{num_dialogues}")
                    teacher_chunks.append(int(num_dialogues))
                    return {"generated": num_dialogues, "total_saved": sum(teacher_chunks), "error": ""}

                def _fake_reasoning(state_dir, model_id, num_samples, stop_event=None, progress_callback=None):
                    events.append(f"reasoning_sub:{num_samples}")
                    reasoning_chunks.append(int(num_samples))
                    return {"generated": num_samples, "filtered_ai": 0, "total_saved": sum(reasoning_chunks), "error": ""}

                orig_dt = brain.language_cortex.train_language_cortex

                def _wrap_dt(*args, **kwargs):
                    events.append("stage4_train")
                    return orig_dt(*args, **kwargs)

                plan = {
                    "parquet": {"enabled": False},
                    "bielik_dialogues": {"enabled": True, "num_dialogues": 1100},
                    "bielik_reasoning": {"enabled": True, "num_samples": 450},
                    "dialogue_train": {"enabled": True, "epochs": 2, "target_loss": 0.05},
                    "knowledge": {"enabled": False},
                    "sleep": {"enabled": False},
                }

                with (
                    patch.object(brain, "_offload_brain_to_cpu", side_effect=_fake_offload),
                    patch.object(brain, "_restore_brain_to_device", side_effect=_fake_restore),
                    patch.object(brain.language_cortex, "train_language_cortex", side_effect=_wrap_dt),
                    patch("brain.agent.run_bielik_teacher_subprocess", side_effect=_fake_teacher),
                    patch("brain.agent.run_bielik_reasoning_subprocess", side_effect=_fake_reasoning),
                ):
                    res = brain.run_training_pipeline(plan)

                self.assertTrue(res["completed"])
                self.assertEqual(teacher_chunks, [500, 500, 100])
                self.assertEqual(reasoning_chunks, [300, 150])
                # Zero premature training between Bielik stages; only 1 stage4_train call total
                self.assertEqual(events.count("stage4_train"), 1)
                # No restore_gpu between teacher_sub and reasoning_sub; restore happens after reasoning_sub and before stage4_train
                last_teacher_idx = max(i for i, e in enumerate(events) if e.startswith("teacher_sub:"))
                first_reas_idx = min(i for i, e in enumerate(events) if e.startswith("reasoning_sub:"))
                last_reas_idx = max(i for i, e in enumerate(events) if e.startswith("reasoning_sub:"))
                train_idx = events.index("stage4_train")
                self.assertFalse(any(e == "restore_gpu" for e in events[last_teacher_idx:first_reas_idx]))
                self.assertTrue(any(e == "restore_gpu" for e in events[last_reas_idx:train_idx]))
            finally:
                brain.living_brain.stop_spontaneous_clock()

    def test_03_freeze_wernicke_preserves_exact_sdr_and_trains_recalled_columns(self) -> None:
        """Stage 4 with freeze_wernicke=True preserves exact Wernicke weights & SDRs while updating Broca."""
        cfg = _small_cfg(ffn_mode="dense")
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            try:
                brain.auto_sleep = False
                brain.auto_web = False

                probe_texts = ["Warszawa jest stolicą Polski.", "Słońce świeci nad rzeką."]
                sdr_before = brain.language_cortex.perceive_texts(probe_texts).sdr.detach().clone()
                w_params_before = [p.detach().clone() for p in brain.language_cortex.wernicke_module.parameters()]
                broca_lm_before = brain.language_cortex.broca_module.lm_head.weight.detach().clone()

                res = brain.run_training_pipeline(
                    {
                        "parquet": {"enabled": False},
                        "bielik_dialogues": {"enabled": False},
                        "bielik_reasoning": {"enabled": False},
                        "dialogue_train": {
                            "enabled": True,
                            "epochs": 6,
                            "target_loss": 0.01,
                            "freeze_wernicke": True,
                            "train_recalled_columns": True,
                        },
                        "knowledge": {"enabled": False},
                        "sleep": {"enabled": False},
                    }
                )
                self.assertTrue(res["completed"])

                sdr_after = brain.language_cortex.perceive_texts(probe_texts).sdr.detach().clone()
                self.assertTrue(torch.equal(sdr_before, sdr_after))
                for p_b, p_a in zip(w_params_before, brain.language_cortex.wernicke_module.parameters()):
                    self.assertTrue(torch.equal(p_b, p_a.detach()))
                    self.assertTrue(p_a.requires_grad, "requires_grad must be restored in finally")

                broca_lm_after = brain.language_cortex.broca_module.lm_head.weight.detach().clone()
                self.assertGreater(float((broca_lm_after - broca_lm_before).abs().max().item()), 1e-6)
            finally:
                brain.living_brain.stop_spontaneous_clock()

    def test_04_stop_study_mid_pipeline_and_state_telemetry(self) -> None:
        """POST /api/study/stop interrupts active pipeline cleanly and updates GET /api/state pipeline_state."""
        cfg = _small_cfg(ffn_mode="dense")
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.auto_sleep = False
            brain.auto_web = False

            entered_stage2 = threading.Event()

            def _blocking_teacher(state_dir, model_id, num_dialogues, extra_topics=None, stop_event=None, progress_callback=None):
                entered_stage2.set()
                while stop_event is not None and not stop_event.is_set():
                    time.sleep(0.02)
                return {"generated": 0, "total_saved": 0, "error": ""}

            srv = BrainServer(brain=brain, static_dir=ROOT_DIR / "static", host="127.0.0.1", port=0).create_server()
            port = srv.server_address[1]
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                with patch("brain.agent.run_bielik_teacher_subprocess", side_effect=_blocking_teacher):
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/train_pipeline",
                        data=json.dumps({
                            "preset": "StopTest",
                            "parquet": {"enabled": False},
                            "bielik_dialogues": {"enabled": True, "num_dialogues": 20},
                            "bielik_reasoning": {"enabled": False},
                            "dialogue_train": {"enabled": True, "epochs": 3},
                            "knowledge": {"enabled": True},
                            "sleep": {"enabled": True, "cycles": 1},
                        }).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        start_res = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(start_res["ok"])
                    self.assertTrue(entered_stage2.wait(timeout=5.0))

                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10) as resp:
                        mid_state = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(mid_state["pipeline_state"]["active"])
                    self.assertEqual(mid_state["pipeline_state"]["current_stage_key"], "bielik_dialogues")

                    stop_req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/study/stop",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(stop_req, timeout=10) as resp:
                        stop_res = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(stop_res["ok"])

                    final_state = None
                    for _ in range(300):
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10) as resp:
                            st = json.loads(resp.read().decode("utf-8"))
                        if not st["pipeline_state"]["active"] and not st["study"]["active"]:
                            final_state = st
                            break
                        time.sleep(0.1)

                    self.assertIsNotNone(final_state)
                    assert final_state is not None
                    statuses = {s["key"]: s["status"] for s in final_state["pipeline_state"]["stages"]}
                    self.assertEqual(statuses["bielik_dialogues"], "stopped")
                    self.assertEqual(statuses["dialogue_train"], "stopped")
                    self.assertEqual(statuses["sleep"], "stopped")
            finally:
                srv.shutdown()
                srv.server_close()
                brain.living_brain.stop_spontaneous_clock()

    def test_05_server_train_pipeline_endpoint_end_to_end(self) -> None:
        """POST /api/train_pipeline runs end-to-end on real BrainServer and finishes with overall_pct=100."""
        cfg = _small_cfg(ffn_mode="dense")
        with tempfile.TemporaryDirectory() as tmp:
            brain = SleepingBrain(cfg=cfg, base_dir=tmp, load_language_mode="sync")
            brain.auto_sleep = False
            brain.auto_web = False

            srv = BrainServer(brain=brain, static_dir=ROOT_DIR / "static", host="127.0.0.1", port=0).create_server()
            port = srv.server_address[1]
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/train_pipeline",
                    data=json.dumps({
                        "preset": "Szybki E2E",
                        "parquet": {"enabled": False},
                        "bielik_dialogues": {"enabled": False},
                        "bielik_reasoning": {"enabled": False},
                        "dialogue_train": {"enabled": True, "epochs": 1, "target_loss": 0.05, "freeze_wernicke": True, "train_recalled_columns": True},
                        "knowledge": {"enabled": True, "include_elementary": True, "include_bielik_facts": False, "turbo_topics": 0},
                        "sleep": {"enabled": True, "cycles": 1},
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(payload["ok"])
                self.assertTrue(payload["started"])
                self.assertEqual(payload["stages"], ["dialogue_train", "knowledge", "sleep"])

                final_state = None
                for _ in range(600):
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=10) as resp:
                        st = json.loads(resp.read().decode("utf-8"))
                    if not st["pipeline_state"]["active"] and not st["study"]["active"]:
                        final_state = st
                        break
                    time.sleep(0.1)

                self.assertIsNotNone(final_state)
                assert final_state is not None
                ps = final_state["pipeline_state"]
                self.assertEqual(ps["overall_pct"], 100.0)
                self.assertEqual([s["status"] for s in ps["stages"]], ["done", "done", "done"])
            finally:
                srv.shutdown()
                srv.server_close()
                brain.living_brain.stop_spontaneous_clock()


if __name__ == "__main__":
    unittest.main(verbosity=2)
