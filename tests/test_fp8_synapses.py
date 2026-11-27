from __future__ import annotations

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
from brain.fp8_synapse import QuantizedSynapseMatrix
from brain.hippocampus import EngramMeta, TensorEngramBank
from brain.living_brain import LivingBrainEngine
from brain.neural_cortex import BiologicalAssociativeCortex


class TestBlackwellFP8Synapses(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_01_quantized_synapse_matrix_numerical_and_stochastic_oja(self) -> None:
        """FP8 E4M3 per-row quantization: no NaN/Inf, native _scaled_mm accuracy, E4M3 clamp, small-lr Oja-Hebb."""
        torch.manual_seed(123)
        w = torch.randn(256, 256, device=self.device, dtype=torch.float32) * 0.15
        w.fill_diagonal_(0.0)
        q = QuantizedSynapseMatrix.from_tensor(w, mode="fp8", device=self.device)

        self.assertEqual(q.mode, "fp8")
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(q.element_size(), 1)
        self.assertTrue(torch.isfinite(q).all())
        self.assertEqual(float(torch.diag(q).abs().max().item()), 0.0)

        # Native Blackwell FP8 _scaled_mm (aligned) & blockwise matvec accuracy
        x_batch = torch.randn(16, 256, device=self.device, dtype=torch.float32)
        y_q = q.matmul_right_t(x_batch)
        y_ref = x_batch @ w.t()
        self.assertTrue(torch.isfinite(y_q).all())
        rel_err = float((y_q - y_ref).norm() / y_ref.norm().clamp(min=1e-6))
        self.assertLess(rel_err, 0.08, f"FP8 matmul relative error too high: {rel_err}")

        # Saturation protection against huge values (> 448)
        huge = torch.randn(32, 64, device=self.device) * 10000.0
        q_huge = QuantizedSynapseMatrix.from_tensor(huge, mode="fp8", device=self.device)
        self.assertTrue(q_huge.is_all_finite())
        self.assertTrue(torch.isfinite(q_huge.float()).all())

        # Small-lr Oja-Hebb updates accumulate cleanly via stochastic rounding without vanishing
        y_sdr = torch.zeros(256, device=self.device)
        y_sdr[:16] = 1.0 / (16.0 ** 0.5)
        x_sdr = torch.zeros(256, device=self.device)
        x_sdr[16:48] = 1.0 / (32.0 ** 0.5)
        before_block = q.float()[:16, 16:48].mean().item()
        for _ in range(30):
            q.oja_hebb_update(y_sdr, x_sdr, lr=0.01, decay=0.04, zero_diag=True)
        after_block = q.float()[:16, 16:48].mean().item()
        self.assertGreater(after_block, before_block, "Small-lr Oja-Hebb updates should increase co-active FP8 weights")
        self.assertEqual(float(torch.diag(q).abs().max().item()), 0.0)
        self.assertTrue(q.is_all_finite())

    def test_02_living_brain_fp8_oja_grow_and_budget_accounting(self) -> None:
        """LivingBrainEngine in FP8 runs cognitive_step, Oja-Hebb, grow_structure(), compaction, and ~2x smaller bytes."""
        lb_fp8 = LivingBrainEngine(
            sensory_dim=256,
            target_device="gpu" if self.device.type == "cuda" else "ram",
            memory_budget_mb=2048.0,
            start_background_clock=False,
            bio_synapse_dtype="fp8",
        )
        lb_bf16 = LivingBrainEngine(
            sensory_dim=256,
            target_device="gpu" if self.device.type == "cuda" else "ram",
            memory_budget_mb=2048.0,
            start_background_clock=False,
            bio_synapse_dtype="bf16",
        )
        self.assertEqual(lb_fp8.bio_synapse_mode, "fp8")
        self.assertEqual(lb_fp8.W_assoc_rec.dtype, torch.float8_e4m3fn)
        self.assertEqual(lb_bf16.W_assoc_rec.dtype, torch.bfloat16)

        # Accounting reflects 1B/param + row_scale (~1.9x-2.0x lower than bf16)
        syn_bytes_fp8 = lb_fp8.W_assoc_rec.nbytes() + lb_fp8.W_in_L1.nbytes() + lb_fp8.W_12.nbytes()
        syn_bytes_bf16 = lb_bf16.W_assoc_rec.nbytes() + lb_bf16.W_in_L1.nbytes() + lb_bf16.W_12.nbytes()
        ratio = syn_bytes_bf16 / float(syn_bytes_fp8)
        self.assertGreater(ratio, 1.90, f"Expected ~2x synapse memory savings, got {ratio:.2f}x")

        # Cognitive steps + Oja-Hebb + grow_structure() without NaN/Inf
        for _ in range(5):
            tel = lb_fp8.cognitive_step(torch.randn(256, device=lb_fp8.device), external_valence=0.5, allow_growth=False)
            self.assertTrue(0.0 <= tel.surprise <= 1.0)
        born = lb_fp8.grow_structure(add_l1=64, add_l2=48, add_l3=32, add_assoc=64, add_m4=16)
        self.assertGreater(born, 0)
        self.assertEqual(lb_fp8.W_assoc_rec.shape, (384 + 64, 384 + 64))
        self.assertEqual(lb_fp8.W_assoc_rec.dtype, torch.float8_e4m3fn)
        self.assertTrue(lb_fp8.W_assoc_rec.is_all_finite())
        self.assertEqual(float(torch.diag(lb_fp8.W_assoc_rec).abs().max().item()), 0.0)

        # Oja-Hebb A <-> C strengthening after growth
        oja = lb_fp8.strengthen_association_oja(
            torch.randn(256, device=lb_fp8.device),
            torch.randn(256, device=lb_fp8.device),
            lr=0.02,
        )
        self.assertGreater(oja["delta_norm"], 0.0)
        self.assertTrue(lb_fp8.W_assoc_rec.is_all_finite())

        # Compaction under tight budget works cleanly in FP8
        lb_fp8._compact_least_active_neurons(shrink_factor=0.85)
        self.assertTrue(lb_fp8.W_assoc_rec.is_all_finite())
        self.assertEqual(float(torch.diag(lb_fp8.W_assoc_rec).abs().max().item()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for real VRAM comparison on RTX 5080")
    def test_03_real_vram_savings_fp8_vs_bf16_on_rtx5080(self) -> None:
        """Real GPU VRAM usage for W_rec (12288x12288) and TensorEngramBank (4096x12288) drops ~2x in FP8 vs BF16."""
        dev = torch.device("cuda")

        # 1) BiologicalAssociativeCortex W_rec (12288 x 12288)
        cfg_bf16 = BrainConfig(bio_synapse_dtype="bf16", num_neurons=12288, num_columns=16, column_dim=64)
        cfg_fp8 = BrainConfig(bio_synapse_dtype="fp8", num_neurons=12288, num_columns=16, column_dim=64)

        torch.cuda.empty_cache()
        base_mem = torch.cuda.memory_allocated()
        c_bf16 = BiologicalAssociativeCortex(cfg_bf16, dev)
        bf16_wrec_bytes = c_bf16.synapse_memory_bytes()
        bf16_vram_delta = torch.cuda.memory_allocated() - base_mem
        del c_bf16
        torch.cuda.empty_cache()

        base_mem2 = torch.cuda.memory_allocated()
        c_fp8 = BiologicalAssociativeCortex(cfg_fp8, dev)
        fp8_wrec_bytes = c_fp8.synapse_memory_bytes()
        fp8_vram_delta = torch.cuda.memory_allocated() - base_mem2
        del c_fp8
        torch.cuda.empty_cache()

        wrec_ratio = bf16_wrec_bytes / float(fp8_wrec_bytes)
        self.assertGreater(wrec_ratio, 1.95, f"Expected ~2x W_rec savings, got {wrec_ratio:.2f}x")
        self.assertLess(fp8_vram_delta, bf16_vram_delta * 0.65)

        # 2) TensorEngramBank (4096 engrams x 12288 neurons)
        sdrs = F.normalize(torch.randn(4096, 12288, device=dev, dtype=torch.float32), dim=-1)
        metas = [EngramMeta(id=f"e{i}") for i in range(4096)]

        bank_bf16 = TensorEngramBank(12288, 492, 10000, dev, bio_synapse_dtype="bf16")
        bank_bf16.add_batch(sdrs, metas, [1.5] * 4096, [3.0] * 4096, day=1)
        engram_bf16_bytes = bank_bf16.engrams.nbytes()
        del bank_bf16
        torch.cuda.empty_cache()

        bank_fp8 = TensorEngramBank(12288, 492, 10000, dev, bio_synapse_dtype="fp8")
        bank_fp8.add_batch(sdrs, metas, [1.5] * 4096, [3.0] * 4096, day=1)
        engram_fp8_bytes = bank_fp8.engrams.nbytes()
        engram_ratio = engram_bf16_bytes / float(engram_fp8_bytes)
        self.assertGreater(engram_ratio, 1.95, f"Expected ~2x engram bank savings, got {engram_ratio:.2f}x")

        print(
            f"\n[RTX 5080 FP8 VRAM Savings] "
            f"W_rec(12288x12288): BF16={bf16_wrec_bytes/1048576:.2f} MB -> FP8={fp8_wrec_bytes/1048576:.2f} MB ({wrec_ratio:.2f}x) | "
            f"Engrams(4096x12288): BF16={engram_bf16_bytes/1048576:.2f} MB -> FP8={engram_fp8_bytes/1048576:.2f} MB ({engram_ratio:.2f}x)"
        )
        del bank_fp8, sdrs
        torch.cuda.empty_cache()

    def test_04_legacy_bf16_fp32_checkpoint_auto_migration_to_fp8(self) -> None:
        """Loading an old bf16/fp32 checkpoint into FP8 SleepingBrain auto-converts W_rec, LivingBrain, and Hippocampus without RuntimeError."""
        cfg_legacy = BrainConfig(
            language_bootstrap_epochs=0,
            bio_synapse_dtype="bf16",
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
            broca_moe_expert_dim=64,
            dmn_enabled=False,
        )
        cfg_fp8 = BrainConfig(
            language_bootstrap_epochs=0,
            bio_synapse_dtype="fp8",
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
            broca_moe_expert_dim=64,
            dmn_enabled=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            old_brain = SleepingBrain(cfg=cfg_legacy, base_dir=tmp, load_language_mode="sync")
            try:
                old_brain.hippocampus.semantic.add_batch(
                    F.normalize(torch.randn(6, 512, device=old_brain.device), dim=-1),
                    [EngramMeta(id=f"leg{i}", preview_label=f"Fakt {i}") for i in range(6)],
                    [2.0] * 6,
                    [4.0] * 6,
                    day=1,
                )
                ckpt_path = old_brain.save_state()

                # Overwrite W_rec and LivingBrain matrices with raw plain bf16/fp32 tensors (exact pre-FP8 checkpoint format!)
                saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                saved["cortex"]["W_rec"] = old_brain.cortex.W_rec.bfloat16().cpu()
                saved["living_brain"]["tensors"]["W_assoc_rec"] = old_brain.living_brain.W_assoc_rec.float().cpu()
                saved["living_brain"]["tensors"]["W_in_L1"] = old_brain.living_brain.W_in_L1.bfloat16().cpu()
                torch.save(saved, ckpt_path)
            finally:
                old_brain.living_brain.stop_spontaneous_clock()

            new_brain = SleepingBrain(cfg=cfg_fp8, base_dir=tmp, load_language_mode="sync")
            try:
                self.assertEqual(new_brain.cortex.W_rec.dtype, torch.float8_e4m3fn)
                self.assertEqual(new_brain.living_brain.W_assoc_rec.dtype, torch.float8_e4m3fn)
                self.assertEqual(new_brain.living_brain.W_in_L1.dtype, torch.float8_e4m3fn)
                self.assertEqual(new_brain.hippocampus.semantic.engrams.dtype, torch.float8_e4m3fn)
                self.assertEqual(len(new_brain.hippocampus.semantic), 6)
                self.assertTrue(new_brain.cortex.W_rec.is_all_finite())
                self.assertTrue(new_brain.living_brain.W_assoc_rec.is_all_finite())

                # Verify inference & recall work seamlessly after migration
                q = F.normalize(torch.randn(512, device=new_brain.device), dim=-1)
                rec = new_brain.hippocampus.semantic.recall_tensor(q, top_k=3, current_day=2, min_sim=-1.0)
                self.assertEqual(len(rec["ids"]), 3)
            finally:
                new_brain.living_brain.stop_spontaneous_clock()


if __name__ == "__main__":
    unittest.main(verbosity=2)
