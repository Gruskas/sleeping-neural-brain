from __future__ import annotations

import logging
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
from brain.neural_language import BrocaByteTransformer, SparseMoEFFN
from brain.sensory_wernicke import amp_autocast, encode_utf8_bytes


class TestBrocaSparseMoE(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_01_config_and_parameter_capacity(self) -> None:
        """Active FFN params at top_k=2 match dense FFN; total FFN capacity grows 4x."""
        cfg_dense = BrainConfig(broca_use_moe=False)
        cfg_moe = BrainConfig(broca_use_moe=True)

        self.assertTrue(cfg_moe.broca_use_moe)
        self.assertEqual(cfg_moe.broca_num_experts, 8)
        self.assertEqual(cfg_moe.broca_moe_top_k, 2)
        self.assertEqual(cfg_moe.broca_moe_expert_dim, 1152)
        self.assertAlmostEqual(cfg_moe.broca_moe_aux_loss_coef, 0.01)
        self.assertAlmostEqual(cfg_moe.broca_moe_z_loss_coef, 0.001)

        broca_dense = BrocaByteTransformer(cfg_dense)
        broca_moe = BrocaByteTransformer(cfg_moe)

        dense_ff_params = sum(p.numel() for p in broca_dense.blocks[0].ff.parameters())
        moe_blk = broca_moe.blocks[0].ff
        assert isinstance(moe_blk, SparseMoEFFN)
        moe_expert_params = moe_blk.w_gate_up.numel() + moe_blk.w_down.numel()
        active_ff_params = moe_expert_params * cfg_moe.broca_moe_top_k // cfg_moe.broca_num_experts

        self.assertEqual(active_ff_params, dense_ff_params)
        self.assertEqual(moe_expert_params, dense_ff_params * 4)

        # Total Broca active params closely match dense Broca (only +router weights)
        self.assertAlmostEqual(
            broca_moe.active_parameters() / broca_dense.total_parameters(),
            1.0,
            delta=0.01,
        )
        print(
            f"\n[MoE Params] Dense={broca_dense.total_parameters()/1e6:.2f}M | "
            f"MoE Total={broca_moe.total_parameters()/1e6:.2f}M | "
            f"MoE Active={broca_moe.active_parameters()/1e6:.2f}M | "
            f"FFN ratio={moe_expert_params/dense_ff_params:.2f}x"
        )

    def test_02_sparse_moe_numerical_equivalence_and_zero_nan(self) -> None:
        """Grouped CUDA bf16 dispatch matches reference per-expert SwiGLU and passes backward without NaN/Inf."""
        torch.manual_seed(42)
        moe = SparseMoEFFN(dim=128, expert_dim=192, num_experts=8, top_k=2, dropout=0.0).to(self.device)
        moe.eval()

        # Check both training-size batch and T=1 single-token inference path
        for shape in ((4, 32, 128), (1, 1, 128), (3, 1, 128)):
            x = torch.randn(*shape, device=self.device, dtype=torch.float32)
            with amp_autocast(self.device):
                out, aux_l, z_l = moe(x, compute_aux=True)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(torch.isfinite(out).all(), f"Non-finite output for shape {shape}")
            self.assertTrue(torch.isfinite(aux_l).all())
            self.assertTrue(torch.isfinite(z_l).all())

            # Compare against explicit fp32 reference implementation
            x_flat = x.reshape(-1, 128)
            logits = F.linear(x_flat.float(), moe.router.weight.float())
            probs = F.softmax(logits, dim=-1)
            top_p, top_i = torch.topk(probs, k=2, dim=-1, sorted=False)
            top_w = top_p / top_p.sum(dim=-1, keepdim=True)
            ref = torch.zeros_like(x_flat)
            for n in range(x_flat.shape[0]):
                for s in range(2):
                    e = int(top_i[n, s].item())
                    gu = x_flat[n] @ moe.w_gate_up[e].float()
                    g, u = gu.chunk(2, dim=-1)
                    y = (F.silu(g) * u) @ moe.w_down[e].float()
                    ref[n] += y * top_w[n, s]
            ref = ref.view_as(x)
            max_diff = float((out.float() - ref.float()).abs().max().item())
            self.assertLess(max_diff, 5e-2, f"Max diff {max_diff} too large for shape {shape}")

    def test_03_forward_backward_grad_ckpt_and_generation_t1(self) -> None:
        """Forward+backward (with & without gradient checkpointing) and T=1 KV-cache generation."""
        for use_ckpt in (False, True):
            cfg = BrainConfig(
                column_dim=128,
                broca_dim=128,
                broca_layers=3,
                broca_heads=4,
                broca_ff_dim=256,
                broca_use_moe=True,
                broca_num_experts=8,
                broca_moe_top_k=2,
                broca_moe_expert_dim=128,
                broca_max_seq_len=128,
                broca_max_new_bytes=24,
                broca_gradient_checkpointing=use_ckpt,
            )
            broca = BrocaByteTransformer(cfg).to(self.device)
            broca.train()
            texts = [
                "Sztuczna sieć neuronowa uczy się języka polskiego bajt po bajcie.",
                "Architektura Mixture of Experts aktywuje dwóch z ośmiu ekspertów.",
                "Kora Broki dekoduje kolumny korowe bez tokenizera.",
                "Test sprawdzający poprawność gradientów oraz brak wartości NaN.",
            ]
            tgt_bytes, _ = encode_utf8_bytes(texts, max_bytes=96, patch_size=1, add_bos_eos=True, device=self.device)
            z_mem = torch.randn(len(texts), 20, 128, device=self.device)
            z_mask = torch.ones(len(texts), 20, dtype=torch.bool, device=self.device)

            broca.zero_grad(set_to_none=True)
            with amp_autocast(self.device):
                ce_l, aux_l, z_l = broca.compute_loss(
                    tgt_bytes, z_mem, memory_mask=z_mask, return_components=True
                )
                loss = ce_l + cfg.broca_moe_aux_loss_coef * aux_l + cfg.broca_moe_z_loss_coef * z_l
            self.assertTrue(torch.isfinite(loss).all())
            loss.backward()

            for name, p in broca.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite grad in {name} (ckpt={use_ckpt})")
            for idx, blk in enumerate(broca.blocks):
                assert isinstance(blk.ff, SparseMoEFFN)
                self.assertIsNotNone(blk.ff.router.weight.grad, f"Missing router grad in layer {idx}")
                self.assertGreater(float(blk.ff.router.weight.grad.float().norm().item()), 0.0)
                self.assertIsNotNone(blk.ff.w_gate_up.grad, f"Missing w_gate_up grad in layer {idx}")
                self.assertIsNotNone(blk.ff.w_down.grad, f"Missing w_down grad in layer {idx}")

            # T=1 autoregressive generation with preallocated KV-cache (stream + batch)
            broca.eval()
            stream_out = "".join(
                broca.generate_stream(z_mem[:1], max_new_bytes=20, temperature=0.7, memory_mask=z_mask[:1])
            )
            self.assertIsInstance(stream_out, str)
            batch_out = broca.generate_batch(
                z_mem[:3], temperatures=[0.5, 0.7, 0.85], max_new_bytes=20, memory_mask=z_mask[:3]
            )
            self.assertEqual(len(batch_out), 3)

    def test_04_expert_load_balancing_after_training_steps(self) -> None:
        """After 20 training steps, all 8 experts stay active across layers (no router collapse)."""
        torch.manual_seed(7)
        cfg = BrainConfig(
            column_dim=128,
            broca_dim=128,
            broca_layers=4,
            broca_heads=4,
            broca_ff_dim=256,
            broca_use_moe=True,
            broca_num_experts=8,
            broca_moe_top_k=2,
            broca_moe_expert_dim=128,
            broca_max_seq_len=128,
        )
        broca = BrocaByteTransformer(cfg).to(self.device)
        broca.train()
        opt = torch.optim.AdamW(broca.parameters(), lr=1.5e-3, weight_decay=0.01)

        samples = [
            "Kot to ssak domowy z rodziny kotowatych, który lubi ciepło.",
            "Słońce jest gwiazdą w centrum Układu Słonecznego.",
            "Fotosynteza przekształca energię światła w związki organiczne.",
            "Procesor wykonuje instrukcje programu komputerowego bardzo szybko.",
            "Warszawa jest stolicą Polski i leży nad rzeką Wisłą.",
            "Pamięć robocza przechowuje aktualnie przetwarzane informacje.",
            "Język polski zawiera znaki diakrytyczne takie jak ą, ć, ę, ł, ń, ó, ś, ź, ż.",
            "Algorytm to skończony ciąg jasno zdefiniowanych kroków obliczeniowych.",
        ] * 2
        tgt_bytes, _ = encode_utf8_bytes(samples, max_bytes=120, patch_size=1, add_bos_eos=True, device=self.device)
        z_mem = torch.randn(len(samples), 16, 128, device=self.device)

        for _ in range(20):
            opt.zero_grad(set_to_none=True)
            with amp_autocast(self.device):
                loss = broca.compute_loss(tgt_bytes, z_mem)
            self.assertTrue(torch.isfinite(loss).all())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(broca.parameters(), max_norm=1.0)
            opt.step()

        util = broca.get_moe_expert_utilization()
        self.assertIsNotNone(util)
        assert util is not None
        util_cpu = util.float().cpu()  # [layers, 8]
        mean_per_expert = util_cpu.mean(dim=0)
        active_experts = int((mean_per_expert > 0.02).sum().item())
        print(
            f"\n[MoE Load Balance after 20 steps] mean per expert: "
            f"{[round(float(x), 4) for x in mean_per_expert.tolist()]} "
            f"(active={active_experts}/8, min_layer_expert={float(util_cpu.min().item()):.4f})"
        )
        self.assertEqual(active_experts, 8, f"Expected all 8 experts active, got {mean_per_expert.tolist()}")
        for layer_i in range(cfg.broca_layers):
            layer_active = int((util_cpu[layer_i] > 0.01).sum().item())
            self.assertEqual(
                layer_active,
                8,
                f"Layer {layer_i} collapsed to {layer_active}/8 experts: {util_cpu[layer_i].tolist()}",
            )

    def test_05_dense_to_moe_checkpoint_migration(self) -> None:
        """Loading an old dense-FFN brain_weights.pt into MoE model succeeds without RuntimeError and logs migration."""
        cfg_dense = BrainConfig(
            language_bootstrap_epochs=0,
            num_neurons=512,
            sdr_k=24,
            num_columns=16,
            column_dim=64,
            wernicke_layers=2,
            wernicke_heads=4,
            wernicke_ff_dim=128,
            broca_dim=64,
            broca_layers=2,
            broca_heads=4,
            broca_ff_dim=128,
            broca_use_moe=False,
            broca_max_seq_len=96,
        )
        cfg_moe = BrainConfig(
            language_bootstrap_epochs=0,
            num_neurons=512,
            sdr_k=24,
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
        )
        with tempfile.TemporaryDirectory() as tmp:
            dense_brain = SleepingBrain(cfg=cfg_dense, base_dir=tmp, load_language_mode="sync")
            dense_qkv_before = dense_brain.language_cortex.broca_module.blocks[0].qkv.weight.detach().cpu().clone()
            ckpt_path = dense_brain.save_state()

            # Also inject an explicit shape-mismatched tensor to test mismatch skipping
            saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            saved["language_cortex"]["broca"]["weights"]["z_summary_proj.weight"] = torch.randn(64, 32)
            torch.save(saved, ckpt_path)

            with self.assertLogs("brain.neural_language", level=logging.WARNING) as log_cm:
                moe_brain = SleepingBrain(cfg=cfg_moe, base_dir=tmp, load_language_mode="sync")

            self.assertTrue(any("migracja checkpointu" in msg.lower() for msg in log_cm.output))
            broca_moe = moe_brain.language_cortex.broca_module
            rep = broca_moe.last_import_report
            self.assertIsNotNone(rep)
            assert rep is not None
            self.assertGreater(len(rep["loaded"]), 10)
            self.assertEqual(len(rep["mismatched"]), 1)  # z_summary_proj.weight
            self.assertEqual(len(rep["missing"]), 2 * 3)  # 2 layers * (router.weight, w_gate_up, w_down)
            self.assertEqual(len(rep["unexpected"]), 2 * 2)  # 2 layers * (ff.w_gate_up.weight, ff.w_down.weight)

            # Shared attention weights were preserved (within bf16 export tolerance)
            moe_qkv_after = broca_moe.blocks[0].qkv.weight.detach().cpu()
            self.assertLess(float((moe_qkv_after - dense_qkv_before).abs().max().item()), 1e-2)

            # Migrated MoE model runs forward+backward cleanly
            tgt_b, _ = encode_utf8_bytes(["Test migracji MoE"], max_bytes=48, patch_size=1, add_bos_eos=True, device=moe_brain.device)
            z_m = torch.randn(1, 16, 64, device=moe_brain.device)
            with amp_autocast(moe_brain.device):
                loss = broca_moe.compute_loss(tgt_b, z_m)
            self.assertTrue(torch.isfinite(loss).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for RTX 5080 VRAM budget check")
    def test_06_rtx5080_full_config_vram_budget(self) -> None:
        """Full production BrainConfig (D=768, 10 layers, 8 experts, top-2) training step stays well under 16 GB VRAM."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        cfg = BrainConfig()  # full production config
        broca = BrocaByteTransformer(cfg).to(self.device)
        broca.train()
        opt = torch.optim.AdamW(broca.parameters(), lr=cfg.broca_lr, fused=True)

        # Realistic training batch (batch_size=24, seq_len=256 bytes, 80 cortical context slots)
        b, t, m = cfg.broca_batch_size, 256, 80
        tgt_bytes = torch.randint(32, 126, (b, t), device=self.device, dtype=torch.long)
        tgt_bytes[:, 0] = 257  # <BOS>
        z_ctx = torch.randn(b, m, cfg.column_dim, device=self.device, dtype=torch.bfloat16)

        opt.zero_grad(set_to_none=True)
        with amp_autocast(self.device):
            loss = broca.compute_loss(tgt_bytes, z_ctx)
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(broca.parameters(), max_norm=1.0)
        opt.step()
        torch.cuda.synchronize()

        peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

        # Also test T=1 streaming step on full production config
        broca.eval()
        gen = "".join(broca.generate_stream(z_ctx[:1], max_new_bytes=24, temperature=0.7))
        self.assertIsInstance(gen, str)

        print(
            f"\n[RTX 5080 Full Config VRAM] batch={b}, seq={t}, ctx={m} | "
            f"Peak Allocated={peak_alloc_gb:.2f} GB | Peak Reserved={peak_res_gb:.2f} GB (budget: 16.00 GB)"
        )
        self.assertLess(peak_alloc_gb, 14.5, f"Peak allocated VRAM {peak_alloc_gb:.2f} GB too close to 16 GB budget!")

        del opt, broca, tgt_bytes, z_ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
