from __future__ import annotations

import gc
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from brain.config import BrainConfig
from brain.neural_language import BrocaByteTransformer
from brain.peer_memory import PEERMicroExpertFFN, _CachedPEERFunction
from brain.sensory_wernicke import amp_autocast, encode_utf8_bytes


class TestPEERScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_01_cached_peer_function_gradients(self) -> None:
        """_CachedPEERFunction forward, grad_x, and grad_weights match dense PyTorch reference (< 1e-3 rel err)."""
        torch.manual_seed(123)
        grid_size = 8
        num_experts = grid_size * grid_size  # 64
        dim = 32
        n_tokens = 12
        k_tot = 6

        peer = PEERMicroExpertFFN(
            dim=dim,
            grid_size=grid_size,
            num_heads=2,
            top_k=3,
            cache_slots=num_experts,
            peer_lr=0.0,  # Pure read-only gradient check first
            cache_dtype=torch.float32,
            pin_ram=False,
        ).to(self.device)

        x = torch.randn(n_tokens, dim, device=self.device, dtype=torch.float32, requires_grad=True)
        raw_w = torch.randn(n_tokens, k_tot, device=self.device, dtype=torch.float32, requires_grad=True)
        weights = F.softmax(raw_w, dim=-1)
        weights.retain_grad()
        expert_ids = torch.randint(0, num_experts, (n_tokens, k_tot), device=self.device, dtype=torch.long)

        slot_ids = peer._ensure_experts_cached(expert_ids, count_stats=True)
        out = _CachedPEERFunction.apply(x, weights, expert_ids, slot_ids, peer, True)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        # Dense float32 reference using the exact dequantized RAM bank weights
        u_ref = peer.ram_u_q.float().to(self.device) * peer.ram_u_scale.float().to(self.device).unsqueeze(-1)
        v_ref = peer.ram_v_q.float().to(self.device) * peer.ram_v_scale.float().to(self.device).unsqueeze(-1)

        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = weights.detach().clone().requires_grad_(True)
        u_sel = u_ref[expert_ids]  # [N, K, D]
        v_sel = v_ref[expert_ids]  # [N, K, D]
        pre = (x_ref.unsqueeze(1) * u_sel).sum(dim=-1)
        act = F.silu(pre)
        out_ref = ((act * w_ref).unsqueeze(-1) * v_sel).sum(dim=1)
        out_ref.backward(grad_out)

        def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
            return float((a - b).norm().item() / max(1e-8, float(b.norm().item())))

        err_out = _rel_err(out.detach(), out_ref.detach())
        err_gx = _rel_err(x.grad, x_ref.grad)
        err_gw = _rel_err(weights.grad, w_ref.grad)

        self.assertLess(err_out, 1e-3, f"Forward rel err {err_out}")
        self.assertLess(err_gx, 1e-3, f"grad_x rel err {err_gx}")
        self.assertLess(err_gw, 1e-3, f"grad_weights rel err {err_gw}")

        # Full module forward+backward check with router gradients
        peer_train = PEERMicroExpertFFN(
            dim=dim,
            grid_size=grid_size,
            num_heads=2,
            top_k=4,
            cache_slots=num_experts,
            peer_lr=3e-3,
            cache_dtype=torch.float32,
            pin_ram=False,
        ).to(self.device)
        peer_train.train()
        xt = torch.randn(2, 6, dim, device=self.device, dtype=torch.float32, requires_grad=True)
        yt, aux_l, z_l = peer_train(xt, compute_aux=True)
        loss = yt.pow(2).mean() + 0.01 * aux_l + 0.001 * z_l
        loss.backward()

        self.assertEqual(sum(p.numel() for p in peer_train.parameters() if p.dim() == 2 and p.shape[0] == num_experts), 0)
        for name, p in peer_train.named_parameters():
            self.assertIsNotNone(p.grad, f"Missing grad for router param {name}")
            self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite grad for {name}")
            self.assertGreater(float(p.grad.norm().item()), 0.0, f"Zero grad for {name}")

    def test_02_lru_cache_eviction_and_stochastic_rounding_writeback(self) -> None:
        """Small lr=1e-4 updates survive repeated LRU cache eviction -> int8 RAM -> VRAM reload."""
        torch.manual_seed(77)
        dim = 64
        grid_size = 8
        num_experts = grid_size * grid_size  # 64
        cache_slots = 16
        lr = 1e-4

        peer = PEERMicroExpertFFN(
            dim=dim,
            grid_size=grid_size,
            num_heads=2,
            top_k=4,
            cache_slots=cache_slots,
            peer_lr=lr,
            weight_decay=0.0,
            pin_ram=False,
        ).to(self.device)
        peer.train()

        target_experts = torch.arange(0, 8, device=self.device, dtype=torch.long)
        evict_pool_a = torch.arange(16, 32, device=self.device, dtype=torch.long).view(2, 8)
        evict_pool_b = torch.arange(32, 48, device=self.device, dtype=torch.long).view(2, 8)
        evict_pool_c = torch.arange(48, 64, device=self.device, dtype=torch.long).view(2, 8)

        # Snapshot initial dequantized RAM weights for experts 0..7
        t_cpu = target_experts.cpu()
        u_init = peer.ram_u_q.index_select(0, t_cpu).float() * peer.ram_u_scale.index_select(0, t_cpu).float().unsqueeze(-1)
        v_init = peer.ram_v_q.index_select(0, t_cpu).float() * peer.ram_v_scale.index_select(0, t_cpu).float().unsqueeze(-1)

        # Fixed positive input & positive upstream gradient => deterministic update direction on v and u
        x_fixed = torch.full((1, dim), 0.5, device=self.device, dtype=torch.float32, requires_grad=True)
        w_fixed = torch.full((1, 8), 1.0 / 8.0, device=self.device, dtype=torch.float32)
        exp_target = target_experts.unsqueeze(0)  # [1, 8]
        gy_fixed = torch.ones(1, dim, device=self.device, dtype=torch.float32)

        # Compute reference gradient direction for v (w * silu(x·u) * gy > 0 => update pushes v negative)
        with torch.no_grad():
            pre_init = (x_fixed.detach().cpu() * u_init).sum(dim=-1)  # [8]
            act_init = F.silu(pre_init)  # [8]
            expected_dv_sign = -(act_init.unsqueeze(-1) * torch.ones(1, dim)).sign()
            sig_init = torch.sigmoid(pre_init)
            silu_g = sig_init * (1.0 + pre_init * (1.0 - sig_init))
            expected_du_sign = -((v_init.sum(dim=-1) * silu_g).unsqueeze(-1) * x_fixed.detach().cpu()).sign()

        # Cycle 40 times: update target experts 0..7 with lr=1e-4, then flush/evict them out via 16..63
        for _ in range(40):
            x_fixed.grad = None
            slot_ids = peer._ensure_experts_cached(exp_target, count_stats=True)
            out = _CachedPEERFunction.apply(x_fixed, w_fixed, exp_target, slot_ids, peer, True)
            out.backward(gy_fixed)

            # Force eviction of 0..7 by touching 48 disjoint experts (> cache_slots=16) without mutating them
            saved_lr = peer.peer_lr
            peer.peer_lr = 0.0
            for pool in (evict_pool_a, evict_pool_b, evict_pool_c):
                x_ev = torch.zeros(2, dim, device=self.device, dtype=torch.float32)
                w_ev = torch.full((2, 8), 1.0 / 8.0, device=self.device, dtype=torch.float32)
                s_ev = peer._ensure_experts_cached(pool, count_stats=True)
                _CachedPEERFunction.apply(x_ev, w_ev, pool, s_ev, peer, False)
            peer.peer_lr = saved_lr

        flushed = peer.flush_dirty_to_ram()
        self.assertGreater(peer.cache_misses, 100)
        self.assertGreaterEqual(flushed, 0)

        u_final = peer.ram_u_q.index_select(0, t_cpu).float() * peer.ram_u_scale.index_select(0, t_cpu).float().unsqueeze(-1)
        v_final = peer.ram_v_q.index_select(0, t_cpu).float() * peer.ram_v_scale.index_select(0, t_cpu).float().unsqueeze(-1)

        du = u_final - u_init
        dv = v_final - v_init
        self.assertGreater(float(dv.abs().mean().item()), 1e-4, "v weights did not persist updates across evictions")
        self.assertGreater(float(du.abs().mean().item()), 1e-4, "u weights did not persist updates across evictions")

        # Cosine alignment with expected gradient descent step direction must be strongly positive
        cos_v = float(F.cosine_similarity(dv.flatten(), expected_dv_sign.flatten(), dim=0).item())
        cos_u = float(F.cosine_similarity(du.flatten(), expected_du_sign.flatten(), dim=0).item())
        self.assertGreater(cos_v, 0.70, f"Low update direction alignment for v across int8 evictions: {cos_v}")
        self.assertGreater(cos_u, 0.50, f"Low update direction alignment for u across int8 evictions: {cos_u}")

    def test_03_broca_peer_integration_grad_ckpt_and_generation(self) -> None:
        """BrocaByteTransformer(broca_ffn_mode='peer'): 0 AdamW bank params, grad_ckpt, generation, export/import."""
        for use_ckpt in (False, True):
            cfg = BrainConfig(
                column_dim=128,
                broca_dim=128,
                broca_layers=3,
                broca_heads=4,
                broca_ff_dim=256,
                broca_ffn_mode="peer",
                peer_grid_size=16,
                peer_num_heads=4,
                peer_top_k=4,
                peer_cache_slots=128,
                peer_lr=3e-3,
                peer_weight_decay=1e-4,
                broca_max_seq_len=96,
                broca_max_new_bytes=20,
                broca_gradient_checkpointing=use_ckpt,
            )
            broca = BrocaByteTransformer(cfg).to(self.device)
            broca.train()

            # Verify zero bank weights in AdamW parameters()
            adam_params = sum(p.numel() for p in broca.parameters())
            total_params = broca.total_parameters()
            active_params = broca.active_parameters()
            expected_bank = 3 * 2 * (16 * 16) * 128
            self.assertEqual(total_params - adam_params, expected_bank)
            self.assertLess(active_params, total_params)

            # Verify no parameter tensor exceeds router scale
            for name, p in broca.named_parameters():
                self.assertNotIn("ram_", name)
                self.assertNotIn("cache_", name)

            opt = torch.optim.AdamW(broca.parameters(), lr=1e-3, weight_decay=0.01)
            texts = [
                "Sztuczna sieć neuronowa przetwarza bajty UTF-8.",
                "Mikro-eksperci PEER używają pamięci podręcznej LRU.",
                "Test integracji gradient checkpointing dla trybu PEER.",
            ]
            tgt_bytes, _ = encode_utf8_bytes(texts, max_bytes=64, patch_size=1, add_bos_eos=True, device=self.device)
            z_mem = torch.randn(len(texts), 16, 128, device=self.device)
            z_mask = torch.ones(len(texts), 16, dtype=torch.bool, device=self.device)

            opt.zero_grad(set_to_none=True)
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
            opt.step()

            stats = broca.get_peer_cache_stats()
            self.assertEqual(stats["num_peer_layers"], 3)
            self.assertEqual(stats["total_experts"], 3 * 256)
            self.assertEqual(stats["active_experts_per_token"], 16)
            self.assertGreater(stats["ram_bank_mb"], 0.0)
            self.assertGreater(stats["vram_cache_mb"], 0.0)

            # Autoregressive generation (stream + batch)
            broca.eval()
            stream_out = "".join(
                broca.generate_stream(z_mem[:1], max_new_bytes=16, temperature=0.7, memory_mask=z_mask[:1])
            )
            self.assertIsInstance(stream_out, str)
            batch_out = broca.generate_batch(
                z_mem[:2], temperatures=[0.6, 0.8], max_new_bytes=16, memory_mask=z_mask[:2]
            )
            self.assertEqual(len(batch_out), 2)

            # export_state flushes dirty slots and import_state restores cleanly
            with tempfile.TemporaryDirectory() as tmp:
                ckpt_file = Path(tmp) / "peer_broca.pt"
                state = broca.export_state()
                for blk in broca.blocks:
                    if isinstance(blk.ff, PEERMicroExpertFFN):
                        self.assertFalse(bool(blk.ff.slot_dirty.any().item()))
                torch.save(state, ckpt_file)

                broca2 = BrocaByteTransformer(cfg).to(self.device)
                broca2.import_state(torch.load(ckpt_file, map_location="cpu", weights_only=False), device=self.device)
                self.assertTrue(
                    torch.equal(broca.blocks[0].ff.ram_u_q, broca2.blocks[0].ff.ram_u_q)
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required for convergence test")
    def test_04_gpu_convergence_polish_utf8(self) -> None:
        """50 GPU training steps in PEER mode reduce Polish UTF-8 CE loss >= 30% and raise cache hit rate."""
        torch.manual_seed(2026)
        cfg = BrainConfig(
            column_dim=192,
            broca_dim=192,
            broca_layers=4,
            broca_heads=6,
            broca_ff_dim=384,
            broca_ffn_mode="peer",
            peer_grid_size=32,
            peer_num_heads=4,
            peer_top_k=4,
            peer_cache_slots=512,
            peer_lr=5e-3,
            peer_weight_decay=1e-4,
            broca_max_seq_len=128,
            broca_dropout=0.05,
            broca_decoder_byte_dropout=0.05,
            broca_gradient_checkpointing=False,
        )
        broca = BrocaByteTransformer(cfg).to(self.device)
        broca.train()
        opt = torch.optim.AdamW(broca.parameters(), lr=2e-3, weight_decay=0.01, fused=True)

        sentences = [
            "Sztuczna inteligencja uczy się języka polskiego bajt po bajcie.",
            "Warszawa jest stolicą Polski i leży nad rzeką Wisłą.",
            "Kora Broki dekoduje kolumny korowe z pamięci skojarzeniowej.",
            "Mikro-eksperci PEER aktywują rzadką siatkę wag w pamięci RAM.",
            "Głębokie sieci neuronowe przetwarzają znaki diakrytyczne ą, ć, ę, ł, ń, ó, ś, ź, ż.",
            "Algorytmy uczenia maszynowego minimalizują entropię krzyżową na GPU.",
            "Szybka pamięć podręczna LRU ogranicza zużycie pamięci karty graficznej.",
            "Polska literatura i nauka rozwijają się dynamicznie od stuleci.",
        ] * 2
        tgt_bytes, _ = encode_utf8_bytes(sentences, max_bytes=96, patch_size=1, add_bos_eos=True, device=self.device)
        z_mem = torch.randn(len(sentences), 16, 192, device=self.device)

        losses: list[float] = []
        for step in range(50):
            opt.zero_grad(set_to_none=True)
            with amp_autocast(self.device):
                ce_l, aux_l, z_l = broca.compute_loss(tgt_bytes, z_mem, clean_z=True, return_components=True)
                loss = ce_l + 0.01 * aux_l + 0.001 * z_l
            self.assertTrue(torch.isfinite(loss).all())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(broca.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(ce_l.detach().item()))

        ce_start = sum(losses[:3]) / 3.0
        ce_end = sum(losses[-3:]) / 3.0
        drop_pct = (ce_start - ce_end) / max(1e-6, ce_start) * 100.0
        stats = broca.get_peer_cache_stats()

        print(
            f"\n[PEER Polish UTF-8 50-step Convergence] CE start={ce_start:.4f} -> end={ce_end:.4f} "
            f"(drop={drop_pct:.1f}%) | Cache hit rate={stats['hit_rate_pct']:.2f}%"
        )
        self.assertGreaterEqual(drop_pct, 30.0, f"Expected >=30% CE drop, got {drop_pct:.1f}%")
        self.assertGreater(stats["hit_rate_pct"], 50.0, f"Expected cache hit rate > 50%, got {stats['hit_rate_pct']}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required for RTX 5080 scaling benchmark")
    def test_05_rtx5080_1b_and_4b_scaling_benchmark(self) -> None:
        """Benchmark 10-layer PEER banks at 1.01B (M=256) and 4.03B (M=512) params under 6.5 GB VRAM."""
        vram_budget_bytes = 6.5 * (1024 ** 3)
        results: list[dict[str, float | int | str]] = []

        for label, grid_size in (("1B (M=256)", 256), ("4B (M=512)", 512)):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            layers = nn.ModuleList(
                [
                    PEERMicroExpertFFN(
                        dim=768,
                        grid_size=grid_size,
                        num_heads=8,
                        top_k=8,
                        cache_slots=4096,
                        peer_lr=3e-3,
                        weight_decay=1e-4,
                    )
                    for _ in range(10)
                ]
            ).to(self.device)
            layers.train()

            router_params = sum(p.numel() for p in layers.parameters())
            bank_params = sum(l.bank_parameters() for l in layers)
            total_params = router_params + bank_params
            ram_bank_mb = sum(l.ram_bank_bytes() for l in layers) / (1024.0 * 1024.0)

            opt = torch.optim.AdamW(layers.parameters(), lr=1e-3, fused=True)
            x = torch.randn(4, 128, 768, device=self.device, dtype=torch.bfloat16)

            step_times_ms: list[float] = []
            for _ in range(10):
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                opt.zero_grad(set_to_none=True)
                h = x
                aux_tot = torch.zeros((), device=self.device, dtype=torch.float32)
                z_tot = torch.zeros((), device=self.device, dtype=torch.float32)
                with amp_autocast(self.device):
                    for lyr in layers:
                        y, a_l, z_l = lyr(h, compute_aux=True)
                        h = h + y
                        aux_tot = aux_tot + a_l
                        z_tot = z_tot + z_l
                    loss = h.float().pow(2).mean() + 0.01 * aux_tot + 0.001 * z_tot

                self.assertTrue(torch.isfinite(loss).all())
                loss.backward()
                opt.step()

                torch.cuda.synchronize()
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)

            peak_res_bytes = torch.cuda.max_memory_reserved()
            peak_res_mb = peak_res_bytes / (1024.0 * 1024.0)
            hits = sum(l.cache_hits for l in layers)
            misses = sum(l.cache_misses for l in layers)
            hit_rate = (100.0 * hits / max(1, hits + misses))
            avg_step_ms = sum(step_times_ms) / len(step_times_ms)

            results.append(
                {
                    "label": label,
                    "total_params": total_params,
                    "ram_mb": ram_bank_mb,
                    "peak_vram_mb": peak_res_mb,
                    "hit_rate": hit_rate,
                    "step_ms": avg_step_ms,
                }
            )

            self.assertLess(
                peak_res_bytes,
                vram_budget_bytes,
                f"{label} exceeded 6.5 GB VRAM budget: {peak_res_mb:.1f} MB reserved",
            )

            del opt, layers, x, h, loss, aux_tot, z_tot
            gc.collect()
            torch.cuda.empty_cache()

        print("\n=== RTX 5080 PEER Scaling Benchmark (10 layers, D=768, H=8, K=8, C=4096) ===")
        print(
            f"{'Config':<14} | {'Total Params':>14} | {'RAM Bank (MB)':>13} | "
            f"{'Peak VRAM (MB)':>14} | {'Hit Rate (%)':>12} | {'Avg Step (ms)':>13}"
        )
        print("-" * 92)
        for r in results:
            print(
                f"{str(r['label']):<14} | {int(r['total_params']):>14,} | {float(r['ram_mb']):>13.1f} | "
                f"{float(r['peak_vram_mb']):>14.1f} | {float(r['hit_rate']):>12.2f} | {float(r['step_ms']):>13.2f}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
