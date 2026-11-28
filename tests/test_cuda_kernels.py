from __future__ import annotations

import copy
import ctypes
import time
import unittest

import torch
import torch.nn.functional as F

from brain.config import BrainConfig
from brain.cuda_ops import (
    CUDA_EXT_ERROR,
    HAS_CUDA_EXT,
    quantize_rows_cuda,
    shy_prune_rows_cuda,
)
from brain.fp8_synapse import QuantizedSynapseMatrix
from brain.neural_language import BrocaByteTransformer
from brain.peer_memory import PEERMicroExpertFFN


def _available_ram_gb() -> float:
    class _MEMSTAT(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        ms = _MEMSTAT()
        ms.dwLength = ctypes.sizeof(_MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / (1024.0**3)
    except Exception:
        return 8.0


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class TestCUDAKernelsAndGraphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda")
        cls.gpu_name = torch.cuda.get_device_name(0)
        cls.cap = torch.cuda.get_device_capability(0)
        print(
            f"\n[CUDA ENV] device={cls.gpu_name} cc={cls.cap} "
            f"torch={torch.__version__} HAS_CUDA_EXT={HAS_CUDA_EXT} err={CUDA_EXT_ERROR!r}"
        )

    def test_cuda_extension_loaded(self) -> None:
        self.assertTrue(HAS_CUDA_EXT, f"Expected compiled CUDA extension, got error: {CUDA_EXT_ERROR}")

    def test_fp8_and_int8_quantize_parity_and_zeros(self) -> None:
        torch.manual_seed(42)
        src = torch.randn((256, 512), device=self.device, dtype=torch.bfloat16) * 0.5
        # Inject exact zeros including diagonal + an all-zero row + extreme magnitudes
        src.fill_diagonal_(0.0)
        src[0].zero_()
        src[1, :4] = torch.tensor([1e-6, -1e-6, 12.5, -12.5], device=self.device, dtype=torch.bfloat16)

        for mode in ("fp8", "int8"):
            q_cuda = QuantizedSynapseMatrix.from_tensor(
                src, mode=mode, stochastic=False, _force_pytorch=False
            )
            q_ref = QuantizedSynapseMatrix.from_tensor(
                src, mode=mode, stochastic=False, _force_pytorch=True
            )

            # Exact zeros preserved strictly
            self.assertTrue(bool((q_cuda.data.float().diagonal() == 0.0).all().item()))
            self.assertTrue(bool((q_cuda.data[0].float() == 0.0).all().item()))

            # Scale and quantized data match reference closely
            scale_diff = (q_cuda.row_scale.float() - q_ref.row_scale.float()).abs().max().item()
            self.assertLess(scale_diff, 1e-6, f"row_scale mismatch for mode={mode}")

            deq_cuda = q_cuda.float()
            deq_ref = q_ref.float()
            exact_frac = (q_cuda.data.float() == q_ref.data.float()).float().mean().item()
            mean_diff = (deq_cuda - deq_ref).abs().mean().item()
            self.assertGreater(exact_frac, 0.995, f"Too many tie-flips for mode={mode}: {exact_frac}")
            self.assertLess(mean_diff, 1e-4 if mode == "fp8" else 1e-3, f"mean diff for mode={mode}")

    def test_fp8_stochastic_rounding_unbiased(self) -> None:
        torch.manual_seed(123)
        # Value between adjacent FP8 E4M3 grid points: 1.0 and 1.125 (midpoint ~ 1.0625)
        base = torch.full((64, 512), 1.0625, device=self.device, dtype=torch.float32)
        base[:, 0] = 2.0  # row_scale reference anchor (within E4M3 headroom)
        # Wait: let's check actual dequantized value at 1.0625/448 -> scale = 2.0/448 => val*448/2.0 = 238.0
        # Better: anchor at max=448.0 so row_scale = 1.0 exactly, then test values [1.0625, 1.03125, -1.04]
        base[:, 0] = 448.0
        base[:, 1] = 1.0625
        base[:, 2] = 0.0  # Exact zero must remain exact zero even with stochastic rounding

        acc = torch.zeros((64,), device=self.device, dtype=torch.float64)
        n_trials = 160
        for t in range(n_trials):
            q_d, q_s = quantize_rows_cuda(base, 0, True, 1000 + t)
            zero_col = q_d[:, 2].float()
            self.assertTrue(bool((zero_col == 0.0).all().item()))
            deq_col = q_d[:, 1].float() * q_s
            acc += deq_col.double()

        mean_est = (acc / float(n_trials)).mean().item()
        self.assertAlmostEqual(mean_est, 1.0625, delta=0.025)

    def test_fp8_shy_prune_cuda_parity(self) -> None:
        torch.manual_seed(7)
        mat = torch.randn((384, 384), device=self.device, dtype=torch.float32) * 0.04
        mat[torch.abs(mat) < 0.01] = 0.0

        for mode in ("fp8", "int8"):
            q_cuda = QuantizedSynapseMatrix.from_tensor(
                mat, mode=mode, stochastic=False, _force_pytorch=False
            )
            # Clone identical quantized tensor so SHY prune parity is tested cleanly in isolation
            q_ref = q_cuda.clone()

            stats_cuda = q_cuda.apply_shy_prune(
                downscale=0.92, eps=0.02, zero_diag=True, _force_pytorch=False
            )
            stats_ref = q_ref.apply_shy_prune(
                downscale=0.92, eps=0.02, zero_diag=True, _force_pytorch=True
            )

            self.assertEqual(stats_cuda, stats_ref, f"SHY prune stats mismatch for mode={mode}")
            self.assertTrue(bool((q_cuda.data.float().diagonal() == 0.0).all().item()))
            diff = (q_cuda.float() - q_ref.float()).abs().max().item()
            self.assertLess(diff, 1e-5, f"SHY prune float diff for mode={mode}")

    def test_peer_forward_backward_parity_and_stateless_lr_survival(self) -> None:
        torch.manual_seed(99)
        dim = 128
        grid = 32  # E = 1024
        # Compare CUDA extension vs PyTorch fallback starting from identical weights & cache
        peer_ref = PEERMicroExpertFFN(
            dim=dim,
            grid_size=grid,
            num_heads=4,
            top_k=4,
            cache_slots=256,
            peer_lr=3e-3,
            weight_decay=1e-4,
            dropout=0.0,
        ).to(self.device)
        peer_cuda = copy.deepcopy(peer_ref).to(self.device)
        peer_cuda.import_bank_state(peer_ref.export_bank_state())

        peer_ref._use_cuda_ext = False
        peer_cuda._use_cuda_ext = True

        # Router params only in AdamW / parameters()
        bank_attr_names = {
            "ram_u_q",
            "ram_u_scale",
            "ram_v_q",
            "ram_v_scale",
            "cache_u",
            "cache_v",
        }
        param_names = {name for name, _ in peer_cuda.named_parameters()}
        self.assertTrue(param_names.isdisjoint(bank_attr_names))

        x = torch.randn((2, 32, dim), device=self.device, dtype=torch.float32)
        x_ref = x.detach().clone().requires_grad_(True)
        x_cu = x.detach().clone().requires_grad_(True)

        # Eval mode check (pure forward/backward parity without in-place cache mutation divergence)
        peer_ref.eval()
        peer_cuda.eval()
        out_ref, aux_ref, z_ref = peer_ref(x_ref, compute_aux=True)
        out_cu, aux_cu, z_cu = peer_cuda(x_cu, compute_aux=True)

        rel_out = (out_cu - out_ref).norm() / out_ref.norm().clamp(min=1e-6)
        self.assertLess(rel_out.item(), 2e-2)
        self.assertLess((aux_cu - aux_ref).abs().item(), 1e-5)
        self.assertLess((z_cu - z_ref).abs().item(), 1e-5)

        grad_probe = torch.randn_like(out_ref)
        (out_ref * grad_probe).sum().backward()
        (out_cu * grad_probe).sum().backward()
        assert x_ref.grad is not None and x_cu.grad is not None
        rel_gx = (x_cu.grad - x_ref.grad).norm() / x_ref.grad.norm().clamp(min=1e-6)
        self.assertLess(rel_gx.item(), 2e-2)

        assert peer_ref.query_proj.weight.grad is not None and peer_cuda.query_proj.weight.grad is not None
        rel_gw = (peer_cuda.query_proj.weight.grad - peer_ref.query_proj.weight.grad).norm() / (
            peer_ref.query_proj.weight.grad.norm().clamp(min=1e-6)
        )
        self.assertLess(rel_gw.item(), 2e-2)

        # Small LR (1e-4) stateless RMS-SGD survival across C++ LRU eviction -> int8 RAM -> VRAM reload
        small_peer = PEERMicroExpertFFN(
            dim=64,
            grid_size=16,  # E = 256
            num_heads=4,
            top_k=4,
            cache_slots=32,  # Forces frequent LRU eviction
            peer_lr=1e-4,
            weight_decay=0.0,
            dropout=0.0,
        ).to(self.device)
        small_peer._use_cuda_ext = True
        small_peer.train()

        # Fix one input & target, run updates, then force eviction of everything via disjoint noise
        x_fixed = torch.randn((1, 8, 64), device=self.device, dtype=torch.float32)
        target = torch.randn((1, 8, 64), device=self.device, dtype=torch.float32) * 0.3

        with torch.no_grad():
            pre_loss = F.mse_loss(small_peer(x_fixed, compute_aux=False)[0], target).item()

        for _ in range(12):
            small_peer.zero_grad(set_to_none=True)
            y_p, _, _ = small_peer(x_fixed, compute_aux=False)
            loss = F.mse_loss(y_p, target)
            loss.backward()

        # Touch 128 other random inputs to force LRU eviction of modified slots into int8 RAM
        noise = torch.randn((8, 16, 64), device=self.device, dtype=torch.float32) * 5.0
        for k in range(8):
            small_peer(noise[k : k + 1], compute_aux=False)
        small_peer.flush_dirty_to_ram()

        # Clear GPU cache indices so next call must reload from int8 RAM
        small_peer.expert_to_slot.fill_(-1)
        small_peer.slot_to_expert.fill_(-1)
        small_peer.slot_last_step.fill_(-1)
        small_peer.slot_dirty.zero_()

        with torch.no_grad():
            post_loss = F.mse_loss(small_peer(x_fixed, compute_aux=False)[0], target).item()
        self.assertLess(post_loss, pre_loss, f"Expected lr=1e-4 update to survive eviction: {post_loss} vs {pre_loss}")

    def test_cuda_graphs_logits_parity_moe_and_dense(self) -> None:
        torch.manual_seed(2025)
        for mode in ("moe", "dense"):
            cfg = BrainConfig(
                broca_dim=128,
                column_dim=64,
                broca_layers=2,
                broca_heads=4,
                broca_ff_dim=256,
                broca_use_moe=(mode == "moe"),
                broca_ffn_mode=mode,
                broca_num_experts=4,
                broca_moe_top_k=2,
                broca_moe_expert_dim=128,
                broca_dropout=0.0,
                broca_max_seq_len=128,
                broca_use_cuda_graphs=True,
            )
            model = BrocaByteTransformer(cfg).to(self.device).eval()
            kv_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            z_raw = torch.randn((1, 8, cfg.column_dim), device=self.device, dtype=kv_dtype)
            cos_all, sin_all = model._get_rope(40, device=self.device)

            with torch.no_grad():
                from brain.sensory_wernicke import amp_autocast

                with amp_autocast(self.device):
                    z_mem = model.col_norm(z_raw)
                    z_summary = model._pool_z_summary(z_mem)
                    cross_kvs = [blk.precompute_cross_kv(z_mem) for blk in model.blocks]
                    film_caches = [blk.precompute_film(z_summary) for blk in model.blocks]
                    bos_prefix = model._norm_z_proj(model.bos_z_proj(z_summary)).unsqueeze(1)
                    out_z_bias = 0.50 * model._norm_z_proj(model.out_z_proj(z_summary)).unsqueeze(1)

                run_eager, cur_eager, _, used_eager = model._create_stream_step_runner(
                    z_mem=z_mem,
                    z_summary=z_summary,
                    cross_kvs=cross_kvs,
                    film_caches=film_caches,
                    bos_prefix=bos_prefix,
                    out_z_bias=out_z_bias,
                    cos_all=cos_all,
                    sin_all=sin_all,
                    mem_mask=None,
                    max_bytes=36,
                    device=self.device,
                    kv_dtype=kv_dtype,
                    force_eager=True,
                )
                run_graph, cur_graph, _, used_graph = model._create_stream_step_runner(
                    z_mem=z_mem,
                    z_summary=z_summary,
                    cross_kvs=cross_kvs,
                    film_caches=film_caches,
                    bos_prefix=bos_prefix,
                    out_z_bias=out_z_bias,
                    cos_all=cos_all,
                    sin_all=sin_all,
                    mem_mask=None,
                    max_bytes=36,
                    device=self.device,
                    kv_dtype=kv_dtype,
                    force_eager=False,
                )
                self.assertFalse(used_eager)
                self.assertTrue(used_graph, f"CUDA Graph was not captured for mode={mode}")

                prev_logits = None
                distinct_steps = 0
                for step in range(32):
                    tok_id = 65 + (step % 26) if step > 0 else 257  # BOS then 'A'..'Z'
                    cur_eager[0, 0] = tok_id
                    cur_graph[0, 0] = tok_id
                    le = run_eager(step)
                    lg = run_graph(step)
                    self.assertTrue(bool(torch.isfinite(lg).all().item()))
                    max_diff = (le - lg).abs().max().item()
                    cos_sim = F.cosine_similarity(le.unsqueeze(0), lg.unsqueeze(0)).item()
                    self.assertLess(max_diff, 0.15, f"mode={mode} step={step} max_diff={max_diff}")
                    self.assertGreater(cos_sim, 0.999, f"mode={mode} step={step} cos={cos_sim}")
                    if prev_logits is not None and (lg - prev_logits).abs().max().item() > 1e-3:
                        distinct_steps += 1
                    prev_logits = lg

                self.assertGreaterEqual(distinct_steps, 30, "Logits appear frozen across steps")

                # End-to-end generate_stream check
                out_text = "".join(model.generate_stream(z_raw, max_new_bytes=24))
                self.assertIsInstance(out_text, str)

        # Verify PEER mode runs cleanly via generate_stream (stably eager, no CUDA graph error)
        peer_cfg = BrainConfig(
            broca_dim=64,
            column_dim=64,
            broca_layers=1,
            broca_heads=4,
            broca_ff_dim=128,
            broca_ffn_mode="peer",
            peer_grid_size=16,
            peer_num_heads=4,
            peer_top_k=4,
            peer_cache_slots=64,
            broca_dropout=0.0,
            broca_max_seq_len=64,
            broca_use_cuda_graphs=True,
        )
        peer_model = BrocaByteTransformer(peer_cfg).to(self.device).eval()
        z_p = torch.randn((1, 8, 64), device=self.device, dtype=torch.bfloat16)
        out_p = "".join(peer_model.generate_stream(z_p, max_new_bytes=16))
        self.assertIsInstance(out_p, str)

    def test_generate_batch_cuda_graphs_parity(self) -> None:
        torch.manual_seed(4242)
        for mode in ("moe", "dense"):
            cfg = BrainConfig(
                broca_dim=128,
                column_dim=64,
                broca_layers=2,
                broca_heads=4,
                broca_ff_dim=256,
                broca_use_moe=(mode == "moe"),
                broca_ffn_mode=mode,
                broca_num_experts=4,
                broca_moe_top_k=2,
                broca_moe_expert_dim=128,
                broca_dropout=0.0,
                broca_max_seq_len=128,
                broca_use_cuda_graphs=True,
            )
            model = BrocaByteTransformer(cfg).to(self.device).eval()
            kv_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            b = 3
            z_raw = torch.randn((b, 8, cfg.column_dim), device=self.device, dtype=kv_dtype)
            cos_all, sin_all = model._get_rope(40, device=self.device)

            with torch.no_grad():
                from brain.sensory_wernicke import amp_autocast

                with amp_autocast(self.device):
                    z_mem = model.col_norm(z_raw)
                    z_summary = model._pool_z_summary(z_mem)
                    cross_kvs = [blk.precompute_cross_kv(z_mem) for blk in model.blocks]
                    film_caches = [blk.precompute_film(z_summary) for blk in model.blocks]
                    bos_prefix = model._norm_z_proj(model.bos_z_proj(z_summary)).unsqueeze(1)
                    out_z_bias = 0.50 * model._norm_z_proj(model.out_z_proj(z_summary)).unsqueeze(1)

                run_eager, cur_eager, _, _, _, used_eager = model._create_batch_step_runner(
                    batch_size=b,
                    z_mem=z_mem,
                    z_summary=z_summary,
                    cross_kvs=cross_kvs,
                    film_caches=film_caches,
                    bos_prefix=bos_prefix,
                    out_z_bias=out_z_bias,
                    cos_all=cos_all,
                    sin_all=sin_all,
                    mem_mask=None,
                    cache_len=36,
                    device=self.device,
                    kv_dtype=kv_dtype,
                    force_eager=True,
                )
                run_graph, cur_graph, _, _, _, used_graph = model._create_batch_step_runner(
                    batch_size=b,
                    z_mem=z_mem,
                    z_summary=z_summary,
                    cross_kvs=cross_kvs,
                    film_caches=film_caches,
                    bos_prefix=bos_prefix,
                    out_z_bias=out_z_bias,
                    cos_all=cos_all,
                    sin_all=sin_all,
                    mem_mask=None,
                    cache_len=36,
                    device=self.device,
                    kv_dtype=kv_dtype,
                    force_eager=False,
                )
                self.assertFalse(used_eager)
                self.assertTrue(used_graph, f"Batch CUDA Graph not captured for mode={mode}")

                for step in range(24):
                    tok_col = torch.tensor([[65 + (step + r) % 26] for r in range(b)], device=self.device) if step > 0 else torch.full((b, 1), 257, device=self.device)
                    cur_eager.copy_(tok_col)
                    cur_graph.copy_(tok_col)
                    le = run_eager(step)
                    lg = run_graph(step)
                    self.assertTrue(bool(torch.isfinite(lg).all().item()))
                    self.assertLess((le - lg).abs().max().item(), 0.15)
                    self.assertGreater(F.cosine_similarity(le, lg, dim=-1).min().item(), 0.999)

                # End-to-end generate_batch parity with fixed seed
                torch.manual_seed(777)
                out_e = model.generate_batch(z_raw, temperatures=[0.5, 0.6, 0.7], max_new_bytes=24, force_eager=True)
                self.assertFalse(model._last_batch_used_cuda_graph)
                torch.manual_seed(777)
                out_g = model.generate_batch(z_raw, temperatures=[0.5, 0.6, 0.7], max_new_bytes=24, force_eager=False)
                self.assertTrue(model._last_batch_used_cuda_graph)
                self.assertEqual(len(out_e), 3)
                self.assertEqual(len(out_g), 3)

    def test_peer_cache_epoch_skips_redundant_ensure_in_backward(self) -> None:
        from unittest.mock import patch

        peer = PEERMicroExpertFFN(
            dim=128,
            grid_size=16,
            num_heads=4,
            top_k=4,
            cache_slots=256,
            peer_lr=1e-3,
            dropout=0.0,
        ).to(self.device).train()
        x = torch.randn((2, 8, 128), device=self.device, dtype=torch.bfloat16, requires_grad=True)
        # Warmup once so cache is populated and _cache_epoch stabilizes
        y0, _, _ = peer(x, compute_aux=False)
        y0.sum().backward()
        peer.zero_grad(set_to_none=True)

        orig_ensure = peer._ensure_experts_cached
        ensure_calls = 0

        def _count_ensure(*args, **kwargs):
            nonlocal ensure_calls
            ensure_calls += 1
            return orig_ensure(*args, **kwargs)

        with patch.object(peer, "_ensure_experts_cached", side_effect=_count_ensure):
            y1, _, _ = peer(x, compute_aux=False)
            self.assertEqual(ensure_calls, 1)
            y1.sum().backward()
            # Backward must reuse saved slot_ids without re-calling _ensure_experts_cached!
            self.assertEqual(ensure_calls, 1)

    def test_sparse_sdr_to_columns_parity(self) -> None:
        from brain.sensory_wernicke import WernickeByteCortex, k_wta

        cfg = BrainConfig(
            num_neurons=1024,
            sdr_k=48,
            num_columns=8,
            column_dim=128,
            wernicke_layers=1,
            wernicke_heads=4,
            wernicke_ff_dim=256,
        )
        wernicke = WernickeByteCortex(cfg).to(self.device).eval()
        logits = torch.randn((4, cfg.num_neurons), device=self.device, dtype=torch.float32)
        sdr, top_idx, top_val = k_wta(logits, k=cfg.sdr_k, normalize=True)
        sdr_bf = sdr.to(dtype=torch.bfloat16)
        top_val_bf = top_val.to(dtype=torch.bfloat16)
        with torch.no_grad():
            dense_cols = wernicke.sdr_to_columns(sdr_bf)
            sparse_cols = wernicke.sdr_to_columns(sdr_bf, active_indices=top_idx, active_values=top_val_bf)
        max_diff = (dense_cols.float() - sparse_cols.float()).abs().max().item()
        cos = F.cosine_similarity(dense_cols.float().flatten(1), sparse_cols.float().flatten(1), dim=-1).min().item()
        self.assertLess(max_diff, 1e-2, f"Sparse vs dense max_diff={max_diff}")
        self.assertGreater(cos, 0.9999)

    def test_rtx5080_benchmarks(self) -> None:
        results: list[tuple[str, float, float, str]] = []

        # 1) FP8 Synapse 12288 x 12288 quantize + SHY prune
        w = torch.randn((12288, 12288), device=self.device, dtype=torch.bfloat16) * 0.05
        for _ in range(2):
            q_w = QuantizedSynapseMatrix.from_tensor(w, mode="fp8", stochastic=True, _force_pytorch=False)
            q_w.apply_shy_prune(downscale=0.95, eps=0.01, zero_diag=True, _force_pytorch=False)
            q_r = QuantizedSynapseMatrix.from_tensor(w, mode="fp8", stochastic=True, _force_pytorch=True)
            q_r.apply_shy_prune(downscale=0.95, eps=0.01, zero_diag=True, _force_pytorch=True)
        torch.cuda.synchronize()

        def _time_fp8(force_pt: bool, n_iters: int = 5) -> float:
            t0 = time.perf_counter()
            for _ in range(n_iters):
                qm = QuantizedSynapseMatrix.from_tensor(
                    w, mode="fp8", stochastic=True, _force_pytorch=force_pt
                )
                qm.apply_shy_prune(downscale=0.95, eps=0.01, zero_diag=True, _force_pytorch=force_pt)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0 / n_iters

        fp8_pt_ms = _time_fp8(True)
        fp8_cu_ms = _time_fp8(False)
        results.append(("fp8_synapse 12288x12288 (quant+SHY)", fp8_pt_ms, fp8_cu_ms, "exact parity"))
        del w
        torch.cuda.empty_cache()

        # 2) PEER 1B (10 layers x M=256) and 4B (10 layers x M=512 if RAM allows, else 4 layers scaled)
        avail_gb = _available_ram_gb()
        configs_peer = [("PEER 1B (10L, M=256, D=768)", 10, 256)]
        if avail_gb >= 6.5:
            configs_peer.append(("PEER 4B (10L, M=512, D=768)", 10, 512))
        else:
            configs_peer.append(("PEER 4B-scaled (4L*2.5, M=512, D=768)", 4, 512))

        for label, n_layers, grid_m in configs_peer:
            layers = [
                PEERMicroExpertFFN(
                    dim=768,
                    grid_size=grid_m,
                    num_heads=8,
                    top_k=8,
                    cache_slots=4096,
                    peer_lr=3e-3,
                    weight_decay=1e-4,
                    dropout=0.0,
                ).to(self.device)
                for _ in range(n_layers)
            ]
            scale_factor = 10.0 / n_layers
            x_in = torch.randn((4, 64, 768), device=self.device, dtype=torch.bfloat16)

            def _bench_peer(use_ext: bool, n_steps: int = 3) -> tuple[float, float, float]:
                torch.cuda.reset_peak_memory_stats(self.device)
                for lyr in layers:
                    lyr._use_cuda_ext = use_ext
                    lyr.cache_hits = 0
                    lyr.cache_misses = 0
                    lyr.train()
                # Warmup step (populates initial LRU cache)
                for lyr in layers:
                    y, _, _ = lyr(x_in, compute_aux=True)
                    y.sum().backward()
                    lyr.zero_grad(set_to_none=True)
                for lyr in layers:
                    lyr.cache_hits = 0
                    lyr.cache_misses = 0
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_steps):
                    for lyr in layers:
                        y, _, _ = lyr(x_in, compute_aux=True)
                        y.sum().backward()
                        lyr.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000.0 / n_steps * scale_factor
                hits = sum(lyr.cache_hits for lyr in layers)
                misses = sum(lyr.cache_misses for lyr in layers)
                hit_rate = 100.0 * hits / max(1, hits + misses)
                vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
                return ms, hit_rate, vram_mb

            pt_ms, pt_hr, pt_vram = _bench_peer(False)
            cu_ms, cu_hr, cu_vram = _bench_peer(True)
            results.append((label, pt_ms, cu_ms, f"hit={cu_hr:.1f}% VRAM={cu_vram:.0f}MB"))
            del layers
            torch.cuda.empty_cache()

        # 3) BrocaByteTransformer.generate_stream (128B eager vs CUDA Graph)
        cfg_broca = BrainConfig(
            broca_dim=768,
            column_dim=256,
            broca_layers=10,
            broca_heads=12,
            broca_ff_dim=2304,
            broca_ffn_mode="moe",
            broca_use_moe=True,
            broca_num_experts=8,
            broca_moe_top_k=2,
            broca_moe_expert_dim=1152,
            broca_dropout=0.0,
            broca_max_seq_len=256,
            broca_max_new_bytes=128,
            broca_use_cuda_graphs=True,
        )
        broca = BrocaByteTransformer(cfg_broca).to(self.device).eval()
        z_ctx = torch.randn((1, 16, cfg_broca.column_dim), device=self.device, dtype=torch.bfloat16)

        def _bench_broca(use_graphs: bool) -> tuple[float, float]:
            broca.cfg.broca_use_cuda_graphs = use_graphs
            # Warmup once
            _ = "".join(broca.generate_stream(z_ctx, max_new_bytes=128, repetition_penalty=1.0))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            n_runs = 2
            for _ in range(n_runs):
                out_s = "".join(broca.generate_stream(z_ctx, max_new_bytes=128, repetition_penalty=1.0))
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / n_runs
            return dt * 1000.0, 128.0 / max(1e-6, dt)

        eager_ms, eager_bps = _bench_broca(False)
        graph_ms, graph_bps = _bench_broca(True)
        results.append(
            ("Broca.generate_stream 128B", eager_ms, graph_ms, f"{eager_bps:.1f} -> {graph_bps:.1f} B/s")
        )

        # 4) BrocaByteTransformer.generate_batch (B=3, 128B eager vs CUDA Graph)
        z_b3 = z_ctx.expand(3, -1, -1).contiguous()
        _ = broca.generate_batch(z_b3, temperatures=[0.5, 0.6, 0.7], max_new_bytes=128, repetition_penalty=1.0, force_eager=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = broca.generate_batch(z_b3, temperatures=[0.5, 0.6, 0.7], max_new_bytes=128, repetition_penalty=1.0, force_eager=True)
        torch.cuda.synchronize()
        gb_eager_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        _ = broca.generate_batch(z_b3, temperatures=[0.5, 0.6, 0.7], max_new_bytes=128, repetition_penalty=1.0, force_eager=False)
        torch.cuda.synchronize()
        gb_graph_ms = (time.perf_counter() - t0) * 1000.0
        results.append(
            ("Broca.generate_batch (B=3, 128B)", gb_eager_ms, gb_graph_ms, f"{384.0/(gb_eager_ms/1000):.1f} -> {384.0/(gb_graph_ms/1000):.1f} B/s")
        )
        del broca
        torch.cuda.empty_cache()

        # 5) Wernicke.sdr_to_columns (B=16, N=12288, dense vs sparse)
        from brain.sensory_wernicke import WernickeByteCortex, k_wta

        w_cfg = BrainConfig(wernicke_layers=1)
        wernicke = WernickeByteCortex(w_cfg).to(self.device).eval()
        sdr_b16, idx_b16, val_b16 = k_wta(torch.randn((16, w_cfg.num_neurons), device=self.device), k=w_cfg.sdr_k, normalize=True)
        sdr_b16 = sdr_b16.to(dtype=torch.bfloat16)
        val_b16 = val_b16.to(dtype=torch.bfloat16)
        with torch.no_grad():
            for _ in range(3):
                _ = wernicke.sdr_to_columns(sdr_b16)
                _ = wernicke.sdr_to_columns(sdr_b16, active_indices=idx_b16, active_values=val_b16)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                _ = wernicke.sdr_to_columns(sdr_b16)
            torch.cuda.synchronize()
            w_dense_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
            t0 = time.perf_counter()
            for _ in range(20):
                _ = wernicke.sdr_to_columns(sdr_b16, active_indices=idx_b16, active_values=val_b16)
            torch.cuda.synchronize()
            w_sparse_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
        results.append(("Wernicke.sdr_to_columns (B=16)", w_dense_ms, w_sparse_ms, "sparse k-WTA"))
        del wernicke
        torch.cuda.empty_cache()

        print("\n=== RTX 5080 Before vs After Benchmark ===")
        print(f"{'Workload':<38} | {'PyTorch/Eager (ms)':>18} | {'C++/CUDA/Graph (ms)':>19} | {'Speedup':>8} | Notes")
        print("-" * 105)
        for name, before_ms, after_ms, note in results:
            spd = before_ms / max(1e-6, after_ms)
            print(f"{name:<38} | {before_ms:18.2f} | {after_ms:19.2f} | {spd:7.2f}x | {note}")
            self.assertGreater(before_ms, 0.0)
            self.assertGreater(after_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
