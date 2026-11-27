from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cuda_ops import (
    HAS_CUDA_EXT,
    peer_backward_update_cuda,
    peer_cache_ensure_cpp,
    peer_flush_dirty_cpp,
    peer_forward_cuda,
)
from .sensory_wernicke import RMSNorm

_logger = logging.getLogger(__name__)

_PIN_STEP: int = 1 << 62


def _stochastic_round_bf16(x: torch.Tensor) -> torch.Tensor:
    """
    Unbiased stochastic rounding from float32 to bfloat16 using bit-level dithering.
    Preserves small gradient updates (~1e-4) in 7-bit bfloat16 mantissa in expectation.
    """
    x_f32 = x.detach().float().contiguous().clamp(-3.3e38, 3.3e38)
    bits = x_f32.view(torch.int32)
    rand = torch.randint(0, 65536, x_f32.shape, device=x_f32.device, dtype=torch.int32)
    rounded_bits = (bits + rand) & (-65536)
    return rounded_bits.view(torch.float32).to(dtype=torch.bfloat16)


def _quantize_int8_stochastic(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-row int8 quantization with unbiased stochastic rounding.
    Input:
      w: float32 [K, D]
    Returns:
      q: int8 [K, D]
      scale: float16 [K] such that w ~= q.float() * scale.float().unsqueeze(-1)
    """
    w_f = w.detach().float()
    amax = w_f.abs().amax(dim=-1)
    scale_fp16 = (amax / 127.0).clamp(min=1e-6).to(dtype=torch.float16)
    scale_f = scale_fp16.float().unsqueeze(-1).clamp(min=1e-6)
    scaled = (w_f / scale_f).clamp(-127.0, 127.0)
    floor_val = torch.floor(scaled)
    prob = scaled - floor_val
    q = (floor_val + (torch.rand_like(prob) < prob).to(dtype=torch.float32)).clamp(-127.0, 127.0).to(dtype=torch.int8)
    return q, scale_fp16


class _CachedPEERFunction(torch.autograd.Function):
    """
    Custom autograd function over LRU-cached rank-1 micro-experts:
      y(x) = sum_e w_e * SiLU(x · u_e) * v_e
    - forward() is strictly read-only on cache_u / cache_v (safe under gradient_checkpointing).
    - backward() computes exact grad_x and grad_weights, aggregates sparse gradients g_u / g_v
      on active slots in float32, and performs zero-state RMS-normalized SGD in-place with
      stochastic rounding to bfloat16 when training=True and peer_lr > 0.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x_flat: torch.Tensor,
        weights: torch.Tensor,
        expert_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        peer_module: PEERMicroExpertFFN,
        training: bool,
    ) -> torch.Tensor:
        n_tokens, d = x_flat.shape
        k_tot = expert_ids.shape[1]
        cache_u = peer_module.cache_u
        cache_v = peer_module.cache_v
        cache_dtype = peer_module.cache_dtype

        use_cuda_ext = bool(
            getattr(peer_module, "_use_cuda_ext", True)
            and HAS_CUDA_EXT
            and x_flat.is_cuda
            and cache_u.is_cuda
        )
        if use_cuda_ext:
            with torch.no_grad():
                x_in = x_flat.detach().to(dtype=cache_dtype).contiguous()
                out, pre_act = peer_forward_cuda(
                    x_in,
                    slot_ids,
                    weights.detach(),
                    cache_u,
                    cache_v,
                )
            ctx.save_for_backward(x_flat.detach(), weights.detach(), expert_ids, pre_act)
            ctx.peer = peer_module
            ctx.training = bool(training)
            ctx.used_cuda_ext = True
            return out.to(dtype=x_flat.dtype)

        chunk = min(peer_module.token_chunk, max(16, 32768 // max(1, k_tot)))
        out = torch.empty((n_tokens, d), device=x_flat.device, dtype=cache_dtype)

        with torch.no_grad():
            for s in range(0, n_tokens, chunk):
                e = min(n_tokens, s + chunk)
                c_n = e - s
                slot_c = slot_ids[s:e].reshape(-1)
                u_sel = cache_u.index_select(0, slot_c).view(c_n, k_tot, d)
                v_sel = cache_v.index_select(0, slot_c).view(c_n, k_tot, d)
                x_c = x_flat[s:e].detach().to(dtype=cache_dtype)

                pre = torch.bmm(u_sel, x_c.unsqueeze(-1)).squeeze(-1).float()  # [c_n, K_tot]
                act = F.silu(pre)
                aw = (act * weights[s:e].detach().float()).to(dtype=cache_dtype)
                out[s:e] = torch.bmm(aw.unsqueeze(1), v_sel).squeeze(1)

        ctx.save_for_backward(x_flat.detach(), weights.detach(), expert_ids)
        ctx.peer = peer_module
        ctx.training = bool(training)
        ctx.used_cuda_ext = False
        return out.to(dtype=x_flat.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        peer: PEERMicroExpertFFN = ctx.peer
        if getattr(ctx, "used_cuda_ext", False) and HAS_CUDA_EXT and peer.cache_u.is_cuda:
            x_flat, weights, expert_ids, pre_act = ctx.saved_tensors
            if x_flat.is_cuda:
                slot_ids = peer._ensure_experts_cached(expert_ids, count_stats=False)
                lr = float(peer.peer_lr)
                wd = float(peer.weight_decay)
                do_update = bool(ctx.training and lr > 0.0 and x_flat.shape[0] > 0)
                with torch.no_grad():
                    grad_x, grad_weights = peer_backward_update_cuda(
                        grad_output.detach(),
                        x_flat.to(dtype=peer.cache_dtype),
                        slot_ids,
                        weights,
                        pre_act,
                        peer.cache_u,
                        peer.cache_v,
                        peer.slot_dirty,
                        lr,
                        wd,
                        do_update,
                        peer._next_seed(),
                    )
                return (
                    grad_x.to(dtype=x_flat.dtype),
                    grad_weights.to(dtype=weights.dtype),
                    None,
                    None,
                    None,
                    None,
                )

        x_flat, weights, expert_ids = ctx.saved_tensors[:3]
        n_tokens, d = x_flat.shape
        k_tot = expert_ids.shape[1]
        device = x_flat.device
        cache_dtype = peer.cache_dtype

        # Re-resolve slots (100% cache hit in normal batches; safe reload if forward chunked > C)
        slot_ids = peer._ensure_experts_cached(expert_ids, count_stats=False)
        uniq_slots, inv = torch.unique(slot_ids.reshape(-1), return_inverse=True)

        grad_x = torch.empty((n_tokens, d), device=device, dtype=torch.float32)
        grad_weights = torch.empty((n_tokens, k_tot), device=device, dtype=torch.float32)

        lr = float(peer.peer_lr)
        wd = float(peer.weight_decay)
        do_update = bool(ctx.training and lr > 0.0 and uniq_slots.numel() > 0)
        if do_update:
            g_u = torch.zeros((uniq_slots.shape[0], d), device=device, dtype=torch.float32)
            g_v = torch.zeros((uniq_slots.shape[0], d), device=device, dtype=torch.float32)
        else:
            g_u = None
            g_v = None

        chunk = min(peer.token_chunk, max(16, 16384 // max(1, k_tot)))
        cache_u = peer.cache_u
        cache_v = peer.cache_v
        gy_contig = grad_output.detach().contiguous()

        with torch.no_grad():
            for s in range(0, n_tokens, chunk):
                e = min(n_tokens, s + chunk)
                c_n = e - s
                slot_c = slot_ids[s:e].reshape(-1)
                u_sel = cache_u.index_select(0, slot_c).view(c_n, k_tot, d)
                v_sel = cache_v.index_select(0, slot_c).view(c_n, k_tot, d)

                x_c_f = x_flat[s:e].float()
                gy_c_f = gy_contig[s:e].float()
                x_c_w = x_c_f.to(dtype=cache_dtype)
                gy_c_w = gy_c_f.to(dtype=cache_dtype)

                pre = torch.bmm(u_sel, x_c_w.unsqueeze(-1)).squeeze(-1).float()
                sig = torch.sigmoid(pre)
                act = pre * sig
                silu_grad = sig * (1.0 + pre * (1.0 - sig))

                d_c = torch.bmm(v_sel, gy_c_w.unsqueeze(-1)).squeeze(-1).float()  # gy · v_e
                w_c = weights[s:e].float()

                grad_weights[s:e] = d_c * act
                dp_c = d_c * w_c * silu_grad  # dL/d(pre)
                grad_x[s:e] = torch.bmm(dp_c.to(dtype=cache_dtype).unsqueeze(1), u_sel).squeeze(1).float()

                if do_update and g_u is not None and g_v is not None:
                    inv_c = inv[s * k_tot : e * k_tot]
                    g_u.index_add_(0, inv_c, (dp_c.unsqueeze(-1) * x_c_f.unsqueeze(1)).reshape(-1, d))
                    g_v.index_add_(0, inv_c, ((w_c * act).unsqueeze(-1) * gy_c_f.unsqueeze(1)).reshape(-1, d))

            # Stateless RMS-Normalized SGD in-place update AFTER computing exact grad_x/grad_weights
            if do_update and g_u is not None and g_v is not None:
                rms_u = g_u.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-5)
                rms_v = g_v.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-5)
                step_u = (g_u / rms_u).clamp(-2.0, 2.0)
                step_v = (g_v / rms_v).clamp(-2.0, 2.0)

                u_old = cache_u.index_select(0, uniq_slots).float()
                v_old = cache_v.index_select(0, uniq_slots).float()
                decay = 1.0 - lr * wd
                u_new = u_old * decay - lr * step_u
                v_new = v_old * decay - lr * step_v

                if cache_dtype == torch.bfloat16:
                    u_store = _stochastic_round_bf16(u_new)
                    v_store = _stochastic_round_bf16(v_new)
                else:
                    u_store = u_new.to(dtype=cache_dtype)
                    v_store = v_new.to(dtype=cache_dtype)

                cache_u.index_copy_(0, uniq_slots, u_store)
                cache_v.index_copy_(0, uniq_slots, v_store)
                peer.slot_dirty.index_fill_(0, uniq_slots, True)

        return (
            grad_x.to(dtype=x_flat.dtype),
            grad_weights.to(dtype=weights.dtype),
            None,
            None,
            None,
            None,
        )


class PEERMicroExpertFFN(nn.Module):
    """
    Product-Key Memory Experts (PEER) ultra-sparse rank-1 micro-expert FFN:
      - O(sqrt(E)) product-key multi-head routing in float32 (E = grid_size^2).
      - Cold expert bank stored in Pinned CPU RAM as symmetric per-row int8 (~1 B/param).
      - Fixed-budget GPU VRAM LRU cache in bfloat16 with stochastic-rounding int8 write-back.
      - Zero AdamW optimizer states for the expert bank (bank tensors are NOT nn.Parameters).
      - Stateless in-place RMS-normalized SGD with stochastic bf16 rounding during backward().
    """

    def __init__(
        self,
        dim: int,
        grid_size: int = 256,
        num_heads: int = 8,
        top_k: int = 8,
        cache_slots: int = 4096,
        peer_lr: float = 3e-3,
        weight_decay: float = 1e-4,
        key_dim: int | None = None,
        dropout: float = 0.0,
        token_chunk: int = 512,
        pin_ram: bool = True,
        cache_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.grid_size = max(2, int(grid_size))
        self.num_experts = self.grid_size * self.grid_size
        self.num_heads = max(1, int(num_heads))
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.k_tot = self.num_heads * self.top_k
        self.cache_slots = max(self.k_tot, int(cache_slots))
        if self.cache_slots < self.k_tot:
            raise ValueError(
                f"peer_cache_slots ({self.cache_slots}) must be >= peer_num_heads * peer_top_k ({self.k_tot})"
            )

        self.peer_lr = float(peer_lr)
        self.weight_decay = float(weight_decay)
        self.token_chunk = max(16, int(token_chunk))
        self.pin_ram = bool(pin_ram)
        self._explicit_cache_dtype = cache_dtype

        if key_dim is None:
            raw_kd = max(16, min(64, (self.dim // self.num_heads)))
            sub_dim = max(8, raw_kd // 2)
        else:
            sub_dim = max(8, int(key_dim) // 2)
        self.sub_dim = sub_dim
        self.key_dim = 2 * sub_dim

        # Registered nn.Parameters ONLY for the lightweight Product-Key Router (seen by AdamW)
        self.query_proj = nn.Linear(self.dim, self.num_heads * self.key_dim, bias=False)
        self.q_norm_1 = RMSNorm(self.sub_dim)
        self.q_norm_2 = RMSNorm(self.sub_dim)
        self.keys_1 = nn.Parameter(torch.empty(self.num_heads, self.grid_size, self.sub_dim))
        self.keys_2 = nn.Parameter(torch.empty(self.num_heads, self.grid_size, self.sub_dim))
        self.dropout = nn.Dropout(dropout)

        # Cold int8 expert bank in CPU RAM (plain attributes, NEVER nn.Parameter or buffer)
        self.ram_u_q: torch.Tensor
        self.ram_u_scale: torch.Tensor
        self.ram_v_q: torch.Tensor
        self.ram_v_scale: torch.Tensor
        self._init_ram_bank()

        # Hot GPU/CPU LRU cache (plain attributes, NEVER nn.Parameter or buffer)
        self.cache_dtype: torch.dtype
        self.cache_u: torch.Tensor
        self.cache_v: torch.Tensor
        self.expert_to_slot: torch.Tensor
        self.slot_to_expert: torch.Tensor
        self.slot_last_step: torch.Tensor
        self.slot_dirty: torch.Tensor
        self._init_cache(device=torch.device("cpu"))

        self._step_counter: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self._use_cuda_ext: bool = bool(HAS_CUDA_EXT)
        self._stage_u_q: torch.Tensor | None = None
        self._stage_u_scale: torch.Tensor | None = None
        self._stage_v_q: torch.Tensor | None = None
        self._stage_v_scale: torch.Tensor | None = None

        self.reset_parameters()
        self._cuda_seed_counter: int = int(torch.randint(0, 1 << 62, (1,)).item())

    def _next_seed(self) -> int:
        self._cuda_seed_counter = (self._cuda_seed_counter + 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF
        return self._cuda_seed_counter

    def _ensure_staging_buffers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_slots, d = self.cache_slots, self.dim
        if (
            self._stage_u_q is None
            or self._stage_u_q.shape != (c_slots, d)
            or self._stage_v_q is None
            or self._stage_v_q.shape != (c_slots, d)
        ):
            self._stage_u_q = self._alloc_cpu_pinned((c_slots, d), torch.int8)
            self._stage_u_scale = self._alloc_cpu_pinned((c_slots,), torch.float16)
            self._stage_v_q = self._alloc_cpu_pinned((c_slots, d), torch.int8)
            self._stage_v_scale = self._alloc_cpu_pinned((c_slots,), torch.float16)
        assert self._stage_u_scale is not None and self._stage_v_scale is not None
        return self._stage_u_q, self._stage_u_scale, self._stage_v_q, self._stage_v_scale

    def _alloc_cpu_pinned(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        if self.pin_ram and torch.cuda.is_available():
            try:
                return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            except (RuntimeError, MemoryError):
                pass
        return torch.empty(shape, dtype=dtype, device="cpu")

    def _init_ram_bank(self) -> None:
        e_num, d = self.num_experts, self.dim
        self.ram_u_q = self._alloc_cpu_pinned((e_num, d), torch.int8)
        self.ram_u_scale = self._alloc_cpu_pinned((e_num,), torch.float16)
        self.ram_v_q = self._alloc_cpu_pinned((e_num, d), torch.int8)
        self.ram_v_scale = self._alloc_cpu_pinned((e_num,), torch.float16)

        # Direct in-place int8 initialization avoids transient float32 multi-GB RAM spikes at 1B-4B+ scale
        torch.randint(-127, 128, (e_num, d), dtype=torch.int8, out=self.ram_u_q)
        torch.randint(-127, 128, (e_num, d), dtype=torch.int8, out=self.ram_v_q)
        u_bound = math.sqrt(3.0 / float(self.dim))
        v_bound = 0.5 * math.sqrt(3.0 / float(self.dim))
        self.ram_u_scale.fill_(max(1e-5, u_bound / 127.0))
        self.ram_v_scale.fill_(max(1e-5, v_bound / 127.0))

    def _resolve_cache_dtype(self, device: torch.device) -> torch.dtype:
        if self._explicit_cache_dtype is not None:
            return self._explicit_cache_dtype
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32

    def _init_cache(self, device: torch.device) -> None:
        c, d, e_num = self.cache_slots, self.dim, self.num_experts
        self.cache_dtype = self._resolve_cache_dtype(device)
        self.cache_u = torch.zeros((c, d), device=device, dtype=self.cache_dtype)
        self.cache_v = torch.zeros((c, d), device=device, dtype=self.cache_dtype)
        self.expert_to_slot = torch.full((e_num,), -1, device=device, dtype=torch.int32)
        self.slot_to_expert = torch.full((c,), -1, device=device, dtype=torch.int32)
        self.slot_last_step = torch.full((c,), -1, device=device, dtype=torch.int64)
        self.slot_dirty = torch.zeros((c,), device=device, dtype=torch.bool)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.keys_1, mean=0.0, std=0.5)
        nn.init.normal_(self.keys_2, mean=0.0, std=0.5)

    def _apply(self, fn: Any) -> PEERMicroExpertFFN:
        super()._apply(fn)
        target_device = self.keys_1.device
        if self.cache_u.device != target_device:
            self.flush_dirty_to_ram()
            self._init_cache(target_device)
        return self

    def reset_cache_stats(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0

    def bank_parameters(self) -> int:
        return 2 * self.num_experts * self.dim

    def active_bank_parameters(self) -> int:
        return 2 * self.k_tot * self.dim

    def vram_cache_bytes(self) -> int:
        return (
            self.cache_u.numel() * self.cache_u.element_size()
            + self.cache_v.numel() * self.cache_v.element_size()
            + self.expert_to_slot.numel() * self.expert_to_slot.element_size()
            + self.slot_to_expert.numel() * self.slot_to_expert.element_size()
            + self.slot_last_step.numel() * self.slot_last_step.element_size()
            + self.slot_dirty.numel() * self.slot_dirty.element_size()
        )

    def ram_bank_bytes(self) -> int:
        return (
            self.ram_u_q.numel() * self.ram_u_q.element_size()
            + self.ram_u_scale.numel() * self.ram_u_scale.element_size()
            + self.ram_v_q.numel() * self.ram_v_q.element_size()
            + self.ram_v_scale.numel() * self.ram_v_scale.element_size()
        )

    @torch.no_grad()
    def _write_back_slots_to_ram(self, slots: torch.Tensor, expert_ids: torch.Tensor) -> None:
        if slots.numel() == 0:
            return
        u_vals = self.cache_u.index_select(0, slots)
        v_vals = self.cache_v.index_select(0, slots)
        u_q, u_s = _quantize_int8_stochastic(u_vals)
        v_q, v_s = _quantize_int8_stochastic(v_vals)

        exp_cpu = expert_ids.detach().to(device="cpu", dtype=torch.long)
        self.ram_u_q.index_copy_(0, exp_cpu, u_q.cpu())
        self.ram_u_scale.index_copy_(0, exp_cpu, u_s.cpu())
        self.ram_v_q.index_copy_(0, exp_cpu, v_q.cpu())
        self.ram_v_scale.index_copy_(0, exp_cpu, v_s.cpu())

    @torch.no_grad()
    def flush_dirty_to_ram(self) -> int:
        if self._use_cuda_ext and HAS_CUDA_EXT and self.cache_u.is_cuda:
            stg_uq, stg_us, stg_vq, stg_vs = self._ensure_staging_buffers()
            return int(
                peer_flush_dirty_cpp(
                    self.cache_u,
                    self.cache_v,
                    self.slot_to_expert,
                    self.slot_dirty,
                    self.ram_u_q,
                    self.ram_u_scale,
                    self.ram_v_q,
                    self.ram_v_scale,
                    stg_uq,
                    stg_us,
                    stg_vq,
                    stg_vs,
                    self._next_seed(),
                )
            )
        dirty_slots = (self.slot_dirty & (self.slot_to_expert >= 0)).nonzero(as_tuple=False).squeeze(-1)
        n_flushed = int(dirty_slots.numel())
        if n_flushed == 0:
            return 0
        expert_ids = self.slot_to_expert.index_select(0, dirty_slots).long()
        self._write_back_slots_to_ram(dirty_slots, expert_ids)
        self.slot_dirty.index_fill_(0, dirty_slots, False)
        return n_flushed

    @torch.no_grad()
    def export_bank_state(self) -> dict[str, torch.Tensor]:
        self.flush_dirty_to_ram()
        return {
            "ram_u_q": self.ram_u_q.clone().cpu(),
            "ram_u_scale": self.ram_u_scale.clone().cpu(),
            "ram_v_q": self.ram_v_q.clone().cpu(),
            "ram_v_scale": self.ram_v_scale.clone().cpu(),
        }

    @torch.no_grad()
    def import_bank_state(self, bank_state: dict[str, torch.Tensor]) -> bool:
        keys = ("ram_u_q", "ram_u_scale", "ram_v_q", "ram_v_scale")
        if not all(k in bank_state and isinstance(bank_state[k], torch.Tensor) for k in keys):
            return False
        if (
            bank_state["ram_u_q"].shape != self.ram_u_q.shape
            or bank_state["ram_v_q"].shape != self.ram_v_q.shape
            or bank_state["ram_u_scale"].shape != self.ram_u_scale.shape
            or bank_state["ram_v_scale"].shape != self.ram_v_scale.shape
        ):
            return False
        self.ram_u_q.copy_(bank_state["ram_u_q"].to(dtype=torch.int8, device="cpu"))
        self.ram_u_scale.copy_(bank_state["ram_u_scale"].to(dtype=torch.float16, device="cpu"))
        self.ram_v_q.copy_(bank_state["ram_v_q"].to(dtype=torch.int8, device="cpu"))
        self.ram_v_scale.copy_(bank_state["ram_v_scale"].to(dtype=torch.float16, device="cpu"))

        self.expert_to_slot.fill_(-1)
        self.slot_to_expert.fill_(-1)
        self.slot_last_step.fill_(-1)
        self.slot_dirty.zero_()
        self._step_counter = 0
        return True

    @torch.no_grad()
    def _ensure_experts_cached(
        self,
        expert_ids: torch.Tensor,
        uniq_ids: torch.Tensor | None = None,
        count_stats: bool = True,
    ) -> torch.Tensor:
        if uniq_ids is None:
            uniq_ids = torch.unique(expert_ids)
        u_num = int(uniq_ids.numel())
        if u_num > self.cache_slots:
            raise RuntimeError(
                f"Active unique PEER experts ({u_num}) exceed cache_slots ({self.cache_slots})"
            )

        dev = self.cache_u.device
        if self._use_cuda_ext and HAS_CUDA_EXT and dev.type == "cuda":
            stg_uq, stg_us, stg_vq, stg_vs = self._ensure_staging_buffers()
            slot_ids, n_hits, n_miss, new_step = peer_cache_ensure_cpp(
                expert_ids,
                uniq_ids,
                self.cache_u,
                self.cache_v,
                self.expert_to_slot,
                self.slot_to_expert,
                self.slot_last_step,
                self.slot_dirty,
                self.ram_u_q,
                self.ram_u_scale,
                self.ram_v_q,
                self.ram_v_scale,
                stg_uq,
                stg_us,
                stg_vq,
                stg_vs,
                int(self._step_counter),
                self._next_seed(),
            )
            self._step_counter = int(new_step)
            if count_stats:
                self.cache_hits += int(n_hits)
                self.cache_misses += int(n_miss)
            return slot_ids

        slots = self.expert_to_slot.index_select(0, uniq_ids)
        miss_mask = slots < 0
        hit_mask = ~miss_mask

        n_miss = int(miss_mask.sum().item())
        n_hits = u_num - n_miss
        if count_stats:
            self.cache_hits += n_hits
            self.cache_misses += n_miss

        # Pin currently active hits so LRU eviction never removes an expert needed by this batch
        if n_hits > 0:
            hit_slots = slots[hit_mask].long()
            self.slot_last_step.index_fill_(0, hit_slots, _PIN_STEP)

        if n_miss > 0:
            miss_experts = uniq_ids[miss_mask]  # [n_miss] int64 on dev
            _, victim_slots = torch.topk(self.slot_last_step, k=n_miss, largest=False, sorted=False)

            old_experts = self.slot_to_expert.index_select(0, victim_slots)
            evict_mask = old_experts >= 0
            dirty_mask = evict_mask & self.slot_dirty.index_select(0, victim_slots)

            if bool(dirty_mask.any().item()):
                self._write_back_slots_to_ram(
                    victim_slots[dirty_mask],
                    old_experts[dirty_mask].long(),
                )

            if bool(evict_mask.any().item()):
                self.expert_to_slot.index_fill_(0, old_experts[evict_mask].long(), -1)

            miss_exp_cpu = miss_experts.to(device="cpu", dtype=torch.long)
            u_q_cpu = self.ram_u_q.index_select(0, miss_exp_cpu)
            u_s_cpu = self.ram_u_scale.index_select(0, miss_exp_cpu)
            v_q_cpu = self.ram_v_q.index_select(0, miss_exp_cpu)
            v_s_cpu = self.ram_v_scale.index_select(0, miss_exp_cpu)

            non_blocking = dev.type == "cuda" and self.ram_u_q.is_pinned()
            u_q_dev = u_q_cpu.to(device=dev, non_blocking=non_blocking)
            u_s_dev = u_s_cpu.to(device=dev, non_blocking=non_blocking)
            v_q_dev = v_q_cpu.to(device=dev, non_blocking=non_blocking)
            v_s_dev = v_s_cpu.to(device=dev, non_blocking=non_blocking)

            u_deq = (u_q_dev.float() * u_s_dev.float().unsqueeze(-1)).to(dtype=self.cache_dtype)
            v_deq = (v_q_dev.float() * v_s_dev.float().unsqueeze(-1)).to(dtype=self.cache_dtype)

            self.cache_u.index_copy_(0, victim_slots, u_deq)
            self.cache_v.index_copy_(0, victim_slots, v_deq)
            self.slot_dirty.index_fill_(0, victim_slots, False)
            self.slot_to_expert.index_copy_(0, victim_slots, miss_experts.to(dtype=torch.int32))
            self.expert_to_slot.index_copy_(0, miss_experts, victim_slots.to(dtype=torch.int32))

        self._step_counter += 1
        active_slots = self.expert_to_slot.index_select(0, uniq_ids).long()
        self.slot_last_step.index_fill_(0, active_slots, self._step_counter)

        return self.expert_to_slot.index_select(0, expert_ids.reshape(-1)).view_as(expert_ids).long()

    def _run_with_cache_chunks(
        self,
        x_flat: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        uniq_ids = torch.unique(expert_ids)
        if int(uniq_ids.numel()) <= self.cache_slots:
            slot_ids = self._ensure_experts_cached(expert_ids, uniq_ids=uniq_ids, count_stats=True)
            return _CachedPEERFunction.apply(x_flat, weights, expert_ids, slot_ids, self, self.training)

        n_tokens = x_flat.shape[0]
        if n_tokens <= 1:
            raise RuntimeError(
                f"Single token requires {int(uniq_ids.numel())} unique experts > cache_slots ({self.cache_slots})"
            )
        mid = n_tokens // 2
        left = self._run_with_cache_chunks(x_flat[:mid], expert_ids[:mid], weights[:mid])
        right = self._run_with_cache_chunks(x_flat[mid:], expert_ids[mid:], weights[mid:])
        return torch.cat([left, right], dim=0)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        compute_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        orig_dtype = x.dtype
        d = orig_shape[-1]
        x_flat = x.reshape(-1, d)
        n_tokens = x_flat.shape[0]
        device = x_flat.device

        if n_tokens == 0:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return x, zero, zero

        h_num = self.num_heads
        m_num = self.grid_size
        k = self.top_k
        c = min(m_num, max(32, k * 2))

        # 1) Product-Key Routing O(sqrt(E)) strictly in float32
        with torch.amp.autocast(device_type=device.type, enabled=False):
            x_f32 = x_flat.float()
            q = F.linear(x_f32, self.query_proj.weight.float()).view(n_tokens, h_num, 2, self.sub_dim)
            q1 = F.rms_norm(q[:, :, 0, :], (self.sub_dim,), self.q_norm_1.weight.float(), self.q_norm_1.eps)
            q2 = F.rms_norm(q[:, :, 1, :], (self.sub_dim,), self.q_norm_2.weight.float(), self.q_norm_2.eps)

            scale = 1.0 / math.sqrt(float(self.sub_dim))
            # [H, N, sub_dim] @ [H, sub_dim, M] -> [N, H, M]
            s1 = torch.bmm(q1.transpose(0, 1), self.keys_1.float().transpose(1, 2)).transpose(0, 1) * scale
            s2 = torch.bmm(q2.transpose(0, 1), self.keys_2.float().transpose(1, 2)).transpose(0, 1) * scale

            s1_top, i1_top = torch.topk(s1, k=c, dim=-1, sorted=False)  # [N, H, c]
            s2_top, i2_top = torch.topk(s2, k=c, dim=-1, sorted=False)  # [N, H, c]

            cand = (s1_top.unsqueeze(-1) + s2_top.unsqueeze(-2)).reshape(n_tokens, h_num, c * c)
            top_scores, top_pos = torch.topk(cand, k=k, dim=-1, sorted=False)  # [N, H, K]

            row_pos = torch.div(top_pos, c, rounding_mode="floor")
            col_pos = top_pos - row_pos * c
            row_idx = i1_top.gather(-1, row_pos)  # [N, H, K]
            col_idx = i2_top.gather(-1, col_pos)  # [N, H, K]

            expert_ids = (row_idx * m_num + col_idx).reshape(n_tokens, h_num * k)
            weights = F.softmax(top_scores, dim=-1, dtype=torch.float32).reshape(n_tokens, h_num * k)

            if compute_aux:
                p1 = F.softmax(s1, dim=-1, dtype=torch.float32)  # [N, H, M]
                p2 = F.softmax(s2, dim=-1, dtype=torch.float32)  # [N, H, M]
                row_flat = row_idx.transpose(0, 1).reshape(h_num, -1)  # [H, N*K]
                col_flat = col_idx.transpose(0, 1).reshape(h_num, -1)  # [H, N*K]

                if token_mask is not None:
                    valid_tok = token_mask.reshape(-1, 1, 1).to(device=device, dtype=torch.float32)
                    valid_denom = valid_tok.sum().clamp(min=1.0)
                    p1_mean = (p1 * valid_tok).sum(dim=0) / valid_denom
                    p2_mean = (p2 * valid_tok).sum(dim=0) / valid_denom

                    slot_valid = valid_tok.expand(-1, h_num, k).transpose(0, 1).reshape(h_num, -1)
                    c1 = torch.zeros((h_num, m_num), device=device, dtype=torch.float32)
                    c2 = torch.zeros((h_num, m_num), device=device, dtype=torch.float32)
                    c1.scatter_add_(1, row_flat, slot_valid)
                    c2.scatter_add_(1, col_flat, slot_valid)
                    f1 = c1 / (valid_denom * float(k))
                    f2 = c2 / (valid_denom * float(k))

                    tok_w = valid_tok.squeeze(-1)  # [N, 1]
                    z1 = (torch.logsumexp(s1, dim=-1).pow(2) * tok_w).sum() / (valid_denom * float(h_num))
                    z2 = (torch.logsumexp(s2, dim=-1).pow(2) * tok_w).sum() / (valid_denom * float(h_num))
                else:
                    p1_mean = p1.mean(dim=0)  # [H, M]
                    p2_mean = p2.mean(dim=0)  # [H, M]
                    ones_k = torch.ones((h_num, n_tokens * k), device=device, dtype=torch.float32)
                    c1 = torch.zeros((h_num, m_num), device=device, dtype=torch.float32)
                    c2 = torch.zeros((h_num, m_num), device=device, dtype=torch.float32)
                    c1.scatter_add_(1, row_flat, ones_k)
                    c2.scatter_add_(1, col_flat, ones_k)
                    f1 = c1 / float(n_tokens * k)
                    f2 = c2 / float(n_tokens * k)

                    z1 = torch.logsumexp(s1, dim=-1).pow(2).mean()
                    z2 = torch.logsumexp(s2, dim=-1).pow(2).mean()

                aux_1 = float(m_num) * (f1.detach() * p1_mean).sum(dim=-1).mean()
                aux_2 = float(m_num) * (f2.detach() * p2_mean).sum(dim=-1).mean()
                aux_loss = 0.5 * (aux_1 + aux_2)
                z_loss = 0.5 * (z1 + z2)
            else:
                aux_loss = torch.zeros((), device=device, dtype=torch.float32)
                z_loss = torch.zeros((), device=device, dtype=torch.float32)

        # 2) Execute over LRU-cached rank-1 micro-experts + stateless in-place update in backward
        out_flat = self._run_with_cache_chunks(x_flat, expert_ids, weights)
        out_flat = self.dropout(out_flat)
        out = out_flat.view(orig_shape)
        if out.dtype != orig_dtype:
            out = out.to(dtype=orig_dtype)
        return out, aux_loss, z_loss
