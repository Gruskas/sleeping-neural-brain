from __future__ import annotations

import codecs
import logging
import math
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

from .config import BrainConfig
from .sensory_wernicke import (
    BOS_BYTE,
    BYTE_VOCAB_SIZE,
    EOS_BYTE,
    MASK_BYTE,
    PAD_BYTE,
    RMSNorm,
    SwiGLU,
    amp_autocast,
    apply_rope,
    encode_utf8_bytes,
    precompute_rope,
)

_precompute_rope = precompute_rope
_apply_rope = apply_rope


class _ContiguousGrad(torch.autograd.Function):
    """Ensure upstream grad_output is contiguous before entering fused F.grouped_mm backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.contiguous()


def _grouped_mm_bf16(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    out = F.grouped_mm(mat_a.contiguous(), mat_b.contiguous(), offs=offs)
    return _ContiguousGrad.apply(out)


class SparseMoEFFN(nn.Module):
    """
    Sparse Mixture-of-Experts SwiGLU FFN (8 experts, top-2 router):
      - Router (D -> E), softmax, top-k renormalization (sum=1.0), Switch Transformer load-balancing
        aux_loss, and ST-MoE router z_loss execute strictly in float32 for numerical stability.
      - Experts are E independent SwiGLU blocks stored as stacked 3D weight tensors
        w_gate_up [E, D, 2*H] and w_down [E, H, D] running in native bfloat16 on CUDA.
      - Zero CPU-GPU synchronization dispatch:
          * Small-N inference fast path (T=1 in generate_stream / generate_batch when N*K < E):
            gathers ONLY the selected active expert slices and executes batched bmm with zero sorting.
          * Grouped GPU path (CUDA bfloat16): sorts token assignments on GPU via argsort, builds
            exclusive/inclusive expert offsets via scatter_add_ + cumsum, runs fused F.grouped_mm
            strictly on assigned tokens, and scatters router-weighted outputs via index_add_.
    """

    def __init__(
        self,
        dim: int,
        expert_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.expert_dim = int(expert_dim)
        self.num_experts = max(1, int(num_experts))
        self.top_k = max(1, min(int(top_k), self.num_experts))

        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        self.w_gate_up = nn.Parameter(torch.empty(self.num_experts, self.dim, self.expert_dim * 2))
        self.w_down = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.dim))
        self.dropout = nn.Dropout(dropout)

        # Latest detached routing load stats (GPU tensors, zero CPU-GPU sync)
        self.last_expert_fractions: torch.Tensor | None = None
        self.last_expert_counts: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        bound_in = 1.0 / math.sqrt(self.dim)
        bound_hid = 1.0 / math.sqrt(self.expert_dim)
        nn.init.uniform_(self.w_gate_up, -bound_in, bound_in)
        nn.init.uniform_(self.w_down, -bound_hid, bound_hid)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        compute_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        orig_dtype = x.dtype
        d = orig_shape[-1]
        x_flat = x.reshape(-1, d)  # [N, D]
        n_tokens = x_flat.shape[0]
        device = x_flat.device
        e_num = self.num_experts
        k = self.top_k

        if n_tokens == 0:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return x, zero, zero

        # 1) Router strictly in float32 (even under bfloat16 autocast)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            router_logits = F.linear(x_flat.float(), self.router.weight.float())  # [N, E]
            router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # [N, E]
            topk_probs, topk_idx = torch.topk(router_probs, k=k, dim=-1, sorted=False)  # [N, K]
            topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)  # [N, K], sum=1.0

            if compute_aux:
                flat_idx = topk_idx.reshape(-1)  # [N * K]
                if token_mask is not None:
                    valid_tok = token_mask.reshape(-1).to(device=device, dtype=torch.float32)  # [N]
                    valid_denom = valid_tok.sum().clamp(min=1.0)
                    p_i = (router_probs * valid_tok.unsqueeze(-1)).sum(dim=0) / valid_denom
                    slot_valid = valid_tok.unsqueeze(1).expand(-1, k).reshape(-1)
                    counts_f = torch.zeros((e_num,), device=device, dtype=torch.float32)
                    counts_f.scatter_add_(0, flat_idx, slot_valid)
                    f_i = counts_f / (valid_denom * float(k))
                    z_loss = (torch.logsumexp(router_logits, dim=-1).pow(2) * valid_tok).sum() / valid_denom
                else:
                    p_i = router_probs.mean(dim=0)
                    counts_f = torch.zeros((e_num,), device=device, dtype=torch.float32)
                    counts_f.scatter_add_(
                        0,
                        flat_idx,
                        torch.ones((flat_idx.shape[0],), device=device, dtype=torch.float32),
                    )
                    f_i = counts_f / float(n_tokens * k)
                    z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

                # Standard Switch Transformer load-balancing loss: E * sum_i(f_i * P_i)
                aux_loss = float(e_num) * torch.sum(f_i.detach() * p_i)
                self.last_expert_fractions = f_i.detach()
                self.last_expert_counts = counts_f.detach()
            else:
                aux_loss = torch.zeros((), device=device, dtype=torch.float32)
                z_loss = torch.zeros((), device=device, dtype=torch.float32)

        # 2) Expert execution in native bfloat16 on CUDA (or input dtype on CPU)
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        work_dtype = torch.bfloat16 if use_bf16 else (orig_dtype if orig_dtype.is_floating_point else torch.float32)
        x_work = x_flat if x_flat.dtype == work_dtype else x_flat.to(dtype=work_dtype)
        flat_experts = topk_idx.reshape(-1)  # [M], M = N * K

        # Fast path A: single-step / small-batch inference (N*K < E) -> zero sorting, touch only active experts
        if (not self.training) and (n_tokens * k < e_num):
            w_gu_sel = self.w_gate_up.index_select(0, flat_experts)
            if w_gu_sel.dtype != work_dtype:
                w_gu_sel = w_gu_sel.to(dtype=work_dtype)
            w_dn_sel = self.w_down.index_select(0, flat_experts)
            if w_dn_sel.dtype != work_dtype:
                w_dn_sel = w_dn_sel.to(dtype=work_dtype)
            x_exp = x_work.unsqueeze(1).expand(n_tokens, k, d).reshape(n_tokens * k, 1, d)
            gu = torch.bmm(x_exp, w_gu_sel)  # [M, 1, 2*H]
            gate, up = gu.chunk(2, dim=-1)
            hid = F.silu(gate) * up
            y_exp = torch.bmm(hid, w_dn_sel).view(n_tokens, k, d)
            out_flat = (y_exp * topk_weights.to(dtype=work_dtype).unsqueeze(-1)).sum(dim=1)

        # Fast path B: CUDA bfloat16 fused grouped GEMM (zero CPU-GPU sync, exact token grouping)
        elif (
            use_bf16
            and hasattr(F, "grouped_mm")
            and (self.dim % 8 == 0)
            and (self.expert_dim % 8 == 0)
        ):
            order = torch.argsort(flat_experts, stable=False)
            sorted_experts = flat_experts.index_select(0, order)
            token_indices = torch.div(order, k, rounding_mode="floor")
            sorted_x = x_work.index_select(0, token_indices).contiguous()

            counts_i32 = torch.zeros((e_num,), device=device, dtype=torch.int32)
            counts_i32.scatter_add_(
                0,
                sorted_experts,
                torch.ones((sorted_experts.shape[0],), device=device, dtype=torch.int32),
            )
            offs = counts_i32.cumsum(dim=0, dtype=torch.int32)

            w_gu = self.w_gate_up if self.w_gate_up.dtype == work_dtype else self.w_gate_up.to(dtype=work_dtype)
            w_dn = self.w_down if self.w_down.dtype == work_dtype else self.w_down.to(dtype=work_dtype)

            gu = _grouped_mm_bf16(sorted_x, w_gu, offs)
            gate, up = gu.chunk(2, dim=-1)
            hid = self.dropout(F.silu(gate) * up)
            sorted_out = _grouped_mm_bf16(hid, w_dn, offs)

            sorted_w = topk_weights.reshape(-1).index_select(0, order).to(dtype=work_dtype).unsqueeze(-1)
            out_flat = torch.zeros((n_tokens, d), device=device, dtype=work_dtype)
            out_flat.index_add_(0, token_indices, sorted_out * sorted_w)

        # Fallback path C: CPU / non-bf16 / non-8-aligned dims (grouped by expert, no token loops)
        else:
            out_flat = torch.zeros((n_tokens, d), device=device, dtype=work_dtype)
            flat_tok = torch.arange(n_tokens, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
            flat_w = topk_weights.reshape(-1).to(dtype=work_dtype)
            w_gu = self.w_gate_up if self.w_gate_up.dtype == work_dtype else self.w_gate_up.to(dtype=work_dtype)
            w_dn = self.w_down if self.w_down.dtype == work_dtype else self.w_down.to(dtype=work_dtype)
            for e in range(e_num):
                sel = (flat_experts == e).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                tok_e = flat_tok.index_select(0, sel)
                xe = x_work.index_select(0, tok_e)
                gu = xe @ w_gu[e]
                gate, up = gu.chunk(2, dim=-1)
                hid = self.dropout(F.silu(gate) * up)
                ye = hid @ w_dn[e]
                out_flat.index_add_(0, tok_e, ye * flat_w.index_select(0, sel).unsqueeze(-1))

        out = out_flat.view(orig_shape)
        if out.dtype != orig_dtype:
            out = out.to(dtype=orig_dtype)
        return out, aux_loss, z_loss


class _BrocaDecoderBlock(nn.Module):
    """
    Parallel Byte-Patching Transformer Decoder Block:
      - RMSNorm + Causal Self-Attention (RoPE + F.scaled_dot_product_attention / FlashAttention)
      - RMSNorm + Cross-Attention EXCLUSIVELY over the 16 cortical columns Z* in R^{B x 16 x D}
      - RMSNorm + SparseMoEFFN (8 experts, top-2) or dense SwiGLU FFN
      - Dropout = 0.2 anti-memorization regularization
    """

    def __init__(
        self,
        dim: int,
        col_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.2,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        moe_expert_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout_p = dropout
        self.use_moe = bool(use_moe)

        # 1) Causal Self-Attention
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_self = nn.Linear(dim, dim, bias=False)
        self.drop1 = nn.Dropout(dropout)

        # 2) Cross-Attention exclusively on 16 cortical columns Z*
        self.norm2 = RMSNorm(dim)
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_kv = nn.Linear(col_dim, dim * 2, bias=False)
        self.out_cross = nn.Linear(dim, dim, bias=False)
        self.drop2 = nn.Dropout(dropout)

        # 2b) FiLM conditioning from pooled cortical columns Z* (prevents Cross-Attention collapse)
        self.film_proj = nn.Linear(col_dim, dim * 2, bias=True)
        nn.init.zeros_(self.film_proj.bias)

        # 3) Sparse MoE FFN (or dense SwiGLU fallback when broca_use_moe=False)
        self.norm3 = RMSNorm(dim)
        if self.use_moe:
            self.ff: nn.Module = SparseMoEFFN(
                dim=dim,
                expert_dim=moe_expert_dim,
                num_experts=num_experts,
                top_k=moe_top_k,
                dropout=dropout,
            )
        else:
            self.ff = SwiGLU(dim, ff_dim, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

    def precompute_cross_kv(self, memory_columns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, m, _ = memory_columns.shape
        k, v = self.cross_kv(memory_columns).chunk(2, dim=-1)
        k = k.view(b, m, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, m, self.n_heads, self.head_dim).transpose(1, 2)
        return k, v

    def precompute_film(self, z_summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        film_g, film_b = self.film_proj(z_summary).chunk(2, dim=-1)
        scale = 1.0 + 0.50 * torch.tanh(film_g).unsqueeze(1)
        shift = 0.50 * torch.tanh(film_b).unsqueeze(1)
        return scale, shift

    def forward(
        self,
        x: torch.Tensor,
        memory_columns: torch.Tensor,
        z_summary: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        self_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cross_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        film_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_pos: int | None = None,
        memory_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        compute_aux: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        b, t, d = x.shape

        # 1) Causal self-attention with RoPE + SDPA (supports zero-alloc static KV buffer when cache_pos is set)
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        if self_kv_cache is not None and cache_pos is not None:
            k_buf, v_buf = self_kv_cache
            k_buf[:, :, cache_pos : cache_pos + t, :] = k
            v_buf[:, :, cache_pos : cache_pos + t, :] = v
            k_view = k_buf[:, :, : cache_pos + t, :]
            v_view = v_buf[:, :, : cache_pos + t, :]
            new_self_kv = (k_buf, v_buf)
            is_causal = False
        elif self_kv_cache is not None:
            prev_k, prev_v = self_kv_cache
            k_view = torch.cat([prev_k, k], dim=2)
            v_view = torch.cat([prev_v, v], dim=2)
            new_self_kv = (k_view, v_view)
            is_causal = False
        else:
            k_view, v_view = k, v
            new_self_kv = (k, v)
            is_causal = t > 1

        attn_self = F.scaled_dot_product_attention(
            q,
            k_view,
            v_view,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_self = attn_self.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop1(self.out_self(attn_self))

        # 2) Cross-Attention over variable-length cortical context Z_ctx [B, M_total, D]
        h2 = self.norm2(x)
        cq = self.cross_q(h2).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        if cross_kv_cache is not None:
            ck, cv = cross_kv_cache
        else:
            ck, cv = self.precompute_cross_kv(memory_columns)

        cross_attn_mask = None
        if memory_mask is not None:
            safe_m = memory_mask.to(device=cq.device, dtype=torch.bool)
            if safe_m.shape[1] != ck.shape[2]:
                safe_m = safe_m[:, : ck.shape[2]]
            cross_attn_mask = safe_m[:, None, None, :]

        attn_cross = F.scaled_dot_product_attention(
            cq,
            ck,
            cv,
            attn_mask=cross_attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attn_cross = attn_cross.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop2(self.out_cross(attn_cross))

        # 2b) AdaLN-FiLM modulation from Z* summary on normalized FFN input
        # (prevents exponential 1.5^10 residual stream explosion across 10 layers!)
        if film_cache is not None:
            scale, shift = film_cache
        else:
            scale, shift = self.precompute_film(z_summary)

        # 3) Sparse MoE FFN (or dense SwiGLU)
        h3 = self.norm3(x) * scale + shift
        if self.use_moe:
            ff_out, aux_loss, z_loss = self.ff(h3, token_mask=token_mask, compute_aux=compute_aux)
        else:
            ff_out = self.ff(h3)
            aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
            z_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        x = x + self.drop3(ff_out)
        return x, new_self_kv, aux_loss, z_loss


class BrocaByteTransformer(nn.Module):
    """
    Broca's Area — Parallel UTF-8 Byte-Patching Transformer Decoder:
      - Zero tokens, zero word vocabulary, zero access to user text or Hippocampus strings.
      - Conditioned EXCLUSIVELY via Cross-Attention on 16 cortical columns Z* in R^{B x 16 x D}.
      - Uses RMSNorm, RoPE, SwiGLU, FlashAttention/SDPA, causal local byte-patch Conv1D context,
        dropout=0.2, Gaussian noise + column dropout on Z*, label_smoothing=0.1.
      - Inference strictly uses stochastic sampling (temperature=0.75..0.9, top_p=0.9,
        repetition_penalty=1.15; NEVER pure argmax) and true incremental UTF-8 streaming.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.broca_dim
        self.col_dim = cfg.column_dim
        self.n_heads = cfg.broca_heads
        self.head_dim = cfg.broca_dim // cfg.broca_heads
        self.max_seq_len = cfg.broca_max_seq_len
        self.dropout_p = cfg.broca_dropout

        self.byte_emb = nn.Embedding(BYTE_VOCAB_SIZE, self.dim, padding_idx=PAD_BYTE)
        # Causal local byte-patch Conv1D (kernel=4, left-padded) for morphological UTF-8 smoothing
        self.local_patch_conv = nn.Conv1d(self.dim, self.dim, kernel_size=4, groups=self.dim)
        self.local_patch_gate = nn.Parameter(torch.tensor(0.25))
        self.emb_drop = nn.Dropout(self.dropout_p)

        self.col_norm = RMSNorm(self.col_dim)
        # Column-aware Z* summary (preserves distinct identity of all 16 columns without averaging them out)
        self.z_col_compress = nn.Linear(self.col_dim, 32, bias=False)
        self.z_summary_proj = nn.Linear(cfg.num_columns * 32, self.col_dim, bias=False)
        self.z_summary_norm = RMSNorm(self.col_dim)
        # Direct cortical prefix projection injected at position 0 (<BOS>) & final readout bridge
        self.bos_z_proj = nn.Linear(self.col_dim, self.dim, bias=False)
        self.out_z_proj = nn.Linear(self.col_dim, self.dim, bias=False)
        self.use_moe = bool(getattr(cfg, "broca_use_moe", False))
        self.num_experts = int(getattr(cfg, "broca_num_experts", 8))
        self.moe_top_k = int(getattr(cfg, "broca_moe_top_k", 2))
        self.moe_expert_dim = int(getattr(cfg, "broca_moe_expert_dim", max(64, cfg.broca_ff_dim // 2)))
        self.blocks = nn.ModuleList(
            [
                _BrocaDecoderBlock(
                    dim=self.dim,
                    col_dim=self.col_dim,
                    n_heads=self.n_heads,
                    ff_dim=cfg.broca_ff_dim,
                    dropout=self.dropout_p,
                    use_moe=self.use_moe,
                    num_experts=self.num_experts,
                    moe_top_k=self.moe_top_k,
                    moe_expert_dim=self.moe_expert_dim,
                )
                for _ in range(cfg.broca_layers)
            ]
        )
        self.final_norm = RMSNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, BYTE_VOCAB_SIZE, bias=False)

        # Precompute & cache RoPE cos/sin buffers once (zero per-step trig allocations)
        rope_cos, rope_sin = precompute_rope(self.max_seq_len + 8, self.head_dim, device=torch.device("cpu"))
        self.register_buffer("_rope_cos", rope_cos, persistent=False)
        self.register_buffer("_rope_sin", rope_sin, persistent=False)

        # Optional activation checkpointing for long sequences on 16 GB VRAM
        self.gradient_checkpointing: bool = bool(getattr(cfg, "broca_gradient_checkpointing", False))

        # Training statistics & MoE diagnostics (zero-sync GPU tensors)
        self.trained_steps: int = 0
        self.last_loss: float = 5.5
        self.fluency_score: float = 0.0
        self.last_aux_loss: torch.Tensor | None = None
        self.last_z_loss: torch.Tensor | None = None
        self.last_import_report: dict[str, list[str]] | None = None

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def _get_rope(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= self._rope_cos.shape[0]:
            return self._rope_cos[:seq_len], self._rope_sin[:seq_len]
        return precompute_rope(seq_len, self.head_dim, device=device)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def active_parameters(self) -> int:
        if not self.use_moe:
            return self.total_parameters()
        total = self.total_parameters()
        inactive_frac = float(max(0, self.num_experts - self.moe_top_k)) / float(max(1, self.num_experts))
        expert_params = sum(
            blk.ff.w_gate_up.numel() + blk.ff.w_down.numel()
            for blk in self.blocks
            if isinstance(blk.ff, SparseMoEFFN)
        )
        return int(round(total - expert_params * inactive_frac))

    def get_moe_expert_utilization(self) -> torch.Tensor | None:
        """Return [num_layers, num_experts] tensor of latest per-layer expert load fractions."""
        if not self.use_moe:
            return None
        fracs = [
            blk.ff.last_expert_fractions
            for blk in self.blocks
            if isinstance(blk.ff, SparseMoEFFN) and blk.ff.last_expert_fractions is not None
        ]
        if not fracs:
            return None
        return torch.stack(fracs, dim=0)

    def _update_fluency(self, loss_val: float) -> None:
        self.last_loss = float(loss_val)
        # Smooth mapping: loss ~5.5 -> 0%, loss ~0.9 -> ~96%
        score = max(0.0, min(100.0, (5.2 - loss_val) / 4.3 * 100.0))
        self.fluency_score = round(score, 1)

    def _regularize_memory_columns(self, memory_columns: torch.Tensor, clean_z: bool = False) -> torch.Tensor:
        z = self.col_norm(memory_columns)
        if self.training and not clean_z:
            if self.cfg.broca_z_noise_std > 0.0:
                z = z + torch.randn_like(z) * float(self.cfg.broca_z_noise_std)
            if self.cfg.broca_z_dropout > 0.0:
                z = F.dropout(z, p=float(self.cfg.broca_z_dropout), training=True)
        return z

    def _pool_z_summary(self, z_mem: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, m, _ = z_mem.shape
        n_prim = self.cfg.num_columns
        if m >= n_prim:
            z_primary = z_mem[:, :n_prim, :]
        else:
            z_primary = F.pad(z_mem, (0, 0, 0, n_prim - m), value=0.0)
        comp = torch.tanh(self.z_col_compress(z_primary)).reshape(b, -1)
        primary_sum = self.z_summary_norm(self.z_summary_proj(comp))
        if m > n_prim:
            z_extra = z_mem[:, n_prim:, :]
            if memory_mask is not None:
                ex_m = memory_mask[:, n_prim:].to(device=z_extra.device, dtype=z_extra.dtype).unsqueeze(-1)
                extra_mean = (z_extra * ex_m).sum(dim=1) / ex_m.sum(dim=1).clamp(min=1.0)
            else:
                extra_mean = z_extra.mean(dim=1)
            return self.z_summary_norm(0.85 * primary_sum + 0.15 * extra_mean)
        return primary_sum

    def _norm_z_proj(self, t: torch.Tensor) -> torch.Tensor:
        """Keep bos_z_proj / out_z_proj at bounded unit-RMS scale matching byte_emb & final_norm."""
        if hasattr(F, "rms_norm"):
            return F.rms_norm(t, (self.dim,), eps=1e-6)
        return t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    @torch.no_grad()
    def maybe_repair_collapsed_conditioning(self) -> bool:
        """
        Detect & repair two pathologies that can occur after heavy raw-text Parquet pretraining:
          1. Rank-1 conditioning collapse in `z_summary_proj` (sv[0] >> sv[1]).
          2. Exploded MoE / attention / RMSNorm weight scales that blow the residual stream up to
             10,000–100,000 in bfloat16 and drown out Z_ctx cross-attention.
        Rescaling is direction-preserving so learned Polish syntax stays intact!
        """
        repaired = False
        w = self.z_summary_proj.weight.detach().float()
        if w.ndim == 2 and min(w.shape) >= 4:
            try:
                svs = torch.linalg.svdvals(w)
                sv0 = float(svs[0].item())
                sv1 = float(svs[1].item()) if svs.numel() > 1 else 1.0
                if sv0 > 12.0 and (sv0 / max(sv1, 1e-6)) > 1.65:
                    nn.init.xavier_uniform_(self.z_col_compress.weight)
                    nn.init.xavier_uniform_(self.z_summary_proj.weight)
                    nn.init.xavier_uniform_(self.bos_z_proj.weight)
                    nn.init.xavier_uniform_(self.out_z_proj.weight)
                    repaired = True
            except Exception:
                pass

        # Direction-preserving spectral rescaling of exploded block weights
        for blk in self.blocks:
            for nm in (blk.norm1, blk.norm2, blk.norm3):
                if float(nm.weight.detach().float().abs().max().item()) > 2.5:
                    nm.weight.clamp_(-2.5, 2.5)
                    repaired = True
            if hasattr(blk.ff, "w_gate_up") and isinstance(blk.ff.w_gate_up, torch.Tensor):
                e_n = blk.ff.w_gate_up.shape[0]
                target_gu = 18.0 * math.sqrt(e_n)
                cur_gu = float(blk.ff.w_gate_up.detach().float().norm().item())
                if cur_gu > target_gu * 1.35:
                    blk.ff.w_gate_up.mul_(target_gu / cur_gu)
                    repaired = True
                target_dn = 14.0 * math.sqrt(e_n)
                cur_dn = float(blk.ff.w_down.detach().float().norm().item())
                if cur_dn > target_dn * 1.35:
                    blk.ff.w_down.mul_(target_dn / cur_dn)
                    repaired = True
            for lin in (blk.qkv, blk.out_self, blk.cross_q, blk.cross_kv, blk.out_cross):
                target_w = 26.0
                cur_w = float(lin.weight.detach().float().norm().item())
                if cur_w > target_w * 1.35:
                    lin.weight.mul_(target_w / cur_w)
                    repaired = True
        return repaired

    def conditioning_decorr_loss(
        self,
        memory_columns: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prevent Broca's pooled z_summary and bos_z prefix from collapsing across distinct questions."""
        b = memory_columns.shape[0]
        if b < 2:
            return torch.zeros((), device=memory_columns.device, dtype=torch.float32)
        z_mem = self.col_norm(memory_columns)
        z_sum = self._pool_z_summary(z_mem, memory_mask=memory_mask)
        bos_z = self.bos_z_proj(z_sum)
        eye = torch.eye(b, device=memory_columns.device, dtype=torch.bool)
        z_flat = F.normalize(z_sum.float(), dim=-1, eps=1e-6)
        b_flat = F.normalize(bos_z.float(), dim=-1, eps=1e-6)
        sim_z = (z_flat @ z_flat.T).masked_fill(eye, 0.0)
        sim_b = (b_flat @ b_flat.T).masked_fill(eye, 0.0)
        pen_z = (sim_z - 0.25).clamp(min=0.0).pow(2).mean()
        pen_b = (sim_b - 0.25).clamp(min=0.0).pow(2).mean()
        return 0.50 * (pen_z + pen_b)

    @staticmethod
    def _block_forward_no_cache(
        blk: _BrocaDecoderBlock,
        x: torch.Tensor,
        z_mem: torch.Tensor,
        z_summary: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        memory_mask: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        compute_aux: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _, aux_loss, z_loss = blk(
            x,
            z_mem,
            z_summary=z_summary,
            rope_cos=cos,
            rope_sin=sin,
            memory_mask=memory_mask,
            token_mask=token_mask,
            compute_aux=compute_aux,
        )
        return out, aux_loss, z_loss

    def forward(
        self,
        byte_ids: torch.Tensor,
        memory_columns: torch.Tensor,
        clean_z: bool = False,
        memory_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parallel teacher-forced forward pass for training.
        Args:
          byte_ids: [B, T] int64 target-shifted bytes starting with <BOS>
          memory_columns: [B, M_total, D] composite cortical context Z_ctx
          memory_mask: optional [B, M_total] bool validity mask (True = valid slot)
          token_mask: optional [B, T] bool validity mask for MoE load balancing (excludes PAD)
          return_aux: if True, returns (logits, total_aux_loss, total_z_loss)
        Returns:
          logits: [B, T, 260] (or tuple with accumulated MoE aux_loss and z_loss across all layers)
        """
        b, t = byte_ids.shape
        device = byte_ids.device

        z_mem = self._regularize_memory_columns(memory_columns, clean_z=clean_z)
        z_summary = self._pool_z_summary(z_mem, memory_mask=memory_mask)  # [B, col_dim]

        x = self.byte_emb(byte_ids)  # [B, T, D]
        bos_prefix = self._norm_z_proj(self.bos_z_proj(z_summary)).unsqueeze(1)  # [B, 1, D]
        x = torch.cat([x[:, :1, :] + bos_prefix, x[:, 1:, :]], dim=1) if t > 1 else (x + bos_prefix)

        conv_in = F.pad(x.transpose(1, 2), (3, 0))
        local_ctx = self.local_patch_conv(conv_in).transpose(1, 2)
        x = self.emb_drop(x + self.local_patch_gate * local_ctx)

        cos, sin = self._get_rope(t, device=device)

        total_aux_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_z_loss = torch.zeros((), device=device, dtype=torch.float32)
        want_aux = bool(self.use_moe and (return_aux or self.training))
        use_ckpt = self.training and self.gradient_checkpointing and x.requires_grad
        for blk in self.blocks:
            if use_ckpt:
                x, blk_aux, blk_z = torch.utils.checkpoint.checkpoint(
                    self._block_forward_no_cache,
                    blk,
                    x,
                    z_mem,
                    z_summary,
                    cos,
                    sin,
                    memory_mask,
                    token_mask,
                    want_aux,
                    use_reentrant=False,
                )
            else:
                x, _, blk_aux, blk_z = blk(
                    x,
                    z_mem,
                    z_summary=z_summary,
                    rope_cos=cos,
                    rope_sin=sin,
                    memory_mask=memory_mask,
                    token_mask=token_mask,
                    compute_aux=want_aux,
                )
            if want_aux:
                total_aux_loss = total_aux_loss + blk_aux
                total_z_loss = total_z_loss + blk_z

        if want_aux:
            self.last_aux_loss = total_aux_loss.detach()
            self.last_z_loss = total_z_loss.detach()

        x = self.final_norm(x) + 0.50 * self._norm_z_proj(self.out_z_proj(z_summary)).unsqueeze(1)
        logits = self.lm_head(x)
        if return_aux:
            return logits, total_aux_loss, total_z_loss
        return logits

    def compute_loss(
        self,
        target_byte_ids: torch.Tensor,
        memory_columns: torch.Tensor,
        label_smoothing: float | None = None,
        decoder_byte_dropout: float | None = None,
        clean_z: bool = False,
        memory_mask: torch.Tensor | None = None,
        return_components: bool = False,
        prefix_weight: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute next-byte cross-entropy loss conditioned strictly on Z_ctx, plus Sparse MoE
        load-balancing aux_loss and router z_loss (when broca_use_moe=True).
        Applies decoder input byte dropout (~15%) during teacher-forcing so Broca MUST read Z_ctx
        via Cross-Attention + FiLM + <BOS> prefix + output Z* bridge instead of collapsing.
        When `prefix_weight > 1.0`, upweights the first 16 answer bytes so Z_ctx answer-selection
        gradients are not diluted by 100+ easy continuation bytes.
        """
        inp = target_byte_ids[:, :-1]
        tgt = target_byte_ids[:, 1:]

        drop_p = float(
            self.cfg.broca_decoder_byte_dropout if decoder_byte_dropout is None else decoder_byte_dropout
        )
        if self.training and drop_p > 0.0 and inp.shape[1] > 1:
            maskable = inp < 256
            maskable[:, 0] = False
            drop_mask = (torch.rand(inp.shape, device=inp.device) < drop_p) & maskable
            inp = torch.where(drop_mask, torch.full_like(inp, MASK_BYTE), inp)

        token_mask = tgt != PAD_BYTE
        logits, aux_loss, z_loss = self.forward(
            inp,
            memory_columns,
            clean_z=clean_z,
            memory_mask=memory_mask,
            token_mask=token_mask,
            return_aux=True,
        )
        ls = float(self.cfg.broca_label_smoothing if label_smoothing is None else label_smoothing)
        pw = float(prefix_weight)
        if pw > 1.01 and tgt.shape[1] > 1:
            per_tok = F.cross_entropy(
                logits.reshape(-1, BYTE_VOCAB_SIZE),
                tgt.reshape(-1),
                ignore_index=PAD_BYTE,
                label_smoothing=ls,
                reduction="none",
            ).view(tgt.shape)
            valid = token_mask.to(dtype=per_tok.dtype)
            pos_w = torch.ones((1, tgt.shape[1]), device=tgt.device, dtype=per_tok.dtype)
            # Concentrate answer-selection gradient on byte 0 (after <BOS>) and bytes 1..3,
            # since bytes 4..150 are easily predicted by the unconditional Polish LM prior!
            pos_w[:, 0] = pw * 8.0
            if tgt.shape[1] > 1:
                pos_w[:, 1 : min(4, tgt.shape[1])] = pw * 3.0
            if tgt.shape[1] > 4:
                pos_w[:, 4 : min(12, tgt.shape[1])] = pw
            w = valid * pos_w
            ce_loss = (per_tok * w).sum() / w.sum().clamp(min=1.0)
        else:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, BYTE_VOCAB_SIZE),
                tgt.reshape(-1),
                ignore_index=PAD_BYTE,
                label_smoothing=ls,
            )
        if return_components:
            return ce_loss, aux_loss, z_loss
        if self.use_moe:
            aux_coef = float(getattr(self.cfg, "broca_moe_aux_loss_coef", 0.01))
            z_coef = float(getattr(self.cfg, "broca_moe_z_loss_coef", 0.001))
            return ce_loss + aux_coef * aux_loss + z_coef * z_loss
        return ce_loss

    @torch.no_grad()
    def generate_stream(
        self,
        memory_columns: torch.Tensor,
        max_new_bytes: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> Iterator[str]:
        """
        Fast GPU autoregressive UTF-8 streaming strictly from Z_ctx [1, M_total, D]:
          - Static pre-allocated KV-cache (zero torch.cat per byte)
          - Precomputed Cross-KV + FiLM + out_z_proj (computed ONCE per reply)
          - Vectorized GPU repetition penalty (zero CPU-GPU sync loops)
          - Stochastic nucleus sampling (NEVER pure argmax)
        """
        was_training = self.training
        self.eval()
        try:
            if memory_columns.ndim == 2:
                memory_columns = memory_columns.unsqueeze(0)
            device = next(self.parameters()).device
            kv_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
            mem_mask = None
            if memory_mask is not None:
                mem_mask = memory_mask[:1].to(device=device, dtype=torch.bool)
                if mem_mask.ndim == 1:
                    mem_mask = mem_mask.unsqueeze(0)

            max_bytes = max(8, min(self.max_seq_len - 2, int(max_new_bytes or self.cfg.broca_max_new_bytes)))
            temp = max(0.35, min(1.0, float(temperature if temperature is not None else self.cfg.broca_temperature)))
            p_cut = max(0.1, min(0.99, float(top_p if top_p is not None else self.cfg.broca_top_p)))
            rep_pen = max(1.0, float(repetition_penalty if repetition_penalty is not None else self.cfg.broca_repetition_penalty))

            cos_all, sin_all = self._get_rope(max_bytes + 4, device=device)

            with amp_autocast(device):
                z_mem = self.col_norm(memory_columns[:1].to(device=device, dtype=kv_dtype))
                z_summary = self._pool_z_summary(z_mem, memory_mask=mem_mask)
                cross_kvs = [blk.precompute_cross_kv(z_mem) for blk in self.blocks]
                film_caches = [blk.precompute_film(z_summary) for blk in self.blocks]
                bos_prefix = self._norm_z_proj(self.bos_z_proj(z_summary)).unsqueeze(1)  # [1, 1, D]
                out_z_bias = 0.50 * self._norm_z_proj(self.out_z_proj(z_summary)).unsqueeze(1)  # [1, 1, D]

            # Static pre-allocated KV-cache buffers per layer (eliminates O(T^2) allocations!)
            self_kvs: list[tuple[torch.Tensor, torch.Tensor]] = [
                (
                    torch.empty((1, self.n_heads, max_bytes, self.head_dim), device=device, dtype=kv_dtype),
                    torch.empty((1, self.n_heads, max_bytes, self.head_dim), device=device, dtype=kv_dtype),
                )
                for _ in self.blocks
            ]
            # Fixed double-buffered conv ring (ZERO per-byte allocations!) & repetition penalty mask
            conv_buf = torch.zeros((1, self.dim, 4), device=device, dtype=kv_dtype)
            conv_buf_next = torch.zeros((1, self.dim, 4), device=device, dtype=kv_dtype)
            recent_ids = torch.full((32,), PAD_BYTE, dtype=torch.long, device=device)
            rep_mask = torch.zeros(BYTE_VOCAB_SIZE, dtype=torch.bool, device=device)

            # Static illegal-byte mask on GPU
            static_ban = torch.zeros(BYTE_VOCAB_SIZE, dtype=torch.bool, device=device)
            static_ban[PAD_BYTE] = True
            static_ban[BOS_BYTE] = True
            static_ban[MASK_BYTE] = True
            static_ban[0:10] = True
            static_ban[11:32] = True
            static_ban[127] = True

            utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
            cur_id = torch.tensor([[BOS_BYTE]], dtype=torch.long, device=device)

            for step in range(max_bytes):
                with amp_autocast(device):
                    emb_step = self.byte_emb(cur_id)  # [1, 1, D]
                    if step == 0:
                        emb_step = emb_step + bos_prefix
                    conv_buf_next[:, :, :-1].copy_(conv_buf[:, :, 1:])
                    conv_buf_next[:, :, -1].copy_(emb_step[:, 0, :].to(dtype=kv_dtype))
                    conv_buf, conv_buf_next = conv_buf_next, conv_buf
                    local_ctx = self.local_patch_conv(conv_buf).transpose(1, 2)  # [1, 1, D]
                    x = emb_step + self.local_patch_gate * local_ctx

                    cos_s = cos_all[step : step + 1]
                    sin_s = sin_all[step : step + 1]
                    for i, blk in enumerate(self.blocks):
                        x, self_kvs[i], _, _ = blk(
                            x,
                            z_mem,
                            z_summary=z_summary,
                            rope_cos=cos_s,
                            rope_sin=sin_s,
                            self_kv_cache=self_kvs[i],
                            cross_kv_cache=cross_kvs[i],
                            film_cache=film_caches[i],
                            cache_pos=step,
                            memory_mask=mem_mask,
                            compute_aux=False,
                        )

                    logits = self.lm_head(self.final_norm(x) + out_z_bias)[:, -1, :].float().squeeze(0)  # [260]

                logits.masked_fill_(static_ban, -1e9)
                if step == 0:
                    logits[10] = -1e9
                    logits[32] = -1e9
                if step < 6:
                    logits[EOS_BYTE] = -1e9

                # Fully vectorized zero-alloc GPU repetition penalty
                if rep_pen > 1.0 and step > 0:
                    rep_mask.zero_()
                    safe_rec = recent_ids.clamp(0, BYTE_VOCAB_SIZE - 1)
                    rep_mask.scatter_(0, safe_rec, recent_ids < 256)
                    logits = torch.where(
                        rep_mask,
                        torch.where(logits > 0, logits / rep_pen, logits * rep_pen),
                        logits,
                    )

                # Branchless stochastic nucleus (top-p) sampling (only ONE GPU->CPU sync per byte!)
                probs = F.softmax(logits / temp, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                remove_mask = cum_probs > p_cut
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False
                filtered_probs = sorted_probs.masked_fill(remove_mask, 0.0)
                prob_sum = filtered_probs.sum()
                bad_sum = (prob_sum <= 0.0) | torch.isnan(prob_sum)
                safe_probs = torch.where(bad_sum, probs, filtered_probs / prob_sum.clamp(min=1e-9))
                sampled_pos = torch.multinomial(safe_probs, num_samples=1)
                next_t = torch.where(bad_sum, sampled_pos, sorted_idx[sampled_pos])

                next_id = int(next_t.item())
                if next_id == EOS_BYTE:
                    break

                recent_ids[step % 32] = next_id
                cur_id[0, 0] = next_id

                if 0 <= next_id <= 255:
                    piece = utf8_decoder.decode(bytes([next_id]), final=False)
                    if piece:
                        yield piece

            tail = utf8_decoder.decode(b"", final=True)
            if tail:
                yield tail
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate_batch(
        self,
        memory_columns_batch: torch.Tensor,
        temperatures: list[float],
        max_new_bytes: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        """
        Batched Parallel System-2 Generation on GPU:
        Generates K candidate thoughts simultaneously in ONE GPU batch with static KV-cache,
        precomputed Cross-KV for Z_ctx, vectorized repetition penalty, and ZERO per-byte CPU syncs!
        """
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            kv_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
            b = memory_columns_batch.shape[0]
            mem_mask = memory_mask.to(device=device, dtype=torch.bool) if memory_mask is not None else None

            max_bytes = max(8, min(self.max_seq_len - 2, int(max_new_bytes or self.cfg.broca_max_new_bytes)))
            temps = torch.tensor(
                [max(0.35, min(1.0, float(t))) for t in temperatures],
                device=device,
                dtype=torch.float32,
            ).view(b, 1)
            p_cut = max(0.1, min(0.99, float(top_p if top_p is not None else self.cfg.broca_top_p)))
            rep_pen = max(1.0, float(repetition_penalty if repetition_penalty is not None else self.cfg.broca_repetition_penalty))

            cos_all, sin_all = self._get_rope(max_bytes + 4, device=device)

            with amp_autocast(device):
                z_mem = self.col_norm(memory_columns_batch.to(device=device, dtype=kv_dtype))
                z_summary = self._pool_z_summary(z_mem, memory_mask=mem_mask)
                cross_kvs = [blk.precompute_cross_kv(z_mem) for blk in self.blocks]
                film_caches = [blk.precompute_film(z_summary) for blk in self.blocks]
                bos_prefix = self._norm_z_proj(self.bos_z_proj(z_summary)).unsqueeze(1)  # [B, 1, D]
                out_z_bias = 0.50 * self._norm_z_proj(self.out_z_proj(z_summary)).unsqueeze(1)  # [B, 1, D]

            self_kvs: list[tuple[torch.Tensor, torch.Tensor]] = [
                (
                    torch.empty((b, self.n_heads, max_bytes, self.head_dim), device=device, dtype=kv_dtype),
                    torch.empty((b, self.n_heads, max_bytes, self.head_dim), device=device, dtype=kv_dtype),
                )
                for _ in self.blocks
            ]
            conv_buf = torch.zeros((b, self.dim, 4), device=device, dtype=kv_dtype)
            conv_buf_next = torch.zeros((b, self.dim, 4), device=device, dtype=kv_dtype)
            recent_ids = torch.full((b, 32), PAD_BYTE, dtype=torch.long, device=device)
            rep_mask = torch.zeros((b, BYTE_VOCAB_SIZE), dtype=torch.bool, device=device)
            pad_vec = torch.full((b,), PAD_BYTE, dtype=torch.long, device=device)
            eos_vec = torch.full((b,), EOS_BYTE, dtype=torch.long, device=device)
            finished = torch.zeros((b,), dtype=torch.bool, device=device)
            out_tokens = torch.full((b, max_bytes), PAD_BYTE, dtype=torch.long, device=device)

            static_ban = torch.zeros(BYTE_VOCAB_SIZE, dtype=torch.bool, device=device)
            static_ban[PAD_BYTE] = True
            static_ban[BOS_BYTE] = True
            static_ban[MASK_BYTE] = True
            static_ban[0:10] = True
            static_ban[11:32] = True
            static_ban[127] = True

            cur_ids = torch.full((b, 1), BOS_BYTE, dtype=torch.long, device=device)

            for step in range(max_bytes):
                with amp_autocast(device):
                    emb_step = self.byte_emb(cur_ids)  # [B, 1, D]
                    if step == 0:
                        emb_step = emb_step + bos_prefix
                    conv_buf_next[:, :, :-1].copy_(conv_buf[:, :, 1:])
                    conv_buf_next[:, :, -1].copy_(emb_step[:, 0, :].to(dtype=kv_dtype))
                    conv_buf, conv_buf_next = conv_buf_next, conv_buf
                    local_ctx = self.local_patch_conv(conv_buf).transpose(1, 2)
                    x = emb_step + self.local_patch_gate * local_ctx

                    cos_s = cos_all[step : step + 1]
                    sin_s = sin_all[step : step + 1]
                    for i, blk in enumerate(self.blocks):
                        x, self_kvs[i], _, _ = blk(
                            x,
                            z_mem,
                            z_summary=z_summary,
                            rope_cos=cos_s,
                            rope_sin=sin_s,
                            self_kv_cache=self_kvs[i],
                            cross_kv_cache=cross_kvs[i],
                            film_cache=film_caches[i],
                            cache_pos=step,
                            memory_mask=mem_mask,
                            compute_aux=False,
                        )

                    logits = self.lm_head(self.final_norm(x) + out_z_bias)[:, -1, :].float()  # [B, 260]

                logits[:, static_ban] = -1e9
                if step == 0:
                    logits[:, 10] = -1e9
                    logits[:, 32] = -1e9
                if step < 6:
                    logits[:, EOS_BYTE] = -1e9

                if rep_pen > 1.0 and step > 0:
                    rep_mask.zero_()
                    safe_rec = recent_ids.clamp(0, BYTE_VOCAB_SIZE - 1)
                    rep_mask.scatter_(1, safe_rec, recent_ids < 256)
                    logits = torch.where(
                        rep_mask,
                        torch.where(logits > 0, logits / rep_pen, logits * rep_pen),
                        logits,
                    )

                probs = F.softmax(logits / temps, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                remove_mask = cum_probs > p_cut
                remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                remove_mask[:, 0] = False
                filtered_probs = sorted_probs.masked_fill(remove_mask, 0.0)
                prob_sums = filtered_probs.sum(dim=-1, keepdim=True)
                bad_rows = (prob_sums <= 0.0) | torch.isnan(prob_sums)
                safe_probs = torch.where(bad_rows, probs, filtered_probs / prob_sums.clamp(min=1e-9))
                sampled_pos = torch.multinomial(safe_probs, num_samples=1)  # [B, 1]
                next_ids = torch.where(
                    bad_rows,
                    sampled_pos,
                    sorted_idx.gather(1, sampled_pos),
                ).squeeze(1)  # [B]

                # Branchless GPU state update (zero per-byte CPU->GPU or GPU->CPU sync!)
                out_tokens[:, step] = torch.where(finished, pad_vec, next_ids)
                finished = finished | (next_ids == EOS_BYTE)
                recent_ids[:, step % 32] = torch.where(finished, pad_vec, next_ids)
                cur_ids[:, 0] = torch.where(finished, eos_vec, next_ids)

                if (step + 1) % 16 == 0 and bool(finished.all().item()):
                    break

            # Single GPU -> CPU copy after the entire batch finishes
            rows_cpu = out_tokens.cpu().tolist()
            results: list[dict[str, Any]] = []
            for row in rows_cpu:
                raw_bytes = bytearray(nid for nid in row if 0 <= nid <= 255)
                txt = bytes(raw_bytes).decode("utf-8", errors="ignore").strip()
                results.append({"text": txt, "num_bytes": len(txt.encode("utf-8"))})
            return results
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate(
        self,
        memory_columns: torch.Tensor,
        max_new_bytes: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if memory_columns.ndim == 2:
            memory_columns = memory_columns.unsqueeze(0)
        if memory_mask is not None and memory_mask.ndim == 1:
            memory_mask = memory_mask.unsqueeze(0)
        temp = float(temperature if temperature is not None else self.cfg.broca_temperature)
        return self.generate_batch(
            memory_columns_batch=memory_columns[:1],
            temperatures=[temp],
            max_new_bytes=max_new_bytes,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            memory_mask=memory_mask[:1] if memory_mask is not None else None,
        )[0]

    @torch.no_grad()
    def score_text_confidence(
        self,
        text: str,
        memory_columns: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Compute Broca's own neural confidence (average byte cross-entropy NLL and exp(-NLL))
        for a generated draft conditioned on Z_ctx in one parallel GPU forward pass.
        """
        s = (text or "").strip()
        if len(s) < 3:
            return {"avg_nll": 5.0, "confidence": 0.01}
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            kv_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
            if memory_columns.ndim == 2:
                memory_columns = memory_columns.unsqueeze(0)
            z_mem = memory_columns[:1].to(device=device, dtype=kv_dtype)
            mem_mask = memory_mask[:1].to(device=device, dtype=torch.bool) if memory_mask is not None else None
            if mem_mask is not None and mem_mask.ndim == 1:
                mem_mask = mem_mask.unsqueeze(0)
            tgt_bytes, _ = encode_utf8_bytes(
                [s],
                max_bytes=self.max_seq_len,
                patch_size=1,
                add_bos_eos=True,
                mask_prob=0.0,
                device=device,
            )
            inp = tgt_bytes[:, :-1]
            tgt = tgt_bytes[:, 1:]
            with amp_autocast(device):
                logits = self.forward(inp, z_mem, clean_z=True, memory_mask=mem_mask)
            nll = float(
                F.cross_entropy(
                    logits.float().reshape(-1, BYTE_VOCAB_SIZE),
                    tgt.reshape(-1),
                    ignore_index=PAD_BYTE,
                ).item()
            )
            conf = float(math.exp(-min(10.0, max(0.0, nll))))
            return {"avg_nll": round(nll, 4), "confidence": round(conf, 4)}
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def score_texts_confidence_batch(
        self,
        texts: list[str],
        memory_columns_batch: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
    ) -> list[dict[str, float]]:
        """
        Score K candidate drafts simultaneously in ONE batched GPU forward pass.
        """
        k = len(texts)
        if k == 0:
            return []
        results: list[dict[str, float]] = [{"avg_nll": 5.0, "confidence": 0.01} for _ in range(k)]
        valid_indices = [i for i, t in enumerate(texts) if len((t or "").strip()) >= 3]
        if not valid_indices:
            return results

        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            kv_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
            sel_texts = [(texts[i] or "").strip() for i in valid_indices]
            idx_t = torch.tensor(valid_indices, dtype=torch.long, device=memory_columns_batch.device)
            z_mem = memory_columns_batch.index_select(0, idx_t).to(device=device, dtype=kv_dtype)
            mem_mask = (
                memory_mask.index_select(0, idx_t.to(memory_mask.device)).to(device=device, dtype=torch.bool)
                if memory_mask is not None
                else None
            )
            tgt_bytes, _ = encode_utf8_bytes(
                sel_texts,
                max_bytes=self.max_seq_len,
                patch_size=1,
                add_bos_eos=True,
                mask_prob=0.0,
                device=device,
            )
            inp = tgt_bytes[:, :-1]
            tgt = tgt_bytes[:, 1:]
            with amp_autocast(device):
                logits = self.forward(inp, z_mem, clean_z=True, memory_mask=mem_mask)
            per_tok = F.cross_entropy(
                logits.float().reshape(-1, BYTE_VOCAB_SIZE),
                tgt.reshape(-1),
                ignore_index=PAD_BYTE,
                reduction="none",
            ).view(tgt.shape)
            valid_tok = (tgt != PAD_BYTE).float()
            row_nlls = ((per_tok * valid_tok).sum(dim=1) / valid_tok.sum(dim=1).clamp(min=1.0)).tolist()
            for row_i, nll_v in zip(valid_indices, row_nlls):
                nll = float(nll_v)
                conf = float(math.exp(-min(10.0, max(0.0, nll))))
                results[row_i] = {"avg_nll": round(nll, 4), "confidence": round(conf, 4)}
            return results
        finally:
            if was_training:
                self.train()

    def export_state(self) -> dict[str, Any]:
        return {
            "weights": {k: v.detach().cpu().to(torch.bfloat16) for k, v in self.state_dict().items()},
            "trained_steps": self.trained_steps,
            "last_loss": self.last_loss,
            "fluency_score": self.fluency_score,
        }

    def import_state(self, state: dict[str, Any], device: torch.device) -> dict[str, list[str]]:
        saved = state.get("weights", {})
        cur = self.state_dict()
        merged: dict[str, torch.Tensor] = {}
        loaded_keys: list[str] = []
        mismatched_keys: list[str] = []
        missing_keys: list[str] = []
        unexpected_keys = [k for k in saved.keys() if k not in cur]

        for k, v in cur.items():
            if k in saved:
                sv = saved[k]
                if isinstance(sv, torch.Tensor) and sv.shape == v.shape:
                    merged[k] = sv.to(device=device, dtype=v.dtype)
                    loaded_keys.append(k)
                else:
                    # Safely skip shape-mismatched tensors (e.g. dense FFN -> MoE migration) and keep fresh init
                    merged[k] = v
                    sv_shape = tuple(sv.shape) if isinstance(sv, torch.Tensor) else type(sv).__name__
                    mismatched_keys.append(f"{k} (ckpt={sv_shape}, model={tuple(v.shape)})")
            else:
                merged[k] = v
                missing_keys.append(k)

        self.load_state_dict(merged, strict=False)
        self.trained_steps = int(state.get("trained_steps", 0))
        self.last_loss = float(state.get("last_loss", 5.5))
        self.fluency_score = float(state.get("fluency_score", 0.0))

        report = {
            "loaded": loaded_keys,
            "mismatched": mismatched_keys,
            "missing": missing_keys,
            "unexpected": unexpected_keys,
        }
        self.last_import_report = report
        if mismatched_keys or missing_keys or unexpected_keys:
            extra = ""
            if mismatched_keys:
                extra += f" | mismatch={mismatched_keys[:3]}"
            if missing_keys:
                extra += f" | new_init={missing_keys[:3]}"
            _logger.warning(
                "BrocaByteTransformer migracja checkpointu: wczytano=%d, pominięto_niezgodny_kształt=%d, "
                "zainicjalizowano_nowe=%d, pominięto_stare_klucze=%d%s",
                len(loaded_keys),
                len(mismatched_keys),
                len(missing_keys),
                len(unexpected_keys),
                extra,
            )
        return report
