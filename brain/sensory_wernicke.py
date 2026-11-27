from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BrainConfig

# Pure UTF-8 byte space (0..255) + 4 control bytes — ZERO tokens, ZERO word dictionary
PAD_BYTE = 256
BOS_BYTE = 257
EOS_BYTE = 258
MASK_BYTE = 259
BYTE_VOCAB_SIZE = 260


def amp_autocast(device: torch.device):
    """Enable bfloat16 autocast on CUDA (RTX 5080 / Blackwell), clean fallback on CPU."""
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def maybe_compile_module(module: nn.Module, enabled: bool) -> nn.Module:
    """Optionally apply torch.compile(mode='reduce-overhead') with safe Windows/eager fallback."""
    if not enabled:
        return module
    try:
        return torch.compile(module, mode="reduce-overhead")
    except Exception:
        return module


def configure_torch_backend() -> None:
    """Enable TensorFloat32, cuDNN autotuning, and high matmul precision for Blackwell / RTX 5080."""
    try:
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def precompute_rope(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine/sine tables for Rotary Positional Embeddings (RoPE)."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / float(half)))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Positional Embeddings to [B, H, T, Dh] in native tensor dtype (bf16/fp32)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos_b = cos[None, None, :, :].to(dtype=x.dtype)
    sin_b = sin[None, None, :, :].to(dtype=x.dtype)
    out1 = x1 * cos_b - x2 * sin_b
    out2 = x1 * sin_b + x2 * cos_b
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            w = self.weight if self.weight.dtype == x.dtype else self.weight.to(dtype=x.dtype)
            return F.rms_norm(x, (self.dim,), w, self.eps)
        orig_dtype = x.dtype
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(orig_dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate_up = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(self.dropout(F.silu(gate) * up))


def k_wta(x: torch.Tensor, k: int, normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Biological lateral inhibition: k-Winners-Take-All (k-WTA) producing a Sparse Distributed
    Representation (SDR). Preserves gradients through the selected top-k winners during
    Wernicke-Broca denoising training while remaining exact top-k sparse (0 elsewhere).
    Returns:
      - sdr: [..., N] sparse tensor (L2-normalized if normalize=True)
      - top_idx: [..., k] int64 winner indices
      - top_val: [..., k] positive winner activations (L2-normalized if normalize=True)
    """
    n = x.shape[-1]
    k_eff = max(1, min(int(k), n))
    # Shift winner activations into strictly positive range via softplus after top-k selection
    top_scores, top_idx = torch.topk(x, k=k_eff, dim=-1, sorted=True)
    top_act = F.softplus(top_scores) + 1e-4
    if normalize:
        top_act = F.normalize(top_act, dim=-1, eps=1e-6)
    sdr = torch.zeros_like(x, dtype=top_act.dtype)
    sdr.scatter_(-1, top_idx, top_act)
    return sdr, top_idx, top_act


def encode_utf8_bytes(
    texts: list[str],
    max_bytes: int = 512,
    patch_size: int = 4,
    add_bos_eos: bool = False,
    mask_prob: float = 0.0,
    device: torch.device | None = None,
    pin_memory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fast NumPy-vectorized lossless UTF-8 byte encoder (zero tokenizer, zero word dictionary).
    Returns:
      - byte_ids: [B, L] int64 where L is padded to a multiple of patch_size (when patch_size > 1)
      - valid_mask: [B, L] bool (True on real non-PAD bytes)
    """
    b = len(texts)
    if b == 0:
        empty_len = max(1, patch_size)
        z = torch.full((0, empty_len), PAD_BYTE, dtype=torch.long, device=device)
        m = torch.zeros((0, empty_len), dtype=torch.bool, device=device)
        return z, m

    cap = max(1, max_bytes - (2 if add_bos_eos else 0))
    raw_list = [(t or "").encode("utf-8")[:cap] for t in texts]
    raw_max = max((len(r) for r in raw_list), default=1)
    max_len = raw_max + (2 if add_bos_eos else 0)
    if patch_size > 1:
        max_len = max(patch_size, math.ceil(max_len / patch_size) * patch_size)
    max_len = min(max_bytes, max(1, max_len))
    if patch_size > 1 and max_len % patch_size != 0:
        max_len = max(patch_size, (max_len // patch_size) * patch_size)

    arr = np.full((b, max_len), PAD_BYTE, dtype=np.int64)
    if add_bos_eos:
        for i, raw in enumerate(raw_list):
            n = min(len(raw), max_len - 2)
            arr[i, 0] = BOS_BYTE
            if n > 0:
                arr[i, 1 : 1 + n] = np.frombuffer(raw[:n], dtype=np.uint8)
            arr[i, 1 + n] = EOS_BYTE
    else:
        for i, raw in enumerate(raw_list):
            n = min(len(raw), max_len)
            if n > 0:
                arr[i, :n] = np.frombuffer(raw[:n], dtype=np.uint8)

    byte_ids = torch.from_numpy(arr)
    valid_mask = byte_ids.ne(PAD_BYTE)

    if device is not None and device.type == "cuda":
        if pin_memory:
            byte_ids = byte_ids.pin_memory().to(device, non_blocking=True)
            valid_mask = valid_mask.pin_memory().to(device, non_blocking=True)
        else:
            byte_ids = byte_ids.to(device)
            valid_mask = valid_mask.to(device)
    elif pin_memory and torch.cuda.is_available():
        byte_ids = byte_ids.pin_memory()
        valid_mask = valid_mask.pin_memory()

    if mask_prob > 0.0:
        corruptible = valid_mask & (byte_ids < 256)
        drop = (torch.rand(byte_ids.shape, device=byte_ids.device) < mask_prob) & corruptible
        byte_ids = torch.where(drop, torch.full_like(byte_ids, MASK_BYTE), byte_ids)

    return byte_ids, valid_mask


class _WernickeEncoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_dim, dropout=dropout)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        patch_valid: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, p, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, p, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, p, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, p, self.n_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        attn_mask = None
        if patch_valid is not None:
            # Ensure at least patch 0 is attendable to prevent NaN softmax on empty inputs
            safe_valid = patch_valid.clone()
            safe_valid[:, 0] = True
            attn_mask = safe_valid[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, p, d)
        x = x + self.proj(out)
        x = x + self.ff(self.norm2(x))
        return x


@dataclass
class WernickeOutput:
    columns: torch.Tensor  # [B, 16, D]
    sdr: torch.Tensor  # [B, N] sparse k-WTA unit vector
    reconstructed_columns: torch.Tensor  # [B, 16, D] decoded purely from SDR bottleneck
    active_indices: torch.Tensor  # [B, k] int16
    active_values: torch.Tensor  # [B, k] bfloat16
    sparsity_pct: float
    patch_columns: torch.Tensor | None = None  # [B, P_active, D] sequential RoPE patches (word order & binding)
    patch_mask: torch.Tensor | None = None  # [B, P_active] bool mask (True = valid patch)


class WernickeByteCortex(nn.Module):
    """
    Receptive Sensory Cortex (Wernicke's Area — 100% tokenless UTF-8):
      1. Raw UTF-8 bytes (0..255 + control bytes) -> Byte-Patching Conv1D (patch_size=4, stride=4)
      2. Lightweight RMSNorm + SDPA + SwiGLU patch encoder
      3. Attention projection into 16 cortical columns S in R^{B x 16 x D}
      4. Lateral inhibition k-WTA -> Sparse Distributed Representation (SDR, 3-5% active neurons)
      5. Aligned SDR -> 16-column decoder head so associative cortex attractors z* map directly
         into the same column space expected by BrocaByteTransformer.
    Once passed through k-WTA, raw text vanishes completely from the cognitive pipeline.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_columns = cfg.num_columns
        self.column_dim = cfg.column_dim
        self.num_neurons = cfg.num_neurons
        self.sdr_k = cfg.sdr_k
        self.patch_size = cfg.patch_size
        self.patch_stride = cfg.patch_stride
        self.max_patches = max(16, cfg.max_input_bytes // cfg.patch_stride + 4)
        self.max_patch_slots = max(16, int(getattr(cfg, "wernicke_max_patch_slots", 64)))
        self.rf_anchor_weight = max(0.0, min(0.15, float(getattr(cfg, "wernicke_rf_anchor_weight", 0.12))))

        self.byte_emb = nn.Embedding(cfg.byte_vocab_size, cfg.byte_embed_dim, padding_idx=PAD_BYTE)
        self.patch_conv = nn.Conv1d(
            in_channels=cfg.byte_embed_dim,
            out_channels=cfg.column_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_stride,
        )
        self.patch_norm = RMSNorm(cfg.column_dim)
        self.patch_pos = nn.Parameter(torch.randn(1, self.max_patches, cfg.column_dim) * 0.12)

        self.blocks = nn.ModuleList(
            [
                _WernickeEncoderBlock(
                    dim=cfg.column_dim,
                    n_heads=cfg.wernicke_heads,
                    ff_dim=cfg.wernicke_ff_dim,
                    dropout=0.05,
                )
                for _ in range(cfg.wernicke_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.column_dim)
        self.patch_out_norm = RMSNorm(cfg.column_dim)

        rope_cos, rope_sin = precompute_rope(
            self.max_patches + 8,
            cfg.column_dim // cfg.wernicke_heads,
            device=torch.device("cpu"),
        )
        self.register_buffer("_rope_cos", rope_cos, persistent=False)
        self.register_buffer("_rope_sin", rope_sin, persistent=False)

        # Learned 16 cortical column pooling queries (diverse initialization prevents attention collapse)
        self.col_queries = nn.Parameter(torch.randn(1, cfg.num_columns, cfg.column_dim) * 0.25)
        self.col_q_proj = nn.Linear(cfg.column_dim, cfg.column_dim, bias=False)
        self.col_kv_proj = nn.Linear(cfg.column_dim, cfg.column_dim * 2, bias=False)
        self.col_out_proj = nn.Linear(cfg.column_dim, cfg.column_dim, bias=False)
        self.col_norm = RMSNorm(cfg.column_dim)
        self.col_heads = cfg.wernicke_heads
        self.col_head_dim = cfg.column_dim // cfg.wernicke_heads

        # Factored bottleneck: 16 columns [B, 16, D] -> SDR [B, N] via k-WTA (no bias -> zero static neuron bias)
        self.col_compress = nn.Linear(cfg.column_dim, 64, bias=False)
        self.to_sdr = nn.Linear(cfg.num_columns * 64, cfg.num_neurons, bias=False)

        # Primary sensory receptive fields (lightweight <= 0.15 anchor so learned RoPE patches lead!)
        gen = torch.Generator().manual_seed(20261125)
        rf_byte = F.normalize(torch.randn(cfg.byte_vocab_size, 32, generator=gen), dim=-1)
        rf_byte[PAD_BYTE].zero_()
        rf_byte[MASK_BYTE].zero_()
        self.register_buffer("rf_byte_basis", rf_byte)
        rf_kernel = torch.randn(256, 32, cfg.patch_size, generator=gen) * (1.0 / math.sqrt(32 * cfg.patch_size))
        self.register_buffer("rf_patch_kernel", rf_kernel)
        rf_proj = torch.randn(512, cfg.num_neurons, generator=gen) * (1.0 / math.sqrt(512))
        self.register_buffer("rf_to_sdr", rf_proj)

        # Aligned SDR -> 16 cortical columns [B, 16, D] (for denoising bottleneck & Cortex->Broca bridge)
        self.sdr_expand_in = nn.Linear(cfg.num_neurons, 512, bias=False)
        self.sdr_expand_norm = RMSNorm(512)
        self.sdr_expand_out = nn.Linear(512, cfg.num_columns * cfg.column_dim, bias=False)
        # Fixed orthogonal residual basis so Z* can NEVER collapse to rank-1 during long pre-training
        orth_basis = torch.randn(cfg.num_neurons, cfg.num_columns * cfg.column_dim, generator=gen) * (
            1.0 / math.sqrt(float(cfg.sdr_k))
        )
        self.register_buffer("sdr_orth_basis", orth_basis.to(torch.bfloat16))
        self.sdr_col_norm = RMSNorm(cfg.column_dim)

    def _get_rope(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= self._rope_cos.shape[0]:
            return self._rope_cos[:seq_len], self._rope_sin[:seq_len]
        return precompute_rope(seq_len, self.col_head_dim, device=device)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def sdr_to_columns(self, sdr: torch.Tensor) -> torch.Tensor:
        """
        Map sparse cortical state z [B, N] into 16 cortical columns Z [B, 16, D].
        Combines learned symmetric-tanh expansion with a frozen orthogonal SDR residual basis
        so distinct SDR inputs ALWAYS remain sharply separated in Z* space.
        """
        b = sdr.shape[0]
        h = torch.tanh(self.sdr_expand_norm(self.sdr_expand_in(sdr)))
        learned_cols = self.sdr_expand_out(h).view(b, self.num_columns, self.column_dim)
        if sdr.device.type == "cuda" and self.sdr_orth_basis.dtype == torch.bfloat16:
            # Avoid allocating/copying the 600 MB float32 [12288, 12288] basis on every call!
            orth_cols = (sdr.to(dtype=torch.bfloat16) @ self.sdr_orth_basis).to(dtype=learned_cols.dtype).view(
                b, self.num_columns, self.column_dim
            )
        else:
            orth_cols = (sdr @ self.sdr_orth_basis.to(dtype=sdr.dtype)).view(b, self.num_columns, self.column_dim)
        return self.sdr_col_norm(0.65 * learned_cols + 0.65 * orth_cols)

    def forward(
        self,
        byte_ids: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> WernickeOutput:
        b, seq_len = byte_ids.shape
        if seq_len % self.patch_stride != 0:
            pad_n = self.patch_stride - (seq_len % self.patch_stride)
            byte_ids = F.pad(byte_ids, (0, pad_n), value=PAD_BYTE)
            if valid_mask is not None:
                valid_mask = F.pad(valid_mask, (0, pad_n), value=False)
            seq_len = byte_ids.shape[1]

        # 1) Byte-Patching Conv1D
        emb = self.byte_emb(byte_ids)  # [B, L, E]
        patches = self.patch_conv(emb.transpose(1, 2)).transpose(1, 2)  # [B, P, D]
        p = patches.shape[1]
        pos = self.patch_pos[:, :p, :] if p <= self.max_patches else F.interpolate(
            self.patch_pos.transpose(1, 2), size=p, mode="linear", align_corners=False
        ).transpose(1, 2)
        h = self.patch_norm(patches + pos)

        patch_valid = None
        if valid_mask is not None:
            patch_valid = valid_mask.view(b, p, self.patch_stride).any(dim=-1)  # [B, P]

        # 2) Lightweight RMSNorm + RoPE SDPA + SwiGLU encoder blocks
        cos, sin = self._get_rope(p, device=h.device)
        for blk in self.blocks:
            h = blk(h, patch_valid=patch_valid, rope_cos=cos, rope_sin=sin)
        h = self.final_norm(h)

        # 2b) Sequential sensory patch columns [B, P_active, D] preserving syntactic word order
        p_valid = patch_valid.clone() if patch_valid is not None else torch.ones((b, p), dtype=torch.bool, device=h.device)
        p_valid[:, 0] = True
        if p > self.max_patch_slots:
            p_cols = F.adaptive_avg_pool1d(h.transpose(1, 2), self.max_patch_slots).transpose(1, 2)
            p_mask = F.adaptive_max_pool1d(p_valid.float().unsqueeze(1), self.max_patch_slots).squeeze(1) > 0.5
            p_mask[:, 0] = True
        else:
            p_cols = h
            p_mask = p_valid
        patch_columns = self.patch_out_norm(p_cols) * p_mask.unsqueeze(-1).to(dtype=p_cols.dtype)

        # 3) Project into 16 cortical columns S in R^{B x 16 x D}
        q = self.col_q_proj(self.col_queries.expand(b, -1, -1))
        k, v = self.col_kv_proj(h).chunk(2, dim=-1)
        q = q.view(b, self.num_columns, self.col_heads, self.col_head_dim).transpose(1, 2)
        k = k.view(b, p, self.col_heads, self.col_head_dim).transpose(1, 2)
        v = v.view(b, p, self.col_heads, self.col_head_dim).transpose(1, 2)

        attn_mask = None
        if patch_valid is not None:
            safe_valid = patch_valid.clone()
            safe_valid[:, 0] = True
            attn_mask = safe_valid[:, None, None, :]

        cols = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        cols = cols.transpose(1, 2).contiguous().view(b, self.num_columns, self.column_dim)
        columns = self.col_norm(self.col_out_proj(cols))  # [B, 16, D]

        # 4) Lateral inhibition k-WTA -> Sparse Distributed Representation (SDR) for fast Hippo/Hopfield indexing.
        # Learned RoPE patch columns lead (1 - rf_w); fixed 4-gram basis is strictly a lightweight anchor <= 0.15.
        comp = F.silu(self.col_compress(columns)).reshape(b, -1)  # [B, 16 * 64]
        learned_logits = self.to_sdr(comp)  # [B, N]
        learned_logits = (learned_logits - learned_logits.mean(dim=-1, keepdim=True)) / (
            learned_logits.std(dim=-1, keepdim=True).clamp(min=1e-4)
        )

        rf_w = self.rf_anchor_weight
        if rf_w > 0.0:
            with torch.no_grad():
                rf_emb = F.embedding(byte_ids, self.rf_byte_basis).to(dtype=h.dtype)  # [B, L, 32]
                rf_conv = F.conv1d(
                    rf_emb.transpose(1, 2),
                    self.rf_patch_kernel.to(dtype=h.dtype),
                    stride=1,
                    padding=1,
                ).transpose(1, 2)[:, : byte_ids.shape[1], :]  # [B, L, 256]
                rf_act = F.silu(rf_conv)
                if valid_mask is not None:
                    vm = valid_mask[:, : rf_act.shape[1]]
                    vm_f = vm.unsqueeze(-1).to(dtype=rf_act.dtype)
                    rf_mean = (rf_act * vm_f).sum(dim=1) / vm_f.sum(dim=1).clamp(min=1.0)
                    rf_max = rf_act.masked_fill(~vm.unsqueeze(-1), -1e4).max(dim=1).values
                    rf_max = torch.where(vm_f.sum(dim=1) > 0, rf_max, torch.zeros_like(rf_max))
                else:
                    rf_mean = rf_act.mean(dim=1)
                    rf_max = rf_act.max(dim=1).values
                rf_feat = F.normalize(torch.cat([rf_mean, rf_max], dim=-1), dim=-1, eps=1e-6)
                if byte_ids.device.type == "cuda":
                    with amp_autocast(byte_ids.device):
                        rf_logits = (rf_feat @ self.rf_to_sdr.to(dtype=rf_feat.dtype)).float()
                else:
                    rf_logits = rf_feat.float() @ self.rf_to_sdr.float()
                rf_logits = (rf_logits - rf_logits.mean(dim=-1, keepdim=True)) / (
                    rf_logits.std(dim=-1, keepdim=True).clamp(min=1e-4)
                )
            logits = (1.0 - rf_w) * learned_logits.float() + rf_w * rf_logits
        else:
            logits = learned_logits.float()

        sdr, top_idx, top_val = k_wta(logits, k=self.sdr_k, normalize=True)
        sdr = sdr.to(dtype=columns.dtype)

        # 5) Bottleneck reconstruction into 16 cortical columns (raw text is gone)
        recon_cols = self.sdr_to_columns(sdr)
        sparsity_pct = round(100.0 * (top_idx.shape[-1] / float(self.num_neurons)), 2)

        return WernickeOutput(
            columns=columns,
            sdr=sdr,
            reconstructed_columns=recon_cols,
            active_indices=top_idx.to(torch.int16),
            active_values=top_val.to(torch.bfloat16),
            sparsity_pct=sparsity_pct,
            patch_columns=patch_columns,
            patch_mask=p_mask,
        )

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        device: torch.device,
        batch_size: int = 32,
        mask_prob: float = 0.0,
    ) -> WernickeOutput:
        """Batch-encode UTF-8 strings into 16 cortical columns, sequential patch columns & sparse SDRs."""
        was_training = self.training
        self.eval()
        all_cols: list[torch.Tensor] = []
        all_sdrs: list[torch.Tensor] = []
        all_recon: list[torch.Tensor] = []
        all_idx: list[torch.Tensor] = []
        all_val: list[torch.Tensor] = []
        all_patch_cols: list[torch.Tensor] = []
        all_patch_masks: list[torch.Tensor] = []
        sparsity_pct = round(100.0 * (self.sdr_k / float(self.num_neurons)), 2)

        for i in range(0, len(texts), max(1, batch_size)):
            chunk = texts[i : i + max(1, batch_size)]
            byte_ids, valid_mask = encode_utf8_bytes(
                chunk,
                max_bytes=self.cfg.max_input_bytes,
                patch_size=self.patch_stride,
                add_bos_eos=False,
                mask_prob=mask_prob,
                device=device,
            )
            with amp_autocast(device):
                out = self.forward(byte_ids, valid_mask)
            all_cols.append(out.columns.float())
            all_sdrs.append(out.sdr.float())
            all_recon.append(out.reconstructed_columns.float())
            all_idx.append(out.active_indices)
            all_val.append(out.active_values)
            if out.patch_columns is not None and out.patch_mask is not None:
                all_patch_cols.append(out.patch_columns.float())
                all_patch_masks.append(out.patch_mask)
            sparsity_pct = out.sparsity_pct

        if was_training:
            self.train()

        cat_patch_cols: torch.Tensor | None = None
        cat_patch_masks: torch.Tensor | None = None
        if all_patch_cols:
            max_p = max(t.shape[1] for t in all_patch_cols)
            if any(t.shape[1] != max_p for t in all_patch_cols):
                padded_pc: list[torch.Tensor] = []
                padded_pm: list[torch.Tensor] = []
                for pc, pm in zip(all_patch_cols, all_patch_masks):
                    pad_p = max_p - pc.shape[1]
                    if pad_p > 0:
                        pc = F.pad(pc, (0, 0, 0, pad_p), value=0.0)
                        pm = F.pad(pm, (0, pad_p), value=False)
                    padded_pc.append(pc)
                    padded_pm.append(pm)
                cat_patch_cols = torch.cat(padded_pc, dim=0)
                cat_patch_masks = torch.cat(padded_pm, dim=0)
            else:
                cat_patch_cols = torch.cat(all_patch_cols, dim=0)
                cat_patch_masks = torch.cat(all_patch_masks, dim=0)

        return WernickeOutput(
            columns=torch.cat(all_cols, dim=0),
            sdr=torch.cat(all_sdrs, dim=0),
            reconstructed_columns=torch.cat(all_recon, dim=0),
            active_indices=torch.cat(all_idx, dim=0),
            active_values=torch.cat(all_val, dim=0),
            sparsity_pct=sparsity_pct,
            patch_columns=cat_patch_cols,
            patch_mask=cat_patch_masks,
        )

    def export_state(self) -> dict[str, Any]:
        return {
            "weights": {k: v.detach().cpu().to(torch.bfloat16) for k, v in self.state_dict().items()},
            "num_columns": self.num_columns,
            "column_dim": self.column_dim,
            "num_neurons": self.num_neurons,
            "sdr_k": self.sdr_k,
        }

    def import_state(self, state: dict[str, Any], device: torch.device) -> None:
        saved = state.get("weights", {})
        cur = self.state_dict()
        merged = {}
        for k, v in cur.items():
            if k in saved:
                sv = saved[k]
                if sv.shape == v.shape:
                    merged[k] = sv.to(device=device, dtype=v.dtype)
                elif sv.ndim == v.ndim and sv.ndim >= 2:
                    dst = v.clone()
                    slices = tuple(slice(0, min(a, b)) for a, b in zip(sv.shape, v.shape))
                    dst[slices] = sv[slices].to(device=device, dtype=v.dtype)
                    merged[k] = dst
                else:
                    merged[k] = v
            else:
                merged[k] = v
        self.load_state_dict(merged)
