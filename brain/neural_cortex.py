from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BrainConfig
from .fp8_synapse import QuantizedSynapseMatrix, resolve_bio_synapse_mode
from .sensory_wernicke import k_wta


@dataclass
class CortexThought:
    z_star: torch.Tensor  # [B, N] stable attractor SDR after k-WTA
    columns_star: torch.Tensor  # [B, 16, D] active cortical columns for Broca
    hopfield_weights: torch.Tensor | None  # [B, M] softmax attention over engrams
    hopfield_energy: float
    attractor_steps: int
    sparsity_pct: float
    max_hopfield_sim: float = 0.0
    recalled_columns: torch.Tensor | None = None  # [B, 16, D] recalled Hippocampus columns for Broca Cross-Attention
    context_columns: torch.Tensor | None = None  # [B, M_total, D] composite cortical context Z_ctx
    context_mask: torch.Tensor | None = None  # [B, M_total] bool validity mask (True = valid)


class BiologicalAssociativeCortex(nn.Module):
    """
    Purely tensor-based Biological Associative Cortex (ZERO strings, ZERO concept dictionaries):
      - Modern Continuous Hopfield Network dynamics over stored engram bank E in R^{M x N}:
            z_{t+1} = E^T softmax(beta * E z_t)  (exponential capacity, no 0.14N Hebb ceiling)
      - Combined with local recurrent synaptic matrix W_rec (strictly zero diagonal),
        sensory projection W_in (Predictive Coding), and homeostatic thresholds.
      - Projected via lateral inhibition k-WTA onto stable sparse attractor z* and expanded to
        16 active cortical columns Z* in R^{16 x D} for BrocaByteTransformer.
      - ALL cortical weight updates (W_in, W_rec, threshold) run 100% locally in @torch.no_grad()
        with ZERO global loss.backward() through the cortex.
    """

    def __init__(self, cfg: BrainConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_neurons = cfg.num_neurons
        self.num_columns = cfg.num_columns
        self.column_dim = cfg.column_dim
        self.sensory_dim = cfg.num_columns * cfg.column_dim
        self.sdr_k = cfg.sdr_k
        self.target_sparsity = float(cfg.sdr_k) / float(cfg.num_neurons)

        # Store W_in in bfloat16 on CUDA and recurrent W_rec in native Blackwell FP8 per-row (~2x lower VRAM!)
        self.weight_dtype = (
            torch.bfloat16
            if (device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        self.bio_synapse_mode = resolve_bio_synapse_mode(
            str(getattr(cfg, "bio_synapse_dtype", "fp8")),
            device,
        )

        # Register as non-grad buffers so PyTorch NEVER backpropagates through cortical weights
        w_in = torch.randn(self.num_neurons, self.sensory_dim, device=device, dtype=torch.float32)
        w_in = F.normalize(w_in, dim=-1) * 0.25
        self.register_buffer("W_in", w_in.to(dtype=self.weight_dtype))

        # Sparse initial recurrent connectivity with strictly zero diagonal stored in QuantizedSynapseMatrix (FP8)
        w_rec = torch.randn(self.num_neurons, self.num_neurons, device=device, dtype=torch.float32) * 0.02
        sparse_mask = (torch.rand_like(w_rec) < 0.08).float()
        w_rec = w_rec * sparse_mask
        w_rec.fill_diagonal_(0.0)
        self.W_rec: QuantizedSynapseMatrix = QuantizedSynapseMatrix.from_tensor(
            w_rec,
            mode=self.bio_synapse_mode,
            device=device,
        )
        self.W_rec.fill_diagonal_(0.0)
        del w_in, w_rec, sparse_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

        self.register_buffer("threshold", torch.zeros(self.num_neurons, device=device, dtype=torch.float32))
        self.register_buffer(
            "avg_activity",
            torch.full((self.num_neurons,), self.target_sparsity, device=device, dtype=torch.float32),
        )
        self.register_buffer("last_active_sdr", torch.zeros(self.num_neurons, device=device, dtype=torch.float32))

        # Deterministic size-independent Fourier basis for projecting variable-size
        # LivingBrainEngine.state_assoc (+ top-3 M2 traces) directly into 16 cortical columns Z*.
        gen = torch.Generator(device="cpu").manual_seed(133742)
        self.register_buffer(
            "living_freq_col",
            (torch.rand(self.num_columns, generator=gen, dtype=torch.float32) * 0.45 + 0.05).to(device),
        )
        self.register_buffer(
            "living_phase_col",
            (torch.rand(self.num_columns, generator=gen, dtype=torch.float32) * 6.2831853).to(device),
        )
        self.register_buffer(
            "living_freq_dim",
            (torch.rand(self.column_dim, generator=gen, dtype=torch.float32) * 0.65 + 0.03).to(device),
        )
        self.register_buffer(
            "living_phase_dim",
            (torch.rand(self.column_dim, generator=gen, dtype=torch.float32) * 6.2831853).to(device),
        )

        # Deterministic segment embeddings for composite Z_ctx slots:
        # 0 = sensory patch columns, 1 = cortical/recalled columns, 2 = PFC working memory slots, 3 = System-2 thought slots
        seg_gen = torch.Generator(device="cpu").manual_seed(20261127)
        seg_emb = F.normalize(torch.randn(4, self.column_dim, generator=seg_gen, dtype=torch.float32), dim=-1) * 0.15
        self.register_buffer("segment_embeddings", seg_emb.to(device))

        self.last_sparsity_pct: float = round(100.0 * self.target_sparsity, 2)
        self.last_attractor_steps: int = cfg.attractor_steps
        self.last_prediction_error: float = 0.0
        self.total_local_updates: int = 0

    def total_parameters(self) -> int:
        return int(self.W_in.numel() + self.W_rec.numel() + self.threshold.numel())

    def active_synapses_count(self) -> int:
        return self.W_rec.count_nonzero()

    def synapse_memory_bytes(self) -> int:
        return self.W_rec.nbytes()

    def synapse_memory_mb(self) -> float:
        return round(self.W_rec.nbytes() / (1024.0 * 1024.0), 3)

    @torch.no_grad()
    def reset_synaptic_weights(self) -> None:
        """
        Reset stale local cortical weights (W_in, W_rec, threshold) after heavy Wernicke
        pre-training so the associative cortex re-aligns cleanly with the new Wernicke SDR space.
        """
        w_in = torch.randn(self.num_neurons, self.sensory_dim, device=self.device, dtype=torch.float32)
        self.W_in.copy_((F.normalize(w_in, dim=-1) * 0.25).to(dtype=self.W_in.dtype))
        w_rec = torch.randn(self.num_neurons, self.num_neurons, device=self.device, dtype=torch.float32) * 0.02
        sparse_mask = (torch.rand_like(w_rec) < 0.08).float()
        w_rec.mul_(sparse_mask)
        w_rec.fill_diagonal_(0.0)
        self.W_rec.copy_(w_rec)
        self.W_rec.fill_diagonal_(0.0)
        del w_in, w_rec, sparse_mask
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.threshold.zero_()
        self.avg_activity.fill_(self.target_sparsity)
        self.last_active_sdr.zero_()
        self.last_prediction_error = 0.0

    @torch.no_grad()
    def project_living_to_columns(
        self,
        living_assoc_vec: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Adaptive, size-independent projector from LivingBrainEngine's dynamically growing
        associative state vector (length L = n_assoc) into cortical column space [B, 16, D].
        Existing neuron coordinates [0..old_L-1] keep identical projection vectors across
        grow_structure() calls while newly born neurons [old_L..new_L-1] add fresh expressive signal.
        """
        v = living_assoc_vec.to(device=self.device, dtype=torch.float32)
        if v.ndim == 1:
            v = v.unsqueeze(0)
        elif v.ndim > 2:
            v = v.reshape(v.shape[0], -1)
        if v.shape[0] == 1 and batch_size > 1:
            v = v.expand(batch_size, -1)
        v = F.normalize(v, dim=-1, eps=1e-6)
        l_dim = int(v.shape[-1])
        idx = torch.arange(l_dim, device=self.device, dtype=torch.float32)
        col_basis = torch.cos(idx.unsqueeze(1) * self.living_freq_col.unsqueeze(0) + self.living_phase_col.unsqueeze(0))
        dim_basis = torch.sin(idx.unsqueeze(1) * self.living_freq_dim.unsqueeze(0) + self.living_phase_dim.unsqueeze(0))
        # [B, L, 16] -> [B, 16, L] @ [L, D] -> [B, 16, D]
        weighted = (v.unsqueeze(2) * col_basis.unsqueeze(0)).transpose(1, 2)
        cols = torch.matmul(weighted, dim_basis)
        if hasattr(F, "rms_norm"):
            return F.rms_norm(cols, (self.column_dim,), eps=1e-6)
        return cols * torch.rsqrt(cols.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def build_context_columns(
        self,
        columns_star: torch.Tensor,
        sensory_patch_columns: torch.Tensor | None = None,
        sensory_patch_mask: torch.Tensor | None = None,
        recalled_columns: torch.Tensor | None = None,
        working_memory_slots: torch.Tensor | None = None,
        thought_slots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble composite cortical context Z_ctx [B, M_total, D] with segment embeddings:
          [0..15]      primary attractor columns Z* (+ seg[1]) so Broca summary stays checkpoint-compatible
          [sensory]    active Wernicke patch columns preserving syntactic order & binding (+ seg[0])
          [recalled]   Modern Hopfield / Hippocampus recalled columns (+ seg[1])
          [wm_slots]   discrete Prefrontal Working Memory slots from recent turns (+ seg[2])
          [thoughts]   System-2 sequential latent reasoning chain slots (+ seg[3])
        """
        b = columns_star.shape[0]
        dev = columns_star.device
        dt = columns_star.dtype
        seg = self.segment_embeddings.to(device=dev, dtype=dt)

        parts: list[torch.Tensor] = [columns_star + seg[1].view(1, 1, -1)]
        masks: list[torch.Tensor] = [torch.ones((b, columns_star.shape[1]), dtype=torch.bool, device=dev)]

        if sensory_patch_columns is not None and sensory_patch_columns.numel() > 0:
            p_cols = sensory_patch_columns.to(device=dev, dtype=dt)
            if p_cols.ndim == 2:
                p_cols = p_cols.unsqueeze(0)
            if p_cols.shape[0] == 1 and b > 1:
                p_cols = p_cols.expand(b, -1, -1)
            if sensory_patch_mask is not None:
                p_mask = sensory_patch_mask.to(device=dev, dtype=torch.bool).clone()
                if p_mask.ndim == 1:
                    p_mask = p_mask.unsqueeze(0)
                if p_mask.shape[0] == 1 and b > 1:
                    p_mask = p_mask.expand(b, -1)
                p_mask[:, 0] = True
            else:
                p_mask = torch.ones((b, p_cols.shape[1]), dtype=torch.bool, device=dev)
            parts.append(p_cols + seg[0].view(1, 1, -1))
            masks.append(p_mask)

        if recalled_columns is not None and recalled_columns.numel() > 0:
            r_cols = recalled_columns.to(device=dev, dtype=dt)
            if r_cols.ndim == 2:
                r_cols = r_cols.unsqueeze(0)
            if r_cols.shape[0] == 1 and b > 1:
                r_cols = r_cols.expand(b, -1, -1)
            parts.append(r_cols + seg[1].view(1, 1, -1))
            masks.append(torch.ones((b, r_cols.shape[1]), dtype=torch.bool, device=dev))

        if working_memory_slots is not None and working_memory_slots.numel() > 0:
            wm = working_memory_slots.to(device=dev, dtype=dt)
            if wm.ndim == 2:
                wm = wm.unsqueeze(0)
            if wm.shape[0] == 1 and b > 1:
                wm = wm.expand(b, -1, -1)
            wm = F.rms_norm(wm, (self.column_dim,), eps=1e-6) if hasattr(F, "rms_norm") else (
                wm * torch.rsqrt(wm.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            )
            parts.append(wm + seg[2].view(1, 1, -1))
            masks.append(torch.ones((b, wm.shape[1]), dtype=torch.bool, device=dev))

        if thought_slots is not None and thought_slots.numel() > 0:
            th = thought_slots.to(device=dev, dtype=dt)
            if th.ndim == 2:
                th = th.unsqueeze(0)
            if th.shape[0] == 1 and b > 1:
                th = th.expand(b, -1, -1)
            th = F.rms_norm(th, (self.column_dim,), eps=1e-6) if hasattr(F, "rms_norm") else (
                th * torch.rsqrt(th.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            )
            parts.append(th + seg[3].view(1, 1, -1))
            masks.append(torch.ones((b, th.shape[1]), dtype=torch.bool, device=dev))

        z_ctx = torch.cat(parts, dim=1)
        ctx_mask = torch.cat(masks, dim=1)
        if hasattr(F, "rms_norm"):
            z_ctx = F.rms_norm(z_ctx, (self.column_dim,), eps=1e-6)
        else:
            z_ctx = z_ctx * torch.rsqrt(z_ctx.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        z_ctx = z_ctx * ctx_mask.unsqueeze(-1).to(dtype=z_ctx.dtype)
        return z_ctx, ctx_mask

    @torch.no_grad()
    def perceive_and_think(
        self,
        sensory_columns: torch.Tensor,
        sensory_sdr: torch.Tensor,
        engram_bank: torch.Tensor | None = None,
        sdr_to_columns_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        steps: int | None = None,
        noise_std: float = 0.0,
        update_last_active: bool = True,
        force_hopfield: bool = False,
        hopfield_min_sim: float = 0.82,
        working_memory_sdr: torch.Tensor | None = None,
        working_memory_weight: float = 0.0,
        living_assoc_vec: torch.Tensor | None = None,
        living_blend: float = 0.08,
        neuromod: Any | None = None,
        living_telemetry: Any | None = None,
        sensory_patch_columns: torch.Tensor | None = None,
        sensory_patch_mask: torch.Tensor | None = None,
        working_memory_slots: torch.Tensor | None = None,
        thought_slots: torch.Tensor | None = None,
    ) -> CortexThought:
        """
        Execute 3-6 attractor iterations combining:
          - Modern Continuous Hopfield Network retrieval over engram bank E [M, N]
          - Prefrontal Working Memory SDR trace & discrete turn slots
          - Local recurrent cortical synapses W_rec (zero diagonal) & homeostatic thresholds
          - Lateral inhibition k-WTA
          - Direct adaptive projection from LivingBrainEngine's growing associative state + top-3 M2 traces
          - Composite Z_ctx [B, M_total, D] assembly for Broca Cross-Attention
        """
        n_steps = max(3, min(6, int(steps if steps is not None else self.cfg.attractor_steps)))
        na_val = 0.35
        pe_val = 0.0
        cur_val = 0.0
        if neuromod is not None:
            if isinstance(neuromod, dict):
                na_val = float(neuromod.get("noradrenaline", 0.35))
            else:
                na_val = float(getattr(neuromod, "noradrenaline", 0.35))
        if living_telemetry is not None:
            pe_val = float(getattr(living_telemetry, "surprise", 0.0))
            cur_val = float(getattr(living_telemetry, "intrinsic_curiosity", 0.0))
            if neuromod is None:
                nm_dict = getattr(living_telemetry, "neuromodulators", None) or {}
                if isinstance(nm_dict, dict):
                    na_val = float(nm_dict.get("noradrenaline", na_val))
        if neuromod is not None or living_telemetry is not None:
            na_intensity = max(0.0, min(1.0, max((na_val - 0.85) / 1.15, (pe_val - 0.30) / 0.45)))
            na_steps = 3 + int(round(3.0 * na_intensity))
            n_steps = max(n_steps, min(6, na_steps))

        u = F.normalize(sensory_sdr.to(device=self.device, dtype=torch.float32), dim=-1, eps=1e-6)
        b = u.shape[0]
        s_flat = F.normalize(
            sensory_columns.to(device=self.device, dtype=torch.float32).reshape(b, self.sensory_dim),
            dim=-1,
            eps=1e-6,
        )

        wm_vec: torch.Tensor | None = None
        wm_w = max(0.0, min(0.45, float(working_memory_weight)))
        if working_memory_sdr is not None and working_memory_sdr.numel() == self.num_neurons and wm_w > 0.0:
            wm_vec = F.normalize(
                working_memory_sdr.to(device=self.device, dtype=torch.float32).reshape(1, self.num_neurons),
                dim=-1,
                eps=1e-6,
            ).expand(b, -1)

        rec_w = max(0.05, min(0.40, float(getattr(self.cfg, "cortex_rec_weight", 0.25))))
        pc_col_w = max(0.05, min(0.40, float(getattr(self.cfg, "cortex_pc_col_weight", 0.20))))
        hop_max_w = max(0.15, min(0.50, float(getattr(self.cfg, "cortex_hopfield_weight", 0.35))))
        base_u_w = max(0.20, 1.0 - rec_w - wm_w)

        # Bottom-up sensory drive via local W_in in native weight_dtype (bfloat16 Tensor Cores on GPU!)
        w_dt = self.W_in.dtype
        drive_in = F.normalize((s_flat.to(dtype=w_dt) @ self.W_in.T).float(), dim=-1, eps=1e-6)
        init_act = base_u_w * u + rec_w * drive_in - self.threshold.unsqueeze(0)
        if wm_vec is not None:
            init_act = init_act + wm_w * wm_vec
        z, _, _ = k_wta(init_act, k=self.sdr_k, normalize=True)

        has_bank = engram_bank is not None and engram_bank.numel() > 0 and engram_bank.shape[0] > 0
        e_qmat: QuantizedSynapseMatrix | None = None
        e_mat: torch.Tensor | None = None
        if has_bank:
            if isinstance(engram_bank, QuantizedSynapseMatrix):
                e_qmat = engram_bank if engram_bank.device == self.device else engram_bank.to(device=self.device)
            else:
                e_mat = F.normalize(engram_bank.to(device=self.device, dtype=torch.float32), dim=-1, eps=1e-6).to(dtype=w_dt)

        beta = float(self.cfg.hopfield_beta)
        attn_weights: torch.Tensor | None = None
        energy_t: torch.Tensor | None = None
        best_sim_t: torch.Tensor | None = None
        last_h_hop: torch.Tensor | None = None

        for step_i in range(n_steps):
            # Native Blackwell FP8 Tensor Core / blockwise BF16 matmul (zero full fp32 copy!)
            h_rec = F.normalize(self.W_rec.matmul_right_t(z), dim=-1, eps=1e-6)

            if e_qmat is not None or e_mat is not None:
                if e_qmat is not None:
                    raw_sims = e_qmat.matmul_right_t(z)  # [B, M] native FP8 / blockwise BF16
                    logits = beta * raw_sims
                    attn_weights = F.softmax(logits, dim=-1)  # [B, M]
                    h_hop = F.normalize(e_qmat.matmul_left(attn_weights), dim=-1, eps=1e-6)
                else:
                    assert e_mat is not None
                    raw_sims = (z.to(dtype=w_dt) @ e_mat.T).float()  # [B, M]
                    logits = beta * raw_sims
                    attn_weights = F.softmax(logits, dim=-1)  # [B, M]
                    h_hop = F.normalize((attn_weights.to(dtype=w_dt) @ e_mat).float(), dim=-1, eps=1e-6)
                last_h_hop = h_hop
                lse_t = torch.logsumexp(logits, dim=-1).mean()
                energy_t = -(lse_t / max(1e-6, beta)) + 0.5

                max_sim = raw_sims.max(dim=-1, keepdim=True).values
                if step_i == 0:
                    best_sim_t = max_sim.mean()

                if force_hopfield:
                    hop_gate = torch.ones((b, 1), device=self.device, dtype=torch.float32)
                    w_hop = max(hop_max_w, 0.45) * hop_gate
                else:
                    hop_gate = torch.clamp((max_sim - hopfield_min_sim) / 0.16, min=0.0, max=1.0)
                    w_hop = hop_max_w * hop_gate

                w_u = torch.clamp((1.0 - rec_w - wm_w) - w_hop, min=0.20)
                pre_act = w_u * u + w_hop * h_hop + rec_w * h_rec - self.threshold.unsqueeze(0)
            else:
                pre_act = base_u_w * u + rec_w * h_rec - self.threshold.unsqueeze(0)

            if wm_vec is not None:
                pre_act = pre_act + wm_w * wm_vec
            if noise_std > 0.0:
                pre_act = pre_act + torch.randn_like(pre_act) * noise_std

            z, _, _ = k_wta(pre_act, k=self.sdr_k, normalize=True)

        energy = float(energy_t.item()) if energy_t is not None else 0.0
        best_sim_val = float(best_sim_t.item()) if best_sim_t is not None else 0.0

        sparsity_pct = round(100.0 * (self.sdr_k / float(self.num_neurons)), 2)
        if update_last_active and b > 0:
            self.last_active_sdr.copy_(z[-1].detach())
            self.last_sparsity_pct = sparsity_pct
            self.last_attractor_steps = n_steps

        # Expand stable attractor z* [B, N] to 16 active cortical columns Z* [B, 16, D]
        z_pc_cols = F.normalize(
            (z.to(dtype=w_dt) @ self.W_in).float().view(b, self.num_columns, self.column_dim),
            dim=-1,
            eps=1e-6,
        )
        if sdr_to_columns_fn is not None:
            z_aligned = sdr_to_columns_fn(z.to(dtype=next_param_dtype(sdr_to_columns_fn, torch.float32))).float()
            raw_cols = z_aligned + pc_col_w * z_pc_cols
        else:
            raw_cols = z_pc_cols

        # Direct adaptive projection from LivingBrainEngine (state_assoc + top-3 M2 traces) into Z* (zero .item() sync!)
        if living_assoc_vec is not None and living_assoc_vec.numel() > 0 and living_blend > 0.0:
            v_norm = living_assoc_vec.detach().float().norm()
            has_v = (v_norm > 1e-6).to(dtype=raw_cols.dtype)
            living_cols = self.project_living_to_columns(living_assoc_vec, batch_size=b)
            eff_blend = float(living_blend) * (1.0 + 0.25 * min(1.0, cur_val))
            eff_blend = max(0.0, min(0.22, eff_blend))
            if hasattr(F, "rms_norm"):
                raw_normed = F.rms_norm(raw_cols, (self.column_dim,), eps=1e-6)
            else:
                raw_normed = raw_cols * torch.rsqrt(raw_cols.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            blended_cols = (1.0 - eff_blend) * raw_normed + eff_blend * living_cols
            raw_cols = raw_cols + has_v * (blended_cols - raw_cols)

        if hasattr(F, "rms_norm"):
            columns_star = F.rms_norm(raw_cols, (self.column_dim,), eps=1e-6)
        else:
            columns_star = raw_cols * torch.rsqrt(raw_cols.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        # Project strongly recalled Hippocampus engram into 16 cortical columns for Broca Cross-Attention
        recalled_columns: torch.Tensor | None = None
        if last_h_hop is not None and (force_hopfield or best_sim_val >= hopfield_min_sim):
            if sdr_to_columns_fn is not None:
                rec_raw = sdr_to_columns_fn(
                    last_h_hop.to(dtype=next_param_dtype(sdr_to_columns_fn, torch.float32))
                ).float()
            else:
                rec_raw = F.normalize(
                    (last_h_hop.to(dtype=w_dt) @ self.W_in).float().view(b, self.num_columns, self.column_dim),
                    dim=-1,
                    eps=1e-6,
                )
            if hasattr(F, "rms_norm"):
                recalled_columns = F.rms_norm(rec_raw, (self.column_dim,), eps=1e-6)
            else:
                recalled_columns = rec_raw * torch.rsqrt(rec_raw.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

        context_columns, context_mask = self.build_context_columns(
            columns_star=columns_star,
            sensory_patch_columns=sensory_patch_columns,
            sensory_patch_mask=sensory_patch_mask,
            recalled_columns=recalled_columns,
            working_memory_slots=working_memory_slots,
            thought_slots=thought_slots,
        )

        return CortexThought(
            z_star=z,
            columns_star=columns_star,
            hopfield_weights=attn_weights,
            hopfield_energy=round(float(energy), 4),
            attractor_steps=n_steps,
            sparsity_pct=sparsity_pct,
            max_hopfield_sim=round(best_sim_val, 4),
            recalled_columns=recalled_columns,
            context_columns=context_columns,
            context_mask=context_mask,
        )

    @torch.no_grad()
    def learn_local(
        self,
        sensory_columns: torch.Tensor,
        z_batch: torch.Tensor,
        steps: int = 1,
        noise_std: float = 0.04,
    ) -> dict[str, float | int]:
        """
        100% Local Biological Plasticity in @torch.no_grad() (ZERO global backprop):
          1. Predictive Coding on W_in: minimizes local reconstruction error e = s - s_hat
          2. Local Hebbian learning on W_rec (zero diagonal enforced)
          3. Threshold homeostasis tracking average firing activity vs target k/N sparsity
        """
        b = z_batch.shape[0]
        if b == 0:
            return {
                "prediction_error_start": 0.0,
                "prediction_error_end": 0.0,
                "active_synapses": self.active_synapses_count(),
            }

        s_flat = F.normalize(
            sensory_columns.to(device=self.device, dtype=torch.float32).reshape(b, self.sensory_dim),
            dim=-1,
            eps=1e-6,
        )
        z_base = F.normalize(z_batch.to(device=self.device, dtype=torch.float32), dim=-1, eps=1e-6)

        err_start_t: torch.Tensor | None = None
        err_end_t: torch.Tensor | None = None
        n_steps = max(1, int(steps))
        # Accumulate W_in Predictive Coding in fp32 and update W_rec blockwise in FP8 with stochastic rounding!
        w_in_f = self.W_in.float()

        for step_i in range(n_steps):
            if noise_std > 0.0 and step_i > 0:
                noisy = z_base + torch.randn_like(z_base) * (noise_std * 0.25)
                z, _, _ = k_wta(noisy, k=self.sdr_k, normalize=True)
            else:
                z = z_base

            # 1) Predictive Coding: local reconstruction s_hat = z @ W_in, error = s - s_hat
            s_hat = z @ w_in_f  # [B, 16*D]
            err = s_flat - s_hat
            mse_t = err.pow(2).mean()
            if step_i == 0:
                err_start_t = mse_t
            err_end_t = mse_t

            # Local Hebbian/Predictive delta: dW_in = (z^T @ err) / B with safe row-norm ceiling
            dw_in = (z.T @ err) / float(b)
            eff_pc_lr = float(self.cfg.pc_lr) * (float(self.num_neurons) / float(max(1, self.sdr_k))) * 0.15
            w_in_f.add_(dw_in * eff_pc_lr)
            row_norms = w_in_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            scale = torch.clamp(2.5 / row_norms, max=1.0)
            w_in_f.mul_(scale)

            # 2) Blockwise Local Hebbian plasticity on FP8 W_rec (zero full 12288x12288 fp32 copy!)
            self.W_rec.hebbian_update_from_sdr(
                z,
                lr=float(self.cfg.hebb_lr),
                decay=float(self.cfg.hebb_decay),
                clamp_min=-1.0,
                clamp_max=1.0,
                zero_diag=True,
            )

            # 3) Homeostatic threshold plasticity — strictly bounded to a small fraction (~15%) of
            #    an active k-WTA neuron's amplitude (1/sqrt(sdr_k) ~= 0.039) so homeostasis gently
            #    breaks ties without EVER overpowering the user's sensory SDR `u`!
            fired = (z > 0).float().mean(dim=0)
            self.avg_activity.mul_(0.90).add_(0.10 * fired)
            thr_cap = min(0.006, 0.15 / math.sqrt(float(max(1, self.sdr_k))))
            self.threshold.add_(float(self.cfg.homeostasis_lr) * 0.1 * (self.avg_activity - self.target_sparsity))
            self.threshold.clamp_(-thr_cap, thr_cap)
            self.total_local_updates += 1

        self.W_in.copy_(w_in_f.to(dtype=self.W_in.dtype))
        # Re-evaluate final clean reconstruction error after the last update (single sync at end)
        final_err_t = (s_flat - (z_base @ w_in_f)).pow(2).mean()
        err_start = float(err_start_t.item()) if err_start_t is not None else 0.0
        err_end_raw = float(err_end_t.item()) if err_end_t is not None else 0.0
        final_err = float(final_err_t.item())
        err_end = min(err_end_raw, final_err) if n_steps > 1 else final_err
        self.last_prediction_error = round(err_end, 6)

        return {
            "prediction_error_start": round(err_start, 6),
            "prediction_error_end": round(err_end, 6),
            "active_synapses": self.active_synapses_count(),
        }

    @torch.no_grad()
    def apply_synaptic_homeostasis(
        self,
        downscale: float | None = None,
        prune_eps: float | None = None,
    ) -> dict[str, int]:
        """
        Synaptic Homeostasis Hypothesis (SHY) on FP8 per-row W_rec:
          - Exact multiplicative downscaling on row_scale (`*= 0.94`)
          - Blockwise pruning of weak weights below prune_eps
          - Strictly enforce zero diagonal
        """
        factor = float(downscale if downscale is not None else self.cfg.shy_downscale)
        eps = float(prune_eps if prune_eps is not None else self.cfg.shy_prune_eps)
        return self.W_rec.apply_shy_prune(downscale=factor, eps=eps, zero_diag=True)

    @torch.no_grad()
    def compute_2d_projection(
        self,
        max_nodes: int = 52,
        max_edges: int = 85,
        engram_bank: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Project the real strongest recurrent synapses from W_rec (post-SHY) and neurons active
        after the last k-WTA onto 2D via GPU SVD/PCA for live visualization in static/index.html.
        """
        n_total = self.num_neurons

        # Select top active neurons from last k-WTA + neurons with strongest total recurrent weight
        active_mask = self.last_active_sdr > 0
        active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        if active_indices.numel() > 0:
            active_vals = self.last_active_sdr[active_indices]
            order = torch.argsort(active_vals, descending=True)
            chosen_active = active_indices[order[: min(22, active_indices.numel())]].tolist()
        else:
            chosen_active = []

        neuron_strength = self.W_rec.row_col_abs_sum()
        top_hubs = torch.topk(neuron_strength, k=min(max_nodes, n_total)).indices.tolist()

        selected: list[int] = []
        seen: set[int] = set()
        for idx in chosen_active + top_hubs:
            if idx not in seen:
                seen.add(idx)
                selected.append(int(idx))
                if len(selected) >= max_nodes:
                    break

        if not selected:
            return {"nodes": [], "edges": []}

        cur_dev = self.W_rec.device
        sel_t = torch.tensor(selected, dtype=torch.long, device=cur_dev)
        sub_w = self.W_rec.select_submatrix(sel_t, sel_t)  # [K, K]
        # Build feature matrix from synaptic profiles + engram affinity if available
        feats = self.W_in.index_select(0, sel_t).float()  # [K, sensory_dim]
        if engram_bank is not None and engram_bank.numel() > 0:
            if isinstance(engram_bank, QuantizedSynapseMatrix):
                m_eb = engram_bank.shape[0]
                row_sel = torch.arange(max(0, m_eb - 64), m_eb, device=engram_bank.device)
                e_sub = engram_bank.select_submatrix(row_sel, sel_t.to(engram_bank.device)).to(device=cur_dev).T
            else:
                e_sub = engram_bank.to(device=cur_dev, dtype=torch.float32)[:, sel_t].T  # [K, M]
                if e_sub.shape[1] > 64:
                    e_sub = e_sub[:, -64:]
            feats = torch.cat([F.normalize(feats, dim=-1), F.normalize(e_sub, dim=-1)], dim=-1)

        k_nodes = len(selected)
        if k_nodes == 1:
            coords = [[0.0, 0.0]]
        else:
            centered = feats - feats.mean(dim=0, keepdim=True)
            u, s, _ = torch.linalg.svd(centered, full_matrices=False)
            c2 = u[:, :2] * s[:2]
            if c2.shape[1] == 1:
                c2 = torch.cat([c2, torch.zeros((k_nodes, 1), device=cur_dev)], dim=1)
            max_abs = c2.abs().max().clamp(min=1e-5)
            coords = ((c2 / max_abs) * 0.90).cpu().tolist()

            # Lightweight repulsion so nodes do not stack
            min_dist = 0.14
            for _ in range(6):
                for i in range(k_nodes):
                    xi, yi = coords[i]
                    for j in range(i + 1, k_nodes):
                        xj, yj = coords[j]
                        dx, dy = xi - xj, yi - yj
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < min_dist:
                            ux, uy = (0.7071, 0.7071) if dist < 1e-6 else (dx / dist, dy / dist)
                            push = (min_dist - dist) * 0.35
                            xi = max(-0.94, min(0.94, xi + ux * push))
                            yi = max(-0.94, min(0.94, yi + uy * push))
                            coords[j][0] = max(-0.94, min(0.94, xj - ux * push))
                            coords[j][1] = max(-0.94, min(0.94, yj - uy * push))
                    coords[i][0], coords[i][1] = xi, yi

        # Extract real top edges from sub_w
        sub_abs = sub_w.abs()
        flat_vals = sub_abs.reshape(-1)
        nz = int((flat_vals > 0).sum().item())
        edge_k = min(max_edges, nz)
        edges: list[dict[str, Any]] = []
        degree_map: dict[int, int] = {}
        active_set = set(chosen_active)

        if edge_k > 0:
            _, top_e_idx = torch.topk(flat_vals, k=edge_k)
            for flat_i in top_e_idx.cpu().tolist():
                r = flat_i // k_nodes
                c = flat_i % k_nodes
                if r == c:
                    continue
                w_val = float(sub_w[r, c].item())
                if abs(w_val) < 1e-6:
                    continue
                n_src = selected[c]
                n_dst = selected[r]
                degree_map[n_src] = degree_map.get(n_src, 0) + 1
                degree_map[n_dst] = degree_map.get(n_dst, 0) + 1
                both_active = (n_src in active_set) and (n_dst in active_set)
                edges.append(
                    {
                        "src": f"N{n_src:04d}",
                        "dst": f"N{n_dst:04d}",
                        "rel": f"{w_val:+.3f}",
                        "weight": round(w_val, 4),
                        "strength": round(abs(w_val), 4),
                        "source": "ACTIVE_SYNAPSE" if both_active else "W_REC",
                    }
                )

        nodes: list[dict[str, Any]] = []
        for i, n_idx in enumerate(selected):
            act_val = float(self.last_active_sdr[n_idx].item())
            thr_val = float(self.threshold[n_idx].item())
            nodes.append(
                {
                    "id": f"N{n_idx:04d}",
                    "neuron_idx": n_idx,
                    "x": round(float(coords[i][0]), 4),
                    "y": round(float(coords[i][1]), 4),
                    "active": act_val > 0.0,
                    "activity": round(act_val, 4),
                    "threshold": round(thr_val, 4),
                    "consolidated": degree_map.get(n_idx, 0) > 0,
                    "degree": degree_map.get(n_idx, 0),
                }
            )

        return {"nodes": nodes, "edges": edges}

    def export_state(self) -> dict[str, Any]:
        return {
            "num_neurons": self.num_neurons,
            "sensory_dim": self.sensory_dim,
            "sdr_k": self.sdr_k,
            "bio_synapse_mode": self.bio_synapse_mode,
            "W_in": self.W_in.detach().cpu().to(torch.bfloat16),
            "W_rec": self.W_rec.export_state(),
            "threshold": self.threshold.detach().cpu().float(),
            "avg_activity": self.avg_activity.detach().cpu().float(),
            "last_active_sdr": self.last_active_sdr.detach().cpu().to(torch.bfloat16),
            "last_prediction_error": self.last_prediction_error,
            "total_local_updates": self.total_local_updates,
        }

    def import_state(self, state: dict[str, Any]) -> None:
        if state.get("num_neurons") != self.num_neurons or state.get("sensory_dim") != self.sensory_dim:
            raise ValueError("Incompatible BiologicalAssociativeCortex dimensions in checkpoint")
        self.W_in.copy_(state["W_in"].to(device=self.device, dtype=self.W_in.dtype))
        # Seamlessly migrate both FP8/INT8 state dicts AND legacy bf16/fp32 W_rec tensors to FP8!
        self.W_rec = QuantizedSynapseMatrix.from_state(
            state["W_rec"],
            device=self.device,
            mode=self.bio_synapse_mode,
        )
        self.W_rec.fill_diagonal_(0.0)
        thr_cap = min(0.006, 0.15 / math.sqrt(float(max(1, self.sdr_k))))
        self.threshold.copy_(
            state["threshold"].to(device=self.device, dtype=torch.float32).clamp(-thr_cap, thr_cap)
        )
        self.avg_activity.copy_(state["avg_activity"].to(device=self.device, dtype=torch.float32))
        if "last_active_sdr" in state:
            self.last_active_sdr.copy_(state["last_active_sdr"].to(device=self.device, dtype=torch.float32))
        if "segment_embeddings" in state and isinstance(state["segment_embeddings"], torch.Tensor):
            if state["segment_embeddings"].shape == self.segment_embeddings.shape:
                self.segment_embeddings.copy_(state["segment_embeddings"].to(device=self.device, dtype=torch.float32))
        self.last_prediction_error = float(state.get("last_prediction_error", 0.0))
        self.total_local_updates = int(state.get("total_local_updates", 0))


def next_param_dtype(fn: Any, fallback: torch.dtype) -> torch.dtype:
    owner = getattr(fn, "__self__", None)
    if isinstance(owner, nn.Module):
        for p in owner.parameters():
            return p.dtype
    return fallback
