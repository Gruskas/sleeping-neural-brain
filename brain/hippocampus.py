from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .sensory_wernicke import k_wta


@dataclass
class EngramMeta:
    """
    Lightweight UI diagnostic metadata attached to a tensor engram.
    STRICT ISOLATION RULE:
      `preview_label` may ONLY be read by `/api/state` and `/api/export` for UI tabs.
      `chat`, `chat_stream`, `BiologicalAssociativeCortex`, and `BrocaByteTransformer`
      MUST NEVER read `preview_label`.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    category: str = "wiedza"
    preview_label: str = ""
    sparsity_pct: float = 4.0
    topic: str | None = None
    source_url: str | None = None
    timestamp: float = field(default_factory=time.time)
    pinned: bool = False

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "preview_label": self.preview_label,
            "sparsity_pct": round(float(self.sparsity_pct), 2),
            "topic": self.topic,
            "source_url": self.source_url,
            "timestamp": self.timestamp,
            "pinned": bool(self.pinned),
        }

    @classmethod
    def from_meta_dict(cls, data: dict[str, Any]) -> "EngramMeta":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:8]),
            category=str(data.get("category", "wiedza")),
            preview_label=str(data.get("preview_label", "")),
            sparsity_pct=float(data.get("sparsity_pct", 4.0)),
            topic=data.get("topic"),
            source_url=data.get("source_url"),
            timestamp=float(data.get("timestamp", time.time())),
            pinned=bool(data.get("pinned", False)),
        )


class TensorEngramBank:
    """
    Contiguous GPU/CPU bfloat16 engram bank:
      - `engrams`: [M, N] bfloat16 L2-normalized sparse SDR vectors
      - `strengths`, `importances`, `days_created`, `last_recalled_days`, `recall_counts`: 1D tensors
      - Retrieval runs via a single matrix multiply `engrams @ q` (zero Python loops, zero string matching).
      - Disk checkpoints serialize SDRs sparsely as `(indices int16, values bfloat16)`.
    """

    def __init__(
        self,
        num_neurons: int,
        sdr_k: int,
        max_capacity: int,
        device: torch.device,
    ) -> None:
        self.num_neurons = num_neurons
        self.sdr_k = sdr_k
        self.max_capacity = max_capacity
        self.device = device
        self.clear()

    def clear(self) -> None:
        self.engrams = torch.empty((0, self.num_neurons), dtype=torch.bfloat16, device=self.device)
        self.strengths = torch.empty((0,), dtype=torch.float32, device=self.device)
        self.importances = torch.empty((0,), dtype=torch.float32, device=self.device)
        self.days_created = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.last_recalled_days = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.recall_counts = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.pinned = torch.empty((0,), dtype=torch.bool, device=self.device)
        self.meta: list[EngramMeta] = []

    def __len__(self) -> int:
        return int(self.engrams.shape[0])

    @torch.no_grad()
    def add_batch(
        self,
        sdrs: torch.Tensor,
        metas: list[EngramMeta],
        strengths: list[float] | torch.Tensor,
        importances: list[float] | torch.Tensor,
        day: int,
    ) -> list[str]:
        if sdrs.ndim == 1:
            sdrs = sdrs.unsqueeze(0)
        m = int(sdrs.shape[0])
        if m == 0:
            return []

        norm_sdrs = F.normalize(sdrs.to(device=self.device, dtype=torch.float32), dim=-1, eps=1e-6).to(torch.bfloat16)
        st_t = (
            strengths.to(device=self.device, dtype=torch.float32)
            if isinstance(strengths, torch.Tensor)
            else torch.tensor(strengths, dtype=torch.float32, device=self.device)
        )
        imp_t = (
            importances.to(device=self.device, dtype=torch.float32)
            if isinstance(importances, torch.Tensor)
            else torch.tensor(importances, dtype=torch.float32, device=self.device)
        )
        day_t = torch.full((m,), int(day), dtype=torch.int32, device=self.device)
        rec_t = torch.zeros((m,), dtype=torch.int32, device=self.device)
        pin_t = torch.tensor([bool(item.pinned) for item in metas], dtype=torch.bool, device=self.device)

        self.engrams = torch.cat([self.engrams, norm_sdrs], dim=0)
        self.strengths = torch.cat([self.strengths, st_t], dim=0)
        self.importances = torch.cat([self.importances, imp_t], dim=0)
        self.days_created = torch.cat([self.days_created, day_t], dim=0)
        self.last_recalled_days = torch.cat([self.last_recalled_days, day_t.clone()], dim=0)
        self.recall_counts = torch.cat([self.recall_counts, rec_t], dim=0)
        self.pinned = torch.cat([self.pinned, pin_t], dim=0)
        self.meta.extend(metas)

        self._enforce_capacity()
        return [item.id for item in metas]

    @torch.no_grad()
    def _enforce_capacity(self) -> None:
        n = len(self)
        if n <= self.max_capacity:
            return
        pinned_idx = torch.nonzero(self.pinned, as_tuple=False).squeeze(-1)
        unpinned_idx = torch.nonzero(~self.pinned, as_tuple=False).squeeze(-1)
        if pinned_idx.numel() >= self.max_capacity:
            # Core identity & pinned memories are NEVER evicted
            keep_idx, _ = torch.sort(pinned_idx)
            self._slice_in_place(keep_idx)
            return
        rem_slots = self.max_capacity - int(pinned_idx.numel())
        if int(unpinned_idx.numel()) <= rem_slots:
            return
        # Retain all pinned engrams + highest-priority unpinned engrams (importance, strength, recency)
        priority = self.importances * 2.0 + self.strengths + self.last_recalled_days.float() * 0.05
        unpinned_prio = priority.index_select(0, unpinned_idx)
        top_rel = torch.topk(unpinned_prio, k=rem_slots, largest=True, sorted=False).indices
        keep_unpinned = unpinned_idx.index_select(0, top_rel)
        keep_idx = torch.cat([pinned_idx, keep_unpinned], dim=0)
        keep_idx, _ = torch.sort(keep_idx)
        self._slice_in_place(keep_idx)

    @torch.no_grad()
    def _slice_in_place(self, keep_idx: torch.Tensor) -> None:
        self.engrams = self.engrams.index_select(0, keep_idx).contiguous()
        self.strengths = self.strengths.index_select(0, keep_idx).contiguous()
        self.importances = self.importances.index_select(0, keep_idx).contiguous()
        self.days_created = self.days_created.index_select(0, keep_idx).contiguous()
        self.last_recalled_days = self.last_recalled_days.index_select(0, keep_idx).contiguous()
        self.recall_counts = self.recall_counts.index_select(0, keep_idx).contiguous()
        self.pinned = self.pinned.index_select(0, keep_idx).contiguous()
        keep_list = keep_idx.cpu().tolist()
        self.meta = [self.meta[i] for i in keep_list]

    @torch.no_grad()
    def recall_tensor(
        self,
        query_sdr: torch.Tensor,
        top_k: int,
        current_day: int,
        min_sim: float = 0.05,
    ) -> dict[str, Any]:
        """
        Pure GPU matrix-multiply recall `engrams @ q`.
        NEVER accesses or returns `preview_label`.
        """
        n = len(self)
        if n == 0 or top_k <= 0:
            return {
                "ids": [],
                "scores": [],
                "indices": [],
                "engrams": torch.empty((0, self.num_neurons), dtype=torch.bfloat16, device=self.device),
            }

        q = F.normalize(query_sdr.to(device=self.device, dtype=torch.float32).reshape(-1), dim=-1, eps=1e-6)
        q_bf16 = q.to(torch.bfloat16)
        # Single matmul on GPU: [M, N] @ [N] -> [M]
        sims = torch.mv(self.engrams, q_bf16).float()
        weighted = sims * (0.85 + 0.15 * torch.clamp(self.strengths / 3.0, min=0.2, max=1.5))

        k = min(int(top_k), n)
        vals, idxs = torch.topk(weighted, k=k, largest=True, sorted=True)
        raw_sims = sims.index_select(0, idxs)
        valid_mask = raw_sims >= min_sim
        if not valid_mask.any():
            return {
                "ids": [],
                "scores": [],
                "indices": [],
                "engrams": torch.empty((0, self.num_neurons), dtype=torch.bfloat16, device=self.device),
            }

        sel_idx = idxs[valid_mask]
        sel_scores = raw_sims[valid_mask]

        # Biological testing effect: recalling an engram strengthens it slightly
        self.recall_counts.index_add_(0, sel_idx, torch.ones_like(sel_idx, dtype=torch.int32))
        self.last_recalled_days[sel_idx] = int(current_day)
        self.strengths[sel_idx] = torch.clamp(self.strengths[sel_idx] + 0.05, max=6.0)

        idx_list = sel_idx.cpu().tolist()
        score_list = [round(float(s), 4) for s in sel_scores.cpu().tolist()]
        ids = [self.meta[i].id for i in idx_list]
        return {
            "ids": ids,
            "scores": score_list,
            "indices": idx_list,
            "engrams": self.engrams.index_select(0, sel_idx),
        }

    @torch.no_grad()
    def delete_by_id(self, engram_id: str) -> bool:
        target = engram_id.strip()
        if not target or len(self) == 0:
            return False
        keep = [i for i, m in enumerate(self.meta) if m.id != target]
        if len(keep) == len(self.meta):
            return False
        if not keep:
            self.clear()
            return True
        keep_idx = torch.tensor(keep, dtype=torch.long, device=self.device)
        self._slice_in_place(keep_idx)
        return True

    def clear_preview_labels(self, replacement: str = "") -> None:
        """Erase/overwrite diagnostic preview labels (proves chat never depends on them)."""
        for m in self.meta:
            m.preview_label = replacement

    @torch.no_grad()
    def export_sparse_tensors(self) -> dict[str, Any]:
        """Serialize SDR engrams compactly as sparse (indices int16, values bfloat16)."""
        n = len(self)
        k = min(self.sdr_k, self.num_neurons)
        if n == 0:
            return {
                "indices": torch.empty((0, k), dtype=torch.int16),
                "values": torch.empty((0, k), dtype=torch.bfloat16),
                "strengths": torch.empty((0,), dtype=torch.float32),
                "importances": torch.empty((0,), dtype=torch.float32),
                "days_created": torch.empty((0,), dtype=torch.int32),
                "last_recalled_days": torch.empty((0,), dtype=torch.int32),
                "recall_counts": torch.empty((0,), dtype=torch.int32),
                "pinned": torch.empty((0,), dtype=torch.bool),
            }
        vals, idxs = torch.topk(self.engrams.float(), k=k, dim=-1, largest=True, sorted=True)
        return {
            "indices": idxs.to(dtype=torch.int16).cpu(),
            "values": vals.to(dtype=torch.bfloat16).cpu(),
            "strengths": self.strengths.detach().cpu(),
            "importances": self.importances.detach().cpu(),
            "days_created": self.days_created.detach().cpu(),
            "last_recalled_days": self.last_recalled_days.detach().cpu(),
            "recall_counts": self.recall_counts.detach().cpu(),
            "pinned": self.pinned.detach().cpu(),
        }

    @torch.no_grad()
    def import_sparse_tensors(self, tensor_state: dict[str, Any], meta_list: list[dict[str, Any]]) -> None:
        self.clear()
        idxs: torch.Tensor = tensor_state["indices"]
        vals: torch.Tensor = tensor_state["values"]
        n = int(idxs.shape[0])
        if n == 0:
            return
        dense = torch.zeros((n, self.num_neurons), dtype=torch.bfloat16, device=self.device)
        dense.scatter_(-1, idxs.to(device=self.device, dtype=torch.long), vals.to(device=self.device, dtype=torch.bfloat16))
        self.engrams = dense
        self.strengths = tensor_state["strengths"].to(device=self.device, dtype=torch.float32)
        self.importances = tensor_state["importances"].to(device=self.device, dtype=torch.float32)
        self.days_created = tensor_state["days_created"].to(device=self.device, dtype=torch.int32)
        self.last_recalled_days = tensor_state["last_recalled_days"].to(device=self.device, dtype=torch.int32)
        self.recall_counts = tensor_state["recall_counts"].to(device=self.device, dtype=torch.int32)
        self.meta = [EngramMeta.from_meta_dict(d) for d in meta_list[:n]]
        while len(self.meta) < n:
            self.meta.append(EngramMeta())
        meta_pinned = torch.tensor([bool(m.pinned) for m in self.meta], dtype=torch.bool, device=self.device)
        if "pinned" in tensor_state and isinstance(tensor_state["pinned"], torch.Tensor) and int(tensor_state["pinned"].shape[0]) == n:
            self.pinned = tensor_state["pinned"].to(device=self.device, dtype=torch.bool) | meta_pinned
        else:
            self.pinned = meta_pinned
        for i, p_flag in enumerate(self.pinned.cpu().tolist()):
            self.meta[i].pinned = bool(p_flag)

    def to_meta_list(self) -> list[dict[str, Any]]:
        return [m.to_meta_dict() for m in self.meta]

    @torch.no_grad()
    def to_public_list(self, limit: int = 120, sort_by_strength: bool = False) -> list[dict[str, Any]]:
        """
        Diagnostic view STRICTLY for `/api/state` and `/api/export` UI tabs.
        Includes `strength`, `sparsity_pct`, and diagnostic `preview_label`.
        """
        n = len(self)
        if n == 0:
            return []
        strengths_cpu = self.strengths.detach().cpu().tolist()
        imps_cpu = self.importances.detach().cpu().tolist()
        days_cpu = self.days_created.detach().cpu().tolist()
        rec_days_cpu = self.last_recalled_days.detach().cpu().tolist()
        rec_cnt_cpu = self.recall_counts.detach().cpu().tolist()

        indices = list(range(n))
        if sort_by_strength:
            indices.sort(key=lambda i: (rec_days_cpu[i], strengths_cpu[i], imps_cpu[i]), reverse=True)
            indices = indices[:limit]
        else:
            indices = indices[-limit:]

        out: list[dict[str, Any]] = []
        for i in indices:
            m = self.meta[i]
            out.append(
                {
                    "id": m.id,
                    "category": m.category,
                    "preview_label": m.preview_label,
                    "text": m.preview_label,  # UI tab display alias (never used by chat/Broca)
                    "importance": int(round(imps_cpu[i])),
                    "strength": round(float(strengths_cpu[i]), 2),
                    "sparsity_pct": round(float(m.sparsity_pct), 2),
                    "recall_count": int(rec_cnt_cpu[i]),
                    "day_created": int(days_cpu[i]),
                    "last_recalled_day": int(rec_days_cpu[i]),
                    "topic": m.topic,
                    "source_url": m.source_url,
                    "timestamp": m.timestamp,
                    "pinned": bool(m.pinned),
                }
            )
        return out


class Hippocampus:
    """
    GPU-resident Tensor Hippocampus (bfloat16 engram banks + sparse SDR persistence):
      - `short_term` (Hipokamp): fresh wake SDR engrams before sleep consolidation
      - `semantic` (Pamięć trwała): consolidated generalized SDR prototypes
      - `rem` (REM): hybrid engrams synthesized during vectorized GPU REM dreaming
    """

    def __init__(
        self,
        num_neurons: int = 2048,
        sdr_k: int = 82,
        max_short_term_memories: int = 5000,
        max_semantic_memories: int = 50000,
        max_rem_reflections: int = 500,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_neurons = num_neurons
        self.sdr_k = sdr_k

        self.short_term = TensorEngramBank(num_neurons, sdr_k, max_short_term_memories, self.device)
        self.semantic = TensorEngramBank(num_neurons, sdr_k, max_semantic_memories, self.device)
        self.rem = TensorEngramBank(num_neurons, sdr_k, max_rem_reflections, self.device)
        self.studied_topics: dict[str, dict[str, Any]] = {}

    @torch.no_grad()
    def record_engram(
        self,
        sdr: torch.Tensor,
        preview_label: str = "",
        category: str = "wiedza",
        importance: int = 3,
        day: int = 1,
        strength: float = 1.2,
        sparsity_pct: float = 4.0,
        topic: str | None = None,
        source_url: str | None = None,
        pinned: bool = False,
    ) -> dict[str, Any]:
        meta = EngramMeta(
            category=category,
            preview_label=(preview_label or "")[:200],
            sparsity_pct=sparsity_pct,
            topic=topic,
            source_url=source_url,
            pinned=bool(pinned),
        )
        self.short_term.add_batch(
            sdrs=sdr.unsqueeze(0) if sdr.ndim == 1 else sdr[:1],
            metas=[meta],
            strengths=[float(strength)],
            importances=[float(max(1, min(5, importance)))],
            day=day,
        )
        return {
            "id": meta.id,
            "category": meta.category,
            "preview_label": meta.preview_label,
            "text": meta.preview_label,
            "importance": int(max(1, min(5, importance))),
            "strength": round(float(strength), 2),
            "sparsity_pct": round(float(sparsity_pct), 2),
            "day_created": day,
            "topic": topic,
            "source_url": source_url,
            "pinned": bool(pinned),
        }

    @torch.no_grad()
    def record_engrams_batch(
        self,
        sdrs: torch.Tensor,
        preview_labels: list[str],
        category: str = "wiedza",
        importances: list[int] | None = None,
        day: int = 1,
        sparsity_pct: float = 4.0,
        topic: str | None = None,
        source_url: str | None = None,
        pinned: bool = False,
        pinned_flags: list[bool] | None = None,
    ) -> list[str]:
        m = int(sdrs.shape[0])
        if m == 0:
            return []
        imps = [float(max(1, min(5, x))) for x in (importances or [4] * m)]
        strengths = [1.0 + 0.12 * x for x in imps]
        p_flags = [bool(x) for x in pinned_flags[:m]] if pinned_flags is not None else [bool(pinned)] * m
        while len(p_flags) < m:
            p_flags.append(bool(pinned))
        metas = [
            EngramMeta(
                category=category,
                preview_label=(lbl or "")[:200],
                sparsity_pct=sparsity_pct,
                topic=topic,
                source_url=source_url,
                pinned=p_flags[idx],
            )
            for idx, lbl in enumerate(preview_labels[:m])
        ]
        return self.short_term.add_batch(sdrs, metas, strengths, imps, day=day)

    @torch.no_grad()
    def record_rem_engrams(
        self,
        sdrs: torch.Tensor,
        decoded_labels: list[str],
        day: int = 1,
        sparsity_pct: float = 4.0,
    ) -> list[str]:
        m = int(sdrs.shape[0])
        if m == 0:
            return []
        metas = [
            EngramMeta(
                category="refleksja_rem",
                preview_label=(lbl or "")[:200],
                sparsity_pct=sparsity_pct,
            )
            for lbl in decoded_labels[:m]
        ]
        return self.rem.add_batch(
            sdrs=sdrs,
            metas=metas,
            strengths=[1.8] * m,
            importances=[4.0] * m,
            day=day,
        )

    @torch.no_grad()
    def get_hopfield_bank(self, max_engrams: int = 8192) -> torch.Tensor:
        """
        Return a single contiguous bfloat16 tensor [M_total, N] on GPU for Modern Hopfield dynamics.
        NEVER exposes any string or preview_label.
        """
        parts = [b.engrams for b in (self.short_term, self.semantic, self.rem) if len(b) > 0]
        if not parts:
            return torch.empty((0, self.num_neurons), dtype=torch.bfloat16, device=self.device)
        combined = torch.cat(parts, dim=0)
        if combined.shape[0] > max_engrams:
            combined = combined[-max_engrams:]
        return combined

    @torch.no_grad()
    def recall_all_banks(
        self,
        query_sdr: torch.Tensor,
        current_day: int,
        top_hippo: int = 12,
        top_semantic: int = 16,
        top_rem: int = 6,
        min_sim: float = 0.35,
    ) -> dict[str, dict[str, Any]]:
        """Pure tensor recall across all three banks via single GPU matmuls."""
        return {
            "hippocampus": self.short_term.recall_tensor(query_sdr, top_k=top_hippo, current_day=current_day, min_sim=min_sim),
            "semantic": self.semantic.recall_tensor(query_sdr, top_k=top_semantic, current_day=current_day, min_sim=min_sim),
            "rem": self.rem.recall_tensor(query_sdr, top_k=top_rem, current_day=current_day, min_sim=min_sim),
        }

    @torch.no_grad()
    def consolidate_nrem_light_gpu(
        self,
        merge_threshold: float = 0.86,
        decay: float = 0.96,
        prune_threshold: float = 0.18,
        current_day: int = 1,
    ) -> dict[str, Any]:
        """
        Fully vectorized GPU NREM Light stage:
          1. Compares short-term engrams against long-term semantic bank in one matmul.
          2. Merges close SDRs (>= merge_threshold) into generalized k-WTA prototypes.
          3. Promotes novel engrams to long-term semantic memory.
          4. Applies biological forgetting curve and prunes weak low-importance traces.
        """
        st_n = len(self.short_term)
        merged_count = 0
        new_count = 0
        fresh_engrams = (
            self.short_term.engrams.clone()
            if st_n > 0
            else torch.empty((0, self.num_neurons), dtype=torch.bfloat16, device=self.device)
        )

        if st_n > 0:
            # 1) Deduplicate within short_term first via GPU matmul
            st_f = self.short_term.engrams.float()
            keep_st_mask = torch.ones(st_n, dtype=torch.bool, device=self.device)
            if st_n > 1:
                self_sim = st_f @ st_f.T
                # Upper-triangular mask: if an earlier short-term trace is near-duplicate, merge into it
                # (never drop pinned short-term engrams during deduplication)
                tril_sim = torch.tril(self_sim, diagonal=-1)
                dup_vals, _ = tril_sim.max(dim=1)
                keep_st_mask = (dup_vals < merge_threshold) | self.short_term.pinned
                merged_count += int((~keep_st_mask).sum().item())

            keep_st_idx = torch.nonzero(keep_st_mask, as_tuple=False).squeeze(-1)
            cand_engrams = self.short_term.engrams.index_select(0, keep_st_idx)
            cand_strengths = self.short_term.strengths.index_select(0, keep_st_idx)
            cand_imps = self.short_term.importances.index_select(0, keep_st_idx)
            cand_pinned = self.short_term.pinned.index_select(0, keep_st_idx)
            cand_metas = [self.short_term.meta[i] for i in keep_st_idx.cpu().tolist()]

            # 2) Match candidates against long-term semantic bank in a single GPU matmul
            if len(self.semantic) > 0 and cand_engrams.shape[0] > 0:
                sim_mat = torch.matmul(cand_engrams.float(), self.semantic.engrams.float().T)  # [C, S]
                best_sim, best_sem_idx = sim_mat.max(dim=1)
                merge_mask = best_sim >= merge_threshold
                novel_mask = ~merge_mask

                if merge_mask.any():
                    m_cand = cand_engrams[merge_mask].float()
                    m_targets = best_sem_idx[merge_mask]
                    old_vecs = self.semantic.engrams.index_select(0, m_targets).float()
                    # Generalized prototype update via k-WTA
                    blended = 0.70 * old_vecs + 0.30 * m_cand
                    proto, _, _ = k_wta(blended, k=self.sdr_k, normalize=True)
                    self.semantic.engrams[m_targets] = proto.to(torch.bfloat16)
                    self.semantic.strengths[m_targets] = torch.clamp(self.semantic.strengths[m_targets] + 0.45, max=6.0)
                    self.semantic.importances[m_targets] = torch.maximum(
                        self.semantic.importances[m_targets], cand_imps[merge_mask]
                    )
                    self.semantic.last_recalled_days[m_targets] = int(current_day)
                    # Propagate pinned status onto merged long-term semantic prototypes
                    m_pin_flags = cand_pinned[merge_mask]
                    if bool(m_pin_flags.any().item()):
                        for t_i, p_f in zip(m_targets.cpu().tolist(), m_pin_flags.cpu().tolist()):
                            if p_f:
                                self.semantic.pinned[t_i] = True
                                self.semantic.meta[t_i].pinned = True
                    merged_count += int(merge_mask.sum().item())

                if novel_mask.any():
                    novel_idx = torch.nonzero(novel_mask, as_tuple=False).squeeze(-1)
                    novel_metas = [cand_metas[i] for i in novel_idx.cpu().tolist()]
                    self.semantic.add_batch(
                        sdrs=cand_engrams.index_select(0, novel_idx),
                        metas=novel_metas,
                        strengths=cand_strengths.index_select(0, novel_idx),
                        importances=cand_imps.index_select(0, novel_idx),
                        day=current_day,
                    )
                    new_count = int(novel_idx.numel())
            elif cand_engrams.shape[0] > 0:
                self.semantic.add_batch(
                    sdrs=cand_engrams,
                    metas=cand_metas,
                    strengths=cand_strengths,
                    importances=cand_imps,
                    day=current_day,
                )
                new_count = int(cand_engrams.shape[0])

        # 3) Biological forgetting curve on long-term semantic bank (vectorized GPU)
        #    Core pinned memories (pinned=True) are strictly immune to decay and forgetting!
        forgotten_count = 0
        if len(self.semantic) > 0:
            not_pinned = ~self.semantic.pinned
            decay_mask = not_pinned & (self.semantic.importances < 4.0) & (self.semantic.last_recalled_days < int(current_day))
            self.semantic.strengths = torch.where(
                decay_mask,
                self.semantic.strengths * float(decay),
                self.semantic.strengths,
            )
            forget_mask = not_pinned & (self.semantic.strengths < float(prune_threshold)) & (self.semantic.importances <= 2.0)
            forgotten_count = int(forget_mask.sum().item())
            if forgotten_count > 0:
                keep_idx = torch.nonzero(~forget_mask, as_tuple=False).squeeze(-1)
                if keep_idx.numel() == 0:
                    self.semantic.clear()
                else:
                    self.semantic._slice_in_place(keep_idx)

        return {
            "new_memories": new_count,
            "merged_memories": merged_count,
            "forgotten_memories": forgotten_count,
            "fresh_engrams": fresh_engrams,
            "semantic_count": len(self.semantic),
        }

    @torch.no_grad()
    def sample_nrem_replay_batch(
        self,
        fresh_engrams: torch.Tensor,
        batch_size: int = 48,
        fresh_ratio: float = 0.30,
    ) -> torch.Tensor:
        """
        Build a GPU Experience Replay batch mixing ~30% fresh Hipokamp engrams
        and ~70% consolidated Pamięć trwała engrams.
        """
        n_fresh = int(fresh_engrams.shape[0])
        n_sem = len(self.semantic)
        if n_fresh == 0 and n_sem == 0:
            return torch.empty((0, self.num_neurons), dtype=torch.float32, device=self.device)

        parts: list[torch.Tensor] = []
        target_fresh = min(n_fresh, max(1, int(round(batch_size * fresh_ratio)))) if n_fresh > 0 else 0
        target_sem = min(n_sem, max(0, batch_size - target_fresh)) if n_sem > 0 else 0
        if target_sem == 0 and n_fresh > target_fresh:
            target_fresh = min(n_fresh, batch_size)

        if target_fresh > 0:
            idx_f = torch.randint(0, n_fresh, (target_fresh,), device=self.device)
            parts.append(fresh_engrams.index_select(0, idx_f).float())

        if target_sem > 0:
            weights = torch.clamp(self.semantic.strengths * self.semantic.importances, min=0.1)
            probs = weights / weights.sum()
            idx_s = torch.multinomial(probs, num_samples=target_sem, replacement=(target_sem > n_sem))
            parts.append(self.semantic.engrams.index_select(0, idx_s).float())

        return torch.cat(parts, dim=0)

    def delete_memory(self, memory_id: str) -> bool:
        for bank in (self.short_term, self.semantic, self.rem):
            if bank.delete_by_id(memory_id):
                return True
        return False

    def delete_topic(self, topic: str) -> bool:
        key = topic.strip().lower()
        if not key:
            return False
        if key in self.studied_topics:
            del self.studied_topics[key]
            return True
        for k, v in list(self.studied_topics.items()):
            if str(v.get("topic", "")).strip().lower() == key:
                del self.studied_topics[k]
                return True
        return False

    def clear_short_term(self) -> None:
        self.short_term.clear()

    def reset_all_banks(self) -> None:
        """Clear all engram banks (used when re-encoding after Wernicke space changes)."""
        self.short_term.clear()
        self.semantic.clear()
        self.rem.clear()

    def clear_all_preview_labels(self, replacement: str = "") -> None:
        self.short_term.clear_preview_labels(replacement)
        self.semantic.clear_preview_labels(replacement)
        self.rem.clear_preview_labels(replacement)

    def export_sparse_tensors(self) -> dict[str, Any]:
        return {
            "short_term": self.short_term.export_sparse_tensors(),
            "semantic": self.semantic.export_sparse_tensors(),
            "rem": self.rem.export_sparse_tensors(),
        }

    def export_metadata(self) -> dict[str, Any]:
        return {
            "short_term_meta": self.short_term.to_meta_list(),
            "semantic_meta": self.semantic.to_meta_list(),
            "rem_meta": self.rem.to_meta_list(),
            "studied_topics": self.studied_topics,
        }

    def import_state(self, tensor_state: dict[str, Any], meta_state: dict[str, Any]) -> None:
        self.short_term.import_sparse_tensors(
            tensor_state["short_term"], meta_state.get("short_term_meta", [])
        )
        self.semantic.import_sparse_tensors(
            tensor_state["semantic"], meta_state.get("semantic_meta", [])
        )
        self.rem.import_sparse_tensors(
            tensor_state["rem"], meta_state.get("rem_meta", [])
        )
        self.studied_topics = dict(meta_state.get("studied_topics", {}))
