from __future__ import annotations

from collections import OrderedDict
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .fp8_synapse import QuantizedSynapseMatrix, resolve_bio_synapse_mode
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
    parent_ids: list[str] = field(default_factory=list)
    tag: str | None = None

    def to_meta_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "preview_label": self.preview_label,
            "sparsity_pct": round(float(self.sparsity_pct), 2),
            "topic": self.topic,
            "source_url": self.source_url,
            "timestamp": self.timestamp,
            "pinned": bool(self.pinned),
        }
        if self.parent_ids:
            d["parent_ids"] = list(self.parent_ids)
        if self.tag:
            d["tag"] = self.tag
        return d

    @classmethod
    def from_meta_dict(cls, data: dict[str, Any]) -> "EngramMeta":
        raw_parents = data.get("parent_ids")
        parents = [str(x) for x in raw_parents] if isinstance(raw_parents, (list, tuple)) else []
        raw_tag = data.get("tag")
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:8]),
            category=str(data.get("category", "wiedza")),
            preview_label=str(data.get("preview_label", "")),
            sparsity_pct=float(data.get("sparsity_pct", 4.0)),
            topic=data.get("topic"),
            source_url=data.get("source_url"),
            timestamp=float(data.get("timestamp", time.time())),
            pinned=bool(data.get("pinned", False)),
            parent_ids=parents,
            tag=str(raw_tag) if raw_tag else None,
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
        bio_synapse_dtype: str = "fp8",
    ) -> None:
        self.num_neurons = num_neurons
        self.sdr_k = sdr_k
        self.max_capacity = max_capacity
        self.device = device
        self.bio_synapse_mode = resolve_bio_synapse_mode(bio_synapse_dtype, device)
        self.max_explored_pairs = 4096
        self.explored_pairs: OrderedDict[tuple[str, str], None] = OrderedDict()
        self.clear()

    def clear(self) -> None:
        self.engrams = QuantizedSynapseMatrix.empty(0, self.num_neurons, self.device, mode=self.bio_synapse_mode)
        self.strengths = torch.empty((0,), dtype=torch.float32, device=self.device)
        self.importances = torch.empty((0,), dtype=torch.float32, device=self.device)
        self.days_created = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.last_recalled_days = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.recall_counts = torch.empty((0,), dtype=torch.int32, device=self.device)
        self.pinned = torch.empty((0,), dtype=torch.bool, device=self.device)
        self.meta: list[EngramMeta] = []
        self.explored_pairs.clear()

    def memory_bytes(self) -> int:
        total = self.engrams.nbytes()
        for t in (self.strengths, self.importances, self.days_created, self.last_recalled_days, self.recall_counts, self.pinned):
            total += int(t.numel() * t.element_size())
        return total

    def memory_mb(self) -> float:
        return round(self.memory_bytes() / (1024.0 * 1024.0), 3)

    @staticmethod
    def _canonical_pair(id_a: str, id_c: str) -> tuple[str, str]:
        return (id_a, id_c) if id_a <= id_c else (id_c, id_a)

    def mark_pair_explored(self, id_a: str, id_c: str) -> None:
        key = self._canonical_pair(id_a, id_c)
        if key in self.explored_pairs:
            self.explored_pairs.move_to_end(key)
        else:
            self.explored_pairs[key] = None
            while len(self.explored_pairs) > self.max_explored_pairs:
                self.explored_pairs.popitem(last=False)

    def is_pair_explored(self, id_a: str, id_c: str) -> bool:
        return self._canonical_pair(id_a, id_c) in self.explored_pairs

    def evict_oldest_explored_pairs(self, frac: float = 0.25) -> int:
        """Evict the oldest `frac` (default 25%) entries from `explored_pairs` LRU."""
        if not self.explored_pairs:
            return 0
        n_evict = max(1, int(len(self.explored_pairs) * max(0.0, min(1.0, float(frac)))))
        evicted = 0
        for _ in range(min(n_evict, len(self.explored_pairs))):
            self.explored_pairs.popitem(last=False)
            evicted += 1
        return evicted

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
        if isinstance(sdrs, QuantizedSynapseMatrix):
            sdrs_f = sdrs.float().to(device=self.device)
        else:
            if sdrs.ndim == 1:
                sdrs = sdrs.unsqueeze(0)
            sdrs_f = sdrs.to(device=self.device, dtype=torch.float32)
        m = int(sdrs_f.shape[0])
        if m == 0:
            return []

        norm_sdrs = F.normalize(sdrs_f, dim=-1, eps=1e-6)
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

        self.engrams = self.engrams.cat_rows(norm_sdrs)
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
        # Retain all pinned engrams + highest-priority unpinned engrams by importance & strength (age-independent)
        priority = self.importances * 2.0 + self.strengths
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
        # Native Blackwell FP8 Tensor Core / blockwise BF16 matvec (zero full fp32 copy!)
        sims = self.engrams.matvec(q)
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
            "engrams": self.engrams.dequant_rows(sel_idx, out_dtype=torch.bfloat16),
        }

    @torch.no_grad()
    def sample_2hop_bridge(
        self,
        min_hop_sim: float = 0.45,
        max_direct_sim: float = 0.25,
        max_attempts: int = 16,
        recent_days_window: int = 3,
        current_day: int | None = None,
        mark_explored: bool = True,
    ) -> dict[str, Any] | None:
        """
        GPU 2-hop Concept Graph discovery (A -> B -> C):
          - Sample start node A (weighted by importance, with optional boost for recent days).
          - Find strong 1-hop neighbor B: sim(A, B) >= min_hop_sim (default 0.45), B != A.
          - Find 2-hop target C: sim(B, C) >= min_hop_sim, C not in {A, B}, AND distant from A:
            sim(A, C) <= max_direct_sim (default 0.25), preferring different day / context when available.
          - Deduplicate unordered (id_A, id_C) via LRU `explored_pairs` so the brain explores novel bridges.
        """
        n = len(self)
        if n < 3:
            return None

        # Importance-weighted start sampling with optional boost for recent-day engrams
        weights = self.importances.clamp(min=0.5).float().clone()
        if current_day is not None and recent_days_window > 0:
            recent_mask = self.days_created >= (int(current_day) - int(recent_days_window))
            if recent_mask.any() and random.random() < 0.55:
                weights = torch.where(recent_mask, weights * 2.0, weights)
        k_starts = min(max(1, int(max_attempts)), n)
        start_indices = torch.multinomial(weights, num_samples=k_starts, replacement=False).cpu().tolist()

        def _pack_node(idx: int) -> dict[str, Any]:
            m = self.meta[idx]
            return {
                "index": int(idx),
                "id": m.id,
                "sdr": self.engrams.dequant_row(idx, out_dtype=torch.bfloat16).detach().clone(),
                "day": int(self.days_created[idx].item()),
                "importance": float(self.importances[idx].item()),
                "topic": m.topic,
                "category": m.category,
                "preview_label": m.preview_label,  # ONLY for logging/UI dedup, NEVER input to Broca/Cortex
            }

        for idx_a in start_indices:
            sdr_a = self.engrams.dequant_row(idx_a, out_dtype=torch.float32)
            sim_a = self.engrams.matvec(sdr_a)
            sim_a[idx_a] = -1.0

            b_cand_idx = torch.nonzero(sim_a >= float(min_hop_sim), as_tuple=False).squeeze(-1)
            if b_cand_idx.numel() == 0:
                continue

            # Inspect up to 8 strongest 1-hop neighbors B (shuffled slightly for diversity)
            b_sims = sim_a.index_select(0, b_cand_idx)
            kb = min(8, int(b_cand_idx.numel()))
            top_b_rel = torch.topk(b_sims, k=kb, largest=True, sorted=True).indices
            b_indices = b_cand_idx.index_select(0, top_b_rel).cpu().tolist()
            if len(b_indices) > 2:
                head, tail = b_indices[:2], b_indices[2:]
                random.shuffle(tail)
                b_indices = head + tail

            day_a = int(self.days_created[idx_a].item())
            meta_a = self.meta[idx_a]
            ctx_a = (meta_a.topic or meta_a.category or "").strip().lower()
            diff_day_vec = (self.days_created != day_a).float()

            for idx_b in b_indices:
                sdr_b = self.engrams.dequant_row(idx_b, out_dtype=torch.float32)
                sim_b = self.engrams.matvec(sdr_b)
                c_mask = (sim_b >= float(min_hop_sim)) & (sim_a <= float(max_direct_sim))
                c_mask[idx_a] = False
                c_mask[idx_b] = False
                c_cand_idx = torch.nonzero(c_mask, as_tuple=False).squeeze(-1)
                if c_cand_idx.numel() == 0:
                    continue

                # Rank C candidates: strong B->C bridge + preference for different day & low A-C direct sim
                c_sim_b = sim_b.index_select(0, c_cand_idx)
                c_sim_a = sim_a.index_select(0, c_cand_idx)
                c_diff_d = diff_day_vec.index_select(0, c_cand_idx)
                c_scores = c_sim_b + 0.35 * c_diff_d + 0.15 * (1.0 - c_sim_a.clamp(min=0.0))

                kc = min(24, int(c_cand_idx.numel()))
                top_c_rel = torch.topk(c_scores, k=kc, largest=True, sorted=True).indices
                c_indices = c_cand_idx.index_select(0, top_c_rel).cpu().tolist()
                c_score_vals = c_scores.index_select(0, top_c_rel).cpu().tolist()

                # Tie-break in favor of different topic/context when available
                ranked_c: list[tuple[float, int]] = []
                for idx_c, sc in zip(c_indices, c_score_vals):
                    meta_c = self.meta[idx_c]
                    ctx_c = (meta_c.topic or meta_c.category or "").strip().lower()
                    ctx_bonus = 0.20 if (ctx_a and ctx_c and ctx_a != ctx_c) else 0.0
                    ranked_c.append((float(sc) + ctx_bonus, int(idx_c)))
                ranked_c.sort(key=lambda item: item[0], reverse=True)

                for _, idx_c in ranked_c:
                    id_c = self.meta[idx_c].id
                    if self.is_pair_explored(meta_a.id, id_c):
                        continue
                    if mark_explored:
                        self.mark_pair_explored(meta_a.id, id_c)
                    node_a = _pack_node(idx_a)
                    node_b = _pack_node(idx_b)
                    node_c = _pack_node(idx_c)
                    ctx_c = (self.meta[idx_c].topic or self.meta[idx_c].category or "").strip().lower()
                    return {
                        "a": node_a,
                        "b": node_b,
                        "c": node_c,
                        "sim_ab": round(float(sim_a[idx_b].item()), 4),
                        "sim_bc": round(float(sim_b[idx_c].item()), 4),
                        "sim_ac": round(float(sim_a[idx_c].item()), 4),
                        "cross_day": bool(node_a["day"] != node_c["day"]),
                        "cross_context": bool(ctx_a != ctx_c),
                    }

        return None

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
        # Blockwise top-k export (zero full fp32 copy!)
        idxs_chunks: list[torch.Tensor] = []
        vals_chunks: list[torch.Tensor] = []
        block = 1024
        for r0 in range(0, n, block):
            r1 = min(n, r0 + block)
            r_idx = torch.arange(r0, r1, device=self.device)
            sub_f = self.engrams.dequant_rows(r_idx, out_dtype=torch.float32)
            v_b, i_b = torch.topk(sub_f, k=k, dim=-1, largest=True, sorted=True)
            idxs_chunks.append(i_b.to(dtype=torch.int16).cpu())
            vals_chunks.append(v_b.to(dtype=torch.bfloat16).cpu())
        return {
            "indices": torch.cat(idxs_chunks, dim=0),
            "values": torch.cat(vals_chunks, dim=0),
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
        # Blockwise scatter + direct FP8 per-row quantization (seamless migration of legacy bf16 checkpoints!)
        block = 1024
        q_bank = QuantizedSynapseMatrix.empty(0, self.num_neurons, self.device, mode=self.bio_synapse_mode)
        for r0 in range(0, n, block):
            r1 = min(n, r0 + block)
            dense_blk = torch.zeros((r1 - r0, self.num_neurons), dtype=torch.float32, device=self.device)
            dense_blk.scatter_(
                -1,
                idxs[r0:r1].to(device=self.device, dtype=torch.long),
                vals[r0:r1].to(device=self.device, dtype=torch.float32),
            )
            q_bank = q_bank.cat_rows(dense_blk)
        self.engrams = q_bank
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
                    "parent_ids": list(m.parent_ids),
                    "tag": m.tag,
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
        bio_synapse_dtype: str = "fp8",
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_neurons = num_neurons
        self.sdr_k = sdr_k
        self.bio_synapse_mode = resolve_bio_synapse_mode(bio_synapse_dtype, self.device)

        self.short_term = TensorEngramBank(num_neurons, sdr_k, max_short_term_memories, self.device, bio_synapse_dtype=bio_synapse_dtype)
        self.semantic = TensorEngramBank(num_neurons, sdr_k, max_semantic_memories, self.device, bio_synapse_dtype=bio_synapse_dtype)
        self.rem = TensorEngramBank(num_neurons, sdr_k, max_rem_reflections, self.device, bio_synapse_dtype=bio_synapse_dtype)
        self.studied_topics: dict[str, dict[str, Any]] = {}

    def memory_bytes(self) -> int:
        return self.short_term.memory_bytes() + self.semantic.memory_bytes() + self.rem.memory_bytes()

    def memory_mb(self) -> float:
        return round(self.memory_bytes() / (1024.0 * 1024.0), 3)

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
        parent_ids: list[str] | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        cat_norm = (category or "wiedza").strip().lower()
        is_pinned = bool(pinned) or (cat_norm in {"zasada", "tożsamość", "tozsamosc", "fundament"})
        meta = EngramMeta(
            category=category,
            preview_label=(preview_label or "")[:200],
            sparsity_pct=sparsity_pct,
            topic=topic,
            source_url=source_url,
            pinned=is_pinned,
            parent_ids=list(parent_ids or []),
            tag=tag,
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
            "pinned": is_pinned,
            "parent_ids": list(meta.parent_ids),
            "tag": meta.tag,
        }

    @torch.no_grad()
    def record_2hop_engram(
        self,
        sdr: torch.Tensor,
        thought_text: str,
        parent_ids: list[str],
        tag: str = "dmn_2hop",
        importance: float = 3.0,
        day: int = 1,
        strength: float = 1.5,
        sparsity_pct: float = 4.0,
        target_bank: str | None = None,
    ) -> dict[str, Any]:
        """
        Store an accepted 2-hop synthesized thought engram with parent IDs [A, B, C],
        tag (`dmn_2hop` / `rem_2hop`), pinned=False, and moderate importance (~3.0).
        """
        bank_name = target_bank or ("rem" if tag == "rem_2hop" else "short_term")
        category = "refleksja_rem" if bank_name == "rem" else "mysl_dmn"
        meta = EngramMeta(
            category=category,
            preview_label=(thought_text or "")[:200],
            sparsity_pct=sparsity_pct,
            pinned=False,
            parent_ids=[str(x) for x in parent_ids],
            tag=str(tag),
        )
        bank = self.rem if bank_name == "rem" else (self.semantic if bank_name == "semantic" else self.short_term)
        imp_clamped = float(max(1.0, min(5.0, float(importance))))
        bank.add_batch(
            sdrs=sdr.unsqueeze(0) if sdr.ndim == 1 else sdr[:1],
            metas=[meta],
            strengths=[float(strength)],
            importances=[imp_clamped],
            day=day,
        )
        return {
            "id": meta.id,
            "bank": bank_name,
            "category": meta.category,
            "preview_label": meta.preview_label,
            "importance": round(imp_clamped, 2),
            "strength": round(float(strength), 2),
            "day_created": day,
            "pinned": False,
            "parent_ids": list(meta.parent_ids),
            "tag": meta.tag,
        }

    @torch.no_grad()
    def _sample_combined_2hop_bridge(
        self,
        banks: list[TensorEngramBank],
        min_hop_sim: float,
        max_direct_sim: float,
        max_attempts: int,
        recent_days_window: int,
        current_day: int | None,
        mark_explored: bool,
    ) -> dict[str, Any] | None:
        non_empty = [b for b in banks if len(b) > 0]
        total_n = sum(len(b) for b in non_empty)
        if total_n < 3:
            return None

        sdr_chunks: list[torch.Tensor] = []
        day_chunks: list[torch.Tensor] = []
        imp_chunks: list[torch.Tensor] = []
        meta_all: list[EngramMeta] = []
        for b in non_empty:
            s = b.engrams.dequant_all(out_dtype=torch.float32)
            s = F.normalize(s, dim=-1, eps=1e-6)
            sdr_chunks.append(s)
            day_chunks.append(b.days_created.to(device=s.device))
            imp_chunks.append(b.importances.to(device=s.device))
            meta_all.extend(b.meta)

        e_mat = torch.cat(sdr_chunks, dim=0)
        days_all = torch.cat(day_chunks, dim=0)
        imp_all = torch.cat(imp_chunks, dim=0)
        n = int(e_mat.shape[0])
        if n > 4096:
            keep_idx = torch.topk(imp_all, k=4096, largest=True).indices
            e_mat = e_mat.index_select(0, keep_idx)
            days_all = days_all.index_select(0, keep_idx)
            imp_all = imp_all.index_select(0, keep_idx)
            keep_list = keep_idx.cpu().tolist()
            meta_all = [meta_all[i] for i in keep_list]
            n = len(meta_all)

        weights = imp_all.clamp(min=0.5).float().clone()
        if current_day is not None and recent_days_window > 0:
            recent_mask = days_all >= (int(current_day) - int(recent_days_window))
            if recent_mask.any() and random.random() < 0.55:
                weights = torch.where(recent_mask, weights * 2.0, weights)
        k_starts = min(max(1, int(max_attempts)), n)
        start_indices = torch.multinomial(weights, num_samples=k_starts, replacement=False).cpu().tolist()

        def _pack_node(idx: int) -> dict[str, Any]:
            m = meta_all[idx]
            return {
                "index": int(idx),
                "id": m.id,
                "sdr": e_mat[idx].to(dtype=torch.bfloat16).detach().clone(),
                "day": int(days_all[idx].item()),
                "importance": float(imp_all[idx].item()),
                "topic": m.topic,
                "category": m.category,
                "preview_label": m.preview_label,
            }

        def _build_res(idx_a: int, idx_b: int, idx_c: int, sim_ab: float, sim_bc: float, sim_ac: float) -> dict[str, Any]:
            if mark_explored:
                self.semantic.mark_pair_explored(meta_all[idx_a].id, meta_all[idx_c].id)
            node_a = _pack_node(idx_a)
            node_b = _pack_node(idx_b)
            node_c = _pack_node(idx_c)
            ctx_a = (meta_all[idx_a].topic or meta_all[idx_a].category or "").strip().lower()
            ctx_c = (meta_all[idx_c].topic or meta_all[idx_c].category or "").strip().lower()
            return {
                "a": node_a,
                "b": node_b,
                "c": node_c,
                "sim_ab": round(sim_ab, 4),
                "sim_bc": round(sim_bc, 4),
                "sim_ac": round(sim_ac, 4),
                "cross_day": bool(node_a["day"] != node_c["day"]),
                "cross_context": bool(ctx_a != ctx_c),
            }

        # Stage A: try relaxed thresholds across combined banks
        for idx_a in start_indices:
            sim_a = e_mat @ e_mat[idx_a]
            sim_a[idx_a] = -1.0
            b_cand_idx = torch.nonzero(sim_a >= float(min_hop_sim), as_tuple=False).squeeze(-1)
            if b_cand_idx.numel() == 0:
                continue
            b_sims = sim_a.index_select(0, b_cand_idx)
            kb = min(8, int(b_cand_idx.numel()))
            top_b_rel = torch.topk(b_sims, k=kb, largest=True, sorted=True).indices
            b_indices = b_cand_idx.index_select(0, top_b_rel).cpu().tolist()
            day_a = int(days_all[idx_a].item())
            meta_a = meta_all[idx_a]
            ctx_a = (meta_a.topic or meta_a.category or "").strip().lower()
            diff_day_vec = (days_all != day_a).float()

            for idx_b in b_indices:
                sim_b = e_mat @ e_mat[idx_b]
                c_mask = (sim_b >= float(min_hop_sim)) & (sim_a <= float(max_direct_sim))
                c_mask[idx_a] = False
                c_mask[idx_b] = False
                c_cand_idx = torch.nonzero(c_mask, as_tuple=False).squeeze(-1)
                if c_cand_idx.numel() == 0:
                    continue
                c_sim_b = sim_b.index_select(0, c_cand_idx)
                c_sim_a = sim_a.index_select(0, c_cand_idx)
                c_diff_d = diff_day_vec.index_select(0, c_cand_idx)
                c_scores = c_sim_b + 0.35 * c_diff_d + 0.15 * (1.0 - c_sim_a.clamp(min=0.0))
                kc = min(24, int(c_cand_idx.numel()))
                top_c_rel = torch.topk(c_scores, k=kc, largest=True, sorted=True).indices
                c_indices = c_cand_idx.index_select(0, top_c_rel).cpu().tolist()
                c_score_vals = c_scores.index_select(0, top_c_rel).cpu().tolist()
                ranked_c: list[tuple[float, int]] = []
                for idx_c, sc in zip(c_indices, c_score_vals):
                    meta_c = meta_all[idx_c]
                    ctx_c = (meta_c.topic or meta_c.category or "").strip().lower()
                    ctx_bonus = 0.20 if (ctx_a and ctx_c and ctx_a != ctx_c) else 0.0
                    ranked_c.append((float(sc) + ctx_bonus, int(idx_c)))
                ranked_c.sort(key=lambda item: item[0], reverse=True)
                for _, idx_c in ranked_c:
                    if self.semantic.is_pair_explored(meta_a.id, meta_all[idx_c].id):
                        continue
                    return _build_res(
                        idx_a, idx_b, idx_c,
                        float(sim_a[idx_b].item()), float(sim_b[idx_c].item()), float(sim_a[idx_c].item()),
                    )

        # Stage B: ultimate distant-pair + best-mediator fallback so DMN never starves when >= 3 engrams exist
        best_ac_score = -1e9
        best_pair: tuple[int, int] | None = None
        for idx_a in start_indices:
            sim_a = e_mat @ e_mat[idx_a]
            imp_a = float(imp_all[idx_a].item())
            pair_scores = (imp_a + imp_all) * 0.15 + (1.0 - sim_a.clamp(min=-1.0))
            pair_scores[idx_a] = -1e9
            top_c_idx = torch.topk(pair_scores, k=min(n - 1, 32), largest=True).indices.cpu().tolist()
            for idx_c in top_c_idx:
                if idx_c == idx_a or self.semantic.is_pair_explored(meta_all[idx_a].id, meta_all[idx_c].id):
                    continue
                sc = float(pair_scores[idx_c].item())
                if sc > best_ac_score:
                    best_ac_score = sc
                    best_pair = (idx_a, idx_c)
        if best_pair is None:
            return None
        idx_a, idx_c = best_pair
        sim_a = e_mat @ e_mat[idx_a]
        sim_c = e_mat @ e_mat[idx_c]
        bridge_scores = sim_a + sim_c
        bridge_scores[idx_a] = -1e9
        bridge_scores[idx_c] = -1e9
        idx_b = int(torch.argmax(bridge_scores).item())
        return _build_res(
            idx_a, idx_b, idx_c,
            float(sim_a[idx_b].item()), float(sim_c[idx_b].item()), float(sim_a[idx_c].item()),
        )

    @torch.no_grad()
    def sample_2hop_bridge(
        self,
        min_hop_sim: float = 0.45,
        max_direct_sim: float = 0.25,
        max_attempts: int = 16,
        recent_days_window: int = 3,
        current_day: int | None = None,
        mark_explored: bool = True,
        include_short_term: bool = True,
        relaxed_min_hop_sim: float = 0.22,
        relaxed_max_direct_sim: float = 0.45,
        allow_fallback: bool = True,
    ) -> dict[str, Any] | None:
        """
        Discover a novel 2-hop conceptual bridge A -> B -> C on GPU:
          1. Strict search over `semantic`.
          2. If `semantic` has < 3 engrams or strict search finds nothing (and `allow_fallback=True`),
             combine `semantic` + `short_term` (+ `rem` if needed) with relaxed thresholds and an
             ultimate distant-pair + best-mediator fallback. Evicts oldest 25% of `explored_pairs` if exhausted.
        """
        strict = self.semantic.sample_2hop_bridge(
            min_hop_sim=min_hop_sim,
            max_direct_sim=max_direct_sim,
            max_attempts=max_attempts,
            recent_days_window=recent_days_window,
            current_day=current_day,
            mark_explored=mark_explored,
        )
        if strict is not None or not allow_fallback:
            return strict

        banks: list[TensorEngramBank] = [self.semantic]
        if include_short_term:
            banks.append(self.short_term)
        if sum(len(b) for b in banks) < 3 or len(self.rem) > 0:
            banks.append(self.rem)
        if sum(len(b) for b in banks) < 3:
            return None

        res = self._sample_combined_2hop_bridge(
            banks=banks,
            min_hop_sim=relaxed_min_hop_sim,
            max_direct_sim=relaxed_max_direct_sim,
            max_attempts=max_attempts,
            recent_days_window=recent_days_window,
            current_day=current_day,
            mark_explored=mark_explored,
        )
        if res is None and len(self.semantic.explored_pairs) > 0:
            self.semantic.evict_oldest_explored_pairs(0.25)
            res = self._sample_combined_2hop_bridge(
                banks=banks,
                min_hop_sim=relaxed_min_hop_sim,
                max_direct_sim=relaxed_max_direct_sim,
                max_attempts=max_attempts,
                recent_days_window=recent_days_window,
                current_day=current_day,
                mark_explored=mark_explored,
            )
        return res

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
    def get_hopfield_bank(self, max_engrams: int = 8192) -> QuantizedSynapseMatrix:
        """
        Return a single contiguous FP8 per-row QuantizedSynapseMatrix [M_total, N] on GPU
        for Modern Hopfield dynamics (2x lower VRAM, zero full fp32 copy!).
        NEVER exposes any string or preview_label.
        """
        combined = QuantizedSynapseMatrix.empty(0, self.num_neurons, self.device, mode=self.bio_synapse_mode)
        for bank in (self.short_term, self.semantic, self.rem):
            if len(bank) > 0:
                combined = combined.cat_rows(bank.engrams)
        if combined.shape[0] > max_engrams:
            idx = torch.arange(combined.shape[0] - max_engrams, combined.shape[0], device=self.device)
            combined = combined.select_rows(idx)
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
            else QuantizedSynapseMatrix.empty(0, self.num_neurons, self.device, mode=self.bio_synapse_mode)
        )

        if st_n > 0:
            # 1) Deduplicate within short_term first via GPU matmul
            st_f = self.short_term.engrams.float()
            keep_st_mask = torch.ones(st_n, dtype=torch.bool, device=self.device)
            if st_n > 1:
                self_sim = st_f @ st_f.T
                tril_sim = torch.tril(self_sim, diagonal=-1)
                dup_vals, _ = tril_sim.max(dim=1)
                keep_st_mask = (dup_vals < merge_threshold) | self.short_term.pinned
                merged_count += int((~keep_st_mask).sum().item())

            keep_st_idx = torch.nonzero(keep_st_mask, as_tuple=False).squeeze(-1)
            cand_engrams_f = st_f.index_select(0, keep_st_idx)
            cand_strengths = self.short_term.strengths.index_select(0, keep_st_idx)
            cand_imps = self.short_term.importances.index_select(0, keep_st_idx)
            cand_pinned = self.short_term.pinned.index_select(0, keep_st_idx)
            cand_metas = [self.short_term.meta[i] for i in keep_st_idx.cpu().tolist()]

            # 2) Match candidates against long-term semantic bank in FP8 (zero full fp32 copy!)
            if len(self.semantic) > 0 and cand_engrams_f.shape[0] > 0:
                sim_mat = self.semantic.engrams.matmul_right_t(cand_engrams_f)  # [C, S]
                best_sim, best_sem_idx = sim_mat.max(dim=1)
                merge_mask = best_sim >= merge_threshold
                novel_mask = ~merge_mask

                if merge_mask.any():
                    m_cand = cand_engrams_f[merge_mask]
                    m_targets = best_sem_idx[merge_mask]
                    old_vecs = self.semantic.engrams.dequant_rows(m_targets, out_dtype=torch.float32)
                    # Generalized prototype update via k-WTA -> re-quantized into FP8 rows
                    blended = 0.70 * old_vecs + 0.30 * m_cand
                    proto, _, _ = k_wta(blended, k=self.sdr_k, normalize=True)
                    self.semantic.engrams.set_rows(m_targets, proto, stochastic=False)
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
                        sdrs=cand_engrams_f.index_select(0, novel_idx),
                        metas=novel_metas,
                        strengths=cand_strengths.index_select(0, novel_idx),
                        importances=cand_imps.index_select(0, novel_idx),
                        day=current_day,
                    )
                    new_count = int(novel_idx.numel())
            elif cand_engrams_f.shape[0] > 0:
                self.semantic.add_batch(
                    sdrs=cand_engrams_f,
                    metas=cand_metas,
                    strengths=cand_strengths,
                    importances=cand_imps,
                    day=current_day,
                )
                new_count = int(cand_engrams_f.shape[0])

        # 3) Biological forgetting curve on long-term semantic bank (vectorized GPU)
        #    Pinned memories (pinned=True) and key knowledge (importance >= 4.0) are strictly immune to decay & forgetting!
        forgotten_count = 0
        if len(self.semantic) > 0:
            protected = self.semantic.pinned | (self.semantic.importances >= 4.0)
            decay_mask = (~protected) & (self.semantic.last_recalled_days < int(current_day))
            self.semantic.strengths = torch.where(
                decay_mask,
                self.semantic.strengths * float(decay),
                self.semantic.strengths,
            )
            forget_mask = (~protected) & (self.semantic.strengths < float(prune_threshold)) & (self.semantic.importances <= 2.0)
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
            if isinstance(fresh_engrams, QuantizedSynapseMatrix):
                parts.append(fresh_engrams.dequant_rows(idx_f, out_dtype=torch.float32))
            else:
                parts.append(fresh_engrams.index_select(0, idx_f).float())

        if target_sem > 0:
            weights = torch.clamp(self.semantic.strengths * self.semantic.importances, min=0.1)
            probs = weights / weights.sum()
            idx_s = torch.multinomial(probs, num_samples=target_sem, replacement=(target_sem > n_sem))
            parts.append(self.semantic.engrams.dequant_rows(idx_s, out_dtype=torch.float32))

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
