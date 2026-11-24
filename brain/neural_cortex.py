from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BrainConfig

_PL_TRANS = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def _strip_pl(text: str) -> str:
    return text.translate(_PL_TRANS)


class AssociativeConceptCortex(nn.Module):
    """
    GPU-accelerated Associative Semantic Cortex trained from scratch in PyTorch.
    - Learns dense 64D concept & 32D relation embeddings on the unit hypersphere.
    - Protected by Elastic Weight Consolidation (EWC) across sleep cycles.
    - During REM sleep, performs 2-hop neural dreaming in latent space to discover
      indirect/hidden associations (e.g., A->B and B->C => A~>C).
    """

    def __init__(self, cfg: BrainConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.concept_to_id: dict[str, int] = {}
        self.id_to_concept: list[str] = []
        self.relation_to_id: dict[str, int] = {}
        self.id_to_relation: list[str] = []

        self.concept_emb = nn.Embedding(cfg.max_concepts, cfg.concept_dim)
        self.relation_emb = nn.Embedding(cfg.max_relations, cfg.relation_dim)
        nn.init.normal_(self.concept_emb.weight, mean=0.0, std=0.25)
        nn.init.normal_(self.relation_emb.weight, mean=0.0, std=0.25)

        self.assoc_net = nn.Sequential(
            nn.Linear(cfg.concept_dim + cfg.relation_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, cfg.concept_dim),
        )

        # Free-association dream projector (relation-agnostic latent drift)
        self.dream_net = nn.Sequential(
            nn.Linear(cfg.concept_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, cfg.concept_dim),
        )

        self.to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.concept_lr, weight_decay=1e-4)

        self.fisher: dict[str, torch.Tensor] = {}
        self.anchor_params: dict[str, torch.Tensor] = {}

    def ensure_concept(self, name: str) -> int:
        key = name.strip().lower()[:60]
        if key not in self.concept_to_id:
            if len(self.id_to_concept) >= self.cfg.max_concepts:
                return 0
            idx = len(self.id_to_concept)
            self.concept_to_id[key] = idx
            self.id_to_concept.append(key)
        return self.concept_to_id[key]

    def ensure_relation(self, name: str) -> int:
        key = name.strip().lower()[:40]
        if key not in self.relation_to_id:
            if len(self.id_to_relation) >= self.cfg.max_relations:
                return 0
            idx = len(self.id_to_relation)
            self.relation_to_id[key] = idx
            self.id_to_relation.append(key)
        return self.relation_to_id[key]

    def normalized_concept_matrix(self) -> torch.Tensor:
        n = len(self.id_to_concept)
        if n == 0:
            return torch.empty((0, self.cfg.concept_dim), device=self.device)
        idx = torch.arange(n, device=self.device)
        return F.normalize(self.concept_emb(idx), dim=-1)

    def predict_target_vec(self, src_vec: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        r_vec = F.normalize(self.relation_emb(rel_ids), dim=-1)
        delta = self.assoc_net(torch.cat([src_vec, r_vec], dim=-1))
        return F.normalize(src_vec * 0.35 + delta, dim=-1)

    def predict_dream_vec(self, src_vec: torch.Tensor) -> torch.Tensor:
        delta = self.dream_net(src_vec)
        return F.normalize(src_vec * 0.4 + delta, dim=-1)

    def ewc_penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        total_elems = 0
        for name, param in self.named_parameters():
            if name in self.fisher and param.requires_grad:
                penalty = penalty + (self.fisher[name] * (param - self.anchor_params[name]).pow(2)).sum()
                total_elems += param.numel()
        return 0.5 * self.cfg.ewc_lambda_concept * (penalty / max(1, total_elems))

    def update_fisher(self, src_ids: torch.Tensor, rel_ids: torch.Tensor, dst_ids: torch.Tensor) -> None:
        if len(src_ids) == 0 or len(self.id_to_concept) < 2:
            return
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        all_emb = self.normalized_concept_matrix()
        src_v = all_emb[src_ids]
        pred_v = self.predict_target_vec(src_v, rel_ids)
        logits = (pred_v @ all_emb.T) * 8.0
        loss = F.cross_entropy(logits, dst_ids)
        loss.backward()

        raw_sq: dict[str, torch.Tensor] = {}
        g_sum = 0.0
        g_cnt = 0
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is None:
                    continue
                g2 = param.grad.detach().pow(2)
                raw_sq[name] = g2
                g_sum += float(g2.sum().item())
                g_cnt += g2.numel()
            mean_g2 = max(1e-10, g_sum / max(1, g_cnt))
            for name, param in self.named_parameters():
                if name not in raw_sq:
                    continue
                norm_g2 = (raw_sq[name] / mean_g2).clamp(max=25.0)
                if name in self.fisher:
                    self.fisher[name] = 0.65 * self.fisher[name] + 0.35 * norm_g2
                else:
                    self.fisher[name] = norm_g2.clone()
                self.anchor_params[name] = param.detach().clone()
        self.optimizer.zero_grad(set_to_none=True)

    def resolve_concept(self, query_word: str) -> str | None:
        key = query_word.strip().lower()[:60]
        if not key:
            return None
        if key in self.concept_to_id:
            return key
        key_norm = _strip_pl(key)
        if len(key_norm) >= 4:
            for cand in self.id_to_concept:
                if _strip_pl(cand) == key_norm:
                    return cand
        if len(key_norm) < 5:
            return None
        key_stem = key_norm[: max(4, len(key_norm) - 2)]
        best_cand: str | None = None
        best_rank: tuple[int, int] | None = None
        for cand in self.id_to_concept:
            if abs(len(cand) - len(key)) > 4:
                continue
            cand_norm = _strip_pl(cand)
            cand_stem = cand_norm[: max(4, len(cand_norm) - 2)] if len(cand_norm) >= 4 else cand_norm
            if cand_norm.startswith(key_stem) or (len(cand_norm) >= 4 and key_norm.startswith(cand_stem)):
                rank = (abs(len(cand) - len(key)), len(cand))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_cand = cand
        return best_cand

    @torch.no_grad()
    def query_associations(self, concept: str, top_k: int = 8) -> list[dict[str, Any]]:
        resolved = self.resolve_concept(concept)
        if resolved is None or len(self.id_to_concept) < 2:
            return []
        self.eval()
        all_emb = self.normalized_concept_matrix()
        cid = self.concept_to_id[resolved]
        src_v = all_emb[cid : cid + 1]

        sim_direct = (src_v @ all_emb.T).squeeze(0)
        dream_v = self.predict_dream_vec(src_v)
        sim_dream = (dream_v @ all_emb.T).squeeze(0)

        best_rel_sim = torch.full_like(sim_direct, -1.0)
        best_rel_id = torch.zeros(len(self.id_to_concept), dtype=torch.long, device=self.device)
        for rid in range(len(self.id_to_relation)):
            r_t = torch.tensor([rid], dtype=torch.long, device=self.device)
            pv = self.predict_target_vec(src_v, r_t)
            s_r = (pv @ all_emb.T).squeeze(0)
            better = s_r > best_rel_sim
            best_rel_sim = torch.where(better, s_r, best_rel_sim)
            best_rel_id = torch.where(better, torch.full_like(best_rel_id, rid), best_rel_id)

        combined = 0.45 * best_rel_sim + 0.35 * sim_dream + 0.20 * sim_direct
        combined[cid] = -999.0

        k = min(top_k, len(self.id_to_concept) - 1)
        vals, idxs = torch.topk(combined, k=k)
        results = []
        for score, idx_t in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
            if score < 0.15:
                continue
            rid = int(best_rel_id[idx_t].item())
            rel_name = self.id_to_relation[rid] if rid < len(self.id_to_relation) else "skojarzenie"
            results.append(
                {
                    "concept": self.id_to_concept[idx_t],
                    "score": round(float(score), 3),
                    "via_relation": rel_name,
                    "direct_sim": round(float(sim_direct[idx_t].item()), 3),
                }
            )
        return results

    @torch.no_grad()
    def compute_2d_projection(self, limit: int = 80, concepts: list[str] | None = None) -> dict[str, tuple[float, float]]:
        n_total = len(self.id_to_concept)
        if n_total == 0:
            return {}
        all_mat = self.normalized_concept_matrix()
        if concepts is not None:
            idxs: list[int] = []
            seen: set[int] = set()
            for c in concepts:
                cid = self.concept_to_id.get(c)
                if cid is not None and cid not in seen:
                    seen.add(cid)
                    idxs.append(cid)
                    if len(idxs) >= limit:
                        break
            if not idxs:
                return {}
            mat = all_mat[idxs]
            names = [self.id_to_concept[i] for i in idxs]
            n = len(names)
        else:
            mat = all_mat
            names = self.id_to_concept
            n = n_total
            if n > limit:
                mat = mat[-limit:]
                names = names[-limit:]
                n = limit
        if n == 1:
            return {names[0]: (0.0, 0.0)}
        centered = mat - mat.mean(dim=0, keepdim=True)
        u, s, _ = torch.linalg.svd(centered, full_matrices=False)
        coords_t = u[:, :2] * s[:2]
        if coords_t.shape[1] == 1:
            coords_t = torch.cat([coords_t, torch.zeros((n, 1), device=self.device)], dim=1)
        max_abs = coords_t.abs().max().clamp(min=1e-5)
        coords = ((coords_t / max_abs) * 0.92).cpu().tolist()

        # Lightweight pairwise repulsion so concept labels do not overlap
        min_dist = 0.16
        for _ in range(8):
            for i in range(n):
                xi, yi = coords[i]
                for j in range(i + 1, n):
                    xj, yj = coords[j]
                    dx = xi - xj
                    dy = yi - yj
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < min_dist:
                        if dist < 1e-6:
                            ux, uy = 0.7071, 0.7071
                        else:
                            ux, uy = dx / dist, dy / dist
                        push = (min_dist - dist) * 0.35
                        xi += ux * push
                        yi += uy * push
                        xj -= ux * push
                        yj -= uy * push
                        coords[j][0] = max(-0.95, min(0.95, xj))
                        coords[j][1] = max(-0.95, min(0.95, yj))
                coords[i][0] = max(-0.95, min(0.95, xi))
                coords[i][1] = max(-0.95, min(0.95, yi))

        return {names[i]: (float(coords[i][0]), float(coords[i][1])) for i in range(n)}

    def export_state(self) -> dict[str, Any]:
        return {
            "concept_to_id": self.concept_to_id,
            "id_to_concept": self.id_to_concept,
            "relation_to_id": self.relation_to_id,
            "id_to_relation": self.id_to_relation,
            "weights": {k: v.cpu() for k, v in self.state_dict().items()},
            "fisher": {k: v.cpu() for k, v in self.fisher.items()},
            "anchor": {k: v.cpu() for k, v in self.anchor_params.items()},
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self.concept_to_id = dict(state["concept_to_id"])
        self.id_to_concept = list(state["id_to_concept"])
        self.relation_to_id = dict(state["relation_to_id"])
        self.id_to_relation = list(state["id_to_relation"])
        # Handle resized embedding tables safely if upgrading from an older checkpoint
        saved_weights = state["weights"]
        cur_sd = self.state_dict()
        merged = {}
        for k, cur_v in cur_sd.items():
            if k in saved_weights and saved_weights[k].shape == cur_v.shape:
                merged[k] = saved_weights[k].to(self.device)
            elif k in saved_weights and saved_weights[k].ndim == 2 and cur_v.ndim == 2 and saved_weights[k].shape[1] == cur_v.shape[1]:
                buf = cur_v.clone()
                rows = min(buf.shape[0], saved_weights[k].shape[0])
                buf[:rows] = saved_weights[k][:rows].to(self.device)
                merged[k] = buf
            else:
                merged[k] = cur_v
        self.load_state_dict(merged)
        self.fisher = {
            k: v.to(self.device)
            for k, v in state.get("fisher", {}).items()
            if k in cur_sd and v.shape == cur_sd[k].shape
        }
        self.anchor_params = {
            k: v.to(self.device)
            for k, v in state.get("anchor", {}).items()
            if k in cur_sd and v.shape == cur_sd[k].shape
        }
