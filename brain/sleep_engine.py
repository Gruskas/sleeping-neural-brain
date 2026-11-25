from __future__ import annotations

import contextlib
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .config import BrainConfig
from .hippocampus import ConceptFact, Hippocampus, _cosine, _keyword_overlap
from .neural_cortex import AssociativeConceptCortex


def _locked(lock: threading.RLock | None):
    return lock if lock is not None else contextlib.nullcontext()


@dataclass
class SleepReport:
    night_number: int
    duration_ms: float
    device_name: str
    # Stage 1: NREM Light
    new_facts: int = 0
    reinforced_facts: int = 0
    new_memories: int = 0
    updated_memories: int = 0
    forgotten_memories: int = 0
    # Stage 2: NREM Deep
    concept_loss_start: float = 0.0
    concept_loss_end: float = 0.0
    lora_trained: bool = False
    lora_samples: int = 0
    lora_loss_start: float = 0.0
    lora_loss_end: float = 0.0
    lora_duration_ms: float = 0.0
    # Stage 3: REM Dreaming
    rem_new_links: list[dict[str, Any]] = field(default_factory=list)
    rem_reflections: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SleepEngine:
    """
    Executes the 3 biological sleep stages entirely on GPU:
      1) NREM Light  : Salience staging, contradiction resolution, biological forgetting curve.
      2) NREM Deep   : Concept Cortex consolidation (EWC) + Language Cortex LoRA Replay training on GPU.
      3) REM Dreaming: Multi-hop neural graph dreaming + Polish REM reflection synthesis.
    """

    def __init__(
        self,
        cfg: BrainConfig,
        concept_cortex: AssociativeConceptCortex,
        hippocampus: Hippocampus,
        device: torch.device,
        language_cortex: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.concept_cortex = concept_cortex
        self.hippocampus = hippocampus
        self.device = device
        self.language_cortex = language_cortex

    def run_sleep_cycle(
        self,
        night_number: int,
        stage_callback: Callable[[str], None] | None = None,
        visual_pacing: bool = False,
        state_lock: threading.RLock | None = None,
    ) -> SleepReport:
        t0 = time.perf_counter()
        dev_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        report = SleepReport(night_number=night_number, duration_ms=0.0, device_name=dev_name)

        # ── STAGE 1: NREM LIGHT ──────────────────────────────────────────────
        if stage_callback:
            stage_callback("NREM_LIGHT")

        # Precompute contradiction checks outside state_lock so Ollama HTTP never blocks /api/state
        precomputed_updates: dict[tuple[str, tuple[str, ...]], int | None] = {}
        if self.language_cortex is not None and self.language_cortex.ready:
            with _locked(state_lock):
                stm_snap = [
                    m for m in self.hippocampus.short_term_memories
                    if m.category in ("użytkownik", "zasada", "wydarzenie")
                ]
                sem_snap = [m for m in self.hippocampus.semantic_memories if not m.superseded]
            for m in stm_snap:
                if not sem_snap:
                    break
                if any(
                    old.text.lower() == m.text.lower()
                    or (m.embedding and old.embedding and _cosine(m.embedding, old.embedding) > 0.91)
                    for old in sem_snap
                ):
                    continue
                candidates = sorted(
                    sem_snap,
                    key=lambda o: 0.65 * _cosine(m.embedding, o.embedding) + 0.35 * _keyword_overlap(m.text, o.text),
                    reverse=True,
                )[:3]
                close_cands = [
                    c
                    for c in candidates
                    if (0.65 * _cosine(m.embedding, c.embedding) + 0.35 * _keyword_overlap(m.text, c.text)) >= 0.36
                ]
                if close_cands:
                    cand_texts = [c.text for c in close_cands]
                    precomputed_updates[(m.text, tuple(cand_texts))] = self.language_cortex.check_memory_update(
                        m.text, cand_texts
                    )

        with _locked(state_lock):
            stage1 = self.hippocampus.nrem_stage_facts()
            report.new_facts = stage1["new_facts"]
            report.reinforced_facts = stage1["reinforced_facts"]

            mem_stage1 = self.hippocampus.consolidate_declarative_memories(
                language_cortex=self.language_cortex,
                decay=self.cfg.memory_decay_per_night,
                prune_threshold=self.cfg.memory_prune_threshold,
                current_day=night_number,
                contradiction_checker=(
                    (lambda mt, cts: precomputed_updates.get((mt, tuple(cts))))
                    if self.language_cortex is not None and self.language_cortex.ready
                    else None
                ),
            )
            report.new_memories = mem_stage1["new_memories"]
            report.updated_memories = mem_stage1["updated_memories"]
            report.forgotten_memories = mem_stage1["forgotten_memories"]

            # Automatic relational triple distillation from declarative memories
            if self.language_cortex is not None and hasattr(self.language_cortex, "extract_triples_from_memory"):
                distill_texts: list[str] = list(mem_stage1["new_texts"][:60])
                active_sem = [m for m in self.hippocampus.semantic_memories if not m.superseded]
                if len(self.hippocampus.consolidated_facts) < min(120, len(active_sem)):
                    seen_dt = {t.lower() for t in distill_texts}
                    for sm in sorted(active_sem, key=lambda x: (x.importance, x.strength), reverse=True):
                        if sm.text.lower() not in seen_dt:
                            distill_texts.append(sm.text)
                            seen_dt.add(sm.text.lower())
                        if len(distill_texts) >= 100:
                            break
                distilled_count = 0
                for dtxt in distill_texts:
                    for s, r, d in self.language_cortex.extract_triples_from_memory(dtxt):
                        cf = ConceptFact(
                            src=s.strip().lower()[:60],
                            rel=r.strip().lower()[:40],
                            dst=d.strip().lower()[:60],
                            strength=1.1,
                            day=night_number,
                            source="CONSOLIDATED",
                            consolidated=True,
                        )
                        if cf.src and cf.dst and cf.src != cf.dst and cf.key() not in self.hippocampus.consolidated_facts:
                            self.hippocampus.consolidated_facts[cf.key()] = cf
                            distilled_count += 1
                report.new_facts += distilled_count

            for fact in self.hippocampus.consolidated_facts.values():
                self.concept_cortex.ensure_concept(fact.src)
                self.concept_cortex.ensure_concept(fact.dst)
                self.concept_cortex.ensure_relation(fact.rel)

        if visual_pacing:
            time.sleep(0.35)

        # ── STAGE 2: NREM DEEP (Concept EWC Consolidation) ───────────────────
        if stage_callback:
            stage_callback("NREM_DEEP")
        with _locked(state_lock):
            c_stats = self._consolidate_concept_cortex()
        report.concept_loss_start = c_stats["loss_start"]
        report.concept_loss_end = c_stats["loss_end"]

        # ── STAGE 3: REM DREAMING (Neural Graph + Polish REM Reflections) ────
        if stage_callback:
            stage_callback("REM_DREAMING")
        with _locked(state_lock):
            report.rem_new_links = self._rem_concept_dreaming(day=night_number)
            existing_refls = [r.text for r in self.hippocampus.rem_reflections]
            if not mem_stage1["new_texts"] and len(self.hippocampus.semantic_memories) >= 3:
                active_pool = sorted(
                    [m for m in self.hippocampus.semantic_memories if not m.superseded],
                    key=lambda m: (m.importance, m.strength, m.last_recalled_day),
                    reverse=True,
                )
                refl_new_pool = [m.text for m in active_pool[:16]]
                refl_old_pool = [m.text for m in active_pool[16:32]] or [m.text for m in active_pool[:16]]
            else:
                refl_new_pool = list(mem_stage1["new_texts"])
                refl_old_pool = list(mem_stage1["replay_texts"])

        new_rem_reflections: list[str] = []
        if self.language_cortex is not None and self.language_cortex.ready and refl_new_pool:
            new_rem_reflections = self.language_cortex.synthesize_rem_reflections(
                new_memories=refl_new_pool,
                old_memories=refl_old_pool,
                rem_links=report.rem_new_links,
                existing_reflections=existing_refls,
            )
            if new_rem_reflections:
                if hasattr(self.language_cortex, "encode_texts"):
                    rem_embs = self.language_cortex.encode_texts(new_rem_reflections)
                else:
                    rem_embs = [self.language_cortex.encode_text(r) for r in new_rem_reflections]
                with _locked(state_lock):
                    for refl, emb in zip(new_rem_reflections, rem_embs):
                        self.hippocampus.add_rem_reflection(refl, day=night_number, embedding=emb)
            report.rem_reflections = new_rem_reflections

        if visual_pacing:
            time.sleep(0.35)

        # ── STAGE 2/3 REPLAY: GPU Language Cortex Replay Training ────────────
        if self.language_cortex is not None and self.language_cortex.ready and (mem_stage1["new_texts"] or new_rem_reflections):
            if stage_callback:
                stage_callback("NREM_DEEP")
            train_new = mem_stage1["new_texts"] if mem_stage1["new_texts"] else mem_stage1["replay_texts"][:12]
            train_replay = mem_stage1["replay_texts"] if mem_stage1["new_texts"] else mem_stage1["replay_texts"][12:28]
            lora_stats = self.language_cortex.train_lora_sleep(
                new_memories=train_new,
                replay_memories=train_replay,
                rem_reflections=new_rem_reflections,
                rem_links=report.rem_new_links,
            )
            report.lora_trained = bool(lora_stats.get("trained", False))
            report.lora_samples = int(lora_stats.get("samples", 0))
            report.lora_loss_start = float(lora_stats.get("loss_start", 0.0))
            report.lora_loss_end = float(lora_stats.get("loss_end", 0.0))
            report.lora_duration_ms = float(lora_stats.get("duration_ms", 0.0))

        with _locked(state_lock):
            self.hippocampus.clear_short_term()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        report.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return report

    def _consolidate_concept_cortex(self) -> dict[str, float]:
        facts = list(self.hippocampus.consolidated_facts.values())
        if not facts or len(self.concept_cortex.id_to_concept) < 2:
            return {"loss_start": 0.0, "loss_end": 0.0}

        src_ids = torch.tensor(
            [self.concept_cortex.concept_to_id[f.src] for f in facts],
            dtype=torch.long,
            device=self.device,
        )
        rel_ids = torch.tensor(
            [self.concept_cortex.relation_to_id[f.rel] for f in facts],
            dtype=torch.long,
            device=self.device,
        )
        dst_ids = torch.tensor(
            [self.concept_cortex.concept_to_id[f.dst] for f in facts],
            dtype=torch.long,
            device=self.device,
        )
        weights = torch.tensor(
            [max(0.5, f.strength) for f in facts],
            dtype=torch.float32,
            device=self.device,
        )
        weights = weights / weights.mean().clamp(min=1e-4)

        self.concept_cortex.train()
        loss_start = 0.0
        loss_end = 0.0

        for epoch in range(self.cfg.nrem_concept_epochs):
            all_emb = self.concept_cortex.normalized_concept_matrix()
            src_v = all_emb[src_ids]
            dst_v = all_emb[dst_ids]

            pred_v = self.concept_cortex.predict_target_vec(src_v, rel_ids)
            rel_logits = (pred_v @ all_emb.T) * 7.5
            rel_loss = (F.cross_entropy(rel_logits, dst_ids, reduction="none") * weights).mean()

            dream_v = self.concept_cortex.predict_dream_vec(src_v)
            dream_logits = (dream_v @ all_emb.T) * 6.0
            dream_loss = (F.cross_entropy(dream_logits, dst_ids, reduction="none") * weights).mean()

            pair_cos = (src_v * dst_v).sum(dim=-1)
            prox_loss = (1.0 - pair_cos).clamp(min=0.25).mean()

            ewc = self.concept_cortex.ewc_penalty()
            loss = rel_loss + 0.55 * dream_loss + 0.25 * prox_loss + ewc

            self.concept_cortex.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.concept_cortex.parameters(), 2.0)
            self.concept_cortex.optimizer.step()

            val = float(loss.item())
            if epoch == 0:
                loss_start = val
            loss_end = val

        self.concept_cortex.update_fisher(src_ids, rel_ids, dst_ids)
        return {"loss_start": round(loss_start, 4), "loss_end": round(loss_end, 4)}

    @torch.no_grad()
    def _rem_concept_dreaming(self, day: int) -> list[dict[str, Any]]:
        all_facts = list(self.hippocampus.consolidated_facts.values())
        if len(all_facts) < 2 or len(self.concept_cortex.id_to_concept) < 3:
            return []

        # Cap to top 180 strongest facts so REM dreaming stays fast even on huge graphs
        facts = (
            sorted(all_facts, key=lambda f: f.strength, reverse=True)[:180]
            if len(all_facts) > 180
            else all_facts
        )

        self.concept_cortex.eval()
        all_emb = self.concept_cortex.normalized_concept_matrix()

        known_pairs: set[tuple[str, str]] = set()
        outgoing: dict[str, list[ConceptFact]] = {}
        incoming: dict[str, list[ConceptFact]] = {}
        deg: dict[str, int] = {}
        for f in facts:
            known_pairs.add((f.src, f.dst))
            known_pairs.add((f.dst, f.src))
            outgoing.setdefault(f.src, []).append(f)
            incoming.setdefault(f.dst, []).append(f)
            deg[f.src] = deg.get(f.src, 0) + 1
            deg[f.dst] = deg.get(f.dst, 0) + 1
        for d in self.hippocampus.rem_dream_links.values():
            known_pairs.add((d.src, d.dst))
            known_pairs.add((d.dst, d.src))

        def is_informative_node(name: str) -> bool:
            n = name.strip().lower()
            if len(n) < 3:
                return False
            if re.fullmatch(r"\d+(?:[.,-]\d+)*\s*(?:rok|lat|lata|stopni|metrów|metr|gramów|minut|godzin|dni|miesięcy|bitów|planet|przypadków)?", n):
                return False
            if re.match(r"^\d", n):
                return False
            return True

        candidates: list[tuple[float, str, str, str, str]] = []

        # 1) Transitive 2-hop chains: A -(r1)-> B -(r2)-> C
        for f1 in facts:
            a, r1, b = f1.src, f1.rel, f1.dst
            aid = self.concept_cortex.concept_to_id.get(a)
            bid = self.concept_cortex.concept_to_id.get(b)
            r1_idx = self.concept_cortex.relation_to_id.get(r1)
            if aid is None or bid is None or r1_idx is None:
                continue
            v_a = all_emb[aid : aid + 1]
            v_b = all_emb[bid : bid + 1]
            r1_id = torch.tensor([r1_idx], device=self.device)
            z1 = self.concept_cortex.predict_target_vec(v_a, r1_id)
            z1_dream = F.normalize(0.70 * z1 + 0.30 * v_b + torch.randn_like(z1) * 0.03, dim=-1)

            for f2 in outgoing.get(b, []):
                c = f2.dst
                r2 = f2.rel
                if c == a or (a, c) in known_pairs:
                    continue
                cid = self.concept_cortex.concept_to_id.get(c)
                r2_idx = self.concept_cortex.relation_to_id.get(r2)
                if cid is None or r2_idx is None:
                    continue
                v_c = all_emb[cid : cid + 1]
                r2_id = torch.tensor([r2_idx], device=self.device)
                z2 = self.concept_cortex.predict_target_vec(z1_dream, r2_id)
                dream_free = self.concept_cortex.predict_dream_vec(z1_dream)

                score = 0.65 * float((z2 * v_c).sum().item()) + 0.35 * float((dream_free * v_c).sum().item())
                if score >= self.cfg.rem_similarity_threshold:
                    bridge_desc = f"przez: {b} ({a} → {r1} → {b} → {r2} → {c})"[:120]
                    candidates.append((score, a, "skojarzenie_rem", c, bridge_desc))

        # 2) Convergent shared-target bridges: A -(r1)-> B <-(r2)- C
        for b, in_edges in incoming.items():
            if len(in_edges) < 2 or deg.get(b, 0) > 10:
                continue
            bid = self.concept_cortex.concept_to_id.get(b)
            if bid is None:
                continue
            v_b = all_emb[bid : bid + 1]
            for i in range(len(in_edges)):
                f1 = in_edges[i]
                a, r1 = f1.src, f1.rel
                if not is_informative_node(a):
                    continue
                aid = self.concept_cortex.concept_to_id.get(a)
                r1_idx = self.concept_cortex.relation_to_id.get(r1)
                if aid is None or r1_idx is None:
                    continue
                v_a = all_emb[aid : aid + 1]
                p_a = self.concept_cortex.predict_target_vec(v_a, torch.tensor([r1_idx], device=self.device))
                s_ab = float((p_a * v_b).sum().item())
                d_a = self.concept_cortex.predict_dream_vec(v_a)

                for j in range(i + 1, len(in_edges)):
                    f2 = in_edges[j]
                    c, r2 = f2.src, f2.rel
                    if c == a or (a, c) in known_pairs or not is_informative_node(c):
                        continue
                    cid = self.concept_cortex.concept_to_id.get(c)
                    r2_idx = self.concept_cortex.relation_to_id.get(r2)
                    if cid is None or r2_idx is None:
                        continue
                    v_c = all_emb[cid : cid + 1]
                    p_c = self.concept_cortex.predict_target_vec(v_c, torch.tensor([r2_idx], device=self.device))
                    s_cb = float((p_c * v_b).sum().item())
                    score = (
                        0.45 * float((v_a * v_c).sum().item())
                        + 0.35 * float((d_a * v_c).sum().item())
                        + 0.20 * min(s_ab, s_cb)
                    )
                    if score >= self.cfg.rem_similarity_threshold:
                        bridge_desc = f"wspólny most: {b} ({a} [{r1}] ↔ {c} [{r2}])"[:120]
                        candidates.append((score, a, "skojarzenie_rem", c, bridge_desc))

        # 3) Divergent shared-source bridges: B -(r1)-> A and B -(r2)-> C
        for b, out_edges in outgoing.items():
            if len(out_edges) < 2 or deg.get(b, 0) > 8:
                continue
            for i in range(len(out_edges)):
                f1 = out_edges[i]
                a, r1 = f1.dst, f1.rel
                if not is_informative_node(a):
                    continue
                aid = self.concept_cortex.concept_to_id.get(a)
                if aid is None:
                    continue
                v_a = all_emb[aid : aid + 1]
                d_a = self.concept_cortex.predict_dream_vec(v_a)
                for j in range(i + 1, len(out_edges)):
                    f2 = out_edges[j]
                    c, r2 = f2.dst, f2.rel
                    if c == a or (a, c) in known_pairs or r1 == r2 or not is_informative_node(c):
                        continue
                    cid = self.concept_cortex.concept_to_id.get(c)
                    if cid is None:
                        continue
                    v_c = all_emb[cid : cid + 1]
                    score = 0.50 * float((v_a * v_c).sum().item()) + 0.50 * float((d_a * v_c).sum().item())
                    if score >= self.cfg.rem_similarity_threshold:
                        bridge_desc = f"wspólne źródło: {b} ({r1} / {r2})"[:120]
                        candidates.append((score, a, "skojarzenie_rem", c, bridge_desc))

        candidates.sort(key=lambda x: x[0], reverse=True)
        discovered: list[dict[str, Any]] = []
        used_pairs: set[tuple[str, str]] = set()
        for score, a, rel, c, bridge in candidates:
            if (a, c) in used_pairs or (c, a) in used_pairs:
                continue
            used_pairs.add((a, c))
            dream_fact = ConceptFact(
                src=a,
                rel=rel,
                dst=c,
                strength=round(score, 3),
                day=day,
                source="REM_DREAM",
                bridge=bridge,
                consolidated=True,
            )
            self.hippocampus.rem_dream_links[dream_fact.key()] = dream_fact
            discovered.append({"src": a, "rel": rel, "dst": c, "score": round(score, 3), "bridge": bridge})
            if len(discovered) >= self.cfg.rem_max_new_links_per_night:
                break
        return discovered
