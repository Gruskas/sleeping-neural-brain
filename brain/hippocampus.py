from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

_PL_TRANS = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def _strip_pl(s: str) -> str:
    return s.translate(_PL_TRANS)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _keyword_overlap(q: str, text: str) -> float:
    def tokenize(s: str) -> set[str]:
        cleaned = _strip_pl(s.lower())
        for ch in ",.!?;:\"'()[]{}—-/\n\t":
            cleaned = cleaned.replace(ch, " ")
        return {w for w in cleaned.split() if len(w) >= 3}

    q_words = tokenize(q)
    t_words = tokenize(text)
    if not q_words or not t_words:
        return 0.0
    inter = 0
    for qw in q_words:
        q_stem_len = max(4, min(len(qw) - 2, 6)) if len(qw) >= 4 else len(qw)
        if qw in t_words or (
            len(qw) >= 4
            and any(
                len(tw) >= 4
                and (
                    tw.startswith(qw[:q_stem_len])
                    or qw.startswith(tw[: max(4, min(len(tw) - 2, 6))])
                )
                for tw in t_words
            )
        ):
            inter += 1
    return inter / max(1, len(q_words))


@dataclass
class ConceptFact:
    src: str
    rel: str
    dst: str
    strength: float = 1.0
    day: int = 1
    source: str = "WAKE_EXPERIENCE"  # WAKE_EXPERIENCE | WEB_STUDY | CONSOLIDATED | REM_DREAM
    bridge: str | None = None
    consolidated: bool = False
    timestamp: float = field(default_factory=time.time)

    def key(self) -> tuple[str, str, str]:
        return (self.src, self.rel, self.dst)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConceptFact":
        return cls(**data)


@dataclass
class MemoryTrace:
    """A human-like declarative memory trace with strength, recall reinforcement, topic & GPU embedding."""

    text: str
    category: str = "wiedza"  # użytkownik | wiedza | nauka_z_sieci | zasada | wydarzenie | refleksja_rem
    importance: int = 3  # 1..5
    strength: float = 1.2
    recall_count: int = 0
    day_created: int = 1
    last_recalled_day: int = 1
    consolidated: bool = False
    superseded: bool = False
    source_url: str | None = None
    topic: str | None = None
    embedding: list[float] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)

    def to_public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)
        d["strength"] = round(self.strength, 2)
        return d

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryTrace":
        # Backwards compatible with older checkpoints missing source_url/topic
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class Hippocampus:
    """
    High-capacity human-like dual-memory organ:
      - Short-term buffers (WAKE / active study): up to 5,000 declarative memories + 5,000 graph triples
      - Long-term archives (Post-SLEEP): up to 50,000 semantic memories + 500 REM reflections + concept graph
    """

    def __init__(
        self,
        max_short_term_memories: int = 5000,
        max_short_term_facts: int = 5000,
        max_semantic_memories: int = 50000,
        max_rem_reflections: int = 500,
    ) -> None:
        self.max_short_term_memories = max_short_term_memories
        self.max_short_term_facts = max_short_term_facts
        self.max_semantic_memories = max_semantic_memories
        self.max_rem_reflections = max_rem_reflections

        # Short-term (WAKE & Web Study)
        self.short_term_facts: list[ConceptFact] = []
        self.short_term_memories: list[MemoryTrace] = []

        # Long-term (Post-SLEEP)
        self.consolidated_facts: dict[tuple[str, str, str], ConceptFact] = {}
        self.rem_dream_links: dict[tuple[str, str, str], ConceptFact] = {}
        self.semantic_memories: list[MemoryTrace] = []
        self.rem_reflections: list[MemoryTrace] = []
        self.studied_topics: dict[str, dict[str, Any]] = {}

    def record_fact(
        self,
        src: str,
        rel: str,
        dst: str,
        day: int,
        strength: float = 1.0,
        source: str = "WAKE_EXPERIENCE",
    ) -> ConceptFact:
        fact = ConceptFact(
            src=src.strip().lower()[:60],
            rel=rel.strip().lower()[:40],
            dst=dst.strip().lower()[:60],
            strength=strength,
            day=day,
            source=source,
            consolidated=False,
        )
        for existing in self.short_term_facts:
            if existing.key() == fact.key():
                existing.strength = min(4.0, existing.strength + 0.5)
                existing.timestamp = time.time()
                return existing
        self.short_term_facts.append(fact)
        if len(self.short_term_facts) > self.max_short_term_facts:
            self.short_term_facts = self.short_term_facts[-self.max_short_term_facts :]
        return fact

    def record_memory(
        self,
        text: str,
        category: str,
        importance: int,
        day: int,
        embedding: list[float] | None = None,
        source_url: str | None = None,
        topic: str | None = None,
    ) -> MemoryTrace:
        clean = text.strip()
        emb = embedding or []

        for existing in self.short_term_memories:
            if existing.text.lower() == clean.lower() or (emb and _cosine(existing.embedding, emb) > 0.92):
                existing.strength = min(4.0, existing.strength + 0.4)
                existing.importance = max(existing.importance, importance)
                existing.timestamp = time.time()
                return existing

        trace = MemoryTrace(
            text=clean,
            category=category,
            importance=max(1, min(5, importance)),
            strength=0.8 + importance * 0.28,
            day_created=day,
            last_recalled_day=day,
            consolidated=False,
            source_url=source_url,
            topic=topic,
            embedding=emb,
        )
        self.short_term_memories.append(trace)
        if len(self.short_term_memories) > self.max_short_term_memories:
            # Drop lowest-importance oldest short-term memory only if 5,000+ exceeded
            self.short_term_memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
            self.short_term_memories = self.short_term_memories[: self.max_short_term_memories]
        return trace

    def add_rem_reflection(
        self,
        text: str,
        day: int,
        embedding: list[float] | None = None,
        topic: str | None = None,
    ) -> MemoryTrace:
        clean = text.strip()
        for existing in self.rem_reflections:
            if existing.text.lower() == clean.lower():
                existing.strength = min(5.0, existing.strength + 0.3)
                return existing
        trace = MemoryTrace(
            text=clean,
            category="refleksja_rem",
            importance=4,
            strength=1.6,
            day_created=day,
            last_recalled_day=day,
            consolidated=True,
            topic=topic,
            embedding=embedding or [],
        )
        self.rem_reflections.append(trace)
        if len(self.rem_reflections) > self.max_rem_reflections:
            self.rem_reflections = self.rem_reflections[-self.max_rem_reflections :]
        return trace

    def search_memories(
        self,
        query_vec: list[float],
        query_text: str,
        current_day: int,
        top_hippo: int = 14,
        top_semantic: int = 16,
        top_rem: int = 8,
    ) -> dict[str, list[MemoryTrace]]:
        def score_trace(m: MemoryTrace) -> float:
            if m.superseded:
                return -999.0
            cos = _cosine(query_vec, m.embedding) if (query_vec and m.embedding) else 0.0
            kw_text = _keyword_overlap(query_text, m.text)
            kw_topic = _keyword_overlap(query_text, m.topic or "")
            # Reject completely unrelated memories before adding strength/importance bonuses
            if cos < 0.18 and kw_text == 0.0 and kw_topic == 0.0:
                return 0.0
            # Strong lexical/topic boost so directly learned definitions ALWAYS rank #1!
            exact_boost = 0.48 * kw_text + 0.35 * kw_topic
            strength_bonus = min(0.25, math.log1p(max(0.0, m.strength)) * 0.10)
            core_bonus = 0.08 if m.category == "użytkownik" and m.importance >= 4 else 0.0
            imp_bonus = 0.12 if m.importance >= 5 else 0.0
            return 0.42 * cos + exact_boost + strength_bonus + core_bonus + imp_bonus

        # 1) Short-term Hippocampus (today's chats + recent web study)
        scored_hippo = sorted(self.short_term_memories, key=score_trace, reverse=True)
        hippo_hits = [m for m in scored_hippo if score_trace(m) >= 0.22][:top_hippo]

        # 2) Long-term consolidated Semantic Cortex
        active_sem = [m for m in self.semantic_memories if not m.superseded]
        scored_sem = sorted(active_sem, key=score_trace, reverse=True)
        sem_hits: list[MemoryTrace] = []
        seen_ids: set[str] = set()

        for m in scored_sem:
            if m.id in seen_ids:
                continue
            sc = score_trace(m)
            if sc < 0.22:
                break
            sem_hits.append(m)
            seen_ids.add(m.id)
            m.recall_count += 1
            m.last_recalled_day = current_day
            m.strength = min(6.0, m.strength + 0.15)
            if len(sem_hits) >= top_semantic:
                break

        # 3) REM Reflections
        scored_rem = sorted(self.rem_reflections, key=score_trace, reverse=True)
        rem_hits = [m for m in scored_rem[:top_rem] if score_trace(m) >= 0.22]

        # Separate user profile facts so greetings know user's name without polluting unrelated search pills
        user_profile = [
            m for m in (self.short_term_memories + active_sem)
            if not m.superseded and m.category == "użytkownik"
        ][:6]

        return {
            "hippocampus": hippo_hits,
            "semantic": sem_hits,
            "rem": rem_hits,
            "user_profile": user_profile,
        }

    def consolidate_declarative_memories(
        self,
        language_cortex: Any | None,
        decay: float,
        prune_threshold: float,
        current_day: int,
        contradiction_checker: Callable[[str, list[str]], int | None] | None = None,
    ) -> dict[str, Any]:
        new_count = 0
        updated_count = 0
        reinforced_count = 0
        new_texts: list[str] = []
        seen_new_texts: set[str] = set()

        def add_new_text(txt: str) -> None:
            if txt not in seen_new_texts:
                seen_new_texts.add(txt)
                new_texts.append(txt)

        checker = contradiction_checker
        if checker is None and language_cortex is not None and language_cortex.ready:
            checker = language_cortex.check_memory_update

        active_sem = [m for m in self.semantic_memories if not m.superseded]

        # Only run LLM contradiction check on user/rule memories (skip bulk web encyclopedia facts for speed)
        for m in self.short_term_memories:
            dup = None
            for old in active_sem:
                if old.text.lower() == m.text.lower() or (
                    m.embedding and old.embedding and _cosine(m.embedding, old.embedding) > 0.91
                ):
                    dup = old
                    break

            if dup is not None:
                dup.strength = min(6.0, dup.strength + 0.5)
                dup.importance = max(dup.importance, m.importance)
                dup.last_recalled_day = current_day
                reinforced_count += 1
                add_new_text(dup.text)
                continue

            if (
                checker is not None
                and active_sem
                and m.category in ("użytkownik", "zasada", "wydarzenie")
            ):
                candidates = sorted(
                    active_sem,
                    key=lambda o: 0.65 * _cosine(m.embedding, o.embedding) + 0.35 * _keyword_overlap(m.text, o.text),
                    reverse=True,
                )[:3]
                close_cands = [
                    c
                    for c in candidates
                    if (0.65 * _cosine(m.embedding, c.embedding) + 0.35 * _keyword_overlap(m.text, c.text)) >= 0.36
                ]
                if close_cands:
                    rep_idx = checker(m.text, [c.text for c in close_cands])
                    if rep_idx is not None:
                        close_cands[rep_idx].superseded = True
                        updated_count += 1

            m.consolidated = True
            self.semantic_memories.append(m)
            active_sem.append(m)
            new_count += 1
            add_new_text(m.text)

        # Gentle biological forgetting curve for weak, unreferenced trivial memories
        forgotten_count = 0
        kept_sem: list[MemoryTrace] = []
        for old in self.semantic_memories:
            if old.superseded:
                continue
            if old.importance < 4 and old.last_recalled_day < current_day:
                old.strength *= decay
            if old.strength < prune_threshold and old.importance <= 2:
                forgotten_count += 1
                continue
            kept_sem.append(old)
        if len(kept_sem) > self.max_semantic_memories:
            kept_sem.sort(key=lambda x: (x.importance, x.strength, x.last_recalled_day), reverse=True)
            kept_sem = kept_sem[: self.max_semantic_memories]
        self.semantic_memories = kept_sem

        # Select diverse high-value older memories for Experience Replay during LoRA sleep training
        new_set = set(new_texts)
        replay_pool = sorted(
            [m for m in self.semantic_memories if m.text not in new_set],
            key=lambda x: (x.importance, x.strength, x.recall_count),
            reverse=True,
        )
        replay_texts = [m.text for m in replay_pool[:28]]

        return {
            "new_memories": new_count,
            "updated_memories": updated_count,
            "reinforced_memories": reinforced_count,
            "forgotten_memories": forgotten_count,
            "new_texts": new_texts,
            "replay_texts": replay_texts,
        }

    def nrem_stage_facts(self) -> dict[str, int]:
        raw_fact_count = len(self.short_term_facts)
        new_facts = 0
        reinforced_facts = 0
        for f in self.short_term_facts:
            k = f.key()
            if k in self.consolidated_facts:
                old = self.consolidated_facts[k]
                old.strength = min(5.0, old.strength + f.strength * 0.5)
                old.consolidated = True
                reinforced_facts += 1
            else:
                f.consolidated = True
                f.source = "CONSOLIDATED"
                self.consolidated_facts[k] = f
                new_facts += 1
        return {
            "raw_facts": raw_fact_count,
            "new_facts": new_facts,
            "reinforced_facts": reinforced_facts,
        }

    def delete_memory(self, memory_id: str) -> bool:
        target = memory_id.strip()
        if not target:
            return False
        for attr in ("short_term_memories", "semantic_memories", "rem_reflections"):
            lst: list[MemoryTrace] = getattr(self, attr)
            before = len(lst)
            filtered = [m for m in lst if m.id != target]
            if len(filtered) != before:
                setattr(self, attr, filtered)
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
        self.short_term_facts.clear()
        self.short_term_memories.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "short_term_facts": [f.to_dict() for f in self.short_term_facts],
            "consolidated_facts": [f.to_dict() for f in self.consolidated_facts.values()],
            "rem_dream_links": [f.to_dict() for f in self.rem_dream_links.values()],
            "short_term_memories": [m.to_dict() for m in self.short_term_memories],
            "semantic_memories": [m.to_dict() for m in self.semantic_memories],
            "rem_reflections": [m.to_dict() for m in self.rem_reflections],
            "studied_topics": self.studied_topics,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self.short_term_facts = [ConceptFact.from_dict(d) for d in data.get("short_term_facts", [])]
        self.consolidated_facts = {
            f.key(): f for f in (ConceptFact.from_dict(d) for d in data.get("consolidated_facts", []))
        }
        self.rem_dream_links = {
            f.key(): f for f in (ConceptFact.from_dict(d) for d in data.get("rem_dream_links", []))
        }
        self.short_term_memories = [MemoryTrace.from_dict(d) for d in data.get("short_term_memories", [])]
        self.semantic_memories = [MemoryTrace.from_dict(d) for d in data.get("semantic_memories", [])]
        self.rem_reflections = [MemoryTrace.from_dict(d) for d in data.get("rem_reflections", [])]
        self.studied_topics = dict(data.get("studied_topics", {}))
