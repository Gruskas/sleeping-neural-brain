from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

_PL_TRANS = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")

_STOPWORDS_PL = {
    "jest", "sa", "to", "oraz", "lub", "albo", "czy", "jak", "jaki", "jaka",
    "jakie", "kto", "co", "czym", "gdzie", "kiedy", "dlaczego", "czemu", "dla",
    "przez", "przy", "nad", "pod", "przed", "bez", "jako", "jego", "jej", "ich",
    "tego", "tym", "tej", "ten", "ta", "te", "tak", "nie", "ale", "lecz", "wiec",
    "ze", "sie", "czyli", "jednak", "rowniez", "takze", "poniewaz", "mozna",
    "ktory", "ktora", "ktore", "ktorych", "czesto", "bardzo", "miedzy", "zwiazek",
    "roznica", "porownaj", "laczy", "wplyw", "zalezy", "powiedz", "opowiedz",
    "wyjasnij", "opisz", "podaj", "zdefiniuj", "napisz", "krotko", "temat",
    "slowo", "oznacza", "wtedy", "natomiast", "ponadto", "tylko", "jeszcze",
}

_POLISH_ROOT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("slonc", "slonecz"),
    ("oddychan", "oddech"),
    ("czlowiek", "ludz"),
    ("swiatl", "swietl"),
    ("roslin", "roslinn"),
    ("ziemi", "ziemsk"),
    ("komork", "komor"),
    ("energi", "energet"),
    ("pamiec", "pamiet"),
    ("dzien", "dni"),
    ("procesor", "cpu"),
    ("graficzn", "gpu"),
    ("dysk", "ssd", "hdd"),
    ("ram", "operacyjn"),
    ("tlen", "tlenu", "tlenow"),
    ("dwutlen", "co2"),
    ("wod", "h2o"),
)


def _strip_pl(s: str) -> str:
    return s.translate(_PL_TRANS)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _words_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4 and abs(len(a) - len(b)) <= 5:
        a_stem = a[: max(4, min(len(a) - 2, 6))]
        b_stem = b[: max(4, min(len(b) - 2, 6))]
        if b.startswith(a_stem) or a.startswith(b_stem):
            return True
    for grp in _POLISH_ROOT_GROUPS:
        a_hit = any((a == g or (len(g) >= 4 and a.startswith(g))) for g in grp)
        if a_hit:
            b_hit = any((b == g or (len(g) >= 4 and b.startswith(g))) for g in grp)
            if b_hit:
                return True
    return False


def _tokenize_words_norm(s: str, drop_stopwords: bool = True) -> list[str]:
    cleaned = _strip_pl(s.lower())
    for ch in ",.!?;:\"'()[]{}—-/\n\t":
        cleaned = cleaned.replace(ch, " ")
    raw = [w for w in cleaned.split() if len(w) >= 2]
    if not drop_stopwords:
        return [w for w in raw if len(w) >= 3 or w in ("pi", "o2", "ip", "si", "ai")]
    filtered = [w for w in raw if w not in _STOPWORDS_PL and (len(w) >= 3 or w in ("pi", "o2", "ip", "si", "ai"))]
    return filtered if filtered else [w for w in raw if len(w) >= 3]


def _keyword_overlap(q: str, text: str) -> float:
    q_words = set(_tokenize_words_norm(q, drop_stopwords=True))
    t_words = set(_tokenize_words_norm(text, drop_stopwords=False))
    if not q_words or not t_words:
        return 0.0
    inter = sum(1 for qw in q_words if any(_words_match(qw, tw) for tw in t_words))
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
        self.episodic_memories: list[MemoryTrace] = []
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
            sim_thresh = 0.86 if category == "rozmowa" else 0.92
            if existing.text.lower() == clean.lower() or (emb and _cosine(existing.embedding, emb) > sim_thresh):
                if category == "rozmowa":
                    existing.text = clean
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

    def record_episodic(
        self,
        text: str,
        day: int,
        embedding: list[float] | None = None,
    ) -> MemoryTrace:
        clean = text.strip()
        emb = embedding or []
        for existing in self.episodic_memories[-12:]:
            if existing.text.lower() == clean.lower() or (emb and _cosine(existing.embedding, emb) > 0.90):
                existing.text = clean
                existing.timestamp = time.time()
                return existing
        trace = MemoryTrace(
            text=clean,
            category="rozmowa",
            importance=3,
            strength=1.4,
            day_created=day,
            last_recalled_day=day,
            consolidated=True,
            embedding=emb,
        )
        self.episodic_memories.append(trace)
        if len(self.episodic_memories) > 1000:
            self.episodic_memories = self.episodic_memories[-1000:]
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
        ][:8]

        # Episodic conversation & event memories (preserved across sleeps)
        episodic_pool = [
            m for m in (self.episodic_memories + self.short_term_memories + active_sem)
            if not m.superseded and m.category in ("rozmowa", "wydarzenie")
        ]
        episodic_pool.sort(key=lambda m: (score_trace(m) >= 0.18, m.timestamp), reverse=True)
        episodic_hits = episodic_pool[:8]

        return {
            "hippocampus": hippo_hits,
            "semantic": sem_hits,
            "rem": rem_hits,
            "user_profile": user_profile,
            "episodic": episodic_hits,
        }

    def find_concept_paths(
        self,
        query_terms: list[str],
        max_hops: int = 3,
        max_paths: int = 6,
    ) -> list[dict[str, Any]]:
        """Find multi-hop connecting paths across consolidated, REM, and short-term facts."""
        all_facts = (
            list(self.consolidated_facts.values())
            + list(self.short_term_facts)
            + list(self.rem_dream_links.values())
        )
        if not all_facts or not query_terms:
            return []

        adj: dict[str, list[tuple[str, str, str, str, float, str | None]]] = {}
        nodes_set: set[str] = set()
        for f in all_facts:
            u, rel, v = f.src.strip().lower(), f.rel.strip().lower(), f.dst.strip().lower()
            if not u or not v or u == v:
                continue
            nodes_set.add(u)
            nodes_set.add(v)
            adj.setdefault(u, []).append((v, rel, "out", f.source, float(f.strength), f.bridge))
            adj.setdefault(v, []).append((u, rel, "in", f.source, float(f.strength), f.bridge))

        if not nodes_set:
            return []

        nodes_list = sorted(nodes_set)
        node_norm_map = {n: _strip_pl(n) for n in nodes_list}
        node_words_map = {n: set(_tokenize_words_norm(n, drop_stopwords=False)) for n in nodes_list}

        def resolve_term(term: str) -> list[str]:
            t_clean = term.strip().lower()
            if not t_clean:
                return []
            t_norm = _strip_pl(t_clean)
            t_words = [w for w in _tokenize_words_norm(t_clean, drop_stopwords=True)]
            if not t_words:
                return []
            scored: list[tuple[float, str]] = []
            for n in nodes_list:
                n_norm = node_norm_map[n]
                if n == t_clean or n_norm == t_norm:
                    scored.append((10.0, n))
                    continue
                n_words = node_words_map[n]
                if not n_words:
                    continue
                hits = sum(1 for tw in t_words if any(_words_match(tw, nw) for nw in n_words))
                if hits == 0:
                    continue
                cov_q = hits / len(t_words)
                cov_n = hits / len(n_words)
                substr_bonus = 0.4 if (len(t_norm) >= 4 and (t_norm in n_norm or n_norm in t_norm)) else 0.0
                score = 2.0 * cov_q + 1.2 * cov_n + substr_bonus
                if score >= 1.1:
                    scored.append((score, n))
            scored.sort(key=lambda x: (-x[0], len(x[1])))
            return [n for _, n in scored[:2]]

        resolved_groups: list[list[str]] = []
        seen_resolved: set[str] = set()
        for term in query_terms:
            matches = resolve_term(term)
            fresh = [m for m in matches if m not in seen_resolved]
            if fresh:
                resolved_groups.append(fresh)
                seen_resolved.update(fresh)

        all_resolved = [n for grp in resolved_groups for n in grp]
        if not all_resolved:
            return []

        def format_summary(steps: list[dict[str, Any]]) -> str:
            if not steps:
                return ""
            parts: list[str] = []
            first = steps[0]
            if first["direction"] == "out":
                parts.append(f"{first['src']} —({first['rel']})→ {first['dst']}")
            else:
                parts.append(f"{first['dst']} ←({first['rel']})— {first['src']}")
            for st in steps[1:]:
                if st["direction"] == "out":
                    parts.append(f"—({st['rel']})→ {st['dst']}")
                else:
                    parts.append(f"←({st['rel']})— {st['src']}")
            return " ".join(parts)

        results: list[dict[str, Any]] = []
        seen_summaries: set[str] = set()

        if len(resolved_groups) >= 2:
            # Find shortest connecting paths across distinct resolved concept groups
            candidate_paths: list[tuple[float, dict[str, Any]]] = []
            for i in range(len(resolved_groups)):
                for j in range(i + 1, len(resolved_groups)):
                    for start in resolved_groups[i]:
                        for end in resolved_groups[j]:
                            if start == end:
                                continue
                            # BFS up to max_hops
                            queue: deque[tuple[str, list[dict[str, Any]]]] = deque([(start, [])])
                            visited_depth: dict[str, int] = {start: 0}
                            found_for_pair: list[tuple[float, dict[str, Any]]] = []
                            while queue:
                                cur, path_steps = queue.popleft()
                                if len(path_steps) >= max_hops:
                                    continue
                                edges = sorted(
                                    adj.get(cur, []),
                                    key=lambda e: (e[3] == "REM_DREAM", -e[4]),
                                )
                                for nxt, rel, direction, source, strength, _bridge in edges:
                                    if any(
                                        (s["src"] == nxt or s["dst"] == nxt)
                                        for s in path_steps
                                    ):
                                        continue
                                    step_dict = {
                                        "src": cur if direction == "out" else nxt,
                                        "rel": rel,
                                        "dst": nxt if direction == "out" else cur,
                                        "direction": direction,
                                        "source": source,
                                    }
                                    new_steps = path_steps + [step_dict]
                                    if nxt == end:
                                        bridges = []
                                        cur_node = start
                                        for st in new_steps[:-1]:
                                            cur_node = st["dst"] if st["direction"] == "out" else st["src"]
                                            if cur_node not in (start, end) and cur_node not in bridges:
                                                bridges.append(cur_node)
                                        rem_penalty = sum(0.6 for st in new_steps if st["source"] == "REM_DREAM")
                                        hub_penalty = sum(
                                            0.35 for b in bridges if len(adj.get(b, [])) > 12
                                        )
                                        rank_cost = len(new_steps) + rem_penalty + hub_penalty
                                        candidate_paths.append(
                                            (
                                                rank_cost,
                                                {
                                                    "start": start,
                                                    "end": end,
                                                    "hops": len(new_steps),
                                                    "steps": new_steps,
                                                    "bridges": bridges,
                                                    "summary": format_summary(new_steps),
                                                },
                                            )
                                        )
                                        found_for_pair.append(candidate_paths[-1])
                                        if len(found_for_pair) >= 2:
                                            break
                                    elif len(new_steps) < max_hops:
                                        prev_d = visited_depth.get(nxt)
                                        if prev_d is None or len(new_steps) <= prev_d:
                                            visited_depth[nxt] = len(new_steps)
                                            queue.append((nxt, new_steps))
                                if len(found_for_pair) >= 2:
                                    break

            candidate_paths.sort(key=lambda x: (x[0], x[1]["hops"], x[1]["summary"]))
            for _, p in candidate_paths:
                if p["summary"] in seen_summaries:
                    continue
                seen_summaries.add(p["summary"])
                results.append(p)
                if len(results) >= max_paths:
                    break

        if not results and all_resolved:
            # Single-concept fallback: return top 1-hop and 2-hop chains
            anchor = all_resolved[0]
            seen_pairs: set[tuple[str, str]] = set()
            one_hop = sorted(
                adj.get(anchor, []),
                key=lambda e: (e[3] == "REM_DREAM", -e[4]),
            )
            for nxt, rel, direction, source, strength, _ in one_hop[:4]:
                if (anchor, nxt) in seen_pairs:
                    continue
                seen_pairs.add((anchor, nxt))
                st1 = {
                    "src": anchor if direction == "out" else nxt,
                    "rel": rel,
                    "dst": nxt if direction == "out" else anchor,
                    "direction": direction,
                    "source": source,
                }
                # Try extending 1 more hop if informative
                ext_edges = [
                    e for e in sorted(adj.get(nxt, []), key=lambda e: (e[3] == "REM_DREAM", -e[4]))
                    if e[0] != anchor and (anchor, e[0]) not in seen_pairs
                ]
                if ext_edges and len(results) < 2:
                    nxt2, rel2, dir2, src2, _, _ = ext_edges[0]
                    st2 = {
                        "src": nxt if dir2 == "out" else nxt2,
                        "rel": rel2,
                        "dst": nxt2 if dir2 == "out" else nxt,
                        "direction": dir2,
                        "source": src2,
                    }
                    steps = [st1, st2]
                    summ = format_summary(steps)
                    if summ not in seen_summaries:
                        seen_summaries.add(summ)
                        results.append(
                            {
                                "start": anchor,
                                "end": nxt2,
                                "hops": 2,
                                "steps": steps,
                                "bridges": [nxt],
                                "summary": summ,
                            }
                        )
                        continue
                steps = [st1]
                summ = format_summary(steps)
                if summ not in seen_summaries:
                    seen_summaries.add(summ)
                    results.append(
                        {
                            "start": anchor,
                            "end": nxt,
                            "hops": 1,
                            "steps": steps,
                            "bridges": [],
                            "summary": summ,
                        }
                    )
                if len(results) >= min(4, max_paths):
                    break

        return results[:max_paths]

    def search_memories_multihop(
        self,
        query_vec: list[float],
        query_text: str,
        sub_queries: list[str],
        sub_vecs: list[list[float]],
        bridge_terms: list[str],
        current_day: int,
        top_hippo: int = 14,
        top_semantic: int = 18,
        top_rem: int = 8,
    ) -> dict[str, list[MemoryTrace]]:
        base = self.search_memories(
            query_vec=query_vec,
            query_text=query_text,
            current_day=current_day,
            top_hippo=top_hippo,
            top_semantic=top_semantic,
            top_rem=top_rem,
        )
        if not sub_queries or (len(sub_queries) < 2 and not bridge_terms):
            return base

        hippo_hits = list(base["hippocampus"])
        sem_hits = list(base["semantic"])
        rem_hits = list(base["rem"])
        seen_hippo = {m.id for m in hippo_hits}
        seen_sem = {m.id for m in sem_hits}
        seen_rem = {m.id for m in rem_hits}
        active_sem = [m for m in self.semantic_memories if not m.superseded]

        def score_sub(m: MemoryTrace, sq: str, sv: list[float]) -> float:
            if m.superseded:
                return -999.0
            cos = _cosine(sv, m.embedding) if (sv and m.embedding) else 0.0
            kw_text = _keyword_overlap(sq, m.text)
            kw_topic = _keyword_overlap(sq, m.topic or "")
            if kw_text == 0.0 and kw_topic == 0.0 and cos < 0.45:
                return 0.0
            return 0.42 * cos + 0.52 * kw_text + 0.35 * kw_topic + (0.08 if m.importance >= 5 else 0.0)

        for sq, sv in zip(sub_queries[:5], sub_vecs[:5]):
            for pool, seen_set, out_list, is_sem in (
                (self.short_term_memories, seen_hippo, hippo_hits, False),
                (active_sem, seen_sem, sem_hits, True),
                (self.rem_reflections, seen_rem, rem_hits, False),
            ):
                scored = sorted(pool, key=lambda m: score_sub(m, sq, sv), reverse=True)
                added = 0
                for m in scored:
                    if m.id in seen_set:
                        continue
                    if score_sub(m, sq, sv) < 0.25:
                        break
                    out_list.append(m)
                    seen_set.add(m.id)
                    if is_sem:
                        m.recall_count += 1
                        m.last_recalled_day = current_day
                        m.strength = min(6.0, m.strength + 0.12)
                    added += 1
                    if added >= 4:
                        break

        for bt in bridge_terms[:4]:
            if not bt or len(bt.strip()) < 3:
                continue
            for pool, seen_set, out_list, is_sem in (
                (self.short_term_memories, seen_hippo, hippo_hits, False),
                (active_sem, seen_sem, sem_hits, True),
            ):
                matches = [
                    m for m in pool
                    if m.id not in seen_set
                    and (_keyword_overlap(bt, m.text) > 0.0 or _keyword_overlap(bt, m.topic or "") > 0.0)
                ]
                matches.sort(key=lambda m: (_keyword_overlap(bt, m.text), m.importance, m.strength), reverse=True)
                for m in matches[:2]:
                    out_list.append(m)
                    seen_set.add(m.id)
                    if is_sem:
                        m.recall_count += 1
                        m.last_recalled_day = current_day

        return {
            "hippocampus": hippo_hits[: top_hippo + 6],
            "semantic": sem_hits[: top_semantic + 8],
            "rem": rem_hits[: top_rem + 4],
            "user_profile": base["user_profile"],
            "episodic": base["episodic"],
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
        for attr in ("short_term_memories", "semantic_memories", "rem_reflections", "episodic_memories"):
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
            "episodic_memories": [m.to_dict() for m in self.episodic_memories],
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
        self.episodic_memories = [MemoryTrace.from_dict(d) for d in data.get("episodic_memories", [])]
        self.studied_topics = dict(data.get("studied_topics", {}))
