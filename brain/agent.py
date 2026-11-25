from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import torch

from .config import BrainConfig
from .curriculum import (
    BROAD_AUTONOMOUS_SYLLABUS,
    ELEMENTARY_FOUNDATION_PACK,
    ELEMENTARY_FOUNDATION_TRIPLES,
    ELEMENTARY_WEB_SYLLABUS,
)
from .hippocampus import Hippocampus
from .language_cortex import LanguageCortex, _fallback_hash_embed, _strip_pl
from .neural_cortex import AssociativeConceptCortex
from .sleep_engine import SleepEngine, SleepReport
from .web_learner import chunk_text, clean_search_query, extract_lead_definition, gather_web_sources, is_http_url


class SleepingBrain:
    """
    Pure Conversational & Autonomous Web-Learning Neural Brain on GPU (PyTorch + CUDA):
      - Natural Polish chat with expanded working memory (24 turns) & high-capacity Hippocampus (5,000+)
      - Autonomous Web Study: searches Wikipedia & the web iteratively, analyzes pages on GPU,
        self-quizzes for gaps until mastery, and uses sleep strictly to connect & consolidate information!
      - 3-stage Informational Sleep on GPU:
          NREM Light (contradictions + forgetting) -> NREM Deep (Concept EWC + LoRA Replay) -> REM Dreaming (creative links & analogies)
    """

    def __init__(
        self,
        cfg: BrainConfig | None = None,
        base_dir: str | Path = ".",
        load_language_mode: str = "async",  # "async" | "sync" | "none"
    ) -> None:
        self.cfg = cfg or BrainConfig()
        self.base_dir = Path(base_dir).resolve()
        self.state_dir = self.base_dir / self.cfg.state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.RLock()

        # Core brain organs
        self.hippocampus = Hippocampus(
            max_short_term_memories=self.cfg.max_short_term_memories,
            max_short_term_facts=self.cfg.max_short_term_facts,
            max_semantic_memories=self.cfg.max_semantic_memories,
            max_rem_reflections=self.cfg.max_rem_reflections,
        )
        self.concept_cortex = AssociativeConceptCortex(self.cfg, self.device)
        self.language_cortex = LanguageCortex(
            cfg=self.cfg,
            device=self.device,
            lora_dir=self.state_dir / "lora_cortex",
        )
        self.sleep_engine = SleepEngine(
            cfg=self.cfg,
            concept_cortex=self.concept_cortex,
            hippocampus=self.hippocampus,
            device=self.device,
            language_cortex=self.language_cortex,
        )

        # Lifecycle & creativity state (no fatigue; sleep is purely for information linking & consolidation)
        self.state_mode: str = "AWAKE"  # AWAKE | STUDYING | NREM_LIGHT | NREM_DEEP | REM_DREAMING
        self.energy: float = self.cfg.max_energy
        self.creativity_level: float = float(self.cfg.creativity_level)
        self.day: int = 1
        self.total_sleeps: int = 0
        self.auto_sleep: bool = True
        self.auto_web: bool = True
        self._active_op: str | None = None  # "chat" | "sleep" | "study"

        # Active Web Study state (visible live in UI)
        self._study_stop_event = threading.Event()
        self.study_state: dict[str, Any] = {
            "active": False,
            "topic": "",
            "round": 0,
            "max_rounds": self.cfg.study_max_rounds,
            "mastery": 0.0,
            "pages_read": 0,
            "facts_learned": 0,
            "current_action": "",
            "sources": [],
            "summary": "",
        }

        # Conversational working memory & visible chat log
        self.working_chat: list[dict[str, str]] = []
        self.chat_log: list[dict[str, Any]] = []
        self.sleep_history: list[dict[str, Any]] = []
        self.event_log: list[dict[str, Any]] = []

        if not self.load_state():
            self._seed_initial_knowledge()

        if load_language_mode == "sync":
            self.language_cortex.load_sync()
        elif load_language_mode == "async":
            self.language_cortex.load_async()

    def _log(self, kind: str, message: str) -> None:
        self.event_log.append(
            {
                "t": time.strftime("%H:%M:%S"),
                "day": self.day,
                "kind": kind,
                "msg": message,
            }
        )
        if len(self.event_log) > 120:
            self.event_log = self.event_log[-120:]

    def _append_chat(
        self,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
        add_to_working: bool = True,
    ) -> dict[str, Any]:
        entry = {
            "role": role,
            "content": content,
            "day": self.day,
            "t": time.strftime("%H:%M:%S"),
            "meta": meta or {},
        }
        self.chat_log.append(entry)
        if len(self.chat_log) > self.cfg.chat_log_limit:
            self.chat_log = self.chat_log[-self.cfg.chat_log_limit :]
        if add_to_working and role in ("user", "assistant"):
            self.working_chat.append({"role": role, "content": content})
            if len(self.working_chat) > self.cfg.working_chat_buffer:
                self.working_chat = self.working_chat[-self.cfg.working_chat_buffer :]
        return entry

    def _acquire_op(self, op_name: str) -> None:
        with self.lock:
            if self._active_op == "study" or self.study_state["active"]:
                raise RuntimeError(
                    f"Mózg jest teraz skupiony na nauce tematu „{self.study_state['topic']}” z sieci — poczekaj chwilkę lub kliknij Stop."
                )
            if self._active_op == "sleep" or self.state_mode in ("NREM_LIGHT", "NREM_DEEP", "REM_DREAMING"):
                raise RuntimeError("Mózg właśnie śpi i konsoliduje pamięć na GPU — poczekaj chwilę aż się obudzi!")
            if self._active_op == "chat":
                raise RuntimeError("Mózg właśnie odpowiada na poprzednią wiadomość — poczekaj chwilkę!")
            self._active_op = op_name

    def _release_op(self, op_name: str) -> None:
        with self.lock:
            if self._active_op == op_name:
                self._active_op = None

    def _seed_initial_knowledge(self) -> None:
        self._append_chat(
            "assistant",
            "Cześć! Jestem Twoim uczącym się mózgiem AI na GPU. Możesz ze mną normalnie rozmawiać i uczyć mnie nowych rzeczy, "
            "albo kazać mi samodzielnie zbadać temat lub link w internecie (np. wpisz `/ucz sieci neuronowe` albo użyj paska „Ucz się z sieci”). "
            "Wszystko trafia najpierw do mojego hipokampu, a podczas snu konsoliduję wiedzę w pamięci trwałej, rdzeniu NREM, grafie pojęć EWC i refleksjach REM!",
            meta={"kind": "greeting"},
            add_to_working=False,
        )
        self._log("INIT", f"Uruchomiono czysty mózg konwersacyjny na {self.gpu_info()['device_name']}.")

    def gpu_info(self) -> dict[str, Any]:
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            return {
                "cuda_available": True,
                "device_name": props.name,
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
                "vram_total_mb": round(props.total_memory / (1024 * 1024), 1),
                "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
                "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2),
            }
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "cuda_version": None,
            "torch_version": torch.__version__,
            "vram_total_mb": 0.0,
            "vram_allocated_mb": 0.0,
            "vram_reserved_mb": 0.0,
        }

    # ── Elementary School Curriculum & Self-Contained Foundation Pack ─────────
    def learn_elementary_foundation(self, consolidate: bool = True, _from_nested: bool = False) -> dict[str, Any]:
        if not _from_nested:
            self._acquire_op("study")
        try:
            with self.lock:
                existing = {
                    m.text.strip().lower()
                    for m in (self.hippocampus.short_term_memories + self.hippocampus.semantic_memories)
                    if not m.superseded
                }
            to_add = [
                item for item in ELEMENTARY_FOUNDATION_PACK
                if item["text"].strip().lower() not in existing
            ]
            embs = self.language_cortex.encode_texts([item["text"] for item in to_add]) if to_add else []

            added_triples = 0
            with self.lock:
                for item, emb in zip(to_add, embs):
                    self.hippocampus.record_memory(
                        text=item["text"],
                        category=item.get("category", "wiedza"),
                        importance=int(item.get("importance", 4)),
                        day=self.day,
                        embedding=emb,
                        topic=item.get("topic"),
                    )
                existing_fact_keys = set(self.hippocampus.consolidated_facts.keys()) | {
                    f.key() for f in self.hippocampus.short_term_facts
                }
                for s, r, d in ELEMENTARY_FOUNDATION_TRIPLES:
                    if (s, r, d) not in existing_fact_keys:
                        self.hippocampus.record_fact(s, r, d, day=self.day, source="CURRICULUM")
                        added_triples += 1

                self.hippocampus.studied_topics["szkoła podstawowa — fundamenty"] = {
                    "topic": "Szkoła podstawowa — fundamenty",
                    "mastery": 95.0,
                    "facts_learned": len(ELEMENTARY_FOUNDATION_PACK),
                    "pages_read": 0,
                    "day": self.day,
                    "summary": "Podstawy 9 przedmiotów: matematyka, przyroda, geografia, fizyka, chemia, historia, język polski, informatyka, wiedza codzienna.",
                    "sources": [],
                }
                if to_add:
                    self._log("STUDY", f"🎓 Wgrano pakiet Szkoła Podstawowa: +{len(to_add)} faktów, +{added_triples} relacji grafu.")

            sleep_rep = None
            if consolidate and (to_add or self.hippocampus.short_term_memories):
                sleep_rep = self.sleep_now(visual_pacing=False, _from_nested=True)

            # Train conversational dialogue phrasing on the new foundation facts
            if to_add:
                with self.lock:
                    all_m = [m.text for m in self.hippocampus.semantic_memories if not m.superseded]
                    user_p = [m.text for m in self.hippocampus.semantic_memories if not m.superseded and m.category == "użytkownik"]
                self.language_cortex.train_language_cortex(
                    all_memories=all_m,
                    episodic_memories=[],
                    chat_turns=[],
                    user_profile=user_p,
                    epochs=8,
                )

            with self.lock:
                total_sem = len([m for m in self.hippocampus.semantic_memories if not m.superseded])
                if not _from_nested:
                    msg = (
                        f"🎓 Wgrałem pakiet **Szkoła Podstawowa** (+{len(to_add)} nowych faktów, +{added_triples} powiązań grafu) "
                        f"i wytrenowałem sieci neuronowe podczas snu na GPU. Łączna pamięć trwała: {total_sem} faktów.\n"
                        f"Możesz mnie teraz pytać o matematykę, przyrodę, geografię, fizykę, chemię, historię, polski lub informatykę!"
                        if to_add
                        else f"🎓 Pakiet **Szkoła Podstawowa** jest już w mojej pamięci ({total_sem} utrwalonych faktów). Pytaj śmiało!"
                    )
                    self._append_chat(
                        "assistant",
                        msg,
                        meta={"kind": "study_done", "topic": "Szkoła Podstawowa", "facts": len(to_add)},
                        add_to_working=False,
                    )
            self.save_state()
            return {
                "added_facts": len(to_add),
                "added_triples": added_triples,
                "total_semantic": total_sem,
                "sleep_report": sleep_rep.to_dict() if sleep_rep else None,
            }
        finally:
            if not _from_nested:
                self._release_op("study")

    def start_curriculum_async(self, deep_web: bool = False, max_topics: int = 12) -> dict[str, Any]:
        if not deep_web:
            return self.learn_elementary_foundation(consolidate=True)
        topics_count = max(1, min(len(ELEMENTARY_WEB_SYLLABUS), int(max_topics)))
        self._acquire_op("study")
        with self.lock:
            self._study_stop_event.clear()
            self.state_mode = "STUDYING"
            self.study_state = {
                "active": True,
                "topic": f"Kurs szkolny z sieci ({topics_count} tematów)",
                "round": 0,
                "max_rounds": topics_count,
                "mastery": 0.0,
                "pages_read": 0,
                "facts_learned": 0,
                "current_action": "Wgrywanie fundamentów i przygotowanie kursu szkolnego...",
                "sources": [],
                "summary": "",
            }
        try:
            t = threading.Thread(
                target=self.run_curriculum_online,
                args=(topics_count, True),
                daemon=True,
            )
            t.start()
        except Exception:
            with self.lock:
                self.study_state["active"] = False
                self.state_mode = "AWAKE"
            self._release_op("study")
            raise
        return {"started": True, "topics": topics_count}

    def run_curriculum_online(self, max_topics: int = 12, _op_acquired: bool = False) -> dict[str, Any]:
        if not _op_acquired:
            self._acquire_op("study")
        try:
            with self.lock:
                self._study_stop_event.clear()
                self.state_mode = "STUDYING"
                self._append_chat(
                    "assistant",
                    f"📚 Rozpoczynam kurs szkolny z polskiej Wikipedii (do {max_topics} tematów) wraz z pakietem fundamentów!",
                    meta={"kind": "study_start", "topic": "Kurs szkolny"},
                    add_to_working=False,
                )

            foundation_res = self.learn_elementary_foundation(consolidate=True, _from_nested=True)
            with self.lock:
                self.state_mode = "STUDYING"
                pending_topics = [
                    t for t in ELEMENTARY_WEB_SYLLABUS
                    if self.hippocampus.studied_topics.get(t.lower(), {}).get("mastery", 0.0) < 75.0
                ][:max_topics]
                self.study_state["max_rounds"] = max(1, len(pending_topics))

            seen_urls: set[str] = set()
            total_pages = 0
            total_facts = int(foundation_res["added_facts"])
            sources_list: list[dict[str, str]] = []

            for idx, topic in enumerate(pending_topics, start=1):
                if self._study_stop_event.is_set():
                    break
                self.study_state["round"] = idx
                self.study_state["topic"] = f"Kurs szkolny: {topic}"
                self.study_state["current_action"] = f"Temat {idx}/{len(pending_topics)}: badam „{topic}” w sieci..."
                docs = gather_web_sources(
                    query=topic,
                    seen_urls=seen_urls,
                    max_docs=2,
                    timeout=self.cfg.study_http_timeout,
                    max_workers=self.cfg.study_fetch_workers,
                )
                topic_facts: list[str] = []
                for doc in docs:
                    if self._study_stop_event.is_set():
                        break
                    total_pages += 1
                    self.study_state["pages_read"] = total_pages
                    sources_list.append({"title": doc.title, "url": doc.url, "source": doc.source})
                    self.study_state["sources"] = sources_list[-12:]
                    lead_def = extract_lead_definition(doc)
                    if lead_def:
                        emb_l = self.language_cortex.encode_text(lead_def)
                        with self.lock:
                            tr = self.hippocampus.record_memory(
                                text=lead_def,
                                category="nauka_z_sieci",
                                importance=5,
                                day=self.day,
                                embedding=emb_l,
                                source_url=doc.url,
                                topic=topic,
                            )
                            topic_facts.append(tr.text)
                            total_facts += 1
                            self.study_state["facts_learned"] = total_facts
                    for ch in chunk_text(doc.text, focus_query=topic, max_chars=2400, max_chunks=1):
                        ext = self.language_cortex.study_document_chunk(topic, doc.title, ch)
                        facts = ext.get("facts", [])
                        embs = self.language_cortex.encode_texts([f["text"] for f in facts])
                        with self.lock:
                            for f_item, emb in zip(facts, embs):
                                tr = self.hippocampus.record_memory(
                                    text=f_item["text"],
                                    category="nauka_z_sieci",
                                    importance=f_item["importance"],
                                    day=self.day,
                                    embedding=emb,
                                    source_url=doc.url,
                                    topic=topic,
                                )
                                topic_facts.append(tr.text)
                                total_facts += 1
                                self.study_state["facts_learned"] = total_facts
                            for s, r, d in ext.get("triples", []):
                                self.hippocampus.record_fact(s, r, d, day=self.day, source="WEB_STUDY")

                if topic_facts:
                    assess = self.language_cortex.assess_study_mastery(topic, topic_facts, round_num=1)
                    with self.lock:
                        self.hippocampus.studied_topics[topic.lower()] = {
                            "topic": topic,
                            "mastery": float(assess["mastery"]),
                            "facts_learned": len(topic_facts),
                            "pages_read": len(docs),
                            "day": self.day,
                            "summary": assess.get("summary", ""),
                            "sources": sources_list[-4:],
                        }
                self.study_state["mastery"] = round((idx / max(1, len(pending_topics))) * 100.0, 1)

            if self.hippocampus.short_term_memories and not self._study_stop_event.is_set():
                self.study_state["current_action"] = "Końcowa konsolidacja wiedzy szkolnej na GPU..."
                self.sleep_now(visual_pacing=False, _from_nested=True)

            with self.lock:
                self._append_chat(
                    "assistant",
                    f"✅ Ukończono kurs szkolny! Przeczytano {total_pages} stron i zapisano {total_facts} faktów w pamięci na GPU.",
                    meta={"kind": "study_done", "topic": "Kurs szkolny", "facts": total_facts},
                    add_to_working=False,
                )
            self.save_state()
            return {"pages_read": total_pages, "facts_learned": total_facts}
        finally:
            with self.lock:
                self.study_state["active"] = False
                self.study_state["current_action"] = ""
                if self.state_mode == "STUDYING":
                    self.state_mode = "AWAKE"
            self._release_op("study")

    def start_turbo_learn_async(self, topics_count: int = 200) -> dict[str, Any]:
        """Fast parallel multi-topic web learning + GPU sleep + language training."""
        count = max(5, min(len(BROAD_AUTONOMOUS_SYLLABUS), int(topics_count)))
        self._acquire_op("study")
        with self.lock:
            self._study_stop_event.clear()
            self.state_mode = "STUDYING"
            self.study_state = {
                "active": True,
                "topic": f"⚡ Turbo-nauka ({count} tematów)",
                "round": 0,
                "max_rounds": count,
                "mastery": 0.0,
                "pages_read": 0,
                "facts_learned": 0,
                "current_action": f"Uruchamiam równoległe pobieranie {count} tematów...",
                "sources": [],
                "summary": "",
            }
        try:
            t = threading.Thread(
                target=self.run_turbo_learn,
                args=(count, True),
                daemon=True,
            )
            t.start()
        except Exception:
            with self.lock:
                self.study_state["active"] = False
                self.state_mode = "AWAKE"
            self._release_op("study")
            raise
        return {"started": True, "topics": count}

    def run_turbo_learn(self, topics_count: int = 200, _op_acquired: bool = False) -> dict[str, Any]:
        if not _op_acquired:
            self._acquire_op("study")
        count = max(5, min(len(BROAD_AUTONOMOUS_SYLLABUS), int(topics_count)))
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with self.lock:
                self._study_stop_event.clear()
                self.state_mode = "STUDYING"
                self._append_chat(
                    "assistant",
                    f"⚡ Rozpoczynam **Turbo-naukę ({count} tematów)**! Pobieram wiedzę równolegle z sieci, "
                    f"a na koniec przeprowadzę sen na GPU i wykształcę model językowy.",
                    meta={"kind": "study_start", "topic": f"Turbo-nauka {count} tematów"},
                    add_to_working=False,
                )

            foundation_res = self.learn_elementary_foundation(consolidate=False, _from_nested=True)
            with self.lock:
                self.state_mode = "STUDYING"
                studied_lower = set(self.hippocampus.studied_topics.keys())
                pending_topics = [t for t in BROAD_AUTONOMOUS_SYLLABUS if t.lower() not in studied_lower][:count]
                if not pending_topics:
                    pending_topics = list(BROAD_AUTONOMOUS_SYLLABUS[:count])
                self.study_state["max_rounds"] = max(1, len(pending_topics))

            total_pages = 0
            total_facts = int(foundation_res.get("added_facts", 0))
            sources_list: list[dict[str, str]] = []
            seen_urls: set[str] = set()
            url_lock = threading.Lock()

            def _fetch_one(topic: str) -> tuple[str, list[Any]]:
                with url_lock:
                    local_seen = set(seen_urls)
                docs = gather_web_sources(
                    query=topic,
                    seen_urls=local_seen,
                    max_docs=1,
                    timeout=min(8, self.cfg.study_http_timeout),
                    max_workers=2,
                )
                with url_lock:
                    for d in docs:
                        seen_urls.add(d.url)
                return topic, docs

            done_topics = 0
            since_last_sleep = 0
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(_fetch_one, t) for t in pending_topics]
                for fut in as_completed(futures):
                    if self._study_stop_event.is_set():
                        break
                    topic, docs = fut.result()
                    done_topics += 1
                    since_last_sleep += 1
                    topic_facts = 0
                    for doc in docs:
                        total_pages += 1
                        sources_list.append({"title": doc.title, "url": doc.url, "source": doc.source})
                        lead_def = extract_lead_definition(doc)
                        if lead_def:
                            emb_l = self.language_cortex.encode_text(lead_def)
                            with self.lock:
                                self.hippocampus.record_memory(
                                    text=lead_def,
                                    category="nauka_z_sieci",
                                    importance=5,
                                    day=self.day,
                                    embedding=emb_l,
                                    source_url=doc.url,
                                    topic=topic,
                                )
                                topic_facts += 1
                                total_facts += 1
                        for ch in chunk_text(doc.text, focus_query=topic, max_chars=2200, max_chunks=1):
                            extracted = self.language_cortex.study_document_chunk(topic, doc.title, ch)
                            facts = extracted.get("facts", [])[:4]
                            if facts:
                                fact_embs = self.language_cortex.encode_texts([item["text"] for item in facts])
                                with self.lock:
                                    for item, emb in zip(facts, fact_embs):
                                        self.hippocampus.record_memory(
                                            text=item["text"],
                                            category="nauka_z_sieci",
                                            importance=item.get("importance", 4),
                                            day=self.day,
                                            embedding=emb,
                                            source_url=doc.url,
                                            topic=topic,
                                        )
                                        topic_facts += 1
                                        total_facts += 1
                                    for s, rel, d in extracted.get("triples", [])[:4]:
                                        self.hippocampus.record_fact(s, rel, d, day=self.day, source="TURBO_WEB")
                    with self.lock:
                        self.hippocampus.studied_topics[topic.lower()] = {
                            "topic": topic,
                            "mastery": 85.0 if topic_facts > 0 else 40.0,
                            "facts_learned": topic_facts,
                            "pages_read": len(docs),
                            "day": self.day,
                            "summary": f"Turbo-nauka: {topic_facts} faktów.",
                            "sources": sources_list[-3:],
                        }
                        self.study_state["round"] = done_topics
                        self.study_state["pages_read"] = total_pages
                        self.study_state["facts_learned"] = total_facts
                        self.study_state["sources"] = sources_list[-12:]
                        self.study_state["topic"] = f"⚡ Mega Turbo-nauka: {topic}"
                        self.study_state["mastery"] = round((done_topics / max(1, len(pending_topics))) * 85.0, 1)
                        self.study_state["current_action"] = f"Pobrano {done_topics}/{len(pending_topics)} tematów (+{total_facts} faktów)..."
                    if since_last_sleep >= 60 and not self._study_stop_event.is_set():
                        self.study_state["current_action"] = f"Śródokresowy sen na GPU ({done_topics}/{len(pending_topics)} tematów)..."
                        self.sleep_now(visual_pacing=False, _from_nested=True)
                        since_last_sleep = 0

            if self.hippocampus.short_term_memories and not self._study_stop_event.is_set():
                self.study_state["current_action"] = "Konsolidacja wiedzy we śnie + kształcenie języka na GPU..."
                self.study_state["mastery"] = 92.0
                self.sleep_now(visual_pacing=False, _from_nested=True)
                self.train_language_now(epochs=16, _from_nested=True)

            self.study_state["mastery"] = 100.0
            with self.lock:
                self._append_chat(
                    "assistant",
                    f"⚡ Zakończyłem **Turbo-naukę**! Przerobiłem **{done_topics} tematów** "
                    f"({total_pages} stron, **+{total_facts} faktów**), skonsolidowałem pamięć we śnie "
                    f"i wykształciłem model językowy na GPU. Możesz już ze mną rozmawiać!",
                    meta={"kind": "study_done", "topic": "Turbo-nauka", "facts": total_facts},
                    add_to_working=False,
                )
            self.save_state()
            return {"topics": done_topics, "pages": total_pages, "facts": total_facts}
        finally:
            with self.lock:
                self.study_state["active"] = False
                self.study_state["current_action"] = ""
                if self.state_mode == "STUDYING":
                    self.state_mode = "AWAKE"
            self._release_op("study")

    def start_continuous_learning_async(self, duration_minutes: int = 60) -> dict[str, Any]:
        """Start autonomous multi-topic web learning + periodic GPU sleep for `duration_minutes`."""
        mins = max(1, min(480, int(duration_minutes)))
        self._acquire_op("study")
        with self.lock:
            self._study_stop_event.clear()
            self.state_mode = "STUDYING"
            self.study_state = {
                "active": True,
                "topic": f"Auto-nauka z sieci ({mins} min)",
                "round": 0,
                "max_rounds": mins,
                "mastery": 0.0,
                "pages_read": 0,
                "facts_learned": 0,
                "current_action": f"Przygotowuję autonomiczną sesję nauki ({mins} min)...",
                "sources": [],
                "summary": "",
            }
        try:
            t = threading.Thread(
                target=self.run_continuous_learning,
                args=(mins, True),
                daemon=True,
            )
            t.start()
        except Exception:
            with self.lock:
                self.study_state["active"] = False
                self.state_mode = "AWAKE"
            self._release_op("study")
            raise
        return {"started": True, "duration_minutes": mins}

    def run_continuous_learning(self, duration_minutes: int = 60, _op_acquired: bool = False) -> dict[str, Any]:
        if not _op_acquired:
            self._acquire_op("study")
        mins = max(1, min(480, int(duration_minutes)))
        deadline = time.monotonic() + mins * 60.0
        try:
            with self.lock:
                self._study_stop_event.clear()
                self.state_mode = "STUDYING"
                self._append_chat(
                    "assistant",
                    f"🚀 Rozpoczynam ciągłą autonomiczną naukę z internetu na **{mins} minut**! "
                    f"Będę uczyć się kolejnych tematów po kolei i co kilka tematów zasypiać, żeby utrwalać wiedzę na GPU.",
                    meta={"kind": "study_start", "topic": f"Auto-nauka {mins} min"},
                    add_to_working=False,
                )

            foundation_res = self.learn_elementary_foundation(consolidate=True, _from_nested=True)
            total_pages = 0
            total_facts = int(foundation_res.get("added_facts", 0))
            topics_done = 0
            sleeps_done = 1 if foundation_res.get("sleep_report") else 0
            seen_urls: set[str] = set()
            sources_list: list[dict[str, str]] = []
            since_last_sleep = 0

            def _next_topic() -> str | None:
                with self.lock:
                    studied_lower = set(self.hippocampus.studied_topics.keys())
                    for t in BROAD_AUTONOMOUS_SYLLABUS:
                        if t.lower() not in studied_lower:
                            return t
                    for cname in list(self.concept_cortex.id_to_concept.values()):
                        if len(cname) >= 4 and cname.lower() not in studied_lower and " " not in cname[:3]:
                            return cname
                return None

            while time.monotonic() < deadline and not self._study_stop_event.is_set():
                topic = _next_topic()
                if not topic:
                    break
                topics_done += 1
                rem_min = max(1, int((deadline - time.monotonic()) / 60.0) + 1)
                with self.lock:
                    self.state_mode = "STUDYING"
                    self.study_state["round"] = topics_done
                    self.study_state["topic"] = f"Auto-nauka ({rem_min} min): {topic}"
                    self.study_state["current_action"] = f"Temat #{topics_done}: badam „{topic}” w sieci..."

                docs = gather_web_sources(
                    query=topic,
                    seen_urls=seen_urls,
                    max_docs=2,
                    timeout=self.cfg.study_http_timeout,
                    max_workers=self.cfg.study_fetch_workers,
                )
                topic_facts_count = 0
                for doc in docs:
                    if self._study_stop_event.is_set() or time.monotonic() >= deadline:
                        break
                    total_pages += 1
                    self.study_state["pages_read"] = total_pages
                    sources_list.append({"title": doc.title, "url": doc.url, "source": doc.source})
                    self.study_state["sources"] = sources_list[-12:]
                    lead_def = extract_lead_definition(doc)
                    if lead_def:
                        emb_l = self.language_cortex.encode_text(lead_def)
                        with self.lock:
                            self.hippocampus.record_memory(
                                text=lead_def,
                                category="nauka_z_sieci",
                                importance=5,
                                day=self.day,
                                embedding=emb_l,
                                source_url=doc.url,
                                topic=topic,
                            )
                            topic_facts_count += 1
                            total_facts += 1
                            self.study_state["facts_learned"] = total_facts
                    for ch in chunk_text(doc.text, focus_query=topic, max_chars=2400, max_chunks=2):
                        if self._study_stop_event.is_set():
                            break
                        extracted = self.language_cortex.study_document_chunk(topic, doc.title, ch)
                        facts = extracted.get("facts", [])
                        fact_embs = self.language_cortex.encode_texts([item["text"] for item in facts])
                        with self.lock:
                            for item, emb in zip(facts, fact_embs):
                                self.hippocampus.record_memory(
                                    text=item["text"],
                                    category="nauka_z_sieci",
                                    importance=item.get("importance", 4),
                                    day=self.day,
                                    embedding=emb,
                                    source_url=doc.url,
                                    topic=topic,
                                )
                                topic_facts_count += 1
                                total_facts += 1
                                self.study_state["facts_learned"] = total_facts
                            for s, rel, d in extracted.get("triples", []):
                                self.hippocampus.record_fact(s, rel, d, day=self.day, source="AUTO_WEB")

                with self.lock:
                    self.hippocampus.studied_topics[topic.lower()] = {
                        "topic": topic,
                        "mastery": 85.0 if topic_facts_count > 0 else 40.0,
                        "facts_learned": topic_facts_count,
                        "pages_read": len(docs),
                        "day": self.day,
                        "summary": f"Zebrano {topic_facts_count} faktów.",
                        "sources": sources_list[-4:],
                    }
                    elapsed_pct = min(99.0, max(2.0, ((mins * 60.0 - max(0.0, deadline - time.monotonic())) / (mins * 60.0)) * 100.0))
                    self.study_state["mastery"] = round(elapsed_pct, 1)

                since_last_sleep += 1
                if self.auto_sleep and (
                    since_last_sleep >= 4
                    or len(self.hippocampus.short_term_memories) >= self.cfg.auto_sleep_pending_threshold
                ):
                    self.study_state["current_action"] = "Sen informacyjny: łączenie zdobytej wiedzy (NREM + REM na GPU)..."
                    self.sleep_now(visual_pacing=False, _from_nested=True)
                    sleeps_done += 1
                    since_last_sleep = 0

            if self.hippocampus.short_term_memories:
                self.study_state["current_action"] = "Końcowa konsolidacja wiedzy na GPU..."
                self.sleep_now(visual_pacing=False, _from_nested=True)
                sleeps_done += 1

            with self.lock:
                self._append_chat(
                    "assistant",
                    f"✅ Zakończyłem ciągłą naukę z sieci! Przerobiłem **{topics_done} tematów** "
                    f"({total_pages} stron, **+{total_facts} faktów**, {sleeps_done} cykli snu na GPU). "
                    f"Możesz teraz ze mną rozmawiać i pytać o zdobytą wiedzę!",
                    meta={"kind": "study_done", "topic": "Auto-nauka", "facts": total_facts},
                    add_to_working=False,
                )
            self.save_state()
            return {"topics": topics_done, "pages": total_pages, "facts": total_facts, "sleeps": sleeps_done}
        finally:
            with self.lock:
                self.study_state["active"] = False
                self.study_state["current_action"] = ""
                if self.state_mode == "STUDYING":
                    self.state_mode = "AWAKE"
            self._release_op("study")

    # ── Detect natural Polish "study from the web" intent ─────────────────────
    @staticmethod
    def _parse_web_study_intent(msg: str) -> str | None:
        s = msg.strip()
        low = s.lower()
        for prefix in ("/ucz ", "/study ", "/zbadaj ", "/naucz "):
            if low.startswith(prefix):
                return s[len(prefix) :].strip()
        low_stripped = _strip_pl(low)
        m = re.match(
            r"^(?:(?:sam\s+)?(?:sie\s+)?naucz(?:\s+sie)?\s+(?:z\s+(?:sieci|internetu)\s+)?(?:o\s+|na\s+temat\s+)?|"
            r"dowiedz\s+sie\s+(?:z\s+(?:sieci|internetu)\s+)?(?:o\s+|czym\s+jest\s+)?|"
            r"zbadaj\s+(?:w\s+(?:sieci|internecie)\s+)?(?:temat\s+)?|"
            r"poczytaj\s+(?:w\s+(?:sieci|internecie)\s+)?(?:o\s+)?|"
            r"przeczytaj\s+w\s+(?:sieci|internecie)\s+(?:o\s+)?)(.+)$",
            low_stripped,
        )
        if m:
            topic = s[m.start(1) : m.end(1)].strip(" .!?")
            if len(topic) >= 2:
                return topic
        return None

    # ── Autonomous Web Learning ("Study until Mastery" + Sleep Consolidation) ─
    def start_study_async(self, topic: str, max_rounds: int | None = None, consolidate_after: bool = True) -> dict[str, Any]:
        clean = topic.strip()
        if not clean:
            raise ValueError("Podaj temat do nauki.")
        if not self.language_cortex.ready:
            raise RuntimeError(self.language_cortex.status_message)

        rounds = max(1, min(8, int(max_rounds or self.cfg.study_max_rounds)))
        self._acquire_op("study")
        with self.lock:
            self._study_stop_event.clear()
            self.state_mode = "STUDYING"
            self.study_state = {
                "active": True,
                "topic": clean,
                "round": 0,
                "max_rounds": rounds,
                "mastery": 0.0,
                "pages_read": 0,
                "facts_learned": 0,
                "current_action": f"Rozpoczynam badanie tematu: {clean}",
                "sources": [],
                "summary": "",
            }
        try:
            t = threading.Thread(
                target=self.study_topic_online,
                args=(clean, rounds, consolidate_after, True),
                daemon=True,
            )
            t.start()
        except Exception:
            with self.lock:
                self.study_state["active"] = False
                self.state_mode = "AWAKE"
            self._release_op("study")
            raise
        return {"started": True, "topic": clean, "max_rounds": rounds}

    def stop_study(self) -> None:
        self._study_stop_event.set()
        if self.study_state["active"]:
            self.study_state["current_action"] = "Zatrzymywanie nauki po bieżącym kroku..."

    def study_topic_online(
        self,
        topic: str,
        max_rounds: int | None = None,
        consolidate_after: bool = True,
        _op_acquired: bool = False,
    ) -> dict[str, Any]:
        """
        Autonomously researches `topic` online (Wikipedia PL/EN + DuckDuckGo web pages + direct URLs),
        analyzes content on GPU into Hippocampus facts & concept graph, self-quizzes for gaps,
        sleeps on GPU to consolidate if tired, and continues until mastery!
        """
        raw_topic = topic.strip()
        if not raw_topic:
            raise ValueError("Podaj temat do nauki.")
        clean_topic = raw_topic if is_http_url(raw_topic) else clean_search_query(raw_topic)
        if not is_http_url(raw_topic) and len(clean_topic) >= len(raw_topic) - 1:
            clean_topic = raw_topic
        rounds = max(1, min(8, int(max_rounds or self.cfg.study_max_rounds)))
        if not self.language_cortex.ready:
            if _op_acquired:
                with self.lock:
                    self.study_state["active"] = False
                    self.state_mode = "AWAKE"
                self._release_op("study")
            raise RuntimeError(self.language_cortex.status_message)

        if not _op_acquired:
            self._acquire_op("study")

        with self.lock:
            self._study_stop_event.clear()
            self.state_mode = "STUDYING"
            self.study_state = {
                "active": True,
                "topic": clean_topic,
                "round": 0,
                "max_rounds": rounds,
                "mastery": 0.0,
                "pages_read": 0,
                "facts_learned": 0,
                "current_action": f"Rozpoczynam badanie tematu: {clean_topic}",
                "sources": [],
                "summary": "",
            }
            self._log("STUDY", f"🌐 Rozpoczęto autonomiczną naukę z sieci: '{clean_topic}' (do {rounds} rund).")
            self._append_chat(
                "assistant",
                f"🌐 Rozpoczynam badanie tematu „{clean_topic}” w sieci! Szukam źródeł, analizuję treść na GPU "
                f"i sprawdzam braki wiedzy aż opanuję ten temat (a po zebraniu wiedzy sen połączy nowe fakty w grafie i pamięci trwałej).",
                meta={"kind": "study_start", "topic": clean_topic},
                add_to_working=False,
            )

        seen_urls: set[str] = set()
        learned_texts: list[str] = []
        if is_http_url(clean_topic):
            pending_queries: list[str] = [clean_topic]
            rounds = 1
            self.study_state["max_rounds"] = 1
        else:
            normalized_qs = self.language_cortex.normalize_study_queries(clean_topic)
            pending_queries = []
            for q in [clean_topic] + normalized_qs:
                if q and q.lower() not in {x.lower() for x in pending_queries}:
                    pending_queries.append(q)
            if normalized_qs and (len(clean_topic.split()) >= 3 or normalized_qs[0].lower() != clean_topic.lower()):
                pending_queries = normalized_qs + [clean_topic]

        sources_list: list[dict[str, str]] = []
        mastery = 0.0
        summary = ""
        sleeps_during_study = 0

        try:
            for r_idx in range(1, rounds + 1):
                if self._study_stop_event.is_set():
                    break
                self.study_state["round"] = r_idx
                query = pending_queries.pop(0) if pending_queries else f"{clean_topic} szczegóły zastosowania"
                self.study_state["current_action"] = f"Runda {r_idx}/{rounds}: szukam źródeł dla „{query}”..."
                with self.lock:
                    self._log("STUDY", f"Runda {r_idx}/{rounds}: wyszukiwanie w sieci '{query}'...")

                docs = gather_web_sources(
                    query=query,
                    seen_urls=seen_urls,
                    max_docs=self.cfg.study_pages_per_round,
                    timeout=self.cfg.study_http_timeout,
                    max_workers=self.cfg.study_fetch_workers,
                )
                if not docs:
                    with self.lock:
                        self._log("STUDY", f"Brak nowych stron dla zapytania '{query}'.")
                    if not pending_queries:
                        break
                    continue

                for doc in docs:
                    if self._study_stop_event.is_set():
                        break
                    self.study_state["current_action"] = f"Runda {r_idx}/{rounds}: analizuję „{doc.title}” na GPU..."
                    sources_list.append({"title": doc.title, "url": doc.url, "source": doc.source})
                    self.study_state["sources"] = sources_list[-12:]
                    self.study_state["pages_read"] += 1

                    # ALWAYS preserve the authoritative verbatim Polish lead definition first!
                    lead_def = extract_lead_definition(doc)
                    if lead_def:
                        emb_lead = self.language_cortex.encode_text(lead_def)
                        with self.lock:
                            lead_trace = self.hippocampus.record_memory(
                                text=lead_def,
                                category="nauka_z_sieci",
                                importance=5,
                                day=self.day,
                                embedding=emb_lead,
                                source_url=doc.url,
                                topic=clean_topic,
                            )
                            if lead_trace.text not in learned_texts:
                                learned_texts.append(lead_trace.text)
                                self.study_state["facts_learned"] = len(learned_texts)

                    chunks = chunk_text(doc.text, focus_query=query, max_chars=2400, max_chunks=2)
                    for ch in chunks:
                        if self._study_stop_event.is_set():
                            break
                        extracted = self.language_cortex.study_document_chunk(clean_topic, doc.title, ch)
                        facts = extracted.get("facts", [])
                        fact_embs = self.language_cortex.encode_texts([item["text"] for item in facts])
                        with self.lock:
                            for item, emb in zip(facts, fact_embs):
                                trace = self.hippocampus.record_memory(
                                    text=item["text"],
                                    category="nauka_z_sieci",
                                    importance=item["importance"],
                                    day=self.day,
                                    embedding=emb,
                                    source_url=doc.url,
                                    topic=clean_topic,
                                )
                                if trace.text not in learned_texts:
                                    learned_texts.append(trace.text)
                                    self.study_state["facts_learned"] = len(learned_texts)

                            for s, rel, d in extracted.get("triples", []):
                                self.hippocampus.record_fact(s, rel, d, day=self.day, source="WEB_STUDY")

                # Self-quiz / metacognition at the end of each round
                self.study_state["current_action"] = f"Runda {r_idx}/{rounds}: sprawdzam poziom opanowania i luki wiedzy..."
                assessment = self.language_cortex.assess_study_mastery(clean_topic, learned_texts, round_num=r_idx)
                mastery = float(assessment["mastery"])
                summary = assessment.get("summary", "")
                self.study_state["mastery"] = mastery
                self.study_state["summary"] = summary

                with self.lock:
                    self._log(
                        "STUDY",
                        f"Po rundzie {r_idx}: opanowanie '{clean_topic}' = {mastery:.0f}% "
                        f"({len(learned_texts)} faktów z {self.study_state['pages_read']} stron).",
                    )
                if mastery >= self.cfg.study_target_mastery:
                    with self.lock:
                        self._log("STUDY", f"Osiągnięto wysoki poziom opanowania ({mastery:.0f}%) — kończę badania!")
                    break

                for nq in assessment.get("next_queries", []):
                    if nq not in pending_queries:
                        pending_queries.append(nq)

            # Record topic mastery in Hippocampus metadata & preserve unrelated working_chat turns
            keyword_source = clean_topic + " " + " ".join(s.get("title", "") for s in sources_list)
            topic_keywords = [
                _strip_pl(w.lower())
                for w in re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9+#]{4,}", keyword_source)
                if _strip_pl(w.lower()) not in ("http", "https", "html")
            ]
            with self.lock:
                should_record_topic = not (
                    self._study_stop_event.is_set()
                    and self.study_state["pages_read"] == 0
                    and clean_topic.lower() in self.hippocampus.studied_topics
                )
                if should_record_topic:
                    self.hippocampus.studied_topics[clean_topic.lower()] = {
                        "topic": clean_topic,
                        "mastery": mastery,
                        "facts_learned": len(learned_texts),
                        "pages_read": self.study_state["pages_read"],
                        "day": self.day,
                        "summary": summary,
                        "sources": sources_list[:10],
                    }

                # Remove only prior assistant "I don't know" turns about this specific topic
                filtered_chat: list[dict[str, str]] = []
                for turn in self.working_chat:
                    if turn.get("role") == "assistant":
                        low_c = _strip_pl(turn.get("content", "").lower())
                        has_ignorance = ("nie wiem" in low_c) or ("/ucz" in low_c)
                        mentions_topic = bool(topic_keywords) and any(kw in low_c for kw in topic_keywords)
                        if has_ignorance and mentions_topic:
                            continue
                    filtered_chat.append(turn)
                self.working_chat = filtered_chat

            # Consolidate newly learned knowledge via GPU Sleep if requested
            sleep_rep = None
            if consolidate_after and self.hippocampus.short_term_memories and not self._study_stop_event.is_set():
                self.study_state["current_action"] = "Konsolidacja zdobytej wiedzy we śnie (NREM + REM na GPU)..."
                sleep_rep = self.sleep_now(visual_pacing=False, _from_nested=True)
                sleeps_during_study += 1

            result = {
                "topic": clean_topic,
                "mastery": mastery,
                "pages_read": self.study_state["pages_read"],
                "facts_learned": len(learned_texts),
                "sleeps": sleeps_during_study,
                "summary": summary,
                "sources": sources_list,
                "sleep_report": sleep_rep.to_dict() if sleep_rep else None,
            }

            summary_line = f"\nPodsumowanie: {summary}" if summary else ""
            with self.lock:
                self._append_chat(
                    "assistant",
                    f"✅ Zakończyłem naukę tematu „{clean_topic}”! Przeanalizowałem {result['pages_read']} źródeł z sieci, "
                    f"wyciągnąłem {result['facts_learned']} faktów (poziom opanowania: {mastery:.0f}%)"
                    f"{' i utrwaliłem je podczas snu w pamięci trwałej oraz grafie pojęć na GPU' if sleeps_during_study else ' (zapisane w hipokampie)'}."
                    f"{summary_line}\nMożesz mnie teraz o to przepytać!",
                    meta={"kind": "study_done", "topic": clean_topic, "mastery": mastery, "facts": result["facts_learned"]},
                    add_to_working=False,
                )
            self.save_state()
            return result
        finally:
            with self.lock:
                self.study_state["active"] = False
                self.study_state["current_action"] = ""
                if self.state_mode == "STUDYING":
                    self.state_mode = "AWAKE"
            self._release_op("study")

    def train_language_now(self, epochs: int | None = None) -> dict[str, Any]:
        self._acquire_op("study")
        try:
            with self.lock:
                all_m = [
                    m.text for m in (self.hippocampus.short_term_memories + self.hippocampus.semantic_memories)
                    if not m.superseded
                ]
                ep_m = [m.text for m in self.hippocampus.episodic_memories]
                user_p = [
                    m.text for m in (self.hippocampus.short_term_memories + self.hippocampus.semantic_memories)
                    if not m.superseded and m.category == "użytkownik"
                ]
                turns = list(self.chat_log[-80:])

            res = self.language_cortex.train_language_cortex(
                all_memories=all_m,
                episodic_memories=ep_m,
                chat_turns=turns,
                user_profile=user_p,
                epochs=epochs,
            )
            with self.lock:
                self._log(
                    "STUDY",
                    f"🗣️ Wykształcono model językowy na GPU ({res['dialogues']} dialogów, "
                    f"loss {res['loss_start']}→{res['loss_end']}, płynność {res['fluency']:.0f}%).",
                )
                self._append_chat(
                    "assistant",
                    f"🗣️ Przeprowadziłem samodzielny trening językowy na GPU ({res['dialogues']} dialogów w {res['duration_ms']:.0f} ms). "
                    f"Strata spadła z {res['loss_start']} do {res['loss_end']}, a moja płynność językowa wynosi teraz **{res['fluency']:.0f}%**!",
                    meta={"kind": "study_done", "topic": "Trening językowy GPU"},
                    add_to_working=False,
                )
            self.save_state()
            return res
        finally:
            self._release_op("study")

    def _prepare_chat_turn(self, msg: str) -> dict[str, Any]:
        # 1) Resolve conversational pronouns/follow-ups & query intent
        with self.lock:
            turns_snap = list(self.working_chat)
        ctx = self.language_cortex.resolve_conversational_context(msg, turns_snap)
        search_q = ctx["search_query"]
        active_subject = ctx["active_subject"]
        intent_info = self.language_cortex.analyze_query_intent(search_q)
        if intent_info["intent"] == "code_request":
            code_q = intent_info.get("code_task") or active_subject
            if code_q:
                search_q = code_q
            sub_concepts = [code_q] if code_q else []
            use_multihop_ctx = False
        else:
            sub_concepts = [sc for sc in intent_info.get("sub_concepts", []) if sc.strip()][:5]
            use_multihop_ctx = intent_info["intent"] != "self_capability" and bool(sub_concepts)

        if use_multihop_ctx:
            vecs = self.language_cortex.encode_texts([search_q] + sub_concepts)
            q_vec = vecs[0]
            sub_vecs = vecs[1:]
        else:
            q_vec = self.language_cortex.encode_text(search_q)
            sub_vecs = []

        extracted = self.language_cortex.extract_memories_from_turn(msg)
        mem_items = extracted.get("memories", [])
        mem_embs = self.language_cortex.encode_texts([item["text"] for item in mem_items])

        # 2) Fast RAM recall, multi-hop graph paths, concept resolution & recording INSIDE self.lock
        with self.lock:
            words = [w.strip(".,!?;:\"'()").lower() for w in search_q.split() if len(w.strip(".,!?;:\"'()")) >= 3]
            if use_multihop_ctx:
                concept_paths = self.hippocampus.find_concept_paths(
                    sub_concepts + words,
                    max_hops=3,
                    max_paths=6,
                )
            else:
                concept_paths = []

            bridge_terms: list[str] = []
            seen_bridges: set[str] = set()
            concept_hints: list[str] = []
            for p in concept_paths[:3]:
                if p.get("summary"):
                    concept_hints.append(f"🔗 {p['summary']}")
                for br in p.get("bridges", []):
                    bn = br.strip().lower()
                    if bn and bn not in seen_bridges:
                        seen_bridges.add(bn)
                        bridge_terms.append(br.strip())

            seen_concepts: set[str] = set()
            for w in sub_concepts + words:
                resolved = self.concept_cortex.resolve_concept(w)
                if resolved and resolved not in seen_concepts:
                    seen_concepts.add(resolved)
                    assocs = self.concept_cortex.query_associations(resolved, top_k=4)
                    if assocs:
                        a_str = ", ".join(f"{a['concept']} ({a['via_relation']})" for a in assocs)
                        concept_hints.append(f"{resolved} → {a_str}")
                        for a in assocs[:2]:
                            ac = a["concept"].strip().lower()
                            if ac and ac not in seen_bridges and a.get("score", 0.0) >= 0.45:
                                seen_bridges.add(ac)
                                bridge_terms.append(a["concept"])

            if use_multihop_ctx and (len(sub_concepts) >= 2 or bridge_terms):
                recalled = self.hippocampus.search_memories_multihop(
                    query_vec=q_vec,
                    query_text=search_q,
                    sub_queries=sub_concepts,
                    sub_vecs=sub_vecs,
                    bridge_terms=bridge_terms,
                    current_day=self.day,
                    top_hippo=self.cfg.recall_top_hippo,
                    top_semantic=self.cfg.recall_top_semantic,
                    top_rem=self.cfg.recall_top_rem,
                )
            else:
                recalled = self.hippocampus.search_memories(
                    query_vec=q_vec,
                    query_text=search_q,
                    current_day=self.day,
                    top_hippo=self.cfg.recall_top_hippo,
                    top_semantic=self.cfg.recall_top_semantic,
                    top_rem=self.cfg.recall_top_rem,
                )

            hippo_texts = [m.text for m in recalled["hippocampus"]]
            sem_texts = [m.text for m in recalled["semantic"]]
            rem_texts = [m.text for m in recalled["rem"]]
            user_profile_texts = [m.text for m in recalled.get("user_profile", [])]
            episodic_texts = [m.text for m in recalled.get("episodic", [])]

            learned_memories: list[dict[str, Any]] = []
            for item, emb in zip(mem_items, mem_embs):
                trace = self.hippocampus.record_memory(
                    text=item["text"],
                    category=item["category"],
                    importance=item["importance"],
                    day=self.day,
                    embedding=emb,
                )
                learned_memories.append(trace.to_public_dict())
                if trace.text not in hippo_texts:
                    hippo_texts.insert(0, trace.text)

            learned_triples: list[dict[str, Any]] = []
            for s, r, d in extracted.get("triples", []):
                f = self.hippocampus.record_fact(s, r, d, day=self.day)
                learned_triples.append(f.to_dict())

            return {
                "hippo_texts": hippo_texts,
                "sem_texts": sem_texts,
                "rem_texts": rem_texts,
                "user_profile": user_profile_texts,
                "episodic_texts": episodic_texts,
                "active_subject": active_subject,
                "sub_concepts": sub_concepts,
                "concept_paths": concept_paths,
                "concept_hints": concept_hints,
                "learned_memories": learned_memories,
                "learned_triples": learned_triples,
                "recent_turns": list(self.working_chat),
                "day": self.day,
                "sleeps": self.total_sleeps,
                "energy": self.energy,
                "state_mode": self.state_mode,
            }

    def _finalize_chat_turn(self, msg: str, reply: str, prep: dict[str, Any]) -> dict[str, Any]:
        learned_memories = prep["learned_memories"]
        hippo_texts = prep["hippo_texts"]
        sem_texts = prep["sem_texts"]
        rem_texts = prep["rem_texts"]
        concept_hints = prep["concept_hints"]

        # Record substantive user turns as episodic conversation memories (separate from short_term_memories)
        low_msg = _strip_pl(msg.lower()).strip(" ?!.,;:")
        skip_ep = (
            len(msg) < 12
            or any(p in low_msg for p in ("ale to nie prawda", "nieprawda", "bzdura", "pisalimi", "pisalismy", "poprzednie waidomosci"))
            or low_msg in ("czesc", "hej", "witaj", "dzien dobry", "dobranoc", "dzieki", "dziekuje", "ale co")
        )
        ep_text = f"W rozmowie (Dzień {prep['day']}) pisałeś: „{msg[:120]}”." if not skip_ep else ""
        ep_emb = self.language_cortex.encode_text(ep_text) if ep_text else []

        with self.lock:
            if ep_text:
                self.hippocampus.record_episodic(ep_text, day=self.day, embedding=ep_emb)
            self._append_chat("user", msg, meta={"learned_count": len(learned_memories)})
            self._append_chat(
                "assistant",
                reply,
                meta={
                    "recalled_semantic": sem_texts[:8],
                    "recalled_hippocampus": hippo_texts[:8],
                    "recalled_rem": rem_texts[:6],
                    "concept_hints": concept_hints[:6],
                    "learned": [m["text"] for m in learned_memories],
                    "thinking": prep.get("thinking", [])[:8],
                },
            )

            self.energy = self.cfg.max_energy
            self.state_mode = "AWAKE"

            if learned_memories:
                self._log(
                    "TEACH",
                    f"Hipokamp zapamiętał z rozmowy ({len(learned_memories)}): "
                    + "; ".join(m["text"] for m in learned_memories[:2]),
                )
            pending_count = len(self.hippocampus.short_term_memories)
            should_auto_sleep = self.auto_sleep and pending_count >= self.cfg.auto_sleep_pending_threshold

        slept = False
        sleep_report_dict = None
        if should_auto_sleep:
            with self.lock:
                self._log(
                    "SLEEP",
                    f"Bufor hipokampu osiągnął {pending_count} nowych informacji — sen łączy je z wiedzą na GPU!",
                )
            rep = self.sleep_now(visual_pacing=False, _from_nested=True)
            slept = True
            sleep_report_dict = rep.to_dict()
        else:
            self.save_state()

        return {
            "reply": reply,
            "slept": slept,
            "sleep_report": sleep_report_dict,
            "learned_memories": learned_memories,
            "learned_triples": prep["learned_triples"],
            "recalled_memories": {
                "hippocampus": hippo_texts,
                "semantic": sem_texts,
                "rem": rem_texts,
            },
            "concept_hints": concept_hints[:6],
            "energy": round(self.energy, 1),
            "state_mode": self.state_mode,
        }

    def _quick_web_lookup(self, topic: str, msg: str, prep: dict[str, Any]) -> list[str]:
        """Lightweight online lookup used when Auto-web is enabled and memory lacks an answer."""
        clean_topic = topic.strip() if is_http_url(topic.strip()) else clean_search_query(topic.strip())
        if not clean_topic:
            clean_topic = topic.strip()

        normalized_qs = self.language_cortex.normalize_study_queries(clean_topic)
        queries = [clean_topic] + [q for q in normalized_qs if q and q.lower() != clean_topic.lower()]

        seen_urls: set[str] = set()
        docs = []
        for q in queries[:2]:
            docs = gather_web_sources(
                query=q,
                seen_urls=seen_urls,
                max_docs=2,
                timeout=min(8, self.cfg.study_http_timeout),
                max_workers=2,
            )
            if docs:
                break

        if not docs:
            return []

        source_titles: list[str] = []
        new_texts: list[str] = []
        for doc in docs:
            source_titles.append(doc.title)
            lead_def = extract_lead_definition(doc)
            if lead_def:
                emb_lead = self.language_cortex.encode_text(lead_def)
                with self.lock:
                    trace = self.hippocampus.record_memory(
                        text=lead_def,
                        category="nauka_z_sieci",
                        importance=5,
                        day=self.day,
                        embedding=emb_lead,
                        source_url=doc.url,
                    )
                    prep["learned_memories"].append(trace.to_public_dict())
                    new_texts.append(trace.text)

            chunk = doc.text[:2200]
            extracted = self.language_cortex.study_document_chunk(clean_topic, doc.title, chunk)
            facts = extracted.get("facts", [])[:5]
            if facts:
                embs = self.language_cortex.encode_texts([item["text"] for item in facts])
                with self.lock:
                    for item, emb in zip(facts, embs):
                        trace = self.hippocampus.record_memory(
                            text=item["text"],
                            category="nauka_z_sieci",
                            importance=item.get("importance", 4),
                            day=self.day,
                            embedding=emb,
                            source_url=doc.url,
                            topic=clean_topic,
                        )
                        prep["learned_memories"].append(trace.to_public_dict())
                        new_texts.append(trace.text)
                    for s, r, d in extracted.get("triples", [])[:6]:
                        f = self.hippocampus.record_fact(s, r, d, day=self.day)
                        prep["learned_triples"].append(f.to_dict())

        q_vec = self.language_cortex.encode_text(msg)
        with self.lock:
            recalled = self.hippocampus.search_memories(
                query_vec=q_vec,
                query_text=msg,
                current_day=self.day,
                top_hippo=self.cfg.recall_top_hippo,
                top_semantic=self.cfg.recall_top_semantic,
                top_rem=self.cfg.recall_top_rem,
            )
            prep["hippo_texts"] = [m.text for m in recalled["hippocampus"]]
            for t in reversed(new_texts):
                if t not in prep["hippo_texts"]:
                    prep["hippo_texts"].insert(0, t)
            prep["sem_texts"] = [m.text for m in recalled["semantic"]]
            prep["rem_texts"] = [m.text for m in recalled["rem"]]
            sub_terms = prep.get("sub_concepts") or [clean_topic]
            prep["concept_paths"] = self.hippocampus.find_concept_paths(sub_terms, max_hops=3, max_paths=6)
            self._log("WEB", f"Auto-sieć sprawdziła '{clean_topic}' ({len(source_titles)} źródła, +{len(new_texts)} faktów).")

        return source_titles

    def _compute_chat_reply(self, msg: str, prep: dict[str, Any]) -> str:
        reply, missing_topic = self.language_cortex._chat_reply_with_meta(
            user_message=msg,
            recent_turns=prep["recent_turns"],
            hippocampus_memories=prep["hippo_texts"][: self.cfg.recall_top_hippo + 6],
            semantic_memories=prep["sem_texts"][: self.cfg.recall_top_semantic + 8],
            rem_reflections=prep["rem_texts"][: self.cfg.recall_top_rem + 4],
            concept_hints=prep["concept_hints"][:6],
            day=prep["day"],
            sleeps=prep["sleeps"],
            energy=prep["energy"],
            state_mode=prep["state_mode"],
            user_profile=prep.get("user_profile", []),
            episodic_memories=prep.get("episodic_texts", []),
            active_subject=prep.get("active_subject", ""),
            concept_paths=prep.get("concept_paths", []),
        )
        thinking = list(self.language_cortex.last_thinking_steps)
        if self.auto_web and missing_topic:
            prev_count = len(prep["learned_memories"])
            thinking.append(f"Uruchamiam Auto-sieć dla tematu „{missing_topic}”...")
            sources = self._quick_web_lookup(missing_topic, msg, prep)
            if sources:
                reply2, still_missing = self.language_cortex._chat_reply_with_meta(
                    user_message=msg,
                    recent_turns=prep["recent_turns"],
                    hippocampus_memories=prep["hippo_texts"][: self.cfg.recall_top_hippo + 6],
                    semantic_memories=prep["sem_texts"][: self.cfg.recall_top_semantic + 8],
                    rem_reflections=prep["rem_texts"][: self.cfg.recall_top_rem + 4],
                    concept_hints=prep["concept_hints"][:6],
                    day=prep["day"],
                    sleeps=prep["sleeps"],
                    energy=prep["energy"],
                    state_mode=prep["state_mode"],
                    user_profile=prep.get("user_profile", []),
                    episodic_memories=prep.get("episodic_texts", []),
                    active_subject=prep.get("active_subject", ""),
                    concept_paths=prep.get("concept_paths", []),
                )
                src_note = ", ".join(sources[:2])
                if not still_missing:
                    thinking.extend(self.language_cortex.last_thinking_steps[-2:])
                    prep["thinking"] = thinking
                    return f"{reply2}\n\n*Sprawdzone w sieci:* {src_note}"
                fresh_facts = [m["text"] for m in prep["learned_memories"][prev_count:prev_count + 3]]
                if fresh_facts:
                    thinking.append(f"Zsyntetyzowano odpowiedź z {len(fresh_facts)} świeżych faktów z sieci ({src_note}).")
                    prep["thinking"] = thinking
                    joined = " ".join(f.rstrip(".") + "." for f in fresh_facts)
                    return f"{joined}\n\n*Sprawdzone w sieci:* {src_note}"
        prep["thinking"] = thinking
        return reply

    # ── Natural Polish Chat + Automatic Human-like Memory Encoding ────────────
    def chat(self, user_message: str) -> dict[str, Any]:
        msg = user_message.strip()
        if not msg:
            raise ValueError("Wiadomość nie może być pusta.")

        if msg.lower() in ("/sen", "/sleep", "idź spać", "idz spac"):
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            rep = self.sleep_now(visual_pacing=False)
            with self.lock:
                last_msg = self.chat_log[-1] if self.chat_log else {"content": "Obudziłem się!"}
            return {
                "reply": last_msg["content"],
                "slept": True,
                "sleep_report": rep.to_dict(),
                "learned_memories": [],
                "recalled_memories": {},
            }

        if msg.lower() in ("/szkola", "/szkoła", "/podstawy"):
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            res = self.learn_elementary_foundation(consolidate=True)
            with self.lock:
                last_msg = self.chat_log[-1] if self.chat_log else {"content": "Wgrano pakiet Szkoła Podstawowa."}
            return {
                "reply": last_msg["content"],
                "slept": bool(res.get("sleep_report")),
                "sleep_report": res.get("sleep_report"),
                "learned_memories": [],
                "recalled_memories": {},
            }

        if msg.lower() in ("/wyksztalc", "/wykształć", "/trenuj"):
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            res = self.train_language_now()
            with self.lock:
                last_msg = self.chat_log[-1] if self.chat_log else {"content": "Wytrenowano model językowy na GPU."}
            return {
                "reply": last_msg["content"],
                "slept": False,
                "learned_memories": [],
                "recalled_memories": {},
                "language_train": res,
            }

        if msg.lower() in ("/szkola siec", "/szkoła sieć", "/szkola web"):
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            info = self.start_curriculum_async(deep_web=True, max_topics=12)
            return {
                "reply": f"📚 Rozpoczynam autonomiczny kurs szkolny z Wikipedii ({info['topics']} tematów + fundamenty). Obserwuj pasek postępu u góry!",
                "study_started": True,
                "slept": False,
                "learned_memories": [],
                "recalled_memories": {},
            }

        if msg.lower().startswith("/ucz turbo"):
            parts = msg.split()
            cnt = 200
            if len(parts) >= 3 and parts[2].isdigit():
                cnt = int(parts[2])
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            info = self.start_turbo_learn_async(topics_count=cnt)
            return {
                "reply": f"⚡ Rozpoczynam Mega Turbo-naukę ({info['topics']} tematów równolegle + sen GPU + trening języka)! Obserwuj pasek postępu u góry.",
                "study_started": True,
                "slept": False,
                "learned_memories": [],
                "recalled_memories": {},
            }

        if msg.lower().startswith("/ucz auto"):
            parts = msg.split()
            mins = 60
            if len(parts) >= 3 and parts[2].isdigit():
                mins = int(parts[2])
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            info = self.start_continuous_learning_async(duration_minutes=mins)
            return {
                "reply": f"🚀 Rozpoczynam ciągłą autonomiczną naukę z sieci na {info['duration_minutes']} min! Obserwuj pasek postępu u góry.",
                "study_started": True,
                "slept": False,
                "learned_memories": [],
                "recalled_memories": {},
            }

        study_topic = self._parse_web_study_intent(msg)
        if study_topic:
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            info = self.start_study_async(study_topic)
            return {
                "reply": f"🌐 Przyjąłem! Zaczynam samodzielnie badać w sieci temat „{study_topic}” (do {info['max_rounds']} rund). Obserwuj postęp u góry!",
                "study_started": True,
                "topic": study_topic,
                "slept": False,
                "learned_memories": [],
                "recalled_memories": {},
            }

        if not self.language_cortex.ready:
            if not self.language_cortex.loading:
                self.language_cortex.load_async()
            raise RuntimeError(self.language_cortex.status_message)

        self._acquire_op("chat")
        try:
            prep = self._prepare_chat_turn(msg)
            reply = self._compute_chat_reply(msg, prep)
            return self._finalize_chat_turn(msg, reply, prep)
        finally:
            self._release_op("chat")

    def chat_stream(self, user_message: str) -> Iterator[dict[str, Any]]:
        msg = user_message.strip()
        if not msg:
            raise ValueError("Wiadomość nie może być pusta.")

        if msg.lower() in ("/sen", "/sleep", "idź spać", "idz spac", "/szkola", "/szkoła", "/podstawy", "/szkola siec", "/szkoła sieć", "/szkola web", "/wyksztalc", "/wykształć", "/trenuj") or msg.lower().startswith("/ucz auto") or msg.lower().startswith("/ucz turbo"):
            res = self.chat(msg)
            yield {"type": "done", **res}
            return

        study_topic = self._parse_web_study_intent(msg)
        if study_topic:
            res = self.chat(msg)
            yield {"type": "done", **res}
            return

        if not self.language_cortex.ready:
            if not self.language_cortex.loading:
                self.language_cortex.load_async()
            raise RuntimeError(self.language_cortex.status_message)

        self._acquire_op("chat")
        try:
            prep = self._prepare_chat_turn(msg)
            reply = self._compute_chat_reply(msg, prep)
            yield {
                "type": "meta",
                "learned_memories": prep["learned_memories"],
                "learned_triples": prep["learned_triples"],
                "recalled_memories": {
                    "hippocampus": prep["hippo_texts"],
                    "semantic": prep["sem_texts"],
                    "rem": prep["rem_texts"],
                },
                "concept_hints": prep["concept_hints"][:6],
                "thinking": prep.get("thinking", [])[:8],
            }

            import time
            tokens = re.findall(r"\S+\s*", reply)
            if not tokens and reply:
                tokens = [reply]
            for tok in tokens:
                yield {"type": "token", "delta": tok}
                time.sleep(0.012)

            final_res = self._finalize_chat_turn(msg, reply, prep)
            yield {"type": "done", **final_res}
        finally:
            self._release_op("chat")

    def sleep_now(self, visual_pacing: bool = False, _from_nested: bool = False) -> SleepReport:
        if not _from_nested:
            self._acquire_op("sleep")
        try:
            with self.lock:
                night_num = self.total_sleeps + 1

            def on_stage(stage: str) -> None:
                labels = {
                    "NREM_LIGHT": "Sen NREM (płytki): selekcja wspomnień, wykrywanie sprzeczności, krzywa zapominania...",
                    "NREM_DEEP": "Sen NREM (głęboki): trening sieci pojęć (EWC) i konsolidacja rdzenia wiedzy na GPU...",
                    "REM_DREAMING": "Sen REM (marzenia): łączenie wspomnień w nowe refleksje i połączenia grafu...",
                }
                with self.lock:
                    self.state_mode = stage
                    self._log("SLEEP", labels.get(stage, stage))

            report = self.sleep_engine.run_sleep_cycle(
                night_number=night_num,
                stage_callback=on_stage,
                visual_pacing=visual_pacing,
                state_lock=self.lock,
            )

            with self.lock:
                self.total_sleeps += 1
                self.day += 1
                self.energy = self.cfg.max_energy
                self.state_mode = "AWAKE"
                # Keep last 4 turns for smooth conversational continuity across sleep while long-term lives in Cortex
                self.working_chat = self.working_chat[-4:]

                self.sleep_history.append(report.to_dict())
                if len(self.sleep_history) > 40:
                    self.sleep_history = self.sleep_history[-40:]

                nrem_note = (
                    f", sieć PyTorch ({report.lora_samples} próbek, loss {report.lora_loss_start}→{report.lora_loss_end})"
                    if report.lora_trained
                    else ""
                )
                msg = (
                    f"Budzenie (Dzień {self.day}) po {report.duration_ms:.0f} ms na {report.device_name}. "
                    f"Utrwalono {report.new_memories} wspomnień{nrem_note}, "
                    f"{len(report.rem_reflections)} refleksji REM, {len(report.rem_new_links)} połączeń grafu!"
                )
                self._log("WAKE", msg)
                for refl in report.rem_reflections:
                    self._log("REM", f"Refleksja REM: {refl}")
                should_greet = (
                    self.language_cortex.ready
                    and not self.study_state["active"]
                    and (report.new_memories > 0 or bool(report.rem_reflections))
                )

            if should_greet:
                greeting = self.language_cortex.generate_morning_greeting(
                    night_number=night_num,
                    new_mem_count=report.new_memories,
                    lora_trained=report.lora_trained,
                    rem_reflections=report.rem_reflections,
                    rem_links=report.rem_new_links,
                )
                with self.lock:
                    self._append_chat(
                        "assistant",
                        f"☀️ [Dzień {self.day} — po Nocy #{night_num}] {greeting}",
                        meta={
                            "kind": "morning_wakeup",
                            "lora_trained": report.lora_trained,
                            "new_memories": report.new_memories,
                            "rem_reflections": report.rem_reflections,
                        },
                        add_to_working=False,
                    )

            self.save_state()
            return report
        finally:
            if not _from_nested:
                self._release_op("sleep")

    def add_manual_memory(self, text: str, category: str = "wiedza", importance: int = 4) -> dict[str, Any]:
        clean = text.strip()
        if len(clean) < 3:
            raise ValueError("Treść wspomnienia musi mieć co najmniej 3 znaki.")
        cat = (category or "wiedza").strip().lower()
        imp = int(max(1, min(5, int(importance))))
        emb = (
            self.language_cortex.encode_text(clean)
            if self.language_cortex.ready
            else _fallback_hash_embed(clean)
        )
        with self.lock:
            trace = self.hippocampus.record_memory(
                text=clean,
                category=cat,
                importance=imp,
                day=self.day,
                embedding=emb,
            )
            self._log("TEACH", f"Dodano ręcznie do hipokampu [{cat}]: {trace.text[:80]}")
        self.save_state()
        return trace.to_public_dict()

    def delete_memory(self, memory_id: str) -> bool:
        with self.lock:
            deleted = self.hippocampus.delete_memory(memory_id)
            if deleted:
                self._log("TEACH", f"Usunięto wspomnienie (id={memory_id}).")
        if deleted:
            self.save_state()
        return deleted

    def delete_studied_topic(self, topic: str) -> bool:
        with self.lock:
            deleted = self.hippocampus.delete_topic(topic)
            if deleted:
                self._log("STUDY", f"Usunięto temat z listy badań: '{topic}'.")
        if deleted:
            self.save_state()
        return deleted

    def export_knowledge(self) -> dict[str, Any]:
        with self.lock:
            active_sem = [m.to_public_dict() for m in self.hippocampus.semantic_memories if not m.superseded]
            return {
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "day": self.day,
                "total_sleeps": self.total_sleeps,
                "model_id": self.cfg.language_model_id,
                "core_knowledge_summary": self.language_cortex.core_knowledge_summary,
                "studied_topics": list(self.hippocampus.studied_topics.values()),
                "semantic_memories": active_sem,
                "short_term_memories": [m.to_public_dict() for m in self.hippocampus.short_term_memories],
                "rem_reflections": [m.to_public_dict() for m in self.hippocampus.rem_reflections],
                "consolidated_facts": [f.to_dict() for f in self.hippocampus.consolidated_facts.values()],
                "rem_dream_links": [f.to_dict() for f in self.hippocampus.rem_dream_links.values()],
            }

    def save_state(self) -> Path:
        with self.lock:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = self.state_dir / "brain_weights.pt"
            meta_path = self.state_dir / "brain_memory.json"
            tmp_ckpt = self.state_dir / "brain_weights.pt.tmp"
            tmp_meta = self.state_dir / "brain_memory.json.tmp"

            torch.save(
                {
                    "concept_cortex": self.concept_cortex.export_state(),
                    "language_cortex": self.language_cortex.export_state(),
                },
                tmp_ckpt,
            )
            os.replace(tmp_ckpt, ckpt_path)

            meta = {
                "embedding_version": "standalone_pytorch_v1",
                "day": self.day,
                "energy": self.cfg.max_energy,
                "creativity_level": self.creativity_level,
                "total_sleeps": self.total_sleeps,
                "auto_sleep": self.auto_sleep,
                "auto_web": self.auto_web,
                "lora_updates_count": self.language_cortex.lora_updates_count,
                "core_knowledge_summary": self.language_cortex.core_knowledge_summary,
                "hippocampus": self.hippocampus.to_dict(),
                "chat_log": self.chat_log[-250:],
                "working_chat": self.working_chat[-self.cfg.working_chat_buffer :],
                "sleep_history": self.sleep_history,
            }
            tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_meta, meta_path)
            return ckpt_path

    def load_state(self) -> bool:
        ckpt_path = self.state_dir / "brain_weights.pt"
        meta_path = self.state_dir / "brain_memory.json"
        if not ckpt_path.exists() or not meta_path.exists():
            return False
        try:
            weights = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if "concept_cortex" in weights:
                self.concept_cortex.import_state(weights["concept_cortex"])
            if "language_cortex" in weights:
                self.language_cortex.import_state(weights["language_cortex"])
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.day = int(meta.get("day", 1))
            self.energy = self.cfg.max_energy
            self.creativity_level = max(0.0, min(1.0, float(meta.get("creativity_level", self.cfg.creativity_level))))
            self.cfg.creativity_level = self.creativity_level
            self.total_sleeps = int(meta.get("total_sleeps", 0))
            self.auto_sleep = bool(meta.get("auto_sleep", True))
            self.auto_web = bool(meta.get("auto_web", True))
            self.language_cortex.lora_updates_count = int(meta.get("lora_updates_count", 0))
            self.language_cortex.core_knowledge_summary = str(meta.get("core_knowledge_summary", ""))
            self.hippocampus.load_dict(meta.get("hippocampus", {}))
            self.chat_log = list(meta.get("chat_log", []))
            self.working_chat = list(meta.get("working_chat", []))
            self.sleep_history = list(meta.get("sleep_history", []))

            # Register loaded memory vocabulary in tokenizer & migrate legacy embeddings in RAM
            all_traces = (
                self.hippocampus.short_term_memories
                + self.hippocampus.semantic_memories
                + self.hippocampus.rem_reflections
            )
            if all_traces:
                self.language_cortex.tokenizer.learn_texts([m.text for m in all_traces])
                if meta.get("embedding_version") != "standalone_pytorch_v1":
                    recomputed = self.language_cortex.encode_texts([m.text for m in all_traces])
                    for trace, vec in zip(all_traces, recomputed):
                        trace.embedding = vec

            self._log(
                "INIT",
                f"Wczytano mózg (Dzień {self.day}, {self.total_sleeps} nocy, "
                f"{len(self.hippocampus.semantic_memories)} wspomnień trwałych).",
            )
            return True
        except Exception as exc:
            self._log("WARN", f"Nie udało się wczytać stanu mózgu ({exc}), start od zera.")
            return False

    def reset_brain(self) -> None:
        self.stop_study()
        with self.lock:
            for fname in ("brain_weights.pt", "brain_memory.json"):
                p = self.state_dir / fname
                if p.exists():
                    p.unlink()
            self.language_cortex.reset_lora()
            self.hippocampus = Hippocampus(
                max_short_term_memories=self.cfg.max_short_term_memories,
                max_short_term_facts=self.cfg.max_short_term_facts,
                max_semantic_memories=self.cfg.max_semantic_memories,
                max_rem_reflections=self.cfg.max_rem_reflections,
            )
            self.concept_cortex = AssociativeConceptCortex(self.cfg, self.device)
            self.sleep_engine = SleepEngine(
                cfg=self.cfg,
                concept_cortex=self.concept_cortex,
                hippocampus=self.hippocampus,
                device=self.device,
                language_cortex=self.language_cortex,
            )
            self.state_mode = "AWAKE"
            self.energy = self.cfg.max_energy
            self.day = 1
            self.total_sleeps = 0
            self._active_op = None
            self.chat_log.clear()
            self.working_chat.clear()
            self.sleep_history.clear()
            self.event_log.clear()
            self._seed_initial_knowledge()

    def build_graph_payload(self, max_edges: int = 90) -> dict[str, Any]:
        raw_edges: list[dict[str, Any]] = []
        for f in list(self.hippocampus.rem_dream_links.values())[-30:]:
            raw_edges.append({"src": f.src, "dst": f.dst, "rel": f.bridge or "REM", "source": "REM_DREAM", "strength": round(f.strength, 2)})
        for f in list(self.hippocampus.consolidated_facts.values())[-40:]:
            raw_edges.append({"src": f.src, "dst": f.dst, "rel": f.rel, "source": "CONSOLIDATED", "strength": round(f.strength, 2)})
        for f in self.hippocampus.short_term_facts[-25:]:
            raw_edges.append({"src": f.src, "dst": f.dst, "rel": f.rel, "source": "HIPPOCAMPUS", "strength": round(f.strength, 2)})

        edges = raw_edges[:max_edges]
        node_names: list[str] = []
        seen_names: set[str] = set()
        degree_map: dict[str, int] = {}
        consolidated_set: set[str] = set()
        for e in edges:
            for key in (e["src"], e["dst"]):
                degree_map[key] = degree_map.get(key, 0) + 1
                if key not in seen_names:
                    seen_names.add(key)
                    node_names.append(key)
                if e["source"] in ("REM_DREAM", "CONSOLIDATED"):
                    consolidated_set.add(key)

        coords = self.concept_cortex.compute_2d_projection(limit=max(90, len(node_names)), concepts=node_names)
        nodes: list[dict[str, Any]] = []
        for name in node_names:
            if name in coords:
                x, y = coords[name]
            else:
                h = abs(hash(name)) % 360
                rad = math.radians(h)
                x, y = math.cos(rad) * 0.88, math.sin(rad) * 0.88
            nodes.append(
                {
                    "id": name,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "consolidated": name in consolidated_set,
                    "degree": degree_map.get(name, 0),
                }
            )
        return {"nodes": nodes, "edges": edges}

    def creativity_score(self) -> float:
        with self.lock:
            base = max(0.0, min(1.0, float(self.creativity_level))) * 55.0
            rem_links_bonus = min(20.0, len(self.hippocampus.rem_dream_links) * 1.2)
            rem_refl_bonus = min(15.0, len(self.hippocampus.rem_reflections) * 1.5)
            graph_bonus = min(10.0, len(self.concept_cortex.id_to_concept) * 0.15)
            return round(min(100.0, base + rem_links_bonus + rem_refl_bonus + graph_bonus), 1)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            active_sem = [x for x in self.hippocampus.semantic_memories if not x.superseded]
            active_sem.sort(key=lambda x: (x.last_recalled_day, x.strength, x.importance), reverse=True)
            pending_info = len(self.hippocampus.short_term_memories)
            info_pressure = round(
                min(100.0, (pending_info / max(1, self.cfg.auto_sleep_pending_threshold)) * 100.0),
                1,
            )
            return {
                "state_mode": self.state_mode,
                "energy": round(self.energy, 1),
                "max_energy": self.cfg.max_energy,
                "sleep_pressure": info_pressure,
                "pending_info_to_link": pending_info,
                "pending_sleep_threshold": self.cfg.auto_sleep_pending_threshold,
                "creativity_level": round(self.creativity_level, 2),
                "creativity_score": self.creativity_score(),
                "day": self.day,
                "total_sleeps": self.total_sleeps,
                "auto_sleep": self.auto_sleep,
                "auto_web": self.auto_web,
                "gpu": self.gpu_info(),
                "study": dict(self.study_state),
                "studied_topics": list(self.hippocampus.studied_topics.values()),
                "language_cortex": {
                    "model_id": self.cfg.language_model_id,
                    "ready": self.language_cortex.ready,
                    "loading": self.language_cortex.loading,
                    "error": self.language_cortex.load_error,
                    "status": self.language_cortex.status_message,
                    "lora_updates": self.language_cortex.lora_updates_count,
                    "total_params": self.language_cortex.total_parameters(),
                    "vocab_used": self.language_cortex.tokenizer.vocab_used,
                    "vocab_size": self.cfg.vocab_size,
                    "fluency": self.language_cortex.transformer_lm.fluency_score,
                    "trained_steps": self.language_cortex.transformer_lm.trained_steps,
                    "last_loss": round(self.language_cortex.transformer_lm.last_loss, 3),
                    "core_knowledge_summary": self.language_cortex.core_knowledge_summary,
                },
                "limits": {
                    "short_term_memories": self.cfg.max_short_term_memories,
                    "semantic_memories": self.cfg.max_semantic_memories,
                    "working_chat_window": self.cfg.working_chat_window,
                    "lora_replay_samples": self.cfg.lora_max_replay_samples,
                },
                "chat_log": self.chat_log[-200:],
                "memory": {
                    "short_term_memory_count": len(self.hippocampus.short_term_memories),
                    "short_term_fact_count": len(self.hippocampus.short_term_facts),
                    "semantic_memory_count": len(active_sem),
                    "consolidated_fact_count": len(self.hippocampus.consolidated_facts),
                    "rem_dream_count": len(self.hippocampus.rem_dream_links),
                    "rem_reflection_count": len(self.hippocampus.rem_reflections),
                    "cortex_concepts": len(self.concept_cortex.id_to_concept),
                    "short_term_memories": [m.to_public_dict() for m in self.hippocampus.short_term_memories[-80:]],
                    "semantic_memories": [m.to_public_dict() for m in active_sem[:120]],
                    "rem_reflections": [m.to_public_dict() for m in self.hippocampus.rem_reflections[-40:]],
                },
                "graph": self.build_graph_payload(),
                "sleep_history": self.sleep_history[-15:],
                "event_log": self.event_log[-35:],
            }
