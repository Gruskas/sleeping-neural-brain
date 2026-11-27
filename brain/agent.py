from __future__ import annotations

import gc
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from .broca_llm import LocalPolishBrocaLLM
from .living_brain import LivingBrainEngine, NeuromodulatorState, trim_process_ram
from .teacher_distill import (
    count_reasoning_dialogues,
    count_teacher_dialogues,
    is_ai_corporate_reply,
    load_reasoning_records,
    load_teacher_dialogues,
    run_bielik_reasoning_subprocess,
    run_bielik_teacher_subprocess,
)
from .config import BrainConfig
from .curriculum import (
    BROAD_AUTONOMOUS_SYLLABUS,
    ELEMENTARY_FOUNDATION_PACK,
    ELEMENTARY_WEB_SYLLABUS,
)
from .hippocampus import Hippocampus
from .language_cortex import LanguageCortex, _strip_pl
from .language_trainer import (
    _short_answer_from_fact,
    PARQUET_CORPORA_CATALOG,
    ensure_parquet_corpora,
    inspect_parquet_corpus,
    iter_parquet_polish_pairs,
    list_parquet_corpora,
    try_prefrontal_math,
)
from .neural_cortex import BiologicalAssociativeCortex
from .sensory_wernicke import configure_torch_backend
from .sleep_engine import SleepEngine, SleepReport
from .web_learner import (
    chunk_text,
    clean_search_query,
    extract_lead_definition,
    fetch_random_wikipedia_titles,
    gather_web_sources,
    is_http_url,
    is_web_boilerplate,
)

CHECKPOINT_VERSION = "tokenless_bio_v1"


class SleepingBrain:
    """
    Tokenless Bio-Inspired Sleeping Neural Brain (PyTorch + CUDA 12.8):
      - Sensory Cortex (WernickeByteCortex): raw UTF-8 bytes -> Byte-Patching Conv1D -> 16 columns -> k-WTA SDR
      - Associative Cortex (BiologicalAssociativeCortex): Modern Continuous Hopfield + local W_rec/W_in/threshold (@torch.no_grad()) -> Z*
      - Speech Cortex (BrocaByteTransformer): Cross-Attention ONLY on Z* -> real UTF-8 byte streaming
      - Tensor Hippocampus (GPU bfloat16 engram banks + sparse int16/bfloat16 checkpoints)
      - Vectorized GPU Sleep (NREM Light -> NREM Deep + SHY -> REM Dreaming)
    """

    def __init__(
        self,
        cfg: BrainConfig | None = None,
        base_dir: str | Path = ".",
        load_language_mode: str = "async",  # "async" | "sync" | "none"
    ) -> None:
        configure_torch_backend()
        self.cfg = cfg or BrainConfig()
        self.base_dir = Path(base_dir).resolve()
        self.state_dir = self.base_dir / self.cfg.state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.RLock()

        # Core biological brain organs
        self.hippocampus = Hippocampus(
            num_neurons=self.cfg.num_neurons,
            sdr_k=self.cfg.sdr_k,
            max_short_term_memories=self.cfg.max_short_term_memories,
            max_semantic_memories=self.cfg.max_semantic_memories,
            max_rem_reflections=self.cfg.max_rem_reflections,
            device=self.device,
            bio_synapse_dtype=self.cfg.bio_synapse_dtype,
        )
        self.cortex = BiologicalAssociativeCortex(self.cfg, self.device)
        self.concept_cortex = self.cortex  # attribute alias for external callers
        self.language_cortex = LanguageCortex(
            cfg=self.cfg,
            device=self.device,
            lora_dir=self.state_dir / "lora_cortex",
        )
        self.sleep_engine = SleepEngine(
            cfg=self.cfg,
            cortex=self.cortex,
            hippocampus=self.hippocampus,
            device=self.device,
            language_cortex=self.language_cortex,
        )

        # Rolling sensory buffer strictly for Broca denoising motor practice during sleep (NEVER read in chat)
        self._wake_broca_buffer: list[str] = []
        # Prefrontal Working Memory SDR Ring (fast SDR cueing) + Discrete Multi-Slot GPU Buffer [K_wm, D]
        self._working_sdr_ring: list[torch.Tensor] = []
        wm_cap = max(2, int(getattr(self.cfg, "working_memory_slots", 8)))
        wm_dt = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        self._wm_slot_buf = torch.zeros((wm_cap, self.cfg.column_dim), device=self.device, dtype=wm_dt)
        self._wm_slot_count: int = 0

        # Lifecycle & creativity state
        self.state_mode: str = "AWAKE"
        self.energy: float = self.cfg.max_energy
        self.creativity_level: float = float(self.cfg.creativity_level)
        self.day: int = 1
        self.total_sleeps: int = 0
        self.auto_sleep: bool = True
        self.auto_web: bool = True
        self._active_op: str | None = None

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

        self._save_lock = threading.Lock()
        self._save_thread: threading.Thread | None = None
        self.working_chat: list[dict[str, str]] = []
        self.chat_log: list[dict[str, Any]] = []
        self.sleep_history: list[dict[str, Any]] = []
        self.event_log: list[dict[str, Any]] = []
        self.dmn_thoughts: list[dict[str, Any]] = []
        self.training_dialogue_feed: list[dict[str, Any]] = []
        self._training_feed_seq: int = 0

        # Autonomous DMN idle-thinking state (preempted immediately on user chat)
        self._last_user_activity_ts: float = time.monotonic()
        self._last_dmn_think_ts: float = 0.0
        self._user_interrupt_event = threading.Event()
        self._dmn_running = threading.Event()

        self.broca_llm: LocalPolishBrocaLLM | None = (
            LocalPolishBrocaLLM(
                cfg=self.cfg,
                device=self.device,
                lora_dir=self.state_dir / "lora_cortex",
            )
            if self.cfg.use_local_llm_broca
            else None
        )

        # Dynamically growing zero-backprop Living Brain (M1 L1-L3/Assoc, M2 dynamic, M4 curiosity, NA/DA/ACh)
        # Background spontaneous theta/gamma clock runs continuously via non-blocking lock.acquire(blocking=False)
        self.living_brain = LivingBrainEngine(
            sensory_dim=512,
            target_device="gpu" if self.device.type == "cuda" else "ram",
            memory_budget_mb=4096.0,
            start_background_clock=True,
            bio_synapse_dtype=self.cfg.bio_synapse_dtype,
        )
        self.living_brain.set_idle_callback(self._on_background_clock_tick)

        if not self.load_state():
            self._seed_initial_knowledge()

        if load_language_mode == "sync":
            self.language_cortex.load_sync()
            if self.broca_llm is not None:
                self.broca_llm.load_sync()
        elif load_language_mode == "async":
            self.language_cortex.load_async()
            if self.broca_llm is not None:
                self.broca_llm.load_async()

    @property
    def brain_log(self) -> list[dict[str, Any]]:
        return self.event_log

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

    def _is_busy_for_dmn(self) -> bool:
        return (
            self._user_interrupt_event.is_set()
            or self._active_op is not None
            or bool(self.study_state.get("active", False))
            or self.state_mode != "AWAKE"
            or not self.language_cortex.ready
            or self.living_brain._pause_event.is_set()
        )

    def synthesize_2hop_thought(
        self,
        engram_a: dict[str, Any] | torch.Tensor,
        engram_b: dict[str, Any] | torch.Tensor,
        engram_c: dict[str, Any] | torch.Tensor,
        tag: str = "dmn_2hop",
        importance: float = 3.0,
        bridge_meta: dict[str, Any] | None = None,
        non_blocking: bool = False,
        abort_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """
        Synthesize an autonomous 2-hop thought connecting A and C through bridge B:
          1. Build multi-slot Z_ctx [A, B, C] & generate + run Quality Gate via LanguageCortex.
          2. If accepted:
             - Save new unpinned engram into Hippocampus (`importance=3.0`, `parent_ids=[A,B,C]`, `tag`).
             - Gently strengthen A <-> C in LivingBrainEngine via local Oja-Hebb plasticity.
             - Log the accepted thought and its A --(B)--> C bridge into `event_log` (`brain_log`) and `dmn_thoughts`.
        """
        with self.lock:
            sem_bank = self.hippocampus.semantic.engrams if len(self.hippocampus.semantic) > 0 else None
            cur_day = int(self.day)
            creativity = float(self.creativity_level)

        res = self.language_cortex.synthesize_2hop_thought(
            engram_a=engram_a,
            engram_b=engram_b,
            engram_c=engram_c,
            cortex=self.cortex,
            engram_bank=sem_bank,
            creativity_level=creativity,
            abort_event=abort_event,
            non_blocking_lock=non_blocking,
        )
        if not res.get("accepted"):
            return res

        if abort_event is not None and abort_event.is_set():
            res["accepted"] = False
            res["aborted"] = True
            return res

        id_a = str(engram_a["id"]) if isinstance(engram_a, dict) and "id" in engram_a else "A"
        id_b = str(engram_b["id"]) if isinstance(engram_b, dict) and "id" in engram_b else "B"
        id_c = str(engram_c["id"]) if isinstance(engram_c, dict) and "id" in engram_c else "C"
        lbl_a = (str(engram_a.get("preview_label", "")) if isinstance(engram_a, dict) else "").strip() or f"#{id_a}"
        lbl_b = (str(engram_b.get("preview_label", "")) if isinstance(engram_b, dict) else "").strip() or f"#{id_b}"
        lbl_c = (str(engram_c.get("preview_label", "")) if isinstance(engram_c, dict) else "").strip() or f"#{id_c}"
        thought_text = str(res["thought_text"]).strip()
        thought_sdr = res["thought_sdr"]

        # 1) Record accepted thought engram into Hippocampus (pinned=False, moderate importance=3.0)
        if non_blocking:
            if not self.lock.acquire(blocking=False):
                res["accepted"] = False
                res["aborted"] = True
                return res
        else:
            self.lock.acquire()
        try:
            if abort_event is not None and abort_event.is_set():
                res["accepted"] = False
                res["aborted"] = True
                return res
            saved_engram = self.hippocampus.record_2hop_engram(
                sdr=thought_sdr,
                thought_text=thought_text,
                parent_ids=[id_a, id_b, id_c],
                tag=tag,
                importance=float(importance),
                day=cur_day,
                strength=1.5,
                sparsity_pct=self.cortex.last_sparsity_pct,
                target_bank="rem" if tag == "rem_2hop" else "short_term",
            )
            log_kind = "REM" if tag == "rem_2hop" else "DMN"
            self._log(
                log_kind,
                f"💡 Myśl 2-hop ({tag}): „{thought_text}” | Most: [{id_a}] „{lbl_a[:56]}” "
                f"—([{id_b}] „{lbl_b[:44]}”)→ [{id_c}] „{lbl_c[:56]}”",
            )
            record_entry = {
                "t": time.strftime("%H:%M:%S"),
                "day": cur_day,
                "tag": tag,
                "engram_id": saved_engram["id"],
                "bank": saved_engram["bank"],
                "thought": thought_text,
                "parent_ids": [id_a, id_b, id_c],
                "parent_labels": {"a": lbl_a[:80], "b": lbl_b[:80], "c": lbl_c[:80]},
                "sim_ab": bridge_meta.get("sim_ab") if bridge_meta else None,
                "sim_bc": bridge_meta.get("sim_bc") if bridge_meta else None,
                "sim_ac": bridge_meta.get("sim_ac") if bridge_meta else None,
                "joint_coherence": res.get("joint_coherence"),
            }
            self.dmn_thoughts.append(record_entry)
            if len(self.dmn_thoughts) > 80:
                self.dmn_thoughts = self.dmn_thoughts[-80:]
        finally:
            self.lock.release()

        # 2) Gently strengthen A <-> C in LivingBrainEngine via local Oja-Hebb plasticity
        sdr_a = engram_a["sdr"] if isinstance(engram_a, dict) else engram_a
        sdr_c = engram_c["sdr"] if isinstance(engram_c, dict) else engram_c
        oja_stats = self.living_brain.strengthen_association_oja(
            vec_a=sdr_a,
            vec_c=sdr_c,
            lr=0.015,
            thought_vec=thought_sdr,
        )

        res["engram_id"] = saved_engram["id"]
        res["engram"] = saved_engram
        res["oja_stats"] = oja_stats
        return res

    def run_dmn_2hop_step(self, non_blocking: bool = True) -> dict[str, Any] | None:
        """Execute a single autonomous Default Mode Network 2-hop step (sample_2hop_bridge -> synthesize_2hop_thought)."""
        if self._dmn_running.is_set() or self._is_busy_for_dmn():
            return None
        self._dmn_running.set()
        try:
            if non_blocking:
                if not self.lock.acquire(blocking=False):
                    return None
            else:
                self.lock.acquire()
            try:
                if self._is_busy_for_dmn() or len(self.hippocampus.semantic) < 3:
                    return None
                bridge = self.hippocampus.sample_2hop_bridge(
                    min_hop_sim=float(getattr(self.cfg, "dmn_bridge_min_hop_sim", 0.45)),
                    max_direct_sim=float(getattr(self.cfg, "dmn_bridge_max_direct_sim", 0.25)),
                    current_day=self.day,
                )
            finally:
                self.lock.release()

            if bridge is None:
                return None

            return self.synthesize_2hop_thought(
                engram_a=bridge["a"],
                engram_b=bridge["b"],
                engram_c=bridge["c"],
                tag="dmn_2hop",
                importance=3.0,
                bridge_meta=bridge,
                non_blocking=non_blocking,
                abort_event=self._user_interrupt_event,
            )
        finally:
            self._dmn_running.clear()

    def _on_background_clock_tick(self) -> None:
        """
        Invoked from LivingBrainEngine's existing background clock thread (`start_background_clock=True`).
        Fires a single non-blocking DMN 2-hop step when user has been idle >= `idle_think_delay_sec`
        and no training/sleep/chat is active; yields immediately if user sends a message.
        """
        if not bool(getattr(self.cfg, "dmn_enabled", True)):
            return
        if self._is_busy_for_dmn() or self._dmn_running.is_set():
            return
        now = time.monotonic()
        idle_delay = float(getattr(self.cfg, "idle_think_delay_sec", 50.0))
        think_interval = float(getattr(self.cfg, "dmn_think_interval_sec", 40.0))
        if (now - self._last_user_activity_ts) < idle_delay:
            return
        if (now - self._last_dmn_think_ts) < think_interval:
            return
        self._last_dmn_think_ts = now
        self.run_dmn_2hop_step(non_blocking=True)

    def _push_training_dialogue(
        self,
        *,
        mode: str,
        tag: str,
        prompt: str,
        prompt_role: str = "Pytanie",
        teacher_reply: str = "",
        teacher_role: str = "",
        brain_reply: str = "",
        brain_role: str = "",
    ) -> None:
        p_txt = (prompt or "").strip()
        t_txt = (teacher_reply or "").strip()
        b_txt = (brain_reply or "").strip()
        if not p_txt and not t_txt and not b_txt:
            return
        self._training_feed_seq += 1
        self.training_dialogue_feed.append(
            {
                "id": self._training_feed_seq,
                "t": time.strftime("%H:%M:%S"),
                "mode": mode,
                "tag": tag,
                "prompt_role": prompt_role,
                "prompt": p_txt,
                "teacher_role": teacher_role,
                "teacher_reply": t_txt,
                "brain_role": brain_role,
                "brain_reply": b_txt,
            }
        )
        if len(self.training_dialogue_feed) > 500:
            self.training_dialogue_feed = self.training_dialogue_feed[-500:]

    def clear_training_dialogue_feed(self) -> None:
        with self.lock:
            self.training_dialogue_feed.clear()

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

    def _buffer_wake_stimulus(self, texts: list[str]) -> None:
        for t in texts:
            clean = (t or "").strip()
            if len(clean) >= 12:
                self._wake_broca_buffer.append(clean)
        if len(self._wake_broca_buffer) > 320:
            self._wake_broca_buffer = self._wake_broca_buffer[-320:]

    def _push_working_sdr(self, sdr: torch.Tensor) -> None:
        vec = sdr.detach().flatten().float()
        norm = float(vec.norm().item())
        if norm > 1e-6:
            vec = vec / norm
        self._working_sdr_ring.append(vec.to(device=self.device, dtype=torch.bfloat16))
        cap = max(4, int(self.cfg.working_sdr_capacity))
        if len(self._working_sdr_ring) > cap:
            self._working_sdr_ring = self._working_sdr_ring[-cap:]

    def _get_working_memory_sdr(
        self,
        query_sdr: torch.Tensor | None = None,
        max_slots: int = 6,
        focus_recent: bool = False,
    ) -> torch.Tensor | None:
        """
        Prefrontal Cortex (PFC) discrete slot working memory with query-gated attention:
        maintains the last `max_slots` (default 6) turn SDR slots and attends to the most
        relevant & recent slots instead of flattening 64 turns into one blurred average.
        """
        if not self._working_sdr_ring:
            return None
        slot_cap = 3 if focus_recent else max(2, int(max_slots))
        slots_list = self._working_sdr_ring[-slot_cap:]
        k = len(slots_list)
        slots = torch.stack([v.float() for v in slots_list], dim=0)  # [K, N]
        ages = torch.arange(k - 1, -1, -1, device=self.device, dtype=torch.float32)

        if query_sdr is not None and query_sdr.numel() == self.cfg.num_neurons and not focus_recent:
            q = F.normalize(query_sdr.to(device=self.device, dtype=torch.float32).reshape(-1), dim=-1, eps=1e-6)
            sims = slots @ q  # [K]
            logits = 6.0 * sims - 0.18 * ages
            weights = F.softmax(logits, dim=0)
        else:
            # Explicit follow-up ("dlaczego?", "rozwiń to") focuses sharply on the immediate last turns
            logits = -0.55 * ages
            weights = F.softmax(logits, dim=0)

        acc = (weights.unsqueeze(1) * slots).sum(dim=0)
        norm = float(acc.norm().item())
        if norm <= 1e-6:
            return None
        return (acc / norm).to(dtype=torch.bfloat16)

    def _push_working_memory_slot(
        self,
        cols_or_patches: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Store a normalized discrete turn representation in the [K_wm, D] GPU working memory buffer."""
        t = cols_or_patches.detach().to(device=self.device, dtype=torch.float32)
        if t.ndim == 3:
            t = t[0]
        if t.ndim == 1:
            vec = t
        else:
            if mask is not None:
                m = mask.to(device=self.device, dtype=torch.float32).reshape(-1)
                if m.shape[0] == t.shape[0]:
                    m_col = m.unsqueeze(-1)
                    vec = (t * m_col).sum(dim=0) / m_col.sum().clamp(min=1.0)
                else:
                    vec = t.mean(dim=0)
            else:
                vec = t.mean(dim=0)
        vec = F.normalize(vec, dim=-1, eps=1e-6).to(dtype=self._wm_slot_buf.dtype)
        cap = self._wm_slot_buf.shape[0]
        if self._wm_slot_count < cap:
            self._wm_slot_buf[self._wm_slot_count].copy_(vec)
            self._wm_slot_count += 1
        else:
            self._wm_slot_buf[:-1].copy_(self._wm_slot_buf[1:].clone())
            self._wm_slot_buf[-1].copy_(vec)

    def _get_working_memory_slots(self, max_slots: int | None = None) -> torch.Tensor | None:
        """Return discrete normalized GPU working memory turn slots [1, K_active, D] without collapsing them."""
        if self._wm_slot_count <= 0:
            return None
        k = self._wm_slot_count if max_slots is None else min(self._wm_slot_count, max(1, int(max_slots)))
        return self._wm_slot_buf[self._wm_slot_count - k : self._wm_slot_count].unsqueeze(0)

    @staticmethod
    def _is_followup_query(msg: str) -> bool:
        s = msg.strip()
        low = _strip_pl(s.lower())
        greetings = ("siema", "czesc", "hej", "hejka", "elo", "dzien dobry", "dobry wieczor", "witaj", "pa", "nara")
        if low in greetings or any(low.startswith(g + " ") for g in greetings):
            return False
        followup_cues = (
            "dlaczego", "czemu", "rozwin", "powiedz wiecej", "wiecej o tym", "a co", "a jak",
            "a dlaczego", "i co", "co dalej", "podaj przyklad", "jak to", "dokladniej",
            "o co chodzi", "serio", "no i", "wyjasnij prosciej", "prosciej", "krotko",
        )
        return any(low.startswith(c) or c in low for c in followup_cues)

    @staticmethod
    def _extract_conversational_qa_pairs(msg: str) -> list[tuple[str, str]]:
        """
        Detect personal / direct conversational teaching statements (e.g. giving the brain a name,
        introducing oneself, or 'zapamiętaj że ...') so Broca can adapt online in ~180 ms via
        pure neural weight plasticity (zero template bypassing in generation!).
        """
        s = msg.strip()
        if not s or s.endswith("?") or s.startswith("/"):
            return []

        # 1. User assigns a name to the brain ("nadaję ci imię: Krzysztof", "nazywasz się Adam")
        m_bot = re.search(
            r"(?:nadaj[eę]\s+ci\s+imi[eę]|masz\s+na\s+imi[eę]|nazywasz\s+si[eę]|"
            r"b[eę]dziesz\s+mia[lł]\s+na\s+imi[eę]|b[eę]dziesz\s+si[eę]\s+nazywa[cć]|"
            r"od\s+teraz\s+(?:jeste[sś]|masz\s+na\s+imi[eę]|nazywasz\s+si[eę]))"
            r"[\s:,\-–—]+([A-ZĄĆĘŁŃÓŚŹŻa-ząćęłńóśźż0-9_-]{2,24})",
            s,
            flags=re.IGNORECASE,
        )
        if m_bot:
            raw_name = m_bot.group(1).strip(".,!?;:\"'()[]")
            name = raw_name[0].upper() + raw_name[1:] if raw_name else raw_name
            if len(name) >= 2:
                return [
                    (s, f"Super, zapamiętałem! Od teraz mam na imię {name}."),
                    ("jak masz na imię", f"Mam na imię {name}."),
                    ("jak masz na imie", f"Mam na imię {name}."),
                    ("jak się nazywasz", f"Nazywam się {name}."),
                    ("jak sie nazywasz", f"Nazywam się {name}."),
                    ("twoje imię", f"Mam na imię {name}."),
                ]

        # 2. User introduces themselves ("mam na imię Jan", "nazywam się Kasia")
        m_usr = re.search(
            r"(?:^|\b)(?:mam\s+na\s+imi[eę]|nazywam\s+si[eę]|m[oó]w\s+mi)"
            r"[\s:,\-–—]+([A-ZĄĆĘŁŃÓŚŹŻa-ząćęłńóśźż]{2,24})",
            s,
            flags=re.IGNORECASE,
        )
        if m_usr:
            raw_name = m_usr.group(1).strip(".,!?;:\"'()[]")
            name = raw_name[0].upper() + raw_name[1:] if raw_name else raw_name
            if len(name) >= 2:
                return [
                    (s, f"Miło mi Cię poznać, {name}! Zapamiętałem Twoje imię."),
                    ("jak mam na imię", f"Masz na imię {name}."),
                    ("jak mam na imie", f"Masz na imię {name}."),
                    ("jak się nazywam", f"Nazywasz się {name}."),
                    ("kim jestem", f"Jesteś {name} i rozmawiasz teraz ze mną."),
                ]

        # 3. Direct user memory teaching ("zapamiętaj że X to Y" / "zapamiętaj że ...")
        m_rem = re.match(r"^(?:zapami[eę]taj|pami[eę]taj)\s*,?\s*(?:[zż]e\s+)?(.{6,200})$", s, flags=re.IGNORECASE)
        if m_rem:
            fact = m_rem.group(1).strip().rstrip(".")
            pairs = [(s, f"Zapamiętałem: {fact}.")]
            if " to " in fact.lower():
                subj, pred = re.split(r"\s+to\s+", fact, maxsplit=1, flags=re.IGNORECASE)
                subj = subj.strip()
                pred = pred.strip()
                if 2 <= len(subj) <= 48 and len(pred) >= 4:
                    pairs.append((f"co to jest {subj.lower()}", f"{subj} to {pred}."))
                    pairs.append((f"co to {subj.lower()}", f"{subj} to {pred}."))
            return pairs

        return []

    def _acquire_op(self, op_name: str) -> None:
        with self.lock:
            if self._active_op == "study" or self.study_state["active"]:
                raise RuntimeError(
                    f"Mózg jest teraz skupiony na nauce tematu „{self.study_state['topic']}” — poczekaj chwilę lub kliknij Stop."
                )
            if self._active_op == "sleep" or self.state_mode in ("NREM_LIGHT", "NREM_DEEP", "REM_DREAMING"):
                raise RuntimeError("Mózg właśnie śpi i konsoliduje pamięć na GPU — poczekaj chwilę aż się obudzi!")
            if self._active_op == "chat":
                raise RuntimeError("Mózg właśnie odpowiada na poprzednią wiadomość — poczekaj chwilkę!")
            self._active_op = op_name
            if op_name in ("study", "sleep"):
                self.living_brain.set_paused(True)

    def _release_op(self, op_name: str) -> None:
        with self.lock:
            if self._active_op == op_name:
                self._active_op = None
                if op_name in ("study", "sleep"):
                    self.living_brain.set_paused(False)

    def _seed_initial_knowledge(self) -> None:
        self._append_chat(
            "assistant",
            "Uruchomiono beztokenową korę neuronową (Wernicke Bajty UTF-8 → Modern Hopfield + k-WTA SDR → BrocaByteTransformer). "
            "Użyj `/ucz <temat>`, `/szkola`, `/wyksztalc` lub `/sen`, albo pisz bezpośrednio po polsku.",
            meta={"kind": "greeting"},
            add_to_working=False,
        )
        self._log("INIT", f"Uruchomiono beztokenową korę na {self.gpu_info()['device_name']}.")

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

    def _sampling_temperature(self) -> float:
        # Centered at broca_temperature (0.55) to avoid random word distortion while staying stochastic
        base = float(self.cfg.broca_temperature)
        delta = (max(0.0, min(1.0, float(self.creativity_level))) - 0.80) * 0.25
        return round(max(0.42, min(0.75, base + delta)), 3)

    # ── Sensory Ingestion Helper (Wernicke -> GPU Tensor Engrams) ────────────
    def _ingest_sensory_texts(
        self,
        texts: list[str],
        category: str = "wiedza",
        importances: list[int] | None = None,
        topic: str | None = None,
        source_url: str | None = None,
        pinned: bool = False,
        pinned_flags: list[bool] | None = None,
    ) -> list[str]:
        clean_texts = [t.strip() for t in texts if t and len(t.strip()) >= 8]
        if not clean_texts:
            return []
        w_out = self.language_cortex.perceive_texts(clean_texts)
        with self.lock:
            ids = self.hippocampus.record_engrams_batch(
                sdrs=w_out.sdr,
                preview_labels=clean_texts,
                category=category,
                importances=importances or [4] * len(clean_texts),
                day=self.day,
                sparsity_pct=w_out.sparsity_pct,
                topic=topic,
                source_url=source_url,
                pinned=pinned,
                pinned_flags=pinned_flags,
            )
            self._buffer_wake_stimulus(clean_texts)
        return ids

    # ── Elementary Curriculum (Sensory Engrams + Broca Pre-warming) ──────────
    def learn_elementary_foundation(self, consolidate: bool = True, _from_nested: bool = False) -> dict[str, Any]:
        if not _from_nested:
            self._acquire_op("study")
        try:
            with self.lock:
                already_done = "szkoła podstawowa — fundamenty" in self.hippocampus.studied_topics

            added_count = 0
            if not already_done:
                texts = [str(item["text"]) for item in ELEMENTARY_FOUNDATION_PACK]
                imps = [int(item.get("importance", 4)) for item in ELEMENTARY_FOUNDATION_PACK]
                ids = self._ingest_sensory_texts(
                    texts=texts,
                    category="wiedza",
                    importances=imps,
                    topic="Szkoła podstawowa — fundamenty",
                    pinned_flags=[imp >= 4 for imp in imps],
                )
                added_count = len(ids)
                with self.lock:
                    self.hippocampus.studied_topics["szkoła podstawowa — fundamenty"] = {
                        "topic": "Szkoła podstawowa — fundamenty",
                        "mastery": 95.0,
                        "facts_learned": added_count,
                        "pages_read": 0,
                        "day": self.day,
                        "summary": "Bodźce zmysłowe (9 przedmiotów) zakodowane w SDR przez Wernickego.",
                        "sources": [],
                    }
                    self._log("STUDY", f"🎓 Zakodowano pakiet Szkoła Podstawowa jako {added_count} engramów SDR.")

            sleep_rep = None
            if consolidate and (added_count > 0 or len(self.hippocampus.short_term) > 0):
                sleep_rep = self.sleep_now(visual_pacing=False, _from_nested=True)

            with self.lock:
                total_sem = len(self.hippocampus.semantic)
                if not _from_nested:
                    msg = (
                        f"🎓 Zakodowano pakiet Szkoła Podstawowa (+{added_count} engramów SDR) i skonsolidowano korę we śnie na GPU. "
                        f"Pamięć trwała: {total_sem} engramów."
                    )
                    self._append_chat(
                        "assistant",
                        msg,
                        meta={"kind": "study_done", "topic": "Szkoła Podstawowa", "facts": added_count},
                        add_to_working=False,
                    )
            self.save_state()
            return {
                "added_facts": added_count,
                "added_triples": 0,
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
                "current_action": "Kodowanie fundamentów i przygotowanie kursu...",
                "sources": [],
                "summary": "",
            }
        try:
            t = threading.Thread(target=self.run_curriculum_online, args=(topics_count, True), daemon=True)
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
            foundation_res = self.learn_elementary_foundation(consolidate=True, _from_nested=True)
            with self.lock:
                self.state_mode = "STUDYING"
                pending_topics = [
                    t
                    for t in ELEMENTARY_WEB_SYLLABUS
                    if self.hippocampus.studied_topics.get(t.lower(), {}).get("mastery", 0.0) < 75.0
                ][:max_topics]
                if not pending_topics:
                    pending_topics = list(ELEMENTARY_WEB_SYLLABUS[:max_topics])
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
                self.study_state["current_action"] = f"Temat {idx}/{len(pending_topics)}: pobieram bodźce dla „{topic}”..."
                docs = gather_web_sources(
                    query=topic,
                    seen_urls=seen_urls,
                    max_docs=2,
                    timeout=self.cfg.study_http_timeout,
                    max_workers=self.cfg.study_fetch_workers,
                )
                topic_count = 0
                for doc in docs:
                    if self._study_stop_event.is_set():
                        break
                    total_pages += 1
                    self.study_state["pages_read"] = total_pages
                    sources_list.append({"title": doc.title, "url": doc.url, "source": doc.source})
                    self.study_state["sources"] = sources_list[-12:]
                    stimuli: list[str] = []
                    lead = extract_lead_definition(doc)
                    if lead:
                        stimuli.append(lead)
                    for ch in chunk_text(doc.text, focus_query=topic, max_chars=2400, max_chunks=1):
                        stimuli.extend(self.language_cortex.extract_sensory_stimuli(ch, max_items=6))
                    added = self._ingest_sensory_texts(
                        stimuli, category="nauka_z_sieci", topic=topic, source_url=doc.url
                    )
                    topic_count += len(added)
                    total_facts += len(added)
                    self.study_state["facts_learned"] = total_facts

                with self.lock:
                    self.hippocampus.studied_topics[topic.lower()] = {
                        "topic": topic,
                        "mastery": 85.0 if topic_count > 0 else 30.0,
                        "facts_learned": topic_count,
                        "pages_read": len(docs),
                        "day": self.day,
                        "summary": f"Zakodowano {topic_count} engramów SDR z sieci.",
                        "sources": sources_list[-4:],
                    }
                self.study_state["mastery"] = round((idx / max(1, len(pending_topics))) * 100.0, 1)

            if len(self.hippocampus.short_term) > 0 and not self._study_stop_event.is_set():
                self.study_state["current_action"] = "Konsolidacja NREM + SHY + REM na GPU..."
                self.sleep_now(visual_pacing=False, _from_nested=True)

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
                "current_action": f"Pobieranie bodźców dla {count} tematów...",
                "sources": [],
                "summary": "",
            }
        try:
            t = threading.Thread(target=self.run_turbo_learn, args=(count, True), daemon=True)
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
            with self.lock:
                self._study_stop_event.clear()
                self.state_mode = "STUDYING"
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
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(_fetch_one, t) for t in pending_topics]
                for fut in as_completed(futures):
                    if self._study_stop_event.is_set():
                        break
                    topic, docs = fut.result()
                    done_topics += 1
                    since_last_sleep += 1
                    topic_engrams = 0
                    for doc in docs:
                        total_pages += 1
                        sources_list.append({"title": doc.title, "url": doc.url, "source": doc.source})
                        stimuli: list[str] = []
                        lead = extract_lead_definition(doc)
                        if lead:
                            stimuli.append(lead)
                        for ch in chunk_text(doc.text, focus_query=topic, max_chars=2200, max_chunks=1):
                            stimuli.extend(self.language_cortex.extract_sensory_stimuli(ch, max_items=5))
                        added = self._ingest_sensory_texts(
                            stimuli, category="nauka_z_sieci", topic=topic, source_url=doc.url
                        )
                        topic_engrams += len(added)
                        total_facts += len(added)
                    with self.lock:
                        self.hippocampus.studied_topics[topic.lower()] = {
                            "topic": topic,
                            "mastery": 85.0 if topic_engrams > 0 else 40.0,
                            "facts_learned": topic_engrams,
                            "pages_read": len(docs),
                            "day": self.day,
                            "summary": f"Turbo-nauka: {topic_engrams} engramów SDR.",
                            "sources": sources_list[-3:],
                        }
                        self.study_state["round"] = done_topics
                        self.study_state["pages_read"] = total_pages
                        self.study_state["facts_learned"] = total_facts
                        self.study_state["sources"] = sources_list[-12:]
                        self.study_state["topic"] = f"⚡ Turbo-nauka: {topic}"
                        self.study_state["mastery"] = round((done_topics / max(1, len(pending_topics))) * 85.0, 1)
                        self.study_state["current_action"] = f"Zakodowano {done_topics}/{len(pending_topics)} tematów (+{total_facts} SDR)..."
                    if since_last_sleep >= 60 and not self._study_stop_event.is_set():
                        self.sleep_now(visual_pacing=False, _from_nested=True)
                        since_last_sleep = 0

            if len(self.hippocampus.short_term) > 0 and not self._study_stop_event.is_set():
                self.study_state["current_action"] = "Sen NREM+SHY+REM oraz pre-warming Broki na GPU..."
                self.sleep_now(visual_pacing=False, _from_nested=True)
                self.train_language_now(epochs=10, _from_nested=True)

            self.study_state["mastery"] = 100.0
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
                "current_action": f"Sesja auto-nauki ({mins} min)...",
                "sources": [],
                "summary": "",
            }
        try:
            t = threading.Thread(target=self.run_continuous_learning, args=(mins, True), daemon=True)
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
        t_start = time.monotonic()
        deadline = t_start + mins * 60.0
        try:
            with self.lock:
                self._study_stop_event.clear()
                self.state_mode = "STUDYING"
                already_has_school = "szkoła podstawowa — fundamenty" in self.hippocampus.studied_topics
                self._log("STUDY", f"⏱️ Rozpoczęto Auto-naukę ({mins} min, GPU + sieć + sen NREM/REM).")
                self._append_chat(
                    "assistant",
                    f"⏱️ Rozpoczynam **Auto-naukę ({mins} min)** z internetu! Będę samodzielnie badać nowe tematy "
                    f"(w tym świeże hasła z polskiej Wikipedii), uczyć się ich na GPU i konsolidować wiedzę we śnie. "
                    f"Możesz zatrzymać w każdej chwili przyciskiem **Stop**.",
                    meta={"kind": "study_start", "topic": f"Auto-nauka ({mins} min)"},
                    add_to_working=False,
                )

            total_facts = 0
            sleeps_done = 0
            if not already_has_school:
                foundation_res = self.learn_elementary_foundation(consolidate=True, _from_nested=True)
                total_facts = int(foundation_res.get("added_facts", 0))
                sleeps_done = 1 if foundation_res.get("sleep_report") else 0

            with self.lock:
                self.state_mode = "STUDYING"

            total_pages = 0
            topics_done = 0
            seen_urls: set[str] = set()
            sources_list: list[dict[str, str]] = []
            since_last_sleep = 0
            random_wiki_queue: list[str] = []

            while time.monotonic() < deadline and not self._study_stop_event.is_set():
                with self.lock:
                    studied_lower = set(self.hippocampus.studied_topics.keys())
                    topic = next((t for t in BROAD_AUTONOMOUS_SYLLABUS if t.lower() not in studied_lower), None)

                # Once all 314 syllabus topics are mastered, keep discovering INFINITE fresh Polish Wikipedia topics!
                if not topic:
                    if not random_wiki_queue:
                        fresh_titles = fetch_random_wikipedia_titles(lang="pl", limit=18, timeout=self.cfg.study_http_timeout)
                        with self.lock:
                            studied_lower = set(self.hippocampus.studied_topics.keys())
                        random_wiki_queue = [t for t in fresh_titles if t.lower() not in studied_lower]
                    if random_wiki_queue:
                        topic = random_wiki_queue.pop(0)
                    else:
                        # Fallback if offline/rate-limited: brief pause before retrying or exit on stop
                        if self._study_stop_event.wait(timeout=3.0):
                            break
                        continue

                topics_done += 1
                elapsed_min = (time.monotonic() - t_start) / 60.0
                pct = round(min(99.0, max(2.0, (elapsed_min / max(0.1, float(mins))) * 100.0)), 1)
                with self.lock:
                    self.study_state["round"] = topics_done
                    self.study_state["max_rounds"] = max(topics_done, mins)
                    self.study_state["mastery"] = pct
                    self.study_state["topic"] = f"Auto-nauka: {topic}"
                    self.study_state["current_action"] = f"Temat #{topics_done} ({elapsed_min:.1f}/{mins} min): badam „{topic}”..."

                docs = gather_web_sources(
                    query=topic,
                    seen_urls=seen_urls,
                    max_docs=2,
                    timeout=self.cfg.study_http_timeout,
                    max_workers=self.cfg.study_fetch_workers,
                )
                topic_cnt = 0
                best_lead = ""
                for doc in docs:
                    if self._study_stop_event.is_set() or time.monotonic() >= deadline:
                        break
                    total_pages += 1
                    self.study_state["pages_read"] = total_pages
                    sources_list.append({"title": doc.title, "url": doc.url, "source": doc.source})
                    self.study_state["sources"] = sources_list[-12:]
                    stimuli: list[str] = []
                    lead = extract_lead_definition(doc)
                    if lead:
                        stimuli.append(lead)
                        if not best_lead:
                            best_lead = lead
                    for ch in chunk_text(doc.text, focus_query=topic, max_chars=2400, max_chunks=2):
                        stimuli.extend(self.language_cortex.extract_sensory_stimuli(ch, max_items=6))
                    added = self._ingest_sensory_texts(
                        stimuli, category="nauka_z_sieci", topic=topic, source_url=doc.url
                    )
                    topic_cnt += len(added)
                    total_facts += len(added)
                    self.study_state["facts_learned"] = total_facts

                # Fast zero-forgetting Broca plasticity so the brain can actually answer questions on new topics
                if best_lead and len(best_lead) >= 24 and not self._study_stop_event.is_set():
                    try:
                        self.language_cortex.adapt_qa_online(
                            f"co to jest {topic.lower()}",
                            best_lead,
                            cortex=self.cortex,
                            steps=14,
                        )
                    except Exception:
                        pass

                with self.lock:
                    self.hippocampus.studied_topics[topic.lower()] = {
                        "topic": topic,
                        "mastery": 85.0 if topic_cnt > 0 else 40.0,
                        "facts_learned": topic_cnt,
                        "pages_read": len(docs),
                        "day": self.day,
                        "summary": f"Zakodowano {topic_cnt} engramów SDR.",
                        "sources": sources_list[-4:],
                    }

                since_last_sleep += 1
                if self.auto_sleep and (
                    since_last_sleep >= 4 or len(self.hippocampus.short_term) >= self.cfg.auto_sleep_pending_threshold
                ):
                    with self.lock:
                        self.study_state["current_action"] = f"Sen NREM+SHY+REM po {topics_done} tematach na GPU..."
                    self.sleep_now(visual_pacing=False, _from_nested=True)
                    sleeps_done += 1
                    since_last_sleep = 0

            if len(self.hippocampus.short_term) > 0:
                self.sleep_now(visual_pacing=False, _from_nested=True)
                sleeps_done += 1

            stopped = self._study_stop_event.is_set()
            with self.lock:
                self.study_state["mastery"] = 100.0
                self._log(
                    "STUDY",
                    f"⏱️ {'Zatrzymano' if stopped else 'Zakończono'} Auto-naukę: {topics_done} tematów, "
                    f"{total_pages} stron, +{total_facts} engramów SDR, {sleeps_done} nocy snu.",
                )
                self._append_chat(
                    "assistant",
                    f"{'⏹️ Zatrzymano i zapisano' if stopped else '✅ Zakończono'} **Auto-naukę**! "
                    f"Zbadano **{topics_done} nowych tematów** ({total_pages} źródeł, **+{total_facts} engramów SDR**, "
                    f"{sleeps_done} cykli snu NREM+REM na GPU).",
                    meta={"kind": "study_done", "topic": f"Auto-nauka ({topics_done} tematów)"},
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

    @staticmethod
    def _parse_web_study_intent(msg: str) -> str | None:
        s = msg.strip()
        low = s.lower()
        for prefix in ("/ucz ", "/study ", "/zbadaj ", "/naucz "):
            if low.startswith(prefix):
                return s[len(prefix) :].strip()
        return None

    def start_study_async(self, topic: str, max_rounds: int | None = None, consolidate_after: bool = True) -> dict[str, Any]:
        clean = topic.strip()
        if not clean:
            raise ValueError("Podaj temat do nauki.")
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
                "current_action": f"Pobieram bodźce dla tematu: {clean}",
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
            self.study_state["current_action"] = "Zatrzymywanie nauki..."

    def study_topic_online(
        self,
        topic: str,
        max_rounds: int | None = None,
        consolidate_after: bool = True,
        _op_acquired: bool = False,
    ) -> dict[str, Any]:
        """
        Fetch web sources via `web_learner.py` and feed them purely as sensory stimuli into
        Wernicke -> SDR engrams in Hippocampus + Broca denoising pre-warming buffer (NEVER as quotes).
        """
        raw_topic = topic.strip()
        if not raw_topic:
            raise ValueError("Podaj temat do nauki.")
        clean_topic = raw_topic if is_http_url(raw_topic) else clean_search_query(raw_topic)
        if not clean_topic:
            clean_topic = raw_topic
        rounds = 1 if is_http_url(clean_topic) else max(1, min(8, int(max_rounds or self.cfg.study_max_rounds)))

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
                "current_action": f"Szukam źródeł dla „{clean_topic}”...",
                "sources": [],
                "summary": "",
            }
            self._log("STUDY", f"🌐 Nauka z sieci: '{clean_topic}' (do {rounds} rund).")

        seen_urls: set[str] = set()
        pending_queries = (
            [clean_topic]
            if is_http_url(clean_topic)
            else self.language_cortex.normalize_study_queries(clean_topic) or [clean_topic]
        )
        sources_list: list[dict[str, str]] = []
        total_engrams = 0
        sleeps_during_study = 0

        try:
            for r_idx in range(1, rounds + 1):
                if self._study_stop_event.is_set():
                    break
                self.study_state["round"] = r_idx
                query = pending_queries.pop(0) if pending_queries else f"{clean_topic} szczegóły"
                self.study_state["current_action"] = f"Runda {r_idx}/{rounds}: pobieram „{query}”..."

                docs = gather_web_sources(
                    query=query,
                    seen_urls=seen_urls,
                    max_docs=self.cfg.study_pages_per_round,
                    timeout=self.cfg.study_http_timeout,
                    max_workers=self.cfg.study_fetch_workers,
                )
                if not docs:
                    if not pending_queries:
                        break
                    continue

                for doc in docs:
                    if self._study_stop_event.is_set():
                        break
                    self.study_state["current_action"] = f"Koduję bodźce z „{doc.title}” przez Wernickego..."
                    sources_list.append({"title": doc.title, "url": doc.url, "source": doc.source})
                    self.study_state["sources"] = sources_list[-12:]
                    self.study_state["pages_read"] += 1

                    stimuli: list[str] = []
                    lead = extract_lead_definition(doc)
                    if lead:
                        stimuli.append(lead)
                    for ch in chunk_text(doc.text, focus_query=query, max_chars=2400, max_chunks=2):
                        stimuli.extend(self.language_cortex.extract_sensory_stimuli(ch, max_items=6))
                    added = self._ingest_sensory_texts(
                        stimuli, category="nauka_z_sieci", topic=clean_topic, source_url=doc.url
                    )
                    total_engrams += len(added)
                    self.study_state["facts_learned"] = total_engrams

                mastery = min(100.0, round(25.0 + total_engrams * 7.5, 1)) if total_engrams > 0 else 0.0
                self.study_state["mastery"] = mastery
                if mastery >= self.cfg.study_target_mastery:
                    break

            mastery = float(self.study_state["mastery"])
            summary = f"Zakodowano {total_engrams} engramów SDR z {self.study_state['pages_read']} źródeł."
            with self.lock:
                self.hippocampus.studied_topics[clean_topic.lower()] = {
                    "topic": clean_topic,
                    "mastery": mastery,
                    "facts_learned": total_engrams,
                    "pages_read": self.study_state["pages_read"],
                    "day": self.day,
                    "summary": summary,
                    "sources": sources_list[:10],
                }

            sleep_rep = None
            if consolidate_after and len(self.hippocampus.short_term) > 0 and not self._study_stop_event.is_set():
                self.study_state["current_action"] = "Sen NREM + SHY + REM na GPU..."
                sleep_rep = self.sleep_now(visual_pacing=False, _from_nested=True)
                sleeps_during_study += 1

            result = {
                "topic": clean_topic,
                "mastery": mastery,
                "pages_read": self.study_state["pages_read"],
                "facts_learned": total_engrams,
                "sleeps": sleeps_during_study,
                "summary": summary,
                "sources": sources_list,
                "sleep_report": sleep_rep.to_dict() if sleep_rep else None,
            }
            with self.lock:
                self._append_chat(
                    "assistant",
                    f"✅ Zakończono naukę „{clean_topic}”: {result['pages_read']} źródeł, +{total_engrams} engramów SDR (opanowanie {mastery:.0f}%).",
                    meta={"kind": "study_done", "topic": clean_topic, "mastery": mastery, "facts": total_engrams},
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

    def train_language_now(self, epochs: int | None = None, _from_nested: bool = False) -> dict[str, Any]:
        eff_epochs = int(epochs if epochs is not None else self.cfg.language_train_epochs)
        if not _from_nested:
            self._acquire_op("study")
            with self.lock:
                self._study_stop_event.clear()
                self.state_mode = "STUDYING"
                self.training_dialogue_feed.clear()
                self.study_state = {
                    "active": True,
                    "topic": "🗣️ Dostrajanie dialogów i Q&A (GPU)",
                    "round": 0,
                    "max_rounds": eff_epochs,
                    "mastery": 0.0,
                    "pages_read": 0,
                    "facts_learned": 0,
                    "current_action": "Rozpoczynam naukę dialogów i odpowiedzi (Wernicke → Kora → Broca)...",
                    "sources": [],
                    "summary": "",
                }

        def _on_dialogue_progress(st: dict[str, Any]) -> None:
            ep = int(st.get("epoch", 0))
            tot = max(1, int(st.get("total_epochs", eff_epochs)))
            phase = str(st.get("phase", "dialogue_train"))
            loss_s = float(st.get("loss_start", 0.0))
            loss_n = float(st.get("loss_now", 0.0))
            flu = float(st.get("fluency", 0.0))
            dial_n = int(st.get("dialogues", 0))
            pct = round(90.0 * ep / tot, 1) if phase == "dialogue_train" else round(90.0 + 10.0 * ep / tot, 1)
            phase_lbl = "Szlifowanie trudnych par" if phase == "dialogue_polish" else "Epoka"
            with self.lock:
                if not _from_nested:
                    self.study_state["round"] = ep
                    self.study_state["max_rounds"] = tot
                    self.study_state["facts_learned"] = dial_n
                    self.study_state["mastery"] = pct
                    self.study_state["current_action"] = (
                        f"{phase_lbl} {ep}/{tot} · {dial_n} par · Loss: {loss_s:.2f}→{loss_n:.2f} · Płynność: {flu:.0f}%"
                    )
                if st.get("fresh_probe") and st.get("sample_q"):
                    self._push_training_dialogue(
                        mode="dialogue_train",
                        tag=f"{phase_lbl} {ep}/{tot} · loss {loss_n:.2f}",
                        prompt_role="Pytanie",
                        prompt=str(st["sample_q"]),
                        teacher_role="Wzorzec",
                        teacher_reply=str(st.get("sample_target", "")),
                        brain_role="Twój mózg (na żywo)",
                        brain_reply=str(st.get("sample_brain", "")) or "(jeszcze składa pierwsze słowa...)",
                    )

        try:
            # Fine-tune dialogue & Q&A cleanly through Wernicke -> Cortex -> Broca without stale engram interference
            res = self.language_cortex.train_language_cortex(
                epochs=eff_epochs,
                include_curriculum=True,
                cortex=self.cortex,
                engram_bank=None,
                progress_callback=_on_dialogue_progress,
                stop_event=self._study_stop_event if not _from_nested else None,
            )
            # Re-align cortical synaptic weights & curriculum engrams to the updated Wernicke SDR space
            with self.lock:
                had_curriculum = "szkoła podstawowa — fundamenty" in self.hippocampus.studied_topics
                self.cortex.reset_synaptic_weights()
                self.hippocampus.reset_all_banks()

            if had_curriculum:
                texts = [str(item["text"]) for item in ELEMENTARY_FOUNDATION_PACK]
                imps = [int(item.get("importance", 4)) for item in ELEMENTARY_FOUNDATION_PACK]
                w_re = self.language_cortex.perceive_texts(texts)
                from .hippocampus import EngramMeta

                with self.lock:
                    self.hippocampus.semantic.add_batch(
                        sdrs=w_re.sdr,
                        metas=[
                            EngramMeta(
                                category="wiedza",
                                preview_label=txt[:200],
                                sparsity_pct=w_re.sparsity_pct,
                                topic="Szkoła podstawowa — fundamenty",
                            )
                            for txt in texts
                        ],
                        strengths=[1.5] * len(texts),
                        importances=[float(x) for x in imps],
                        day=self.day,
                    )
                    self.cortex.learn_local(w_re.columns, w_re.sdr, steps=12, noise_std=0.02)
                    self.cortex.apply_synaptic_homeostasis()

            with self.lock:
                self._log(
                    "STUDY",
                    f"🗣️ Dostrajanie dialogowe + Q&A na GPU ({res['dialogues']} par, {res['epochs_run']} epok, "
                    f"loss {res['loss_start']}→{res['loss_end']}, płynność {res['fluency']:.0f}%).",
                )
                if not _from_nested:
                    self._append_chat(
                        "assistant",
                        f"🗣️ Zakończono dostrajanie dialogowe + Q&A (`Wernicke → Kora → Broca`, {res['dialogues']} par, "
                        f"{res['epochs_run']} epok, loss **{res['loss_start']} → {res['loss_end']}**, płynność **{res['fluency']:.0f}%**). "
                        f"Kora i engramy zostały wyrównane do nowej przestrzeni Wernickego — możesz już rozmawiać!",
                        meta={"kind": "study_done", "topic": "Dostrajanie dialogowe GPU"},
                        add_to_working=False,
                    )
            self.save_state()
            return res
        finally:
            if not _from_nested:
                with self.lock:
                    self.study_state["active"] = False
                    self.study_state["current_action"] = ""
                    if self.state_mode == "STUDYING":
                        self.state_mode = "AWAKE"
                self._release_op("study")

    # ── Streaming Polish Grammar & Fluency Training from Local Parquet(s) ────
    def _approved_parquet_paths(self) -> list[Path]:
        return [
            self.base_dir / name
            for name in PARQUET_CORPORA_CATALOG
            if (self.base_dir / name).is_file()
        ]

    def _resolve_parquet_paths(self, filename: str = "__ALL__") -> list[Path]:
        raw = (filename or "__ALL__").strip()
        if raw in ("__ALL__", "all", "*"):
            return self._approved_parquet_paths()
        cand = Path(raw)
        if cand.name in PARQUET_CORPORA_CATALOG:
            if cand.is_absolute():
                return [cand] if cand.is_file() else []
            p = self.base_dir / cand.name
            if p.is_file():
                return [p]
        found = self._approved_parquet_paths()
        return [found[0]] if found else []

    def _resolve_parquet_path(self, filename: str = "train-00000-of-00006.parquet") -> Path:
        paths = self._resolve_parquet_paths(filename)
        if paths:
            return paths[0]
        return self.base_dir / filename

    def get_parquet_corpus_info(self, filename: str = "__ALL__") -> dict[str, Any]:
        """Discover only the approved Polish Parquet corpora in the root folder with lightweight mtime caching."""
        found = self._approved_parquet_paths()
        sig = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in found)
        cached = getattr(self, "_parquet_info_cache", None)
        if cached is not None and cached.get("_sig") == sig:
            out = dict(cached)
            out.pop("_sig", None)
            return out

        files_info = [inspect_parquet_corpus(p) for p in found]
        if not files_info:
            res: dict[str, Any] = {
                "exists": False,
                "path": str(self.base_dir / "train-00000-of-00006.parquet"),
                "filename": "train-00000-of-00006.parquet",
                "size_mb": 0.0,
                "num_rows": 0,
                "file_count": 0,
                "total_size_mb": 0.0,
                "total_rows": 0,
                "files": [],
            }
        else:
            total_mb = round(sum(float(x["size_mb"]) for x in files_info), 2)
            total_rows = int(sum(int(x["num_rows"]) for x in files_info))
            primary = next(
                (x for x in files_info if x["filename"] == "train-00000-of-00006.parquet"),
                files_info[0],
            )
            res = {
                "exists": True,
                "path": str(primary["path"]),
                "filename": str(primary["filename"]),
                "size_mb": total_mb if len(files_info) > 1 else float(primary["size_mb"]),
                "num_rows": total_rows if len(files_info) > 1 else int(primary["num_rows"]),
                "file_count": len(files_info),
                "total_size_mb": total_mb,
                "total_rows": total_rows,
                "files": files_info,
            }
        self._parquet_info_cache = {"_sig": sig, **res}
        return res

    def start_parquet_polish_async(
        self,
        filename: str = "__ALL__",
        max_steps: int = 5000,
        batch_size: int | None = None,
        lr: float | None = None,
    ) -> dict[str, Any]:
        raw = (filename or "__ALL__").strip()
        paths = self._resolve_parquet_paths(raw)
        will_download = False
        if not paths:
            if raw in ("__ALL__", "all", "*"):
                will_download = True
                target_names = list(PARQUET_CORPORA_CATALOG.keys())
            elif raw in PARQUET_CORPORA_CATALOG:
                will_download = True
                target_names = [raw]
            else:
                raise FileNotFoundError(f"Nie znaleziono pliku Parquet w głównym folderze: {filename}")
        else:
            target_names = [x.name for x in paths]
        label = (
            target_names[0]
            if len(target_names) == 1
            else f"Wszystkie ({len(target_names)} pliki: {', '.join(target_names)})"
        )
        steps = max(20, min(1000000, int(max_steps)))
        default_bs = self._default_parquet_batch_size()
        bs = int(batch_size) if batch_size is not None else default_bs
        self._acquire_op("study")
        with self.lock:
            self._study_stop_event.clear()
            self.state_mode = "STUDYING"
            self.study_state = {
                "active": True,
                "topic": f"🇵🇱 Polski i gramatyka ({label})",
                "round": 0,
                "max_rounds": steps,
                "mastery": 0.0,
                "pages_read": 0,
                "facts_learned": 0,
                "current_action": (
                    f"Pobieram brakujący korpus Parquet z HuggingFace: {label}..."
                    if will_download
                    else f"Otwieram strumień Parquet: {label} na GPU (bfloat16)..."
                ),
                "sources": [],
                "summary": "",
            }
        try:
            t = threading.Thread(
                target=self.train_polish_from_parquet,
                args=(filename, steps, bs, lr, True),
                daemon=True,
            )
            t.start()
        except Exception:
            with self.lock:
                self.study_state["active"] = False
                self.state_mode = "AWAKE"
            self._release_op("study")
            raise
        return {
            "started": True,
            "filename": label,
            "files": target_names,
            "downloading": will_download,
            "max_steps": steps,
            "batch_size": bs,
        }

    def _default_parquet_batch_size(self) -> int:
        if self.device.type != "cuda":
            return 16
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
        except Exception:
            vram_gb = 12.0
        # Measured on RTX 5080 (16 GB) with Sparse MoE Broca (273M params, 8 experts, top-2)
        # + gradient checkpointing: bs=32 keeps peak VRAM comfortably at ~8-10 GB
        # (ZERO Windows WDDM Shared RAM spill, full Tensor-Core throughput, identical quality).
        if self.cfg.broca_dim >= 640:
            if getattr(self.cfg, "broca_use_moe", False):
                return 32 if vram_gb >= 14.0 else 24
            return 48 if vram_gb >= 14.0 else 32
        return 96

    def train_polish_from_parquet(
        self,
        parquet_path: str | Path | None = None,
        max_steps: int = 5000,
        batch_size: int | None = None,
        lr: float | None = None,
        _op_acquired: bool = False,
    ) -> dict[str, Any]:
        """
        Stream Polish sentences from one or all local *.parquet corpora (e.g. `train-00000-of-00006.parquet`
        + `000_00000.parquet`) on GPU (bfloat16) to train fluent Polish spelling, grammar, and coherent
        sentence generation in BrocaByteTransformer through Wernicke's k-WTA SDR bottleneck.
        """
        raw_arg = str(parquet_path).strip() if parquet_path else "__ALL__"
        paths = self._resolve_parquet_paths(raw_arg)
        if not paths:
            download_targets: list[str] | None = None
            if raw_arg in ("__ALL__", "all", "*"):
                download_targets = list(PARQUET_CORPORA_CATALOG.keys())
            elif raw_arg in PARQUET_CORPORA_CATALOG:
                download_targets = [raw_arg]
            if download_targets:
                if not _op_acquired:
                    self._acquire_op("study")
                    _op_acquired = True
                with self.lock:
                    self._study_stop_event.clear()
                    self.state_mode = "STUDYING"
                    self.study_state["active"] = True
                    self.study_state["current_action"] = (
                        f"Pobieram korpus Parquet z HuggingFace ({', '.join(download_targets)})..."
                    )

                def _on_dl_progress(dl: dict[str, Any]) -> None:
                    with self.lock:
                        self.study_state["mastery"] = float(dl.get("pct", 0.0))
                        total_mb = dl.get("total_mb", 0.0)
                        total_txt = f" / {total_mb} MB ({dl.get('pct', 0.0)}%)" if total_mb else ""
                        self.study_state["current_action"] = (
                            f"Pobieram {dl['filename']}: {dl['downloaded_mb']} MB{total_txt}..."
                        )

                try:
                    paths = ensure_parquet_corpora(
                        self.base_dir,
                        filenames=download_targets,
                        progress_cb=_on_dl_progress,
                        stop_event=self._study_stop_event,
                    )
                    self._parquet_info_cache = None
                except InterruptedError:
                    with self.lock:
                        self.study_state["active"] = False
                        self.study_state["current_action"] = "Przerwano pobieranie korpusu Parquet."
                        self.state_mode = "AWAKE"
                    self._release_op("study")
                    return {"stopped": True, "downloaded": False}
                except Exception as exc:
                    with self.lock:
                        self.study_state["active"] = False
                        self.study_state["current_action"] = f"Błąd pobierania Parquet: {exc}"
                        self.state_mode = "AWAKE"
                    self._release_op("study")
                    raise
            if not paths:
                if _op_acquired:
                    with self.lock:
                        self.study_state["active"] = False
                        self.state_mode = "AWAKE"
                    self._release_op("study")
                raise FileNotFoundError(f"Nie znaleziono pliku Parquet: {raw_arg}")

        label = paths[0].name if len(paths) == 1 else f"Wszystkie ({len(paths)} pliki: {', '.join(x.name for x in paths)})"
        steps = max(1, min(1000000, int(max_steps)))
        default_bs = self._default_parquet_batch_size()
        bs = int(batch_size) if batch_size is not None else default_bs
        if not _op_acquired:
            self._acquire_op("study")

        with self.lock:
            self._study_stop_event.clear()
            self.state_mode = "STUDYING"
            self.training_dialogue_feed.clear()
            self.study_state = {
                "active": True,
                "topic": f"🇵🇱 Polski i gramatyka ({label})",
                "round": 0,
                "max_rounds": steps,
                "mastery": 0.0,
                "pages_read": 0,
                "facts_learned": 0,
                "current_action": f"Strumieniowy trening gramatyki z {label} (batch={bs}, bfloat16)...",
                "sources": [],
                "summary": "",
            }
            self._log("STUDY", f"🇵🇱 Start treningu języka polskiego i gramatyki z {label} ({steps} kroków, batch={bs}).")
            self._append_chat(
                "assistant",
                f"🇵🇱 Rozpoczynam naukę pisania i polskiej gramatyki z `{label}` ({steps} kroków, batch={bs}, `bfloat16` na GPU). "
                f"Możesz w każdej chwili zatrzymać trening przyciskiem **Stop** — dotychczasowe postępy zostaną zapisane!",
                meta={"kind": "study_start", "topic": label},
                add_to_working=False,
            )

        def _on_progress(st: dict[str, Any]) -> None:
            pct = round(100.0 * st["step"] / max(1, st["max_steps"]), 1)
            preview = f" · Próbka: „{st['sample_preview'][:75]}”" if st.get("sample_preview") else ""
            with self.lock:
                self.study_state["round"] = st["step"]
                self.study_state["max_rounds"] = st["max_steps"]
                self.study_state["pages_read"] = st["step"]
                self.study_state["facts_learned"] = st["samples"]
                self.study_state["mastery"] = pct
                self.study_state["current_action"] = (
                    f"Krok {st['step']}/{st['max_steps']} · Zdań: {st['samples']:,} ({st['samples_per_sec']:.0f}/s) · "
                    f"Loss: {st['loss_start']:.2f}→{st['loss_now']:.2f} · Płynność: {st['fluency']:.0f}%{preview}"
                )
                if st.get("fresh_probe") and st.get("sample_prompt"):
                    s_prompt = str(st["sample_prompt"])
                    s_target = str(st.get("sample_target", ""))
                    self._push_training_dialogue(
                        mode="parquet",
                        tag=f"Parquet krok {st['step']}/{st['max_steps']} · loss {st['loss_now']:.2f}",
                        prompt_role="Zdanie z korpusu",
                        prompt=s_prompt,
                        teacher_role="Cel (dalszy ciąg)" if s_target and s_target != s_prompt else "",
                        teacher_reply=s_target if s_target and s_target != s_prompt else "",
                        brain_role="Twój mózg (Broca na żywo)",
                        brain_reply=str(st.get("sample_preview", "")) or "(składa pierwsze litery...)",
                    )
                elif st["step"] == 1 and st.get("sample_prompt"):
                    s_prompt = str(st["sample_prompt"])
                    s_target = str(st.get("sample_target", ""))
                    self._push_training_dialogue(
                        mode="parquet",
                        tag=f"Parquet start · krok 1/{st['max_steps']}",
                        prompt_role="Czytam z korpusu",
                        prompt=s_prompt,
                        teacher_role="Cel" if s_target and s_target != s_prompt else "",
                        teacher_reply=s_target if s_target and s_target != s_prompt else "",
                    )

        def _on_checkpoint() -> None:
            self.save_state_async()

        try:
            # Start from a rotating row group based on trained_steps so repeated runs read fresh articles
            start_rg = int(self.language_cortex.broca_module.trained_steps // 200)
            res = self.language_cortex.train_parquet_polish_stream(
                pair_iterable_factory=lambda: iter_parquet_polish_pairs(
                    paths if len(paths) > 1 else paths[0],
                    continuation_ratio=0.35,
                    min_chars=32,
                    max_chars=230,
                    arrow_batch_size=128,
                    start_row_group=start_rg,
                    min_quality_score=0.45,
                    min_language_score=0.85,
                ),
                max_steps=steps,
                batch_size=bs,
                lr=lr,
                early_stop_loss=self.cfg.broca_early_stop_loss,
                mask_prob=0.16,
                stop_event=self._study_stop_event,
                progress_callback=_on_progress,
                checkpoint_callback=_on_checkpoint,
                checkpoint_every=2000,
            )

            # Re-align Hippocampus, Cortex & Dialogue Q&A after corpus pre-training:
            # 1) Automatically polish Broca back to Polish dialogue + Q&A (if substantial pre-training ran)
            #    so the brain never stays stuck in Wikipedia-continuation mode!
            # 2) Reset stale W_in / W_rec / threshold and re-encode curriculum entries in the updated Wernicke space.
            if res["trained"]:
                with self.lock:
                    had_curriculum = "szkoła podstawowa — fundamenty" in self.hippocampus.studied_topics
                    self.cortex.reset_synaptic_weights()
                    self.hippocampus.reset_all_banks()

                if res["steps"] >= 50:
                    with self.lock:
                        self.study_state["current_action"] = "Dostrajam dialogi i Q&A po nauce z Parquet (GPU)..."

                    def _on_post_pq_dialogue(st: dict[str, Any]) -> None:
                        if st.get("fresh_probe") and st.get("sample_q"):
                            with self.lock:
                                self._push_training_dialogue(
                                    mode="dialogue_train",
                                    tag=f"Wyrównanie dialogów {st['epoch']}/{st['total_epochs']} · loss {st['loss_now']:.2f}",
                                    prompt_role="Pytanie",
                                    prompt=str(st["sample_q"]),
                                    teacher_role="Wzorzec",
                                    teacher_reply=str(st.get("sample_target", "")),
                                    brain_role="Twój mózg (na żywo)",
                                    brain_reply=str(st.get("sample_brain", "")) or "(odpowiada...)",
                                )

                    saved_teacher_pairs = load_teacher_dialogues(self.state_dir)
                    self.language_cortex.train_language_cortex(
                        extra_qa_pairs=saved_teacher_pairs or None,
                        epochs=90,
                        include_curriculum=True,
                        cortex=self.cortex,
                        engram_bank=None,
                        progress_callback=_on_post_pq_dialogue,
                        stop_event=self._study_stop_event,
                    )

                reencode_texts: list[str] = []
                reencode_imps: list[int] = []
                if had_curriculum:
                    for item in ELEMENTARY_FOUNDATION_PACK:
                        reencode_texts.append(str(item["text"]))
                        reencode_imps.append(int(item.get("importance", 4)))

                if reencode_texts:
                    w_re = self.language_cortex.perceive_texts(reencode_texts)
                    with self.lock:
                        self.hippocampus.semantic.add_batch(
                            sdrs=w_re.sdr,
                            metas=[
                                __import__("brain.hippocampus", fromlist=["EngramMeta"]).EngramMeta(
                                    category="wiedza",
                                    preview_label=txt[:200],
                                    sparsity_pct=w_re.sparsity_pct,
                                    topic="Szkoła podstawowa — fundamenty",
                                    pinned=(imp >= 4),
                                )
                                for txt, imp in zip(reencode_texts, reencode_imps)
                            ],
                            strengths=[1.5] * len(reencode_texts),
                            importances=[float(x) for x in reencode_imps],
                            day=self.day,
                        )
                        self.cortex.learn_local(w_re.columns, w_re.sdr, steps=12, noise_std=0.03)
                        self.cortex.apply_synaptic_homeostasis()
                    self._log(
                        "STUDY",
                        f"🔄 Zresetowano W_in/W_rec przed dostrajaniem dialogów i przekodowano {len(reencode_texts)} engramów do nowej przestrzeni Wernickego.",
                    )
                else:
                    self._log(
                        "STUDY",
                        "🔄 Zresetowano przestarzałe wagi kory (W_in/W_rec) i wyczyszczono nieaktualne engramy po pre-trainingu.",
                    )

            stopped = self._study_stop_event.is_set()
            preview_note = f"\nPrzykładowa próbka Broki: *„{res['sample_preview']}”*" if res.get("sample_preview") else ""
            with self.lock:
                self.hippocampus.studied_topics[label.lower()] = {
                    "topic": f"🇵🇱 Polski & gramatyka ({label})",
                    "mastery": float(res["fluency"]),
                    "facts_learned": int(res["samples"]),
                    "pages_read": int(res["steps"]),
                    "day": self.day,
                    "summary": f"Loss {res['loss_start']}→{res['loss_end']} · Płynność {res['fluency']:.0f}% ({res['samples']:,} zdań).",
                    "sources": [],
                }
                self._log(
                    "STUDY",
                    f"🇵🇱 {'Zatrzymano' if stopped else 'Zakończono'} trening z {label}: "
                    f"{res['steps']} kroków, {res['samples']:,} zdań, loss {res['loss_start']}→{res['loss_end']}, "
                    f"płynność {res['fluency']:.0f}%.",
                )
                self._append_chat(
                    "assistant",
                    f"{'⏹️ Zatrzymano i zapisano' if stopped else '✅ Zakończono'} naukę języka polskiego z `{label}`!\n"
                    f"Przerobiono **{res['samples']:,} polskich zdań** ({res['steps']} kroków w {res['duration_ms']/1000:.1f}s), "
                    f"Broca Loss: **{res['loss_start']} → {res['loss_end']}**, płynność: **{res['fluency']:.0f}%**.{preview_note}\n"
                    f"Dialogi i Q&A zostały automatycznie wyrównane — możesz od razu rozmawiać!",
                    meta={"kind": "study_done", "topic": label, "facts": res["samples"]},
                    add_to_working=False,
                )
            self.save_state()
            trim_process_ram()
            return res
        finally:
            with self.lock:
                self.study_state["active"] = False
                self.study_state["current_action"] = ""
                if self.state_mode == "STUDYING":
                    self.state_mode = "AWAKE"
            self._release_op("study")

    # Alias matching corpus pretrain terminology
    run_corpus_pretrain = train_polish_from_parquet

    def start_teacher_distill_async(
        self,
        num_dialogues: int = 80,
        train_epochs: int = 160,
        skip_generation: bool = False,
    ) -> dict[str, Any]:
        """
        Launch 2-phase Bielik Teacher -> Self-Sufficient Brain distillation in background:
          Phase 1: Temporarily offload brain to CPU RAM, run local cached Bielik-4.5B-v3-heretic
                   to generate natural Polish conversational pairs saved to teacher_dialogues.jsonl,
                   then completely unload Bielik from VRAM.
          Phase 2: Train our OWN Wernicke + Broca + SDR Attractor Bank on RTX 5080 (bfloat16)
                   so the brain writes natural Polish replies completely WITHOUT Bielik!
        """
        n_dial = max(5, min(1000, int(num_dialogues)))
        n_ep = max(20, min(600, int(train_epochs)))
        self._acquire_op("study")
        with self.lock:
            self._study_stop_event.clear()
            self.state_mode = "STUDYING"
            self.study_state = {
                "active": True,
                "topic": "🎓 Nauka dialogów z Bielikiem → Twój mózg",
                "round": 0,
                "max_rounds": n_dial,
                "mastery": 0.0,
                "pages_read": 0,
                "facts_learned": count_teacher_dialogues(self.state_dir),
                "current_action": "Przygotowuję sesję nauki dialogów (Nauczyciel Bielik → Twój mózg)...",
                "sources": [],
                "summary": "",
            }
        try:
            t = threading.Thread(
                target=self.train_with_bielik_teacher,
                args=(n_dial, n_ep, bool(skip_generation), True),
                daemon=True,
            )
            t.start()
        except Exception:
            with self.lock:
                self.study_state["active"] = False
                self.state_mode = "AWAKE"
            self._release_op("study")
            raise
        return {
            "started": True,
            "num_dialogues": n_dial,
            "train_epochs": n_ep,
            "skip_generation": bool(skip_generation),
            "existing_dialogues": count_teacher_dialogues(self.state_dir),
        }

    @staticmethod
    def _is_substantive_teacher_knowledge(q: str, a: str) -> bool:
        """Filter out trivial greetings/chatter so only substantive Bielik knowledge enters Hipokamp."""
        q_low = _strip_pl((q or "").strip().lower()).rstrip("?!., ")
        a_clean = (a or "").strip()
        if len(a_clean) < 28 or len(a_clean.split()) < 4:
            return False
        chatter_qs = {
            "siema", "siema co tam", "elo", "elo mordo", "hej", "hejka", "czesc", "czesc jak leci",
            "yo co slychac", "spoko a u ciebie", "co tam u cb", "jak mija dzien", "haha dobre",
            "xddd", "spoko dzieki", "nie gadaj", "wez przestan", "dobranoc ide spac",
            "narazie spadam", "zaraz wracam", "dobra niewazne", "no i co teraz", "i co z tego",
            "jestes spoko", "lubie z toba gadac", "dobra robota", "halo jestes tam",
        }
        if q_low in chatter_qs:
            return False
        return True

    def _ingest_teacher_pairs_to_short_term(
        self,
        pairs: list[tuple[str, str]],
        max_items: int = 80,
    ) -> dict[str, int]:
        """
        Encode substantive knowledge from Bielik teacher dialogues into Hipokamp (short-term memory),
        local associative synapses W_in/W_rec, and LivingBrainEngine (M1/M2/M4) so the brain actively
        absorbs facts during training and later links them into long-term memory during sleep.
        """
        with self.lock:
            existing_labels = {
                (m.preview_label or "").strip().lower()
                for m in (self.hippocampus.short_term.meta + self.hippocampus.semantic.meta)
                if m.preview_label
            }

        texts: list[str] = []
        importances: list[int] = []
        seen_local: set[str] = set()
        for q, a in pairs:
            if is_ai_corporate_reply(a):
                continue
            if not self._is_substantive_teacher_knowledge(q, a):
                continue
            lbl = a.strip()[:200]
            key = lbl.lower()
            if key in existing_labels or key in seen_local:
                continue
            seen_local.add(key)
            texts.append(lbl)
            importances.append(4 if len(lbl) >= 75 else 3)
            if len(texts) >= max_items:
                break

        if not texts:
            return {"engrams_added": 0, "living_steps": 0, "living_neurons_born": 0}

        w_out = self.language_cortex.perceive_texts(texts)
        with self.lock:
            ids = self.hippocampus.record_engrams_batch(
                sdrs=w_out.sdr,
                preview_labels=texts,
                category="wiedza_bielik",
                importances=importances,
                day=self.day,
                sparsity_pct=w_out.sparsity_pct,
                topic="Nauczyciel Bielik",
            )
            self.cortex.learn_local(w_out.columns, w_out.sdr, steps=6, noise_std=0.015)
            self.cortex.apply_synaptic_homeostasis()
            self._buffer_wake_stimulus(texts)

        # Let LivingBrainEngine (M1/M2/M4) actively absorb & grow from a representative slice of new facts
        n_cols = int(w_out.columns.shape[0])
        lb_cap = min(n_cols, 16)
        lb_steps = 0
        lb_born = 0
        if lb_cap > 0:
            step_stride = max(1, n_cols // lb_cap)
            for idx in range(0, n_cols, step_stride):
                if lb_steps >= lb_cap:
                    break
                try:
                    tel = self.living_brain.cognitive_step(
                        sensory_vec=w_out.columns[idx],
                        external_valence=0.35,
                        allow_growth=True,
                        spontaneous=False,
                    )
                    lb_steps += 1
                    lb_born += int(tel.neurons_added)
                except Exception:
                    break

        return {
            "engrams_added": len(ids),
            "living_steps": lb_steps,
            "living_neurons_born": lb_born,
        }

    def train_with_bielik_teacher(
        self,
        num_dialogues: int = 80,
        train_epochs: int = 160,
        skip_generation: bool = False,
        _op_acquired: bool = False,
    ) -> dict[str, Any]:
        if not _op_acquired:
            self._acquire_op("study")
        generated_n = 0
        gen_err = ""
        session_new_pairs: list[tuple[str, str]] = []
        with self.lock:
            self.training_dialogue_feed.clear()
        try:
            if not skip_generation:
                with self.lock:
                    self._log("STUDY", f"Start nauki dialogów z nauczycielem Bielikiem ({num_dialogues} par)...")
                    self._append_chat(
                        "assistant",
                        f"Rozpoczynam naukę dialogów z lokalnym nauczycielem **Bielik-4.5B** ({num_dialogues} nowych par). "
                        f"Po wygenerowaniu wyładuję Bielika z GPU i wytrenuję wyłącznie Twój własny mózg.",
                        meta={"kind": "study_start", "topic": "Nauczyciel Bielik"},
                        add_to_working=False,
                    )
                    extra_topics = [m.preview_label[:60] for m in self.hippocampus.semantic.meta[-60:] if m.preview_label]

                # Temporarily move our large biological brain to CPU RAM so Bielik gets clean 16 GB VRAM
                moved_to_cpu = False
                if self.device.type == "cuda":
                    with self.language_cortex._op_lock:
                        self.language_cortex._wernicke_raw.to("cpu")
                        self.language_cortex._broca_raw.to("cpu")
                        self.cortex.to("cpu")
                        moved_to_cpu = True
                    torch.cuda.empty_cache()

                gen_seen = 0

                def _on_gen_progress(st: dict[str, Any]) -> None:
                    nonlocal gen_seen
                    done = int(st.get("done", 0))
                    tot = max(1, int(st.get("total", num_dialogues)))
                    pct = round(50.0 * done / tot, 1)
                    new_pairs = st.get("new_pairs") or []
                    with self.lock:
                        self.study_state["round"] = done
                        self.study_state["max_rounds"] = tot
                        self.study_state["mastery"] = pct
                        self.study_state["facts_learned"] = int(st.get("generated", done))
                        self.study_state["current_action"] = str(st.get("message", "Generuję dialogi..."))
                        for pair in new_pairs:
                            if not isinstance(pair, dict):
                                continue
                            pq = str(pair.get("q", ""))
                            pa = str(pair.get("a", ""))
                            if pq and pa:
                                session_new_pairs.append((pq, pa))
                            gen_seen += 1
                            self._push_training_dialogue(
                                mode="teacher_gen",
                                tag=f"Bielik · dialog #{gen_seen}/{tot}",
                                prompt_role="Pytanie",
                                prompt=pq,
                                teacher_role="Bielik (Nauczyciel)",
                                teacher_reply=pa,
                            )

                gen_err = ""
                try:
                    gen_res = run_bielik_teacher_subprocess(
                        state_dir=self.state_dir,
                        model_id=self.cfg.teacher_model_id,
                        num_dialogues=num_dialogues,
                        extra_topics=extra_topics,
                        stop_event=self._study_stop_event,
                        progress_callback=_on_gen_progress,
                    )
                    generated_n = int(gen_res["generated"])
                    gen_err = str(gen_res.get("error", "") or "")
                    if gen_err:
                        with self.lock:
                            self._log("ERROR", gen_err)
                finally:
                    if moved_to_cpu:
                        with self.language_cortex._op_lock:
                            self.language_cortex._wernicke_raw.to(self.device)
                            self.language_cortex._broca_raw.to(self.device)
                            self.cortex.to(self.device)
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                    trim_process_ram()

            all_teacher_pairs = load_teacher_dialogues(self.state_dir)
            all_reasoning_pairs = [
                (r["q"], r["a"])
                for r in load_reasoning_records(self.state_dir)
                if not is_ai_corporate_reply(r["a"])
            ]
            combined_bielik_pairs = all_teacher_pairs + all_reasoning_pairs

            # Ingest substantive knowledge from Bielik into Hipokamp BEFORE training so Hopfield engram_bank
            # already contains the newly learned Bielik facts during Broca/Wernicke training!
            knowledge_source = session_new_pairs if session_new_pairs else all_teacher_pairs
            st_ingest = self._ingest_teacher_pairs_to_short_term(
                knowledge_source,
                max_items=max(80, min(500, len(knowledge_source))),
            ) if knowledge_source else {"engrams_added": 0, "living_steps": 0, "living_neurons_born": 0}
            with self.lock:
                engram_bank = self.hippocampus.get_hopfield_bank()

            base_pct = 50.0 if not skip_generation else 0.0
            span_pct = 100.0 - base_pct
            with self.lock:
                self.study_state["round"] = 0
                self.study_state["max_rounds"] = train_epochs
                self.study_state["mastery"] = max(base_pct, 5.0)
                self.study_state["facts_learned"] = len(combined_bielik_pairs)
                self.study_state["current_action"] = (
                    f"Bielik wyładowany z VRAM. Trenuję własny mózg na {len(combined_bielik_pairs)} dialogach/faktach + bazie "
                    f"({train_epochs} epok, bfloat16 GPU)..."
                )

            def _on_brain_train_progress(st: dict[str, Any]) -> None:
                ep = int(st.get("epoch", 0))
                tot = max(1, int(st.get("total_epochs", train_epochs)))
                phase = str(st.get("phase", "dialogue_train"))
                loss_s = float(st.get("loss_start", 0.0))
                loss_n = float(st.get("loss_now", 0.0))
                flu = float(st.get("fluency", 0.0))
                if phase == "dialogue_train":
                    pct = round(base_pct + (span_pct * 0.85) * ep / tot, 1)
                else:
                    pct = round(base_pct + (span_pct * 0.85) + (span_pct * 0.10) * ep / tot, 1)
                phase_lbl = "Szlifowanie mózgu" if phase == "dialogue_polish" else "Trening mózgu · epoka"
                with self.lock:
                    self.study_state["round"] = ep
                    self.study_state["max_rounds"] = tot
                    self.study_state["mastery"] = min(96.0, pct)
                    self.study_state["current_action"] = (
                        f"{phase_lbl} {ep}/{tot} · Loss: {loss_s:.2f}→{loss_n:.2f} · Płynność: {flu:.0f}%"
                    )
                    if st.get("fresh_probe") and st.get("sample_q"):
                        self._push_training_dialogue(
                            mode="dialogue_train",
                            tag=f"{phase_lbl} {ep}/{tot} · loss {loss_n:.2f}",
                            prompt_role="Pytanie",
                            prompt=str(st["sample_q"]),
                            teacher_role="Bielik / Wzorzec",
                            teacher_reply=str(st.get("sample_target", "")),
                            brain_role="Twój mózg (na żywo)",
                            brain_reply=str(st.get("sample_brain", "")) or "(jeszcze składa pierwsze słowa...)",
                        )

            probe_source = session_new_pairs if session_new_pairs else all_teacher_pairs
            train_res = self.language_cortex.train_language_cortex(
                extra_qa_pairs=combined_bielik_pairs,
                epochs=train_epochs,
                include_curriculum=True,
                cortex=self.cortex,
                engram_bank=engram_bank,
                progress_callback=_on_brain_train_progress,
                stop_event=self._study_stop_event,
                priority_probe_pairs=probe_source,
            )
            self.language_cortex.ensure_familiarity_bank(force_rebuild=True)
            self.save_state()

            st_note = (
                f" Zapisano **+{st_ingest['engrams_added']} engramów wiedzy** do Hipokampu (pamięć krótkotrwała — gotowe do połączenia we śnie)"
                if st_ingest["engrams_added"] > 0
                else ""
            )
            ng_note = (
                f" oraz pobudzono Living Brain (+{st_ingest['living_neurons_born']} nowych neuronów)"
                if st_ingest["living_neurons_born"] > 0
                else ""
            )
            err_note = f" ⚠️ {gen_err}" if gen_err else ""
            tail_note = f"{st_note}{ng_note}." if (st_note or ng_note) else ""
            msg_done = (
                f"Nauka z Bielikiem zakończona! Wygenerowano nowych dialogów: **{generated_n}** "
                f"(łącznie w bazie: **{len(all_teacher_pairs)}**). "
                f"Bielik został wyładowany z GPU, a Twój własny mózg zakończył trening "
                f"(loss `{train_res['loss_start']} → {train_res['loss_end']}`, płynność **{train_res['fluency']:.0f}%**)."
                f"{tail_note}{err_note}"
            )
            with self.lock:
                self.study_state["mastery"] = 100.0
                self.study_state["current_action"] = msg_done
                self._log("STUDY", msg_done)
                self._append_chat("assistant", msg_done, meta={"kind": "study_done"}, add_to_working=False)
            return {
                "generated": generated_n,
                "total_teacher_dialogues": len(all_teacher_pairs),
                "short_term_engrams_added": st_ingest["engrams_added"],
                "living_neurons_born": st_ingest["living_neurons_born"],
                "train_result": train_res,
            }
        finally:
            with self.lock:
                self.study_state["active"] = False
                self.study_state["current_action"] = ""
                if self.state_mode == "STUDYING":
                    self.state_mode = "AWAKE"
            self._release_op("study")

    def start_reasoning_bielik_async(
        self,
        num_samples: int = 90,
        train_epochs: int = 160,
        skip_generation: bool = False,
    ) -> dict[str, Any]:
        """Launch 3-category Bielik reasoning training (Relational + Causal + Ignorance) in background."""
        n_samples = max(6, min(600, int(num_samples)))
        n_ep = max(20, min(600, int(train_epochs)))
        self._acquire_op("study")
        with self.lock:
            self._study_stop_event.clear()
            self.state_mode = "STUDYING"
            self.study_state = {
                "active": True,
                "topic": "🧠 Rozumowanie z Bielikiem (Relacje + Logika + Niewiedza)",
                "round": 0,
                "max_rounds": n_samples,
                "mastery": 0.0,
                "pages_read": 0,
                "facts_learned": count_reasoning_dialogues(self.state_dir),
                "current_action": "Przygotowuję trening rozumowania z Bielikiem (Relacje + Logika + Niewiedza)...",
                "sources": [],
                "summary": "",
            }
        try:
            t = threading.Thread(
                target=self.train_reasoning_with_bielik,
                args=(n_samples, n_ep, bool(skip_generation), True),
                daemon=True,
            )
            t.start()
        except Exception:
            with self.lock:
                self.study_state["active"] = False
                self.state_mode = "AWAKE"
            self._release_op("study")
            raise
        return {
            "started": True,
            "num_samples": n_samples,
            "train_epochs": n_ep,
            "skip_generation": bool(skip_generation),
            "existing_reasoning": count_reasoning_dialogues(self.state_dir),
        }

    def train_reasoning_with_bielik(
        self,
        num_samples: int = 90,
        train_epochs: int = 160,
        skip_generation: bool = False,
        _op_acquired: bool = False,
    ) -> dict[str, Any]:
        if not _op_acquired:
            self._acquire_op("study")
        generated_n = 0
        filtered_ai_n = 0
        gen_err = ""
        session_new_records: list[dict[str, str]] = []
        session_new_pairs: list[tuple[str, str]] = []
        cat_labels = {"relational": "Relacje", "causal": "Logika", "ignorance": "Niewiedza"}
        with self.lock:
            self.training_dialogue_feed.clear()
        try:
            if not skip_generation:
                with self.lock:
                    self._log("STUDY", f"Start nauki rozumowania z Bielikiem ({num_samples} przykładów: Relacje + Logika + Niewiedza)...")
                    self._append_chat(
                        "assistant",
                        f"Rozpoczynam naukę rozumowania z lokalnym **Bielik-4.5B** ({num_samples} przykładów: "
                        f"Relacje, Logika przyczynowa, Naturalna niewiedza). Formułki AI są blokowane.",
                        meta={"kind": "study_start", "topic": "Rozumowanie z Bielikiem"},
                        add_to_working=False,
                    )

                moved_to_cpu = False
                if self.device.type == "cuda":
                    with self.language_cortex._op_lock:
                        self.language_cortex._wernicke_raw.to("cpu")
                        self.language_cortex._broca_raw.to("cpu")
                        self.cortex.to("cpu")
                        moved_to_cpu = True
                    torch.cuda.empty_cache()

                gen_seen = 0

                def _on_gen_progress(st: dict[str, Any]) -> None:
                    nonlocal gen_seen
                    done = int(st.get("done", 0))
                    tot = max(1, int(st.get("total", num_samples)))
                    pct = round(50.0 * done / tot, 1)
                    new_pairs = st.get("new_pairs") or []
                    with self.lock:
                        self.study_state["round"] = done
                        self.study_state["max_rounds"] = tot
                        self.study_state["mastery"] = pct
                        self.study_state["facts_learned"] = int(st.get("generated", done))
                        self.study_state["current_action"] = str(st.get("message", "Generuję przykłady rozumowania..."))
                        for pair in new_pairs:
                            if not isinstance(pair, dict):
                                continue
                            pq = str(pair.get("q", ""))
                            pa = str(pair.get("a", ""))
                            pcat = str(pair.get("category", "reasoning"))
                            if pq and pa and not is_ai_corporate_reply(pa):
                                session_new_records.append({"q": pq, "a": pa, "category": pcat})
                                session_new_pairs.append((pq, pa))
                                gen_seen += 1
                                c_lbl = cat_labels.get(pcat, pcat)
                                self._push_training_dialogue(
                                    mode="teacher_gen",
                                    tag=f"Bielik · {c_lbl} #{gen_seen}/{tot}",
                                    prompt_role="Pytanie",
                                    prompt=pq,
                                    teacher_role="Bielik (Nauczyciel)",
                                    teacher_reply=pa,
                                )

                gen_err = ""
                try:
                    gen_res = run_bielik_reasoning_subprocess(
                        state_dir=self.state_dir,
                        model_id=self.cfg.teacher_model_id,
                        num_samples=num_samples,
                        stop_event=self._study_stop_event,
                        progress_callback=_on_gen_progress,
                    )
                    generated_n = int(gen_res.get("generated", 0))
                    filtered_ai_n = int(gen_res.get("filtered_ai", 0))
                    gen_err = str(gen_res.get("error", "") or "")
                    if gen_err:
                        with self.lock:
                            self._log("ERROR", gen_err)
                finally:
                    if moved_to_cpu:
                        with self.language_cortex._op_lock:
                            self.language_cortex._wernicke_raw.to(self.device)
                            self.language_cortex._broca_raw.to(self.device)
                            self.cortex.to(self.device)
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                    trim_process_ram()

            all_reasoning_records = load_reasoning_records(self.state_dir)
            all_reasoning_pairs = [(r["q"], r["a"]) for r in all_reasoning_records if not is_ai_corporate_reply(r["a"])]
            all_teacher_pairs = load_teacher_dialogues(self.state_dir)
            combined_bielik_pairs = all_teacher_pairs + all_reasoning_pairs

            # Ingest substantive causal chains into Hippocampus BEFORE training so Hopfield engram_bank
            # includes them during Wernicke/Broca training!
            causal_src = session_new_records if session_new_records else all_reasoning_records
            causal_pairs = [(r["q"], r["a"]) for r in causal_src if r.get("category") == "causal"]
            st_ingest = self._ingest_teacher_pairs_to_short_term(
                causal_pairs,
                max_items=max(40, min(250, len(causal_pairs))),
            ) if causal_pairs else {"engrams_added": 0, "living_steps": 0, "living_neurons_born": 0}
            with self.lock:
                engram_bank = self.hippocampus.get_hopfield_bank()

            base_pct = 50.0 if not skip_generation else 0.0
            span_pct = 100.0 - base_pct
            with self.lock:
                self.study_state["round"] = 0
                self.study_state["max_rounds"] = train_epochs
                self.study_state["mastery"] = max(base_pct, 5.0)
                self.study_state["facts_learned"] = len(combined_bielik_pairs)
                self.study_state["current_action"] = (
                    f"Bielik wyładowany z VRAM. Trenuję Wernicke + Broca na {len(all_reasoning_pairs)} przykładach "
                    f"rozumowania + {len(all_teacher_pairs)} dialogach ({train_epochs} epok)..."
                )

            def _on_brain_train_progress(st: dict[str, Any]) -> None:
                ep = int(st.get("epoch", 0))
                tot = max(1, int(st.get("total_epochs", train_epochs)))
                phase = str(st.get("phase", "dialogue_train"))
                loss_s = float(st.get("loss_start", 0.0))
                loss_n = float(st.get("loss_now", 0.0))
                flu = float(st.get("fluency", 0.0))
                if phase == "dialogue_train":
                    pct = round(base_pct + (span_pct * 0.85) * ep / tot, 1)
                else:
                    pct = round(base_pct + (span_pct * 0.85) + (span_pct * 0.10) * ep / tot, 1)
                phase_lbl = "Szlifowanie rozumowania" if phase == "dialogue_polish" else "Trening rozumowania · epoka"
                with self.lock:
                    self.study_state["round"] = ep
                    self.study_state["max_rounds"] = tot
                    self.study_state["mastery"] = min(96.0, pct)
                    self.study_state["current_action"] = (
                        f"{phase_lbl} {ep}/{tot} · Loss: {loss_s:.2f}→{loss_n:.2f} · Płynność: {flu:.0f}%"
                    )
                    if st.get("fresh_probe") and st.get("sample_q"):
                        self._push_training_dialogue(
                            mode="dialogue_train",
                            tag=f"{phase_lbl} {ep}/{tot} · loss {loss_n:.2f}",
                            prompt_role="Pytanie",
                            prompt=str(st["sample_q"]),
                            teacher_role="Bielik (Wzorzec)",
                            teacher_reply=str(st.get("sample_target", "")),
                            brain_role="Twój mózg (na żywo)",
                            brain_reply=str(st.get("sample_brain", "")) or "(jeszcze składa pierwsze słowa...)",
                        )

            probe_source = session_new_pairs if session_new_pairs else all_reasoning_pairs
            train_res = self.language_cortex.train_language_cortex(
                extra_qa_pairs=combined_bielik_pairs,
                epochs=train_epochs,
                include_curriculum=True,
                cortex=self.cortex,
                engram_bank=engram_bank,
                progress_callback=_on_brain_train_progress,
                stop_event=self._study_stop_event,
                priority_probe_pairs=probe_source,
            )
            self.language_cortex.ensure_familiarity_bank(force_rebuild=True)
            self.save_state()

            err_note = f" ⚠️ {gen_err}" if gen_err else ""
            msg_done = (
                f"Trening rozumowania z Bielikiem zakończony! Nowe przykłady: **{generated_n}** "
                f"(łącznie rozumowania: **{len(all_reasoning_pairs)}**, odrzucono formułek AI: **{filtered_ai_n}**). "
                f"Loss: `{train_res['loss_start']} → {train_res['loss_end']}`, płynność: **{train_res['fluency']:.0f}%**.{err_note}"
            )
            with self.lock:
                self.study_state["mastery"] = 100.0
                self.study_state["current_action"] = msg_done
                self._log("STUDY", msg_done)
                self._append_chat("assistant", msg_done, meta={"kind": "study_done"}, add_to_working=False)
            return {
                "generated": generated_n,
                "filtered_ai": filtered_ai_n,
                "total_reasoning_dialogues": len(all_reasoning_pairs),
                "short_term_engrams_added": st_ingest["engrams_added"],
                "living_neurons_born": st_ingest["living_neurons_born"],
                "train_result": train_res,
            }
        finally:
            with self.lock:
                self.study_state["active"] = False
                self.study_state["current_action"] = ""
                if self.state_mode == "STUDYING":
                    self.state_mode = "AWAKE"
            self._release_op("study")

    # ── Tokenless Cognitive Chat & True Real-Time UTF-8 Streaming ────────────
    def _handle_system_command(self, msg: str) -> dict[str, Any] | None:
        low = msg.lower()
        if low in ("/sen", "/sleep", "idź spać", "idz spac"):
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            rep = self.sleep_now(visual_pacing=False)
            summary = (
                f"☀️ Zakończono sen (Noc #{rep.night_number}, {rep.duration_ms:.0f} ms): "
                f"Pred Error {rep.prediction_error_start:.4f}→{rep.prediction_error_end:.4f}, "
                f"SHY przycięto {rep.pruned_synapses} synaps (aktywnych {rep.active_synapses}), "
                f"REM +{len(rep.rem_reflections)}."
            )
            return {
                "reply": summary,
                "slept": True,
                "sleep_report": rep.to_dict(),
                "learned_memories": [],
                "recalled_memories": {},
            }

        if low in ("/szkola", "/szkoła", "/podstawy"):
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            res = self.learn_elementary_foundation(consolidate=True)
            return {
                "reply": f"🎓 Wgrano pakiet Szkoła Podstawowa (+{res['added_facts']} engramów SDR, pamięć trwała: {res['total_semantic']}).",
                "slept": bool(res.get("sleep_report")),
                "sleep_report": res.get("sleep_report"),
                "learned_memories": [],
                "recalled_memories": {},
            }

        if low in ("/wyksztalc", "/wykształć", "/trenuj"):
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            res = self.train_language_now()
            return {
                "reply": f"🗣️ Przetrenowano BrocaByteTransformer (loss {res['loss_start']}→{res['loss_end']}, płynność {res['fluency']:.0f}%).",
                "slept": False,
                "learned_memories": [],
                "recalled_memories": {},
                "language_train": res,
            }

        if low.startswith("/parquet") or low in ("/ucz polski", "/polski"):
            parts = msg.split()
            steps = int(parts[-1]) if len(parts) >= 2 and parts[-1].isdigit() else 5000
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            info = self.start_parquet_polish_async(max_steps=steps)
            return {
                "reply": (
                    f"🇵🇱 Rozpoczęto naukę języka polskiego i gramatyki z `{info['filename']}` "
                    f"({info['max_steps']} kroków na GPU w bfloat16). Obserwuj pasek postępu u góry!"
                ),
                "study_started": True,
                "slept": False,
                "learned_memories": [],
                "recalled_memories": {},
            }

        if low in ("/szkola siec", "/szkoła sieć", "/szkola web"):
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            info = self.start_curriculum_async(deep_web=True, max_topics=12)
            return {
                "reply": f"📚 Rozpoczęto kurs szkolny z Wikipedii ({info['topics']} tematów).",
                "study_started": True,
                "slept": False,
                "learned_memories": [],
                "recalled_memories": {},
            }

        if low.startswith("/ucz turbo"):
            parts = msg.split()
            cnt = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 200
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            info = self.start_turbo_learn_async(topics_count=cnt)
            return {
                "reply": f"⚡ Rozpoczęto Turbo-naukę ({info['topics']} tematów).",
                "study_started": True,
                "slept": False,
                "learned_memories": [],
                "recalled_memories": {},
            }

        if low.startswith("/ucz auto"):
            parts = msg.split()
            mins = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 60
            with self.lock:
                self._append_chat("user", msg, add_to_working=False)
            info = self.start_continuous_learning_async(duration_minutes=mins)
            return {
                "reply": f"🚀 Rozpoczęto auto-naukę ({info['duration_minutes']} min).",
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
                "reply": f"🌐 Rozpoczęto badanie tematu „{study_topic}” (do {info['max_rounds']} rund).",
                "study_started": True,
                "topic": study_topic,
                "slept": False,
                "learned_memories": [],
                "recalled_memories": {},
            }
        return None

    @staticmethod
    def _is_question_or_command(msg: str) -> bool:
        s = msg.strip()
        if not s or s.endswith("?") or s.startswith("/"):
            return True
        low = _strip_pl(s.lower())
        q_prefixes = (
            "co ", "co to", "kto ", "kim ", "czym ", "jak ", "jakie ", "jaki ", "jaka ",
            "kiedy ", "gdzie ", "dlaczego", "czemu", "ile ", "czy ", "powiedz ", "opowiedz ",
            "wyjasnij ", "napisz ", "podaj ", "siema", "czesc", "hej", "elo", "nic tak", "tak tylko",
        )
        return any(low.startswith(p) for p in q_prefixes)

    @staticmethod
    def _is_factual_web_query(msg: str) -> bool:
        """
        Strict encyclopedic-question detector for live Auto-sieć so conversational chatter
        ('nie wiem', 'ty mi powiedz', 'ale o co chodzi', 'wiem to') NEVER triggers web scraping!
        """
        s = msg.strip()
        low = _strip_pl(s.lower()).rstrip("?!., ")
        if not low or low.startswith("/"):
            return False
        chatter_block = (
            "nie wiem", "wiem to", "juz to wiem", "znam to", "o co chodzi", "ty mi powiedz",
            "co tam", "jak tam", "co slychac", "jak sie masz", "co u ciebie", "a co jak",
            "dlaczego ty", "czemu ty", "co robisz", "co myslisz", "powiedz cos", "opowiedz cos",
        )
        if any(b in low for b in chatter_block):
            return False
        factual_prefixes = (
            "co to jest ", "co to sa ", "co to ", "co znaczy ", "co oznacza ",
            "kto to jest ", "kto to byl ", "kto to ", "kim jest ", "kim byl ", "kim byla ",
            "kto wynalazl ", "kto napisal ", "kto stworzyl ",
            "czym jest ", "czym sa ", "definicja ", "znaczenie ",
            "kiedy byl ", "kiedy byla ", "gdzie lezy ", "gdzie jest ",
        )
        if any(low.startswith(p) for p in factual_prefixes) and len(low) >= 8:
            q_clean = clean_search_query(s)
            return 2 <= len(q_clean) <= 48 and 1 <= len(q_clean.split()) <= 4
        return False

    def _prepare_tensor_thought(
        self,
        msg: str,
        temperature: float | None = None,
        for_stream: bool = False,
    ) -> tuple[dict[str, Any], Iterator[str]]:
        """
        Execute purely tensor cognitive pathway with Prefrontal Working Memory SDR & Wernicke Self-Critique:
          UTF-8 bytes -> Wernicke (S, u_sdr) -> Familiarity/OOD check + Auto-sieć (if enabled)
            -> System-2 Multi-Path Cortical Deliberation (Precyzyjna / Kontekstowa / Kreatywna)
            -> Wernicke Auditory Self-Critique -> Broca stream(Z*_best).
        STRICT ISOLATION: `preview_label` is NEVER read anywhere in this method or downstream.
        """
        # Ensure familiarity bank is never empty after training/loading so OOD gating stays active
        self.language_cortex.ensure_familiarity_bank()

        # Online conversational plasticity: if user assigns a name or explicitly teaches a fact,
        # update Broca's neural weights cleanly (with Wernicke frozen!) and PIN core identity/rules!
        conv_note = ""
        conv_pairs = self._extract_conversational_qa_pairs(msg)
        if conv_pairs:
            self.language_cortex.adapt_qa_pairs_online(conv_pairs, cortex=self.cortex, steps=52, pinned=True)
            conv_note = f"Szybka plastyczność konwersacyjna Broki (+{len(conv_pairs)} przypiętych powiązań Q&A bez dryfu Wernickego)"
            self._log("TEACH", conv_note)

        with self.lock:
            engram_bank = self.hippocampus.get_hopfield_bank()
            wm_count = len(self._working_sdr_ring)
            wm_slots = self._get_working_memory_slots()

        temp = temperature if temperature is not None else self._sampling_temperature()
        w_out, q_fam, _ = self.language_cortex.perceive_for_chat(msg)
        is_followup = self._is_followup_query(msg)
        with self.lock:
            wm_sdr = self._get_working_memory_sdr(
                query_sdr=w_out.sdr[0],
                max_slots=6,
                focus_recent=is_followup,
            )

        # Apply PFC Working Memory SDR on contextual follow-ups and novel short replies so the brain
        # genuinely carries conversation context without contaminating standalone factual questions.
        if wm_sdr is not None and q_fam < 0.78 and (is_followup or len(msg) <= 24):
            wm_weight = 0.28 if is_followup else 0.22
        else:
            wm_weight = 0.0

        # Auto-sieć: ONLY for genuine encyclopedic/factual definition queries (NEVER conversational chatter!),
        # and strictly using authoritative Polish Wiktionary + Polish Wikipedia (zero cookie-banner web spam!).
        auto_web_note = ""
        auto_web_adapted = False
        if self.auto_web and self._is_factual_web_query(msg) and q_fam < 0.78 and not conv_pairs:
            try:
                q_clean = clean_search_query(msg)
                docs = gather_web_sources(
                    query=q_clean,
                    seen_urls=set(),
                    max_docs=1,
                    timeout=6,
                    max_workers=2,
                    encyclopedic_only=True,
                )
                if docs:
                    raw_lead = extract_lead_definition(docs[0])
                    lead = _short_answer_from_fact(raw_lead, max_chars=185) if raw_lead else ""
                    if lead and len(lead) >= 24 and not lead.endswith("?") and not is_web_boilerplate(lead):
                        self._ingest_sensory_texts(
                            [lead], category="nauka_z_sieci", importances=[4], topic=q_clean, source_url=docs[0].url
                        )
                        new_loss = self.language_cortex.adapt_qa_online(msg, lead, cortex=self.cortex, steps=60)
                        auto_web_adapted = True
                        with self.lock:
                            engram_bank = self.hippocampus.get_hopfield_bank()
                        w_out, q_fam, _ = self.language_cortex.perceive_for_chat(msg)
                        auto_web_note = (
                            f"Auto-sieć sprawdziła „{docs[0].title}”, zakodowała do SDR i zaktualizowała Z* (loss={new_loss:.2f})"
                        )
                        self._log("WEB", auto_web_note)
            except Exception:
                pass

        # Step Living Brain (M1/M2/M4) BEFORE cortical deliberation so its growing associative state,
        # top-3 M2 traces, neuromodulators (NA/DA/ACh), and M4 curiosity directly steer Z* and Broca!
        lb_tel = None
        lb_neuromod: NeuromodulatorState | None = None
        living_assoc_vec: torch.Tensor | None = None
        lb_note = ""
        try:
            lb_tel = self.living_brain.cognitive_step(
                sensory_vec=w_out.columns[0],
                external_valence=0.35 if conv_pairs else (0.25 if q_fam >= 0.42 else -0.10),
                allow_growth=True,
                spontaneous=False,
                pinned=bool(conv_pairs),
            )
            nm = lb_tel.neuromodulators
            lb_neuromod = NeuromodulatorState(
                noradrenaline=float(nm.get("noradrenaline", self.living_brain.neuromod.noradrenaline)),
                dopamine=float(nm.get("dopamine", self.living_brain.neuromod.dopamine)),
                acetylcholine=float(nm.get("acetylcholine", self.living_brain.neuromod.acetylcholine)),
            )
            living_assoc_vec = self.living_brain.get_cortical_association_vector(top_k_m2=3)
            ng_txt = f" · +Neurogeneza({lb_tel.neurons_added})" if lb_tel.neurons_added > 0 else ""
            exp_txt = " · Eksploracja M4" if lb_tel.boredom_exploration else ""
            pin_txt = " · [M2 pinned]" if conv_pairs else ""
            lb_note = (
                f"Living Brain M1/M2/M4 → Z*: PE={lb_tel.surprise:.3f}, ciekawość={lb_tel.intrinsic_curiosity:.2f}, "
                f"NA={lb_neuromod.noradrenaline:.2f} DA={lb_neuromod.dopamine:.2f} ACh={lb_neuromod.acetylcholine:.2f}"
                f"{ng_txt}{exp_txt}{pin_txt}"
            )
        except Exception:
            pass

        # Let the brain ALWAYS think continuously via System-2 3-path cortical deliberation + Working Memory SDR
        # instead of blocking novel Polish sentences behind a rigid OOD wall!
        # Only pure gibberish (non-Polish / q_fam < 0.42 or failed neural self-critique NLL > 1.45) falls back to uncertainty.
        ood_note = ""
        critique_notes: list[str] = []
        used_temp = temp
        has_fam_bank = self.language_cortex._known_q_sdrs.shape[0] > 0
        alpha_ratio = sum(1 for ch in msg if ch.isalpha()) / float(max(1, len(msg)))
        looks_gibberish = (alpha_ratio < 0.45 and len(msg) >= 6) or (
            has_fam_bank and q_fam < 0.42 and not conv_pairs and wm_weight == 0.0
        )
        if looks_gibberish:
            _, thought, stream_iter = self.language_cortex.think_and_stream(
                user_message="__nieznane_pytanie__",
                cortex=self.cortex,
                engram_bank=None,
                temperature=min(temp, 0.55),
                max_new_bytes=self.cfg.broca_max_new_bytes,
            )
            ood_note = f"Nierozpoznany bodziec (q_fam={q_fam:.2f}) → naturalna odpowiedź o braku wiedzy"
        else:
            # System-2 batched parallel deliberation + Auditory Self-Hearing & Active Self-Correction
            # Only inject PFC working-memory turn slots when wm_weight > 0 (follow-up queries),
            # so standalone greetings/questions match their clean training Z_ctx distribution!
            rehearsal_bytes = min(240, self.cfg.broca_max_new_bytes)
            eff_living_blend = 0.02 if q_fam >= 0.78 else 0.08
            delib = self.language_cortex.deliberate_and_critique(
                user_message=msg,
                cortex=self.cortex,
                engram_bank=engram_bank,
                working_memory_sdr=wm_sdr,
                working_memory_weight=wm_weight,
                creativity_level=self.creativity_level,
                base_temperature=temp,
                max_new_bytes=rehearsal_bytes,
                num_candidates=self.cfg.self_critique_candidates,
                living_assoc_vec=living_assoc_vec,
                living_blend=eff_living_blend,
                neuromod=lb_neuromod,
                living_telemetry=lb_tel,
                working_memory_slots=wm_slots if wm_weight > 0.0 else None,
            )
            w_out = delib["w_out"]
            thought = delib["thought"]
            used_temp = float(delib["temperature"])
            critique_notes = list(delib["critique_notes"])

            # Neural self-critique safeguard: ONLY reject if Broca itself cannot form a coherent Polish sentence
            if has_fam_bank and (delib["avg_nll"] > 1.45 or delib["score"] < 0.50) and q_fam < 0.78:
                _, thought, stream_iter = self.language_cortex.think_and_stream(
                    user_message="__nieznane_pytanie__",
                    cortex=self.cortex,
                    engram_bank=None,
                    temperature=min(temp, 0.55),
                    max_new_bytes=self.cfg.broca_max_new_bytes,
                )
                ood_note = f"Samokrytyka odrzuciła niespójny szkic (NLL={delib['avg_nll']:.2f}) → stan niepewności"
            else:
                # Stream the exact self-heard & self-corrected reply (zero wasted 4th unverified pass!)
                draft = delib["draft"]
                if draft:
                    if for_stream:
                        def _chunked_verified(txt: str) -> Iterator[str]:
                            for part in re.findall(r"\S+\s*|\s+", txt):
                                yield part
                        stream_iter = _chunked_verified(draft)
                    else:
                        stream_iter = iter([draft])
                else:
                    ctx_cols = thought.context_columns if thought.context_columns is not None else thought.columns_star
                    stream_iter = self.language_cortex.broca_module.generate_stream(
                        memory_columns=ctx_cols,
                        max_new_bytes=self.cfg.broca_max_new_bytes,
                        temperature=used_temp,
                        top_p=self.cfg.broca_top_p,
                        repetition_penalty=self.cfg.broca_repetition_penalty,
                        memory_mask=thought.context_mask,
                    )

        with self.lock:
            recalled = self.hippocampus.recall_all_banks(
                query_sdr=thought.z_star[0],
                current_day=self.day,
                top_hippo=self.cfg.recall_top_hippo,
                top_semantic=self.cfg.recall_top_semantic,
                top_rem=self.cfg.recall_top_rem,
                min_sim=0.35,
            )

            # Encode substantive declarative statements as fresh SDR engrams in Hipokamp
            # (pin core identity/rule statements from _extract_conversational_qa_pairs!)
            learned_engrams: list[dict[str, Any]] = []
            is_pinned_stmt = bool(conv_pairs)
            if is_pinned_stmt or (len(msg) >= 14 and not self._is_question_or_command(msg)):
                engram_info = self.hippocampus.record_engram(
                    sdr=w_out.sdr[0],
                    preview_label=msg[:180],
                    category="zasada" if is_pinned_stmt else "użytkownik",
                    importance=5 if is_pinned_stmt else 3,
                    day=self.day,
                    strength=2.5 if is_pinned_stmt else 1.2,
                    sparsity_pct=w_out.sparsity_pct,
                    pinned=is_pinned_stmt,
                )
                learned_engrams.append(
                    {
                        "id": engram_info["id"],
                        "sparsity_pct": engram_info["sparsity_pct"],
                        "pinned": bool(engram_info.get("pinned", False)),
                    }
                )
                self._buffer_wake_stimulus([msg])

            hippo_refs = [f"engram:{eid} ({sc:.2f})" for eid, sc in zip(recalled["hippocampus"]["ids"][:6], recalled["hippocampus"]["scores"][:6])]
            sem_refs = [f"engram:{eid} ({sc:.2f})" for eid, sc in zip(recalled["semantic"]["ids"][:6], recalled["semantic"]["scores"][:6])]
            rem_refs = [f"rem:{eid} ({sc:.2f})" for eid, sc in zip(recalled["rem"]["ids"][:4], recalled["rem"]["scores"][:4])]
            thinking_telemetry = [
                f"Wernicke UTF-8 → patche RoPE + 16 kolumn S → k-WTA SDR ({w_out.sparsity_pct:.2f}% aktywnych)",
                f"Pamięć robocza SDR: {wm_count}/{self.cfg.working_sdr_capacity} · Sloty PFC: {self._wm_slot_count}/{self._wm_slot_buf.shape[0]} · Kreatywność={self.creativity_level*100:.0f}%",
                f"Modern Hopfield + W_rec ({thought.attractor_steps} kroków, q_fam={q_fam:.2f}, max_sim={thought.max_hopfield_sim:.2f}) → Z_ctx",
            ]
            if lb_note:
                thinking_telemetry.append(lb_note)
            thinking_telemetry.extend(critique_notes)
            thinking_telemetry.append(
                f"BrocaByteTransformer: próbkowanie UTF-8 (T={used_temp:.2f}, top_p={self.cfg.broca_top_p}, rep={self.cfg.broca_repetition_penalty})"
            )
            if conv_note:
                thinking_telemetry.insert(0, conv_note)
            if auto_web_note:
                thinking_telemetry.insert(0, auto_web_note)
            if ood_note:
                thinking_telemetry.insert(0, ood_note)

        prep = {
            "learned_memories": learned_engrams,
            "learned_triples": [],
            "hippo_refs": hippo_refs,
            "sem_refs": sem_refs,
            "rem_refs": rem_refs,
            "thinking": thinking_telemetry,
            "sdr_sparsity_pct": thought.sparsity_pct,
            "hopfield_energy": thought.hopfield_energy,
            "attractor_steps": thought.attractor_steps,
            "temperature": used_temp,
            "user_sdr": w_out.sdr[0].detach(),
            "user_patch_columns": w_out.patch_columns[0].detach() if w_out.patch_columns is not None else w_out.columns[0].detach(),
            "user_patch_mask": w_out.patch_mask[0].detach() if w_out.patch_mask is not None else None,
            "thought": thought,
            "living_da": float(lb_neuromod.dopamine) if lb_neuromod is not None else 0.0,
            "is_ood": bool(ood_note),
            "already_adapted_online": bool(conv_pairs) or auto_web_adapted,
        }
        return prep, stream_iter

    def _prepare_hybrid_llm_thought(
        self,
        msg: str,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], Iterator[str]]:
        """
        Hybrid Biological + Local Polish LLM Broca pathway (Option B):
          1. Wernicke UTF-8 -> 16 columns S -> k-WTA SDR (4%)
          2. BiologicalAssociativeCortex (Modern Continuous Hopfield + Prefrontal Working Memory SDR)
             updates cortex.last_active_sdr for the live 2D synaptic graph
          3. Hippocampus SDR cosine recall (semantic + short-term + REM associations)
          4. Optional authoritative Auto-sieć (Polish Wiktionary / Wikipedia ONLY on explicit factual queries)
          5. Genuine token-by-token streaming from LocalPolishBrocaLLM (Qwen2.5-3B-Instruct / Bielik + LoRA)
             with full multi-turn working conversation context and ZERO hardcoded canned responses!
        """
        assert self.broca_llm is not None
        if not self.broca_llm.ready:
            if not self.broca_llm.load_sync():
                raise RuntimeError(self.broca_llm.load_error or "Nie udało się załadować lokalnego modelu Broki.")

        with self.lock:
            engram_bank = self.hippocampus.get_hopfield_bank()
            wm_count = len(self._working_sdr_ring)
            history_snapshot = list(self.working_chat[-64:])

        w_out = self.language_cortex.perceive_one(msg)
        with self.lock:
            wm_sdr = self._get_working_memory_sdr(
                query_sdr=w_out.sdr[0],
                max_slots=6,
                focus_recent=self._is_followup_query(msg),
            )
        wm_weight = 0.22 if wm_sdr is not None else 0.0
        thought = self.cortex.perceive_and_think(
            sensory_columns=w_out.columns,
            sensory_sdr=w_out.sdr,
            engram_bank=engram_bank,
            sdr_to_columns_fn=self.language_cortex.wernicke_module.sdr_to_columns,
            steps=self.cfg.attractor_steps,
            noise_std=0.01 * self.creativity_level,
            update_last_active=True,
            working_memory_sdr=wm_sdr,
            working_memory_weight=wm_weight,
        )

        with self.lock:
            recalled = self.hippocampus.recall_all_banks(
                query_sdr=thought.z_star[0],
                current_day=self.day,
                top_hippo=self.cfg.recall_top_hippo,
                top_semantic=self.cfg.recall_top_semantic,
                top_rem=self.cfg.recall_top_rem,
                min_sim=0.38,
            )

        # Optional Auto-sieć ONLY for explicit factual/encyclopedic questions when Hippocampus match is weak
        top_sem_sim = max(recalled["semantic"]["scores"], default=0.0)
        top_hip_sim = max(recalled["hippocampus"]["scores"], default=0.0)
        best_mem_sim = max(top_sem_sim, top_hip_sim)
        web_context = ""
        auto_web_note = ""
        if self.auto_web and self._is_factual_web_query(msg) and best_mem_sim < 0.72:
            try:
                q_clean = clean_search_query(msg)
                docs = gather_web_sources(
                    query=q_clean,
                    seen_urls=set(),
                    max_docs=1,
                    timeout=6,
                    max_workers=2,
                    encyclopedic_only=True,
                )
                if docs:
                    raw_lead = extract_lead_definition(docs[0])
                    lead = _short_answer_from_fact(raw_lead, max_chars=260) if raw_lead else ""
                    if lead and len(lead) >= 24 and not lead.endswith("?") and not is_web_boilerplate(lead):
                        web_context = lead
                        self._ingest_sensory_texts(
                            [lead], category="nauka_z_sieci", importances=[4], topic=q_clean, source_url=docs[0].url
                        )
                        auto_web_note = f"Auto-sieć sprawdziła „{docs[0].title}” i zapisała ślad SDR w Hipokampie"
                        self._log("WEB", auto_web_note)
            except Exception:
                pass

        with self.lock:
            # Resolve biological SDR memory cues for Broca LLM synthesis (never pasted 1:1!)
            sem_map = {m.id: m.preview_label for m in self.hippocampus.semantic.meta if m.preview_label}
            hip_map = {m.id: m.preview_label for m in self.hippocampus.short_term.meta if m.preview_label}
            rem_map = {m.id: m.preview_label for m in self.hippocampus.rem.meta if m.preview_label}

            sem_texts = [sem_map[eid] for eid in recalled["semantic"]["ids"][:12] if eid in sem_map]
            hip_texts = [hip_map[eid] for eid in recalled["hippocampus"]["ids"][:10] if eid in hip_map]
            rem_texts = [rem_map[eid] for eid in recalled["rem"]["ids"][:6] if eid in rem_map]

            # Record substantive declarative statements (e.g. names, preferences, taught facts) into Hipokamp
            learned_engrams: list[dict[str, Any]] = []
            if len(msg) >= 12 and not self._is_question_or_command(msg):
                engram_info = self.hippocampus.record_engram(
                    sdr=w_out.sdr[0],
                    preview_label=msg[:320],
                    category="użytkownik",
                    importance=4,
                    day=self.day,
                    strength=1.35,
                    sparsity_pct=w_out.sparsity_pct,
                    pinned=True,
                )
                learned_engrams.append({"id": engram_info["id"], "sparsity_pct": engram_info["sparsity_pct"]})
                self._buffer_wake_stimulus([msg])

            hippo_refs = [f"engram:{eid} ({sc:.2f})" for eid, sc in zip(recalled["hippocampus"]["ids"][:6], recalled["hippocampus"]["scores"][:6])]
            sem_refs = [f"engram:{eid} ({sc:.2f})" for eid, sc in zip(recalled["semantic"]["ids"][:6], recalled["semantic"]["scores"][:6])]
            rem_refs = [f"rem:{eid} ({sc:.2f})" for eid, sc in zip(recalled["rem"]["ids"][:4], recalled["rem"]["scores"][:4])]

            sys_prompt = self.broca_llm.build_system_prompt(
                day=self.day,
                total_sleeps=self.total_sleeps,
                creativity_level=self.creativity_level,
                recalled_semantic=sem_texts,
                recalled_hippo=hip_texts,
                recalled_rem=rem_texts,
                web_context=web_context,
                sdr_sparsity_pct=thought.sparsity_pct,
            )

            short_llm = self.broca_llm.model_id.split("/")[-1]
            eff_temp = temperature if temperature is not None else round(0.35 + 0.55 * self.creativity_level, 2)
            thinking_telemetry = [
                f"Wernicke UTF-8 → 16 kolumn S [16×{self.cfg.column_dim}] → k-WTA SDR ({w_out.sparsity_pct:.2f}% aktywnych)",
                f"Pamięć robocza SDR: {wm_count}/{self.cfg.working_sdr_capacity} · Kontekst dialogu: {len(history_snapshot)} tur · Kreatywność={self.creativity_level*100:.0f}%",
                f"Modern Hopfield + Hipokamp ({thought.attractor_steps} kroków, pamięć trwała: {len(sem_texts)}, hipokamp: {len(hip_texts)}, REM: {len(rem_texts)})",
                f"Broca LLM ({short_llm} + LoRA, bf16): generowanie na żywo (T={eff_temp:.2f})",
            ]
            if auto_web_note:
                thinking_telemetry.insert(0, auto_web_note)

        stream_iter = self.broca_llm.generate_stream(
            user_message=msg,
            history=history_snapshot,
            system_prompt=sys_prompt,
            creativity_level=self.creativity_level,
            temperature=temperature,
            max_new_tokens=self.cfg.broca_llm_max_new_tokens,
        )

        prep = {
            "learned_memories": learned_engrams,
            "learned_triples": [],
            "hippo_refs": hippo_refs,
            "sem_refs": sem_refs,
            "rem_refs": rem_refs,
            "thinking": thinking_telemetry,
            "sdr_sparsity_pct": thought.sparsity_pct,
            "hopfield_energy": thought.hopfield_energy,
            "attractor_steps": thought.attractor_steps,
            "temperature": eff_temp,
            "user_sdr": w_out.sdr[0].detach(),
            "thought": thought,
            "is_llm_broca": True,
        }
        return prep, stream_iter

    def _finalize_chat_turn(self, msg: str, reply: str, prep: dict[str, Any]) -> dict[str, Any]:
        # Auditory feedback loop: brain reads its own reply through Wernicke, evaluates it,
        # triggers lightweight Dopamine-gated online reinforcement on clean high-score turns,
        # and stores BOTH user + assistant SDRs into the Prefrontal Working Memory SDR Ring.
        reply_sdr: torch.Tensor | None = None
        reply_patches: torch.Tensor | None = None
        reply_patch_mask: torch.Tensor | None = None
        final_critique_note = ""
        da_reinforce_note = ""
        if reply and len(reply.strip()) >= 3:
            try:
                reply_out = self.language_cortex.perceive_one(reply)
                reply_sdr = reply_out.sdr[0].detach()
                reply_patches = (
                    reply_out.patch_columns[0].detach()
                    if reply_out.patch_columns is not None
                    else reply_out.columns[0].detach()
                )
                reply_patch_mask = reply_out.patch_mask[0].detach() if reply_out.patch_mask is not None else None
                if prep.get("is_llm_broca"):
                    qual = self.language_cortex.score_polish_quality(reply, creativity_level=self.creativity_level)
                    final_critique_note = f"Pętla zwrotna Wernickego (SDR odpowiedzi zapisany): jakość={qual*100:.0f}%"
                else:
                    audit = self.language_cortex.analyze_self_heard_draft(
                        reply,
                        prep["thought"],
                        creativity_level=self.creativity_level,
                        heard_sdr=reply_sdr,
                    )
                    final_critique_note = (
                        f"Samokontrola końcowa (Wernicke widzi odpowiedź): ocena={audit['total']*100:.0f}% "
                        f"(jakość={audit['quality']*100:.0f}%, spójność={audit['alignment']:.2f})"
                    )
                    # NOTE: Do NOT call adapt_qa_online on unverified self-generated replies!
                    # Blind online reinforcement on Broca's own guesses can permanently burn off-topic
                    # hallucinations into weights & _learned_qa_replay. Online adaptation is reserved
                    # for explicitly user-taught facts and verified Auto-sieć definitions.
                    lb_da = float(prep.get("living_da", 0.0))
                    if lb_da >= 0.85 and not audit["needs_correction"] and float(audit["total"]) >= 0.78:
                        da_reinforce_note = (
                            f"Dopamina (DA={lb_da:.2f}) + czysta samokrytyka → odpowiedź zatwierdzona (bez nadpisywania wag Broki)"
                        )
            except Exception:
                pass

        with self.lock:
            if "user_sdr" in prep and prep["user_sdr"] is not None:
                self._push_working_sdr(prep["user_sdr"])
            if prep.get("user_patch_columns") is not None:
                self._push_working_memory_slot(prep["user_patch_columns"], prep.get("user_patch_mask"))
            if reply_sdr is not None:
                self._push_working_sdr(reply_sdr)
            elif prep.get("thought") is not None:
                self._push_working_sdr(prep["thought"].z_star[0].detach())
            if reply_patches is not None:
                self._push_working_memory_slot(reply_patches, reply_patch_mask)
            elif prep.get("thought") is not None:
                self._push_working_memory_slot(prep["thought"].columns_star[0].detach())
            thinking_list = list(prep["thinking"])
            if final_critique_note:
                thinking_list.append(
                    f"{final_critique_note} · Pamięć robocza SDR: {len(self._working_sdr_ring)}/{self.cfg.working_sdr_capacity}"
                )
            if da_reinforce_note:
                thinking_list.append(da_reinforce_note)

            self._append_chat("user", msg, meta={"learned_count": len(prep["learned_memories"])})
            self._append_chat(
                "assistant",
                reply,
                meta={
                    "recalled_semantic": prep["sem_refs"],
                    "recalled_hippocampus": prep["hippo_refs"],
                    "recalled_rem": prep["rem_refs"],
                    "thinking": thinking_list,
                    "sdr_sparsity_pct": prep["sdr_sparsity_pct"],
                    "hopfield_energy": prep["hopfield_energy"],
                },
            )
            self.energy = self.cfg.max_energy
            self.state_mode = "AWAKE"
            pending_count = len(self.hippocampus.short_term)
            should_auto_sleep = self.auto_sleep and pending_count >= self.cfg.auto_sleep_pending_threshold

        slept = False
        sleep_report_dict = None
        if should_auto_sleep:
            rep = self.sleep_now(visual_pacing=False, _from_nested=True)
            slept = True
            sleep_report_dict = rep.to_dict()
        else:
            self.save_state_async()

        return {
            "reply": reply,
            "slept": slept,
            "sleep_report": sleep_report_dict,
            "learned_memories": prep["learned_memories"],
            "learned_triples": [],
            "recalled_memories": {
                "hippocampus": prep["hippo_refs"],
                "semantic": prep["sem_refs"],
                "rem": prep["rem_refs"],
            },
            "concept_hints": [],
            "thinking": thinking_list,
            "sdr_sparsity_pct": prep["sdr_sparsity_pct"],
            "hopfield_energy": prep["hopfield_energy"],
            "attractor_steps": prep["attractor_steps"],
            "working_sdr_count": len(self._working_sdr_ring),
            "energy": round(self.energy, 1),
            "state_mode": self.state_mode,
        }

    def chat(self, user_message: str, temperature: float | None = None) -> dict[str, Any]:
        self._last_user_activity_ts = time.monotonic()
        self._user_interrupt_event.set()
        try:
            msg = user_message.strip()
            if not msg:
                raise ValueError("Wiadomość nie może być pusta.")
            cmd_res = self._handle_system_command(msg)
            if cmd_res is not None:
                return cmd_res

            math_ans = try_prefrontal_math(msg)
            if math_ans is not None:
                with self.lock:
                    self._append_chat("user", msg)
                    self._append_chat(
                        "assistant",
                        math_ans,
                        meta={"thinking": ["Moduł arytmetyki symbolicznej (kora przedczołowa AST)"]},
                    )
                self.save_state()
                return {
                    "reply": math_ans,
                    "slept": False,
                    "sleep_report": None,
                    "learned_memories": [],
                    "learned_triples": [],
                    "recalled_memories": {"hippocampus": [], "semantic": [], "rem": []},
                    "concept_hints": [],
                    "energy": round(self.energy, 1),
                    "state_mode": self.state_mode,
                }

            self._acquire_op("chat")
            try:
                if self.broca_llm is not None:
                    prep, stream_iter = self._prepare_hybrid_llm_thought(msg, temperature=temperature)
                else:
                    prep, stream_iter = self._prepare_tensor_thought(msg, temperature=temperature)
                reply = "".join(stream_iter).strip()
                return self._finalize_chat_turn(msg, reply, prep)
            finally:
                self._release_op("chat")
        finally:
            self._last_user_activity_ts = time.monotonic()
            self._user_interrupt_event.clear()

    def chat_stream(self, user_message: str, temperature: float | None = None) -> Iterator[dict[str, Any]]:
        """
        True real-time UTF-8 streaming directly from Broca (zero time.sleep!).
        """
        self._last_user_activity_ts = time.monotonic()
        self._user_interrupt_event.set()
        try:
            msg = user_message.strip()
            if not msg:
                raise ValueError("Wiadomość nie może być pusta.")
            cmd_res = self._handle_system_command(msg)
            if cmd_res is not None:
                yield {"type": "done", **cmd_res}
                return

            math_ans = try_prefrontal_math(msg)
            if math_ans is not None:
                res = self.chat(msg)
                yield {
                    "type": "meta",
                    "learned_memories": [],
                    "learned_triples": [],
                    "recalled_memories": {"hippocampus": [], "semantic": [], "rem": []},
                    "concept_hints": [],
                    "thinking": ["Moduł arytmetyki symbolicznej (kora przedczołowa AST)"],
                }
                yield {"type": "token", "delta": math_ans}
                yield {"type": "done", **res}
                return

            self._acquire_op("chat")
            try:
                if self.broca_llm is not None:
                    prep, stream_iter = self._prepare_hybrid_llm_thought(msg, temperature=temperature)
                else:
                    prep, stream_iter = self._prepare_tensor_thought(msg, temperature=temperature, for_stream=True)
                yield {
                    "type": "meta",
                    "learned_memories": prep["learned_memories"],
                    "learned_triples": [],
                    "recalled_memories": {
                        "hippocampus": prep["hippo_refs"],
                        "semantic": prep["sem_refs"],
                        "rem": prep["rem_refs"],
                    },
                    "concept_hints": [],
                    "thinking": prep["thinking"],
                    "sdr_sparsity_pct": prep["sdr_sparsity_pct"],
                    "hopfield_energy": prep["hopfield_energy"],
                }

                chunks: list[str] = []
                for piece in stream_iter:
                    chunks.append(piece)
                    yield {"type": "token", "delta": piece}

                reply = "".join(chunks).strip()
                final_res = self._finalize_chat_turn(msg, reply, prep)
                yield {"type": "done", **final_res}
            finally:
                self._release_op("chat")
        finally:
            self._last_user_activity_ts = time.monotonic()
            self._user_interrupt_event.clear()

    # ── Vectorized GPU Sleep ─────────────────────────────────────────────────
    def sleep_now(self, visual_pacing: bool = False, _from_nested: bool = False) -> SleepReport:
        if not _from_nested:
            self._acquire_op("sleep")
        try:
            with self.lock:
                night_num = self.total_sleeps + 1
                wake_texts = list(self._wake_broca_buffer)

            def on_stage(stage: str) -> None:
                labels = {
                    "NREM_LIGHT": "Sen NREM (płytki): scalanie prototypów SDR i krzywa zapominania na GPU...",
                    "NREM_DEEP": "Sen NREM (głęboki): lokalny Predictive Coding + Hebb + SHY + Broca na GPU...",
                    "REM_DREAMING": "Sen REM: wektorowe mieszanie odległych engramów i atraktor Hopfielda na GPU...",
                }
                with self.lock:
                    self.state_mode = stage
                    self._log("SLEEP", labels.get(stage, stage))

            report = self.sleep_engine.run_sleep_cycle(
                night_number=night_num,
                stage_callback=on_stage,
                visual_pacing=visual_pacing,
                state_lock=self.lock,
                wake_texts_for_broca=wake_texts,
            )

            # Real GPU LoRA consolidation during NREM Deep Sleep when local Polish LLM Broca is active
            if self.broca_llm is not None and self.broca_llm.ready:
                with self.lock:
                    recent_dialogues: list[tuple[str, str]] = []
                    wc = self.working_chat[-48:]
                    for i in range(len(wc) - 1):
                        if wc[i].get("role") == "user" and wc[i + 1].get("role") == "assistant":
                            recent_dialogues.append((wc[i]["content"], wc[i + 1]["content"]))
                    sleep_facts = list(wake_texts[-40:]) + [
                        m.preview_label for m in self.hippocampus.semantic.meta[-24:] if m.preview_label
                    ]
                lora_res = self.broca_llm.train_lora_sleep(
                    facts=sleep_facts,
                    dialogues=recent_dialogues,
                )
                if lora_res.get("trained"):
                    report.lora_trained = True
                    report.lora_samples = int(lora_res["samples"])
                    report.broca_loss_start = float(lora_res["loss_start"])
                    report.broca_loss_end = float(lora_res["loss_end"])
                    report.lora_duration_ms = float(lora_res["duration_ms"])

            # Bidirectional NREM Memory Unification: Hippocampus <-> LivingBrainEngine (M2) <-> Cortex
            hippo_to_m2 = 0
            m2_to_cortex = 0
            try:
                with self.lock:
                    sem_n = len(self.hippocampus.semantic)
                    if sem_n > 0:
                        k_sync = min(16, sem_n)
                        prio = (
                            self.hippocampus.semantic.importances * 2.0
                            + self.hippocampus.semantic.strengths
                            + self.hippocampus.semantic.pinned.float() * 4.0
                        )
                        top_idx = torch.topk(prio, k=k_sync, largest=True, sorted=False).indices
                        sync_sdrs = self.hippocampus.semantic.engrams.dequant_rows(top_idx, out_dtype=torch.float32)
                        sync_imps = self.hippocampus.semantic.importances.index_select(0, top_idx).cpu().tolist()
                        sync_pins = self.hippocampus.semantic.pinned.index_select(0, top_idx).cpu().tolist()
                        sync_cols = self.language_cortex.wernicke_module.sdr_to_columns(sync_sdrs).detach()
                    else:
                        sync_cols = None
                        sync_imps = []
                        sync_pins = []
                if sync_cols is not None and sync_cols.shape[0] > 0:
                    hippo_to_m2 = self.living_brain.encode_external_engrams(
                        sensory_vecs=sync_cols,
                        importances=sync_imps,
                        pinned_flags=sync_pins,
                    )

                m2_vecs, _ = self.living_brain.export_top_m2_assoc_vectors(top_k=6)
                if m2_vecs.shape[0] > 0:
                    from .sensory_wernicke import k_wta

                    with self.lock:
                        m2_cols_list = [
                            self.cortex.project_living_to_columns(m2_vecs[i], batch_size=1)
                            for i in range(int(m2_vecs.shape[0]))
                        ]
                        m2_cols = torch.cat(m2_cols_list, dim=0)  # [K, 16, D]
                        s_flat = F.normalize(
                            m2_cols.to(device=self.device, dtype=torch.float32).reshape(m2_cols.shape[0], -1),
                            dim=-1,
                            eps=1e-6,
                        )
                        m2_sdrs, _, _ = k_wta(
                            (s_flat.to(dtype=self.cortex.W_in.dtype) @ self.cortex.W_in.T).float(),
                            k=self.cfg.sdr_k,
                            normalize=True,
                        )
                        self.cortex.learn_local(m2_cols, m2_sdrs, steps=3, noise_std=0.01)
                        m2_to_cortex = int(m2_vecs.shape[0])
            except Exception:
                pass

            # REM Phase 2-hop Concept Graph synthesis: connect today's consolidated facts with earlier knowledge
            rem_2hop_steps = max(0, int(getattr(self.cfg, "rem_2hop_steps", 5)))
            rem_2hop_accepted = 0
            if rem_2hop_steps > 0 and self.language_cortex.ready:
                for _ in range(rem_2hop_steps):
                    with self.lock:
                        if len(self.hippocampus.semantic) < 3:
                            break
                        bridge = self.hippocampus.sample_2hop_bridge(
                            min_hop_sim=float(getattr(self.cfg, "dmn_bridge_min_hop_sim", 0.45)),
                            max_direct_sim=float(getattr(self.cfg, "dmn_bridge_max_direct_sim", 0.25)),
                            current_day=self.day,
                        )
                    if bridge is None:
                        break
                    syn = self.synthesize_2hop_thought(
                        engram_a=bridge["a"],
                        engram_b=bridge["b"],
                        engram_c=bridge["c"],
                        tag="rem_2hop",
                        importance=3.0,
                        bridge_meta=bridge,
                        non_blocking=False,
                    )
                    if syn.get("accepted"):
                        rem_2hop_accepted += 1
                        report.rem_reflections.append(syn["thought_text"])
                        report.rem_new_links.append(
                            {
                                "src_engram": bridge["a"]["id"],
                                "bridge_engram": bridge["b"]["id"],
                                "dst_engram": bridge["c"]["id"],
                                "rem_id": syn.get("engram_id"),
                                "sim_ab": bridge["sim_ab"],
                                "sim_bc": bridge["sim_bc"],
                                "sim_ac": bridge["sim_ac"],
                                "source": "REM_2HOP",
                            }
                        )

            # Offline M2 -> M1 Replay + SHY Homeostasis + Fatigue Reset in Living Brain
            lb_sleep = self.living_brain.sleep_consolidate_replay(replay_steps=16)
            if hippo_to_m2 > 0 or m2_to_cortex > 0 or rem_2hop_accepted > 0:
                with self.lock:
                    self._log(
                        "SLEEP",
                        f"🔗 Unifikacja NREM Hipokamp ↔ LivingBrain M2: +{hippo_to_m2} engramów do M2, "
                        f"{m2_to_cortex} śladów M2 w korze, REM 2-hop: +{rem_2hop_accepted} nowych myśli.",
                    )

            with self.lock:
                self._wake_broca_buffer.clear()
                self.total_sleeps += 1
                self.day += 1
                self.energy = self.cfg.max_energy
                self.state_mode = "AWAKE"
                self.working_chat = self.working_chat[-32:]
                self._working_sdr_ring = self._working_sdr_ring[-16:]
                report.pruned_synapses += int(lb_sleep.get("pruned_synapses", 0))

                self.sleep_history.append(report.to_dict())
                if len(self.sleep_history) > 40:
                    self.sleep_history = self.sleep_history[-40:]

                self.language_cortex.core_knowledge_summary = (
                    f"Noc #{night_num} · Engramów trwałych: {len(self.hippocampus.semantic)} · "
                    f"REM: {len(self.hippocampus.rem)} · Aktywnych synaps W_rec: {report.active_synapses} "
                    f"(przycięto SHY: {report.pruned_synapses}) · Pred Error: "
                    f"{report.prediction_error_start:.4f}→{report.prediction_error_end:.4f}"
                )

                self._log(
                    "WAKE",
                    f"Budzenie (Dzień {self.day}, {report.duration_ms:.0f} ms na {report.device_name}): "
                    f"+{report.new_memories} SDR, PredErr {report.prediction_error_start:.4f}→{report.prediction_error_end:.4f}, "
                    f"SHY przycięto {report.pruned_synapses} (aktywnych {report.active_synapses}), REM +{len(report.rem_reflections)}.",
                )
            self.save_state()
            return report
        finally:
            if not _from_nested:
                self._release_op("sleep")

    # ── Manual Engram CRUD ───────────────────────────────────────────────────
    def add_manual_memory(
        self,
        text: str,
        category: str = "wiedza",
        importance: int = 4,
        pinned: bool | None = None,
    ) -> dict[str, Any]:
        clean = text.strip()
        if len(clean) < 3:
            raise ValueError("Treść bodźca musi mieć co najmniej 3 znaki.")
        cat = (category or "wiedza").strip().lower()
        imp = int(max(1, min(5, int(importance))))
        is_pinned = bool(pinned) if pinned is not None else (
            cat in {"zasada", "tożsamość", "tozsamosc", "fundament"} or imp >= 4
        )
        w_out = self.language_cortex.perceive_one(clean)
        with self.lock:
            info = self.hippocampus.record_engram(
                sdr=w_out.sdr[0],
                preview_label=clean,
                category=cat,
                importance=imp,
                day=self.day,
                strength=2.5 if is_pinned else (1.2 + 0.1 * imp),
                sparsity_pct=w_out.sparsity_pct,
                pinned=is_pinned,
            )
            self._buffer_wake_stimulus([clean])
            self._log("TEACH", f"Zakodowano bodziec SDR [{cat}{', pinned' if is_pinned else ''}] (id={info['id']}): {clean[:72]}")
        self.save_state()
        return info

    def delete_memory(self, memory_id: str) -> bool:
        with self.lock:
            deleted = self.hippocampus.delete_memory(memory_id)
            if deleted:
                self._log("TEACH", f"Usunięto engram (id={memory_id}).")
        if deleted:
            self.save_state()
        return deleted

    def delete_studied_topic(self, topic: str) -> bool:
        with self.lock:
            deleted = self.hippocampus.delete_topic(topic)
            if deleted:
                self._log("STUDY", f"Usunięto temat: '{topic}'.")
        if deleted:
            self.save_state()
        return deleted

    def export_knowledge(self) -> dict[str, Any]:
        with self.lock:
            return {
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_version": CHECKPOINT_VERSION,
                "day": self.day,
                "total_sleeps": self.total_sleeps,
                "model_id": self.cfg.language_model_id,
                "architecture": self.cfg.architecture_label,
                "core_knowledge_summary": self.language_cortex.core_knowledge_summary,
                "studied_topics": list(self.hippocampus.studied_topics.values()),
                "semantic_memories": self.hippocampus.semantic.to_public_list(limit=5000, sort_by_strength=True),
                "short_term_memories": self.hippocampus.short_term.to_public_list(limit=5000),
                "rem_reflections": self.hippocampus.rem.to_public_list(limit=500),
            }

    # ── Sparse SDR Persistence & Automatic Legacy Checkpoint Migration ───────
    def _backup_legacy_files(self, paths: list[Path], reason: str) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        moved: list[str] = []
        for p in paths:
            if p.exists():
                bak = p.with_name(f"{p.name}.legacy_{ts}.bak")
                idx = 1
                while bak.exists():
                    bak = p.with_name(f"{p.name}.legacy_{ts}_{idx}.bak")
                    idx += 1
                os.replace(p, bak)
                moved.append(f"{p.name} -> {bak.name}")
        if moved:
            self._log("WARN", f"Migracja starego checkpointu ({reason}): {', '.join(moved)}.")

    def save_state(self) -> Path:
        with self.lock:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = self.state_dir / "brain_weights.pt"
            meta_path = self.state_dir / "brain_memory.json"
            tmp_ckpt = self.state_dir / "brain_weights.pt.tmp"
            tmp_meta = self.state_dir / "brain_memory.json.tmp"

            torch.save(
                {
                    "version": CHECKPOINT_VERSION,
                    "cortex": self.cortex.export_state(),
                    "language_cortex": self.language_cortex.export_state(),
                    "hippocampus_engrams": self.hippocampus.export_sparse_tensors(),
                    "living_brain": self.living_brain.export_state(),
                    "working_memory_slots": self._wm_slot_buf[: self._wm_slot_count].detach().cpu().to(torch.bfloat16),
                },
                tmp_ckpt,
            )
            os.replace(tmp_ckpt, ckpt_path)

            meta = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "day": self.day,
                "energy": self.cfg.max_energy,
                "creativity_level": self.creativity_level,
                "total_sleeps": self.total_sleeps,
                "auto_sleep": self.auto_sleep,
                "auto_web": self.auto_web,
                "lora_updates_count": self.language_cortex.lora_updates_count,
                "core_knowledge_summary": self.language_cortex.core_knowledge_summary,
                "hippocampus_meta": self.hippocampus.export_metadata(),
                "chat_log": self.chat_log[-1000:],
                "working_chat": self.working_chat[-self.cfg.working_chat_buffer :],
                "sleep_history": self.sleep_history,
                "dmn_thoughts": self.dmn_thoughts[-80:],
            }
            tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_meta, meta_path)
            trim_process_ram()
            return ckpt_path

    def save_state_async(self) -> None:
        """Non-blocking background checkpoint save so chat stays instant even with a ~490M brain."""
        with self._save_lock:
            if self._save_thread is not None and self._save_thread.is_alive():
                return

            def _bg_save() -> None:
                try:
                    self.save_state()
                except Exception:
                    pass

            self._save_thread = threading.Thread(target=_bg_save, daemon=True)
            self._save_thread.start()

    def load_state(self) -> bool:
        ckpt_path = self.state_dir / "brain_weights.pt"
        meta_path = self.state_dir / "brain_memory.json"
        if not ckpt_path.exists() and not meta_path.exists():
            return False

        if not ckpt_path.exists() or not meta_path.exists():
            self._backup_legacy_files([ckpt_path, meta_path], "niekompletny checkpoint")
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("checkpoint_version") != CHECKPOINT_VERSION:
                self._backup_legacy_files([ckpt_path, meta_path], "stara wersja tokenowa JSON")
                return False

            weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if not isinstance(weights, dict) or weights.get("version") != CHECKPOINT_VERSION:
                self._backup_legacy_files([ckpt_path, meta_path], "stara wersja tokenowa PT")
                return False

            self.cortex.import_state(weights["cortex"])
            self.language_cortex.import_state(weights["language_cortex"])
            broca_rep = getattr(self.language_cortex._broca_raw, "last_import_report", None)
            if broca_rep and (broca_rep.get("mismatched") or broca_rep.get("missing") or broca_rep.get("unexpected")):
                self._log(
                    "INIT",
                    f"Migracja BrocaByteTransformer (MoE): wczytano {len(broca_rep['loaded'])} tensorów, "
                    f"pominięto niezgodne kształty={len(broca_rep['mismatched'])}, "
                    f"zainicjalizowano nowe={len(broca_rep['missing'])}, "
                    f"pominięto stare={len(broca_rep['unexpected'])}.",
                )
            self.hippocampus.import_state(weights["hippocampus_engrams"], meta.get("hippocampus_meta", {}))
            if "living_brain" in weights:
                self.living_brain.import_state(weights["living_brain"])
            self._wm_slot_buf.zero_()
            self._wm_slot_count = 0
            if "working_memory_slots" in weights and isinstance(weights["working_memory_slots"], torch.Tensor):
                saved_wm = weights["working_memory_slots"]
                if saved_wm.ndim == 2 and saved_wm.shape[1] == self._wm_slot_buf.shape[1] and saved_wm.shape[0] > 0:
                    take = min(int(saved_wm.shape[0]), int(self._wm_slot_buf.shape[0]))
                    self._wm_slot_buf[:take].copy_(
                        saved_wm[-take:].to(device=self.device, dtype=self._wm_slot_buf.dtype)
                    )
                    self._wm_slot_count = take

            del weights
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            self.day = int(meta.get("day", 1))
            self.energy = self.cfg.max_energy
            self.creativity_level = max(0.0, min(1.0, float(meta.get("creativity_level", self.cfg.creativity_level))))
            self.cfg.creativity_level = self.creativity_level
            self.total_sleeps = int(meta.get("total_sleeps", 0))
            self.auto_sleep = bool(meta.get("auto_sleep", True))
            self.auto_web = bool(meta.get("auto_web", True))
            self.language_cortex.lora_updates_count = int(meta.get("lora_updates_count", 0))
            self.language_cortex.core_knowledge_summary = str(meta.get("core_knowledge_summary", ""))
            self.chat_log = list(meta.get("chat_log", []))
            self.working_chat = list(meta.get("working_chat", []))
            self.sleep_history = list(meta.get("sleep_history", []))
            self.dmn_thoughts = list(meta.get("dmn_thoughts", []))[-80:]

            self._log(
                "INIT",
                f"Wczytano beztokenowy mózg (Dzień {self.day}, {self.total_sleeps} nocy, "
                f"{len(self.hippocampus.semantic)} engramów trwałych).",
            )
            return True
        except Exception as exc:
            self._backup_legacy_files([ckpt_path, meta_path], f"niezgodny format: {exc}")
            return False

    def reset_brain(self) -> None:
        self.stop_study()
        with self.lock:
            for fname in ("brain_weights.pt", "brain_memory.json"):
                p = self.state_dir / fname
                if p.exists():
                    p.unlink()
            self.language_cortex.reset_models()
            self.living_brain.reset()
            if self.broca_llm is not None:
                self.broca_llm.reset_adapter()
            self.hippocampus = Hippocampus(
                num_neurons=self.cfg.num_neurons,
                sdr_k=self.cfg.sdr_k,
                max_short_term_memories=self.cfg.max_short_term_memories,
                max_semantic_memories=self.cfg.max_semantic_memories,
                max_rem_reflections=self.cfg.max_rem_reflections,
                device=self.device,
                bio_synapse_dtype=self.cfg.bio_synapse_dtype,
            )
            self.cortex = BiologicalAssociativeCortex(self.cfg, self.device)
            self.concept_cortex = self.cortex
            self.sleep_engine = SleepEngine(
                cfg=self.cfg,
                cortex=self.cortex,
                hippocampus=self.hippocampus,
                device=self.device,
                language_cortex=self.language_cortex,
            )
            self._wake_broca_buffer.clear()
            self._working_sdr_ring.clear()
            self._wm_slot_buf.zero_()
            self._wm_slot_count = 0
            self.state_mode = "AWAKE"
            self.energy = self.cfg.max_energy
            self.day = 1
            self.total_sleeps = 0
            self._active_op = None
            self.chat_log.clear()
            self.working_chat.clear()
            self.sleep_history.clear()
            self.event_log.clear()
            self.dmn_thoughts.clear()
            self.training_dialogue_feed.clear()
            self._seed_initial_knowledge()

    def build_graph_payload(self, max_edges: int = 85) -> dict[str, Any]:
        return self.cortex.compute_2d_projection(
            max_nodes=52,
            max_edges=max_edges,
            engram_bank=self.hippocampus.semantic.engrams if len(self.hippocampus.semantic) > 0 else None,
        )

    def creativity_score(self) -> float:
        with self.lock:
            base = max(0.0, min(1.0, float(self.creativity_level))) * 60.0
            rem_bonus = min(25.0, len(self.hippocampus.rem) * 2.5)
            sem_bonus = min(15.0, len(self.hippocampus.semantic) * 0.1)
            return round(min(100.0, base + rem_bonus + sem_bonus), 1)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            pending_info = len(self.hippocampus.short_term)
            info_pressure = round(
                min(100.0, (pending_info / max(1, self.cfg.auto_sleep_pending_threshold)) * 100.0),
                1,
            )
            w_params = self.language_cortex.wernicke_parameters()
            c_params = self.cortex.total_parameters()
            b_params = self.language_cortex.broca_parameters()
            active_syn = self.cortex.active_synapses_count()
            llm = self.broca_llm
            if llm is not None:
                short_llm = llm.model_id.split("/")[-1]
                lc_model_id = llm.model_id
                lc_arch = f"Hybryda: Wernicke SDR + Hopfield + {short_llm} LoRA"
                lc_ready = llm.ready
                lc_loading = llm.loading
                lc_error = llm.load_error
                lc_status = llm.status_message
                lc_lora_updates = llm.lora_updates_count + self.language_cortex.lora_updates_count
                lc_broca_params = llm.total_params_count or b_params
                lc_fluency = 100.0 if llm.ready else 0.0
                lc_loss = round(llm.last_loss, 3)
            else:
                lc_model_id = self.cfg.language_model_id
                lc_arch = self.cfg.architecture_label
                lc_ready = self.language_cortex.ready
                lc_loading = self.language_cortex.loading
                lc_error = self.language_cortex.load_error
                lc_status = self.language_cortex.status_message
                lc_lora_updates = self.language_cortex.lora_updates_count
                lc_broca_params = b_params
                lc_fluency = self.language_cortex.broca_module.fluency_score
                lc_loss = round(self.language_cortex.broca_module.last_loss, 3)

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
                "living_brain": self.living_brain.snapshot(),
                "parquet_corpus": self.get_parquet_corpus_info(),
                "teacher_distill": {
                    "teacher_model": self.cfg.teacher_model_id,
                    "saved_dialogues": count_teacher_dialogues(self.state_dir),
                    "saved_reasoning": count_reasoning_dialogues(self.state_dir),
                },
                "self_correction_enabled": self.cfg.self_correction_enabled,
                "study": dict(self.study_state),
                "studied_topics": list(self.hippocampus.studied_topics.values()),
                "language_cortex": {
                    "model_id": lc_model_id,
                    "architecture": lc_arch,
                    "ready": lc_ready,
                    "loading": lc_loading,
                    "error": lc_error,
                    "status": lc_status,
                    "lora_updates": lc_lora_updates,
                    "total_params": w_params + c_params + lc_broca_params,
                    "wernicke_params": w_params,
                    "cortex_params": c_params,
                    "broca_params": lc_broca_params,
                    "num_columns": self.cfg.num_columns,
                    "column_dim": self.cfg.column_dim,
                    "num_neurons": self.cfg.num_neurons,
                    "sdr_k": self.cfg.sdr_k,
                    "sdr_sparsity_pct": self.cortex.last_sparsity_pct,
                    "active_synapses": active_syn,
                    "bio_synapse_dtype": self.cortex.bio_synapse_mode,
                    "cortex_w_rec_mb": self.cortex.synapse_memory_mb(),
                    "hippocampus_engrams_mb": self.hippocampus.memory_mb(),
                    "attractor_steps": self.cortex.last_attractor_steps,
                    "fluency": lc_fluency,
                    "trained_steps": self.language_cortex.broca_module.trained_steps,
                    "last_loss": lc_loss,
                    "core_knowledge_summary": self.language_cortex.core_knowledge_summary,
                },
                "limits": {
                    "short_term_memories": self.cfg.max_short_term_memories,
                    "semantic_memories": self.cfg.max_semantic_memories,
                    "working_chat_window": self.cfg.working_chat_window,
                    "working_sdr_capacity": self.cfg.working_sdr_capacity,
                    "max_input_bytes": self.cfg.max_input_bytes,
                    "max_new_tokens": self.cfg.broca_llm_max_new_tokens,
                    "num_columns": self.cfg.num_columns,
                },
                "chat_log": self.chat_log[-400:],
                "memory": {
                    "short_term_memory_count": len(self.hippocampus.short_term),
                    "short_term_fact_count": active_syn,
                    "semantic_memory_count": len(self.hippocampus.semantic),
                    "consolidated_fact_count": active_syn,
                    "rem_dream_count": len(self.hippocampus.rem),
                    "rem_reflection_count": len(self.hippocampus.rem),
                    "working_sdr_count": len(self._working_sdr_ring),
                    "cortex_concepts": self.cfg.num_neurons,
                    "short_term_memories": self.hippocampus.short_term.to_public_list(limit=80),
                    "semantic_memories": self.hippocampus.semantic.to_public_list(limit=120, sort_by_strength=True),
                    "rem_reflections": self.hippocampus.rem.to_public_list(limit=40),
                },
                "graph": self.build_graph_payload(),
                "sleep_history": self.sleep_history[-15:],
                "event_log": self.event_log[-35:],
                "dmn_thoughts": self.dmn_thoughts[-25:],
                "training_dialogue_feed": self.training_dialogue_feed[-80:],
            }
