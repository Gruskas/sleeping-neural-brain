from __future__ import annotations

import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F

from .config import BrainConfig
from .language_trainer import (
    build_denoising_corpus,
    build_dialogue_qa_pairs,
    expand_question_paraphrases,
    split_clean_sentences,
)
from .neural_cortex import BiologicalAssociativeCortex, CortexThought
from .neural_language import BrocaByteTransformer
from .sensory_wernicke import (
    WernickeByteCortex,
    WernickeOutput,
    amp_autocast,
    configure_torch_backend,
    encode_utf8_bytes,
    maybe_compile_module,
)
from .teacher_distill import is_ai_corporate_reply

_PL_TRANS = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
_CASUAL_FILLER_RE = re.compile(
    r"\b(?:kurwa|kurde|ku[zź]wa|cholera|cholerka|ja\s+pierdol[eę]|pierdol[eę]|do\s+cholery|ej|halo|no\s+we[zź])\b",
    flags=re.IGNORECASE,
)


def _strip_pl(text: str) -> str:
    return text.translate(_PL_TRANS)


def _chat_perception_variants(text: str) -> list[str]:
    """
    Build up to 4 normalized surface variants (raw, lowercase, stripped-filler/punctuation)
    for case-insensitive UTF-8 byte perception in Wernicke (so 'Siema', 'SIEMA', 'Cześć',
    'Co to jest kot', and 'CO TAM KURWA' all map cleanly to their trained SDR attractors!).
    """
    raw = (text or "").strip()
    if not raw:
        return [" "]
    out: list[str] = []
    seen: set[str] = set()

    def _add(v: str) -> None:
        s = re.sub(r"\s+", " ", v).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    _add(raw)
    low = raw.lower()
    _add(low)
    no_punct = low.rstrip("?!.,;: ").strip()
    _add(no_punct)
    cleaned = _CASUAL_FILLER_RE.sub(" ", no_punct)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-?!")
    if len(cleaned) >= 2:
        _add(cleaned)
        _add(_strip_pl(cleaned))
    elif len(no_punct) >= 2:
        _add(_strip_pl(no_punct))
    return out[:4]


class LanguageCortex:
    """
    Thin neurobiological pipeline coordinator (ZERO templates, ZERO quote-gluing, ZERO tokens):
      Raw UTF-8 bytes
        -> WernickeByteCortex (Byte-Patching Conv1D + RMSNorm/SwiGLU -> 16 cols S -> k-WTA SDR)
        -> BiologicalAssociativeCortex (Modern Continuous Hopfield + W_rec + k-WTA -> 16 cols Z*)
        -> BrocaByteTransformer (Cross-Attention ONLY on Z* -> stochastic UTF-8 byte stream)

    Broca training runs strictly as denoising autoencoding through the k-WTA SDR bottleneck
    with label_smoothing=0.1, Gaussian noise + column dropout on Z*, AdamW(weight_decay=0.05),
    and early stopping around loss 0.85-1.0 so it learns fluent Polish syntax without memorizing quotes.
    """

    def __init__(self, cfg: BrainConfig, device: torch.device, lora_dir: Path | None = None) -> None:
        configure_torch_backend()
        self.cfg = cfg
        self.device = device
        self.lora_dir = lora_dir
        self._op_lock = threading.RLock()

        self._wernicke_raw = WernickeByteCortex(cfg).to(device)
        self._broca_raw = BrocaByteTransformer(cfg).to(device)

        self.wernicke = maybe_compile_module(self._wernicke_raw, cfg.enable_torch_compile)
        self.broca = maybe_compile_module(self._broca_raw, cfg.enable_torch_compile)
        self._device_label = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"

        self.ready: bool = True
        self.loading: bool = False
        self.load_error: str | None = None
        self.lora_updates_count: int = 0
        self.core_knowledge_summary: str = ""
        self.status_message: str = ""
        # Pure GPU bfloat16 Modern Hopfield paraphrase -> canonical question SDR attractor bank
        self._known_q_sdrs: torch.Tensor = torch.empty((0, cfg.num_neurons), dtype=torch.bfloat16, device=device)
        self._canonical_q_sdrs: torch.Tensor = torch.empty((0, cfg.num_neurons), dtype=torch.bfloat16, device=device)
        # Rolling Q&A replay pairs learned online so NREM sleep preserves them instead of forgetting
        self._learned_qa_replay: list[tuple[str, str]] = []
        # Pinned identity / rule question keys (lowercase) immune to [-200:] eviction
        self._pinned_qa_keys: set[str] = set()
        self._refresh_status()

    @property
    def wernicke_module(self) -> WernickeByteCortex:
        return self._wernicke_raw

    @property
    def broca_module(self) -> BrocaByteTransformer:
        return self._broca_raw

    def wernicke_parameters(self) -> int:
        return self._wernicke_raw.total_parameters()

    def broca_parameters(self) -> int:
        return self._broca_raw.total_parameters()

    def total_parameters(self) -> int:
        return self.wernicke_parameters() + self.broca_parameters()

    def _refresh_status(self) -> None:
        m_params = self.total_parameters() / 1_000_000.0
        fluency = self._broca_raw.fluency_score
        self.status_message = (
            f"Gotowa: {self.cfg.architecture_label} ({self._device_label} · "
            f"{m_params:.2f}M param. Wernicke+Broca · {self.cfg.num_columns} kolumn · płynność {fluency:.0f}%)"
        )

    def load_sync(self) -> bool:
        with self._op_lock:
            self._wernicke_raw.to(self.device)
            self._broca_raw.to(self.device)
            if self._broca_raw.trained_steps == 0 and self.cfg.language_bootstrap_epochs > 0:
                boot_corpus = build_denoising_corpus(
                    extra_texts=None,
                    include_base=True,
                    include_curriculum=False,
                    max_samples=64,
                )
                self._train_denoising_epochs_unlocked(
                    texts=boot_corpus,
                    epochs=self.cfg.language_bootstrap_epochs,
                    lr=self.cfg.broca_lr,
                    batch_size=min(self.cfg.broca_batch_size, 16),
                    early_stop_loss=self.cfg.broca_early_stop_loss,
                    mask_prob=0.16,
                )
            self.ready = True
            self.loading = False
            self.load_error = None
            self._refresh_status()
            return True

    def load_async(self) -> None:
        self.load_sync()

    def shutdown(self) -> None:
        return

    def reset_models(self) -> None:
        with self._op_lock:
            self._wernicke_raw = WernickeByteCortex(self.cfg).to(self.device)
            self._broca_raw = BrocaByteTransformer(self.cfg).to(self.device)
            self.wernicke = maybe_compile_module(self._wernicke_raw, self.cfg.enable_torch_compile)
            self.broca = maybe_compile_module(self._broca_raw, self.cfg.enable_torch_compile)
            self.lora_updates_count = 0
            self.core_knowledge_summary = ""
            self._known_q_sdrs = torch.empty((0, self.cfg.num_neurons), dtype=torch.bfloat16, device=self.device)
            self._canonical_q_sdrs = torch.empty((0, self.cfg.num_neurons), dtype=torch.bfloat16, device=self.device)
            self._learned_qa_replay = []
            self._pinned_qa_keys = set()
            self._refresh_status()

    def export_state(self) -> dict[str, Any]:
        with self._op_lock:
            return {
                "wernicke": self._wernicke_raw.export_state(),
                "broca": self._broca_raw.export_state(),
                "lora_updates_count": self.lora_updates_count,
                "known_q_sdrs": self._known_q_sdrs.detach().cpu(),
                "canonical_q_sdrs": self._canonical_q_sdrs.detach().cpu(),
                "learned_qa_replay": list(self._learned_qa_replay),
                "pinned_qa_keys": sorted(self._pinned_qa_keys),
            }

    def import_state(self, state: dict[str, Any]) -> None:
        with self._op_lock:
            self._wernicke_raw.import_state(state["wernicke"], device=self.device)
            self._broca_raw.import_state(state["broca"], device=self.device)
            self.lora_updates_count = int(state.get("lora_updates_count", 0))
            if "known_q_sdrs" in state and isinstance(state["known_q_sdrs"], torch.Tensor):
                self._known_q_sdrs = state["known_q_sdrs"].to(device=self.device, dtype=torch.bfloat16)
            else:
                self._known_q_sdrs = torch.empty((0, self.cfg.num_neurons), dtype=torch.bfloat16, device=self.device)
            if "canonical_q_sdrs" in state and isinstance(state["canonical_q_sdrs"], torch.Tensor):
                self._canonical_q_sdrs = state["canonical_q_sdrs"].to(device=self.device, dtype=torch.bfloat16)
            else:
                self._canonical_q_sdrs = torch.empty((0, self.cfg.num_neurons), dtype=torch.bfloat16, device=self.device)
            self._pinned_qa_keys = {
                str(k).strip().lower() for k in state.get("pinned_qa_keys", []) if str(k).strip()
            }
            raw_replay = [
                (str(x[0]), str(x[1])) for x in state.get("learned_qa_replay", []) if isinstance(x, (list, tuple)) and len(x) == 2
            ]
            pinned_entries = [p for p in raw_replay if p[0].lower() in self._pinned_qa_keys]
            unpinned_entries = [p for p in raw_replay if p[0].lower() not in self._pinned_qa_keys]
            rem_slots = max(0, 200 - len(pinned_entries))
            self._learned_qa_replay = pinned_entries + (unpinned_entries[-rem_slots:] if rem_slots > 0 else [])
            if (
                self._known_q_sdrs.shape[0] == 0
                or self._canonical_q_sdrs.shape[0] != self._known_q_sdrs.shape[0]
            ) and self._broca_raw.trained_steps > 0:
                self.ensure_familiarity_bank(force_rebuild=True)
            self._refresh_status()

    @torch.no_grad()
    def ensure_familiarity_bank(self, force_rebuild: bool = False) -> int:
        """
        Guarantee the GPU paraphrase -> canonical SDR attractor bank is populated whenever Broca
        has trained weights, so paraphrase snapping & OOD gating can NEVER silently disable themselves.
        """
        with self._op_lock:
            if not force_rebuild and self._known_q_sdrs.shape[0] > 0 and self._canonical_q_sdrs.shape[0] == self._known_q_sdrs.shape[0]:
                return int(self._known_q_sdrs.shape[0])
            if self._broca_raw.trained_steps <= 0:
                return 0
            pairs = build_dialogue_qa_pairs(
                include_curriculum=True,
                extra_pairs=list(self._learned_qa_replay),
                min_chars=2,
            )
            self._register_known_questions([q for q, _ in pairs], replace=True)
            return int(self._known_q_sdrs.shape[0])

    @torch.no_grad()
    def _register_known_questions(self, questions: list[str], replace: bool = False) -> None:
        """
        Encode canonical trained questions and their natural Polish surface paraphrases into a
        paired GPU bfloat16 Modern Hopfield attractor bank (`_known_q_sdrs -> _canonical_q_sdrs`).
        Zero text is retained in the bank — strictly sparse SDR vectors!
        """
        canon_list: list[str] = []
        canon_idx_map: dict[str, int] = {}
        variant_texts: list[str] = []
        variant_to_canon: list[int] = []
        seen_v: set[str] = set()

        for raw_q in questions:
            q = (raw_q or "").strip()
            if len(q) < 2 or q.startswith("__"):
                continue
            ckey = q.lower()
            if ckey not in canon_idx_map:
                canon_idx_map[ckey] = len(canon_list)
                canon_list.append(q)
            c_idx = canon_idx_map[ckey]
            for variant in expand_question_paraphrases(q):
                for v_form in (variant, variant.lower()):
                    if v_form not in seen_v:
                        seen_v.add(v_form)
                        variant_texts.append(v_form)
                        variant_to_canon.append(c_idx)

        if not variant_texts or not canon_list:
            return

        w_canon = self._wernicke_raw.encode_texts(canon_list, device=self.device, batch_size=64)
        canon_sdrs = F.normalize(w_canon.sdr.float(), dim=-1, eps=1e-6).to(torch.bfloat16)

        w_var = self._wernicke_raw.encode_texts(variant_texts, device=self.device, batch_size=64)
        var_sdrs = F.normalize(w_var.sdr.float(), dim=-1, eps=1e-6).to(torch.bfloat16)
        idx_t = torch.tensor(variant_to_canon, dtype=torch.long, device=self.device)
        mapped_canon = canon_sdrs[idx_t]

        max_bank = 6144
        if replace or self._known_q_sdrs.shape[0] == 0:
            self._known_q_sdrs = var_sdrs[-max_bank:].contiguous()
            self._canonical_q_sdrs = mapped_canon[-max_bank:].contiguous()
        else:
            self._known_q_sdrs = torch.cat([self._known_q_sdrs, var_sdrs], dim=0)[-max_bank:].contiguous()
            self._canonical_q_sdrs = torch.cat([self._canonical_q_sdrs, mapped_canon], dim=0)[-max_bank:].contiguous()

    @torch.no_grad()
    def match_known_attractor(self, query_sdr: torch.Tensor) -> tuple[float, torch.Tensor | None]:
        """
        Compute continuous familiarity score against the GPU bfloat16 SDR bank WITHOUT
        overwriting/replacing the input SDR (so novel syntax & word order are always preserved).
        """
        if self._known_q_sdrs.shape[0] == 0:
            return 1.0, None
        q = F.normalize(query_sdr.to(device=self.device, dtype=torch.float32).reshape(-1), dim=-1, eps=1e-6).to(
            torch.bfloat16
        )
        sims = torch.mv(self._known_q_sdrs, q).float()
        best_val = sims.max(dim=0).values
        sim_f = round(float(best_val.item()), 4)
        return sim_f, None

    @torch.no_grad()
    def familiarity_score(self, query_sdr: torch.Tensor) -> float:
        """Single GPU matmul checking if query_sdr matches any trained dialogue/Q&A paraphrase."""
        sim_f, _ = self.match_known_attractor(query_sdr)
        return sim_f

    @torch.no_grad()
    def perceive_for_chat(self, user_message: str) -> tuple[WernickeOutput, float, torch.Tensor | None]:
        """
        Case- & punctuation-tolerant familiarity scoring on GPU while preserving the REAL
        input structure (`w_batch[0]`, including sequential `patch_columns` and `patch_mask`).
        """
        variants = _chat_perception_variants(user_message)
        with self._op_lock:
            w_batch = self._wernicke_raw.encode_texts(variants, device=self.device, batch_size=len(variants))
            best_sim = -1.0
            for i in range(len(variants)):
                sim_i, _ = self.match_known_attractor(w_batch.sdr[i])
                if sim_i > best_sim:
                    best_sim = sim_i
            single_out = WernickeOutput(
                columns=w_batch.columns[:1],
                sdr=w_batch.sdr[:1],
                reconstructed_columns=w_batch.reconstructed_columns[:1],
                active_indices=w_batch.active_indices[:1],
                active_values=w_batch.active_values[:1],
                sparsity_pct=w_batch.sparsity_pct,
                patch_columns=w_batch.patch_columns[:1] if w_batch.patch_columns is not None else None,
                patch_mask=w_batch.patch_mask[:1] if w_batch.patch_mask is not None else None,
            )
            return single_out, best_sim, None

    # ── Sensory Perception (Wernicke: UTF-8 bytes -> 16 columns S + k-WTA SDR) ──
    @torch.no_grad()
    def perceive_texts(self, texts: list[str], mask_prob: float = 0.0) -> WernickeOutput:
        if not texts:
            raise ValueError("Cannot perceive empty text batch")
        with self._op_lock:
            return self._wernicke_raw.encode_texts(
                texts,
                device=self.device,
                batch_size=self.cfg.embed_batch_size,
                mask_prob=mask_prob,
            )

    @torch.no_grad()
    def perceive_one(self, text: str) -> WernickeOutput:
        return self.perceive_texts([text or " "])

    # ── Isolated Broca Decoding (STRICTLY from Z* / Z_ctx) ─────────────────
    @torch.no_grad()
    def decode_columns(
        self,
        columns_star: torch.Tensor,
        max_new_bytes: int | None = None,
        temperature: float | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        with self._op_lock:
            return self._broca_raw.generate(
                memory_columns=columns_star,
                max_new_bytes=max_new_bytes,
                temperature=temperature,
                top_p=self.cfg.broca_top_p,
                repetition_penalty=self.cfg.broca_repetition_penalty,
                memory_mask=memory_mask,
            )

    @staticmethod
    def score_polish_quality(text: str, creativity_level: float = 0.6) -> float:
        """
        Evaluate inner Broca draft for Polish linguistic quality without any external model:
          - Penalizes empty/tiny drafts, repetitive word loops ('ciekawe i ciekawe'),
            non-Polish symbol spam, and unbalanced brackets.
          - Rewards clean sentence endings ('.', '!', '?'), good alphabetic ratio, and
            lexical diversity (scaled by creativity_level).
        Returns a quality score in [0.0, 1.0].
        """
        s = (text or "").strip()
        if len(s) < 6:
            return 0.05
        score = 0.55
        if s[-1] in ".!?":
            score += 0.18
        if s[0].isupper() or s[0] in "„\"":
            score += 0.07
        letters = sum(1 for ch in s if ch.isalpha())
        alpha_ratio = letters / float(max(1, len(s)))
        if alpha_ratio < 0.55:
            score -= 0.30
        elif alpha_ratio >= 0.72:
            score += 0.08

        words = [w.strip(".,!?;:„”\"()[]—-").lower() for w in s.split() if w.strip(".,!?;:„”\"()[]—-")]
        if len(words) >= 3:
            uniq_ratio = len(set(words)) / float(len(words))
            # Strong penalty for word loops ("ciekawe i ciekawe", "jest jest jest")
            adj_dups = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
            score -= 0.22 * adj_dups
            if uniq_ratio < 0.60:
                score -= 0.25
            elif uniq_ratio >= 0.85:
                score += 0.08 + 0.06 * max(0.0, min(1.0, creativity_level))
            # Check bigram repetition loops
            if len(words) >= 6:
                bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
                bg_ratio = len(set(bigrams)) / float(len(bigrams))
                if bg_ratio < 0.75:
                    score -= 0.25

        if s.count("(") != s.count(")"):
            score -= 0.08
        if "\ufffd" in s:
            score -= 0.35
        return round(max(0.0, min(1.0, score)), 4)

    @torch.no_grad()
    def analyze_self_heard_draft(
        self,
        draft_text: str,
        thought: CortexThought,
        creativity_level: float = 0.6,
        heard_sdr: torch.Tensor | None = None,
        conf_info: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Biological Arcuate Fasciculus Auditory Self-Hearing & Error Analysis:
        The brain literally 'hears' its own generated draft through Wernicke, compares the heard SDR
        against its intended cortical state Z*, checks Broca neural confidence & Polish coherence,
        and diagnoses concrete flaws so it can actively self-correct BEFORE speaking!
        """
        s = (draft_text or "").strip()
        quality = self.score_polish_quality(s, creativity_level=creativity_level)
        if len(s) < 4:
            return {
                "total": 0.05,
                "quality": quality,
                "alignment": 0.0,
                "confidence": 0.01,
                "avg_nll": 5.0,
                "issues": ["zbyt krótka / pusta odpowiedź"],
                "needs_correction": True,
                "heard_sdr": heard_sdr,
            }

        if heard_sdr is None:
            with self._op_lock:
                fb_out = self._wernicke_raw.encode_texts([s], device=self.device, batch_size=1)
            heard_sdr = fb_out.sdr[0]
        if conf_info is None:
            ctx_cols = thought.context_columns if thought.context_columns is not None else thought.columns_star
            with self._op_lock:
                conf_info = self._broca_raw.score_text_confidence(
                    s,
                    ctx_cols,
                    memory_mask=thought.context_mask,
                )

        align = float(
            F.cosine_similarity(
                heard_sdr.float().unsqueeze(0),
                thought.z_star[0].float().unsqueeze(0),
            ).item()
        )
        align_norm = max(0.0, min(1.0, (align + 0.10) / 0.75))
        conf = float(conf_info["confidence"])
        nll = float(conf_info["avg_nll"])
        total = round(0.45 * conf + 0.45 * quality + 0.10 * align_norm, 4)

        issues: list[str] = []
        if nll > 1.05:
            issues.append(f"niska pewność Broki (NLL={nll:.2f})")
        if quality < 0.75:
            issues.append(f"słaba płynność polska ({quality*100:.0f}%)")
        if align < 0.16:
            issues.append(f"rozbieżność słuchowa z intencją Z* (cos={align:.2f})")
        words = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+", s.lower())
        if len(words) >= 6 and (len(set(words)) / float(len(words))) < 0.52:
            issues.append("pętla powtórzeń słów")
        if len(s) >= 24 and s[-1] not in ".!?\"')»…":
            issues.append("ucięte zdanie bez zakończenia")
        if "\ufffd" in s:
            issues.append("uszkodzone znaki UTF-8")

        thresh = float(getattr(self.cfg, "self_correction_threshold", 0.78))
        needs_correction = bool(issues) or (total < thresh)
        return {
            "total": total,
            "quality": round(quality, 4),
            "alignment": round(align, 4),
            "confidence": round(conf, 4),
            "avg_nll": round(nll, 4),
            "issues": issues,
            "needs_correction": needs_correction,
            "heard_sdr": heard_sdr,
        }

    @torch.no_grad()
    def self_critique_reply(
        self,
        draft_text: str,
        thought: CortexThought,
        creativity_level: float = 0.6,
    ) -> dict[str, float]:
        """
        Biological Arcuate Fasciculus auditory feedback + Broca neural confidence critic:
        Wernicke reads Broca's own output while Broca evaluates its own byte perplexity under Z*.
        """
        diag = self.analyze_self_heard_draft(draft_text, thought, creativity_level=creativity_level)
        return {
            "total": float(diag["total"]),
            "quality": float(diag["quality"]),
            "alignment": float(diag["alignment"]),
            "confidence": float(diag["confidence"]),
            "avg_nll": float(diag["avg_nll"]),
        }

    @torch.no_grad()
    def synthesize_2hop_thought(
        self,
        engram_a: dict[str, Any] | torch.Tensor,
        engram_b: dict[str, Any] | torch.Tensor,
        engram_c: dict[str, Any] | torch.Tensor,
        cortex: BiologicalAssociativeCortex,
        engram_bank: torch.Tensor | None = None,
        creativity_level: float = 0.70,
        max_new_bytes: int = 128,
        num_candidates: int = 2,
        min_joint_coherence: float = 0.16,
        max_copy_sim: float = 0.88,
        abort_event: threading.Event | None = None,
        non_blocking_lock: bool = False,
    ) -> dict[str, Any]:
        """
        Autonomous 2-hop Concept Graph synthesis (A -> B -> C):
          1. Build multi-slot Z_ctx with distinct segment-embedded slots for [A, B, C]
             around their settled cortical bridge attractor Z*.
          2. Generate candidate short insight(s) connecting A and C through BrocaByteTransformer
             strictly from tensor Z_ctx (ZERO string/preview_label input to Broca).
          3. Pass each candidate through the 2-stage Quality Gate:
             (a) Auditory self-critique (`analyze_self_heard_draft`) must pass with
                 `needs_correction=False` and zero word repetition loops.
             (b) Novelty vs A and C (SDR cosine < max_copy_sim and not a copy of A/C)
                 plus meaningful coherence with their joint context (>= min_joint_coherence).
        """
        from .sensory_wernicke import k_wta

        def _extract_sdr(item: dict[str, Any] | torch.Tensor) -> torch.Tensor:
            t = item["sdr"] if isinstance(item, dict) else item
            return F.normalize(t.to(device=self.device, dtype=torch.float32).reshape(-1), dim=-1, eps=1e-6)

        def _extract_label(item: dict[str, Any] | torch.Tensor) -> str:
            return str(item.get("preview_label", "") or "").strip() if isinstance(item, dict) else ""

        if abort_event is not None and abort_event.is_set():
            return {"accepted": False, "aborted": True, "thought_text": "", "rejection_reasons": ["user_interrupt"]}

        sdr_a = _extract_sdr(engram_a)
        sdr_b = _extract_sdr(engram_b)
        sdr_c = _extract_sdr(engram_c)

        # Acquire language_cortex lock (non-blocking in idle DMN so chat/training are NEVER stalled)
        if non_blocking_lock:
            if not self._op_lock.acquire(blocking=False):
                return {"accepted": False, "aborted": True, "thought_text": "", "rejection_reasons": ["lock_busy"]}
        else:
            self._op_lock.acquire()

        try:
            if abort_event is not None and abort_event.is_set():
                return {"accepted": False, "aborted": True, "thought_text": "", "rejection_reasons": ["user_interrupt"]}

            self._wernicke_raw.eval()
            self._broca_raw.eval()

            # 1) Project A, B, C SDRs into distinct cortical column slots + settled bridge attractor
            sdrs_abc = torch.stack([sdr_a, sdr_b, sdr_c], dim=0)  # [3, N]
            with amp_autocast(self.device):
                cols_abc = self._wernicke_raw.sdr_to_columns(sdrs_abc)  # [3, 16, D]
            cols_a = self._normalize_z_columns(cols_abc[0:1])  # [1, 16, D]
            cols_b = self._normalize_z_columns(cols_abc[1:2])  # [1, 16, D]
            cols_c = self._normalize_z_columns(cols_abc[2:3])  # [1, 16, D]

            bridge_raw = 0.42 * sdr_a + 0.16 * sdr_b + 0.42 * sdr_c
            bridge_sdr, _, _ = k_wta(bridge_raw.unsqueeze(0), k=self.cfg.sdr_k, normalize=True)  # [1, N]
            with amp_autocast(self.device):
                bridge_cols = self._normalize_z_columns(self._wernicke_raw.sdr_to_columns(bridge_sdr))

            thought = cortex.perceive_and_think(
                sensory_columns=bridge_cols,
                sensory_sdr=bridge_sdr,
                engram_bank=engram_bank,
                sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
                steps=self.cfg.attractor_steps,
                noise_std=0.02 * max(0.2, float(creativity_level)),
                update_last_active=False,
            )

            # Distinct multi-slot Z_ctx with segment embeddings:
            #   primary Z* (seg 1) | A (seg 0) | B (seg 2) | C (seg 3)
            z_ctx, z_ctx_mask = cortex.build_context_columns(
                columns_star=self._normalize_z_columns(thought.columns_star),
                sensory_patch_columns=cols_a,
                working_memory_slots=cols_b,
                thought_slots=cols_c,
            )
            thought.context_columns = z_ctx
            thought.context_mask = z_ctx_mask

            if abort_event is not None and abort_event.is_set():
                return {"accepted": False, "aborted": True, "thought_text": "", "rejection_reasons": ["user_interrupt"]}

            # 2) Generate candidate short thoughts through BrocaByteTransformer
            k_cand = max(1, min(3, int(num_candidates)))
            base_temp = max(0.45, min(0.85, float(self.cfg.broca_temperature) + 0.08 * float(creativity_level)))
            temps = [max(0.40, min(0.92, base_temp + 0.07 * (i - (k_cand - 1) / 2.0))) for i in range(k_cand)]
            gen_batch = self._broca_raw.generate_batch(
                memory_columns_batch=z_ctx.expand(k_cand, -1, -1),
                temperatures=temps,
                max_new_bytes=max(32, min(220, int(max_new_bytes))),
                memory_mask=z_ctx_mask.expand(k_cand, -1),
            )
        finally:
            self._op_lock.release()

        if abort_event is not None and abort_event.is_set():
            return {"accepted": False, "aborted": True, "thought_text": "", "rejection_reasons": ["user_interrupt"]}

        # 3) Quality Gate: Auditory self-critique + zero repetition loops + novelty vs A/C + joint coherence
        joint_ac = F.normalize(0.5 * sdr_a + 0.5 * sdr_c, dim=-1, eps=1e-6)
        joint_abc = F.normalize(0.4 * sdr_a + 0.2 * sdr_b + 0.4 * sdr_c, dim=-1, eps=1e-6)
        lbl_a = _extract_label(engram_a).lower()
        lbl_c = _extract_label(engram_c).lower()
        words_a = set(re.findall(r"[a-ząćęłńóśźż0-9]{2,}", lbl_a))
        words_c = set(re.findall(r"[a-ząćęłńóśźż0-9]{2,}", lbl_c))

        best_accepted: dict[str, Any] | None = None
        best_rejected: dict[str, Any] | None = None

        for item in gen_batch:
            if abort_event is not None and abort_event.is_set():
                return {"accepted": False, "aborted": True, "thought_text": "", "rejection_reasons": ["user_interrupt"]}
            draft = (item.get("text") or "").strip()
            diag = self.analyze_self_heard_draft(draft, thought, creativity_level=creativity_level)
            heard_sdr = diag.get("heard_sdr")

            reasons: list[str] = []
            if diag["needs_correction"]:
                reasons.extend(diag["issues"] or [f"audit_score={diag['total']:.2f}"])

            words = re.findall(r"[a-ząćęłńóśźż0-9]+", draft.lower())
            if len(draft) < 12 or len(words) < 3:
                reasons.append("zbyt krótka myśl")
            if re.search(r"\b([a-ząćęłńóśźż0-9]{2,})\s+\1\b", draft.lower()):
                reasons.append("zapętlenie powtarzających się słów")
            if len(words) >= 4 and (len(set(words)) / float(len(words))) < 0.55:
                reasons.append("niska różnorodność leksykalna")

            sim_a = 0.0
            sim_b = 0.0
            sim_c = 0.0
            joint_coh = float(diag.get("alignment", 0.0))
            if heard_sdr is not None:
                h_u = F.normalize(heard_sdr.to(device=self.device, dtype=torch.float32).reshape(-1), dim=-1, eps=1e-6)
                sim_a = float(torch.dot(h_u, sdr_a).item())
                sim_b = float(torch.dot(h_u, sdr_b).item())
                sim_c = float(torch.dot(h_u, sdr_c).item())
                sim_joint_ac = float(torch.dot(h_u, joint_ac).item())
                sim_joint_abc = float(torch.dot(h_u, joint_abc).item())
                joint_coh = max(sim_joint_ac, sim_joint_abc, float(diag.get("alignment", 0.0)))
                if sim_a >= float(max_copy_sim) or sim_c >= float(max_copy_sim):
                    reasons.append(f"zbyt podobna do rodzica (sim_a={sim_a:.2f}, sim_c={sim_c:.2f})")
                if joint_coh < float(min_joint_coherence):
                    reasons.append(f"słaba spójność z kontekstem A+C (coh={joint_coh:.2f})")
            else:
                reasons.append("brak heard_sdr")

            # Post-generation text dedup guard against copying A or C verbatim
            draft_low = draft.lower()
            if (lbl_a and draft_low == lbl_a) or (lbl_c and draft_low == lbl_c):
                reasons.append("dokładna kopia A lub C")
            draft_wset = set(re.findall(r"[a-ząćęłńóśźż0-9]{2,}", draft_low))
            for ref_set, tag_r in ((words_a, "A"), (words_c, "C")):
                if len(ref_set) >= 4 and len(draft_wset) >= 4:
                    jacc = len(draft_wset & ref_set) / float(max(1, len(draft_wset | ref_set)))
                    if jacc >= 0.85:
                        reasons.append(f"kopia treści {tag_r} (jaccard={jacc:.2f})")

            cand_res = {
                "accepted": len(reasons) == 0,
                "aborted": False,
                "thought_text": draft,
                "thought_sdr": heard_sdr.detach().clone() if heard_sdr is not None else thought.z_star[0].detach().clone(),
                "bridge_columns": thought.columns_star.detach().clone(),
                "z_ctx_shape": tuple(z_ctx.shape),
                "audit": {k: v for k, v in diag.items() if k != "heard_sdr"},
                "sim_to_a": round(sim_a, 4),
                "sim_to_b": round(sim_b, 4),
                "sim_to_c": round(sim_c, 4),
                "joint_coherence": round(joint_coh, 4),
                "rejection_reasons": reasons,
            }
            if cand_res["accepted"]:
                if best_accepted is None or cand_res["audit"]["total"] > best_accepted["audit"]["total"]:
                    best_accepted = cand_res
            else:
                if best_rejected is None or cand_res["audit"]["total"] > best_rejected["audit"]["total"]:
                    best_rejected = cand_res

        if best_accepted is not None:
            return best_accepted
        return best_rejected or {
            "accepted": False,
            "aborted": False,
            "thought_text": "",
            "rejection_reasons": ["brak kandydatów"],
        }

    @torch.no_grad()
    def deliberate_and_critique(
        self,
        user_message: str,
        cortex: BiologicalAssociativeCortex,
        engram_bank: torch.Tensor | None,
        working_memory_sdr: torch.Tensor | None = None,
        working_memory_weight: float = 0.0,
        creativity_level: float = 0.7,
        base_temperature: float | None = None,
        max_new_bytes: int | None = None,
        num_candidates: int | None = None,
        living_assoc_vec: torch.Tensor | None = None,
        living_blend: float = 0.08,
        neuromod: Any | None = None,
        living_telemetry: Any | None = None,
        working_memory_slots: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Sequential System-2 Latent Chain-of-Thought + Arcuate Fasciculus Self-Hearing & Active Self-Correction:
          - Step 1 (Precyzyjna): Grounded associative perception over real Wernicke patches + PFC slots.
          - Step 2 (Kontekstowa): Appends Step-1 latent thought summary as a cortical slot & runs 2nd Hopfield hop.
          - Step 3 (Kreatywna): Appends Step-2 latent thought summary as another slot for exploratory synthesis.
          - Evaluates all chain states in ONE batched Broca decode + Arcuate Fasciculus auditory critique.
        """
        base_t = float(base_temperature if base_temperature is not None else self.cfg.broca_temperature)
        default_steps = int(getattr(self.cfg, "system2_reasoning_steps", self.cfg.self_critique_candidates))
        k_cand = max(1, min(3, int(num_candidates or default_steps)))
        cre = max(0.0, min(1.0, float(creativity_level)))
        max_b = int(max_new_bytes or self.cfg.broca_max_new_bytes)

        na_val = 0.35
        pe_val = 0.0
        cur_val = 0.0
        boredom_exp = False
        has_nm = (neuromod is not None) or (living_telemetry is not None)
        if neuromod is not None:
            if isinstance(neuromod, dict):
                na_val = float(neuromod.get("noradrenaline", 0.35))
            else:
                na_val = float(getattr(neuromod, "noradrenaline", 0.35))
        if living_telemetry is not None:
            pe_val = float(getattr(living_telemetry, "surprise", 0.0))
            cur_val = float(getattr(living_telemetry, "intrinsic_curiosity", 0.0))
            boredom_exp = bool(getattr(living_telemetry, "boredom_exploration", False))
            if neuromod is None:
                nm_dict = getattr(living_telemetry, "neuromodulators", None) or {}
                if isinstance(nm_dict, dict):
                    na_val = float(nm_dict.get("noradrenaline", na_val))

        na_arousal = (
            max(0.0, min(1.0, max((na_val - 0.85) / 1.15, (pe_val - 0.30) / 0.45)))
            if has_nm
            else 0.0
        )
        cur_boost = (
            max(0.0, min(1.0, max(0.0, (cur_val - 0.30) / 0.70) + (0.45 if boredom_exp else 0.0)))
            if living_telemetry is not None
            else 0.0
        )

        eff_steps = max(self.cfg.attractor_steps, min(6, 3 + int(round(3.0 * na_arousal)))) if has_nm else self.cfg.attractor_steps
        eff_base_t = max(0.38, base_t - 0.14 * na_arousal) if has_nm else base_t
        cre_temp = max(eff_base_t, min(0.95, eff_base_t + 0.14 * cre + 0.16 * cur_boost))
        cre_noise = 0.010 * cre + 0.008 * cur_boost

        w_out, q_sim, _ = self.perceive_for_chat(user_message)

        thoughts: list[CortexThought] = []
        temps: list[float] = []
        labels: list[str] = []
        latent_slots: list[torch.Tensor] = []
        chain_notes: list[str] = []

        # Step 1: Precyzyjna (grounded perception over actual Wernicke patch_columns + SDR + PFC slots)
        th0 = cortex.perceive_and_think(
            sensory_columns=w_out.columns,
            sensory_sdr=w_out.sdr,
            engram_bank=engram_bank,
            sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
            steps=eff_steps,
            noise_std=0.0,
            update_last_active=False,
            hopfield_min_sim=0.85,
            working_memory_sdr=working_memory_sdr if working_memory_weight > 0.0 else None,
            working_memory_weight=working_memory_weight * 0.5,
            living_assoc_vec=living_assoc_vec,
            living_blend=living_blend,
            neuromod=neuromod,
            living_telemetry=living_telemetry,
            sensory_patch_columns=w_out.patch_columns,
            sensory_patch_mask=w_out.patch_mask,
            working_memory_slots=working_memory_slots,
        )
        thoughts.append(th0)
        temps.append(round(max(0.36, eff_base_t - 0.08), 3))
        labels.append("Precyzyjna")
        latent_slots.append(F.normalize(th0.columns_star.mean(dim=1, keepdim=True), dim=-1, eps=1e-6))

        # Step 2: Kontekstowa (feeds Step-1 latent thought slot + Z*_0 back into Associative Cortex & Hopfield)
        if k_cand >= 2:
            thought_cat_1 = torch.cat(latent_slots, dim=1)
            th1 = cortex.perceive_and_think(
                sensory_columns=th0.columns_star,
                sensory_sdr=th0.z_star,
                engram_bank=engram_bank,
                sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
                steps=min(6, eff_steps + 1),
                noise_std=0.0,
                update_last_active=False,
                hopfield_min_sim=0.72,
                working_memory_sdr=working_memory_sdr if working_memory_weight > 0.0 else None,
                working_memory_weight=working_memory_weight,
                living_assoc_vec=living_assoc_vec,
                living_blend=living_blend,
                neuromod=neuromod,
                living_telemetry=living_telemetry,
                sensory_patch_columns=w_out.patch_columns,
                sensory_patch_mask=w_out.patch_mask,
                working_memory_slots=working_memory_slots,
                thought_slots=thought_cat_1,
            )
            if th1.recalled_columns is None and th0.recalled_columns is not None:
                th1.recalled_columns = th0.recalled_columns
                th1.context_columns, th1.context_mask = cortex.build_context_columns(
                    columns_star=th1.columns_star,
                    sensory_patch_columns=w_out.patch_columns,
                    sensory_patch_mask=w_out.patch_mask,
                    recalled_columns=th1.recalled_columns,
                    working_memory_slots=working_memory_slots,
                    thought_slots=thought_cat_1,
                )
            thoughts.append(th1)
            temps.append(round(eff_base_t, 3))
            labels.append("Kontekstowa")
            latent_slots.append(F.normalize(th1.columns_star.mean(dim=1, keepdim=True), dim=-1, eps=1e-6))
            chain_notes.append(
                f"Łańcuch System-2 (krok 1→2): hop1_sim={th0.max_hopfield_sim:.2f} → hop2_sim={th1.max_hopfield_sim:.2f}"
            )

        # Step 3: Kreatywna (feeds accumulated latent thought chain [h_0, h_1] for broader synthesis)
        if k_cand >= 3:
            prev_th = thoughts[-1]
            thought_cat_2 = torch.cat(latent_slots, dim=1)
            th2 = cortex.perceive_and_think(
                sensory_columns=prev_th.columns_star,
                sensory_sdr=prev_th.z_star,
                engram_bank=engram_bank,
                sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
                steps=eff_steps,
                noise_std=cre_noise,
                update_last_active=False,
                hopfield_min_sim=0.80,
                working_memory_sdr=working_memory_sdr if working_memory_weight > 0.0 else None,
                working_memory_weight=working_memory_weight * 0.5,
                living_assoc_vec=living_assoc_vec,
                living_blend=living_blend,
                neuromod=neuromod,
                living_telemetry=living_telemetry,
                sensory_patch_columns=w_out.patch_columns,
                sensory_patch_mask=w_out.patch_mask,
                working_memory_slots=working_memory_slots,
                thought_slots=thought_cat_2,
            )
            if th2.recalled_columns is None and prev_th.recalled_columns is not None:
                th2.recalled_columns = prev_th.recalled_columns
                th2.context_columns, th2.context_mask = cortex.build_context_columns(
                    columns_star=th2.columns_star,
                    sensory_patch_columns=w_out.patch_columns,
                    sensory_patch_mask=w_out.patch_mask,
                    recalled_columns=th2.recalled_columns,
                    working_memory_slots=working_memory_slots,
                    thought_slots=thought_cat_2,
                )
            thoughts.append(th2)
            temps.append(round(cre_temp, 3))
            labels.append("Kreatywna")
            latent_slots.append(F.normalize(th2.columns_star.mean(dim=1, keepdim=True), dim=-1, eps=1e-6))

        # Assemble padded Z_ctx batch [K, M_max, D] + memory_mask [K, M_max] for ONE batched GPU decode!
        ctx_list = [
            (th.context_columns[:1] if th.context_columns is not None else th.columns_star[:1])
            for th in thoughts
        ]
        mask_list = [
            (
                th.context_mask[:1]
                if th.context_mask is not None
                else torch.ones((1, c.shape[1]), dtype=torch.bool, device=c.device)
            )
            for th, c in zip(thoughts, ctx_list)
        ]
        m_max = max(c.shape[1] for c in ctx_list)
        padded_ctx: list[torch.Tensor] = []
        padded_masks: list[torch.Tensor] = []
        for c, m in zip(ctx_list, mask_list):
            pad_m = m_max - c.shape[1]
            if pad_m > 0:
                c = F.pad(c, (0, 0, 0, pad_m), value=0.0)
                m = F.pad(m, (0, pad_m), value=False)
            padded_ctx.append(c)
            padded_masks.append(m)
        mem_batch = torch.cat(padded_ctx, dim=0)
        mask_batch = torch.cat(padded_masks, dim=0)
        has_recalled_mem = any(th.recalled_columns is not None for th in thoughts)

        with self._op_lock:
            gen_batch = self._broca_raw.generate_batch(
                memory_columns_batch=mem_batch,
                temperatures=temps,
                max_new_bytes=max_b,
                top_p=self.cfg.broca_top_p,
                repetition_penalty=self.cfg.broca_repetition_penalty,
                memory_mask=mask_batch,
            )
            conf_list = self._broca_raw.score_texts_confidence_batch(
                [g["text"] for g in gen_batch],
                mem_batch,
                memory_mask=mask_batch,
            )

        draft_texts = [g["text"] if g["text"] else " " for g in gen_batch]

        # Batch auditory self-hearing: Wernicke listens to ALL generated drafts simultaneously on GPU
        with self._op_lock:
            fb_batch = self._wernicke_raw.encode_texts(draft_texts, device=self.device, batch_size=len(draft_texts))

        candidates: list[dict[str, Any]] = []
        best_idx = 0
        best_score = -1.0
        critique_notes: list[str] = []
        if has_nm and (na_arousal > 0.05 or cur_boost > 0.05):
            critique_notes.append(
                f"Neuromodulacja inferencji: NA={na_val:.2f} (kroki={eff_steps}, T_baz={eff_base_t:.2f}) · "
                f"Ciekawość M4={cur_val:.2f}{'+eksploracja' if boredom_exp else ''} (T_kreat={cre_temp:.2f})"
            )
        if chain_notes:
            critique_notes.append(
                chain_notes[-1]
                + f" · Z_ctx={m_max} slotów"
                + (" (z kolumnami Hipokampu)" if has_recalled_mem else "")
            )
        p1_conf = float(conf_list[0]["confidence"])

        for idx in range(len(thoughts)):
            draft = gen_batch[idx]["text"]
            diag = self.analyze_self_heard_draft(
                draft_text=draft,
                thought=thoughts[idx],
                creativity_level=cre,
                heard_sdr=fb_batch.sdr[idx],
                conf_info=conf_list[idx],
            )
            cre_q_min = 0.78 if cur_boost >= 0.35 else 0.85
            cre_c_min = max(0.50, p1_conf - (0.08 if cur_boost >= 0.35 else 0.04))
            cre_bonus = (
                (0.015 * cre + 0.045 * cur_boost)
                if (labels[idx] == "Kreatywna" and diag["quality"] >= cre_q_min and diag["confidence"] >= cre_c_min)
                else 0.0
            )
            total = round(min(1.0, float(diag["total"]) + cre_bonus), 4)
            cand = {
                "label": labels[idx],
                "thought": thoughts[idx],
                "temperature": temps[idx],
                "draft": draft,
                "score": total,
                "confidence": float(diag["confidence"]),
                "avg_nll": float(diag["avg_nll"]),
                "quality": float(diag["quality"]),
                "alignment": float(diag["alignment"]),
                "issues": list(diag["issues"]),
                "needs_correction": bool(diag["needs_correction"]),
                "heard_sdr": fb_batch.sdr[idx],
                "thought_slots": idx,
                "context_slots": int(
                    thoughts[idx].context_columns.shape[1]
                    if thoughts[idx].context_columns is not None
                    else thoughts[idx].columns_star.shape[1]
                ),
            }
            candidates.append(cand)
            critique_notes.append(
                f"Ścieżka {cand['label']} (T={cand['temperature']:.2f}): ocena={total*100:.0f}% "
                f"(pewność={cand['confidence']*100:.0f}%, NLL={cand['avg_nll']:.2f}, jakość={cand['quality']*100:.0f}%)"
            )
            if total > best_score:
                best_score = total
                best_idx = idx

        winner = dict(candidates[best_idx])
        self_corrected = False

        # Active Auditory Self-Correction ("słyszę co mówię -> analizuję -> poprawiam"):
        if bool(getattr(self.cfg, "self_correction_enabled", True)) and winner["needs_correction"] and winner["draft"]:
            from .sensory_wernicke import k_wta

            err_signal = winner["thought"].z_star[0].float() - winner["heard_sdr"].float()
            corrected_pre = winner["thought"].z_star[0].float() + 0.35 * err_signal
            corr_sdr, _, _ = k_wta(corrected_pre.unsqueeze(0), k=self.cfg.sdr_k, normalize=True)
            corr_thought = cortex.perceive_and_think(
                sensory_columns=w_out.columns,
                sensory_sdr=corr_sdr.to(dtype=w_out.sdr.dtype),
                engram_bank=None,
                sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
                steps=eff_steps,
                noise_std=0.0,
                update_last_active=False,
                hopfield_min_sim=0.92,
                living_assoc_vec=living_assoc_vec,
                living_blend=living_blend,
                neuromod=neuromod,
                living_telemetry=living_telemetry,
                sensory_patch_columns=w_out.patch_columns,
                sensory_patch_mask=w_out.patch_mask,
                working_memory_slots=working_memory_slots,
                thought_slots=torch.cat(latent_slots, dim=1) if latent_slots else None,
            )
            corr_temp = max(0.36, round(winner["temperature"] - 0.10, 3))
            corr_rep = round(self.cfg.broca_repetition_penalty + 0.10, 2)
            corr_ctx = corr_thought.context_columns if corr_thought.context_columns is not None else corr_thought.columns_star
            with self._op_lock:
                rev_gen = self._broca_raw.generate(
                    memory_columns=corr_ctx,
                    max_new_bytes=max_b,
                    temperature=corr_temp,
                    top_p=max(0.75, self.cfg.broca_top_p - 0.05),
                    repetition_penalty=corr_rep,
                    memory_mask=corr_thought.context_mask,
                )
            rev_diag = self.analyze_self_heard_draft(
                draft_text=rev_gen["text"],
                thought=corr_thought,
                creativity_level=cre,
            )
            issues_str = ", ".join(winner["issues"][:2]) if winner["issues"] else "niska ocena ogólna"
            if float(rev_diag["total"]) > float(winner["score"]) + 0.01 and len(rev_gen["text"].strip()) >= 4:
                old_score = winner["score"]
                winner["thought"] = corr_thought
                winner["draft"] = rev_gen["text"]
                winner["temperature"] = corr_temp
                winner["score"] = float(rev_diag["total"])
                winner["confidence"] = float(rev_diag["confidence"])
                winner["avg_nll"] = float(rev_diag["avg_nll"])
                winner["quality"] = float(rev_diag["quality"])
                winner["alignment"] = float(rev_diag["alignment"])
                winner["needs_correction"] = bool(rev_diag["needs_correction"])
                self_corrected = True
                critique_notes.append(
                    f"🎧 Słyszę siebie → wykryto: {issues_str} → aktywna samokorekta Z* poprawiła odpowiedź "
                    f"({old_score*100:.0f}% → {winner['score']*100:.0f}%)"
                )
            else:
                critique_notes.append(
                    f"🎧 Słyszę siebie → sprawdziłem poprawkę ({issues_str}), ale pierwotny szkic pozostał najlepszy "
                    f"({winner['score']*100:.0f}%)"
                )
        else:
            critique_notes.append(
                f"🎧 Słyszę własną odpowiedź → spójność słuchowa i polszczyzna OK ({winner['score']*100:.0f}%, pewność={winner['confidence']*100:.0f}%)"
            )

        # Commit winning thought SDR to cortex.last_active_sdr so 2D graph highlights the final thought!
        cortex.last_active_sdr.copy_(winner["thought"].z_star[-1].detach())
        cortex.last_sparsity_pct = winner["thought"].sparsity_pct
        cortex.last_attractor_steps = winner["thought"].attractor_steps

        critique_notes.append(
            f"Samokrytyka wybrała ścieżkę „{winner['label']}” ({winner['score']*100:.0f}%, pewność={winner['confidence']*100:.0f}%) z {len(candidates)} kroków."
        )

        return {
            "w_out": w_out,
            "thought": winner["thought"],
            "draft": winner["draft"],
            "temperature": winner["temperature"],
            "score": winner["score"],
            "confidence": winner["confidence"],
            "avg_nll": winner["avg_nll"],
            "quality": winner["quality"],
            "needs_correction": bool(winner["needs_correction"]),
            "label": winner["label"],
            "self_corrected": self_corrected,
            "reasoning_steps": len(thoughts),
            "critique_notes": critique_notes,
            "candidates": [
                {
                    "label": c["label"],
                    "score": c["score"],
                    "confidence": c["confidence"],
                    "avg_nll": c["avg_nll"],
                    "quality": c["quality"],
                    "alignment": c["alignment"],
                    "temperature": c["temperature"],
                    "thought_slots": c["thought_slots"],
                    "context_slots": c["context_slots"],
                }
                for c in candidates
            ],
        }

    @torch.no_grad()
    def think_and_stream(
        self,
        user_message: str,
        cortex: BiologicalAssociativeCortex,
        engram_bank: torch.Tensor | None,
        temperature: float | None = None,
        max_new_bytes: int | None = None,
        working_memory_sdr: torch.Tensor | None = None,
        working_memory_weight: float = 0.0,
        working_memory_slots: torch.Tensor | None = None,
    ) -> tuple[WernickeOutput, CortexThought, Iterator[str]]:
        """
        Full zero-token cognitive pathway:
          UTF-8 bytes -> Wernicke (patch_columns, S, u_sdr) -> BiologicalAssociativeCortex (Z_ctx) -> Broca stream(Z_ctx).
        """
        with self._op_lock:
            w_out = self._wernicke_raw.encode_texts([user_message], device=self.device, batch_size=1)
            thought = cortex.perceive_and_think(
                sensory_columns=w_out.columns,
                sensory_sdr=w_out.sdr,
                engram_bank=engram_bank,
                sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
                steps=self.cfg.attractor_steps,
                noise_std=0.0,
                update_last_active=True,
                working_memory_sdr=working_memory_sdr,
                working_memory_weight=working_memory_weight,
                sensory_patch_columns=w_out.patch_columns,
                sensory_patch_mask=w_out.patch_mask,
                working_memory_slots=working_memory_slots,
            )

        ctx_cols = thought.context_columns if thought.context_columns is not None else thought.columns_star

        def _stream_gen() -> Iterator[str]:
            with self._op_lock:
                yield from self._broca_raw.generate_stream(
                    memory_columns=ctx_cols,
                    max_new_bytes=max_new_bytes or self.cfg.broca_max_new_bytes,
                    temperature=temperature,
                    top_p=self.cfg.broca_top_p,
                    repetition_penalty=self.cfg.broca_repetition_penalty,
                    memory_mask=thought.context_mask,
                )

        return w_out, thought, _stream_gen()

    # ── Denoising Autoencoder Training through k-WTA SDR Bottleneck ──────────
    def _build_optimizer(self, lr: float, cond_lr_mult: float = 1.0) -> torch.optim.Optimizer:
        # Keep zero weight decay on Z* conditioning & Wernicke bottleneck pathways
        # so AdamW NEVER decays Z* conditioning or column/SDR projections toward zero.
        cond_keys = (
            "cross_", "film_", "bos_z_", "out_z_", "z_col_", "z_summary_", "sdr_expand_",
            "col_queries", "col_q_proj", "col_kv_proj", "col_out_proj", "col_compress", "to_sdr",
        )
        decay_params: list[torch.nn.Parameter] = []
        norm_params: list[torch.nn.Parameter] = []
        cond_params: list[torch.nn.Parameter] = []
        for mod in (self._wernicke_raw, self._broca_raw):
            for name, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                if any(k in name for k in cond_keys):
                    cond_params.append(p)
                elif p.ndim < 2 or "norm" in name or "router" in name:
                    norm_params.append(p)
                else:
                    decay_params.append(p)
        groups = [
            {"params": decay_params, "lr": lr, "weight_decay": float(self.cfg.broca_weight_decay)},
            {"params": norm_params, "lr": lr, "weight_decay": 0.0},
            {"params": cond_params, "lr": lr * float(cond_lr_mult), "weight_decay": 0.0},
        ]
        use_fused = self.device.type == "cuda"
        try:
            return torch.optim.AdamW(groups, lr=lr, fused=use_fused)
        except Exception:
            return torch.optim.AdamW(groups, lr=lr, fused=False)

    @staticmethod
    def _z_decorrelation_loss(z_cols: torch.Tensor) -> torch.Tensor:
        """Lightweight batch decorrelation penalty preventing Z* collapse across distinct inputs."""
        b = z_cols.shape[0]
        if b < 2:
            return torch.zeros((), device=z_cols.device, dtype=torch.float32)
        flat = F.normalize(z_cols.float().reshape(b, -1), dim=-1, eps=1e-6)
        sim = flat @ flat.T
        eye = torch.eye(b, device=z_cols.device, dtype=torch.bool)
        off = sim.masked_fill(eye, 0.0)
        return 0.65 * (off - 0.20).clamp(min=0.0).pow(2).mean()

    def _wernicke_bottleneck_loss(self, w_out: WernickeOutput, z_star: torch.Tensor) -> torch.Tensor:
        """Preserve sharp k-WTA SDR & column diversity even when patch_columns is present in Z_ctx."""
        dec_z = self._z_decorrelation_loss(z_star)
        dec_cols = self._z_decorrelation_loss(w_out.columns)
        dec_sdr = self._z_decorrelation_loss(w_out.sdr)
        align_recon = 0.25 * F.mse_loss(w_out.reconstructed_columns.float(), w_out.columns.detach().float())
        return dec_z + dec_cols + 0.5 * dec_sdr + align_recon

    def _train_denoising_epochs_unlocked(
        self,
        texts: list[str],
        epochs: int,
        lr: float,
        batch_size: int,
        early_stop_loss: float,
        mask_prob: float = 0.18,
    ) -> dict[str, Any]:
        clean_texts = [t.strip() for t in texts if t and len(t.strip()) >= 8]
        if not clean_texts:
            return {
                "trained": False,
                "samples": 0,
                "epochs_run": 0,
                "loss_start": round(self._broca_raw.last_loss, 4),
                "loss_end": round(self._broca_raw.last_loss, 4),
                "fluency": self._broca_raw.fluency_score,
                "duration_ms": 0.0,
            }

        t0 = time.perf_counter()
        self._wernicke_raw.train()
        self._broca_raw.train()
        opt = self._build_optimizer(lr=lr)

        loss_start: float | None = None
        loss_end: float = self._broca_raw.last_loss
        epochs_run = 0
        bs = max(1, int(batch_size))
        is_cuda = self.device.type == "cuda"
        ep_loss_gpu = torch.zeros((), device=self.device, dtype=torch.float32)

        try:
            for _ in range(max(1, int(epochs))):
                epochs_run += 1
                order = list(clean_texts)
                random.shuffle(order)
                ep_loss_gpu.zero_()
                ep_batches = 0

                for i in range(0, len(order), bs):
                    batch = order[i : i + bs]
                    in_bytes, in_mask = encode_utf8_bytes(
                        batch,
                        max_bytes=self.cfg.max_input_bytes,
                        patch_size=self.cfg.patch_stride,
                        add_bos_eos=False,
                        mask_prob=mask_prob,
                        device=self.device,
                        pin_memory=is_cuda,
                    )
                    tgt_bytes, _ = encode_utf8_bytes(
                        batch,
                        max_bytes=self.cfg.broca_max_seq_len,
                        patch_size=1,
                        add_bos_eos=True,
                        mask_prob=0.0,
                        device=self.device,
                        pin_memory=is_cuda,
                    )
                    opt.zero_grad(set_to_none=True)
                    with amp_autocast(self.device):
                        w_out = self.wernicke(in_bytes, in_mask)
                        z_cols = self._normalize_z_columns(w_out.reconstructed_columns)
                        if w_out.patch_columns is not None and w_out.patch_mask is not None:
                            z_ctx = self._normalize_z_columns(torch.cat([z_cols, w_out.patch_columns], dim=1))
                            prim_m = torch.ones((z_cols.shape[0], z_cols.shape[1]), dtype=torch.bool, device=self.device)
                            z_ctx_mask = torch.cat([prim_m, w_out.patch_mask], dim=1)
                        else:
                            z_ctx = z_cols
                            z_ctx_mask = None
                        loss = self._broca_raw.compute_loss(tgt_bytes, z_ctx, memory_mask=z_ctx_mask) + self._wernicke_bottleneck_loss(w_out, z_cols)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self._wernicke_raw.parameters()) + list(self._broca_raw.parameters()),
                        max_norm=1.0,
                    )
                    opt.step()

                    ep_loss_gpu += loss.detach().float()
                    ep_batches += 1
                    self._broca_raw.trained_steps += 1

                loss_end = float((ep_loss_gpu / float(max(1, ep_batches))).item())
                if loss_start is None:
                    loss_start = loss_end
                self._broca_raw._update_fluency(loss_end)
                # Early stopping around 0.85-1.0 to avoid 1:1 memorization
                if loss_end <= early_stop_loss:
                    break
        finally:
            self._wernicke_raw.eval()
            self._broca_raw.eval()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._refresh_status()
        return {
            "trained": True,
            "samples": len(clean_texts),
            "epochs_run": epochs_run,
            "loss_start": round(float(loss_start if loss_start is not None else loss_end), 4),
            "loss_end": round(float(loss_end), 4),
            "fluency": self._broca_raw.fluency_score,
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def train_denoising_sleep(
        self,
        texts: list[str],
        epochs: int | None = None,
        cortex: BiologicalAssociativeCortex | None = None,
    ) -> dict[str, Any]:
        """
        Sleep consolidation for Broca that NEVER resets or degrades dialogue tuning:
        - When dialogue Q&A has been trained (`_known_q_sdrs` populated), replays online-learned Q&A
          + Polish dialogues gently at low LR so sleep strengthens dialogue instead of overwriting it.
        """
        from .language_trainer import BASE_HUMAN_DIALOGUES

        fresh = [t.strip() for t in texts if t and len(t.strip()) >= 12]
        if not fresh and not self._learned_qa_replay:
            return {
                "trained": False,
                "samples": 0,
                "epochs_run": 0,
                "loss_start": round(self._broca_raw.last_loss, 4),
                "loss_end": round(self._broca_raw.last_loss, 4),
                "fluency": self._broca_raw.fluency_score,
                "duration_ms": 0.0,
            }

        if cortex is not None and self._known_q_sdrs.shape[0] > 0:
            pinned_replay = [p for p in self._learned_qa_replay if p[0].lower() in self._pinned_qa_keys]
            unpinned_replay = [p for p in self._learned_qa_replay if p[0].lower() not in self._pinned_qa_keys][-40:]
            replay_pairs = pinned_replay + unpinned_replay + random.sample(
                BASE_HUMAN_DIALOGUES, k=min(48, len(BASE_HUMAN_DIALOGUES))
            )
            for txt in fresh[:16]:
                replay_pairs.append((txt, txt))
            return self.train_dialogue_finetune(
                pairs=replay_pairs,
                cortex=cortex,
                engram_bank=None,
                epochs=min(3, max(1, int(epochs or 2))),
                lr=1.5e-4,
                batch_size=32,
                label_smoothing=0.05,
                early_stop_loss=0.52,
                decoder_byte_dropout=0.08,
                cortex_noise_std=0.01,
                register_questions=False,
            )

        replay_pool = [a for _, a in BASE_HUMAN_DIALOGUES]
        replay_k = min(len(replay_pool), max(16, len(fresh) * 3))
        mixed = fresh + random.sample(replay_pool, k=replay_k)
        with self._op_lock:
            res = self._train_denoising_epochs_unlocked(
                texts=mixed,
                epochs=min(2, int(epochs if epochs is not None else self.cfg.cortex_sleep_epochs)),
                lr=min(1.5e-4, self.cfg.broca_lr * 0.15),
                batch_size=self.cfg.broca_batch_size,
                early_stop_loss=self.cfg.broca_early_stop_loss,
                mask_prob=0.14,
            )
            if res["trained"]:
                self.lora_updates_count += 1
            return res

    def adapt_qa_pairs_online(
        self,
        pairs: list[tuple[str, str]],
        cortex: BiologicalAssociativeCortex,
        steps: int = 18,
        pinned: bool = False,
        lightweight: bool = False,
    ) -> float | None:
        """
        Fast (~180 ms on RTX 5080) ZERO-FORGETTING online Broca synaptic plasticity for newly taught
        facts or personal conversational updates (e.g. brain's name, user's name, Auto-sieć definitions).
        Protections against catastrophic forgetting:
          1. Keeps Wernicke FROZEN so existing SDRs in Hippocampus and `_known_q_sdrs` NEVER drift.
          2. Mixes >= 64 diverse dialogue + curriculum + pinned + uncertainty replay pairs (~85% of every batch).
          3. Applies EWC-style L2 anchor regularization toward pre-update Broca weights so unrelated
             Q&A answers (e.g. 'co to jest kot', 'co to jest fotosynteza') stay 100% intact.
          4. Pinned entries (`pinned=True`) can NEVER be evicted from `_learned_qa_replay`.
        """
        clean_new = [
            (q.strip(), a.strip()[:210])
            for q, a in pairs
            if q and a and len(q.strip()) >= 2 and len(a.strip()) >= 6 and not is_ai_corporate_reply(a)
        ]
        if not clean_new:
            return None

        replay_pool = build_dialogue_qa_pairs(include_curriculum=True, min_chars=2)
        new_q_low = {q.lower() for q, _ in clean_new}
        # Always include pinned core identity/rule pairs + recent unpinned taught facts + broad Q&A replay
        prior_pinned = [p for p in self._learned_qa_replay if p[0].lower() in self._pinned_qa_keys and p[0].lower() not in new_q_low][-24:]
        prior_unpinned = [p for p in self._learned_qa_replay if p[0].lower() not in self._pinned_qa_keys and p[0].lower() not in new_q_low][-12:]
        ood_anchors = [p for p in replay_pool if p[0].startswith("__")][:4]
        general_pool = [p for p in replay_pool if p[0].lower() not in new_q_low and not p[0].startswith("__")]
        replay = ood_anchors + prior_pinned + prior_unpinned + random.sample(general_pool, k=min(60, len(general_pool)))

        # Keep new pairs <= ~15% of batch so Broca adapts cleanly without overwriting global weights
        repeat_mult = 1 if lightweight else (2 if len(clean_new) <= 4 else 1)
        batch_pairs = clean_new * repeat_mult + replay

        with self._op_lock:
            self._wernicke_raw.eval()
            self._broca_raw.train()
            broca_decay, broca_no_decay, broca_cond = [], [], []
            cond_keys = ("cross_attn", "film", "bos_z_proj", "out_z_proj", "col_norm", "z_col_compress", "z_summary_proj")
            for name, p in self._broca_raw.named_parameters():
                if not p.requires_grad:
                    continue
                if any(k in name for k in cond_keys):
                    broca_cond.append(p)
                elif p.ndim < 2 or "norm" in name or "router" in name:
                    broca_no_decay.append(p)
                else:
                    broca_decay.append(p)
            # Snapshot pre-update weights for EWC-style L2 anchor penalty (zero catastrophic forgetting)
            anchor_params = [(p, p.detach().clone()) for p in self._broca_raw.parameters() if p.requires_grad]

            kwargs: dict[str, Any] = {}
            if self.device.type == "cuda":
                kwargs["fused"] = True
            lr_scale = 0.45 if lightweight else 1.0
            opt = torch.optim.AdamW(
                [
                    {"params": broca_decay, "lr": 3.5e-4 * lr_scale, "weight_decay": 0.01},
                    {"params": broca_no_decay, "lr": 3.5e-4 * lr_scale, "weight_decay": 0.0},
                    {"params": broca_cond, "lr": 1.8e-3 * lr_scale, "weight_decay": 0.0},
                ],
                **kwargs,
            )

            qs = [x[0] for x in batch_pairs]
            ans = [x[1] for x in batch_pairs]
            n_new_rows = len(clean_new) * repeat_mult
            is_cuda = self.device.type == "cuda"
            in_bytes, in_mask = encode_utf8_bytes(
                qs,
                max_bytes=self.cfg.max_input_bytes,
                patch_size=self.cfg.patch_stride,
                add_bos_eos=False,
                mask_prob=0.0,
                device=self.device,
                pin_memory=is_cuda,
            )
            tgt_bytes, _ = encode_utf8_bytes(
                ans,
                max_bytes=self.cfg.broca_max_seq_len,
                patch_size=1,
                add_bos_eos=True,
                mask_prob=0.0,
                device=self.device,
                pin_memory=is_cuda,
            )

            with torch.no_grad():
                w_out = self._wernicke_raw(in_bytes, in_mask)
                thought = cortex.perceive_and_think(
                    sensory_columns=w_out.columns,
                    sensory_sdr=w_out.sdr,
                    engram_bank=None,
                    sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
                    steps=self.cfg.attractor_steps,
                    noise_std=0.0,
                    update_last_active=False,
                    sensory_patch_columns=w_out.patch_columns,
                    sensory_patch_mask=w_out.patch_mask,
                )
                z_star = (thought.context_columns if thought.context_columns is not None else thought.columns_star).detach()
                z_mask = thought.context_mask.detach() if thought.context_mask is not None else None

            broca_params = list(self._broca_raw.parameters())
            min_steps = 4 if lightweight else 16
            max_steps = max(min_steps, min(64, int(steps)))
            ewc_weight = 0.012 if lightweight else 0.004
            last_new_loss = 5.0
            last_new_t = torch.tensor(5.0, device=self.device, dtype=torch.float32)
            moe_aux_coef = float(getattr(self.cfg, "broca_moe_aux_loss_coef", 0.01)) if self._broca_raw.use_moe else 0.0
            moe_z_coef = float(getattr(self.cfg, "broca_moe_z_loss_coef", 0.001)) if self._broca_raw.use_moe else 0.0
            try:
                for step_i in range(max_steps):
                    opt.zero_grad(set_to_none=True)
                    with amp_autocast(self.device):
                        ce_new, aux_new, z_new = self._broca_raw.compute_loss(
                            tgt_bytes[:n_new_rows],
                            z_star[:n_new_rows],
                            label_smoothing=0.03,
                            decoder_byte_dropout=0.02,
                            clean_z=True,
                            memory_mask=z_mask[:n_new_rows] if z_mask is not None else None,
                            return_components=True,
                        )
                        ce_rep, aux_rep, z_rep = self._broca_raw.compute_loss(
                            tgt_bytes[n_new_rows:],
                            z_star[n_new_rows:],
                            label_smoothing=0.05,
                            decoder_byte_dropout=0.05,
                            clean_z=True,
                            memory_mask=z_mask[n_new_rows:] if z_mask is not None else None,
                            return_components=True,
                        )
                        loss_new = ce_new + moe_aux_coef * aux_new + moe_z_coef * z_new
                        loss_rep = ce_rep + moe_aux_coef * aux_rep + moe_z_coef * z_rep
                        ewc_pen = sum((p - p0).pow(2).sum() for p, p0 in anchor_params)
                        loss = 0.65 * loss_new + 0.35 * loss_rep + ewc_weight * ewc_pen
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(broca_params, max_norm=1.0)
                    opt.step()
                    self._broca_raw.trained_steps += 1
                    last_new_t = ce_new.detach().float()
                    if ((step_i + 1) % 4 == 0 or step_i + 1 == max_steps) and float(last_new_t.item()) <= 0.28:
                        break
                last_new_loss = float(last_new_t.item())
            finally:
                self._broca_raw.eval()

            self._register_known_questions([q for q, _ in clean_new], replace=False)
            if pinned:
                for q, _ in clean_new:
                    self._pinned_qa_keys.add(q.lower())
            # Deduplicate replay buffer by lowercase question and NEVER evict pinned core entries!
            existing = [p for p in self._learned_qa_replay if p[0].lower() not in new_q_low]
            merged = existing + clean_new
            pinned_entries = [p for p in merged if p[0].lower() in self._pinned_qa_keys]
            unpinned_entries = [p for p in merged if p[0].lower() not in self._pinned_qa_keys]
            rem_slots = max(0, 200 - len(pinned_entries))
            self._learned_qa_replay = pinned_entries + (unpinned_entries[-rem_slots:] if rem_slots > 0 else [])
            return round(last_new_loss, 4)

    def adapt_qa_online(
        self,
        question: str,
        answer: str,
        cortex: BiologicalAssociativeCortex,
        steps: int = 56,
        pinned: bool = False,
        lightweight: bool = False,
    ) -> float:
        """Single-pair wrapper around zero-drift online Broca adaptation."""
        res = self.adapt_qa_pairs_online(
            [(question, answer)],
            cortex=cortex,
            steps=steps,
            pinned=pinned,
            lightweight=lightweight,
        )
        return float(res) if res is not None else 5.0

    def _normalize_z_columns(self, cols: torch.Tensor) -> torch.Tensor:
        """Unify Z* RMS scale across training and BiologicalAssociativeCortex.perceive_and_think()."""
        if hasattr(F, "rms_norm"):
            return F.rms_norm(cols, (self.cfg.column_dim,), eps=1e-6)
        return cols * torch.rsqrt(cols.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def train_dialogue_finetune(
        self,
        pairs: list[tuple[str, str]],
        cortex: BiologicalAssociativeCortex,
        engram_bank: torch.Tensor | None = None,
        epochs: int = 120,
        lr: float | None = None,
        batch_size: int | None = None,
        label_smoothing: float | None = None,
        early_stop_loss: float | None = None,
        decoder_byte_dropout: float | None = None,
        cortex_noise_std: float = 0.015,
        register_questions: bool = True,
        progress_callback: Any | None = None,
        stop_event: threading.Event | None = None,
        priority_probe_pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Tokenless Dialogue + Q&A Fine-Tuning through the full cognitive pipeline:
          Wernicke(question) -> BiologicalAssociativeCortex (attractor + noise, @torch.no_grad())
            -> RMSNorm(Z*) -> BrocaByteTransformer(answer)
        - Accepts short inputs down to 2 characters (NO 60-char floor!).
        - Uses differential LR (2x on Z* conditioning pathways: cross-attn, FiLM, BOS/out bridges)
          + cosine decay + label_smoothing=0.05 + decoder byte dropout so Broca reliably answers
          the user's actual question conditioned on Z*.
        """
        import math

        clean_pairs = [
            (q.strip(), a.strip())
            for q, a in pairs
            if q and a and len(q.strip()) >= 2 and len(a.strip()) >= 2
        ]
        if not clean_pairs:
            return {
                "trained": False,
                "samples": 0,
                "dialogues": 0,
                "epochs_run": 0,
                "loss_start": round(self._broca_raw.last_loss, 4),
                "loss_end": round(self._broca_raw.last_loss, 4),
                "fluency": self._broca_raw.fluency_score,
                "duration_ms": 0.0,
            }

        t0 = time.perf_counter()
        eff_lr = float(lr if lr is not None else self.cfg.broca_dialogue_lr)
        eff_ls = float(label_smoothing if label_smoothing is not None else self.cfg.broca_dialogue_label_smoothing)
        eff_stop = float(early_stop_loss if early_stop_loss is not None else self.cfg.broca_dialogue_early_stop_loss)
        eff_drop = float(0.08 if decoder_byte_dropout is None else decoder_byte_dropout)
        is_cuda = self.device.type == "cuda"
        # VRAM-aware GPU batch sizing (stay well below WDDM paging cliff while saturating Tensor Cores!)
        if batch_size is not None:
            bs = int(batch_size)
        elif is_cuda:
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
            except Exception:
                vram_gb = 12.0
            # Keep peak activations safely inside physical 16 GB VRAM (zero WDDM RAM paging!)
            bs = 48 if vram_gb >= 14.0 else 32
        else:
            bs = 24
        bs = max(2, bs)
        total_epochs = max(1, int(epochs))
        all_params = list(self._wernicke_raw.parameters()) + list(self._broca_raw.parameters())

        # Pre-encode all UTF-8 byte sequences ONCE on CPU before the epoch loop
        # (eliminates 160x redundant Python string encoding + regex diacritic stripping!).
        import numpy as np
        from .sensory_wernicke import BOS_BYTE, EOS_BYTE, MASK_BYTE, PAD_BYTE

        in_cap = max(1, int(self.cfg.max_input_bytes))
        tgt_cap = max(1, int(self.cfg.broca_max_seq_len) - 2)
        patch_stride = max(1, int(self.cfg.patch_stride))

        pre_q_orig: list[np.ndarray] = []
        pre_q_alt: list[np.ndarray] = []
        pre_ans: list[np.ndarray] = []
        pre_ans_lens: list[int] = []
        for raw_q, raw_a in clean_pairs:
            qb = raw_q.encode("utf-8")[:in_cap]
            pre_q_orig.append(np.frombuffer(qb, dtype=np.uint8))
            if not raw_q.startswith("__"):
                alt_s = _strip_pl(raw_q).rstrip("?!.").strip()
                alt_b = (alt_s if len(alt_s) >= 2 else raw_q).encode("utf-8")[:in_cap]
                pre_q_alt.append(np.frombuffer(alt_b, dtype=np.uint8))
            else:
                pre_q_alt.append(pre_q_orig[-1])
            ab = raw_a.encode("utf-8")[:tgt_cap]
            pre_ans.append(np.frombuffer(ab, dtype=np.uint8))
            pre_ans_lens.append(len(ab))

        def _pack_preencoded_batch(
            idxs: list[int],
            use_alt_prob: float,
            mask_p: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            b_n = len(idxs)
            q_arrs = [
                pre_q_alt[i] if (use_alt_prob > 0.0 and random.random() < use_alt_prob) else pre_q_orig[i]
                for i in idxs
            ]
            a_arrs = [pre_ans[i] for i in idxs]
            q_max = max((len(x) for x in q_arrs), default=1)
            if patch_stride > 1:
                q_max = max(patch_stride, math.ceil(q_max / patch_stride) * patch_stride)
            q_max = min(in_cap, max(1, q_max))
            if patch_stride > 1 and q_max % patch_stride != 0:
                q_max = max(patch_stride, (q_max // patch_stride) * patch_stride)
            a_max = min(self.cfg.broca_max_seq_len, max(3, max((len(x) for x in a_arrs), default=1) + 2))

            q_np = np.full((b_n, q_max), PAD_BYTE, dtype=np.int64)
            a_np = np.full((b_n, a_max), PAD_BYTE, dtype=np.int64)
            for r, arr in enumerate(q_arrs):
                n = min(len(arr), q_max)
                if n > 0:
                    q_np[r, :n] = arr[:n]
            for r, arr in enumerate(a_arrs):
                n = min(len(arr), a_max - 2)
                a_np[r, 0] = BOS_BYTE
                if n > 0:
                    a_np[r, 1 : 1 + n] = arr[:n]
                a_np[r, 1 + n] = EOS_BYTE

            in_t = torch.from_numpy(q_np)
            mask_t = in_t.ne(PAD_BYTE)
            tgt_t = torch.from_numpy(a_np)
            if is_cuda:
                in_b = in_t.pin_memory().to(self.device, non_blocking=True)
                in_m = mask_t.pin_memory().to(self.device, non_blocking=True)
                tg_b = tgt_t.pin_memory().to(self.device, non_blocking=True)
            else:
                in_b = in_t.to(self.device)
                in_m = mask_t.to(self.device)
                tg_b = tgt_t.to(self.device)
            if mask_p > 0.0:
                corruptible = in_m & (in_b < 256)
                drop = (torch.rand(in_b.shape, device=self.device) < mask_p) & corruptible
                in_b = torch.where(drop, torch.full_like(in_b, MASK_BYTE), in_b)
            return in_b, in_m, tg_b

        with self._op_lock:
            self._wernicke_raw.train()
            self._broca_raw.train()
            cond_mult = 2.0
            opt = self._build_optimizer(lr=eff_lr, cond_lr_mult=cond_mult)

            loss_start: float | None = None
            loss_end: float = self._broca_raw.last_loss
            epochs_run = 0
            ep_loss_gpu = torch.zeros((), device=self.device, dtype=torch.float32)

            probe_pool = [
                (q.strip(), a.strip())
                for q, a in (priority_probe_pairs or [])
                if q and a and not q.strip().startswith("__") and 2 <= len(q.strip()) <= 180 and len(a.strip()) >= 4
            ]
            if not probe_pool:
                probe_pool = [
                    (q, a) for q, a in clean_pairs
                    if not q.startswith("__") and 2 <= len(q) <= 140 and len(a) >= 6
                ]
            if probe_pool:
                random.shuffle(probe_pool)
            probe_idx = 0
            last_probe_t = 0.0
            last_cb_t = 0.0

            def _emit_dialogue_progress(phase: str, ep_num: int, ep_tot: int, force_probe: bool = False) -> None:
                nonlocal probe_idx, last_probe_t, last_cb_t
                if progress_callback is None:
                    return
                now = time.perf_counter()
                do_probe = bool(probe_pool) and (
                    force_probe
                    or last_probe_t == 0.0
                    or (now - last_probe_t) >= 4.0
                )
                if not do_probe and (now - last_cb_t) < 0.55:
                    return
                sample_q = ""
                sample_target = ""
                sample_brain = ""
                if do_probe:
                    sample_q, sample_target = probe_pool[probe_idx % len(probe_pool)]
                    probe_idx += 1
                    self._wernicke_raw.eval()
                    self._broca_raw.eval()
                    try:
                        with torch.no_grad():
                            w_p = self.perceive_one(sample_q)
                            t_p = cortex.perceive_and_think(
                                sensory_columns=w_p.columns,
                                sensory_sdr=w_p.sdr,
                                engram_bank=engram_bank,
                                sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
                                steps=self.cfg.attractor_steps,
                                noise_std=0.0,
                                update_last_active=False,
                                sensory_patch_columns=w_p.patch_columns,
                                sensory_patch_mask=w_p.patch_mask,
                            )
                            gen_p = self.decode_columns(
                                t_p.context_columns if t_p.context_columns is not None else t_p.columns_star,
                                max_new_bytes=min(96, self.cfg.broca_max_seq_len),
                                temperature=0.55,
                                memory_mask=t_p.context_mask,
                            )
                            sample_brain = str(gen_p.get("text", "")).strip()
                    except Exception:
                        sample_brain = ""
                    finally:
                        self._wernicke_raw.train()
                        self._broca_raw.train()
                    last_probe_t = time.perf_counter()
                last_cb_t = time.perf_counter()
                try:
                    progress_callback(
                        {
                            "phase": phase,
                            "epoch": ep_num,
                            "total_epochs": ep_tot,
                            "loss_start": round(float(loss_start if loss_start is not None else loss_end), 4),
                            "loss_now": round(float(loss_end), 4),
                            "fluency": self._broca_raw.fluency_score,
                            "dialogues": len(clean_pairs),
                            "sample_q": sample_q,
                            "sample_target": sample_target,
                            "sample_brain": sample_brain,
                            "fresh_probe": do_probe,
                        }
                    )
                except Exception:
                    pass

            all_indices = list(range(len(clean_pairs)))
            # Keep at most 2 async batches queued on GPU so VRAM stays flat (zero WDDM RAM spill!)
            cuda_ev_ring = [torch.cuda.Event(), torch.cuda.Event()] if is_cuda else []
            ev_slot = 0
            ev_armed = 0
            try:
                for ep in range(total_epochs):
                    if stop_event is not None and stop_event.is_set():
                        break
                    epochs_run += 1
                    prog = float(ep) / float(max(1, total_epochs - 1))
                    scale = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))
                    opt.param_groups[0]["lr"] = eff_lr * scale
                    opt.param_groups[1]["lr"] = eff_lr * scale
                    opt.param_groups[2]["lr"] = eff_lr * cond_mult * scale

                    # Length-bucketed mini-batches cut wasted padding FLOPs by ~40% while shuffling batch order!
                    order_idx = sorted(all_indices, key=lambda idx: pre_ans_lens[idx] + random.randint(-8, 8))
                    batches = [order_idx[i : i + bs] for i in range(0, len(order_idx), bs)]
                    random.shuffle(batches)
                    ep_loss_gpu.zero_()
                    ep_batches = 0

                    for batch_idxs in batches:
                        if is_cuda and ev_armed >= len(cuda_ev_ring):
                            cuda_ev_ring[ev_slot].synchronize()
                        in_bytes, in_mask, tgt_bytes = _pack_preencoded_batch(batch_idxs, use_alt_prob=0.25, mask_p=0.04)

                        opt.zero_grad(set_to_none=True)
                        with amp_autocast(self.device):
                            w_out = self.wernicke(in_bytes, in_mask)
                            z_wernicke = self._normalize_z_columns(w_out.reconstructed_columns)

                            with torch.no_grad():
                                thought = cortex.perceive_and_think(
                                    sensory_columns=w_out.columns.detach(),
                                    sensory_sdr=w_out.sdr.detach(),
                                    engram_bank=engram_bank,
                                    sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
                                    steps=self.cfg.attractor_steps,
                                    noise_std=cortex_noise_std,
                                    update_last_active=False,
                                    sensory_patch_columns=w_out.patch_columns.detach() if w_out.patch_columns is not None else None,
                                    sensory_patch_mask=w_out.patch_mask.detach() if w_out.patch_mask is not None else None,
                                )
                                z_cortex = thought.columns_star.to(dtype=z_wernicke.dtype)

                            z_star = self._normalize_z_columns(z_wernicke + (z_cortex - z_wernicke.detach()))
                            use_patches = random.random() >= 0.30
                            z_ctx, z_ctx_mask = cortex.build_context_columns(
                                columns_star=z_star,
                                sensory_patch_columns=w_out.patch_columns if use_patches else None,
                                sensory_patch_mask=w_out.patch_mask if use_patches else None,
                                recalled_columns=thought.recalled_columns,
                            )
                            ce_loss, aux_loss, z_loss = self._broca_raw.compute_loss(
                                tgt_bytes,
                                z_ctx,
                                label_smoothing=eff_ls,
                                decoder_byte_dropout=eff_drop,
                                clean_z=True,
                                memory_mask=z_ctx_mask,
                                return_components=True,
                            )
                            moe_reg = (
                                float(getattr(self.cfg, "broca_moe_aux_loss_coef", 0.01)) * aux_loss
                                + float(getattr(self.cfg, "broca_moe_z_loss_coef", 0.001)) * z_loss
                            ) if self._broca_raw.use_moe else 0.0
                            loss = ce_loss + moe_reg + self._wernicke_bottleneck_loss(w_out, z_star)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                        opt.step()

                        # Accumulate on GPU (ZERO per-step .item() CUDA sync stalls!)
                        ep_loss_gpu += ce_loss.detach().float()
                        ep_batches += 1
                        self._broca_raw.trained_steps += 1
                        if is_cuda:
                            cuda_ev_ring[ev_slot].record()
                            ev_slot = (ev_slot + 1) % len(cuda_ev_ring)
                            ev_armed += 1

                    loss_end = float((ep_loss_gpu / float(max(1, ep_batches))).item())
                    if loss_start is None:
                        loss_start = loss_end
                    self._broca_raw._update_fluency(loss_end)
                    hit_stop = loss_end <= eff_stop
                    is_last = (ep + 1 >= total_epochs) or hit_stop
                    _emit_dialogue_progress(
                        "dialogue_train",
                        epochs_run,
                        total_epochs,
                        force_probe=(epochs_run == 1 or (is_last and (time.perf_counter() - last_probe_t) >= 0.8)),
                    )
                    if hit_stop:
                        break

                # Batched Predictive-Coding Hard-Example Scan (~25x faster than batch=1 loop!)
                if total_epochs >= 40 and len(clean_pairs) >= 40 and (stop_event is None or not stop_event.is_set()):
                    self._wernicke_raw.eval()
                    self._broca_raw.eval()
                    hard_indices: list[int] = []
                    eval_bs = bs if is_cuda else 16
                    from .sensory_wernicke import BYTE_VOCAB_SIZE
                    with torch.no_grad():
                        for i in range(0, len(all_indices), eval_bs):
                            chunk_idxs = all_indices[i : i + eval_bs]
                            in_b, in_m, tgt_b = _pack_preencoded_batch(chunk_idxs, use_alt_prob=0.0, mask_p=0.0)
                            with amp_autocast(self.device):
                                w_c = self._wernicke_raw(in_b, in_m)
                                t_c = cortex.perceive_and_think(
                                    sensory_columns=w_c.columns,
                                    sensory_sdr=w_c.sdr,
                                    engram_bank=None,
                                    sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
                                    steps=self.cfg.attractor_steps,
                                    noise_std=0.0,
                                    update_last_active=False,
                                    sensory_patch_columns=w_c.patch_columns,
                                    sensory_patch_mask=w_c.patch_mask,
                                )
                                eval_ctx = t_c.context_columns if t_c.context_columns is not None else t_c.columns_star
                                logits = self._broca_raw.forward(
                                    tgt_b[:, :-1],
                                    eval_ctx,
                                    clean_z=True,
                                    memory_mask=t_c.context_mask,
                                )
                            tgt_next = tgt_b[:, 1:]
                            per_tok = F.cross_entropy(
                                logits.float().reshape(-1, BYTE_VOCAB_SIZE),
                                tgt_next.reshape(-1),
                                ignore_index=PAD_BYTE,
                                reduction="none",
                            ).view(tgt_next.shape)
                            valid_tok = (tgt_next != PAD_BYTE).float()
                            row_nlls = (per_tok * valid_tok).sum(dim=1) / valid_tok.sum(dim=1).clamp(min=1.0)
                            for j, nll_v in enumerate(row_nlls.tolist()):
                                if nll_v > 0.50:
                                    hard_indices.append(chunk_idxs[j])

                    if hard_indices:
                        self._wernicke_raw.train()
                        self._broca_raw.train()
                        opt_p = self._build_optimizer(lr=eff_lr * 0.45, cond_lr_mult=2.0)
                        p_loss_gpu = torch.zeros((), device=self.device, dtype=torch.float32)
                        for p_ep in range(28):
                            if stop_event is not None and stop_event.is_set():
                                break
                            replay_k = min(len(all_indices), max(32, len(hard_indices)))
                            polish_order = sorted(
                                hard_indices + random.sample(all_indices, k=replay_k),
                                key=lambda idx: pre_ans_lens[idx],
                            )
                            p_batches = [polish_order[i : i + bs] for i in range(0, len(polish_order), bs)]
                            random.shuffle(p_batches)
                            p_loss_gpu.zero_()
                            p_cnt = 0
                            for p_batch_idxs in p_batches:
                                in_bytes, in_mask, tgt_bytes = _pack_preencoded_batch(
                                    p_batch_idxs, use_alt_prob=0.0, mask_p=0.02
                                )
                                opt_p.zero_grad(set_to_none=True)
                                with amp_autocast(self.device):
                                    w_out = self.wernicke(in_bytes, in_mask)
                                    z_wernicke = self._normalize_z_columns(w_out.reconstructed_columns)
                                    with torch.no_grad():
                                        thought = cortex.perceive_and_think(
                                            sensory_columns=w_out.columns.detach(),
                                            sensory_sdr=w_out.sdr.detach(),
                                            engram_bank=engram_bank,
                                            sdr_to_columns_fn=self._wernicke_raw.sdr_to_columns,
                                            steps=self.cfg.attractor_steps,
                                            noise_std=0.005,
                                            update_last_active=False,
                                            sensory_patch_columns=w_out.patch_columns.detach() if w_out.patch_columns is not None else None,
                                            sensory_patch_mask=w_out.patch_mask.detach() if w_out.patch_mask is not None else None,
                                        )
                                        z_cortex = thought.columns_star.to(dtype=z_wernicke.dtype)
                                    z_star = self._normalize_z_columns(z_wernicke + (z_cortex - z_wernicke.detach()))
                                    use_patches = random.random() >= 0.30
                                    z_ctx, z_ctx_mask = cortex.build_context_columns(
                                        columns_star=z_star,
                                        sensory_patch_columns=w_out.patch_columns if use_patches else None,
                                        sensory_patch_mask=w_out.patch_mask if use_patches else None,
                                        recalled_columns=thought.recalled_columns,
                                    )
                                    ce_loss, aux_loss, z_loss = self._broca_raw.compute_loss(
                                        tgt_bytes,
                                        z_ctx,
                                        label_smoothing=0.04,
                                        decoder_byte_dropout=0.04,
                                        clean_z=True,
                                        memory_mask=z_ctx_mask,
                                        return_components=True,
                                    )
                                    moe_reg = (
                                        float(getattr(self.cfg, "broca_moe_aux_loss_coef", 0.01)) * aux_loss
                                        + float(getattr(self.cfg, "broca_moe_z_loss_coef", 0.001)) * z_loss
                                    ) if self._broca_raw.use_moe else 0.0
                                    loss = ce_loss + moe_reg + self._wernicke_bottleneck_loss(w_out, z_star)
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                                opt_p.step()
                                p_loss_gpu += ce_loss.detach().float()
                                p_cnt += 1
                                self._broca_raw.trained_steps += 1
                            loss_end = float((p_loss_gpu / float(max(1, p_cnt))).item())
                            self._broca_raw._update_fluency(loss_end)
                            hit_p_stop = loss_end <= 0.36
                            _emit_dialogue_progress(
                                "dialogue_polish",
                                p_ep + 1,
                                28,
                                force_probe=(p_ep == 0 or hit_p_stop or p_ep == 27),
                            )
                            if hit_p_stop:
                                break
            finally:
                self._wernicke_raw.eval()
                self._broca_raw.eval()
                if is_cuda:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            if register_questions:
                self._register_known_questions([q for q, _ in clean_pairs], replace=True)
            self.lora_updates_count += 1
            self._refresh_status()

            return {
                "trained": True,
                "samples": len(clean_pairs),
                "dialogues": len(clean_pairs),
                "epochs_run": epochs_run,
                "loss_start": round(float(loss_start if loss_start is not None else loss_end), 4),
                "loss_end": round(float(loss_end), 4),
                "fluency": self._broca_raw.fluency_score,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    def train_language_cortex(
        self,
        extra_texts: list[str] | None = None,
        epochs: int | None = None,
        include_curriculum: bool = True,
        cortex: BiologicalAssociativeCortex | None = None,
        engram_bank: torch.Tensor | None = None,
        extra_qa_pairs: list[tuple[str, str]] | None = None,
        progress_callback: Any | None = None,
        stop_event: threading.Event | None = None,
        priority_probe_pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        if cortex is not None:
            merged_extra = list(self._learned_qa_replay)
            if extra_qa_pairs:
                merged_extra.extend(extra_qa_pairs)
            pairs = build_dialogue_qa_pairs(
                include_curriculum=include_curriculum,
                extra_pairs=merged_extra,
                min_chars=2,
            )
            return self.train_dialogue_finetune(
                pairs=pairs,
                cortex=cortex,
                engram_bank=engram_bank,
                epochs=int(epochs if epochs is not None else self.cfg.language_train_epochs),
                lr=self.cfg.broca_dialogue_lr,
                batch_size=None,
                label_smoothing=self.cfg.broca_dialogue_label_smoothing,
                early_stop_loss=self.cfg.broca_dialogue_early_stop_loss,
                progress_callback=progress_callback,
                stop_event=stop_event,
                priority_probe_pairs=priority_probe_pairs,
            )
        corpus = build_denoising_corpus(
            extra_texts=extra_texts,
            include_base=True,
            include_curriculum=include_curriculum,
            max_samples=1200,
        )
        with self._op_lock:
            res = self._train_denoising_epochs_unlocked(
                texts=corpus,
                epochs=int(epochs if epochs is not None else self.cfg.language_train_epochs),
                lr=self.cfg.broca_lr,
                batch_size=self.cfg.broca_batch_size,
                early_stop_loss=self.cfg.broca_early_stop_loss,
                mask_prob=0.18,
            )
            if res["trained"]:
                self.lora_updates_count += 1
            res["dialogues"] = res["samples"]
            return res

    def train_streaming_batches(
        self,
        chunk_iterable_factory: Any,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float | None = None,
        early_stop_loss: float | None = None,
        mask_prob: float = 0.18,
        max_steps_per_epoch: int | None = None,
    ) -> dict[str, Any]:
        """
        Streaming bfloat16 denoising training for large local UTF-8 text corpora (100-300 MB)
        without loading all text into RAM.
        """
        t0 = time.perf_counter()
        eff_lr = float(lr if lr is not None else self.cfg.broca_lr)
        stop_loss = float(early_stop_loss if early_stop_loss is not None else self.cfg.broca_early_stop_loss)
        bs = max(1, int(batch_size))

        with self._op_lock:
            self._wernicke_raw.train()
            self._broca_raw.train()
            opt = self._build_optimizer(lr=eff_lr)

            loss_start: float | None = None
            loss_end: float = self._broca_raw.last_loss
            total_samples = 0
            total_batches = 0
            epochs_run = 0

            try:
                for _ in range(max(1, int(epochs))):
                    epochs_run += 1
                    ep_loss_sum = 0.0
                    ep_batches = 0
                    batch_buf: list[str] = []

                    is_cuda = self.device.type == "cuda"

                    def _flush_batch(items: list[str]) -> float:
                        nonlocal loss_start, total_samples, total_batches
                        in_bytes, in_mask = encode_utf8_bytes(
                            items,
                            max_bytes=self.cfg.max_input_bytes,
                            patch_size=self.cfg.patch_stride,
                            add_bos_eos=False,
                            mask_prob=mask_prob,
                            device=self.device,
                            pin_memory=is_cuda,
                        )
                        tgt_bytes, _ = encode_utf8_bytes(
                            items,
                            max_bytes=self.cfg.broca_max_seq_len,
                            patch_size=1,
                            add_bos_eos=True,
                            mask_prob=0.0,
                            device=self.device,
                            pin_memory=is_cuda,
                        )
                        opt.zero_grad(set_to_none=True)
                        with amp_autocast(self.device):
                            w_out = self.wernicke(in_bytes, in_mask)
                            z_cols = self._normalize_z_columns(w_out.reconstructed_columns)
                            if w_out.patch_columns is not None and w_out.patch_mask is not None:
                                z_ctx = self._normalize_z_columns(torch.cat([z_cols, w_out.patch_columns], dim=1))
                                prim_m = torch.ones((z_cols.shape[0], z_cols.shape[1]), dtype=torch.bool, device=self.device)
                                z_ctx_mask = torch.cat([prim_m, w_out.patch_mask], dim=1)
                            else:
                                z_ctx = z_cols
                                z_ctx_mask = None
                            loss = self._broca_raw.compute_loss(tgt_bytes, z_ctx, memory_mask=z_ctx_mask)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(self._wernicke_raw.parameters()) + list(self._broca_raw.parameters()),
                            max_norm=1.0,
                        )
                        opt.step()

                        val = float(loss.detach().item())
                        if loss_start is None:
                            loss_start = val
                        total_samples += len(items)
                        total_batches += 1
                        self._broca_raw.trained_steps += 1
                        return val

                    stream: Iterable[str] = chunk_iterable_factory()
                    for chunk in stream:
                        batch_buf.append(chunk)
                        if len(batch_buf) >= bs:
                            ep_loss_sum += _flush_batch(batch_buf)
                            ep_batches += 1
                            batch_buf.clear()
                            if max_steps_per_epoch is not None and ep_batches >= max_steps_per_epoch:
                                break
                    if batch_buf and (max_steps_per_epoch is None or ep_batches < max_steps_per_epoch):
                        ep_loss_sum += _flush_batch(batch_buf)
                        ep_batches += 1
                        batch_buf.clear()

                    if ep_batches > 0:
                        loss_end = ep_loss_sum / ep_batches
                        self._broca_raw._update_fluency(loss_end)
                        if loss_end <= stop_loss:
                            break
            finally:
                self._wernicke_raw.eval()
                self._broca_raw.eval()

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            if total_batches > 0:
                self.lora_updates_count += 1
            self._refresh_status()

            return {
                "trained": total_batches > 0,
                "samples": total_samples,
                "batches": total_batches,
                "epochs_run": epochs_run,
                "loss_start": round(float(loss_start if loss_start is not None else loss_end), 4),
                "loss_end": round(float(loss_end), 4),
                "fluency": self._broca_raw.fluency_score,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

    def train_parquet_polish_stream(
        self,
        pair_iterable_factory: Any,
        max_steps: int = 3000,
        batch_size: int = 128,
        lr: float | None = None,
        early_stop_loss: float | None = None,
        mask_prob: float = 0.16,
        stop_event: threading.Event | None = None,
        progress_callback: Any | None = None,
        checkpoint_callback: Any | None = None,
        checkpoint_every: int = 500,
    ) -> dict[str, Any]:
        """
        High-throughput RTX 5080 bfloat16 streaming Polish grammar & syntax trainer:
          - Background producer thread overlaps Parquet IO + NumPy UTF-8 encoding + length-bucketing
            + pinned host memory transfers with GPU compute (zero CPU stalls!).
          - Length-bucketed batches cut wasted padding FLOPs by ~40% while shuffling bucket order.
          - Accumulates loss on GPU without per-step `.item()` CUDA sync barriers (syncs every 16 steps).
          - Trains WernickeByteCortex + BrocaByteTransformer on `(src, tgt)` pairs (65% denoising +
            35% next-sentence continuation) with linear warmup + cosine LR decay & AdamW(fused=True).
        """
        import math
        import queue

        t0 = time.perf_counter()
        bs = max(2, int(batch_size))
        default_lr = self.cfg.broca_lr * (1.35 if bs >= 128 else 1.0)
        base_lr = float(lr if lr is not None else default_lr)
        stop_loss = float(early_stop_loss if early_stop_loss is not None else self.cfg.broca_early_stop_loss)
        steps_cap = max(1, int(max_steps))
        warmup_steps = min(150, max(10, steps_cap // 12))
        is_cuda = self.device.type == "cuda"

        all_params = list(self._wernicke_raw.parameters()) + list(self._broca_raw.parameters())
        opt = self._build_optimizer(lr=base_lr)

        # Deeper async CPU prefetch queue with length-bucketing & pinned memory (zero GPU stalls!)
        prefetch_q: queue.Queue[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str, str] | None] = queue.Queue(maxsize=12)
        local_stop = threading.Event()
        pool_target = bs * 16

        def _producer() -> None:
            produced_steps = 0
            try:
                while produced_steps < steps_cap and not local_stop.is_set() and (stop_event is None or not stop_event.is_set()):
                    pool: list[tuple[str, str]] = []
                    any_item = False
                    for pair in pair_iterable_factory():
                        if local_stop.is_set() or (stop_event is not None and stop_event.is_set()):
                            break
                        any_item = True
                        pool.append(pair)
                        if len(pool) >= pool_target:
                            pool.sort(key=lambda x: len(x[1]))
                            chunks = [pool[i : i + bs] for i in range(0, len(pool), bs) if len(pool[i : i + bs]) == bs]
                            random.shuffle(chunks)
                            for ch in chunks:
                                if produced_steps >= steps_cap or local_stop.is_set() or (stop_event is not None and stop_event.is_set()):
                                    break
                                srcs = [x[0] for x in ch]
                                tgts = [x[1] for x in ch]
                                in_b, in_m = encode_utf8_bytes(
                                    srcs,
                                    max_bytes=self.cfg.max_input_bytes,
                                    patch_size=self.cfg.patch_stride,
                                    add_bos_eos=False,
                                    mask_prob=0.0,
                                    device=None,
                                    pin_memory=is_cuda,
                                )
                                tg_b, _ = encode_utf8_bytes(
                                    tgts,
                                    max_bytes=self.cfg.broca_max_seq_len,
                                    patch_size=1,
                                    add_bos_eos=True,
                                    mask_prob=0.0,
                                    device=None,
                                    pin_memory=is_cuda,
                                )
                                prefetch_q.put((in_b, in_m, tg_b, len(srcs), srcs[0], tgts[0]))
                                produced_steps += 1
                            pool.clear()
                            if produced_steps >= steps_cap:
                                break
                    if pool and produced_steps < steps_cap and not local_stop.is_set() and (stop_event is None or not stop_event.is_set()):
                        pool.sort(key=lambda x: len(x[1]))
                        for i in range(0, len(pool), bs):
                            ch = pool[i : i + bs]
                            if not ch or produced_steps >= steps_cap:
                                break
                            srcs = [x[0] for x in ch]
                            tgts = [x[1] for x in ch]
                            in_b, in_m = encode_utf8_bytes(
                                srcs,
                                max_bytes=self.cfg.max_input_bytes,
                                patch_size=self.cfg.patch_stride,
                                add_bos_eos=False,
                                mask_prob=0.0,
                                device=None,
                                pin_memory=is_cuda,
                            )
                            tg_b, _ = encode_utf8_bytes(
                                tgts,
                                max_bytes=self.cfg.broca_max_seq_len,
                                patch_size=1,
                                add_bos_eos=True,
                                mask_prob=0.0,
                                device=None,
                                pin_memory=is_cuda,
                            )
                            prefetch_q.put((in_b, in_m, tg_b, len(srcs), srcs[0], tgts[0]))
                            produced_steps += 1
                    if not any_item:
                        break
            finally:
                prefetch_q.put(None)

        prod_thread = threading.Thread(target=_producer, daemon=True)
        prod_thread.start()

        loss_start: float | None = None
        ema_loss: float | None = None
        total_samples = 0
        steps_done = 0
        last_sample_text = ""
        last_probe_prompt = ""
        last_probe_target = ""
        last_probe_t = 0.0
        sync_interval = 16
        loss_window = torch.zeros((), device=self.device, dtype=torch.float32)
        window_count = 0
        # 2-slot CUDA Event ring: bounds in-flight GPU queue to <= 2 steps so VRAM NEVER
        # snowballs into Windows WDDM Shared RAM (eliminates 5 GB -> 20 GB sawtooth spikes!)
        # while keeping 100% CPU/GPU pipeline overlap.
        cuda_ev_ring = [torch.cuda.Event(), torch.cuda.Event()] if is_cuda else []
        ev_slot = 0
        ev_armed = 0

        def _set_lr(step_i: int) -> float:
            if step_i < warmup_steps:
                cur_lr = base_lr * float(step_i + 1) / float(max(1, warmup_steps))
            else:
                prog = min(1.0, float(step_i - warmup_steps) / float(max(1, steps_cap - warmup_steps)))
                cur_lr = base_lr * (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            return cur_lr

        try:
            with self._op_lock:
                self._wernicke_raw.train()
                self._broca_raw.train()

            while steps_done < steps_cap:
                if stop_event is not None and stop_event.is_set():
                    break
                if is_cuda and ev_armed >= len(cuda_ev_ring):
                    cuda_ev_ring[ev_slot].synchronize()
                item = prefetch_q.get()
                if item is None:
                    break
                in_cpu, mask_cpu, tgt_cpu, batch_n, batch_src, batch_tgt = item
                cur_lr = _set_lr(steps_done)

                in_bytes = in_cpu.to(self.device, non_blocking=is_cuda)
                in_mask = mask_cpu.to(self.device, non_blocking=is_cuda)
                tgt_bytes = tgt_cpu.to(self.device, non_blocking=is_cuda)

                if mask_prob > 0.0:
                    corruptible = in_mask & (in_bytes < 256)
                    drop = (torch.rand(in_bytes.shape, device=self.device) < mask_prob) & corruptible
                    in_bytes = torch.where(drop, torch.full_like(in_bytes, 259), in_bytes)

                with self._op_lock:
                    opt.zero_grad(set_to_none=True)
                    with amp_autocast(self.device):
                        w_out = self.wernicke(in_bytes, in_mask)
                        z_cols = self._normalize_z_columns(w_out.reconstructed_columns)
                        if w_out.patch_columns is not None and w_out.patch_mask is not None:
                            z_ctx = self._normalize_z_columns(torch.cat([z_cols, w_out.patch_columns], dim=1))
                            prim_m = torch.ones((z_cols.shape[0], z_cols.shape[1]), dtype=torch.bool, device=self.device)
                            z_ctx_mask = torch.cat([prim_m, w_out.patch_mask], dim=1)
                        else:
                            z_ctx = z_cols
                            z_ctx_mask = None
                        ce_loss, aux_loss, z_loss = self._broca_raw.compute_loss(
                            tgt_bytes,
                            z_ctx,
                            memory_mask=z_ctx_mask,
                            return_components=True,
                        )
                        moe_reg = (
                            float(getattr(self.cfg, "broca_moe_aux_loss_coef", 0.01)) * aux_loss
                            + float(getattr(self.cfg, "broca_moe_z_loss_coef", 0.001)) * z_loss
                        ) if self._broca_raw.use_moe else 0.0
                        loss = ce_loss + moe_reg + self._wernicke_bottleneck_loss(w_out, z_cols)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    opt.step()
                    self._broca_raw.trained_steps += 1
                    loss_window += ce_loss.detach().float()
                    window_count += 1
                del in_bytes, in_mask, tgt_bytes, w_out, z_cols, z_ctx, z_ctx_mask, ce_loss, aux_loss, z_loss, moe_reg, loss
                if is_cuda:
                    cuda_ev_ring[ev_slot].record()
                    ev_slot = (ev_slot + 1) % len(cuda_ev_ring)
                    ev_armed += 1

                total_samples += batch_n
                steps_done += 1

                if steps_done == 1 or window_count >= sync_interval or steps_done >= steps_cap:
                    avg_val = float((loss_window / float(max(1, window_count))).item())
                    loss_window.zero_()
                    was_first_window = (steps_done > 1) and (ema_loss == loss_start)
                    window_count = 0
                    if loss_start is None:
                        loss_start = avg_val
                    if ema_loss is None or was_first_window:
                        ema_loss = avg_val
                    else:
                        ema_loss = 0.70 * ema_loss + 0.30 * avg_val
                    self._broca_raw._update_fluency(ema_loss)
                    self._refresh_status()

                    fresh_probe = False
                    now_p = time.perf_counter()
                    if (
                        steps_done == 16
                        or steps_done % 256 == 0
                        or steps_done == steps_cap
                        or (now_p - last_probe_t) >= 4.5
                    ):
                        probe_src = batch_src or "Język polski ma bogatą gramatykę i poprawną składnię."
                        probe_tgt = batch_tgt or probe_src
                        with self._op_lock:
                            self._wernicke_raw.eval()
                            self._broca_raw.eval()
                        try:
                            probe_out = self.perceive_one(probe_src)
                            p_cols = probe_out.reconstructed_columns
                            p_mask = None
                            if probe_out.patch_columns is not None and probe_out.patch_mask is not None:
                                p_cols = self._normalize_z_columns(
                                    torch.cat([p_cols.to(self.device), probe_out.patch_columns.to(self.device)], dim=1)
                                )
                                p_mask = torch.cat(
                                    [
                                        torch.ones((1, probe_out.reconstructed_columns.shape[1]), dtype=torch.bool, device=self.device),
                                        probe_out.patch_mask.to(self.device),
                                    ],
                                    dim=1,
                                )
                            gen = self.decode_columns(
                                p_cols,
                                max_new_bytes=90,
                                temperature=0.72,
                                memory_mask=p_mask,
                            )
                            last_sample_text = gen["text"]
                            last_probe_prompt = probe_src
                            last_probe_target = probe_tgt
                            fresh_probe = True
                            last_probe_t = time.perf_counter()
                        finally:
                            with self._op_lock:
                                self._wernicke_raw.train()
                                self._broca_raw.train()

                    if progress_callback is not None:
                        elapsed = max(0.001, time.perf_counter() - t0)
                        progress_callback(
                            {
                                "step": steps_done,
                                "max_steps": steps_cap,
                                "samples": total_samples,
                                "loss_start": round(float(loss_start or 0.0), 4),
                                "loss_now": round(float(ema_loss or 0.0), 4),
                                "fluency": self._broca_raw.fluency_score,
                                "lr": cur_lr,
                                "samples_per_sec": round(total_samples / elapsed, 1),
                                "sample_preview": last_sample_text,
                                "sample_prompt": last_probe_prompt if fresh_probe else batch_src,
                                "sample_target": last_probe_target if fresh_probe else batch_tgt,
                                "fresh_probe": fresh_probe,
                            }
                        )

                    if ema_loss is not None and steps_done >= warmup_steps and ema_loss <= stop_loss:
                        break

                if checkpoint_callback is not None and checkpoint_every > 0 and steps_done % checkpoint_every == 0:
                    checkpoint_callback()
        finally:
            local_stop.set()
            while not prefetch_q.empty():
                try:
                    prefetch_q.get_nowait()
                except queue.Empty:
                    break
            prod_thread.join(timeout=2.0)

            if window_count > 0:
                avg_val = float((loss_window / float(window_count)).item())
                if loss_start is None:
                    loss_start = avg_val
                ema_loss = avg_val if ema_loss is None else (0.80 * ema_loss + 0.20 * avg_val)
                self._broca_raw._update_fluency(ema_loss)

            with self._op_lock:
                self._wernicke_raw.eval()
                self._broca_raw.eval()
                if steps_done > 0:
                    self.lora_updates_count += 1
                self._refresh_status()

        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return {
            "trained": steps_done > 0,
            "steps": steps_done,
            "samples": total_samples,
            "loss_start": round(float(loss_start if loss_start is not None else self._broca_raw.last_loss), 4),
            "loss_end": round(float(ema_loss if ema_loss is not None else self._broca_raw.last_loss), 4),
            "fluency": self._broca_raw.fluency_score,
            "sample_preview": last_sample_text,
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # ── Minimal Query Cleaning & Sensory Splitting for brain/web_learner.py ──
    @staticmethod
    def normalize_study_queries(raw_topic: str) -> list[str]:
        q = raw_topic.strip()
        if not q:
            return []
        typo_rules = [
            (r"\bwzk", "wsk"),
            (r"\bwzp", "wsp"),
            (r"\bwzt", "wst"),
            (r"\bwskaznik([a-z]*)", r"wskaźnik\1"),
            (r"\bzolw([a-z]*)", r"żółw\1"),
            (r"\bzrodl([a-z]*)", r"źródł\1"),
            (r"\bpamiec([a-z]*)", r"pamięć\1"),
            (r"\bczastk([a-z]*)", r"cząstk\1"),
            (r"\brownan([a-z]*)", r"równan\1"),
            (r"\bsiec([a-z]*)", r"sieć\1"),
            (r"\bcpp\b", "C++"),
            (r"\bnode\.?ks\b|\bnodejs\b", "Node.js"),
            (r"\bcsharp\b", "C#"),
        ]
        fixed = q
        for pat, rep in typo_rules:
            fixed = re.sub(pat, rep, fixed, flags=re.I)
        fixed = re.sub(r"\s+", " ", fixed).strip()
        out = [fixed]
        if fixed.lower() != q.lower():
            out.append(q)
        return out

    @staticmethod
    def extract_sensory_stimuli(text: str, max_items: int = 6) -> list[str]:
        return split_clean_sentences(text, min_chars=24, max_chars=240)[:max_items]
