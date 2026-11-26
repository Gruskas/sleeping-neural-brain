from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import torch

from .config import BrainConfig


class LocalPolishBrocaLLM:
    """
    Hybrid Biological Broca Speech Cortex powered by a local uncensored/abliterated Polish LLM
    (default: `MassivDash/Bielik-4.5B-v3-heretic`, fallback: `huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated`)
    running natively in `bfloat16` on GPU (RTX 5080) with live `peft` LoRA sleep consolidation.

    Design principles:
      - Zero hardcoded canned answers in code: all replies are genuinely generated from the
        neural model conditioned on the biological brain's state, recalled Hippocampus SDR memories,
        REM creative associations, Prefrontal Working Memory conversation window, and creativity level.
      - True token-by-token real-time streaming via `TextIteratorStreamer` (zero artificial time.sleep).
      - During NREM sleep, trains the LoRA adapter (`brain_state/lora_cortex`) on newly consolidated
        memories & conversations so learned knowledge and personality persist in weights.
    """

    def __init__(
        self,
        cfg: BrainConfig,
        device: torch.device,
        lora_dir: Path,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.lora_dir = Path(lora_dir)
        self.model_id = os.environ.get("BROCA_LLM_MODEL", "").strip() or cfg.broca_llm_model_id

        self._lock = threading.RLock()
        self._load_thread: threading.Thread | None = None

        self.tokenizer: Any = None
        self.model: Any = None
        self.ready: bool = False
        self.loading: bool = False
        self.load_error: str | None = None
        self.status_message: str = f"Oczekiwanie na załadowanie lokalnego Broki ({self.model_id})..."
        self.total_params_count: int = 0
        self.trainable_lora_params: int = 0
        self.lora_updates_count: int = 0
        self.last_loss: float = 0.0

    def load_async(self) -> None:
        with self._lock:
            if self.ready or self.loading:
                return
            self.loading = True
            self.load_error = None
            self.status_message = f"Ładowanie lokalnego modelu Broki ({self.model_id}) na GPU (bfloat16)..."
            self._load_thread = threading.Thread(target=self.load_sync, daemon=True)
            self._load_thread.start()

    def load_sync(self) -> bool:
        with self._lock:
            if self.ready:
                return True
            self.loading = True
            self.load_error = None
            self.status_message = f"Ładowanie {self.model_id} + LoRA na GPU..."

        t0 = time.perf_counter()
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else (
                torch.float16 if self.device.type == "cuda" else torch.float32
            )

            # Prefer local cache first, then download primary model, then offline fallback if needed
            def _load_pair(mid: str) -> tuple[Any, Any]:
                try:
                    t_ = AutoTokenizer.from_pretrained(mid, local_files_only=True)
                    m_ = AutoModelForCausalLM.from_pretrained(
                        mid,
                        dtype=dtype,
                        attn_implementation="sdpa",
                        local_files_only=True,
                    )
                    return t_, m_
                except Exception:
                    t_ = AutoTokenizer.from_pretrained(mid)
                    m_ = AutoModelForCausalLM.from_pretrained(
                        mid,
                        dtype=dtype,
                        attn_implementation="sdpa",
                    )
                    return t_, m_

            try:
                tok, base_model = _load_pair(self.model_id)
            except Exception:
                fb = getattr(self.cfg, "broca_llm_fallback_model_id", "")
                if fb and fb != self.model_id:
                    self.model_id = fb
                    tok, base_model = _load_pair(self.model_id)
                else:
                    raise

            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            base_model.to(self.device)
            base_model.config.use_cache = True

            # Load persisted LoRA sleep adapter ONLY if compatible with the current base model & rank
            adapter_cfg_file = self.lora_dir / "adapter_config.json"
            model: Any = None
            if adapter_cfg_file.exists():
                try:
                    import json as _json
                    acfg = _json.loads(adapter_cfg_file.read_text(encoding="utf-8"))
                    saved_base = str(acfg.get("base_model_name_or_path", "")).strip()
                    saved_r = int(acfg.get("r", 0))
                    if saved_base == self.model_id and saved_r == int(self.cfg.broca_llm_lora_r):
                        model = PeftModel.from_pretrained(base_model, str(self.lora_dir), is_trainable=True)
                except Exception:
                    model = None

            if model is None:
                lora_cfg = LoraConfig(
                    r=int(self.cfg.broca_llm_lora_r),
                    lora_alpha=int(self.cfg.broca_llm_lora_alpha),
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                )
                model = get_peft_model(base_model, lora_cfg)

            model.eval()
            total_p = sum(p.numel() for p in model.parameters())
            train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            elapsed = time.perf_counter() - t0

            with self._lock:
                self.tokenizer = tok
                self.model = model
                self.total_params_count = int(total_p)
                self.trainable_lora_params = int(train_p)
                self.ready = True
                self.loading = False
                self.load_error = None
                short_name = self.model_id.split("/")[-1]
                self.status_message = (
                    f"Gotowy: {short_name} + LoRA ({total_p / 1e9:.2f}B param. · "
                    f"{train_p / 1e6:.2f}M LoRA · bf16 załadowano w {elapsed:.1f}s)"
                )
            return True
        except Exception as exc:
            with self._lock:
                self.ready = False
                self.loading = False
                self.load_error = str(exc)
                self.status_message = f"Błąd ładowania {self.model_id}: {exc}"
            return False

    def reset_adapter(self) -> None:
        """Reset only the LoRA sleep weights (keeps base model resident on GPU for instant reset)."""
        with self._lock:
            if self.lora_dir.exists():
                shutil.rmtree(self.lora_dir, ignore_errors=True)
            self.lora_updates_count = 0
            self.last_loss = 0.0
            if not self.ready or self.model is None:
                return
            try:
                from peft import LoraConfig, get_peft_model

                base = self.model.unload() if hasattr(self.model, "unload") else self.model.get_base_model()
                lora_cfg = LoraConfig(
                    r=int(self.cfg.broca_llm_lora_r),
                    lora_alpha=int(self.cfg.broca_llm_lora_alpha),
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                )
                self.model = get_peft_model(base, lora_cfg)
                self.model.eval()
            except Exception:
                pass

    def build_system_prompt(
        self,
        day: int,
        total_sleeps: int,
        creativity_level: float,
        recalled_semantic: list[str],
        recalled_hippo: list[str],
        recalled_rem: list[str],
        web_context: str = "",
        sdr_sparsity_pct: float = 4.0,
    ) -> str:
        """
        Build a clean, dynamic Polish cognitive context reflecting the brain's real biological state,
        recalled Hippocampus SDR memories, and live creativity setting — zero canned replies!
        """
        cre = max(0.0, min(1.0, float(creativity_level)))
        if cre >= 0.75:
            style_note = "Twój styl myślenia jest teraz twórczy, żywy, błyskotliwy i otwarty na ciekawe skojarzenia."
        elif cre <= 0.35:
            style_note = "Twój styl myślenia jest teraz rzeczowy, konkretny, precyzyjny i analityczny."
        else:
            style_note = "Twój styl myślenia łączy naturalną, luźną rozmowę z rzeczową, konkretną wiedzą."

        lines = [
            "Jesteś Sleeping Neural Brain — lokalnym, uczącym się mózgiem neuronowym działającym na karcie graficznej użytkownika.",
            "Twoja architektura składa się z kory odbiorczej Wernickego (bajty UTF-8 -> SDR k-WTA), kory asocjacyjnej Modern Hopfield, "
            "Hipokampu, pamięci roboczej SDR, cyklu snu NREM/SHY/REM oraz obszaru mowy Broki z adaptacją LoRA.",
            f"Stan biologiczny: Dzień {day}, przespanych nocy: {total_sleeps}, aktywność korowa SDR: {sdr_sparsity_pct:.2f}%, "
            f"poziom kreatywności: {cre * 100:.0f}%.",
            style_note,
            "",
            "Zasady naturalnej rozmowy:",
            "- Odpowiadaj zawsze po polsku — naturalnie, bezpośrednio, szczerze i z własną osobowością, bez sztucznego moralizowania, pouczania i sztywnych formułek.",
            "- Dopasowuj długość odpowiedzi do pytania: na krótkie zagajenie odpowiadaj luźno i zwięźle, a na złożone pytania, prośby o wyjaśnienia, opowiadania czy kod odpowiadaj wyczerpująco i bez sztucznego skracania.",
            "- Rozumiesz potoczny język, slang, przekleństwa, literówki (np. 'SUEPR', 'poweidz'), skróty i luźny ton — odpowiadaj na luzie, po ludzku i na temat.",
            "- Jeśli użytkownik nada Ci imię albo przedstawi się, zapamiętaj to i naturalnie tego używaj w rozmowie.",
            "- Korzystaj z przywołanych wspomnień z pamięci jako ze swojej wiedzy, ale ZAWSZE formułuj myśli własnymi słowami (nigdy nie wklejaj surowych notatek 1:1).",
            "- Jeśli czegoś nie wiesz albo użytkownik wpisze losowy ciąg znaków, zareaguj krótko i po ludzku — bez sztywnych komunikatów o komendach.",
        ]

        mem_blocks: list[str] = []
        if web_context:
            mem_blocks.append(f"[Świeżo sprawdzone z sieci]: {web_context}")
        if recalled_semantic:
            mem_blocks.append("[Pamięć trwała — przywołane ślady]:\n" + "\n".join(f"- {m}" for m in recalled_semantic[:12]))
        if recalled_hippo:
            mem_blocks.append("[Hipokamp — świeże wspomnienia z dziś]:\n" + "\n".join(f"- {m}" for m in recalled_hippo[:10]))
        if recalled_rem and cre >= 0.55:
            mem_blocks.append("[Skojarzenia ze snu REM]:\n" + "\n".join(f"- {m}" for m in recalled_rem[:6]))

        if mem_blocks:
            lines.append("")
            lines.append("Kontekst przywołany przez Twój Hipokamp i Korę:")
            lines.extend(mem_blocks)

        return "\n".join(lines)

    def _build_chat_messages(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        user_message: str,
        max_history_turns: int = 64,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for turn in history[-max_history_turns:]:
            role = turn.get("role", "")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content[:4000]})
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate_stream(
        self,
        user_message: str,
        history: list[dict[str, str]],
        system_prompt: str,
        creativity_level: float = 0.75,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        """
        True token-by-token real-time Polish streaming directly from the local GPU LLM.
        """
        if not self.ready or self.model is None or self.tokenizer is None:
            raise RuntimeError("Lokalny model Broki jeszcze się ładuje na GPU — poczekaj kilka sekund.")

        from transformers import TextIteratorStreamer

        cre = max(0.0, min(1.0, float(creativity_level)))
        temp = float(temperature) if temperature is not None else round(0.25 + 0.50 * cre, 3)
        temp = max(0.15, min(1.0, temp))
        top_p = round(0.80 + 0.12 * cre, 3)
        max_toks = max(64, min(4096, int(max_new_tokens or self.cfg.broca_llm_max_new_tokens)))

        messages = self._build_chat_messages(system_prompt, history, user_message)

        with self._lock:
            self.model.eval()
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=90.0,
            )
            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_toks,
                "do_sample": True,
                "temperature": temp,
                "top_p": top_p,
                "top_k": 40,
                "repetition_penalty": 1.08,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            err_box: list[Exception] = []

            def _worker() -> None:
                try:
                    with torch.inference_mode():
                        self.model.generate(**gen_kwargs)
                except Exception as exc:
                    err_box.append(exc)
                    try:
                        streamer.end()
                    except Exception:
                        pass

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            try:
                for text_chunk in streamer:
                    if text_chunk:
                        yield text_chunk
            finally:
                t.join(timeout=2.0)
            if err_box:
                raise err_box[0]

    def train_lora_sleep(
        self,
        facts: list[str],
        dialogues: list[tuple[str, str]] | None = None,
        epochs: int | None = None,
        lr: float | None = None,
    ) -> dict[str, Any]:
        """
        Real GPU LoRA consolidation during NREM Deep Sleep:
        trains the PEFT LoRA adapter (`q_proj`, `v_proj`) on the day's new consolidated memories
        and conversation turns in `bfloat16`, then saves `brain_state/lora_cortex/`.
        """
        t0 = time.perf_counter()
        if not self.ready or self.model is None or self.tokenizer is None:
            return {"trained": False, "samples": 0, "loss_start": 0.0, "loss_end": 0.0, "duration_ms": 0.0}

        samples: list[str] = []
        seen: set[str] = set()
        for f in (facts or []):
            clean = (f or "").strip()
            if len(clean) >= 14 and clean.lower() not in seen:
                seen.add(clean.lower())
                msgs = [
                    {"role": "system", "content": "Jesteś Sleeping Neural Brain. Utrwalasz wiedzę podczas snu NREM."},
                    {"role": "user", "content": "Opowiedz o tym, czego się dzisiaj nauczyłeś."},
                    {"role": "assistant", "content": clean[:640]},
                ]
                samples.append(self.tokenizer.apply_chat_template(msgs, tokenize=False))

        for u_txt, a_txt in (dialogues or []):
            u_c = (u_txt or "").strip()
            a_c = (a_txt or "").strip()
            if len(u_c) >= 2 and len(a_c) >= 6:
                key = f"{u_c.lower()}||{a_c.lower()}"
                if key not in seen:
                    seen.add(key)
                    msgs = [
                        {"role": "system", "content": "Jesteś Sleeping Neural Brain — naturalnym polskim mózgiem neuronowym."},
                        {"role": "user", "content": u_c[:512]},
                        {"role": "assistant", "content": a_c[:640]},
                    ]
                    samples.append(self.tokenizer.apply_chat_template(msgs, tokenize=False))

        samples = samples[:64]
        if not samples:
            return {"trained": False, "samples": 0, "loss_start": self.last_loss, "loss_end": self.last_loss, "duration_ms": 0.0}

        n_epochs = max(1, min(4, int(epochs or self.cfg.broca_llm_sleep_epochs)))
        eff_lr = float(lr or self.cfg.broca_llm_lora_lr)

        with self._lock:
            lora_params = [p for p in self.model.parameters() if p.requires_grad]
            if not lora_params:
                return {"trained": False, "samples": 0, "loss_start": 0.0, "loss_end": 0.0, "duration_ms": 0.0}

            prev_use_cache = getattr(self.model.config, "use_cache", True)
            self.model.config.use_cache = False
            self.model.train()
            opt = torch.optim.AdamW(lora_params, lr=eff_lr, weight_decay=0.01)

            loss_start: float | None = None
            loss_end = self.last_loss
            try:
                for _ in range(n_epochs):
                    ep_sum = 0.0
                    ep_n = 0
                    for i in range(0, len(samples), 2):
                        batch_texts = samples[i : i + 2]
                        enc = self.tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512,
                        ).to(self.device)
                        labels = enc["input_ids"].clone()
                        labels[enc["attention_mask"] == 0] = -100

                        opt.zero_grad(set_to_none=True)
                        out = self.model(**enc, labels=labels)
                        loss = out.loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                        opt.step()

                        v = float(loss.detach().item())
                        if loss_start is None:
                            loss_start = v
                        ep_sum += v
                        ep_n += 1
                    if ep_n > 0:
                        loss_end = ep_sum / float(ep_n)

                self.lora_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(self.lora_dir))
                self.lora_updates_count += 1
                self.last_loss = round(float(loss_end), 4)
            finally:
                self.model.eval()
                self.model.config.use_cache = prev_use_cache
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return {
            "trained": True,
            "samples": len(samples),
            "loss_start": round(float(loss_start if loss_start is not None else loss_end), 4),
            "loss_end": round(float(loss_end), 4),
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def shutdown(self) -> None:
        with self._lock:
            self.model = None
            self.tokenizer = None
            self.ready = False
            self.loading = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
