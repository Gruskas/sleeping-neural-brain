from __future__ import annotations

import ast
import math
import operator
import re
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Iterator

import torch

from .config import BrainConfig
from .neural_language import (
    CorticalSemanticEncoder,
    CorticalTransformerLM,
    SubwordTokenizer,
    _strip_pl,
)


def _compress_normalize(vec: list[float], target_dim: int = 256) -> list[float]:
    if not vec:
        return []
    n = len(vec)
    if n >= target_dim:
        out: list[float] = []
        for i in range(target_dim):
            start = (i * n) // target_dim
            end = max(start + 1, ((i + 1) * n) // target_dim)
            out.append(sum(vec[start:end]) / (end - start))
    else:
        out = list(vec) + [0.0] * (target_dim - n)
    norm = math.sqrt(sum(x * x for x in out))
    if norm < 1e-8:
        return [0.0] * target_dim
    return [round(x / norm, 5) for x in out]


def _fallback_hash_embed(clean: str, target_dim: int = 256) -> list[float]:
    if not clean:
        return []
    vec = [0.0] * target_dim
    low = clean.lower()
    if len(low) < 3:
        vec[zlib.crc32(low.encode("utf-8")) % target_dim] += 1.0
    else:
        for i in range(len(low) - 2):
            gram = low[i : i + 3]
            vec[zlib.crc32(gram.encode("utf-8")) % target_dim] += 1.0
    return _compress_normalize(vec, target_dim=target_dim)


def _cosine_list(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


_STOPWORDS_PL = {
    "jest", "są", "sa", "to", "oraz", "lub", "albo", "czy", "jak", "jaki", "jaka",
    "jakie", "kto", "co", "czym", "gdzie", "kiedy", "dlaczego", "dla", "przez",
    "przy", "nad", "pod", "przed", "bez", "jako", "jego", "jej", "ich", "tego",
    "tym", "tej", "ten", "ta", "te", "tak", "nie", "ale", "lecz", "więc", "wiec",
    "że", "ze", "się", "sie", "w", "we", "z", "ze", "na", "do", "od", "po", "o",
    "u", "za", "i", "a", "no", "krótko", "krotko", "wyjaśnij", "wyjasnij",
    "powiedz", "opowiedz", "opisz", "zdefiniuj", "napisz", "podaj", "temat",
}


def _tokenize_stems(text: str) -> set[str]:
    stripped = _strip_pl(text.lower())
    words = re.findall(r"[a-z0-9+#]{2,}", stripped)
    stems: set[str] = set()
    for w in words:
        if w in _STOPWORDS_PL:
            continue
        stems.add(w)
        if len(w) >= 5:
            stems.add(w[: max(4, len(w) - 2)])
        if len(w) >= 6:
            stems.add(w[:6])
    return stems


def _stem_overlap(query: str, candidate: str) -> float:
    q_words = [
        w for w in re.findall(r"[a-z0-9+#]{2,}", _strip_pl(query.lower()))
        if w not in _STOPWORDS_PL
    ]
    if not q_words:
        return 0.0
    cand_stems = _tokenize_stems(candidate)
    if not cand_stems:
        return 0.0
    hits = 0
    for qw in q_words:
        q_stem = qw[: max(4, len(qw) - 2)] if len(qw) >= 5 else qw
        if qw in cand_stems or q_stem in cand_stems or any(
            cs.startswith(q_stem) or (len(cs) >= 4 and qw.startswith(cs[: max(4, len(cs) - 2)]))
            for cs in cand_stems
        ):
            hits += 1
    return hits / max(1, len(q_words))


def _clean_fact_text(txt: str) -> str:
    s = re.sub(r"==+[^=]+==+", " ", txt)
    s = re.sub(r"\[\d+\]|\[[a-z]\]", "", s)
    s = re.sub(r"\(\d+\.\d+\)", "", s)
    s = re.sub(r"zob\.\s*też\s+[^.\n]+w\s+Wikipedii", "", s, flags=re.I)
    s = re.sub(r"rzeczownik,\s*rodzaj\s+(?:męski|żeński|nijaki)(?:\s+osobowy)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" ;,-")
    return s


def _split_polish_sentences(text: str) -> list[str]:
    protected = text
    abbrs = (
        "np.", "m.in.", "ok.", "łac.", "gr.", "p.n.e.", "n.e.", "ang.", "fr.",
        "por.", "zob.", "tzw.", "wg.", "tj.", "itd.", "itp.", "dr.", "prof.",
        "ryc.", "str.", "r.", "w.", "tys.", "mln.", "mld.",
    )
    for a in abbrs:
        safe = a.replace(".", "@@DOT@@")
        protected = re.sub(re.escape(a), safe, protected, flags=re.I)
    protected = re.sub(r"\b([A-ZŁŚŻŹĆŃ])\.", r"\1@@DOT@@", protected)
    protected = re.sub(r"(\d+)\.(\d+)", r"\1@@DOT@@\2", protected)

    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", protected)
    sentences: list[str] = []
    for p in raw_parts:
        restored = p.replace("@@DOT@@", ".").strip()
        restored = _clean_fact_text(restored)
        if 24 <= len(restored) <= 440:
            sentences.append(restored)
    return sentences


def _extract_query_subject(query: str) -> str:
    s = query.strip()
    patterns = [
        r"^(?:no\s+to\s+|a\s+|powiedz\s+mi\s+|powiedz\s+|wyjaśnij(?:\s+krótko|\s+w\s+skrócie)?\s+|"
        r"wyjasnij(?:\s+krotko|\s+w\s+skrocie)?\s+|wytłumacz\s+|opowiedz(?:\s+o)?\s+|opisz\s+|podaj\s+|zdefiniuj\s+)+",
        r"^(?:kto\s+to\s+(?:jest|był|byl)|co\s+to\s+(?:jest|są|sa)|czym\s+(?:jest|są|sa)|kim\s+(?:jest|był|byl)|"
        r"jaka\s+jest|jaki\s+jest|jakie\s+s[aą]|kiedy\s+(?:by[lł]|by[lł]a|odby[lł]\s+si[eę]|mia[lł]\s+miejsce)|"
        r"gdzie\s+(?:le[zż]y|znajduje\s+si[eę])|ile\s+[a-ząćęłńóśźż]+\s+(?:ma|liczy)|"
        r"jak\s+(?:działają|dzialaja|działa|dziala)|o\s+co\s+chodzi\s+z|na\s+czym\s+polega)\s+",
    ]
    for _ in range(4):
        prev = s
        for pat in patterns:
            s = re.sub(pat, "", s, flags=re.I).strip(" ?.!,:;")
        if s == prev:
            break
    return s or query.strip(" ?.!,:;")


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _safe_eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _safe_eval_ast(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and abs(right) < 1e-12:
            raise ZeroDivisionError("Nie można dzielić przez zero.")
        if isinstance(node.op, ast.Pow):
            if abs(right) > 200 or abs(left) > 1e6:
                raise ValueError("Zbyt duża potęga.")
        return float(_BIN_OPS[type(node.op)](left, right))
    raise ValueError("Nieobsługiwane wyrażenie matematyczne.")


def _fmt_num(val: float) -> str:
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.6f}".rstrip("0").rstrip(".")


def _try_solve_math(msg: str) -> str | None:
    raw = msg.strip()
    low = _strip_pl(raw.lower())

    # 1) Procent z liczby: "25% z 200"
    m_pct = re.search(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:z|ze)\s*(\d+(?:[.,]\d+)?)", low)
    if m_pct:
        pct = float(m_pct.group(1).replace(",", "."))
        base = float(m_pct.group(2).replace(",", "."))
        res = (pct / 100.0) * base
        return f"Obliczenie: {_fmt_num(pct)}% z {_fmt_num(base)} = **{_fmt_num(res)}**."

    # 2) Pierwiastek kwadratowy: "pierwiastek z 144"
    m_sqrt = re.search(r"pierwiastek(?:\s+kwadratowy)?\s+z[e]?\s*(-?\d+(?:[.,]\d+)?)", low)
    if m_sqrt:
        val = float(m_sqrt.group(1).replace(",", "."))
        if val < 0:
            return "Pierwiastek kwadratowy z liczby ujemnej nie istnieje w zbiorze liczb rzeczywistych."
        return f"Pierwiastek kwadratowy z {_fmt_num(val)} wynosi **{_fmt_num(math.sqrt(val))}**."

    # 3) Standardowe wyrażenia arytmetyczne
    cleaned = re.sub(
        r"^(?:ile\s+(?:to\s+jest|to|jest|wynosi)|oblicz|policz|wylicz)\s+",
        "",
        low,
    ).strip(" ?.=!")
    cleaned = cleaned.replace("×", "*").replace("÷", "/").replace("^", "**")
    cleaned = re.sub(r"\b(?:dodac|plus)\b", "+", cleaned)
    cleaned = re.sub(r"\b(?:odjac|minus)\b", "-", cleaned)
    cleaned = re.sub(r"\b(?:pomnozone\s+przez|pomnozyc\s+przez|razy)\b", "*", cleaned)
    cleaned = re.sub(r"\b(?:podzielone\s+przez|podzielic\s+przez)\b", "/", cleaned)
    cleaned = re.sub(r"(\d)\s*:\s*(\d)", r"\1 / \2", cleaned)
    cleaned = re.sub(r"(\d+(?:[.,]\d+)?)\s+do\s+kwadratu", r"(\1)**2", cleaned)
    cleaned = re.sub(r"(\d+(?:[.,]\d+)?)\s+do\s+szescianu", r"(\1)**3", cleaned)
    cleaned = re.sub(r"\bdo\s+potegi\b", "**", cleaned)
    cleaned = cleaned.replace(",", ".")

    if not re.search(r"\d", cleaned):
        return None
    if not any(op in cleaned for op in ("+", "-", "*", "/", "%")):
        return None
    if not re.fullmatch(r"[0-9.\s+\-*/()%]+", cleaned):
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
        res = _safe_eval_ast(tree)
        return f"Wynik działania `{cleaned}` to **{_fmt_num(res)}**."
    except ZeroDivisionError:
        return "W matematyce nie można dzielić przez zero."
    except Exception:
        return None


class LanguageCortex:
    """
    100% Self-Contained PyTorch GPU Language & Semantic Cortex (zero external LLMs, zero Ollama):
      - `SubwordTokenizer`: lossless UTF-8 byte + dynamic Polish/code vocabulary
      - `CorticalSemanticEncoder`: 256D GPU encoder fine-tuned contrastively during NREM sleep
      - `CorticalTransformerLM`: 4-layer causal Transformer trained on GPU via sleep replay
      - Polish neuro-linguistic analyzer/synthesizer grounded strictly in Hippocampus/Cortex memory
    """

    def __init__(self, cfg: BrainConfig, device: torch.device, lora_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.device = device
        self.lora_dir = lora_dir
        self._op_lock = threading.Lock()

        self.tokenizer = SubwordTokenizer(max_vocab_size=cfg.vocab_size)
        self.semantic_encoder = CorticalSemanticEncoder(
            vocab_size=cfg.vocab_size,
            ngram_buckets=cfg.ngram_buckets,
            d_model=cfg.cortex_dim,
        ).to(device)
        self.transformer_lm = CorticalTransformerLM(
            vocab_size=cfg.vocab_size,
            d_model=cfg.cortex_dim,
            n_layers=cfg.cortex_layers,
            n_heads=cfg.cortex_heads,
            ff_dim=cfg.cortex_ff_dim,
            max_seq_len=cfg.cortex_max_seq_len,
        ).to(device)

        self.ready: bool = True
        self.loading: bool = False
        self.load_error: str | None = None
        self.lora_updates_count: int = 0
        self.core_knowledge_summary: str = ""
        self.available_models: list[str] = [cfg.language_model_id]
        self.status_message: str = ""
        self._refresh_status()

    def total_parameters(self) -> int:
        enc_params = sum(p.numel() for p in self.semantic_encoder.parameters())
        lm_params = self.transformer_lm.total_parameters()
        return enc_params + lm_params

    def _refresh_status(self) -> None:
        dev_label = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        m_params = self.total_parameters() / 1_000_000.0
        self.status_message = (
            f"Gotowa: {self.cfg.language_model_id} ({dev_label} · {m_params:.2f}M param. · 100% lokalna)"
        )

    def load_sync(self) -> bool:
        with self._op_lock:
            self.semantic_encoder.to(self.device)
            self.transformer_lm.to(self.device)
            self.ready = True
            self.loading = False
            self.load_error = None
            self._refresh_status()
            return True

    def load_async(self) -> None:
        self.load_sync()

    def shutdown(self) -> None:
        return

    def save_lora(self) -> None:
        return

    def reset_lora(self) -> None:
        with self._op_lock:
            self.tokenizer = SubwordTokenizer(max_vocab_size=self.cfg.vocab_size)
            self.semantic_encoder = CorticalSemanticEncoder(
                vocab_size=self.cfg.vocab_size,
                ngram_buckets=self.cfg.ngram_buckets,
                d_model=self.cfg.cortex_dim,
            ).to(self.device)
            self.transformer_lm = CorticalTransformerLM(
                vocab_size=self.cfg.vocab_size,
                d_model=self.cfg.cortex_dim,
                n_layers=self.cfg.cortex_layers,
                n_heads=self.cfg.cortex_heads,
                ff_dim=self.cfg.cortex_ff_dim,
                max_seq_len=self.cfg.cortex_max_seq_len,
            ).to(self.device)
            self.lora_updates_count = 0
            self.core_knowledge_summary = ""
            self._refresh_status()

    def export_state(self) -> dict[str, Any]:
        with self._op_lock:
            return {
                "tokenizer": self.tokenizer.export_state(),
                "semantic_encoder": self.semantic_encoder.export_state(),
                "transformer_lm": self.transformer_lm.export_state(),
            }

    def import_state(self, state: dict[str, Any]) -> None:
        with self._op_lock:
            if "tokenizer" in state:
                self.tokenizer.import_state(state["tokenizer"])
            if "semantic_encoder" in state:
                self.semantic_encoder.import_state(state["semantic_encoder"])
            if "transformer_lm" in state:
                self.transformer_lm.import_state(state["transformer_lm"])
            self._refresh_status()

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._op_lock:
            self.tokenizer.learn_texts(texts)
            return self.semantic_encoder.encode_texts(
                texts,
                self.tokenizer,
                batch_size=self.cfg.embed_batch_size,
            )

    def encode_text(self, text: str) -> list[float]:
        clean = text.strip()
        if not clean:
            return []
        return self.encode_texts([clean])[0]

    def normalize_study_queries(self, raw_topic: str) -> list[str]:
        q = raw_topic.strip()
        if not q:
            return []

        # Fix common Polish phonetic typos and missing diacritics
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
            (r"\bts\b", "TypeScript"),
        ]
        fixed = q
        for pat, repl in typo_rules:
            fixed = re.sub(pat, repl, fixed, flags=re.I)
        fixed = re.sub(r"\s+", " ", fixed).strip()

        low_s = _strip_pl(fixed.lower())
        prog_langs = ("c++", "c#", "python", "rust", "java", "javascript", "typescript", "programow", "kod")
        has_prog = any(pl in low_s for pl in prog_langs)

        if has_prog:
            core_terms = [
                w for w in re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9+#]{3,}", fixed)
                if _strip_pl(w.lower()) not in ("cpp", "c++", "c#", "python", "rust", "java", "javascript", "typescript", "jest", "oraz")
            ]
            main_term = core_terms[0] if core_terms else fixed
            candidates = [
                f"{fixed} programowanie",
                f"{main_term} informatyka",
                fixed,
            ]
        elif len(fixed.split()) <= 2:
            candidates = [fixed, f"{fixed} definicja historia"]
        else:
            candidates = [fixed]

        dedup: list[str] = []
        for c in candidates:
            if c and c.lower() not in {x.lower() for x in dedup}:
                dedup.append(c)
        return dedup[:3]

    def study_document_chunk(self, topic: str, title: str, chunk: str) -> dict[str, Any]:
        sentences = _split_polish_sentences(chunk)
        if not sentences:
            return {"facts": [], "triples": []}

        anchor = f"{topic} {title}"
        all_vecs = self.encode_texts([anchor] + sentences)
        anchor_vec = all_vecs[0]
        sent_vecs = all_vecs[1:]

        def_cues = (
            " – ", " — ", " to ", " jest ", " są ", " oznacza ", " był ", " była ",
            " przechowuje ", " służy ", " umożliwia ", " polega na ", " nazywa się ",
            " definiuje ", " wskazuje ", " zawiera ",
        )
        noise_cues = ("zobacz też", "przypisy", "bibliografia", "linki zewnętrzne", "kategoria:", "isbn")

        scored: list[tuple[float, str, list[float], bool]] = []
        for sent, vec in zip(sentences, sent_vecs):
            low = sent.lower()
            if any(nc in low for nc in noise_cues):
                continue
            cos = _cosine_list(anchor_vec, vec)
            ov = max(_stem_overlap(topic, sent), _stem_overlap(title, sent))
            is_def = any(cue in low for cue in def_cues)
            def_bonus = 0.16 if is_def else 0.0
            score = 0.48 * cos + 0.36 * ov + def_bonus
            if score >= 0.20 or ov > 0.0:
                scored.append((score, sent, vec, is_def))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected_facts: list[dict[str, Any]] = []
        selected_vecs: list[list[float]] = []
        for score, sent, vec, is_def in scored:
            if any(_cosine_list(vec, prev_v) > 0.88 for prev_v in selected_vecs):
                continue
            imp = 5 if (is_def and score >= 0.44) else (4 if score >= 0.32 else 3)
            selected_facts.append({"text": sent, "importance": imp})
            selected_vecs.append(vec)
            if len(selected_facts) >= 7:
                break

        triples: list[tuple[str, str, str]] = []
        seen_t: set[tuple[str, str, str]] = set()
        topic_key = topic.strip().lower()[:50]
        title_key = re.sub(r"^Słownik\s*\([A-Z]+\):\s*", "", title).strip().lower()[:50]

        rel_patterns = [
            (r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9+#\s-]{3,42}?)\s+(?:–|—|-|:)\s+([^.,;]{4,55})", "oznacza"),
            (r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9+#\s-]{3,42}?)\s+(?:to|jest)\s+([^.,;]{4,55})", "jest"),
            (r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9+#\s-]{3,42}?)\s+(przechowuje|wskazuje|umożliwia|oznacza|tworzy|emituje|powoduje)\s+([^.,;]{4,55})", None),
        ]
        for item in selected_facts:
            txt = item["text"]
            for pat, fixed_rel in rel_patterns:
                m = re.match(pat, txt)
                if m:
                    s = m.group(1).strip().lower()[:50]
                    if fixed_rel is None:
                        r = m.group(2).strip().lower()[:35]
                        d = m.group(3).strip().lower()[:50]
                    else:
                        r = fixed_rel
                        d = m.group(2).strip().lower()[:50]
                    if s and r and d and s != d and (s, r, d) not in seen_t:
                        seen_t.add((s, r, d))
                        triples.append((s, r, d))
                        break

        if topic_key and title_key and topic_key != title_key and len(triples) < 6:
            t_link = (topic_key, "dotyczy", title_key)
            if t_link not in seen_t:
                triples.append(t_link)

        return {"facts": selected_facts, "triples": triples[:6]}

    def extract_memories_from_turn(self, user_message: str) -> dict[str, Any]:
        clean_msg = user_message.strip()
        if len(clean_msg) < 6:
            return {"memories": [], "triples": []}

        interrogative_starts = (
            "co ", "kto ", "czym ", "jak ", "gdzie ", "kiedy ", "dlaczego ", "po co", "skad ",
            "czy ", "ile ", "jaki", "jaka", "jakie", "komu", "kogo", "no to", "powiedz",
            "wyjasnij", "wytlumacz", "opisz", "opowiedz", "podaj", "daj ", "pokaz",
            "napisz", "zrob", "stworz", "sprawdz", "pomoz", "popraw", "naucz", "zbadaj",
            "ale to nie", "to nie prawda", "nie prawda", "nieprawda", "bzdura",
            "mylisz sie", "chodzi", "okej", "ok ", "dzieki", "czesc", "hej", "witaj",
        )
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_msg) if s.strip()]
        memories: list[dict[str, Any]] = []
        triples: list[tuple[str, str, str]] = []

        for s in sentences:
            low_raw = s.lower().lstrip(" ,.-")
            low_s = _strip_pl(low_raw)
            if s.endswith("?") or len(low_s) < 8:
                continue
            if any(low_s.startswith(p) for p in interrogative_starts):
                continue

            matched = False
            user_patterns = [
                (r"(?:mam na imi[eę]|nazywam si[eę])\s+([A-ZŁŚŻŹĆŃa-ząćęłńóśźż-]{2,30})", "Użytkownik ma na imię {val}.", "ma_na_imię", 5),
                (r"(?:mieszkam w|mieszkam na)\s+([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9\s-]{2,40})", "Użytkownik mieszka w {val}.", "mieszka_w", 5),
                (r"pochodz[eę] z\s+([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9\s-]{2,40})", "Użytkownik pochodzi z {val}.", "pochodzi_z", 4),
                (r"interesuj[eę] si[eę]\s+([^.,;!?]{3,70})", "Użytkownik interesuje się {val}.", "interesuje_się", 4),
                (r"(?:bardzo\s+)?lubi[eę]\s+([^.,;!?]{3,70})", "Użytkownik lubi {val}.", "lubi", 4),
                (r"uwielbiam\s+([^.,;!?]{3,70})", "Użytkownik uwielbia {val}.", "uwielbia", 4),
                (r"wol[eę]\s+([^.,;!?]{3,70})", "Użytkownik woli {val}.", "woli", 4),
                (r"nie lubi[eę]\s+([^.,;!?]{3,70})", "Użytkownik nie lubi {val}.", "nie_lubi", 4),
                (r"pracuj[eę]\s+(?:jako|w)\s+([^.,;!?]{3,60})", "Użytkownik pracuje jako/w {val}.", "pracuje", 5),
                (r"mam\s+(\d{1,3})\s+lat", "Użytkownik ma {val} lat.", "ma_lat", 4),
                (r"(?:używam|uzywam|korzystam z|programuj[eę] w|pisz[eę] w)\s+([^.,;!?]{2,60})", "Użytkownik używa {val}.", "używa", 4),
            ]
            for pat, tmpl, rel, imp in user_patterns:
                for m in re.finditer(pat, s, flags=re.I):
                    val = m.group(1).strip(" .!?,;")
                    if val:
                        memories.append({"text": tmpl.format(val=val), "importance": imp, "category": "użytkownik"})
                        triples.append(("użytkownik", rel, val.lower()[:50]))
                        matched = True

            if matched:
                continue

            # Explicit coding rules / preferences
            if re.match(r"^(?:zasada:|preferuj[eę]|zawsze\s+|nigdy\s+nie\s+)", low_raw):
                memories.append({"text": s.rstrip(".") + ".", "importance": 5, "category": "zasada"})
                continue

            # Explicit teaching or clean declarative definitions
            teach_m = re.match(r"^(?:zapami[eę]taj(?:\s*,?\s*[zż]e)?\s*:?\s*)(.+)$", s, flags=re.I)
            if teach_m:
                fact = teach_m.group(1).strip()
                if len(fact) >= 6:
                    memories.append({"text": fact.rstrip(".") + ".", "importance": 5, "category": "wiedza"})
                continue

            def_m = re.match(r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9+#\s-]{2,40}?)\s+(to|jest|oznacza)\s+([^!?]{5,140})$", s)
            if def_m and len(s) >= 18:
                subj = def_m.group(1).strip().lower()[:50]
                rel = def_m.group(2).strip().lower()
                obj = def_m.group(3).strip().rstrip(".").lower()[:50]
                memories.append({"text": s.rstrip(".") + ".", "importance": 4, "category": "wiedza"})
                if subj and obj and subj != obj:
                    triples.append((subj, rel, obj))

        # Deduplicate memories preserving order
        uniq_mems: list[dict[str, Any]] = []
        seen_txt: set[str] = set()
        for m in memories:
            key = m["text"].lower()
            if key not in seen_txt:
                seen_txt.add(key)
                uniq_mems.append(m)

        return {"memories": uniq_mems[:12], "triples": triples[:12]}

    def assess_study_mastery(self, topic: str, learned_facts: list[str], round_num: int) -> dict[str, Any]:
        if not learned_facts:
            return {"mastery": 10.0, "next_queries": [f"{topic} definicja"], "summary": "Jeszcze brak wiedzy."}

        vecs = self.encode_texts(learned_facts[-20:])
        # Measure semantic diversity across learned facts
        unique_clusters = 0
        kept_vecs: list[list[float]] = []
        for v in vecs:
            if not any(_cosine_list(v, kv) > 0.85 for kv in kept_vecs):
                kept_vecs.append(v)
                unique_clusters += 1

        has_def = any(
            any(cue in f.lower() for cue in ("znaczenie słowa", " to ", " jest ", " – ", " — ", "oznacza"))
            for f in learned_facts
        )
        base_score = min(68.0, len(learned_facts) * 7.5 + unique_clusters * 4.5)
        def_bonus = 14.0 if has_def else 0.0
        round_bonus = min(18.0, round_num * 6.5)
        mastery = round(min(96.0, base_score + def_bonus + round_bonus), 1)

        # Find promising follow-up concepts from learned facts
        topic_stems = _tokenize_stems(topic)
        next_queries: list[str] = []
        for f in learned_facts:
            for cand in re.findall(r"\b[A-ZŁŚŻŹĆŃ][a-ząćęłńóśźż]{3,}(?:\s+[A-ZŁŚŻŹĆŃ][a-ząćęłńóśźż]{3,})?", f):
                if _strip_pl(cand.lower()) not in topic_stems and cand not in next_queries:
                    next_queries.append(cand)
                    if len(next_queries) >= 2:
                        break
            if len(next_queries) >= 2:
                break
        if not next_queries:
            next_queries = [f"{topic} zastosowania"]

        summary_facts = [_clean_fact_text(f) for f in learned_facts[:2]]
        summary = " ".join(summary_facts)[:320]
        return {"mastery": mastery, "next_queries": next_queries[:2], "summary": summary}

    def synthesize_rem_reflections(self, new_memories: list[str], old_memories: list[str]) -> list[str]:
        if not new_memories:
            return []
        pool_new = new_memories[:12]
        pool_old = old_memories[:16] if old_memories else new_memories[1:12]
        if not pool_old:
            return []

        vecs_new = self.encode_texts(pool_new)
        vecs_old = self.encode_texts(pool_old)

        best_pairs: list[tuple[float, str, str, list[str]]] = []
        for mn, vn in zip(pool_new, vecs_new):
            sn = _tokenize_stems(mn)
            for mo, vo in zip(pool_old, vecs_old):
                if mn.lower() == mo.lower():
                    continue
                cos = _cosine_list(vn, vo)
                if 0.25 <= cos <= 0.85:
                    shared = sorted(w for w in (sn & _tokenize_stems(mo)) if len(w) >= 4)
                    score = cos + min(0.25, len(shared) * 0.08)
                    best_pairs.append((score, mn, mo, shared))

        best_pairs.sort(key=lambda x: x[0], reverse=True)
        reflections: list[str] = []
        used_texts: set[str] = set()
        for _, mn, mo, shared in best_pairs:
            if mn in used_texts or mo in used_texts:
                continue
            used_texts.add(mn)
            used_texts.add(mo)
            c1 = _clean_fact_text(mn).rstrip(".")
            c2 = _clean_fact_text(mo).rstrip(".")
            if shared:
                bridge = ", ".join(shared[:2])
                refl = f"Skojarzenie REM ({bridge}): {c1[:110]} — łączy się z: {c2[:110]}."
            else:
                refl = f"Wniosek REM: {c1[:110]} uzupełnia pamięć o: {c2[:110]}."
            reflections.append(refl)
            if len(reflections) >= 2:
                break
        return reflections

    def check_memory_update(self, new_fact: str, candidate_old_facts: list[str]) -> int | None:
        if not candidate_old_facts:
            return None
        new_low = _strip_pl(new_fact.lower())
        singleton_prefixes = (
            "uzytkownik ma na imie ",
            "uzytkownik mieszka w ",
            "uzytkownik pochodzi z ",
            "uzytkownik pracuje ",
        )
        for pref in singleton_prefixes:
            if new_low.startswith(pref):
                for idx, old in enumerate(candidate_old_facts):
                    if _strip_pl(old.lower()).startswith(pref):
                        return idx
        if new_low.startswith("uzytkownik ma ") and new_low.endswith(" lat."):
            for idx, old in enumerate(candidate_old_facts):
                old_low = _strip_pl(old.lower())
                if old_low.startswith("uzytkownik ma ") and old_low.endswith(" lat."):
                    return idx
        return None

    def train_lora_sleep(
        self,
        new_memories: list[str],
        replay_memories: list[str],
        rem_reflections: list[str],
    ) -> dict[str, Any]:
        """
        Real NREM Deep Sleep GPU Training (100% self-contained PyTorch):
          1) Expands tokenizer vocabulary with newly encountered Polish/code tokens
          2) Contrastively trains `CorticalSemanticEncoder` on GPU
          3) Trains `CorticalTransformerLM` via Experience Replay on GPU
          4) Updates structured `core_knowledge_summary`
        """
        if not self.ready or not new_memories:
            return {"trained": False, "samples": 0, "loss_start": 0.0, "loss_end": 0.0, "duration_ms": 0.0}

        t0 = time.perf_counter()
        sample_limit = max(self.cfg.lora_max_replay_samples, min(200, len(new_memories) + 20))
        all_items = (new_memories[:160] + replay_memories[:40])[:sample_limit]

        # Build contrastive (anchor_prompt, memory) pairs
        pairs: list[tuple[str, str]] = []
        sequences: list[str] = []
        for item in all_items:
            clean = _clean_fact_text(item)
            if len(clean) < 8:
                continue
            words = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9+#]{3,}", clean)
            hint = " ".join(words[:3]) if words else clean[:24]
            pairs.append((hint, clean))
            sequences.append(f"<Q> {hint} <A> {clean}")
        for refl in rem_reflections[:6]:
            clean_r = _clean_fact_text(refl)
            if len(clean_r) >= 10:
                sequences.append(f"<Q> refleksja <A> {clean_r}")

        with self._op_lock:
            self.tokenizer.learn_texts(all_items + rem_reflections)
            enc_s, enc_e = self.semantic_encoder.train_contrastive(
                pairs=pairs[:80],
                tokenizer=self.tokenizer,
                epochs=max(4, self.cfg.cortex_sleep_epochs // 2),
                lr=self.cfg.cortex_lr,
            )
            lm_s, lm_e = self.transformer_lm.train_sleep_replay(
                sequences=sequences,
                tokenizer=self.tokenizer,
                epochs=self.cfg.cortex_sleep_epochs,
                lr=self.cfg.cortex_lr,
                batch_size=32 if len(sequences) > 40 else 16,
            )

            # Distill structured Core Knowledge Summary deterministically from highest-signal facts
            existing_lines = [
                ln.strip()
                for ln in self.core_knowledge_summary.splitlines()
                if ln.strip().startswith("- ")
            ]
            combined_pool = []
            seen_k: set[str] = set()
            for raw in all_items + [ln[2:] for ln in existing_lines]:
                c = _clean_fact_text(raw)
                k = c.lower()
                if c and k not in seen_k:
                    seen_k.add(k)
                    combined_pool.append(c)

            user_pts = [x for x in combined_pool if x.lower().startswith("użytkownik ") or "zasada" in x.lower()][:5]
            know_pts = [x for x in combined_pool if x not in user_pts][:7]
            sections: list[str] = []
            if user_pts:
                sections.append("1) Profil i zasady użytkownika:\n" + "\n".join(f"- {u}" for u in user_pts))
            if know_pts:
                sections.append("2) Kluczowa utrwalona wiedza:\n" + "\n".join(f"- {k[:180]}" for k in know_pts))
            self.core_knowledge_summary = "\n".join(sections)[:2400]
            self.lora_updates_count += 1

        dur_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        loss_start = lm_s if lm_s > 0 else enc_s
        loss_end = lm_e if lm_e > 0 else enc_e
        return {
            "trained": True,
            "samples": len(sequences),
            "loss_start": loss_start,
            "loss_end": loss_end,
            "encoder_loss_start": enc_s,
            "encoder_loss_end": enc_e,
            "duration_ms": dur_ms,
        }

    def chat_reply(
        self,
        user_message: str,
        recent_turns: list[dict[str, str]],
        hippocampus_memories: list[str],
        semantic_memories: list[str],
        rem_reflections: list[str],
        concept_hints: list[str],
        day: int,
        sleeps: int,
        energy: float,
        state_mode: str,
        user_profile: list[str] | None = None,
    ) -> str:
        reply, _ = self._chat_reply_with_meta(
            user_message=user_message,
            recent_turns=recent_turns,
            hippocampus_memories=hippocampus_memories,
            semantic_memories=semantic_memories,
            rem_reflections=rem_reflections,
            concept_hints=concept_hints,
            day=day,
            sleeps=sleeps,
            energy=energy,
            state_mode=state_mode,
            user_profile=user_profile,
        )
        return reply

    def _chat_reply_with_meta(
        self,
        user_message: str,
        recent_turns: list[dict[str, str]],
        hippocampus_memories: list[str],
        semantic_memories: list[str],
        rem_reflections: list[str],
        concept_hints: list[str],
        day: int,
        sleeps: int,
        energy: float,
        state_mode: str,
        user_profile: list[str] | None = None,
    ) -> tuple[str, str | None]:
        msg = user_message.strip()
        low = _strip_pl(msg.lower()).strip(" ?!.,;:")

        # 1) Correction / disagreement reaction
        if any(p in low for p in ("ale to nie prawda", "to nie prawda", "nieprawda", "bzdura", "mylisz sie")):
            return (
                "Rozumiem, dzięki za zwrócenie uwagi! Nie zapisałem tej reakcji jako faktu. "
                "Napisz proszę poprawną informację zdaniem twierdzącym albo użyj `/ucz <temat>`, "
                "abym sprawdził wiarygodne źródła i zaktualizował pamięć.",
                None,
            )

        # Find user name if known
        user_name = ""
        for mem in (user_profile or []) + hippocampus_memories + semantic_memories:
            m_name = re.search(r"Użytkownik ma na imię\s+([^\s.,!?;]+)", mem, flags=re.I)
            if m_name:
                user_name = m_name.group(1)
                break
        name_part = f" {user_name}" if user_name else ""

        # 2) Thanks / acknowledgement / reactions
        if low in ("dzieki", "dziekuje", "dzieki wielkie", "thx"):
            return (f"Nie ma za co{name_part}! Pytaj śmiało o kolejne rzeczy albo po prostu porozmawiajmy.", None)

        if low in ("wow", "lal", "mega", "cool", "extra", "ekstra", "super", "swietnie", "genialnie", "niesamowite", "niezle", "fajnie", "dobre", "brawo", "szacun"):
            return (f"Dzięki{name_part}! Staram się jak mogę — im więcej rozmawiamy i uczymy się razem, tym lepiej działam. O czym chcesz teraz pogadać?", None)

        if low in ("spoko", "okej", "ok", "rozumiem", "jasne", "aha", "dobrze", "git", "pewnie", "zgadza sie", "tak", "nie", "moze", "chyba", "nom", "taa", "no", "raczej"):
            return (f"Jasne{name_part}! Jeśli chcesz o coś zapytać, coś policzyć albo czegoś mnie nauczyć — pisz śmiało.", None)

        if low in ("haha", "hehe", "hihi", "xd", "lol", "lmao"):
            return (f"Cieszę się, że humor dopisuje{name_part}! O czym pogadamy dalej?", None)

        # 2b) Compliments & criticism (including common Polish typos like "moadry jestes")
        is_compliment = any(
            p in low
            for p in (
                "madry jestes", "moadry jestes", "jestes madry", "jestes moadry",
                "jestes super", "super jestes", "fajny jestes", "jestes fajny",
                "dobry jestes", "jestes dobry", "inteligentny", "sprytny",
                "lubie cie", "najlepszy", "kozacki", "genialny jestes",
            )
        )
        if is_compliment:
            return (
                f"Miło mi to słyszeć{name_part}! Uczę się na bieżąco z naszych rozmów i z internetu, "
                f"a w nocy utrwalam wszystko w sieci neuronowej na GPU. Chcesz mnie o coś zapytać albo pogadać na jakiś temat?",
                None,
            )

        is_criticism = any(
            p in low
            for p in ("glupi jestes", "jestes glupi", "slabo", "slaby jestes", "bez sensu", "nic nie umiesz", "nie dziala")
        )
        if is_criticism:
            return (
                f"Przepraszam, jeśli odpowiedź nie była trafna{name_part}. Dopiero się uczę — "
                f"możesz poprawić mnie zdaniem twierdzącym albo włączyć `🌐 Auto-sieć`, żebym sam sprawdzał odpowiedzi w internecie!",
                None,
            )

        # 2c) Small talk / mood / jokes
        if any(p in low for p in ("jak sie masz", "co slychac", "co u ciebie", "wszystko ok", "jak leci", "co tam")):
            return (
                f"U mnie wszystko w porządku{name_part}! Energia na poziomie {energy:.0f}% "
                f"(Dzień #{day}, przespanych nocy: {sleeps}). W czym mogę Ci pomóc albo o czym porozmawiamy?",
                None,
            )

        if any(p in low for p in ("nudzi mi sie", "pogadajmy", "porozmawiajmy")):
            return (
                f"Chętnie porozmawiam{name_part}! Możesz zapytać mnie o dowolną ciekawostkę "
                f"(z włączonym `🌐 Auto-sieć` od razu ją sprawdzę w internecie), możemy policzyć jakieś zadanie albo po prostu pogadać.",
                None,
            )

        if any(p in low for p in ("zart", "kawal")):
            return (
                "Dlaczego sieć neuronowa poszła spać? Bo bez fazy REM w ogóle nie kojarzyła faktów! 😄",
                None,
            )

        # 3) Farewell
        if low in ("dobranoc", "do zobaczenia", "pa", "nara", "czesc na dzis"):
            sleep_note = " Masz w hipokampie świeże wspomnienia — możesz wpisać `/sen`, abym je utrwalił na GPU." if hippocampus_memories else ""
            return (f"Do zobaczenia{name_part}!{sleep_note}", None)

        # 4) Conversational capability / "o czym mogę z tobą rozmawiać" (tolerant of Polish typos)
        is_capability_q = (
            any(p in low for p in ("z toba", "ze mna", "potrafisz", "umiesz", "do czego sluzysz", "jak dzialasz", "co robisz"))
            and any(v in low for v in ("rozmaw", "rozmiac", "pogad", "gadac", "moge", "mozemy", "potraf", "umiesz", "dzialasz", "robisz", "sluzysz", "pomoc"))
        ) or low in ("co potrafisz", "co umiesz", "kim jestes", "o czym mozemy rozmawiac", "o czym moge z toba rozmawiac")
        if is_capability_q:
            return (
                f"Możemy rozmawiać po polsku na wiele tematów{name_part}! Oto co potrafię jako samowystarczalny mózg PyTorch:\n"
                "- **Rozmowa i pamięć**: zapamiętuję fakty o Tobie oraz informacje, których mnie uczysz zdaniami twierdzącymi.\n"
                "- **Auto-sieć**: gdy nie znam odpowiedzi, sam sprawdzam ją w internecie i od razu odpowiadam.\n"
                "- **Podstawy szkolne jednym kliknięciem**: kliknij **`🎓 Szkoła Podstawowa`** (albo wpisz `/szkola`), a wgram ~140 podstawowych pojęć z różnych przedmiotów.\n"
                "- **Matematyka szkolna**: policzę działania, potęgi, procenty i pierwiastki (np. `ile to 12 * 15`, `25% z 200`, `pierwiastek z 144`).",
                None,
            )

        # 5) Short conversational follow-ups / confusion ("ale co", "dlaczego", "nie rozumiem", "o co chodzi")
        follow_up_phrases = (
            "ale co", "co", "no i", "i co", "o co chodzi", "dlaczego", "czemu",
            "jak to", "nie rozumiem", "rozwin", "powiedz wiecej", "wiecej", "serio",
        )
        if low in follow_up_phrases or (len(low.split()) <= 2 and any(low == fp for fp in follow_up_phrases)):
            last_bot = ""
            for t in reversed(recent_turns or []):
                if t.get("role") == "assistant":
                    last_bot = t.get("content", "").strip()
                    break
            if last_bot:
                return (
                    "Chodziło mi o naszą ostatnią rozmowę. Napisz dokładniej, które pojęcie mam wyjaśnić albo policzyć "
                    "(np. `co to jest fotosynteza`, `ile to 7 * 8`).",
                    None,
                )
            return (
                "Napisz proszę pełne pytanie lub konkretne pojęcie (np. `co to jest grawitacja`, `ile to 15% z 200`), "
                "albo po prostu napisz, o czym chcesz porozmawiać!",
                None,
            )

        # 6) Pure greeting / status question
        is_greeting = bool(
            re.match(r"^(?:czesc|hej|hejka|siema|siemanko|witaj|witam|elo|yo|dzien dobry|dobry wieczor|jak sie czujesz)[!.\s?]*$", low)
        )
        if is_greeting:
            sleep_hint = " Czuję się już lekko śpiący — przyda się `/sen`, aby skonsolidować pamięć na GPU." if energy <= self.cfg.drowsy_threshold else ""
            return (
                f"Cześć{name_part}! Jestem w pełni samowystarczalnym mózgiem PyTorch na GPU "
                f"(Dzień #{day}, energia {energy:.0f}%, przespanych nocy: {sleeps}). "
                f"Możesz ze mną normalnie porozmawiać, zapytać o dowolny temat albo czegoś mnie nauczyć.{sleep_hint}",
                None,
            )

        # 7) Memory self-inspection questions ("co pamiętasz", "co o mnie wiesz", "czego się nauczyłeś")
        is_mem_query = any(
            p in low for p in ("co pamietasz", "co juz pamietasz", "co o mnie wiesz", "czego sie nauczyles", "jakie tematy")
        )
        if is_mem_query:
            all_mems = []
            seen = set()
            for x in (user_profile or []) + hippocampus_memories + semantic_memories:
                c = _clean_fact_text(x)
                if c and c.lower() not in seen:
                    seen.add(c.lower())
                    all_mems.append(c)
            if not all_mems and not self.core_knowledge_summary:
                return ("Na razie moja pamięć jest jeszcze pusta. Kliknij `🎓 Szkoła Podstawowa` (lub wpisz `/szkola`), naucz mnie czegoś w rozmowie albo wpisz `/ucz <temat>`!", None)
            lines = ["Oto najważniejsze rzeczy, które pamiętam:"]
            for item in all_mems[:8]:
                lines.append(f"- {item}")
            if rem_reflections:
                lines.append("\nWnioski ze snu REM:")
                for r in rem_reflections[:2]:
                    lines.append(f"- {_clean_fact_text(r)}")
            return ("\n".join(lines), None)

        # 8) Elementary school math solver
        math_ans = _try_solve_math(msg)
        if math_ans is not None:
            return (math_ans, None)

        # 9) Did user just state/teach something declarative (no question)?
        is_question = msg.endswith("?") or any(
            low.startswith(p)
            for p in (
                "co ", "kto ", "czym ", "jak ", "jaka ", "jaki ", "jakie ", "gdzie ", "kiedy ",
                "dlaczego ", "czy ", "ile ", "no to", "powiedz", "wyjasnij", "wytlumacz",
                "opisz", "opowiedz", "podaj", "zdefiniuj",
            )
        )
        extracted_now = self.extract_memories_from_turn(msg)
        if extracted_now["memories"] and not is_question:
            learned_str = "; ".join(m["text"] for m in extracted_now["memories"][:3])
            return (f"Zapamiętałem w hipokampie: {learned_str}", None)

        # 10) Knowledge / definition / explanation query grounded strictly in recalled memory
        subj = _extract_query_subject(msg)
        subj_low = _strip_pl(subj.lower())
        conversational_tokens = {"moge", "mozesz", "mozemy", "toba", "mna", "ciebie", "jestem", "jestes", "rozmiac", "rozmawiac", "pogadac", "ale", "czy"}
        subj_words = [w for w in re.findall(r"[a-z0-9+#]{2,}", subj_low) if w not in _STOPWORDS_PL]
        is_non_topic_phrase = (
            len(subj) <= 2
            or not subj_words
            or any(w in conversational_tokens for w in subj_words)
        )

        candidates: list[str] = []
        seen_c: set[str] = set()
        for raw in hippocampus_memories + semantic_memories:
            c = _clean_fact_text(raw)
            if c and c.lower() not in seen_c:
                seen_c.add(c.lower())
                candidates.append(c)

        if not candidates:
            if is_non_topic_phrase:
                return (
                    "Nie jestem pewien, o co dokładnie pytasz. Podaj proszę konkretne zagadnienie "
                    "(np. `co to jest fotosynteza`), obliczenie (np. `ile to 8 * 7`) albo kliknij `🎓 Szkoła Podstawowa` (`/szkola`).",
                    None,
                )
            return (
                f"Nie mam jeszcze w swojej pamięci informacji na temat „{subj}”. "
                f"Wpisz `/ucz {subj}`, a sam znajdę wiarygodne źródła w sieci i się tego nauczę!",
                subj,
            )

        # Rank candidates jointly using GPU Semantic Encoder + Polish stem overlap + Transformer LM score
        vecs = self.encode_texts([subj, msg] + candidates)
        subj_vec = vecs[0]
        msg_vec = vecs[1]
        cand_vecs = vecs[2:]
        with self._op_lock:
            lm_scores = self.transformer_lm.sequence_scores(candidates, self.tokenizer)

        ranked: list[tuple[float, float, float, str, list[float]]] = []
        for cand, cvec, lms in zip(candidates, cand_vecs, lm_scores):
            ov_subj = _stem_overlap(subj, cand)
            ov_msg = _stem_overlap(msg, cand)
            ov = max(ov_subj, ov_msg)
            subj_cos = _cosine_list(subj_vec, cvec)
            msg_cos = _cosine_list(msg_vec, cvec)
            cos = 0.75 * subj_cos + 0.25 * msg_cos
            is_def = any(cue in cand.lower() for cue in ("znaczenie słowa", " to ", " jest ", " – ", " — ", "oznacza", "składa się"))
            def_bonus = 0.14 if is_def else 0.0
            lm_bonus = max(-0.05, min(0.08, (lms + 6.0) * 0.015))
            total = 0.48 * ov + 0.36 * cos + def_bonus + lm_bonus
            ranked.append((total, ov, subj_cos, cand, cvec))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top_score, top_ov, top_subj_cos, _, _ = ranked[0]

        # Anti-hallucination gate: require genuine relevance to the asked subject
        if top_ov < 0.25 and top_subj_cos < 0.52:
            if is_non_topic_phrase:
                return (
                    "Nie do końca rozumiem, o jaki temat pytasz. Zadaj proszę pytanie o konkretne pojęcie "
                    "albo wpisz `/szkola`, abym wgrał podstawową wiedzę szkolną.",
                    None,
                )
            return (
                f"Nie jestem pewien odpowiedzi na temat „{subj}” — nie znalazłem pasujących faktów w pamięci. "
                f"Użyj `/ucz {subj}`, abym zbadał ten temat w internecie!",
                subj,
            )

        relevant = [item for item in ranked if item[1] >= 0.25 or item[2] >= max(0.50, top_subj_cos - 0.06)]
        chosen_texts: list[str] = []
        chosen_vecs: list[list[float]] = []
        max_facts = 3 if any(w in low for w in ("krotko", "w skrocie")) else 5

        for _, _, _, cand, cvec in relevant:
            if any(_cosine_list(cvec, pv) > 0.84 for pv in chosen_vecs):
                continue
            chosen_texts.append(cand)
            chosen_vecs.append(cvec)
            if len(chosen_texts) >= max_facts:
                break

        if not chosen_texts:
            chosen_texts = [ranked[0][3]]

        lead = chosen_texts[0].rstrip(".") + "."
        extras = [t.rstrip(".") + "." for t in chosen_texts[1:]]

        subj_title = (subj[0].upper() + subj[1:]) if subj else subj
        first_word = lead.split(" ", 1)[0].lower() if lead else ""
        if first_word in ("to", "jest", "są", "oznacza", "był", "była", "ktoś", "osoba"):
            lead_fmt = lead[0].lower() + lead[1:]
        else:
            lead_fmt = lead
        parts: list[str] = [
            f"**{subj_title}** — {lead_fmt}"
            if not lead.lower().startswith(subj.lower()) and len(subj.split()) <= 3
            else lead
        ]
        if extras:
            parts.append(" ".join(extras))

        # Add relevant REM/graph insight if it overlaps with the subject
        rel_rem = [r for r in rem_reflections if _stem_overlap(subj, r) >= 0.25]
        if rel_rem:
            parts.append(f"\n*Skojarzenie REM:* {_clean_fact_text(rel_rem[0])}")
        elif concept_hints:
            rel_hints = [h for h in concept_hints if _stem_overlap(subj, h) >= 0.25]
            if rel_hints:
                parts.append(f"\n*Powiązania w grafie EWC:* {'; '.join(rel_hints[:2])}")

        # If asking about C++ pointers and memory confirms C++/pointer, add a crisp grounded C++ snippet
        if ("wskaznik" in low or "wskaźnik" in low) and ("c++" in low or "cpp" in low):
            parts.append(
                "\n```cpp\n"
                "int liczba = 42;\n"
                "int* wskaznik = &liczba; // przechowuje adres zmiennej\n"
                "int wartosc = *wskaznik; // dereferencja -> 42\n"
                "```"
            )

        if state_mode == "DROWSY" or energy <= self.cfg.drowsy_threshold:
            parts.append("\n*(Energia niska — przyda się `/sen`, aby utrwalić świeżą wiedzę w sieci neuronowej na GPU.)*")

        return ("\n".join(parts).strip(), None)

    def chat_reply_stream(
        self,
        user_message: str,
        recent_turns: list[dict[str, str]],
        hippocampus_memories: list[str],
        semantic_memories: list[str],
        rem_reflections: list[str],
        concept_hints: list[str],
        day: int,
        sleeps: int,
        energy: float,
        state_mode: str,
        user_profile: list[str] | None = None,
    ) -> Iterator[str]:
        reply = self.chat_reply(
            user_message=user_message,
            recent_turns=recent_turns,
            hippocampus_memories=hippocampus_memories,
            semantic_memories=semantic_memories,
            rem_reflections=rem_reflections,
            concept_hints=concept_hints,
            day=day,
            sleeps=sleeps,
            energy=energy,
            state_mode=state_mode,
            user_profile=user_profile,
        )
        # Stream smoothly in natural word chunks
        tokens = re.findall(r"\S+\s*", reply)
        if not tokens:
            if reply:
                yield reply
            return
        buf = ""
        for idx, tok in enumerate(tokens, start=1):
            buf += tok
            if idx % 3 == 0:
                yield buf
                buf = ""
        if buf:
            yield buf

    def generate_morning_greeting(
        self,
        night_number: int,
        new_mem_count: int,
        lora_trained: bool,
        rem_reflections: list[str],
        rem_links: list[dict[str, Any]],
    ) -> str:
        parts = [f"Dzień dobry! Przespałem Noc #{night_number} na GPU i utrwaliłem {new_mem_count} nowych wspomnień."]
        if rem_reflections:
            parts.append(f"Mój sen REM wyłonił wniosek: {_clean_fact_text(rem_reflections[0])}")
        elif rem_links:
            top = rem_links[0]
            parts.append(f"W grafie pojęć EWC pojawiło się nowe połączenie: {top['src']} → {top['dst']}.")
        return " ".join(parts)
