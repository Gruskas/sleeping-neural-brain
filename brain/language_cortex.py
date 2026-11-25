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
from .language_trainer import (
    BASE_HUMAN_DIALOGUES,
    build_training_dialogues,
    clean_human_fact,
)
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
    "jakie", "kto", "co", "czym", "gdzie", "kiedy", "dlaczego", "czemu", "dla", "przez",
    "przy", "nad", "pod", "przed", "bez", "jako", "jego", "jej", "ich", "tego",
    "tym", "tej", "ten", "ta", "te", "tak", "nie", "ale", "lecz", "więc", "wiec",
    "że", "ze", "się", "sie", "w", "we", "z", "na", "do", "od", "po", "o",
    "u", "za", "i", "a", "no", "krótko", "krotko", "wyjaśnij", "wyjasnij",
    "powiedz", "opowiedz", "opisz", "zdefiniuj", "napisz", "podaj", "temat",
    "czyli", "jednak", "również", "rowniez", "także", "takze", "ponieważ", "poniewaz",
    "można", "mozna", "który", "ktory", "która", "ktora", "które", "ktore",
    "których", "ktorych", "często", "czesto", "bardzo", "między", "miedzy",
    "związek", "zwiazek", "różnica", "roznica", "porównaj", "porownaj",
    "łączy", "laczy", "wpływ", "wplyw", "zależy", "zalezy", "natomiast",
    "ponadto", "wtedy", "słowo", "slowo", "oznacza", "wiele", "każdy", "kazdy",
    "inny", "inne", "innych", "tylko", "jeszcze", "będzie", "bedzie", "może", "moze",
    "był", "byl", "była", "byla", "było", "bylo", "sposób", "sposob", "polega",
    "następnie", "nastepnie", "najpierw", "potem", "później", "pozniej",
    "wcześniej", "wczesniej", "obecnie", "zwykle", "gdyż", "gdyz", "zatem",
    "równocześnie", "rownoczesnie", "jednocześnie", "jednoczesnie", "podczas",
    "pomimo", "mimo", "wobec",
}

_STOPWORDS_PL_NORM = {_strip_pl(w) for w in _STOPWORDS_PL}

_SHORT_DOMAIN_ACRONYMS = {
    "c++", "c#", "ai", "ip", "si", "pi", "o2", "co2", "h2o", "cpu", "gpu",
    "ram", "ssd", "hdd", "dna", "rna", "sql", "dns", "psu", "onz",
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


_WEAK_BRIDGE_WORDS_NORM = {
    "zycie", "zycia", "zyciu", "swiat", "swiata", "swiecie", "czas", "czasu",
    "miejsce", "miejsca", "czesc", "czesci", "forma", "formy", "rodzaj",
    "przyklad", "przykladu", "czlowiek", "czlowieka", "ludzie", "ludzi",
    "proces", "procesu", "uklad", "ukladu", "system", "systemu", "gwiazda",
    "dzialanie", "dzialania", "powietrzu", "wokol",
}


def _word_matches(a: str, b: str) -> bool:
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


def _extract_meaningful_words(text: str, keep_originals: bool = False) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9+#]{2,}", text or "")
    out: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        low_orig = tok.lower()
        norm = _strip_pl(low_orig)
        if norm.isdigit():
            continue
        if norm in _STOPWORDS_PL_NORM:
            continue
        if len(norm) < 3 and norm not in _SHORT_DOMAIN_ACRONYMS:
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(low_orig if keep_originals else norm)
    return out


def _shared_meaningful_concepts(text_a: str, text_b: str, allow_weak: bool = False) -> list[str]:
    words_a = _extract_meaningful_words(text_a, keep_originals=True)
    words_b = _extract_meaningful_words(text_b, keep_originals=False)
    if not words_a or not words_b:
        return []
    shared: list[str] = []
    seen_norm: set[str] = set()
    for wa in words_a:
        na = _strip_pl(wa)
        if na in seen_norm:
            continue
        if not allow_weak and na in _WEAK_BRIDGE_WORDS_NORM:
            continue
        if any(_word_matches(na, wb) for wb in words_b):
            seen_norm.add(na)
            shared.append(wa)
    return shared


def _lc_first(text: str) -> str:
    s = (text or "").strip()
    if len(s) > 1 and s[0].isupper() and not s[:3].isupper():
        return s[0].lower() + s[1:]
    return s


def _trim_sentence_clean(text: str, max_len: int = 165) -> str:
    clean = _clean_fact_text(text)
    if not clean:
        return ""
    if len(clean) <= max_len:
        return clean if clean[-1] in ".!?" else clean + "."
    sents = _split_polish_sentences(clean)
    if sents and len(sents[0]) <= max_len:
        acc = sents[0]
        if len(sents) > 1 and len(acc) + 1 + len(sents[1]) <= max_len:
            acc = f"{acc} {sents[1]}"
        return acc if acc[-1] in ".!?" else acc + "."
    cut = clean[:max_len]
    # Prefer clause boundary or last complete word
    for sep in ("; ", ", ", " "):
        idx = cut.rfind(sep)
        if idx >= max(28, max_len // 2):
            cut = cut[:idx]
            break
    cut = cut.rstrip(" ,;:-")
    return cut + "." if cut and cut[-1] not in ".!?" else cut


def _is_noisy_memory(text: str) -> bool:
    low = (text or "").strip().lower()
    if len(low) < 16 or low.endswith("?"):
        return True
    noise_markers = (
        "rzeczownik, rodzaj",
        "czasowniki przechodnie",
        "czasownik przechodni",
        "źródło: „https://",
        "zródło: „https://",
        "dopuszczalne w grach",
        "słownik ortograficzny",
        "odmienność tak",
        "rejestr wyraz",
    )
    return any(m in low for m in noise_markers)


def _is_bare_definition(text: str) -> bool:
    low = (text or "").strip().lower()
    return (
        low.startswith("słowo „")
        or low.startswith("znaczenie słowa")
        or " oznacza:" in low
    )


def _tokenize_stems(text: str) -> set[str]:
    stripped = _strip_pl(text.lower())
    words = re.findall(r"[a-z0-9+#]{2,}", stripped)
    stems: set[str] = set()
    for w in words:
        if w in _STOPWORDS_PL_NORM:
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
        if w not in _STOPWORDS_PL_NORM and (len(w) >= 3 or w in _SHORT_DOMAIN_ACRONYMS)
    ]
    if not q_words:
        return 0.0
    cand_words = [
        w for w in re.findall(r"[a-z0-9+#]{2,}", _strip_pl(candidate.lower()))
        if len(w) >= 2
    ]
    if not cand_words:
        return 0.0
    hits = sum(1 for qw in q_words if any(_word_matches(qw, cw) for cw in cand_words))
    return hits / max(1, len(q_words))


def _clean_fact_text(txt: str) -> str:
    cleaned = clean_human_fact(txt)
    if cleaned:
        return cleaned
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
            d_model=cfg.embed_dim,
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
        self.online_dialogues: list[str] = []
        self.last_thinking_steps: list[str] = []
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
        fluency = self.transformer_lm.fluency_score
        self.status_message = (
            f"Gotowa: {self.cfg.language_model_id} ({dev_label} · {m_params:.2f}M param. · płynność {fluency:.0f}%)"
        )

    def load_sync(self) -> bool:
        with self._op_lock:
            self.semantic_encoder.to(self.device)
            self.transformer_lm.to(self.device)
            if self.transformer_lm.trained_steps == 0:
                boot_seqs = build_training_dialogues([], [], [], [], max_total=64)
                self.tokenizer.learn_texts(boot_seqs)
                self.transformer_lm.train_sleep_replay(
                    sequences=boot_seqs,
                    tokenizer=self.tokenizer,
                    epochs=self.cfg.language_bootstrap_epochs,
                    lr=self.cfg.cortex_lr,
                    batch_size=32,
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

    def save_lora(self) -> None:
        return

    def reset_lora(self) -> None:
        with self._op_lock:
            self.tokenizer = SubwordTokenizer(max_vocab_size=self.cfg.vocab_size)
            self.semantic_encoder = CorticalSemanticEncoder(
                vocab_size=self.cfg.vocab_size,
                ngram_buckets=self.cfg.ngram_buckets,
                d_model=self.cfg.embed_dim,
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
            self.online_dialogues = []
            self._refresh_status()

    def export_state(self) -> dict[str, Any]:
        with self._op_lock:
            return {
                "tokenizer": self.tokenizer.export_state(),
                "semantic_encoder": self.semantic_encoder.export_state(),
                "transformer_lm": self.transformer_lm.export_state(),
                "online_dialogues": list(self.online_dialogues[-200:]),
            }

    def import_state(self, state: dict[str, Any]) -> None:
        with self._op_lock:
            if "tokenizer" in state:
                self.tokenizer.import_state(state["tokenizer"])
            if "semantic_encoder" in state:
                self.semantic_encoder.import_state(state["semantic_encoder"])
            if "transformer_lm" in state:
                self.transformer_lm.import_state(state["transformer_lm"])
            if "online_dialogues" in state:
                self.online_dialogues = list(state.get("online_dialogues", []))[-200:]
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

    def extract_triples_from_memory(self, text: str, topic: str | None = None) -> list[tuple[str, str, str]]:
        """Extract concise relational (src, rel, dst) triples from Polish declarative sentences."""
        clean = _clean_fact_text(text)
        if not clean or _is_noisy_memory(clean):
            return []

        rel_patterns = [
            (
                r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9+#\s-]{2,42}?)\s+"
                r"(składa się z|składają się z|należy do|należą do|polega na|krąży wokół|leży nad|"
                r"wytwarza|wytwarzają|zużywa|zużywają|wymaga|powoduje|powodują|przechowuje|przechowują|"
                r"dostarcza|dostarczają|kieruje|chroni|łączy|wykorzystuje|przeprowadza|przeprowadzają|"
                r"odkrył|odkryła|napisał|ma stolicę|ma wzór|emituje|wskazuje|umożliwia|tworzy|zawiera)\s+"
                r"([^.,;!?]{3,55})",
                None,
            ),
            (r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9+#\s-]{2,42}?)\s+(?:–|—|-|:)\s+([^.,;!?]{4,55})", "oznacza"),
            (r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9+#\s-]{2,42}?)\s+(?:to|jest|są)\s+([^.,;!?]{4,55})", "jest"),
        ]

        triples: list[tuple[str, str, str]] = []
        seen_t: set[tuple[str, str, str]] = set()
        for sent in (_split_polish_sentences(clean) or [clean]):
            for pat, fixed_rel in rel_patterns:
                m = re.match(pat, sent, flags=re.I)
                if not m:
                    continue
                s = m.group(1).strip(" „”:-").lower()[:50]
                if fixed_rel is None:
                    r = m.group(2).strip().lower()[:35]
                    d = m.group(3).strip(" „”.").lower()[:50]
                else:
                    r = fixed_rel
                    d = m.group(2).strip(" „”.").lower()[:50]
                if len(s) >= 2 and len(d) >= 3 and s != d and (s, r, d) not in seen_t:
                    seen_t.add((s, r, d))
                    triples.append((s, r, d))
                    break
            if len(triples) >= 4:
                break
        return triples

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
        noise_cues = (
            "zobacz też", "przypisy", "bibliografia", "linki zewnętrzne", "kategoria:", "isbn",
            "dopuszczalne w grach", "słownik ortograficzny", "rejestr wyraz", "odmienność tak",
            "x/twitter", "koalar", "aktualizacja 20",
        )

        scored: list[tuple[float, str, list[float], bool]] = []
        for raw_sent, vec in zip(sentences, sent_vecs):
            sent = clean_human_fact(raw_sent)
            if len(sent) < 20:
                continue
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

        for item in selected_facts:
            for tr in self.extract_triples_from_memory(item["text"], topic=topic):
                if tr not in seen_t:
                    seen_t.add(tr)
                    triples.append(tr)
                if len(triples) >= 6:
                    break
            if len(triples) >= 6:
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
            brain_name_m = re.search(
                r"(?:twoje imi[eę] to|nazywasz si[eę]|masz na imi[eę]|b[eę]d[eę] ci[eę] nazywa[cć])\s+([A-ZŁŚŻŹĆŃa-ząćęłńóśźż-]{2,30})",
                s,
                flags=re.I,
            )
            if brain_name_m:
                bval = brain_name_m.group(1).strip(" .!?,;").capitalize()
                if bval:
                    memories.append({"text": f"Mózg ma nadane przez użytkownika imię {bval}.", "importance": 5, "category": "użytkownik"})
                    triples.append(("mózg", "ma_imię", bval.lower()))
                    matched = True

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
                (r"(?:studiuj[eę]|ucz[eę] si[eę])\s+([^.,;!?]{3,60})", "Użytkownik studiuje lub uczy się {val}.", "uczy_się", 4),
                (r"(?:tworz[eę]|buduj[eę]|robi[eę]\s+projekt)\s+([^.,;!?]{3,70})", "Użytkownik pracuje nad projektem: {val}.", "tworzy", 4),
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

            if re.match(r"^(?:wczoraj|dzisiaj|dziś|ostatnio|kupi[lł]em|kupi[lł]am|zrobi[lł]em|zrobi[lł]am)\b", low_raw):
                memories.append({"text": f"Wydarzenie z życia użytkownika: {s.rstrip('.')}.", "importance": 4, "category": "wydarzenie"})
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

    def synthesize_rem_reflections(
        self,
        new_memories: list[str],
        old_memories: list[str],
        rem_links: list[dict[str, Any]] | None = None,
        existing_reflections: list[str] | None = None,
    ) -> list[str]:
        if not new_memories and not rem_links:
            return []

        pool_new = [_clean_fact_text(m) for m in new_memories if not _is_noisy_memory(m)]
        pool_new = [m for m in pool_new if m][:18]
        pool_old = [_clean_fact_text(m) for m in old_memories if not _is_noisy_memory(m)]
        pool_old = [m for m in pool_old if m][:24]
        if not pool_old and len(pool_new) >= 2:
            pool_old = list(pool_new)

        existing_clean = [_clean_fact_text(r) for r in (existing_reflections or []) if _clean_fact_text(r)]
        existing_vecs = self.encode_texts(existing_clean[-40:]) if existing_clean else []

        def is_dup_reflection(cand: str, cand_vec: list[float] | None = None) -> bool:
            clow = cand.lower()
            if any(clow == e.lower() or clow[:48] == e.lower()[:48] for e in existing_clean):
                return True
            if cand_vec and existing_vecs:
                if any(_cosine_list(cand_vec, ev) > 0.85 for ev in existing_vecs):
                    return True
            return False

        reflections: list[str] = []
        used_texts: set[str] = set()

        # Pass 1: Graph-Chain Reflections grounded in discovered REM links
        all_mems = pool_new + [m for m in pool_old if m not in pool_new]
        if rem_links:
            for link in rem_links[:6]:
                src = str(link.get("src", "")).strip()
                dst = str(link.get("dst", "")).strip()
                bridge_raw = str(link.get("bridge", "")).strip()
                if not src or not dst or src == dst:
                    continue
                bridge_node = ""
                bm = re.search(r"^(?:przez|wspólny most|wspólne źródło):\s*([^(\n]+)", bridge_raw, flags=re.I)
                if bm:
                    bridge_node = bm.group(1).strip()

                m_src = None
                m_dst = None
                for m in all_mems:
                    if m in used_texts:
                        continue
                    if _stem_overlap(src, m) >= 0.5 and (not bridge_node or _stem_overlap(bridge_node, m) >= 0.4):
                        m_src = m
                        break
                if m_src is None:
                    for m in all_mems:
                        if m not in used_texts and _stem_overlap(src, m) >= 0.5:
                            m_src = m
                            break

                for m in all_mems:
                    if m in used_texts or m == m_src:
                        continue
                    if _stem_overlap(dst, m) >= 0.5 and (not bridge_node or _stem_overlap(bridge_node, m) >= 0.4):
                        m_dst = m
                        break
                if m_dst is None:
                    for m in all_mems:
                        if m not in used_texts and m != m_src and _stem_overlap(dst, m) >= 0.5:
                            m_dst = m
                            break

                if m_src and m_dst:
                    c1 = _trim_sentence_clean(m_src, 130)
                    c2 = _trim_sentence_clean(m_dst, 130)
                    b_label = bridge_node or "wspólny kontekst"
                    refl = (
                        f"Połączenie faktów ({src} ↔ {dst} przez „{b_label}”): "
                        f"{c1} Z kolei {_lc_first(c2)}"
                    )
                elif bridge_node:
                    refl = f"Skojarzenie REM: pojęcia „{src}” i „{dst}” łączą się poprzez „{bridge_node}” ({bridge_raw})."
                else:
                    continue

                r_vec = self.encode_text(refl)
                if not is_dup_reflection(refl, r_vec):
                    reflections.append(refl)
                    existing_clean.append(refl)
                    existing_vecs.append(r_vec)
                    if m_src:
                        used_texts.add(m_src)
                    if m_dst:
                        used_texts.add(m_dst)
                if len(reflections) >= 2:
                    break

        # Pass 2: Cross-topic semantic & conceptual bridges across memories
        if pool_new and pool_old:
            vecs_new = self.encode_texts(pool_new)
            vecs_old = self.encode_texts(pool_old) if pool_old is not pool_new else vecs_new

            best_pairs: list[tuple[float, str, str, list[str]]] = []
            for mn, vn in zip(pool_new, vecs_new):
                for mo, vo in zip(pool_old, vecs_old):
                    if mn.lower() == mo.lower() or mn.lower()[:30] == mo.lower()[:30]:
                        continue
                    cos = _cosine_list(vn, vo)
                    if cos > 0.80 or _stem_overlap(mn, mo) > 0.50:
                        continue
                    shared = _shared_meaningful_concepts(mn, mo)
                    if not ((len(shared) >= 1 and 0.18 <= cos <= 0.80) or (len(shared) == 0 and 0.46 <= cos <= 0.78)):
                        continue
                    stub_bonus = 0.06 if (not _is_bare_definition(mn) and not _is_bare_definition(mo)) else 0.0
                    score = cos + min(0.32, len(shared) * 0.14) + stub_bonus
                    best_pairs.append((score, mn, mo, shared))

            best_pairs.sort(key=lambda x: x[0], reverse=True)
            for _, mn, mo, shared in best_pairs:
                if mn in used_texts or mo in used_texts:
                    continue
                c1 = _trim_sentence_clean(mn, 135)
                c2 = _trim_sentence_clean(mo, 135)
                if not c1 or not c2:
                    continue
                if shared:
                    bridge = ", ".join(shared[:2])
                    refl = f"Wniosek REM (most: {bridge}): {c1} Wiąże się to z tym, że {_lc_first(c2)}"
                else:
                    refl = f"Wniosek REM: {c1} Uzupełnia to fakt, że {_lc_first(c2)}"
                r_vec = self.encode_text(refl)
                if is_dup_reflection(refl, r_vec):
                    continue
                used_texts.add(mn)
                used_texts.add(mo)
                reflections.append(refl)
                existing_clean.append(refl)
                existing_vecs.append(r_vec)
                if len(reflections) >= 4:
                    break

        return reflections[:4]

    def check_memory_update(self, new_fact: str, candidate_old_facts: list[str]) -> int | None:
        if not candidate_old_facts:
            return None
        new_low = _strip_pl(new_fact.lower())
        singleton_prefixes = (
            "uzytkownik ma na imie ",
            "mozg ma nadane przez uzytkownika imie ",
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
        rem_links: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Real NREM Deep Sleep GPU Training (100% self-contained PyTorch):
          1) Expands tokenizer vocabulary with newly encountered Polish/code tokens
          2) Contrastively trains `CorticalSemanticEncoder` on GPU (including cross-fact bridges)
          3) Trains `CorticalTransformerLM` via Experience Replay on GPU
          4) Updates structured `core_knowledge_summary`
        """
        if not self.ready or (not new_memories and not rem_reflections):
            return {"trained": False, "samples": 0, "loss_start": 0.0, "loss_end": 0.0, "duration_ms": 0.0}

        t0 = time.perf_counter()
        sample_limit = max(self.cfg.lora_max_replay_samples, min(200, len(new_memories) + 20))
        all_items = (new_memories[:160] + replay_memories[:40])[:sample_limit]

        # Build contrastive pairs (single-fact + cross-fact bridge pairs)
        pairs: list[tuple[str, str]] = []
        cleaned_items: list[str] = []
        for item in all_items:
            clean = _clean_fact_text(item)
            if len(clean) < 8:
                continue
            cleaned_items.append(clean)
            mwords = _extract_meaningful_words(clean, keep_originals=True)
            hint = " ".join(mwords[:3]) if mwords else clean[:24]
            pairs.append((hint, clean))

        # Add cross-fact bridge contrastive pairs so the encoder embeds connected topics closer
        for i in range(min(30, len(cleaned_items))):
            fi = cleaned_items[i]
            if _is_noisy_memory(fi):
                continue
            for j in range(i + 1, min(40, len(cleaned_items))):
                fj = cleaned_items[j]
                if _is_noisy_memory(fj):
                    continue
                shared = _shared_meaningful_concepts(fi, fj)
                if shared:
                    si = _extract_meaningful_words(fi, keep_originals=True)
                    sj = _extract_meaningful_words(fj, keep_originals=True)
                    if si and sj and si[0] != sj[0]:
                        pairs.append((f"{si[0]} {sj[0]} {shared[0]}", f"{fi[:110]} {fj[:110]}"))
                        if len(pairs) >= 100:
                            break
            if len(pairs) >= 100:
                break

        for link in (rem_links or [])[:8]:
            src = str(link.get("src", "")).strip()
            dst = str(link.get("dst", "")).strip()
            bridge = str(link.get("bridge", "")).strip()
            if src and dst:
                pairs.append((f"{src} {dst}", f"{src} łączy się z {dst} ({bridge})"))

        sequences = build_training_dialogues(
            memories=all_items,
            episodic_memories=[x for x in all_items if "w rozmowie" in x.lower() or "wydarzenie" in x.lower()],
            chat_turns=[],
            user_profile=[x for x in all_items if x.lower().startswith("użytkownik ") or "imię" in x.lower()],
            max_total=320,
        )
        for refl in rem_reflections[:6]:
            clean_r = _clean_fact_text(refl)
            if len(clean_r) >= 10:
                rwords = _extract_meaningful_words(clean_r, keep_originals=True)
                sequences.append(f"<MEM> {clean_r} <CTX> sen REM <USER> czego nowego się domyśliłeś? <BOT> {clean_r}")
                if len(rwords) >= 2:
                    sequences.append(
                        f"<MEM> {clean_r} <CTX> łączenie faktów <USER> jaki jest związek między {rwords[0]} a {rwords[1]}? <BOT> {clean_r}"
                    )
        if self.online_dialogues:
            sequences.extend(self.online_dialogues[-60:])

        with self._op_lock:
            self.tokenizer.learn_texts(all_items + rem_reflections + sequences)
            enc_s, enc_e = self.semantic_encoder.train_contrastive(
                pairs=pairs[:96],
                tokenizer=self.tokenizer,
                epochs=max(4, self.cfg.cortex_sleep_epochs // 2),
                lr=self.cfg.cortex_lr,
            )
            lm_s, lm_e = self.transformer_lm.train_sleep_replay(
                sequences=sequences,
                tokenizer=self.tokenizer,
                epochs=self.cfg.cortex_sleep_epochs,
                lr=self.cfg.cortex_lr,
                batch_size=48 if len(sequences) > 64 else 24,
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

            user_pts = [x for x in combined_pool if x.lower().startswith("użytkownik ") or "zasada" in x.lower() or "imię" in x.lower()][:5]
            know_pts = [x for x in combined_pool if x not in user_pts][:7]
            sections: list[str] = []
            if user_pts:
                sections.append("1) Profil i zasady użytkownika:\n" + "\n".join(f"- {u}" for u in user_pts))
            if know_pts:
                sections.append("2) Kluczowa utrwalona wiedza:\n" + "\n".join(f"- {k[:180]}" for k in know_pts))
            self.core_knowledge_summary = "\n".join(sections)[:2400]
            self.lora_updates_count += 1
            self._refresh_status()

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
            "fluency": self.transformer_lm.fluency_score,
            "duration_ms": dur_ms,
        }

    def train_language_cortex(
        self,
        all_memories: list[str],
        episodic_memories: list[str],
        chat_turns: list[dict[str, str]],
        user_profile: list[str],
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """Self-train the GPU Transformer LM & Semantic Encoder on human Polish dialogues."""
        t0 = time.perf_counter()
        ep = max(2, int(epochs or self.cfg.language_train_epochs))
        seqs = build_training_dialogues(
            memories=all_memories,
            episodic_memories=episodic_memories,
            chat_turns=chat_turns,
            user_profile=user_profile,
            max_total=450,
        )
        if self.online_dialogues:
            seqs.extend(self.online_dialogues[-100:])

        pairs: list[tuple[str, str]] = []
        for m in all_memories[:80]:
            cm = _clean_fact_text(m)
            if len(cm) >= 12:
                words = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9+#]{3,}", cm)
                pairs.append((" ".join(words[:3]) if words else cm[:20], cm))

        with self._op_lock:
            self.tokenizer.learn_texts(seqs + all_memories[:120])
            if len(pairs) >= 2:
                self.semantic_encoder.train_contrastive(
                    pairs=pairs[:80],
                    tokenizer=self.tokenizer,
                    epochs=max(4, ep // 3),
                    lr=self.cfg.cortex_lr,
                )
            lm_s, lm_e = self.transformer_lm.train_sleep_replay(
                sequences=seqs,
                tokenizer=self.tokenizer,
                epochs=ep,
                lr=self.cfg.cortex_lr,
                batch_size=48,
            )
            self.lora_updates_count += 1
            self._refresh_status()

        dur_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "trained": True,
            "dialogues": len(seqs),
            "loss_start": lm_s,
            "loss_end": lm_e,
            "fluency": self.transformer_lm.fluency_score,
            "trained_steps": self.transformer_lm.trained_steps,
            "duration_ms": dur_ms,
        }

    def resolve_conversational_context(
        self,
        user_message: str,
        recent_turns: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Resolve pronouns and short follow-ups against recent chat messages."""
        msg = user_message.strip()
        low = _strip_pl(msg.lower()).strip(" ?!.,;:")

        active_subject = ""
        ignored_subjects = {
            "czesc", "hej", "witaj", "dzien dobry", "dobranoc", "dzieki", "dziekuje",
            "super", "ale co", "co", "dlaczego", "czemu", "jak to", "nie rozumiem",
            "o czy moge z toba rozmiac", "co potrafisz", "co umiesz", "co pamietasz",
            "co wczesniej pisalismy", "o czym rozmawialismy",
        }
        for turn in reversed(recent_turns[-8:]):
            txt = turn.get("content", "").strip()
            if not txt:
                continue
            if turn.get("role") == "user":
                cand = _extract_query_subject(txt)
                cand_low = _strip_pl(cand.lower())
                if (
                    cand
                    and len(cand) >= 3
                    and cand_low not in ignored_subjects
                    and not any(w in cand_low for w in ("pisalismy", "pisalimi", "rozmawialismy", "moge", "toba"))
                ):
                    active_subject = cand
                    break
            elif turn.get("role") == "assistant" and not active_subject:
                m_bold = re.search(r"\*\*([^*]{3,40})\*\*", txt)
                if m_bold:
                    active_subject = m_bold.group(1).strip()
                    break

        pronouns = {
            "on", "ona", "ono", "oni", "one", "niego", "niej", "nim", "nia", "nich",
            "jego", "jej", "ich", "tego", "tym", "tej", "to", "tam", "wtedy", "stamtad",
        }
        followup_starts = (
            "a ", "i ", "a kiedy", "kiedy zyl", "kiedy powstal", "kiedy byla", "ile ma",
            "ile mial", "powiedz wiecej", "powiedz cos wiecej", "cos wiecej", "wiecej o",
            "rozwin", "a z czym", "a dlaczego", "a czemu", "jak to dziala", "podaj przyklad",
        )
        words = re.findall(r"[a-z0-9+#]{2,}", low)
        has_pronoun = any(w in pronouns for w in words)
        has_followup_prefix = any(low.startswith(p) for p in followup_starts)
        is_short_followup = len(words) <= 5 and (has_pronoun or has_followup_prefix)

        if active_subject and (has_pronoun or is_short_followup):
            return {
                "is_followup": True,
                "active_subject": active_subject,
                "search_query": f"{active_subject} {msg}",
            }
        return {
            "is_followup": False,
            "active_subject": active_subject,
            "search_query": msg,
        }

    def analyze_query_intent(self, query: str) -> dict[str, Any]:
        """Classify query intent and decompose compound/multi-entity queries into sub-concepts."""
        msg = (query or "").strip()
        low = _strip_pl(msg.lower()).strip(" ?!.,;:")

        # 1) Assistant-directed capability / local action questions
        if re.match(
            r"^(?:czy\s+)?(?:ty\s+)?(?:mozesz|moglbys|moglabys|potrafisz|umiesz|masz\s+dostep|umialbys)\b",
            low,
        ):
            return {
                "intent": "self_capability",
                "detail_kind": None,
                "sub_intents": [],
                "sub_concepts": [],
                "is_compound": False,
            }

        sub_intents: list[str] = []
        if re.search(r"\bkiedy\b|\bw\s+ktorym\s+roku\b|\bjak\s+dlugo\b", low):
            sub_intents.append("when")
        if re.search(r"\bgdzie\b|\bskad\b|\bdokad\b", low):
            sub_intents.append("where")
        if re.search(r"\bile\b|\bilu\b|\bjak\s+duzo\b|\bjak\s+wiele\b", low):
            sub_intents.append("how_many")
        if re.search(r"\bkto\b|\bkim\b|\bczyim\b", low):
            sub_intents.append("who")

        # 2) Detect primary intent
        is_connection = bool(
            re.search(
                r"\b(?:zwiazek|zwiazki|laczy|lacza|powiazan\w*|wspoln\w*|relacj\w*|zaleznosc\w*|jak\s+ma\s+sie|wplyw\w*\s+na|zalezy\s+od)\b",
                low,
            )
        )
        is_comparison = bool(
            re.search(
                r"\b(?:porownaj|roznic\w*|czym\s+sie\s+rozni\w*|vs\.?|kontra|podobienstw\w*)\b",
                low,
            )
        ) or (" a czym " in low)
        is_causal = bool(
            re.search(
                r"\b(?:dlaczego|czemu|po\s+co|skad\s+sie\s+bierze|co\s+sie\s+stanie|co\s+by\s+bylo|bez\s+czego|co\s+powoduje|jak\s+wplywa|wazne\s+dla)\b",
                low,
            )
        ) or low.startswith("dlaczego ") or low.startswith("czemu ") or (" bez " in low and any(w in low for w in ("nie byloby", "nie ma", "brak", "zabrak")))
        is_how = bool(
            re.search(
                r"\b(?:jak\s+dziala\w*|w\s+jaki\s+sposob|na\s+czym\s+polega|jak\s+przebiega|jak\s+powstaje|jak\s+zachodzi|jak\s+odbywa|do\s+czego\s+sluzy)\b",
                low,
            )
        )
        is_yes_no = low.startswith("czy ") and not is_comparison and not is_connection
        is_explicit_def = bool(
            re.match(
                r"^(?:no\s+to\s+|a\s+)?(?:co\s+to\s+(?:jest|sa)|czym\s+(?:jest|sa)|kim\s+(?:jest|byl)|kto\s+to\s+(?:jest|byl)|zdefiniuj|co\s+oznacza|znaczenie\s+slowa)\b",
                low,
            )
        )

        # 3) Decompose into sub-concepts for compound / multi-entity queries
        raw_clean = msg.strip(" ?!.")
        # Strip leading framing phrases
        framing_patterns = [
            r"^(?:powiedz\s+mi|wyjaśnij(?:\s+krótko)?|wyjasnij(?:\s+krotko)?|wytłumacz|wytlumacz|opowiedz(?:\s+o)?|opisz|podaj|porównaj|porownaj)\s+",
            r"^(?:jaki\s+jest\s+związek\s+(?:między|pomiedzy|pomiędzy)|jaki\s+jest\s+zwiazek\s+(?:miedzy|pomiedzy)|co\s+łączy|co\s+laczy|jaka\s+jest\s+różnica\s+(?:między|pomiędzy)|jaka\s+jest\s+roznica\s+(?:miedzy|pomiedzy)|czym\s+się\s+różni|czym\s+sie\s+rozni)\s+",
            r"^(?:dlaczego\s+bez|dlaczego|czemu|czy|jak\s+działa|jak\s+dziala|na\s+czym\s+polega|w\s+jaki\s+sposób|w\s+jaki\s+sposob)\s+",
        ]
        body = raw_clean
        for _ in range(3):
            prev = body
            for fp in framing_patterns:
                body = re.sub(fp, "", body, flags=re.I).strip(" ?!.,;:")
            if body == prev:
                break

        # Split around compound connectors (między X a Y, X i Y, X oraz Y, X vs Y, bez X nie byłoby Y)
        clauses = re.split(
            r"\s+(?:oraz|vs\.?|kontra|a\s+także|a\s+takze)\s+|,\s*(?:a\s+)?|\s+\ba\b\s+|\s+\bi\b\s+|\s+(?:na|od|w|z|przez)\s+(?=(?:[A-ZŁŚŻŹĆŃ]|[a-ząćęłńóśźż]{4,}))",
            body,
            flags=re.I,
        )
        # Clean causal fragments like "Słońca nie byłoby tlenu"
        split_clauses: list[str] = []
        for cl in clauses:
            for sub_cl in re.split(r"\b(?:nie\s+byłoby|nie\s+byloby|nie\s+ma|powoduje|prowadzi\s+do|daje)\b", cl, flags=re.I):
                sub_cl = sub_cl.strip(" ?!.,;:")
                if sub_cl:
                    split_clauses.append(sub_cl)

        sub_concepts: list[str] = []
        seen_sc: set[str] = set()
        for cl in split_clauses:
            subj_cl = _extract_query_subject(cl)
            mwords = _extract_meaningful_words(subj_cl, keep_originals=True)
            if not mwords:
                continue
            # Preserve readable short phrase (up to 4 meaningful words)
            phrase = " ".join(mwords[:4]) if len(mwords) > 4 else subj_cl.strip(" ?!.,;:")
            # Remove leading/trailing stopwords from phrase
            p_tokens = [
                t for t in re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9+#-]+", phrase)
                if _strip_pl(t.lower()) not in _STOPWORDS_PL_NORM
            ]
            if not p_tokens:
                continue
            clean_phrase = " ".join(p_tokens[:4])
            key_norm = _strip_pl(clean_phrase.lower())
            if key_norm not in seen_sc and len(key_norm) >= 2:
                seen_sc.add(key_norm)
                sub_concepts.append(clean_phrase)

        # For single-concept explicit definitions or simple queries, keep _extract_query_subject as single concept if split collapsed/over-split
        base_subj = _extract_query_subject(msg)
        base_mwords = _extract_meaningful_words(base_subj, keep_originals=True)
        if not sub_concepts and base_mwords:
            sub_concepts = [base_subj]

        is_multi_part = (
            len(sub_intents) >= 2
            or (len(sub_concepts) >= 2 and not is_explicit_def and not is_connection and not is_comparison and not is_causal and not is_how and not is_yes_no)
            or (" i kiedy " in low or " i kto " in low or " i gdzie " in low or " i ile " in low)
        )

        if is_connection:
            intent = "connection"
        elif is_comparison:
            intent = "comparison"
        elif is_causal:
            intent = "causal_why"
        elif is_how:
            intent = "how_process"
        elif is_yes_no:
            intent = "yes_no"
        elif is_multi_part:
            intent = "multi_part"
        elif sub_intents:
            intent = "factual_detail"
        elif is_explicit_def:
            intent = "definition"
        else:
            intent = "open_question"

        # Keep single clean subject for simple explicit definitions like "co to jest kwark powabny"
        if intent == "definition" and len(base_mwords) >= 1 and len(sub_concepts) > 1 and not any(sep in low for sep in (" i ", " oraz ", ",")):
            sub_concepts = [base_subj]

        return {
            "intent": intent,
            "detail_kind": sub_intents[0] if sub_intents else None,
            "sub_intents": sub_intents,
            "sub_concepts": sub_concepts[:5],
            "is_compound": len(sub_concepts) >= 2 or len(sub_intents) >= 2,
        }

    def _format_human_learn_ack(self, mems: list[dict[str, Any]]) -> str:
        phrases: list[str] = []
        brain_named = ""
        user_named = ""
        for m in mems[:4]:
            t = m["text"]
            bm = re.search(r"Mózg ma nadane przez użytkownika imię\s+([^\s.,!?;]+)", t, flags=re.I)
            if bm:
                brain_named = bm.group(1)
                continue
            um = re.search(r"Użytkownik ma na imię\s+([^\s.,!?;]+)", t, flags=re.I)
            if um:
                user_named = um.group(1)
                continue
            conv = t.rstrip(".")
            conv = re.sub(r"^Użytkownik mieszka w\s+", "mieszkasz w ", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik pochodzi z\s+", "pochodzisz z ", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik interesuje się\s+", "interesujesz się ", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik lubi\s+", "lubisz ", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik uwielbia\s+", "uwielbiasz ", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik woli\s+", "wolisz ", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik nie lubi\s+", "nie lubisz ", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik pracuje jako/w\s+", "pracujesz jako ", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik ma\s+(\d+)\s+lat", r"masz \1 lat", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik używa\s+", "używasz ", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik studiuje lub uczy się\s+", "uczysz się ", conv, flags=re.I)
            conv = re.sub(r"^Użytkownik pracuje nad projektem:\s+", "pracujesz nad projektem ", conv, flags=re.I)
            conv = re.sub(r"^Wydarzenie z życia użytkownika:\s+", "", conv, flags=re.I)
            phrases.append(conv[0].lower() + conv[1:] if conv else conv)

        parts: list[str] = []
        if brain_named:
            parts.append(f"Super, od teraz mam na imię {brain_named} — już to sobie zapisałem!")
        if user_named:
            parts.append(f"Miło Cię poznać, {user_named}!")
        if phrases:
            joined = ", ".join(phrases)
            if parts:
                parts.append(f"Zapamiętam też, że {joined}.")
            else:
                parts.append(f"Jasne, zapamiętam sobie, że {joined}.")
        return " ".join(parts) if parts else "Jasne, już to sobie zapisałem!"

    def _try_transformer_reply(
        self,
        mem_hint: str,
        ctx_hint: str,
        user_msg: str,
        reference_texts: list[str],
    ) -> str | None:
        if self.transformer_lm.fluency_score < 45.0 or not reference_texts:
            return None
        prompt = (
            f"<MEM> {mem_hint[:220] if mem_hint else 'brak'} "
            f"<CTX> {ctx_hint[:140] if ctx_hint else 'początek rozmowy'} "
            f"<USER> {user_msg[:120]} <BOT>"
        )
        with self._op_lock:
            gen = self.transformer_lm.generate_reply(prompt, self.tokenizer, max_new_tokens=75, temperature=0.28)
        cand = gen["text"].strip()
        if len(cand) < 18 or gen["mean_prob"] < 0.50:
            return None
        ref_blob = " ".join(reference_texts)
        ref_nums = set(re.findall(r"\d+", ref_blob))
        gen_nums = set(re.findall(r"\d+", cand))
        if gen_nums and not gen_nums.issubset(ref_nums):
            return None
        if _stem_overlap(cand, ref_blob) < 0.28:
            return None
        return cand

    def _extract_entity_clause(self, fact: str, entity: str) -> str:
        """Pick the most relevant clause from a compound sentence for a specific entity."""
        clean = _clean_fact_text(fact)
        if not clean:
            return ""
        # Split compound sentences into focused clauses
        clauses = [
            c.strip(" ,;.")
            for c in re.split(
                r";\s+|,\s+(?:a|natomiast|podczas\s+gdy|zaś|jednak)\s+|,\s+(?=(?:[a-ząćęłńóśźż]+\s+){1,2}(?:przechowuje|służy|ma|jest|wykonuje|przyspiesza|dostarcza|odpowiada))",
                clean,
                flags=re.I,
            )
            if len(c.strip(" ,;.")) >= 10
        ]
        if len(clauses) <= 1:
            return clean if clean.endswith((".", "!", "?")) else clean + "."
        best = max(clauses, key=lambda cl: (_stem_overlap(entity, cl), -len(cl)))
        if _stem_overlap(entity, best) > 0.0:
            res = best[0].upper() + best[1:] if len(best) > 1 and best[0].islower() else best
            return res if res.endswith((".", "!", "?")) else res + "."
        return clean if clean.endswith((".", "!", "?")) else clean + "."

    def _deliberate_best_answer(
        self,
        msg: str,
        intent_info: dict[str, Any],
        effective_subj: str,
        is_contextual_followup: bool,
        active_subject: str,
        chosen_texts: list[str],
        bridges: list[str],
        concept_paths: list[dict[str, Any]],
        rem_reflections: list[str],
        concept_hints: list[str],
        missing_concepts: list[str],
        grounded_concepts: list[str],
        thinking: list[str],
    ) -> str:
        """
        Generate 2-4 grounded candidate answers tailored to the user's intent,
        score them on intent fit, concept coverage, connection depth, and LM fluency,
        and pick the strongest answer (recording the reasoning in `thinking`).
        """
        intent = intent_info["intent"]
        detail_kind = intent_info.get("detail_kind")
        sub_intents = intent_info.get("sub_intents", [])
        sub_concepts = intent_info.get("sub_concepts", [])
        cleaned = [clean_human_fact(t) or t.rstrip(".") + "." for t in chosen_texts if t.strip()]
        cleaned = [c if c[-1] in ".!?" else c + "." for c in cleaned]

        matching_rem: list[str] = []
        for r in rem_reflections:
            rc = clean_human_fact(r)
            if not rc:
                continue
            rc_body = re.sub(
                r"^(?:Wniosek\s+REM(?:\s*\([^)]*\))?|Połączenie\s+faktów(?:\s*\([^)]*\))?|Skojarzenie\s+REM(?:\s*\([^)]*\))?):\s*",
                "",
                rc,
                flags=re.I,
            ).strip()
            rc_for_match = re.sub(
                r"\b(?:uzupełnia\s+pamięć\s+o:|wiąże\s+się\s+to\s+z\s+tym,\s+że|z\s+kolei)\b",
                " ",
                rc_body,
                flags=re.I,
            )
            ov_sub = sum(1 for sc in sub_concepts if _stem_overlap(sc, rc_for_match) >= 0.45)
            min_sub = 2 if (intent in ("connection", "comparison") and len(sub_concepts) >= 2) else 1
            if ov_sub >= min_sub or (intent not in ("connection", "comparison") and _stem_overlap(effective_subj, rc_for_match) >= 0.30):
                if rc_body and rc_body.lower() not in " ".join(cleaned).lower():
                    matching_rem.append(rc_body)

        path_summaries = [p["summary"] for p in (concept_paths or []) if p.get("summary")]
        top_bridge = bridges[0] if bridges else ""

        candidates: list[tuple[str, str]] = []

        if intent == "connection":
            # Strategy A: Lead with discovered bridge / chain, then connect supporting facts
            f1 = _trim_sentence_clean(cleaned[0], 150)
            f2 = _trim_sentence_clean(cleaned[1], 150) if len(cleaned) > 1 else ""
            if top_bridge and f2:
                cand_conn = (
                    f"Łączy je przede wszystkim **{top_bridge}**: {f1} "
                    f"Jednocześnie {_lc_first(f2)}"
                )
            elif f2:
                cand_conn = f"{f1} Z kolei {_lc_first(f2)}"
            else:
                cand_conn = f1
            if path_summaries:
                cand_conn += f" (Łańcuch pojęć: {path_summaries[0]}.)"
            elif matching_rem:
                cand_conn += f" {matching_rem[0]}"
            candidates.append(("łączenie faktów (most + łańcuch)", cand_conn))

            if len(cleaned) >= 2:
                joined = " ".join(_trim_sentence_clean(c, 145) for c in cleaned[:3])
                prefix = f"Kluczowy związek opiera się na pojęciu **{top_bridge}**. " if top_bridge else ""
                candidates.append(("synteza powiązanych faktów", f"{prefix}{joined}"))

        elif intent == "comparison":
            # Strategy A: Focused per-entity clause contrast
            if len(sub_concepts) >= 2:
                c_parts: list[str] = []
                used_clauses: set[str] = set()
                for sc in sub_concepts[:3]:
                    best_f = max(cleaned, key=lambda f: _stem_overlap(sc, f))
                    if _stem_overlap(sc, best_f) > 0.0:
                        cl = self._extract_entity_clause(best_f, sc)
                        if cl.lower() not in used_clauses:
                            used_clauses.add(cl.lower())
                            c_parts.append(cl)
                if len(c_parts) >= 2:
                    comp_txt = f"{c_parts[0]} Natomiast {_lc_first(c_parts[1])}"
                    if len(c_parts) > 2:
                        comp_txt += f" Z kolei {_lc_first(c_parts[2])}"
                    if top_bridge:
                        comp_txt += f" Łączy je wspólny kontekst: **{top_bridge}**."
                    candidates.append(("porównanie aspektów", comp_txt))
            if len(cleaned) >= 2:
                candidates.append(
                    (
                        "zestawienie pełnych faktów",
                        f"{_trim_sentence_clean(cleaned[0], 150)} Z kolei {_lc_first(_trim_sentence_clean(cleaned[1], 150))}",
                    )
                )
            else:
                candidates.append(("zestawienie z jednego faktu", cleaned[0]))

        elif intent in ("causal_why", "how_process"):
            # Order cause/source -> bridge mechanism -> effect when multiple facts exist
            ordered = list(cleaned)
            if intent == "causal_why" and len(ordered) >= 2 and len(sub_concepts) >= 2:
                cause_sc = sub_concepts[0]
                effect_sc = sub_concepts[1]
                # Prefer a fact mentioning cause + bridge (not just an incidental orbit fact)
                ordered.sort(
                    key=lambda f: (
                        _stem_overlap(cause_sc, f) > 0 and (_stem_overlap(effect_sc, f) > 0 or (bool(top_bridge) and _stem_overlap(top_bridge, f) > 0)),
                        _stem_overlap(effect_sc, f) > 0,
                        _stem_overlap(cause_sc, f) > 0,
                    ),
                    reverse=True,
                )
            lead_f = _trim_sentence_clean(ordered[0], 165)
            if len(ordered) >= 2:
                second_f = _trim_sentence_clean(ordered[1], 160)
                if intent == "causal_why":
                    if top_bridge and _stem_overlap(top_bridge, " ".join(ordered[:2])) > 0:
                        cand_c = (
                            f"Kluczową rolę odgrywa tu **{top_bridge}**: {lead_f} "
                            f"Dlatego {_lc_first(second_f)}"
                        )
                    else:
                        cand_c = f"Wynika to z tego, że {_lc_first(lead_f)} Ponadto {_lc_first(second_f)}"
                else:
                    cand_c = f"{lead_f} Kolejny krok polega na tym, że {_lc_first(second_f)}"
                if len(ordered) >= 3 and any(_stem_overlap(sc, ordered[2]) > 0 for sc in sub_concepts[:2]):
                    cand_c += f" {_trim_sentence_clean(ordered[2], 145)}"
                candidates.append(("wyjaśnienie przyczynowo-skutkowe", cand_c))
            else:
                if intent == "causal_why":
                    candidates.append(("uzasadnienie bezpośrednie", f"Dzieje się tak dlatego, że {_lc_first(lead_f)}"))
                else:
                    candidates.append(("opis mechanizmu", lead_f))

            if matching_rem:
                candidates.append(("wyjaśnienie z wnioskiem REM", f"{lead_f} {matching_rem[0]}"))

        elif intent == "yes_no":
            # Pick the clause matching the query best to verify polarity accurately
            best_fact = cleaned[0]
            focused = self._extract_entity_clause(best_fact, effective_subj or msg)
            f_low = _strip_pl(focused.lower())
            q_low = _strip_pl(msg.lower())
            neg_in_fact = bool(re.search(r"\b(?:nie|niedozwolone|brak|nigdy|niemozliwe|bez)\b", f_low))
            neg_in_q = bool(re.search(r"\b(?:nie|brak|nigdy)\b", q_low))
            if neg_in_fact != neg_in_q:
                polarity = "**Nie.**"
            else:
                polarity = "**Tak.**"
            cand_yn = f"{polarity} {focused}"
            if len(cleaned) > 1:
                extra_yn = _trim_sentence_clean(cleaned[1], 130)
                cov_extra = sum(1 for sc in sub_concepts if _stem_overlap(sc, extra_yn) > 0)
                if extra_yn.lower() not in cand_yn.lower() and cov_extra >= min(2, len(sub_concepts)):
                    cand_yn += f" {extra_yn}"
            candidates.append(("weryfikacja tak/nie", cand_yn))
            candidates.append(("uzasadnienie opisowe", f"{focused}"))

        elif intent in ("multi_part", "factual_detail"):
            parts: list[str] = []
            seen_p: set[str] = set()
            if len(sub_concepts) >= 2:
                for sc in sub_concepts[:3]:
                    bf = max(cleaned, key=lambda f: _stem_overlap(sc, f))
                    if _stem_overlap(sc, bf) > 0.0 and bf.lower() not in seen_p:
                        seen_p.add(bf.lower())
                        parts.append(_trim_sentence_clean(bf, 155))
            for c in cleaned[:3]:
                tc = _trim_sentence_clean(c, 155)
                if tc.lower() not in seen_p:
                    seen_p.add(tc.lower())
                    parts.append(tc)
            # For when/who multi-part queries, prioritize facts containing years or named persons
            if "when" in sub_intents and len(parts) > 1:
                parts.sort(key=lambda p: bool(re.search(r"\b\d{3,4}\b", p)), reverse=True)
            direct_ans = " ".join(parts[:3])
            candidates.append(("bezpośrednia odpowiedź wielowątkowa", direct_ans))
            if len(cleaned) == 1:
                candidates.append(("odpowiedź z pamięci", f"Z tego co pamiętam, {_lc_first(cleaned[0])}"))

        else:
            # open_question or definition
            if is_contextual_followup and active_subject and active_subject.lower() not in cleaned[0].lower():
                candidates.append(("kontynuacja kontekstowa", f"Jeśli chodzi o {active_subject}, to {_lc_first(cleaned[0])}"))
            if intent == "definition":
                candidates.append(("definicja klarowna", f"Najprościej mówiąc: {_lc_first(cleaned[0])}"))
                candidates.append(("odpowiedź bezpośrednia", cleaned[0]))
            else:
                # Prefer direct synthesis over generic "Najprościej mówiąc:" for open questions
                synth = _trim_sentence_clean(cleaned[0], 165)
                if len(cleaned) > 1:
                    synth += f" Ponadto {_lc_first(_trim_sentence_clean(cleaned[1], 150))}"
                if len(cleaned) > 2:
                    synth += f" {_trim_sentence_clean(cleaned[2], 140)}"
                candidates.append(("synteza wiedzy", synth))
                candidates.append(("odpowiedź z pamięci", f"Z tego co pamiętam, {_lc_first(cleaned[0])}"))

        # Add matching REM insight or graph path if helpful and not already included
        if matching_rem and candidates:
            base_lbl, base_txt = candidates[0]
            if matching_rem[0].lower() not in base_txt.lower() and intent not in ("yes_no", "definition"):
                candidates.append((f"{base_lbl} + REM", f"{base_txt} Patrząc szerzej: {matching_rem[0]}"))

        # Optional neural Transformer LM candidate (only if genuinely fluent & grounded)
        ctx_hint = f"Rozmawialiśmy o: {active_subject}." if active_subject else f"Intencja: {intent}."
        lm_reply = self._try_transformer_reply(
            mem_hint=" ".join(cleaned[:2]),
            ctx_hint=ctx_hint,
            user_msg=msg,
            reference_texts=cleaned,
        )
        if lm_reply:
            candidates.append(("Transformer LM", lm_reply))

        # Deduplicate candidates
        uniq_cands: list[tuple[str, str]] = []
        seen_cand_txt: set[str] = set()
        for lbl, txt in candidates:
            t_clean = txt.strip()
            if t_clean and t_clean.lower() not in seen_cand_txt:
                seen_cand_txt.add(t_clean.lower())
                uniq_cands.append((lbl, t_clean))

        if not uniq_cands:
            uniq_cands = [("odpowiedź bazowa", cleaned[0] if cleaned else "")]

        # Score candidates jointly across 4 criteria
        cand_texts = [txt for _, txt in uniq_cands]
        with self._op_lock:
            lm_scores = self.transformer_lm.sequence_scores(cand_texts, self.tokenizer)
        min_lm = min(lm_scores)
        max_lm = max(lm_scores)
        lm_span = max(1e-4, max_lm - min_lm)

        scored_cands: list[tuple[float, str, str]] = []
        target_concepts = grounded_concepts or sub_concepts
        for (lbl, txt), lms in zip(uniq_cands, lm_scores):
            t_low = txt.lower()
            # 1) Intent fit (0..0.38)
            intent_fit = 0.24
            is_def_style = t_low.startswith(("najprościej mówiąc:", "słowo „", "znaczenie słowa"))
            if intent == "definition":
                intent_fit = 0.36 if is_def_style else 0.28
            else:
                if is_def_style:
                    intent_fit = 0.08  # penalize definition boilerplate on non-definition questions
                elif intent == "connection" and any(k in t_low for k in ("łączy", "łańcuch", "jednocześnie", "z kolei", "most")):
                    intent_fit = 0.37
                elif intent == "comparison":
                    if lbl.startswith("porównanie aspektów"):
                        intent_fit = 0.44
                    elif any(k in t_low for k in ("natomiast", "z kolei", "łączy je")):
                        intent_fit = 0.28
                    else:
                        intent_fit = 0.22
                elif intent in ("causal_why", "how_process") and any(k in t_low for k in ("dlatego", "wynika", "kluczową", "kolejny krok", "ponieważ", "dzięki")):
                    intent_fit = 0.36
                elif intent == "yes_no":
                    intent_fit = 0.44 if (t_low.startswith("**tak.**") or t_low.startswith("**nie.**")) else 0.18
                elif intent in ("multi_part", "factual_detail"):
                    intent_fit = 0.34
                elif intent == "open_question":
                    intent_fit = 0.33

            # 2) Concept coverage (0..0.28)
            if target_concepts:
                cov = sum(1 for sc in target_concepts if _stem_overlap(sc, txt) > 0.0) / len(target_concepts)
            else:
                cov = 1.0 if _stem_overlap(effective_subj, txt) > 0.0 else 0.5
            cov_score = 0.28 * cov

            # 3) Connection depth (0..0.20)
            conn_score = 0.06
            if top_bridge and _stem_overlap(top_bridge, txt) > 0.0:
                conn_score += 0.07
            if len(cleaned) >= 2 and sum(1 for c in cleaned[:3] if _stem_overlap(c[:50], txt) >= 0.4) >= 2:
                conn_score += 0.05
            if "patrząc szerzej" in t_low or "łańcuch pojęć" in t_low:
                conn_score += 0.02

            # 4) LM fluency (0..0.14)
            norm_lm = (lms - min_lm) / lm_span if len(uniq_cands) > 1 else 0.5
            lm_fit = 0.14 * norm_lm

            total = intent_fit + cov_score + conn_score + lm_fit
            scored_cands.append((total, lbl, txt))

        scored_cands.sort(key=lambda x: x[0], reverse=True)
        best_score, best_lbl, best_reply = scored_cands[0]

        if missing_concepts and grounded_concepts:
            missing_str = ", ".join(missing_concepts[:2])
            best_reply += (
                f"\n\n*(Brakuje mi jeszcze pewnych danych o: „{missing_str}” — "
                f"możesz użyć `/ucz {missing_concepts[0]}`, żebym uzupełnił lukę.)*"
            )

        summary_scores = ", ".join(f"{lbl}={sc:.2f}" for sc, lbl, _ in scored_cands[:3])
        thinking.append(f"Porównanie wariantów odpowiedzi ({summary_scores}) → wybrano: „{best_lbl}”.")
        return best_reply.strip()

    def _compose_human_reply(
        self,
        subj: str,
        chosen_texts: list[str],
        is_contextual_followup: bool,
        active_subject: str,
        rem_reflections: list[str] | None = None,
        concept_hints: list[str] | None = None,
    ) -> str:
        """Turn grounded memory facts into natural, conversational Polish prose ranked by the Transformer LM."""
        cleaned = [clean_human_fact(t) for t in chosen_texts if clean_human_fact(t)]
        if not cleaned:
            cleaned = [t.rstrip(".") + "." for t in chosen_texts]

        lead = cleaned[0].rstrip(".") + "."
        lead_body = lead[0].lower() + lead[1:] if len(lead) > 1 and lead[0].isupper() and not lead[:3].isupper() else lead

        if is_contextual_followup and active_subject and active_subject.lower() not in lead.lower():
            base_variants = [f"Jeśli chodzi o {active_subject}, to {lead_body}"]
        else:
            base_variants = [
                f"Najprościej mówiąc: {lead_body}",
                f"Z tego co pamiętam, {lead_body}",
                lead,
            ]

        if len(base_variants) > 1:
            with self._op_lock:
                scores = self.transformer_lm.sequence_scores(base_variants, self.tokenizer)
            best_idx = max(range(len(base_variants)), key=lambda i: scores[i])
            main_text = base_variants[best_idx]
        else:
            main_text = base_variants[0]

        if len(cleaned) > 1:
            extra = cleaned[1].rstrip(".") + "."
            extra_body = extra[0].lower() + extra[1:] if len(extra) > 1 and extra[0].isupper() and not extra[:3].isupper() else extra
            main_text = f"{main_text} Warto też dodać, że {extra_body}"
        if len(cleaned) > 2:
            extra2 = cleaned[2].rstrip(".") + "."
            main_text = f"{main_text} {extra2}"

        # Weave in a matching REM reflection or EWC graph association for deeper reasoning
        if rem_reflections:
            rel_rem = [r for r in rem_reflections if _stem_overlap(subj, r) >= 0.25]
            if rel_rem:
                r_clean = clean_human_fact(rel_rem[0])
                if r_clean and r_clean.lower() not in main_text.lower():
                    main_text = f"{main_text} Patrząc szerzej: {r_clean}"
        elif concept_hints:
            rel_hints = [h for h in concept_hints if _stem_overlap(subj, h) >= 0.25]
            if rel_hints:
                main_text = f"{main_text} (Powiązane pojęcia w grafie: {rel_hints[0]}.)"
        return main_text.strip()

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
        episodic_memories: list[str] | None = None,
        active_subject: str = "",
        concept_paths: list[dict[str, Any]] | None = None,
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
            episodic_memories=episodic_memories,
            active_subject=active_subject,
            concept_paths=concept_paths,
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
        episodic_memories: list[str] | None = None,
        active_subject: str = "",
        concept_paths: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str | None]:
        msg = user_message.strip()
        low = _strip_pl(msg.lower()).strip(" ?!.,;:")
        thinking: list[str] = [f"Analizuję wiadomość: „{msg[:90]}”."]
        self.last_thinking_steps = thinking

        # 1) Correction / disagreement / opinion reaction
        if any(
            p in low
            for p in (
                "ale to nie prawda", "to nie prawda", "nieprawda", "bzdura", "bzdury",
                "glupoty", "mylisz sie", "nie uwazam", "nie zgadzam", "zmien zdanie",
                "zmien swoje zdanie", "zmien swje zdanie", "nie masz racji",
                "zle mowisz", "zle piszesz", "co ty gadasz", "co ty piszesz",
            )
        ):
            thinking.append("Wykryto niezgodę lub korektę ze strony użytkownika — nie narzucam starego wpisu i proszę o jego wersję.")
            return (
                "Rozumiem, że masz inne zdanie — dzięki za poprawkę! Nie będę upierać się przy tamtym wpisie. "
                "Napisz proszę, jak Ty to widzisz zdaniem twierdzącym, albo podaj temat do `/ucz <temat>`, żebym sprawdził inne źródła.",
                None,
            )

        # Find user name & brain name if known
        user_name = ""
        brain_name = ""
        for mem in (user_profile or []) + hippocampus_memories + semantic_memories:
            if not user_name:
                m_name = re.search(r"Użytkownik ma na imię\s+([^\s.,!?;]+)", mem, flags=re.I)
                if m_name:
                    user_name = m_name.group(1)
            if not brain_name:
                b_name = re.search(
                    r"(?:Mózg ma nadane przez użytkownika imię|Twoje imi[eę] to)\s+([^\s.,!?;]+)",
                    mem,
                    flags=re.I,
                )
                if b_name:
                    brain_name = b_name.group(1).capitalize()
        name_part = f" {user_name}" if user_name else ""

        # 1b) Brain name question
        if any(p in low for p in ("jak masz na imie", "jak sie nazywasz", "jak ci na imie")):
            if brain_name:
                return (f"Mam na imię {brain_name} — tak mnie nazwałeś i trzymam się tego imienia! W czym mogę Ci pomóc{name_part}?", None)
            return (f"Nie nadałeś mi jeszcze imienia{name_part}. Możesz napisać np. „Twoje imię to Krzysztof”, a od razu je zapamiętam!", None)

        # 1c) Episodic recall: questions about previous messages / shared memories ("co wcześniej pisaliśmy")
        is_episodic_q = (
            any(k in low for k in ("wczesniej", "wczenij", "poprzedni", "przed chwila", "ostatnio", "naszej rozmow"))
            and any(v in low for v in ("pisal", "mowil", "rozmawi", "gadal", "wiadomosc", "waidomosc", "pamietasz"))
        ) or any(p in low for p in ("co pisalem", "o czym rozmawialismy", "co wczesniej pisalismy", "co wczenij pisalimi"))
        if is_episodic_q:
            prev_user_msgs = [
                t.get("content", "").strip()
                for t in (recent_turns or [])
                if t.get("role") == "user" and t.get("content", "").strip()
            ]
            prev_clean = [
                m for m in prev_user_msgs[-6:]
                if not any(kw in _strip_pl(m.lower()) for kw in ("pisalimi", "pisalismy", "poprzednie waidomosci"))
            ]
            ep_list = [clean_human_fact(e) for e in (episodic_memories or []) if clean_human_fact(e)]
            if prev_clean or ep_list:
                lines = [f"Jasne{name_part}, pamiętam co wcześniej pisaliśmy:"]
                for pm in prev_clean[-3:]:
                    lines.append(f"- pisałeś: „{pm}”")
                for ep in ep_list[:4]:
                    if not any(pm.lower()[:25] in ep.lower() for pm in prev_clean):
                        lines.append(f"- {ep}")
                if active_subject:
                    lines.append(f"Ostatnio rozmawialiśmy głównie o temacie: {active_subject}.")
                return ("\n".join(lines), None)
            return (f"Dopiero zaczęliśmy tę rozmowę{name_part} — napisz mi o czymś, a na pewno to zapamiętam na przyszłość!", None)

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

        # 2d) Short summary request for the previous assistant reply ("w skrócie", "krótko", "streść")
        if low in ("w skrocie", "krotko", "krocej", "stresc", "stresc to", "podsumuj", "podsumuj to"):
            last_bot = ""
            for t in reversed(recent_turns or []):
                if t.get("role") == "assistant":
                    last_bot = t.get("content", "").strip()
                    break
            if last_bot:
                clean_bot = re.sub(r"\*[^*]+:\*.*$", "", last_bot, flags=re.S).strip()
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_bot) if s.strip()]
                short_ans = sents[0] if sents else clean_bot[:220]
                return (f"W skrócie: {short_ans}", None)
            return ("Napisz najpierw pytanie lub temat, a potem mogę go streścić!", None)

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

        # 4b) Assistant-directed capability / local action questions ("mozesz pliki przeszukiwac?")
        intent_info = self.analyze_query_intent(msg)
        if intent_info["intent"] == "self_capability":
            thinking.append("Wykryto pytanie o własne możliwości działania/narzędzia asystenta — odpowiadam uczciwie o zakresie funkcji.")
            if any(k in low for k in ("plik", "folder", "katalog", "dysk", "komputer", "pulpit", "terminal", "system")):
                return (
                    f"Nie mam bezpośredniego dostępu do plików ani folderów na Twoim komputerze{name_part} — działam wyłącznie "
                    "wewnątrz własnej pamięci i modelu PyTorch w tej aplikacji.\n"
                    "Mogę jednak od razu pomóc na inne sposoby: wklej tu fragment tekstu albo zapamiętaj fakty zdaniami twierdzącymi, "
                    "użyj `/ucz <temat lub adres URL>`, a także łączę pojęcia w grafie i liczę zadania matematyczne.",
                    None,
                )
            return (
                f"Jako lokalny mózg PyTorch potrafię rozmawiać po polsku, zapamiętywać fakty, łączyć je przez graf skojarzeń i sen REM, "
                f"liczyć zadania matematyczne oraz uczyć się nowych tematów z sieci przez `/ucz <temat>` (lub Auto-sieć){name_part}.",
                None,
            )

        # 5) Short conversational follow-ups / confusion without a specific subject
        follow_up_phrases = (
            "ale co", "co", "no i", "i co", "o co chodzi", "dlaczego", "czemu",
            "jak to", "nie rozumiem", "serio",
        )
        if low in follow_up_phrases:
            if active_subject:
                return (
                    f"Chodziło mi o temat „{active_subject}”, o którym przed chwilą rozmawialiśmy. "
                    f"Napisz dokładniej, co chciałbyś w nim rozwinąć albo zapytać!",
                    None,
                )
            return (
                f"Nie jestem pewien, do czego dokładnie nawiązujesz{name_part}. Napisz proszę pełniej, o co zapytać albo co policzyć "
                "(np. `co to jest fotosynteza` albo `ile to 7 * 8`).",
                None,
            )

        # 6) Pure greeting / status question
        is_greeting = bool(
            re.match(r"^(?:czesc|hej|hejka|siema|siemanko|witaj|witam|elo|yo|dzien dobry|dobry wieczor|jak sie czujesz)[!.\s?]*$", low)
        )
        if is_greeting:
            me = f"Mam na imię {brain_name}. " if brain_name else ""
            sleep_hint = " Jestem już trochę zmęczony — przyda się `/sen`, żebym utrwalił dzisiejsze rozmowy." if energy <= self.cfg.drowsy_threshold else ""
            return (
                f"Cześć{name_part}! {me}Dzisiaj jest mój {day}. dzień pracy (energia {energy:.0f}%), "
                f"wszystko działa lokalnie na moim własnym modelu GPU. O czym dzisiaj pogadamy?{sleep_hint}",
                None,
            )

        # 7) Memory self-inspection questions ("co pamiętasz", "co o mnie wiesz", "czego się nauczyłeś")
        is_mem_query = any(
            p in low for p in ("co pamietasz", "co juz pamietasz", "co o mnie wiesz", "czego sie nauczyles", "jakie tematy")
        )
        if is_mem_query:
            all_mems = []
            seen = set()
            for x in (user_profile or []) + (episodic_memories or []) + hippocampus_memories + semantic_memories:
                c = _clean_fact_text(x)
                if c and c.lower() not in seen:
                    seen.add(c.lower())
                    all_mems.append(c)
            if not all_mems and not self.core_knowledge_summary:
                return (f"Na razie moja pamięć jest jeszcze świeża{name_part}. Możemy po prostu porozmawiać, albo wgraj pakiet `/szkola`!", None)
            lines = [f"Pamiętam już sporo rzeczy{name_part}:"]
            for item in all_mems[:8]:
                lines.append(f"- {item}")
            if rem_reflections:
                lines.append("\nA ze snu REM skojarzyłem jeszcze:")
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
            return (self._format_human_learn_ack(extracted_now["memories"]), None)

        # 10) Knowledge / multi-hop reasoning / contextual follow-up query grounded in recalled memory
        subj = _extract_query_subject(msg)
        subj_low = _strip_pl(subj.lower())
        conversational_tokens = {"moge", "mozesz", "mozemy", "toba", "mna", "ciebie", "jestem", "jestes", "rozmiac", "rozmawiac", "pogadac", "ale", "czy"}
        subj_words = [w for w in re.findall(r"[a-z0-9+#]{2,}", subj_low) if w not in _STOPWORDS_PL_NORM]

        pronoun_or_followup = any(
            w in {"on", "ona", "ono", "niego", "niej", "nim", "nia", "jego", "jej", "tego", "tym", "tej", "to", "zyl", "powstal", "wiecej"}
            for w in subj_words
        ) or len(subj_words) == 0
        is_contextual_followup = bool(active_subject) and (pronoun_or_followup or low.startswith(("a ", "i ", "powiedz cos wiecej", "rozwin")))
        effective_subj = active_subject if is_contextual_followup else subj
        eff_low = _strip_pl(effective_subj.lower())
        eff_words = [w for w in re.findall(r"[a-z0-9+#]{2,}", eff_low) if w not in _STOPWORDS_PL_NORM]

        if is_contextual_followup and active_subject:
            intent_info = self.analyze_query_intent(f"{active_subject} {msg}")
        sub_concepts: list[str] = list(intent_info.get("sub_concepts", []))
        if is_contextual_followup and active_subject and active_subject.lower() not in [s.lower() for s in sub_concepts]:
            sub_concepts = [active_subject] + sub_concepts[:4]
        intent = intent_info["intent"]
        detail_kind = intent_info.get("detail_kind")

        is_non_topic_phrase = (
            len(effective_subj) <= 2
            or (not eff_words and not sub_concepts)
            or (len(sub_concepts) <= 1 and any(w in conversational_tokens for w in eff_words))
        )

        raw_pool = hippocampus_memories + semantic_memories
        clean_candidates: list[str] = []
        fallback_candidates: list[str] = []
        seen_c: set[str] = set()
        for raw in raw_pool:
            c = _clean_fact_text(raw)
            if not c or c.lower() in seen_c:
                continue
            seen_c.add(c.lower())
            fallback_candidates.append(c)
            if not _is_noisy_memory(c):
                clean_candidates.append(c)
        candidates = clean_candidates if clean_candidates else fallback_candidates

        if not candidates:
            if is_non_topic_phrase:
                return (
                    "Nie do końca złapałem, o co pytasz. Podaj proszę konkretne pojęcie "
                    "(np. `co to jest fotosynteza`), działanie matematyczne albo wpisz `/szkola`, żebym poznał podstawy.",
                    None,
                )
            ask_topic = sub_concepts[0] if sub_concepts else effective_subj
            return (
                f"Szczerze mówiąc nie mam jeszcze w pamięci informacji o „{ask_topic}”. "
                f"Jeśli wpiszesz `/ucz {ask_topic}`, sam doczytam o tym w sieci i zaraz pogadamy!",
                ask_topic,
            )

        # Rank candidates jointly using GPU Semantic Encoder + intent fit + sub-concept coverage + Transformer LM score
        sub_list = sub_concepts[:5] if sub_concepts else [effective_subj]
        vecs = self.encode_texts([effective_subj, f"{effective_subj} {msg}"] + sub_list + candidates)
        subj_vec = vecs[0]
        msg_vec = vecs[1]
        sub_vecs = vecs[2 : 2 + len(sub_list)]
        cand_vecs = vecs[2 + len(sub_list) :]
        with self._op_lock:
            lm_scores = self.transformer_lm.sequence_scores(candidates, self.tokenizer)

        subj_mwords = _extract_meaningful_words(effective_subj)
        ranked: list[tuple[float, float, float, int, list[str], str, list[float]]] = []
        for cand, cvec, lms in zip(candidates, cand_vecs, lm_scores):
            ov_subj = _stem_overlap(effective_subj, cand)
            ov_msg = _stem_overlap(msg, cand)
            ov = max(ov_subj, 0.85 * ov_msg)

            cand_mwords = _extract_meaningful_words(cand)
            matched_subj_words = sum(
                1 for sw in subj_mwords if any(_word_matches(sw, cw) for cw in cand_mwords)
            )
            covered_subs = [
                sc for sc in sub_list
                if _stem_overlap(sc, cand) >= 0.45
                or (len(_extract_meaningful_words(sc)) == 1 and _stem_overlap(sc, cand) > 0.0)
            ]

            subj_cos = _cosine_list(subj_vec, cvec)
            msg_cos = _cosine_list(msg_vec, cvec)
            best_sub_cos = max((_cosine_list(sv, cvec) for sv in sub_vecs), default=subj_cos)
            cos = 0.55 * max(subj_cos, best_sub_cos) + 0.45 * msg_cos

            # Intent-aligned bonuses / penalties
            intent_adj = 0.0
            cand_low = cand.lower()
            is_def_cue = any(cue in cand_low for cue in ("oznacza", " to ", " jest ", " – ", " — ", "składa się", "był "))
            if intent == "definition":
                intent_adj += 0.10 if is_def_cue else 0.0
            else:
                if _is_bare_definition(cand):
                    intent_adj -= 0.18
                if intent in ("causal_why", "how_process", "connection") and any(
                    cue in cand_low for cue in ("powoduje", "wytwarza", "dostarcza", "wymaga", "dzięki", "polega", "przechowuje", "służy", "umożliwia", "tworzy")
                ):
                    intent_adj += 0.12
                if (detail_kind == "when" or "when" in intent_info.get("sub_intents", [])) and re.search(r"\b\d{3,4}\b", cand):
                    intent_adj += 0.14
                if (detail_kind == "how_many" or "how_many" in intent_info.get("sub_intents", [])) and re.search(r"\b(?:\d+|jeden|dwa|trzy|cztery|pięć|sześć|siedem|osiem|ośmiu|dziewięć|dziesięć|dwanaście)\b", cand_low):
                    intent_adj += 0.12
                if detail_kind == "where" and any(cue in cand_low for cue in ("leży", "znajduje", "nad ", "w ", "na ")):
                    intent_adj += 0.08
                if intent == "yes_no" and any(cue in cand_low for cue in ("nie ", "niedozwolone", "tak", "zawsze", "nigdy", "każdy")):
                    intent_adj += 0.10

            sub_cov_bonus = 0.10 * min(2, len(covered_subs)) if len(sub_list) >= 2 else 0.0
            lm_bonus = max(-0.04, min(0.07, (lms + 6.0) * 0.014))
            total = 0.44 * ov + 0.34 * cos + intent_adj + sub_cov_bonus + lm_bonus
            ranked.append((total, ov, max(subj_cos, best_sub_cos), matched_subj_words, covered_subs, cand, cvec))

        thinking.append(
            f"Intencja: {intent}"
            + (f" ({detail_kind})" if detail_kind else "")
            + (f" · pojęcia: {', '.join(sub_list)}" if sub_list else f" · temat: „{effective_subj}”")
            + f". Przeszukano {len(candidates)} wspomnień."
        )
        ranked.sort(key=lambda x: x[0], reverse=True)
        top_score, top_ov, top_subj_cos, top_matched_words, _, _, _ = ranked[0]

        # Grounding check: strict for single-concept queries, multi-concept aware for compound queries
        if len(sub_list) >= 2:
            facts_by_concept: dict[str, list[tuple[float, float, str, list[float]]]] = {}
            grounded_concepts: list[str] = []
            missing_concepts: list[str] = []
            path_nodes_text = " ".join(
                f"{p.get('start', '')} {p.get('end', '')}" for p in (concept_paths or [])
            )
            for sc in sub_list:
                sc_mwords = _extract_meaningful_words(sc)
                sc_in_paths = bool(path_nodes_text and _stem_overlap(sc, path_nodes_text) >= 0.45)
                sc_hits: list[tuple[float, float, str, list[float]]] = []
                for total, ov, scos, _, cov_subs, cand, cvec in ranked:
                    ov_sc = _stem_overlap(sc, cand)
                    cand_mw = _extract_meaningful_words(cand)
                    matched_cnt = sum(1 for sw in sc_mwords if any(_word_matches(sw, cw) for cw in cand_mw))
                    if len(sc_mwords) >= 2:
                        ok = matched_cnt >= 2 or (ov_sc >= 0.5 and (scos >= 0.32 or sc_in_paths))
                    else:
                        ok = ov_sc > 0.0 or sc in cov_subs
                    if ok:
                        sc_hits.append((total + 0.35 * ov_sc, ov_sc, cand, cvec))
                sc_hits.sort(key=lambda x: x[0], reverse=True)
                if sc_hits:
                    facts_by_concept[sc] = sc_hits
                    grounded_concepts.append(sc)
                else:
                    missing_concepts.append(sc)

            if not grounded_concepts:
                lookup_subj = " ".join(sub_list[:2])
                thinking.append(f"Brak ugruntowanych faktów dla pojęć ({lookup_subj}) — potrzebna nauka z sieci.")
                return (
                    f"Nie wiem jeszcze wystarczająco dużo o „{lookup_subj}”, żeby nie zmyślać. "
                    f"Wpisz `/ucz {lookup_subj}`, a chętnie się tego nauczę z internetu!",
                    lookup_subj,
                )

            # Collect bridges across graph paths, shared meaningful concepts, REM reflections & concept hints
            bridges: list[str] = []
            seen_br: set[str] = set()

            def add_bridge(b: str) -> None:
                bn = _strip_pl(b.strip().lower())
                if (
                    b
                    and len(bn) >= 3
                    and not re.match(r"^\d", bn)
                    and bn not in _STOPWORDS_PL_NORM
                    and bn not in _WEAK_BRIDGE_WORDS_NORM
                    and bn not in seen_br
                ):
                    seen_br.add(bn)
                    bridges.append(b.strip())

            for p in (concept_paths or []):
                for br in p.get("bridges", []):
                    add_bridge(br)

            # Balanced fact selection: pick top fact per grounded concept + connecting bridge facts
            chosen_texts: list[str] = []
            chosen_vecs: list[list[float]] = []
            for sc in grounded_concepts:
                for _, _, cand, cvec in facts_by_concept[sc]:
                    if cand in chosen_texts:
                        break
                    if any(_cosine_list(cvec, pv) > 0.88 for pv in chosen_vecs):
                        continue
                    chosen_texts.append(cand)
                    chosen_vecs.append(cvec)
                    break

            # Look for a bridge fact connecting >= 2 grounded facts or covering discovered graph bridges
            sub_norm_words = {_strip_pl(w) for sc in sub_list for w in _extract_meaningful_words(sc)}
            for i in range(len(chosen_texts)):
                for j in range(i + 1, len(chosen_texts)):
                    for sh in _shared_meaningful_concepts(chosen_texts[i], chosen_texts[j]):
                        if not any(_word_matches(_strip_pl(sh), sw) for sw in sub_norm_words):
                            add_bridge(sh)

            # Pull in an intermediate bridge fact if graph paths found a bridge not yet in chosen_texts
            for br in bridges[:3]:
                if len(chosen_texts) >= 4:
                    break
                if any(_stem_overlap(br, ct) >= 0.5 for ct in chosen_texts):
                    continue
                br_cands = [item for item in ranked if _stem_overlap(br, item[5]) >= 0.5]
                if br_cands:
                    _, _, _, _, _, bcand, bvec = br_cands[0]
                    if bcand not in chosen_texts and not any(_cosine_list(bvec, pv) > 0.86 for pv in chosen_vecs):
                        chosen_texts.append(bcand)
                        chosen_vecs.append(bvec)

            # Fill up to max_facts with highest-ranked remaining facts
            max_facts = 1 if any(w in low for w in ("krotko", "w skrocie")) else 4
            for _, _, _, _, _, cand, cvec in ranked:
                if len(chosen_texts) >= max_facts:
                    break
                if cand in chosen_texts or any(_cosine_list(cvec, pv) > 0.84 for pv in chosen_vecs):
                    continue
                chosen_texts.append(cand)
                chosen_vecs.append(cvec)

            # Recompute shared bridges after bridge fact insertion
            for i in range(len(chosen_texts)):
                for j in range(i + 1, len(chosen_texts)):
                    for sh in _shared_meaningful_concepts(chosen_texts[i], chosen_texts[j]):
                        if not any(_word_matches(_strip_pl(sh), sw) for sw in sub_norm_words):
                            add_bridge(sh)
            for i in range(len(chosen_texts)):
                for j in range(i + 1, len(chosen_texts)):
                    for sh in _shared_meaningful_concepts(chosen_texts[i], chosen_texts[j]):
                        add_bridge(sh)

            if concept_paths:
                thinking.append(f"Ścieżka w grafie pojęć: {concept_paths[0]['summary']}.")
            if bridges:
                thinking.append(f"Wykryte mosty łączące fakty: {', '.join(bridges[:3])}.")
            thinking.append(f"Połączono {len(chosen_texts)} faktów (pokryte pojęcia: {', '.join(grounded_concepts)}).")

            final_reply = self._deliberate_best_answer(
                msg=msg,
                intent_info=intent_info,
                effective_subj=effective_subj,
                is_contextual_followup=is_contextual_followup,
                active_subject=active_subject,
                chosen_texts=chosen_texts[:max_facts],
                bridges=bridges,
                concept_paths=concept_paths or [],
                rem_reflections=rem_reflections,
                concept_hints=concept_hints,
                missing_concepts=missing_concepts,
                grounded_concepts=grounded_concepts,
                thinking=thinking,
            )
        else:
            thinking.append(
                f"Najlepsze dopasowanie: pokrycie słów={top_ov:.2f}, podobieństwo semantyczne={top_subj_cos:.2f}, "
                f"trafione rdzenie={top_matched_words}."
            )
            no_lexical_grounding = top_ov < 0.12
            multi_word_mismatch = len(eff_words) >= 2 and top_matched_words < 2 and top_ov < 0.45
            if no_lexical_grounding or (top_ov < 0.25 and top_subj_cos < 0.55) or multi_word_mismatch:
                if is_non_topic_phrase:
                    thinking.append("Brak wystarczającego pokrycia w pamięci i fraza nie jest nazwą tematu — proszę o doprecyzowanie.")
                    return (
                        "Nie jestem pewien, o jaki dokładnie temat pytasz. Napisz proszę nazwę pojęcia "
                        "albo kliknij `🎓 Szkoła Podstawowa` (`/szkola`).",
                        None,
                    )
                lookup_subj = effective_subj
                if len(eff_words) == 1 and recent_turns and not is_contextual_followup:
                    for t in reversed(recent_turns):
                        if t.get("role") == "user":
                            prev_subj = _extract_query_subject(t.get("content", ""))
                            prev_words = [w for w in re.findall(r"[a-z0-9+#]{3,}", _strip_pl(prev_subj.lower())) if w not in _STOPWORDS_PL_NORM]
                            if prev_words and prev_subj.lower() != effective_subj.lower():
                                lookup_subj = f"{effective_subj} {prev_words[-1]}"
                            break
                thinking.append(f"Za mała pewność odpowiedzi dla „{lookup_subj}” — potrzebna nauka z sieci (Auto-sieć lub /ucz).")
                return (
                    f"Nie wiem jeszcze wystarczająco dużo o „{effective_subj}”, żeby nie zmyślać. "
                    f"Wpisz `/ucz {lookup_subj}`, a chętnie się tego nauczę z internetu!",
                    lookup_subj,
                )

            relevant = [
                item for item in ranked
                if (item[1] >= 0.30 or item[2] >= max(0.58, top_subj_cos - 0.05))
                and not (len(eff_words) >= 2 and item[3] < 2 and item[2] < 0.68)
            ]
            chosen_texts = []
            chosen_vecs = []
            max_facts = 1 if any(w in low for w in ("krotko", "w skrocie")) else 3

            for _, _, _, _, _, cand, cvec in relevant:
                if any(_cosine_list(cvec, pv) > 0.84 for pv in chosen_vecs):
                    continue
                chosen_texts.append(cand)
                chosen_vecs.append(cvec)
                if len(chosen_texts) >= max_facts:
                    break

            if not chosen_texts:
                chosen_texts = [ranked[0][5]]

            bridges = []
            for p in (concept_paths or []):
                bridges.extend(p.get("bridges", []))
            if concept_paths:
                thinking.append(f"Powiązania w grafie: {concept_paths[0]['summary']}.")

            final_reply = self._deliberate_best_answer(
                msg=msg,
                intent_info=intent_info,
                effective_subj=effective_subj,
                is_contextual_followup=is_contextual_followup,
                active_subject=active_subject,
                chosen_texts=chosen_texts,
                bridges=bridges,
                concept_paths=concept_paths or [],
                rem_reflections=rem_reflections,
                concept_hints=concept_hints,
                missing_concepts=[],
                grounded_concepts=sub_list,
                thinking=thinking,
            )

        if ("wskaznik" in low or "wskaźnik" in low) and ("c++" in low or "cpp" in low):
            final_reply += (
                "\n```cpp\n"
                "int liczba = 42;\n"
                "int* wskaznik = &liczba; // przechowuje adres zmiennej\n"
                "int wartosc = *wskaznik; // dereferencja -> 42\n"
                "```"
            )

        if state_mode == "DROWSY" or energy <= self.cfg.drowsy_threshold:
            final_reply += "\n(Jestem już lekko śpiący — przyda się `/sen`, żebym utrwalił nowe rzeczy na GPU.)"

        ctx_hint = f"Rozmawialiśmy o: {active_subject}." if active_subject else ""
        sample_seq = (
            f"<MEM> {' '.join(chosen_texts[:2])[:220]} "
            f"<CTX> {ctx_hint or 'początek rozmowy'} "
            f"<USER> {msg[:120]} <BOT> {final_reply[:220]}"
        )
        self.online_dialogues.append(sample_seq)
        if len(self.online_dialogues) > 200:
            self.online_dialogues = self.online_dialogues[-200:]

        return (final_reply, None)

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
        episodic_memories: list[str] | None = None,
        active_subject: str = "",
        concept_paths: list[dict[str, Any]] | None = None,
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
            episodic_memories=episodic_memories,
            active_subject=active_subject,
            concept_paths=concept_paths,
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
