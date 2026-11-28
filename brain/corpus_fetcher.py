from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Allowed Polish + Latin + standard punctuation characters (protects UTF-8 byte bucketing)
_ALLOWED_CHAR_RE = re.compile(
    r"^[A-Za-z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻàáâäèéêëìíîïòóôöùúûüýÿçñšžčřďťňěůĺľŕ"
    r"\s.,;:!?()\[\]\"'„”«»\-–—/+*=°%€$£§@#&\n\r\t]+$"
)
_POLISH_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
_PL_STOPWORDS_RE = re.compile(
    r"\b(?:i|w|z|na|do|się|nie|jest|że|to|jak|od|po|dla|przez|oraz|czy|ale|lub|też|już|tylko|jako|który|która|które)\b",
    flags=re.IGNORECASE,
)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]{1,120}>")
_WIKI_REF_RE = re.compile(r"\[\s*(?:\d+|przypis potrzebny|potrzebne źródło|edytuj|citation needed)\s*\]", flags=re.IGNORECASE)
_WIKI_HEADING_RE = re.compile(r"^={2,6}\s*(.+?)\s*={2,6}$", flags=re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{4,}")

_CONCEPTNET_REL_PL: dict[str, str] = {
    "IsA": "jest rodzajem",
    "PartOf": "jest częścią",
    "Causes": "powoduje",
    "UsedFor": "służy do",
    "CapableOf": "potrafi",
    "HasProperty": "ma cechę",
    "AtLocation": "znajduje się w",
    "HasPrerequisite": "wymaga",
    "MotivatedByGoal": "ma na celu",
    "Antonym": "jest przeciwieństwem",
    "Synonym": "znaczy podobnie do",
}

_WORDNET_REL_PL: dict[str, str] = {
    "hypernym": "jest rodzajem",
    "hyponym": "obejmuje pojęcie",
    "meronym": "zawiera część",
    "part_meronym": "zawiera część",
    "holonym": "należy do całości",
    "part_holonym": "należy do całości",
    "antonym": "jest przeciwieństwem",
}

CONCEPTNET_ALLOWED_RELS = set(_CONCEPTNET_REL_PL.keys())


def sanitize_polish_text(text: str) -> str:
    """Clean Wikipedia/Markdown headings, footnotes, HTML tags, URLs and whitespace."""
    if not text:
        return ""
    s = str(text)
    s = _WIKI_HEADING_RE.sub(r"\1.", s)
    s = _MD_LINK_RE.sub(r"\1", s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = _WIKI_REF_RE.sub("", s)
    s = _URL_RE.sub(" ", s)
    # Remove bullet/table pipes and collapse whitespace
    lines: list[str] = []
    for raw_ln in s.splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        if ln.startswith(("|", "{|", "|}", "!", "* ", "• ", "- ")) and len(ln) < 24:
            continue
        if ln.count("|") >= 3:
            continue
        ln = re.sub(r"^[\*\•\-\+]+\s*", "", ln)
        lines.append(ln)
    s = " ".join(lines)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def polish_text_metrics(text: str) -> dict[str, float | int]:
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))
    if n_chars == 0:
        return {"chars": 0, "utf8_bytes": 0, "pl_diacritic_ratio": 0.0, "digit_ratio": 0.0}
    n_pl = sum(1 for ch in text if ch in _POLISH_DIACRITICS)
    non_ws = sum(1 for ch in text if not ch.isspace())
    n_digits = sum(1 for ch in text if ch.isdigit())
    return {
        "chars": n_chars,
        "utf8_bytes": n_bytes,
        "pl_diacritic_ratio": round(n_pl / max(1, non_ws), 4),
        "digit_ratio": round(n_digits / max(1, non_ws), 4),
    }


def is_clean_polish_utf8(
    text: str,
    *,
    min_chars: int = 30,
    max_utf8_bytes: int = 1400,
    require_polish_signal: bool = True,
) -> bool:
    """
    Validate text for UTF-8 byte-level Broca/Wernicke training:
      - Rejects URLs, HTML/code noise, excessive digits/tables, repeated chars, non-Latin scripts.
      - Enforces UTF-8 byte budget so 32B sequence bucketing stays clean.
    """
    if not text:
        return False
    if len(text) < min_chars:
        return False
    utf8_len = len(text.encode("utf-8"))
    if utf8_len < min_chars or utf8_len > max_utf8_bytes:
        return False
    if _REPEAT_CHAR_RE.search(text):
        return False
    if "http://" in text.lower() or "https://" in text.lower() or "www." in text.lower():
        return False
    if any(tok in text for tok in ("{", "}", "\\u", "<!--", "-->", "function(", "SELECT ", "CREATE TABLE")):
        return False

    # Check non-allowed character fraction (tolerance <= 1%)
    if not _ALLOWED_CHAR_RE.match(text):
        bad = 0
        for ch in text:
            if not _ALLOWED_CHAR_RE.match(ch):
                bad += 1
        if (bad / float(len(text))) > 0.01:
            return False

    non_ws = sum(1 for ch in text if not ch.isspace())
    if non_ws == 0:
        return False
    digits = sum(1 for ch in text if ch.isdigit())
    letters = sum(1 for ch in text if ch.isalpha())
    if (digits / float(non_ws)) > 0.18:
        return False
    if (letters / float(non_ws)) < 0.55:
        return False

    if require_polish_signal and len(text) >= 50:
        has_pl_char = any(ch in _POLISH_DIACRITICS for ch in text)
        has_pl_stop = bool(_PL_STOPWORDS_RE.search(text))
        if not (has_pl_char or has_pl_stop):
            return False
    return True


def split_into_clean_chunks(text: str, *, min_chars: int = 35, max_utf8_bytes: int = 1100) -> list[str]:
    """Split long clean articles into paragraph/sentence chunks respecting UTF-8 byte budget."""
    clean = sanitize_polish_text(text)
    if not clean:
        return []
    if len(clean.encode("utf-8")) <= max_utf8_bytes:
        return [clean] if is_clean_polish_utf8(clean, min_chars=min_chars, max_utf8_bytes=max_utf8_bytes) else []

    sents = re.split(r"(?<=[.!?])\s+", clean)
    chunks: list[str] = []
    cur: list[str] = []
    cur_bytes = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        sb = len(s.encode("utf-8"))
        if sb > max_utf8_bytes:
            continue
        if cur and (cur_bytes + sb + 1) > max_utf8_bytes:
            cand = " ".join(cur)
            if is_clean_polish_utf8(cand, min_chars=min_chars, max_utf8_bytes=max_utf8_bytes):
                chunks.append(cand)
            cur = [s]
            cur_bytes = sb
        else:
            cur.append(s)
            cur_bytes += sb + (1 if cur else 0)
    if cur:
        cand = " ".join(cur)
        if is_clean_polish_utf8(cand, min_chars=min_chars, max_utf8_bytes=max_utf8_bytes):
            chunks.append(cand)
    return chunks


class CorpusSanitizer:
    """Deduplicating UTF-8 Polish sanitizer writing clean JSONL files."""

    def __init__(self, min_chars: int = 30, max_utf8_bytes: int = 1200) -> None:
        self.min_chars = int(min_chars)
        self.max_utf8_bytes = int(max_utf8_bytes)
        self.seen_hashes: set[str] = set()
        self.rejected: int = 0
        self.duplicates: int = 0

    @staticmethod
    def _hash_text(text: str) -> str:
        norm = re.sub(r"\s+", " ", text.lower()).strip()
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()

    def accept(self, text: str, *, require_polish_signal: bool = True) -> dict[str, Any] | None:
        clean = sanitize_polish_text(text)
        if not is_clean_polish_utf8(
            clean,
            min_chars=self.min_chars,
            max_utf8_bytes=self.max_utf8_bytes,
            require_polish_signal=require_polish_signal,
        ):
            self.rejected += 1
            return None
        h = self._hash_text(clean)
        if h in self.seen_hashes:
            self.duplicates += 1
            return None
        self.seen_hashes.add(h)
        m = polish_text_metrics(clean)
        return {
            "text": clean,
            "utf8_bytes": m["utf8_bytes"],
            "pl_diacritic_ratio": m["pl_diacritic_ratio"],
        }


@dataclass
class CorpusPaths:
    root: Path
    raw: Path
    processed: Path

    @classmethod
    def from_base_dir(cls, base_dir: str | Path = ".") -> "CorpusPaths":
        root = Path(base_dir).resolve() / "data" / "corpora"
        raw = root / "raw"
        processed = root / "processed"
        raw.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        return cls(root=root, raw=raw, processed=processed)


def _write_jsonl(path: Path, records: Iterator[dict[str, Any]] | list[dict[str, Any]], append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def fetch_tinystories_pl(
    paths: CorpusPaths,
    *,
    max_docs: int = 500,
    min_score: float = 3.0,
) -> dict[str, Any]:
    """
    Stage 1 warm-up: Download Dxniz/TinyStories-Multilingual and keep strictly Polish records
    (`language_code == "pl"`), prioritizing high `score`.
    """
    from datasets import load_dataset

    out_path = paths.processed / "tinystories_pl.jsonl"
    sanitizer = CorpusSanitizer(min_chars=35, max_utf8_bytes=1200)
    written = 0

    def _iter_rows() -> Iterator[dict[str, Any]]:
        try:
            ds = load_dataset("Dxniz/TinyStories-Multilingual", split="train", streaming=True)
            for row in ds:
                yield row
        except Exception:
            import requests

            url = "https://huggingface.co/datasets/Dxniz/TinyStories-Multilingual/resolve/main/tinystories.jsonl"
            with requests.get(url, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue

    with out_path.open("w", encoding="utf-8") as fh:
        for row in _iter_rows():
            if str(row.get("language_code", "")).strip().lower() != "pl":
                continue
            sc = row.get("score")
            if sc is not None:
                try:
                    if float(sc) < min_score:
                        continue
                except Exception:
                    pass
            txt = row.get("output") or row.get("text") or ""
            for chunk in split_into_clean_chunks(str(txt), min_chars=35, max_utf8_bytes=1100):
                acc = sanitizer.accept(chunk, require_polish_signal=True)
                if acc is None:
                    continue
                rec = {"source": "tinystories_pl", "stage": "stage1", "score": sc, **acc}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if max_docs > 0 and written >= max_docs:
                    return {
                        "source": "tinystories_pl",
                        "path": str(out_path),
                        "docs": written,
                        "bytes": out_path.stat().st_size,
                    }
    return {
        "source": "tinystories_pl",
        "path": str(out_path),
        "docs": written,
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


def fetch_wikipedia_pl(
    paths: CorpusPaths,
    *,
    max_docs: int = 300,
) -> dict[str, Any]:
    """
    Download authentic Polish Wikipedia articles via Hugging Face `wikimedia/wikipedia`
    (auto-detecting the `.pl` config) with fallback to `speakleash` `plwiki`.
    """
    out_path = paths.processed / "wikipedia_pl.jsonl"
    sanitizer = CorpusSanitizer(min_chars=45, max_utf8_bytes=1200)
    written = 0

    def _iter_wiki() -> Iterator[str]:
        try:
            from datasets import get_dataset_config_names, load_dataset

            cfgs = get_dataset_config_names("wikimedia/wikipedia")
            pl_cfgs = sorted([c for c in cfgs if c.endswith(".pl")], reverse=True)
            cfg_name = pl_cfgs[0] if pl_cfgs else "20231101.pl"
            ds = load_dataset("wikimedia/wikipedia", cfg_name, split="train", streaming=True)
            for row in ds:
                txt = row.get("text") or ""
                if txt:
                    yield str(txt)
            return
        except Exception:
            pass

        from speakleash import Speakleash

        sl = Speakleash(str(paths.raw / "speakleash"))
        for ds_info in sl.datasets:
            if "plwiki" in ds_info.name.lower():
                for item in ds_info.ext_data:
                    txt = item[0] if isinstance(item, tuple) else str(item)
                    if txt:
                        yield txt
                return

    with out_path.open("w", encoding="utf-8") as fh:
        for article in _iter_wiki():
            chunks = split_into_clean_chunks(article, min_chars=45, max_utf8_bytes=1100)
            for chunk in chunks[:3]:
                acc = sanitizer.accept(chunk, require_polish_signal=True)
                if acc is None:
                    continue
                rec = {"source": "wikipedia_pl", "stage": "stage2", **acc}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if max_docs > 0 and written >= max_docs:
                    return {
                        "source": "wikipedia_pl",
                        "path": str(out_path),
                        "docs": written,
                        "bytes": out_path.stat().st_size,
                    }
    return {
        "source": "wikipedia_pl",
        "path": str(out_path),
        "docs": written,
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


def fetch_polqa(
    paths: CorpusPaths,
    *,
    max_docs: int = 300,
) -> dict[str, Any]:
    """Download `ipipan/polqa` Polish open-domain Q&A with Wikipedia evidence passages."""
    import ast
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    out_path = paths.processed / "polqa.jsonl"
    sanitizer = CorpusSanitizer(min_chars=8, max_utf8_bytes=1200)
    written = 0

    pq_path = hf_hub_download(
        "ipipan/polqa",
        filename="pairs/train/0000.parquet",
        revision="refs/convert/parquet",
        repo_type="dataset",
        local_dir=str(paths.raw / "polqa"),
    )
    pf = pq.ParquetFile(pq_path)

    with out_path.open("w", encoding="utf-8") as fh:
        for batch in pf.iter_batches(batch_size=256):
            cols = batch.schema.names
            q_col = batch.column("question").to_pylist()
            a_col = batch.column("answers").to_pylist()
            p_col = batch.column("passage_text").to_pylist() if "passage_text" in cols else [""] * len(q_col)
            rel_col = batch.column("relevant").to_pylist() if "relevant" in cols else [True] * len(q_col)

            for q_raw, ans_raw, passage_raw, is_rel in zip(q_col, a_col, p_col, rel_col):
                if is_rel is False:
                    continue
                q = sanitize_polish_text(str(q_raw or ""))
                parsed_ans = ans_raw
                if isinstance(ans_raw, str) and ans_raw.startswith("[") and ans_raw.endswith("]"):
                    try:
                        parsed_ans = ast.literal_eval(ans_raw)
                    except Exception:
                        parsed_ans = ans_raw
                if isinstance(parsed_ans, (list, tuple)):
                    a = sanitize_polish_text(str(parsed_ans[0] if parsed_ans else ""))
                else:
                    a = sanitize_polish_text(str(parsed_ans or ""))
                passage = sanitize_polish_text(str(passage_raw or ""))
                if not q or not a:
                    continue
                if not is_clean_polish_utf8(q, min_chars=8, max_utf8_bytes=400, require_polish_signal=False):
                    continue
                if not is_clean_polish_utf8(a, min_chars=2, max_utf8_bytes=300, require_polish_signal=False):
                    continue
                p_clean = ""
                if passage:
                    p_chunks = split_into_clean_chunks(passage, min_chars=30, max_utf8_bytes=1000)
                    p_clean = p_chunks[0] if p_chunks else ""
                key_check = sanitizer.accept(f"{q} {a}", require_polish_signal=False)
                if key_check is None:
                    continue
                rec = {
                    "source": "polqa",
                    "stage": "hippocampus_qa",
                    "q": q,
                    "a": a,
                    "passage": p_clean,
                    "utf8_bytes": len(f"{q} {a}".encode("utf-8")),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if max_docs > 0 and written >= max_docs:
                    return {
                        "source": "polqa",
                        "path": str(out_path),
                        "docs": written,
                        "bytes": out_path.stat().st_size,
                    }

    return {
        "source": "polqa",
        "path": str(out_path),
        "docs": written,
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


def fetch_speakleash(
    paths: CorpusPaths,
    *,
    max_docs: int = 300,
) -> dict[str, Any]:
    """
    Inspect SpeakLeash manifest dynamically and stream ONLY clean curated Polish subsets
    (Wolne Lektury, Wikipedia `plwiki`, `biblioteka_nauki_pl_corpus`, `plwikisource`;
    no dirty web-crawl), filtering by SpeakLeash quality metadata (`quality == "HIGH"`
    or `quality_ai.HIGH >= 70`). Streams `.jsonl.zst` on the fly when `max_docs > 0`.
    """
    import requests
    import zstandard
    from speakleash.config_loader import ConfigLoader
    from speakleash.dataset import SpeakleashDataset

    out_path = paths.processed / "speakleash.jsonl"
    sl_dir = paths.raw / "speakleash"
    sl_dir.mkdir(parents=True, exist_ok=True)
    sanitizer = CorpusSanitizer(min_chars=40, max_utf8_bytes=1200)

    base_url = ConfigLoader.load_config()["url_datasets_text_pl"]
    manifest_resp = requests.get(base_url + "speakleash.json", timeout=20)
    manifest_resp.raise_for_status()
    all_entries = manifest_resp.json()

    preferred_tokens = ("wolne_lektury", "plwiki", "biblioteka_nauki", "plwikisource", "wikibooks", "wikinews")
    blocked_tokens = ("common_crawl", "crawl", "twitter", "reddit", "wykop", "blog", "forum")

    selected_names: list[tuple[int, str]] = []
    for entry in all_entries:
        name = str(entry.get("name") or "")
        name_l = name.lower()
        if not name or any(bt in name_l for bt in blocked_tokens):
            continue
        if any(pt in name_l for pt in preferred_tokens):
            prio = 0 if "wolne_lektury" in name_l else (1 if "biblioteka_nauki" in name_l else (2 if "plwiki" in name_l else 3))
            selected_names.append((prio, name))
    selected_names.sort(key=lambda x: x[0])

    candidates: list[tuple[int, int, SpeakleashDataset]] = []
    for prio, name in selected_names[:6]:
        ds_info = SpeakleashDataset(name, base_url, str(sl_dir))
        fsize = int(ds_info.jsonl_zst_file_size or 0)
        candidates.append((prio, fsize, ds_info))
    if max_docs > 0:
        # Prefer compact archives first for fast bounded fetches while keeping Wolne Lektury + Science + Wiki
        candidates.sort(key=lambda x: (x[0], x[1]))

    def _iter_ds_stream(ds_obj: SpeakleashDataset) -> Iterator[tuple[str, dict[str, Any]]]:
        if max_docs <= 0:
            ext = ds_obj.ext_data
            if ext is not None:
                for item in ext:
                    if isinstance(item, tuple) and len(item) >= 2:
                        yield str(item[0]), (item[1] if isinstance(item[1], dict) else {})
                    else:
                        yield str(item), {}
            return

        url = f"{ds_obj.url}{ds_obj.name}.jsonl.zst"
        with requests.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            dctx = zstandard.ZstdDecompressor()
            reader = dctx.stream_reader(resp.raw)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for raw_line in text_stream:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except Exception:
                    continue
                yield str(obj.get("text") or ""), (obj.get("meta") if isinstance(obj.get("meta"), dict) else {})

    written = 0
    per_ds_cap = max(15, max_docs // max(1, min(3, len(candidates)))) if max_docs > 0 else 0

    with out_path.open("w", encoding="utf-8") as fh:
        for _, _, ds_info in candidates:
            if max_docs > 0 and written >= max_docs:
                break
            ds_written = 0
            try:
                stream_it = _iter_ds_stream(ds_info)
            except Exception:
                continue
            for txt, meta in stream_it:
                q_tag = str(meta.get("quality", "")).upper()
                q_ai = meta.get("quality_ai")
                ai_high = float(q_ai.get("HIGH", 0.0)) if isinstance(q_ai, dict) else 0.0
                if q_tag and q_tag != "HIGH" and ai_high < 70.0:
                    continue
                if not txt or len(txt) < 50:
                    continue
                for chunk in split_into_clean_chunks(txt, min_chars=40, max_utf8_bytes=1100)[:2]:
                    acc = sanitizer.accept(chunk, require_polish_signal=True)
                    if acc is None:
                        continue
                    rec = {
                        "source": f"speakleash:{ds_info.name}",
                        "stage": "stage1" if "wolne_lektury" in ds_info.name.lower() else "stage2",
                        **acc,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    ds_written += 1
                    if (max_docs > 0 and written >= max_docs) or (per_ds_cap > 0 and ds_written >= per_ds_cap and len(candidates) > 1):
                        break
                if (max_docs > 0 and written >= max_docs) or (per_ds_cap > 0 and ds_written >= per_ds_cap and len(candidates) > 1):
                    break

    return {
        "source": "speakleash",
        "path": str(out_path),
        "docs": written,
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


def fetch_polish_dynaword(
    paths: CorpusPaths,
    *,
    max_docs: int = 300,
    revision: str = "v0.2.6",
) -> dict[str, Any]:
    """Stream curated Polish texts from `SlayerLab/polish-dynaword` (pinned revision `v0.2.6`)."""
    from datasets import load_dataset

    out_path = paths.processed / "polish_dynaword.jsonl"
    sanitizer = CorpusSanitizer(min_chars=40, max_utf8_bytes=1200)
    written = 0

    def _iter_dynaword() -> Iterator[dict[str, Any]]:
        try:
            ds = load_dataset("SlayerLab/polish-dynaword", split="train", revision=revision, streaming=True)
            for row in ds:
                yield row
            return
        except Exception:
            from huggingface_hub import HfApi, hf_hub_download
            import pyarrow.parquet as pq

            api = HfApi()
            files = [
                f for f in api.list_repo_files("SlayerLab/polish-dynaword", revision=revision, repo_type="dataset")
                if f.endswith(".parquet")
            ]
            for f in files[:3]:
                local_pq = hf_hub_download(
                    "SlayerLab/polish-dynaword",
                    filename=f,
                    revision=revision,
                    repo_type="dataset",
                    local_dir=str(paths.raw / "dynaword"),
                )
                pf = pq.ParquetFile(local_pq)
                for batch in pf.iter_batches(batch_size=128):
                    cols = batch.schema.names
                    t_col = "text" if "text" in cols else cols[0]
                    for val in batch.column(t_col).to_pylist():
                        yield {"text": val, "source": f}

    with out_path.open("w", encoding="utf-8") as fh:
        for row in _iter_dynaword():
            txt = row.get("text") or row.get("content") or ""
            sub_src = row.get("source") or row.get("dataset") or "dynaword"
            for chunk in split_into_clean_chunks(str(txt), min_chars=40, max_utf8_bytes=1100)[:2]:
                acc = sanitizer.accept(chunk, require_polish_signal=True)
                if acc is None:
                    continue
                rec = {
                    "source": f"polish_dynaword:{sub_src}",
                    "stage": "stage2",
                    **acc,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if max_docs > 0 and written >= max_docs:
                    return {
                        "source": "polish_dynaword",
                        "path": str(out_path),
                        "docs": written,
                        "bytes": out_path.stat().st_size,
                    }

    return {
        "source": "polish_dynaword",
        "path": str(out_path),
        "docs": written,
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


def fetch_finepdfs_edu_pl(
    paths: CorpusPaths,
    *,
    max_docs: int = 300,
) -> dict[str, Any]:
    """Stream Polish educational PDF texts (`pol_Latn`) from `HuggingFaceFW/finepdfs-edu`."""
    from datasets import load_dataset

    out_path = paths.processed / "finepdfs_edu_pl.jsonl"
    sanitizer = CorpusSanitizer(min_chars=45, max_utf8_bytes=1200)
    written = 0

    ds = load_dataset("HuggingFaceFW/finepdfs-edu", name="pol_Latn", split="train", streaming=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in ds:
            txt = row.get("text") or ""
            for chunk in split_into_clean_chunks(str(txt), min_chars=45, max_utf8_bytes=1100)[:2]:
                acc = sanitizer.accept(chunk, require_polish_signal=True)
                if acc is None:
                    continue
                rec = {
                    "source": "finepdfs_edu:pol_Latn",
                    "stage": "stage3",
                    **acc,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                if max_docs > 0 and written >= max_docs:
                    return {
                        "source": "finepdfs_edu_pl",
                        "path": str(out_path),
                        "docs": written,
                        "bytes": out_path.stat().st_size,
                    }

    return {
        "source": "finepdfs_edu_pl",
        "path": str(out_path),
        "docs": written,
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


def _clean_cn_term(uri: str) -> str:
    # /c/pl/kot_domowy -> kot domowy
    parts = uri.strip().split("/")
    if len(parts) < 4:
        return ""
    term = parts[3].replace("_", " ").strip()
    return re.sub(r"\s+", " ", term)


def fetch_conceptnet_pl(
    paths: CorpusPaths,
    *,
    max_edges: int = 2000,
    min_weight: float = 1.0,
) -> dict[str, Any]:
    """
    Stream ConceptNet 5 assertions (`conceptnet-assertions-5.7.0.csv.gz` with HTTP resume/HF fallback)
    and extract ONLY Polish `/c/pl/` <-> `/c/pl/` edges with weight >= `min_weight`.
    """
    import requests

    out_path = paths.processed / "conceptnet_pl_edges.jsonl"
    gz_cache = paths.raw / "conceptnet-assertions-5.7.0.csv.gz"
    s3_url = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
    hf_url = "https://huggingface.co/datasets/CleverThis/conceptnet/resolve/main/conceptnet-assertions-5.7.0.csv.gz"

    edges_written = 0
    seen_edges: set[tuple[str, str, str]] = set()

    def _process_tsv_line(line: str, fh: Any) -> bool:
        nonlocal edges_written
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 5:
            return False
        rel_raw, start, end, meta_json = parts[1], parts[2], parts[3], parts[4]
        if not (start.startswith("/c/pl/") and end.startswith("/c/pl/")):
            return False
        rel = rel_raw.rsplit("/", 1)[-1]
        if rel not in CONCEPTNET_ALLOWED_RELS:
            return False
        try:
            meta = json.loads(meta_json)
            weight = float(meta.get("weight", 1.0))
        except Exception:
            weight = 1.0
        if weight < min_weight:
            return False
        a = _clean_cn_term(start)
        b = _clean_cn_term(end)
        if not a or not b or a == b or len(a) < 2 or len(b) < 2:
            return False
        if not (_ALLOWED_CHAR_RE.match(a) and _ALLOWED_CHAR_RE.match(b)):
            return False
        key = (a.lower(), rel, b.lower())
        if key in seen_edges:
            return False
        seen_edges.add(key)
        fh.write(
            json.dumps(
                {"source": "conceptnet_pl", "a": a, "rel": rel, "b": b, "weight": round(weight, 3)},
                ensure_ascii=False,
            )
            + "\n"
        )
        edges_written += 1
        return max_edges > 0 and edges_written >= max_edges

    with out_path.open("w", encoding="utf-8") as fh:
        if gz_cache.exists() and gz_cache.stat().st_size > 1024 * 1024 and max_edges <= 0:
            with gzip.open(gz_cache, mode="rt", encoding="utf-8", errors="ignore") as gz:
                for line in gz:
                    if _process_tsv_line(line, fh):
                        break
        else:
            # Stream gzipped S3 (or HF fallback) on the fly so bounded smoke fetches finish quickly
            done = False
            for url in (s3_url, hf_url):
                try:
                    with requests.get(url, stream=True, timeout=30) as resp:
                        resp.raise_for_status()
                        decomp = zlib.decompressobj(16 + zlib.MAX_WBITS)
                        buf = ""
                        for chunk in resp.iter_content(chunk_size=131072):
                            if not chunk:
                                continue
                            data = decomp.decompress(chunk).decode("utf-8", errors="ignore")
                            buf += data
                            while "\n" in buf:
                                line, buf = buf.split("\n", 1)
                                if _process_tsv_line(line, fh):
                                    done = True
                                    break
                            if done:
                                break
                    if edges_written > 0:
                        break
                except Exception:
                    continue

    return {
        "source": "conceptnet_pl",
        "path": str(out_path),
        "edges": edges_written,
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


def fetch_plwordnet(
    paths: CorpusPaths,
    *,
    max_edges: int = 2000,
) -> dict[str, Any]:
    """
    Download Polish WordNet (`omw-pl:2.0`, fallback `omw-pl:1.4`) via `wn`
    (plus optional local `data/corpora/raw/plwordnet_4_2.xml` if placed by user)
    and extract real Polish synsets, definitions, and lexical relations.
    """
    import wn

    out_path = paths.processed / "plwordnet_edges.jsonl"
    wn_dir = paths.raw / "wn_data"
    wn_dir.mkdir(parents=True, exist_ok=True)
    wn.config.data_directory = str(wn_dir)

    xml_path = paths.raw / "plwordnet_4_2.xml"
    edges_written = 0
    seen: set[tuple[str, str, str]] = set()

    if xml_path.exists():
        try:
            import plwordnet  # type: ignore

            pw = plwordnet.load(str(xml_path))
            with out_path.open("w", encoding="utf-8") as fh:
                for syn in getattr(pw, "synsets", {}).values():
                    units = list(getattr(syn, "lexical_units", []) or [])
                    if not units:
                        continue
                    head = str(getattr(units[0], "name", "") or "").strip()
                    if len(head) < 2 or not _ALLOWED_CHAR_RE.match(head):
                        continue
                    defn = sanitize_polish_text(str(getattr(syn, "definition", "") or ""))
                    if defn and is_clean_polish_utf8(defn, min_chars=12, max_utf8_bytes=400, require_polish_signal=False):
                        k_def = (head.lower(), "definition", defn.lower())
                        if k_def not in seen:
                            seen.add(k_def)
                            fh.write(json.dumps({"source": "plwordnet_xml", "a": head, "rel": "definition", "b": defn, "weight": 1.0}, ensure_ascii=False) + "\n")
                            edges_written += 1
                            if max_edges > 0 and edges_written >= max_edges:
                                break
                if edges_written > 0:
                    return {
                        "source": "plwordnet",
                        "path": str(out_path),
                        "edges": edges_written,
                        "bytes": out_path.stat().st_size,
                    }
        except Exception:
            pass

    lex_spec = "omw-pl:2.0"
    try:
        if not wn.lexicons(lexicon="omw-pl*"):
            try:
                wn.download("omw-pl:2.0")
            except Exception:
                lex_spec = "omw-pl:1.4"
                wn.download("omw-pl:1.4")
        else:
            lex_spec = wn.lexicons(lexicon="omw-pl*")[0].specifier()
    except Exception:
        lex_spec = "omw-pl:1.4"
        wn.download("omw-pl:1.4")

    en_spec: str | None = None
    try:
        if not wn.lexicons(lexicon="omw-en*"):
            wn.download("omw-en:2.0")
        en_lex = wn.lexicons(lexicon="omw-en*")
        if en_lex:
            en_spec = en_lex[0].specifier()
    except Exception:
        en_spec = None

    wordnet = wn.Wordnet(lex_spec)
    ili_pl: dict[str, list[str]] = {}
    for syn in wordnet.synsets():
        lemmas = [l.strip() for l in syn.lemmas() if l and len(l.strip()) >= 2 and _ALLOWED_CHAR_RE.match(l.strip())]
        if lemmas and syn.ili:
            ili_pl[str(syn.ili)] = lemmas

    with out_path.open("w", encoding="utf-8") as fh:
        # 1) Traverse cross-ILI taxonomy relations (hypernym, hyponym, meronym, holonym) mapped back to Polish lemmas
        if en_spec:
            en_wn = wn.Wordnet(en_spec)
            for en_syn in en_wn.synsets():
                if not en_syn.ili or str(en_syn.ili) not in ili_pl:
                    continue
                head = ili_pl[str(en_syn.ili)][0]
                for rel_name, getter in (
                    ("hypernym", en_syn.hypernyms),
                    ("hyponym", en_syn.hyponyms),
                    ("meronym", en_syn.meronyms),
                    ("holonym", en_syn.holonyms),
                ):
                    try:
                        targets = getter()
                    except Exception:
                        targets = []
                    for tgt in targets[:4]:
                        t_ili = str(tgt.ili) if tgt.ili else ""
                        if t_ili not in ili_pl:
                            continue
                        b = ili_pl[t_ili][0]
                        if b.lower() == head.lower():
                            continue
                        key = (head.lower(), rel_name, b.lower())
                        if key in seen:
                            continue
                        seen.add(key)
                        fh.write(
                            json.dumps(
                                {"source": "plwordnet", "a": head, "rel": rel_name, "b": b, "weight": 1.0},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        edges_written += 1
                        if max_edges > 0 and edges_written >= max_edges:
                            return {
                                "source": "plwordnet",
                                "path": str(out_path),
                                "edges": edges_written,
                                "bytes": out_path.stat().st_size,
                            }

        # 2) Also extract direct Polish synset relations, definitions, and intra-synset synonyms
        for syn in wordnet.synsets():
            lemmas = [l.strip() for l in syn.lemmas() if l and len(l.strip()) >= 2 and _ALLOWED_CHAR_RE.match(l.strip())]
            if not lemmas:
                continue
            head = lemmas[0]

            defn = sanitize_polish_text(syn.definition() or "")
            if defn and is_clean_polish_utf8(defn, min_chars=12, max_utf8_bytes=400, require_polish_signal=False):
                key_def = (head.lower(), "definition", defn.lower())
                if key_def not in seen:
                    seen.add(key_def)
                    fh.write(
                        json.dumps(
                            {"source": "plwordnet", "a": head, "rel": "definition", "b": defn, "weight": 1.0},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    edges_written += 1
                    if max_edges > 0 and edges_written >= max_edges:
                        break

            for alt in lemmas[1:3]:
                if alt.lower() == head.lower():
                    continue
                key_syn = (head.lower(), "Synonym", alt.lower())
                if key_syn not in seen:
                    seen.add(key_syn)
                    fh.write(
                        json.dumps(
                            {"source": "plwordnet", "a": head, "rel": "Synonym", "b": alt, "weight": 1.0},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    edges_written += 1
                    if max_edges > 0 and edges_written >= max_edges:
                        break
            if max_edges > 0 and edges_written >= max_edges:
                break

    return {
        "source": "plwordnet",
        "path": str(out_path),
        "edges": edges_written,
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


def _edge_to_polish_sentence(a: str, rel: str, b: str) -> str:
    if rel == "definition":
        return f"{a} — {b}." if not b.endswith(".") else f"{a} — {b}"
    pred = _CONCEPTNET_REL_PL.get(rel) or _WORDNET_REL_PL.get(rel) or "łączy się z"
    return f"{a} {pred} {b}."


def build_2hop_knowledge_graph(
    paths: CorpusPaths,
    *,
    max_facts: int = 2000,
    max_2hop: int = 1000,
    max_negatives: int = 400,
) -> dict[str, Any]:
    """
    Build a directed knowledge graph from downloaded ConceptNet-PL + plWordNet edges,
    emitting:
      - `knowledge_graph_facts.jsonl`: natural Polish factual sentences from real edges
      - `knowledge_graph_2hop.jsonl`: genuine 2-hop paths (A, rel1, B, rel2, C) with no direct A-C edge
      - `knowledge_graph_negatives.jsonl`: unrelated concept pairs for honest 'nie wiem' training
    """
    edges: list[dict[str, Any]] = []
    for fname in ("conceptnet_pl_edges.jsonl", "plwordnet_edges.jsonl"):
        ep = paths.processed / fname
        if not ep.exists():
            continue
        with ep.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    edges.append(json.loads(line))
                except Exception:
                    continue

    adj: dict[str, list[tuple[str, str, float, str]]] = defaultdict(list)
    undirected: dict[str, set[str]] = defaultdict(set)
    facts_out: list[dict[str, Any]] = []
    seen_facts: set[str] = set()

    for e in edges:
        a = str(e.get("a", "")).strip()
        rel = str(e.get("rel", "")).strip()
        b = str(e.get("b", "")).strip()
        src = str(e.get("source", "graph"))
        w = float(e.get("weight", 1.0))
        if not a or not b or a.lower() == b.lower():
            continue
        sent = _edge_to_polish_sentence(a, rel, b)
        if sent not in seen_facts and len(facts_out) < max_facts:
            seen_facts.add(sent)
            facts_out.append({"source": src, "a": a, "rel": rel, "b": b, "text": sent})
        if rel != "definition":
            ak = a.lower()
            bk = b.lower()
            adj[ak].append((rel, b, w, src))
            undirected[ak].add(bk)
            undirected[bk].add(ak)

    # Canonical case lookup
    canon_label: dict[str, str] = {}
    for e in edges:
        a = str(e.get("a", "")).strip()
        b = str(e.get("b", "")).strip()
        if a:
            canon_label.setdefault(a.lower(), a)
        if b and e.get("rel") != "definition":
            canon_label.setdefault(b.lower(), b)

    paths_2hop: list[dict[str, Any]] = []
    seen_ac: set[tuple[str, str]] = set()
    for a_low, nbrs_b in adj.items():
        if len(paths_2hop) >= max_2hop:
            break
        a_lbl = canon_label.get(a_low, a_low)
        for rel1, b_lbl, _, src1 in nbrs_b:
            b_low = b_lbl.lower()
            for rel2, c_lbl, _, src2 in adj.get(b_low, ()):
                c_low = c_lbl.lower()
                if c_low == a_low or c_low == b_low:
                    continue
                if c_low in undirected.get(a_low, ()):
                    continue
                pair_k = (a_low, c_low) if a_low <= c_low else (c_low, a_low)
                if pair_k in seen_ac:
                    continue
                seen_ac.add(pair_k)
                paths_2hop.append(
                    {
                        "a": a_lbl,
                        "rel1": rel1,
                        "b": b_lbl,
                        "rel2": rel2,
                        "c": c_lbl,
                        "fact_ab": _edge_to_polish_sentence(a_lbl, rel1, b_lbl),
                        "fact_bc": _edge_to_polish_sentence(b_lbl, rel2, c_lbl),
                        "sources": [src1, src2],
                    }
                )
                if len(paths_2hop) >= max_2hop:
                    break
            if len(paths_2hop) >= max_2hop:
                break

    # Negative pairs (disconnected concepts for honest ignorance training)
    nodes = [k for k in adj.keys() if len(canon_label.get(k, "")) >= 3]
    negatives: list[dict[str, Any]] = []
    if len(nodes) >= 4:
        step = max(1, len(nodes) // max(1, min(len(nodes), max_negatives)))
        for i in range(0, len(nodes), step):
            if len(negatives) >= max_negatives:
                break
            a_low = nodes[i]
            b_low = nodes[(i + len(nodes) // 2 + 1) % len(nodes)]
            if a_low == b_low or b_low in undirected.get(a_low, ()):
                continue
            a_lbl = canon_label.get(a_low, a_low)
            b_lbl = canon_label.get(b_low, b_low)
            negatives.append(
                {
                    "a": a_lbl,
                    "c": b_lbl,
                    "q": f"Jaki jest potwierdzony związek między „{a_lbl}” a „{b_lbl}”?",
                    "a_text": f"Nie mam w swojej pamięci potwierdzonego powiązania między „{a_lbl}” a „{b_lbl}”.",
                }
            )

    f_path = paths.processed / "knowledge_graph_facts.jsonl"
    h_path = paths.processed / "knowledge_graph_2hop.jsonl"
    n_path = paths.processed / "knowledge_graph_negatives.jsonl"
    _write_jsonl(f_path, facts_out)
    _write_jsonl(h_path, paths_2hop)
    _write_jsonl(n_path, negatives)

    return {
        "facts": len(facts_out),
        "paths_2hop": len(paths_2hop),
        "negatives": len(negatives),
        "facts_path": str(f_path),
        "paths_2hop_path": str(h_path),
        "negatives_path": str(n_path),
    }


def fetch_all_polish_corpora(
    base_dir: str | Path = ".",
    *,
    max_docs_per_source: int = 200,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch all 5 Polish corpus blocks + build the 2-hop knowledge graph."""
    paths = CorpusPaths.from_base_dir(base_dir)
    selected = set(sources) if sources else {
        "tinystories", "wikipedia", "polqa", "speakleash", "dynaword", "finepdfs", "conceptnet", "plwordnet"
    }
    report: dict[str, Any] = {"corpora_dir": str(paths.root), "results": {}}
    t0 = time.perf_counter()

    if "tinystories" in selected:
        report["results"]["tinystories_pl"] = fetch_tinystories_pl(paths, max_docs=max_docs_per_source)
    if "wikipedia" in selected:
        report["results"]["wikipedia_pl"] = fetch_wikipedia_pl(paths, max_docs=max_docs_per_source)
    if "polqa" in selected:
        report["results"]["polqa"] = fetch_polqa(paths, max_docs=max_docs_per_source)
    if "speakleash" in selected:
        report["results"]["speakleash"] = fetch_speakleash(paths, max_docs=max_docs_per_source)
    if "dynaword" in selected:
        report["results"]["polish_dynaword"] = fetch_polish_dynaword(paths, max_docs=max_docs_per_source)
    if "finepdfs" in selected:
        report["results"]["finepdfs_edu_pl"] = fetch_finepdfs_edu_pl(paths, max_docs=max_docs_per_source)
    if "conceptnet" in selected:
        report["results"]["conceptnet_pl"] = fetch_conceptnet_pl(paths, max_edges=max(500, max_docs_per_source * 5))
    if "plwordnet" in selected:
        report["results"]["plwordnet"] = fetch_plwordnet(paths, max_edges=max(500, max_docs_per_source * 5))
    if "conceptnet" in selected or "plwordnet" in selected:
        report["results"]["knowledge_graph"] = build_2hop_knowledge_graph(
            paths,
            max_facts=max(500, max_docs_per_source * 6),
            max_2hop=max(200, max_docs_per_source * 3),
            max_negatives=max(100, max_docs_per_source),
        )

    report["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return report


def iter_processed_corpus_texts(
    base_dir: str | Path = ".",
    *,
    filenames: tuple[str, ...] = (
        "tinystories_pl.jsonl",
        "speakleash.jsonl",
        "wikipedia_pl.jsonl",
        "polish_dynaword.jsonl",
        "finepdfs_edu_pl.jsonl",
    ),
    max_docs: int | None = None,
) -> Iterator[str]:
    """Stream clean UTF-8 Polish documents from `data/corpora/processed/*.jsonl` round-robin."""
    proc_dir = Path(base_dir).resolve() / "data" / "corpora" / "processed"
    if not proc_dir.is_dir():
        return
    files = [proc_dir / fn for fn in filenames if (proc_dir / fn).is_file() and (proc_dir / fn).stat().st_size > 0]
    if not files:
        return

    def _iter_file(p: Path) -> Iterator[str]:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                txt = str(rec.get("text") or "").strip()
                if txt:
                    yield txt

    iters = [_iter_file(p) for p in files]
    yielded = 0
    active = list(iters)
    while active:
        nxt: list[Iterator[str]] = []
        for it in active:
            try:
                yield next(it)
                yielded += 1
                if max_docs is not None and yielded >= max_docs:
                    return
                nxt.append(it)
            except StopIteration:
                continue
        active = nxt


def load_corpus_qa_pairs(base_dir: str | Path = ".", *, max_pairs: int = 800) -> list[tuple[str, str]]:
    """Load Q&A pairs from `polqa.jsonl` and `knowledge_graph_negatives.jsonl`."""
    proc_dir = Path(base_dir).resolve() / "data" / "corpora" / "processed"
    if not proc_dir.is_dir():
        return []
    pairs: list[tuple[str, str]] = []
    for fn, a_key in (("polqa.jsonl", "a"), ("knowledge_graph_negatives.jsonl", "a_text")):
        p = proc_dir / fn
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(pairs) >= max_pairs:
                    return pairs
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                q = str(rec.get("q") or "").strip()
                a = str(rec.get(a_key) or "").strip()
                if q and a:
                    pairs.append((q, a))
    return pairs[:max_pairs]


def load_corpus_2hop_facts(base_dir: str | Path = ".", *, max_items: int = 800) -> list[dict[str, Any]]:
    """Load factual statements and 2-hop graph paths for Hippocampus ingestion."""
    proc_dir = Path(base_dir).resolve() / "data" / "corpora" / "processed"
    if not proc_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for fn, kind in (("knowledge_graph_facts.jsonl", "fact"), ("knowledge_graph_2hop.jsonl", "2hop"), ("polqa.jsonl", "polqa")):
        p = proc_dir / fn
        if not p.is_file():
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(items) >= max_items:
                    return items
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if kind == "fact" and rec.get("text"):
                    items.append({"kind": "fact", "text": str(rec["text"]), "topic": "Graf Wiedzy (ConceptNet/Słowosieć)"})
                elif kind == "2hop" and rec.get("fact_ab") and rec.get("fact_bc"):
                    items.append({
                        "kind": "2hop",
                        "text": f"{rec['fact_ab']} {rec['fact_bc']}",
                        "a": str(rec.get("a", "")),
                        "b": str(rec.get("b", "")),
                        "c": str(rec.get("c", "")),
                        "topic": "Mosty 2-hop (ConceptNet/Słowosieć)",
                    })
                elif kind == "polqa" and rec.get("passage"):
                    items.append({"kind": "polqa", "text": str(rec["passage"]), "topic": "PolQA (Wikipedia)"})
    return items[:max_items]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch & sanitize real Polish training corpora.")
    parser.add_argument("--base-dir", default=".", help="Workspace root directory")
    parser.add_argument("--max-docs-per-source", type=int, default=200, help="Max docs per source (0 = unbounded)")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional subset: tinystories wikipedia polqa speakleash dynaword finepdfs conceptnet plwordnet",
    )
    args = parser.parse_args()
    rep = fetch_all_polish_corpora(
        base_dir=args.base_dir,
        max_docs_per_source=args.max_docs_per_source,
        sources=args.sources,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
