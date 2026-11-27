from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html.parser import HTMLParser


USER_AGENT = "SleepingNeuralBrain/2.1 (Educational Local GPU Study Agent; Python urllib)"
_PL_TRANS = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def _strip_pl(text: str) -> str:
    return text.translate(_PL_TRANS)


def is_http_url(s: str) -> bool:
    low = s.strip().lower()
    return low.startswith("http://") or low.startswith("https://")


class _ReadableHTMLParser(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "iframe"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self.page_title = ""
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low = tag.lower()
        if low == "title":
            self._in_title = True
        elif low in self.SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and low in ("p", "li", "h1", "h2", "h3", "h4", "br", "div", "article"):
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        low = tag.lower()
        if low == "title":
            self._in_title = False
        elif low in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.page_title += data
        elif self._skip_depth == 0:
            txt = data.strip()
            if txt:
                self._chunks.append(txt + " ")

    def get_text(self) -> str:
        raw = html.unescape("".join(self._chunks))
        lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
        lines = [ln for ln in lines if len(ln) >= 30 and not is_web_boilerplate(ln)]
        return "\n".join(lines)


_WEB_BOILERPLATE_RE = re.compile(
    r"(?:polityk[aęi]\s+prywatno|pliki\s+cookie|\bcookies?\b|"
    r"korzystaj[aą]c\s+z\s+(?:tego|niego|strony|serwisu|witryny)|"
    r"zgadzasz\s+si[eę]\s+na|wyra[zż]asz\s+zgod[eę]\s+na|"
    r"warunki\s+(?:korzystania|u[zż]ytkowania|i\s+polityk)|"
    r"akceptuj\s+wszystk|zarz[aą]dzaj\s+zgod|\brodo\b|\bgdpr\b|"
    r"zaloguj\s+si[eę]|zarejestruj\s+si[eę]|utw[oó]rz\s+konto|"
    r"wszelkie\s+prawa\s+zastrze[zż]one|all\s+rights\s+reserved|"
    r"privacy\s+policy|terms\s+of\s+(?:service|use)|accept\s+all\s+cookies|"
    r"zapisz\s+si[eę]\s+do\s+newsletter)",
    flags=re.IGNORECASE,
)


def is_web_boilerplate(text: str) -> bool:
    """Detect cookie banners, privacy notices, login prompts, and non-content web chrome."""
    s = (text or "").strip()
    if not s:
        return True
    if _WEB_BOILERPLATE_RE.search(s):
        return True
    return False


@dataclass
class WebDocument:
    title: str
    url: str
    text: str
    source: str  # wiktionary_pl | wikipedia_pl | wikipedia_en | web


def clean_search_query(raw: str) -> str:
    """Strip conversational Polish filler so search engines receive clean topic keywords."""
    q = raw.strip()
    if is_http_url(q):
        return q
    patterns = [
        r"^(?:sam\s+si[eę]\s+naucz|naucz\s+si[eę]|dowiedz\s+si[eę]|zbadaj|sprawd[zź]|powiedz\s+mi|wyja[sś]nij|wyt[lł]umacz|poczytaj(?:\s+o)?)\s+",
        r"^(?:z\s+sieci|z\s+internetu|w\s+sieci|w\s+internecie)\s+",
        r"^(?:co\s+to\s+(?:jest|s[aą])|kto\s+to\s+(?:jest|by[lł])|czym\s+(?:jest|s[aą])|kto\s+(?:wygra[lł]|zdoby[lł]|wynalaz[lł]|napisa[lł]|stworzy[lł]|odkry[lł]|za[lł]o[zż]y[lł])|kiedy\s+(?:by[lł][aoy]?|rozpocz[eę][lł][aoy]?|zako[nń]czy[lł][aoy]?)|gdzie\s+(?:le[zż]y|jest|znajduje\s+si[eę])|jak\s+(?:dzia[lł]a|powstaj[eą]?)|sk[aą]d\s+(?:si[eę]\s+)?(?:bior[aą]|bierze|pochodz[iiaą]|wzi[eę][lł][aoy]?)|dlaczego\s+(?:istniej[aąe]?|s[aą]|jest)?|czemu\s+|czy\s+|o\s+tym\s+|o\s+|na\s+temat\s+)+",
    ]
    for _ in range(4):
        prev = q
        for pat in patterns:
            q = re.sub(pat, "", q, flags=re.IGNORECASE).strip(" .?!,;:")
        if q == prev:
            break
    # Fix common Polish phonetic typos (e.g. 'wzkaźniki' -> 'wskaźniki')
    q = re.sub(r"\bwzk", "wsk", q, flags=re.IGNORECASE)
    q = re.sub(r"\bwzp", "wsp", q, flags=re.IGNORECASE)
    q = re.sub(r"\bwzt", "wst", q, flags=re.IGNORECASE)
    q = re.sub(r"\bwskaznik([a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]*)", r"wskaźnik\1", q, flags=re.IGNORECASE)
    # Common shorthand replacements for Polish technical queries
    q = re.sub(r"\bcpp\b", "C++", q, flags=re.IGNORECASE)
    q = re.sub(r"\bnode\.?ks\b|\bnodejs\b", "Node.js", q, flags=re.IGNORECASE)
    q = re.sub(r"\bcsharp\b", "C#", q, flags=re.IGNORECASE)
    return q or raw.strip()


def _http_get(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(750_000)
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def search_wiktionary(query: str, lang: str = "pl", timeout: int = 12) -> list[WebDocument]:
    """Fetch exact dictionary definitions & etymology from Wiktionary (great for words like 'kunktator')."""
    docs: list[WebDocument] = []
    clean = clean_search_query(query)
    candidates = [clean.lower(), clean]
    if " " in clean:
        candidates = [clean.lower()]
    seen_titles: set[str] = set()
    for word in candidates:
        if not word or len(word) > 50 or word in seen_titles:
            continue
        seen_titles.add(word)
        try:
            t_enc = urllib.parse.quote(word)
            url = (
                f"https://{lang}.wiktionary.org/w/api.php?"
                f"action=query&prop=extracts|info&inprop=url&explaintext=1&exchars=4000&exlimit=max"
                f"&redirects=1&titles={t_enc}&format=json&utf8=1"
            )
            data = json.loads(_http_get(url, timeout=timeout))
            pages = data.get("query", {}).get("pages", {})
            for pid, p in pages.items():
                if str(pid) == "-1":
                    continue
                extract = (p.get("extract") or "").strip()
                title = (p.get("title") or word).strip()
                fullurl = p.get("fullurl") or f"https://{lang}.wiktionary.org/wiki/{urllib.parse.quote(title)}"
                if len(extract) >= 60:
                    docs.append(
                        WebDocument(
                            title=f"Słownik ({lang.upper()}): {title}",
                            url=fullurl,
                            text=extract[:4000],
                            source=f"wiktionary_{lang}",
                        )
                    )
                    return docs
        except Exception:
            continue
    return docs


def _fetch_one_wiki_extract(title: str, lang: str, timeout: int) -> WebDocument | None:
    try:
        t_enc = urllib.parse.quote(title)
        extract_url = (
            f"https://{lang}.wikipedia.org/w/api.php?"
            f"action=query&prop=extracts|info&inprop=url&explaintext=1&exchars=14000&exlimit=max"
            f"&redirects=1&titles={t_enc}&format=json&utf8=1"
        )
        ex_data = json.loads(_http_get(extract_url, timeout=timeout))
        pages = ex_data.get("query", {}).get("pages", {})
        for pid, p in pages.items():
            if str(pid) == "-1":
                continue
            extract = (p.get("extract") or "").strip()
            real_title = (p.get("title") or title).strip()
            fullurl = p.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(real_title)}"
            if len(extract) >= 100:
                return WebDocument(
                    title=real_title,
                    url=fullurl,
                    text=extract,
                    source=f"wikipedia_{lang}",
                )
    except Exception:
        return None
    return None


def _fetch_wiki_extracts(titles: list[str], lang: str, timeout: int, max_workers: int = 3) -> list[WebDocument]:
    """Fetch full plain-text extracts for each Wikipedia title individually (avoids MediaWiki exlimit=1 truncation!)."""
    if not titles:
        return []
    if len(titles) == 1:
        doc = _fetch_one_wiki_extract(titles[0], lang, timeout)
        return [doc] if doc is not None else []
    docs: list[WebDocument] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(titles))) as pool:
        for doc in pool.map(lambda t: _fetch_one_wiki_extract(t, lang, timeout), titles):
            if doc is not None:
                docs.append(doc)
    return docs


def search_wikipedia(query: str, lang: str = "pl", limit: int = 3, timeout: int = 12) -> list[WebDocument]:
    """Search Wikipedia API with automatic typo-suggestion retry and per-title full extraction."""
    clean = clean_search_query(query)
    if not clean:
        return []
    try:
        q_enc = urllib.parse.quote(clean)
        search_url = (
            f"https://{lang}.wikipedia.org/w/api.php?"
            f"action=query&list=search&srsearch={q_enc}&srlimit={limit}&srinfo=suggestion&format=json&utf8=1"
        )
        data = json.loads(_http_get(search_url, timeout=timeout))
        hits = data.get("query", {}).get("search", [])

        # Automatically follow Wikipedia spelling suggestion if no hits (e.g. "wzkaźniki w cpp" -> "wskaźnik w cpp")
        if not hits:
            suggestion = data.get("query", {}).get("searchinfo", {}).get("suggestion")
            if suggestion:
                s_enc = urllib.parse.quote(suggestion)
                retry_url = (
                    f"https://{lang}.wikipedia.org/w/api.php?"
                    f"action=query&list=search&srsearch={s_enc}&srlimit={limit}&format=json&utf8=1"
                )
                data = json.loads(_http_get(retry_url, timeout=timeout))
                hits = data.get("query", {}).get("search", [])

        titles = [h["title"] for h in hits if h.get("title")][:limit]
        if not titles:
            return []
        return _fetch_wiki_extracts(titles, lang=lang, timeout=timeout)
    except Exception:
        return []


def fetch_random_wikipedia_titles(lang: str = "pl", limit: int = 15, timeout: int = 10) -> list[str]:
    """Fetch fresh random main-namespace Wikipedia article titles so Auto-nauka has infinite topics."""
    lim = max(1, min(30, int(limit)))
    try:
        url = (
            f"https://{lang}.wikipedia.org/w/api.php?"
            f"action=query&list=random&rnnamespace=0&rnlimit={lim}&format=json&utf8=1"
        )
        data = json.loads(_http_get(url, timeout=timeout))
        items = data.get("query", {}).get("random", [])
        out: list[str] = []
        for it in items:
            t = str(it.get("title") or "").strip()
            if len(t) >= 3 and ":" not in t and "(ujednoznacznienie)" not in t.lower():
                out.append(t)
        return out
    except Exception:
        return []


def search_duckduckgo_urls(query: str, limit: int = 4, timeout: int = 12) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    clean = clean_search_query(query)
    for base in ("https://lite.duckduckgo.com/lite/?q=", "https://html.duckduckgo.com/html/?q="):
        try:
            url = base + urllib.parse.quote(clean)
            page = _http_get(url, timeout=timeout)
            matches = re.findall(
                r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                page,
                flags=re.IGNORECASE | re.DOTALL,
            )
            seen: set[str] = set()
            for href, raw_title in matches:
                href = html.unescape(href)
                if "uddg=" in href:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    real_urls = parsed.get("uddg")
                    if real_urls:
                        href = real_urls[0]
                if not href.startswith("http"):
                    continue
                if any(
                    bad in href.lower()
                    for bad in ("duckduckgo.com", "youtube.com", "facebook.com", "instagram.com", "tiktok.com", ".pdf")
                ):
                    continue
                if href in seen:
                    continue
                seen.add(href)
                title = re.sub(r"<[^>]+>", "", html.unescape(raw_title)).strip() or href
                if len(title) < 4:
                    continue
                results.append((title, href))
                if len(results) >= limit:
                    return results
        except Exception:
            continue
    return results


def fetch_web_page(title: str, url: str, timeout: int = 12) -> WebDocument | None:
    try:
        raw_html = _http_get(url, timeout=timeout)
        parser = _ReadableHTMLParser()
        parser.feed(raw_html)
        text = parser.get_text()
        if len(text) < 200:
            return None
        parsed_title = re.sub(r"\s+", " ", parser.page_title).strip()
        final_title = parsed_title if (parsed_title and (not title or is_http_url(title))) else title
        return WebDocument(title=final_title, url=url, text=text[:8500], source="web")
    except Exception:
        return None


def extract_lead_definition(doc: WebDocument) -> str | None:
    """
    Extract the authoritative verbatim Polish opening definition from Wiktionary or Wikipedia
    so core names, dates, and meanings are NEVER lost during LLM summarization.
    """
    txt = doc.text.strip()
    if not txt:
        return None

    if doc.source.startswith("wiktionary_"):
        # Pull 'znaczenia' and 'etymologia' sections cleanly
        head = doc.title.split(":", 1)[-1].strip()
        meaning_m = re.search(r"znaczenia:\s*(.+?)(?:\nodmiana:|\nprzykłady:|\nskładnia:|\n==|$)", txt, flags=re.S | re.I)
        etym_m = re.search(r"etymologia:\s*(.+?)(?:\nuwagi:|\ntłumaczenia:|\nźródła:|\n==|$)", txt, flags=re.S | re.I)
        parts: list[str] = []
        if meaning_m:
            m_clean = re.sub(r"\s+", " ", meaning_m.group(1)).strip()
            parts.append(f"Znaczenie słowa „{head}”: {m_clean[:260]}")
        if etym_m:
            e_clean = re.sub(r"\s+", " ", etym_m.group(1)).strip()
            parts.append(f"Etymologia: {e_clean[:200]}")
        if parts:
            return " ".join(parts)
        return f"{doc.title}: " + re.sub(r"\s+", " ", txt[:320]).strip()

    if doc.source.startswith("wikipedia_"):
        # Take the first clean paragraph (up to 480 chars) ending on a sentence boundary
        first_para = txt.split("\n")[0].strip()
        if len(first_para) < 80 and "\n" in txt:
            first_para = " ".join(txt.split("\n")[:2]).strip()
        first_para = re.sub(r"\s+", " ", first_para)
        if len(first_para) > 480:
            cut = first_para[:480]
            last_dot = cut.rfind(".")
            first_para = cut[: last_dot + 1] if last_dot > 180 else cut + "..."
        return first_para if len(first_para) >= 40 else None

    return None


def extract_contextual_answer(doc: WebDocument, question: str, max_chars: int = 230) -> str | None:
    """
    Extract the most question-relevant 1–2 factual Polish sentences from a Wikipedia/Wiktionary
    document (e.g. winner/final for 'kto wygrał...', causes/origins for 'skąd się biorą/dlaczego...'),
    falling back cleanly to `extract_lead_definition(doc)` when the lead is already best.
    """
    txt = (doc.text or "").strip()
    if not txt:
        return None
    if doc.source.startswith("wiktionary_"):
        return extract_lead_definition(doc)

    q_low = _strip_pl((question or "").lower())
    intent_roots: list[str] = []
    if any(k in q_low for k in ("wygral", "zdobyl", "zwyciez", "mistrz")):
        intent_roots.extend(["wygra", "zwyci", "mistrz", "fina", "pokona", "zdoby", "tytu"])
    if any(k in q_low for k in ("skad", "biora", "bierze", "dlaczego", "czemu", "przyczyn", "powstaj")):
        intent_roots.extend(["przyczyn", "pochodz", "biolog", "genety", "powsta", "wynika", "czynnik", "uwarunk", "wywol", "zjawisk", "hormon", "ewoluc"])
    if any(k in q_low for k in ("kiedy", "ktorym roku")):
        intent_roots.extend(["roku", "wieku", "odby", "rozpocz", "zakoncz", "urodz", "zmarl"])
    if any(k in q_low for k in ("gdzie", "lezy")):
        intent_roots.extend(["lezy", "znajduj", "polozon", "kontynen", "panstw", "region", "granicz"])

    # Only run deeper sentence selection when the question asks a specific non-definitional intent
    if not intent_roots:
        return extract_lead_definition(doc)

    q_words = [
        _strip_pl(w.lower())[:6]
        for w in re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]{4,}", clean_search_query(question))
        if _strip_pl(w.lower()) not in ("jest", "biora", "bierze", "dlaczego", "czemu", "kiedy", "gdzie", "skad")
    ]
    # Split first ~7500 chars into clean sentences (skipping wiki section headings == ... ==)
    clean_body = re.sub(r"==+[^=]+==+", "\n", txt[:7500])
    clean_body = re.sub(r"\[\d+\]", "", clean_body)
    raw_sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", clean_body).strip())

    best_score = 0.0
    best_idx = -1
    for idx, sent in enumerate(raw_sents[:90]):
        s = sent.strip()
        if len(s) < 35 or len(s) > 320 or is_web_boilerplate(s):
            continue
        s_fold = _strip_pl(s.lower())
        hits_intent = sum(1 for r in intent_roots if r in s_fold)
        if hits_intent == 0:
            continue
        hits_kw = sum(1 for kw in q_words if kw in s_fold)
        pos_bonus = 0.4 if idx < 8 else (0.15 if idx < 25 else 0.0)
        score = hits_intent * 1.5 + hits_kw * 0.8 + pos_bonus
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx >= 0 and best_score >= 1.5:
        picked = raw_sents[best_idx].strip()
        if len(picked) < max_chars - 60 and best_idx + 1 < len(raw_sents):
            nxt = raw_sents[best_idx + 1].strip()
            if 25 <= len(nxt) <= (max_chars - len(picked) - 2) and not is_web_boilerplate(nxt):
                picked = f"{picked} {nxt}"
        if len(picked) > max_chars:
            cut = picked[:max_chars]
            last_dot = cut.rfind(".")
            picked = cut[: last_dot + 1] if last_dot > 80 else cut.rstrip(" ,;:") + "."
        return picked

    return extract_lead_definition(doc)


def is_relevant_doc(doc: WebDocument, query: str, fallback_topic: str = "") -> bool:
    """Reject unrelated search hits (e.g. Kaposi's sarcoma for 'node.ks')."""
    if doc.source.startswith("wiktionary_"):
        return True
    combined_q = clean_search_query(query)
    if fallback_topic:
        combined_q = f"{combined_q} {clean_search_query(fallback_topic)}"
    q_words = [
        w.lower()
        for w in re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9+#]{3,}", combined_q)
        if _strip_pl(w.lower()) not in ("jest", "oraz", "jako", "jego", "tego", "temat", "dziala", "definicja")
    ]
    if not q_words:
        return True
    haystack = (doc.title + " " + doc.text[:4500]).lower()
    haystack_stripped = _strip_pl(haystack)

    # If the query is clearly about programming, reject unrelated economics/social indicator pages
    prog_markers = ("c++", "c#", "python", "rust", "java", "javascript", "typescript", "programow")
    q_stripped = _strip_pl(combined_q.lower())
    if any(pm in q_stripped for pm in prog_markers):
        cs_terms = ("program", "pamie", "zmienn", "adres", "kod", "funkcj", "typ", "c++", "pointer", "jezyk", "kompil", "referenc")
        if not any(term in haystack_stripped for term in cs_terms):
            return False

    for w in q_words:
        stem = w[: min(len(w), 6)]
        stem_stripped = _strip_pl(stem)
        if stem in haystack or stem_stripped in haystack_stripped:
            return True
    return False


def gather_web_sources(
    query: str,
    seen_urls: set[str],
    max_docs: int = 3,
    timeout: int = 12,
    max_workers: int = 3,
    encyclopedic_only: bool = False,
    prefer_wikipedia: bool = False,
) -> list[WebDocument]:
    """
    Gather high-signal, relevance-checked Polish & English web documents:
      0) Direct HTTP/HTTPS URL fetch when user passes a link directly
      1) Polish Wiktionary (or Polish Wikipedia first when `prefer_wikipedia=True`)
      2) Polish Wikipedia (with automatic typo-suggestion retry & full multi-article fetch)
      3) English Wikipedia fallback (for technical topics like C++ pointers)
      4) General Web via DuckDuckGo Lite/HTML (fetched concurrently)
    """
    if is_http_url(query):
        url = query.strip()
        if url in seen_urls:
            return []
        doc = fetch_web_page(url, url, timeout=timeout)
        if doc is not None:
            seen_urls.add(doc.url)
            return [doc]
        return []

    collected: list[WebDocument] = []
    clean = clean_search_query(query)

    def add_doc(doc: WebDocument | None) -> bool:
        if doc is None or doc.url in seen_urls:
            return False
        if not is_relevant_doc(doc, clean):
            return False
        seen_urls.add(doc.url)
        collected.append(doc)
        return len(collected) >= max_docs

    if prefer_wikipedia:
        for d in search_wikipedia(clean, lang="pl", limit=max_docs, timeout=timeout):
            if add_doc(d):
                return collected
        if len(clean.split()) <= 2:
            for d in search_wiktionary(clean, lang="pl", timeout=timeout):
                if add_doc(d):
                    return collected
    else:
        # 1) Polish Wiktionary for concise terms (1–2 words)
        if len(clean.split()) <= 2:
            for d in search_wiktionary(clean, lang="pl", timeout=timeout):
                if add_doc(d):
                    return collected

        # 2) Polish Wikipedia (now fetches ALL top articles cleanly with typo retry!)
        for d in search_wikipedia(clean, lang="pl", limit=max_docs, timeout=timeout):
            if add_doc(d):
                return collected

    # 3) English Wikipedia (great for programming/science/C++ topics; skipped in strict Polish live Auto-sieć)
    if not encyclopedic_only and len(collected) < max_docs:
        for d in search_wikipedia(clean, lang="en", limit=2, timeout=timeout):
            if add_doc(d):
                return collected

    # 4) General web search via DuckDuckGo Lite/HTML (ONLY for explicit /ucz study, never live chat Auto-sieć!)
    if not encyclopedic_only and len(collected) < max_docs:
        candidates = [
            (title, url)
            for title, url in search_duckduckgo_urls(clean, limit=max_docs + 2, timeout=timeout)
            if url not in seen_urls
        ]
        if candidates:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(candidates))) as pool:
                fetched_docs = list(pool.map(lambda pair: fetch_web_page(pair[0], pair[1], timeout=timeout), candidates))
            for d in fetched_docs:
                if add_doc(d):
                    return collected

    return collected


def chunk_text(text: str, focus_query: str = "", max_chars: int = 2400, max_chunks: int = 2) -> list[str]:
    """
    Select the most relevant chunks of a web document:
      - Always includes the introduction (first ~2400 chars).
      - If `focus_query` keywords appear deeper in a long article (e.g. 'Kunktator' inside 'II wojna punicka'),
        explicitly extracts a window centered around that keyword match!
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = [text[:max_chars]]
    if max_chunks <= 1:
        return chunks

    # Search deeper in the article for focus keyword occurrences
    keywords = [w.lower() for w in re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9+#]{4,}", clean_search_query(focus_query))]
    low_rest = text[max_chars:].lower()
    hit_pos = -1
    for kw in keywords:
        idx = low_rest.find(kw[: min(len(kw), 7)])
        if idx != -1:
            hit_pos = max_chars + idx
            break

    if hit_pos != -1:
        start = max(max_chars, hit_pos - 500)
        chunks.append(text[start : start + max_chars])
    else:
        chunks.append(text[max_chars : max_chars * 2])

    return chunks[:max_chunks]
