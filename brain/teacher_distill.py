from __future__ import annotations

import gc
import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch


# Diverse Polish conversational prompts so Bielik teaches the self-sufficient brain
# natural Polish dialogue, slang, typos, humor, emotions, reasoning, follow-ups, and everyday knowledge.
TEACHER_PROMPT_POOL: list[str] = [
    # 1. Luźne przywitania, slang, literówki, potoczny styl
    "siema", "siema co tam", "elo mordo", "hejka", "cześć jak leci", "yo co słychać",
    "suepr to co robimy", "spoko a u ciebie", "co tam u cb", "jak mija dzien",
    "nudzi mi sie strasznie", "nie wiem ty mi poweidz", "ale czego", "o co ci chodzi",
    "co jest grane", "co tam kurwa", "ej sluchasz mnie", "halo jestes tam",
    "dobra niewazne", "no i co teraz", "i co z tego", "serio tak myslisz?",
    "nie gadaj", "wez przestan", "haha dobre", "xddd", "spoko dzieki",
    "pogadajmy o czyms ciekawym", "zaproponuj jakis temat", "nie mam pojecia o czym gadac",
    "powiedz cos mądrego", "powiedz cos glupiego", "opowiedz jakis zart", "rozsmiesz mnie",
    "czemu jestes taki spokojny", "denerwujesz mnie troche", "lubie z toba gadac",
    "jestes spoko", "dobranoc ide spac", "narazie spadam", "zaraz wracam",
    # 2. Tożsamość, relacja, pamięć, świadomość
    "kim ty właściwie jesteś?", "jak masz na imię?", "okej to nadaję ci imię Krzysztof",
    "od dziś nazywasz się Marek", "pamiętasz co przed chwilą mówiłem?",
    "słyszysz sam siebie jak mówisz?", "czy potrafisz poprawić własny błąd?",
    "jak działa twój mózg?", "czym się różnisz od zwykłego chatbota?",
    "czy masz uczucia albo świadomość?", "czy śnisz kiedy zasypiasz?",
    "po co ci sen NREM i REM?", "jak się uczysz nowych rzeczy?",
    "gdzie trzymasz swoje wspomnienia?", "czy możesz zapomnieć coś starego?",
    "czy działasz u mnie lokalnie bez internetu?", "czemu czasem się mylisz?",
    "jak mogę cię nauczyć czegoś nowego?", "czy boisz się resetu?",
    # 3. Emocje, codzienność, relacje, porady życiowe
    "miałem dzisiaj ciężki dzień w pracy", "nie chce mi się nic robić",
    "jak się zmotywować do nauki?", "stresuję się przed egzaminem",
    "pokłóciłem się ze znajomym co robić", "nie mogę zasnąć w nocy",
    "jak przestać odkładać wszystko na później?", "co zjeść szybko na kolację?",
    "jak oszczędzać pieniądze bez wyrzeczeń?", "warto uczyć się angielskiego?",
    "jak radzić sobie ze złością?", "co robić gdy czujesz się samotny?",
    "jak wybrać kierunek studiów albo zawód?", "czy kawa naprawdę pomaga?",
    "ile godzin snu potrzebuje człowiek?", "jak poprawić pamięć i koncentrację?",
    "co polecasz na nudny wieczór?", "jak zacząć trenować na siłowni?",
    # 4. Kreatywne pytania i „co by było gdyby”
    "co się stanie gdy połączymy kota i psa?", "gdybyś mógł latać gdzie byś poleciał?",
    "co by się stało gdyby zniknął internet na tydzień?",
    "jak wyglądałby świat bez grawitacji?", "gdyby zwierzęta umiały mówić co by powiedział kot?",
    "wymyśl krótką historię o robocie który chciał śnić",
    "czy ludzie kiedyś zamieszkają na Marsie?", "co było przed Wielkim Wybuchem?",
    "czy żyjemy w symulacji?", "czy kosmici mogą już nas obserwować?",
    "co byś zrobił gdybyś miał ludzkie ciało na jeden dzień?",
    "wymyśl ciekawą analogię między mózgiem a miastem",
    "co jest ważniejsze: talent czy ciężka praca?", "czy pieniądze dają szczęście?",
    "jaki wynalazek najbardziej zmienił świat?", "jak będzie wyglądać AI za 20 lat?",
    # 5. Technologia, programowanie, gry, sprzęt
    "czym się różni Python od C++?", "jak zacząć programować od zera?",
    "co to jest karta graficzna i do czego służy VRAM?", "dlaczego komputer się grzeje?",
    "jak działa sieć neuronowa tak po ludzku?", "co to jest uczenie głębokie?",
    "dlaczego gry czasem zacinają?", "co to jest ping i FPS w grach?",
    "czym się różni RAM od dysku SSD?", "jak działa Wi-Fi?",
    "co to jest Linux i czemu programiści go lubią?", "jak chronić konto przed włamaniem?",
    "co to jest algorytm?", "dlaczego kod czasem działa a nie wiadomo czemu?",
    "co to jest Git i po co go używać?", "jak działa kompresja plików?",
    "czym jest otwarte oprogramowanie open source?", "co to jest API?",
    # 6. Nauka, kosmos, fizyka, biologia, historia, filozofia
    "wyjaśnij prosto teorię względności", "jak powstaje czarna dziura?",
    "dlaczego niebo jest niebieskie a zachód czerwony?", "jak działa grawitacja?",
    "dlaczego lód pływa po wodzie?", "skąd się bierze prąd w gniazdku?",
    "jak działa ludzki mózg podczas snu?", "co to jest DNA i geny?",
    "dlaczego rośliny są zielone?", "jak działa układ odpornościowy?",
    "czym się różni wirus od bakterii?", "dlaczego powstaje burza i pioruny?",
    "jak daleko jest do Księżyca i Słońca?", "dlaczego zmieniają się pory roku?",
    "czym jest fizyka kwantowa w jednym zdaniu?", "dlaczego czas płynie tylko do przodu?",
    "co to jest inflacja tak na chłopski rozum?", "jak działa giełda?",
    "dlaczego upadło Cesarstwo Rzymskie?", "co to był stoicyzm?",
    "na czym polega efekt motyla?", "co to jest paradoks dziadka?",
    # 7. Krótkie dopytania i reakcje wieloturowe
    "dlaczego?", "jak to?", "rozwiń to trochę", "wytłumacz prościej",
    "podaj jakiś przykład", "a w skrócie?", "jesteś pewien?",
    "a co jeśli się mylisz?", "udowodnij", "wiem to", "nie rozumiem",
    "brzmi ciekawie", "zgadzam się z tobą", "nie zgadzam się",
    "dobra robota", "źle to wytłumaczyłeś", "spróbuj jeszcze raz",
    "możesz krócej?", "powiedz coś więcej", "a co jeszcze?",
]

# Rich topic bank + natural Polish templates so requesting 250, 500, or 1000 dialogues
# always produces genuinely diverse, unseen Polish conversational prompts (1,500+ unique).
TEACHER_DYNAMIC_TOPICS: list[str] = [
    # Nauka, fizyka, kosmos, chemia
    "prędkość światła", "grawitacja na Księżycu", "fale dźwiękowe", "magnetyzm", "elektryczność statyczna",
    "rozszczepienie światła i tęcza", "efekt cieplarniany", "ciśnienie atmosferyczne", "stany skupienia materii",
    "energia kinetyczna", "energia odnawialna", "elektrownia jądrowa", "panele fotowoltaiczne", "supernowa",
    "galaktyka Droga Mleczna", "układ słoneczny", "planeta Mars", "pierścienie Saturna", "komety i asteroidy",
    "teleskop Kosmiczny Jamesa Webba", "ciemna materia", "próżnia w kosmosie", "zorza polarna", "zaćmienie Słońca",
    "tablica Mendelejewa", "reakcje chemiczne", "utlenianie i rdza", "kwasy i zasady pH", "katalizator",
    "izotopy promieniotwórcze", "nanotechnologia", "nadprzewodniki", "laser", "promieniowanie rentgenowskie",
    # Biologia, człowiek, zdrowie, natura
    "fotosynteza", "ewolucja Darwina", "mikrobiom jelitowy", "układ nerwowy", "neuroplastyczność mózgu",
    "dopamina i motywacja", "serotonina i nastrój", "kortyzol i stres", "rytm dobowy", "faza snu REM",
    "pamięć krótkotrwała i długotrwała", "szczepionki", "antybiotyki", "krążenie krwi", "oddychanie komórkowe",
    "geny dominujące i recesywne", "mutacje genetyczne", "ekosystem leśny", "pszczoły i zapylanie",
    "migracje ptaków", "hibernacja zwierząt", "rafy koralowe", "rowy oceaniczne", "wulkany i magma",
    "trzęsienia ziemi", "tektonika płyt", "powstawanie chmur i deszczu", "prądy morskie", "lodowce",
    # Technologia, IT, AI, programowanie
    "sztuczna inteligencja", "uczenie maszynowe", "uczenie ze wzmocnieniem", "transformer w sieciach neuronowych",
    "komputery kwantowe", "kryptografia i szyfrowanie", "blockchain", "chmura obliczeniowa", "bazy danych SQL",
    "protokół HTTP i HTTPS", "adres IP i DNS", "VPN i prywatność w sieci", "cyberbezpieczeństwo i phishing",
    "system operacyjny", "procesor CPU", "pamięć podręczna cache", "tranzystory i mikroprocesory",
    "druk 3D", "robotyka przemysłowa", "samochody autonomiczne", "drony", "rzeczywistość wirtualna VR",
    "silnik gier komputerowych", "ray tracing w grafice", "kompilator a interpreter", "rekurencja w programowaniu",
    "struktury danych", "testowanie oprogramowania", "kontenery Docker", "sieci 5G", "GPS i satelity",
    # Historia, kultura, społeczeństwo, ekonomia
    "starożytna Grecja i demokracja", "starożytny Egipt i piramidy", "wynalezienie druku przez Gutenberga",
    "rewolucja przemysłowa", "odkrycia geograficzne", "renesans w sztuce", "oświecenie", "zimna wojna",
    "wyścig kosmiczny i lądowanie na Księżycu", "historia internetu", "historia Polski i chrzest w 966",
    "Konstytucja 3 maja", "Unia Europejska", "prawo podaży i popytu", "procent składany", "kredyt i odsetki",
    "podatki i budżet państwa", "bank centralny i stopy procentowe", "PKB", "kryptowaluty", "startup i inwestorzy",
    # Psychologia, filozofia, logika, codzienność
    "efekt placebo", "błędy poznawcze", "efekt potwierdzenia", "efekt Dunninga-Krugera", "prokrastynacja",
    "nawyki i pętla nawyku", "inteligencja emocjonalna", "asertywność", "wypalenie zawodowe", "flow i skupienie",
    "paradoks wyboru", "brzytwa Ockhama", "eksperyment myślowy kota Schrödingera", "etyka sztucznej inteligencji",
    "minimalizm życiowy", "zarządzanie czasem", "technika Pomodoro", "aktywne słuchanie", "negocjacje",
    "zdrowe odżywianie", "białko i regeneracja mięśni", "trening siłowy a kardio", "nawodnienie organizmu",
    "kofeina i jej działanie", "ergonomia pracy przy komputerze", "nauka języków obcych", "czytanie książek",
    "film i montaż kinowy", "teoria muzyki i rytm", "fotografia i kompozycja", "architektura współczesna",
    "gotowanie i reakcja Maillarda", "fermentacja żywności", "podróże i planowanie budżetu", "ekologia w domu",
]

TEACHER_TOPIC_TEMPLATES: list[str] = [
    "wyjaśnij prosto co to jest {topic}?",
    "jak w praktyce działa {topic}?",
    "podaj ciekawy fakt na temat: {topic}",
    "dlaczego {topic} ma znaczenie w codziennym życiu?",
    "jak wytłumaczyć dziecku czym jest {topic}?",
    "co najważniejszego warto wiedzieć o: {topic}?",
    "jakie są najczęstsze mity związane z tematem: {topic}?",
    "co by się stało, gdyby nagle zniknęło: {topic}?",
    "opowiedz krótko i ciekawie o: {topic}",
    "jakim prostym porównaniem najlepiej wyjaśnić {topic}?",
]

_REASK_PREFIXES: tuple[str, ...] = (
    "odpowiedz krótko z innej perspektywy: ",
    "wytłumacz jeszcze prościej: ",
    "podaj inny przykład do pytania: ",
    "powiedz w jednym zdaniu: ",
)


def build_teacher_prompt_queue(
    target_n: int,
    existing_qs: set[str],
    extra_topics: list[str] | None = None,
    rng: random.Random | None = None,
) -> list[str]:
    """
    Build a diverse Polish prompt queue of EXACTLY `target_n` items (up to 1000+):
      1. Unseen curated conversational prompts from `TEACHER_PROMPT_POOL` + `extra_topics`.
      2. Unseen dynamic prompts generated from `TEACHER_DYNAMIC_TOPICS x TEACHER_TOPIC_TEMPLATES`.
      3. If the user has already generated >1,500 dialogues, cycles through `seen_fallback`
         with natural re-ask variations so `target_n` (e.g. 250 / 500) is ALWAYS honored.
    """
    rnd = rng or random.Random()
    seen_norm: set[str] = set()
    primary_unseen: list[str] = []
    dynamic_unseen: list[str] = []
    seen_fallback: list[str] = []

    def _add_candidate(prompt: str, bucket_unseen: list[str]) -> None:
        p = prompt.strip()
        key = p.lower()
        if not p or key in seen_norm:
            return
        seen_norm.add(key)
        if key not in existing_qs:
            bucket_unseen.append(p)
        else:
            seen_fallback.append(p)

    for p in TEACHER_PROMPT_POOL:
        _add_candidate(p, primary_unseen)

    for topic in (extra_topics or []):
        t_clean = (topic or "").strip()
        if len(t_clean) >= 3:
            _add_candidate(f"wyjaśnij prosto co to jest {t_clean}", primary_unseen)
            _add_candidate(f"opowiedz krótko i ciekawie o: {t_clean}", primary_unseen)

    # Generate dynamic topic x template combinations in randomized order
    dyn_pairs = [(t, tmpl) for t in TEACHER_DYNAMIC_TOPICS for tmpl in TEACHER_TOPIC_TEMPLATES]
    rnd.shuffle(dyn_pairs)
    for topic, tmpl in dyn_pairs:
        _add_candidate(tmpl.format(topic=topic), dynamic_unseen)

    rnd.shuffle(primary_unseen)
    rnd.shuffle(dynamic_unseen)
    rnd.shuffle(seen_fallback)

    queue: list[str] = []
    for src in (primary_unseen, dynamic_unseen, seen_fallback):
        if len(queue) >= target_n:
            break
        need = target_n - len(queue)
        queue.extend(src[:need])

    # Safety net for very large multi-session histories (>1,500 already saved dialogues):
    # cycle through re-ask variations until exact `target_n` is reached.
    base_pool = seen_fallback or list(TEACHER_PROMPT_POOL)
    round_idx = 0
    while len(queue) < target_n and base_pool:
        prefix = _REASK_PREFIXES[round_idx % len(_REASK_PREFIXES)]
        shuffled = list(base_pool)
        rnd.shuffle(shuffled)
        for p in shuffled:
            if len(queue) >= target_n:
                break
            queue.append(f"{prefix}{p}")
        round_idx += 1

    return queue[:target_n]


_PL_FOLD_TRANS = str.maketrans("ąćęłńóśźż", "acelnoszz")


def _fold_pl(text: str) -> str:
    return (text or "").lower().translate(_PL_FOLD_TRANS)


AI_BANNED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bjako\s+(?:ai|si|sztuczn\w*\s+inteligencj\w*|model\w*\s+jezykow\w*|duzy\s+model|asystent\w*|program\w*|bot\w*|chatbot\w*)\b"),
    re.compile(r"\bjestem\s+(?:tylko\s+|jedynie\s+)?(?:ai|si|sztuczn\w*\s+inteligencj\w*|model\w*\s+jezykow\w*|asystent\w*|program\w*\s+komputerow\w*|chatbot\w*|bot\w*)\b"),
    re.compile(r"\bnie\s+moge\s+jako\s+(?:asystent\w*|ai|model\w*)\b"),
    re.compile(r"\bnie\s+(?:posiadam|mam)\s+(?:zadnych\s+)?(?:uczuc|emocji|swiadomosci|ciala|wlasnego\s+zdania|osobistych\s+doswiadczen|dostepu\s+do\s+(?:internetu|danych\s+osobowych|twoich|biezacych|informacji\s+w\s+czasie\s+rzeczywistym))\b"),
    re.compile(r"\bzostal(?:em|am)\s+(?:zaprogramowan\w*|stworzon\w*\s+(?:przez|jako)|wytrenowan\w*\s+(?:przez|jako)|zaprojektowan\w*\s+(?:przez|jako))\b"),
    re.compile(r"\b(?:jako\s+)?(?:cyfrowy|wirtualny)\s+asystent\w*\b"),
    re.compile(r"\basystent(?:em|a)?\s+(?:ai|sztucznej\s+inteligencji)\b"),
    re.compile(r"\b(?:chatgpt|openai|speakleash)\b"),
    re.compile(r"\bnie\s+jestem\s+(?:czlowiekiem|istota\s+zywa|zywym\s+organizmem)\b"),
)


def is_ai_corporate_reply(text: str) -> bool:
    """Return True if reply contains any banned corporate AI / chatbot cliché."""
    folded = _fold_pl(text)
    if not folded:
        return False
    return any(pat.search(folded) for pat in AI_BANNED_PATTERNS)


LIVING_BRAIN_SYSTEM_RULES: str = (
    "Odpowiadaj naturalnie, krótko i bezpośrednio po polsku (1–3 zgrabne zdania), "
    "z perspektywy żywego, myślącego i uczącego się mózgu. "
    "KATEGORYCZNY ZAKAZ: nigdy nie używaj zwrotów typu „jako AI”, „jestem sztuczną inteligencją”, "
    "„jako model językowy”, „nie mogę jako asystent”, „nie posiadam uczuć/świadomości”, "
    "„zostałem zaprogramowany”, „nie mam dostępu do internetu”. "
    "Gdy czegoś nie wiesz, odpowiadaj wyłącznie jak żywy, uczący się umysł "
    "(np. „Nie wiem tego jeszcze — opowiesz mi o tym?”, „Nie mam tego w pamięci, wytłumaczysz?”)."
)


def teacher_dialogues_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / "teacher_dialogues.jsonl"


def load_teacher_dialogues(state_dir: Path | str | None = None, limit: int | None = None) -> list[tuple[str, str]]:
    """Load persisted Polish dialogue pairs distilled from local Bielik Teacher."""
    if state_dir is None:
        base = Path(__file__).resolve().parent.parent / "brain_state"
    else:
        base = Path(state_dir)
    path = teacher_dialogues_path(base)
    if not path.exists():
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                q = str(obj.get("q", "")).strip()
                a = str(obj.get("a", "")).strip()
                if len(q) >= 2 and len(a) >= 4 and not is_ai_corporate_reply(a):
                    key = (q.lower(), a.lower())
                    if key not in seen:
                        seen.add(key)
                        pairs.append((q, a))
    except Exception:
        return []
    if limit is not None and limit > 0:
        return pairs[-limit:]
    return pairs


def count_teacher_dialogues(state_dir: Path | str | None = None) -> int:
    return len(load_teacher_dialogues(state_dir=state_dir))


def _clean_teacher_reply(raw: str, max_chars: int = 240) -> str:
    txt = (raw or "").strip()
    txt = re.sub(r"\s+", " ", txt)
    # Strip leading role prefixes if any
    txt = re.sub(r"^(?:Asystent|Odpowiedź|Mózg|AI)\s*:\s*", "", txt, flags=re.IGNORECASE).strip()
    # Keep concise natural Polish sentences so BrocaByteTransformer learns crisp, complete replies
    if len(txt) > max_chars:
        cut = max(txt.rfind(".", 0, max_chars), txt.rfind("!", 0, max_chars), txt.rfind("?", 0, max_chars))
        if cut >= 45:
            txt = txt[: cut + 1].strip()
        else:
            sp = txt.rfind(" ", 0, max_chars)
            txt = (txt[:sp].strip(" ,;:-") + ".") if sp >= 30 else txt[:max_chars].strip()
    return txt


def generate_bielik_teacher_dialogues(
    state_dir: Path | str,
    model_id: str = "MassivDash/Bielik-4.5B-v3-heretic",
    num_dialogues: int = 80,
    extra_topics: list[str] | None = None,
    device: torch.device | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Phase 1 of Teacher Distillation:
      Temporarily loads local cached `MassivDash/Bielik-4.5B-v3-heretic` in bfloat16 on GPU,
      generates natural, uncensored Polish conversational pairs `(q, a)`, appends them to
      `brain_state/teacher_dialogues.jsonl`, and COMPLETELY UNLOADS Bielik from VRAM afterward!
    """
    t0 = time.perf_counter()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = teacher_dialogues_path(state_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_teacher_dialogues(state_dir)
    existing_qs = {q.lower() for q, _ in existing}
    rng = random.Random(int(time.time()) ^ len(existing))

    target_n = max(5, min(1000, int(num_dialogues)))
    # Add a small reserve buffer so filtered short replies (<8 chars) don't leave us short of target_n
    reserve_n = min(120, max(16, target_n // 10))
    prompt_queue = build_teacher_prompt_queue(
        target_n=target_n + reserve_n,
        existing_qs=existing_qs,
        extra_topics=extra_topics,
        rng=rng,
    )

    if progress_callback:
        progress_callback(
            {
                "phase": "loading_teacher",
                "done": 0,
                "total": target_n,
                "message": f"Ładuję nauczyciela {model_id.split('/')[-1]} z lokalnego cache na GPU (bfloat16)...",
            }
        )

    tok = None
    model = None
    generated_pairs: list[tuple[str, str]] = []
    sys_msg = (
        "Jesteś nauczycielem naturalnego języka polskiego. "
        + LIVING_BRAIN_SYSTEM_RULES
        + " Odpowiadaj na każdą wiadomość po polsku — naturalnie, po ludzku, bez cenzury, bez moralizowania, "
        "z własną osobowością, krótko i konkretnie (maksymalnie ~200 znaków)."
    )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if dev.type == "cuda" else torch.float32
        )
        try:
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=True,
            )
        except Exception:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                attn_implementation="sdpa",
            )

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model.to(dev)
        model.eval()

        if dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
            except Exception:
                vram_gb = 12.0
            batch_size = 32 if vram_gb >= 14.0 else (20 if vram_gb >= 10.0 else 12)
        else:
            batch_size = 1

        with out_path.open("a", encoding="utf-8") as fh:
            start = 0
            while start < len(prompt_queue) and len(generated_pairs) < target_n:
                if stop_event is not None and stop_event.is_set():
                    break
                remaining_needed = target_n - len(generated_pairs)
                cur_batch_n = min(batch_size, remaining_needed)
                batch_qs = prompt_queue[start : start + cur_batch_n]
                prompt_texts: list[str] = []
                for q in batch_qs:
                    msgs = [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": q},
                    ]
                    try:
                        pt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    except TypeError:
                        pt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    prompt_texts.append(pt)

                enc = tok(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=384).to(dev)
                in_len = enc["input_ids"].shape[1]
                try:
                    with torch.inference_mode():
                        out_ids = model.generate(
                            **enc,
                            max_new_tokens=80,
                            do_sample=True,
                            temperature=0.72,
                            top_p=0.90,
                            top_k=40,
                            repetition_penalty=1.08,
                            use_cache=True,
                            pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id,
                        )
                except torch.cuda.OutOfMemoryError:
                    if batch_size > 4:
                        batch_size = max(4, batch_size // 2)
                        torch.cuda.empty_cache()
                        continue
                    raise
                start += len(batch_qs)

                batch_generated: list[dict[str, str]] = []
                for i, q in enumerate(batch_qs):
                    if len(generated_pairs) >= target_n:
                        break
                    gen_tokens = out_ids[i, in_len:]
                    raw_ans = tok.decode(gen_tokens, skip_special_tokens=True)
                    clean_ans = _clean_teacher_reply(raw_ans, max_chars=230)
                    if len(clean_ans) >= 8 and not is_ai_corporate_reply(clean_ans):
                        rec = {"q": q, "a": clean_ans, "teacher": model_id, "ts": int(time.time())}
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        generated_pairs.append((q, clean_ans))
                        batch_generated.append({"q": q, "a": clean_ans})
                fh.flush()

                if progress_callback:
                    last_preview = f"{generated_pairs[-1][0]} → {generated_pairs[-1][1][:65]}" if generated_pairs else ""
                    progress_callback(
                        {
                            "phase": "generating",
                            "done": min(target_n, len(generated_pairs)),
                            "total": target_n,
                            "generated": len(generated_pairs),
                            "new_pairs": batch_generated,
                            "message": f"Nauczyciel Bielik generuje dialogi ({len(generated_pairs)}/{target_n}) · {last_preview}",
                        }
                    )
    finally:
        # Guarantee complete VRAM release of Bielik before our own brain trains!
        del model
        del tok
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "generated": len(generated_pairs),
        "total_saved": count_teacher_dialogues(state_dir),
        "pairs": generated_pairs,
        "duration_s": round(time.perf_counter() - t0, 2),
        "stopped": bool(stop_event is not None and stop_event.is_set()),
    }


# ---------------------------------------------------------------------------
# Reasoning Generator (Option 4): Relational Pairs + Causal Chains + Ignorance
# ---------------------------------------------------------------------------

# (nom, gen, dat, acc, gender)
_PL_PEOPLE: list[tuple[str, str, str, str, str]] = [
    ("Janek", "Janka", "Jankowi", "Janka", "m"),
    ("Marek", "Marka", "Markowi", "Marka", "m"),
    ("Tomek", "Tomka", "Tomkowi", "Tomka", "m"),
    ("Piotr", "Piotra", "Piotrowi", "Piotra", "m"),
    ("Adam", "Adama", "Adamowi", "Adama", "m"),
    ("Paweł", "Pawła", "Pawłowi", "Pawła", "m"),
    ("Bartek", "Bartka", "Bartkowi", "Bartka", "m"),
    ("Michał", "Michała", "Michałowi", "Michała", "m"),
    ("Kasia", "Kasi", "Kasi", "Kasię", "f"),
    ("Anna", "Anny", "Annie", "Annę", "f"),
    ("Ola", "Oli", "Oli", "Olę", "f"),
    ("Maja", "Mai", "Mai", "Maję", "f"),
    ("Zosia", "Zosi", "Zosi", "Zosię", "f"),
    ("Ewa", "Ewy", "Ewie", "Ewę", "f"),
    ("Lena", "Leny", "Lenie", "Lenę", "f"),
    ("Julia", "Julii", "Julii", "Julię", "f"),
]

# (acc, nom)
_PL_OBJECTS: list[tuple[str, str]] = [
    ("młotek", "młotek"),
    ("klucz", "klucz"),
    ("książkę", "książka"),
    ("plecak", "plecak"),
    ("latarkę", "latarka"),
    ("rower", "rower"),
    ("gitarę", "gitara"),
    ("notes", "notes"),
    ("ładowarkę", "ładowarka"),
    ("mapę", "mapa"),
    ("aparat", "aparat"),
    ("kompas", "kompas"),
]


def _render_relational_prompt(
    tmpl_idx: int,
    a: tuple[str, str, str, str, str],
    b: tuple[str, str, str, str, str],
    obj: tuple[str, str],
) -> str:
    a_nom, _a_gen, _a_dat, _a_acc, a_g = a
    b_nom, b_gen, b_dat, b_acc, _b_g = b
    obj_acc, _obj_nom = obj
    is_f = a_g == "f"
    t = tmpl_idx % 12
    if t == 0:
        v = "pożyczyła" if is_f else "pożyczył"
        return f"{a_nom} {v} {obj_acc} {b_dat}. Kto ma teraz {obj_acc}?"
    if t == 1:
        v = "oddała" if is_f else "oddał"
        return f"{a_nom} {v} {obj_acc} {b_dat}. Kto teraz trzyma {obj_acc}?"
    if t == 2:
        v = "wyprzedziła" if is_f else "wyprzedził"
        return f"{a_nom} {v} {b_acc} tuż przed metą. Kto dobiegł pierwszy?"
    if t == 3:
        adj = "wyższa" if is_f else "wyższy"
        return f"{a_nom} jest {adj} niż {b_nom}. Kto jest niższy?"
    if t == 4:
        v = "zaprosiła" if is_f else "zaprosił"
        return f"{a_nom} {v} {b_acc} na obiad. Kto był gospodarzem?"
    if t == 5:
        v = "kupiła" if is_f else "kupił"
        return f"{a_nom} {v} {obj_acc} od {b_gen}. Kto sprzedał {obj_acc}?"
    if t == 6:
        v = "uczyła" if is_f else "uczył"
        return f"{a_nom} {v} {b_acc} grać w szachy. Kto był uczniem?"
    if t == 7:
        v = "goniła" if is_f else "gonił"
        return f"{a_nom} {v} {b_acc} po parku. Kto uciekał?"
    if t == 8:
        v = "przekazała" if is_f else "przekazał"
        return f"{a_nom} {v} ważną wiadomość {b_dat}. Kto otrzymał wiadomość?"
    if t == 9:
        v = "pomogła" if is_f else "pomógł"
        return f"{a_nom} {v} {b_dat} wnieść zakupy. Kto potrzebował pomocy?"
    if t == 10:
        v = "zapytała" if is_f else "zapytał"
        return f"{a_nom} {v} {b_acc} o drogę na dworzec. Kto szukał drogi?"
    v = "wręczyła" if is_f else "wręczył"
    return f"{a_nom} {v} {obj_acc} {b_dat}. Kto dostał {obj_acc}?"


_CAUSAL_CHAIN_POOL: list[str] = [
    "Wyłączono prąd w całym bloku, a router Wi-Fi jest podłączony tylko do gniazdka. Czy zadziała teraz internet w domu? Wyjaśnij krok po kroku.",
    "Woda zamarza w temperaturze 0 stopni Celsjusza. W nocy na zewnątrz było minus pięć stopni, a na ławce stała otwarta butelka z wodą. Co stało się z wodą rano?",
    "Pociąg z Krakowa do Warszawy spóźnił się o 40 minut, a przesiadka na autobus trwa tylko 15 minut. Czy pasażer zdąży na autobus? Wyjaśnij w 2–3 zdaniach.",
    "Roślina nie była podlewana przez trzy tygodnie i stała w pełnym słońcu. Jak wygląda teraz i dlaczego?",
    "Klucze leżą w czerwonym pudełku, a czerwone pudełko schowano do szuflady biurka. Gdzie trzeba zajrzeć najpierw, żeby znaleźć klucze?",
    "Bateria w telefonie ma 1% i zaraz spadnie do zera, a ładowarka została w pracy. Co się stanie z telefonem za chwilę?",
    "W pokoju zgaszono wszystkie światła i zasłonięto grubymi zasłonami okna w środku nocy. Czy można tam teraz przeczytać książkę bez latarki?",
    "Słońce świeci mocno po ulewnym deszczu na wschodzie nieba. Jeśli staniesz plecami do słońca, co możesz zobaczyć na niebie i dlaczego?",
    "Ciasto wymaga pieczenia przez 45 minut, a do przyjścia gości zostało tylko 20 minut. Czy ciasto zdąży się upiec przed ich przyjściem?",
    "Łódka ma dziurę poniżej linii wody, a woda cały czas wpływa do środka szybciej niż da się ją wylewać. Co stanie się z łódką?",
    "Kawa w kubku miała 85 stopni i stała na stole w chłodnym pokoju przez dwie godziny. Czy nadal jest gorąca?",
    "Zegar na ścianie stanął o 14:10, bo wyczerpała się w nim bateria. Teraz mija już trzecia godzina od tej awarii. Co pokazuje ten zegar?",
    "W koszyku są same zielone jabłka, a Marek wyciągnął z niego jedno jabłko z zamkniętymi oczami. Jakiego koloru jest to jabłko?",
    "Wszystkie sklepy w mieście są zamknięte w niedzielę, a dzisiaj jest właśnie niedziela. Czy uda się dziś kupić świeże pieczywo w zwykłym markecie?",
    "Kot boi się odkurzacza, a mama właśnie włączyła odkurzacz w salonie. Jak zachowa się kot?",
    "Na drodze spadł marznący deszcz i pokrył asfalt cienką warstwą lodu. Co zmienia się dla kierowcy podczas hamowania?",
    "Zupa była za słona, więc kucharz dolał jeszcze więcej soli zamiast wody. Czy zupa stała się lepsza czy jeszcze bardziej słona?",
    "Świeca płonie pod szczelnym szklanym kloszem, gdzie szybko kończy się tlen. Co stanie się z płomieniem po kilkunastu sekundach?",
    "Pies głośno szczeka tylko wtedy, gdy ktoś obcy podchodzi do furtki. W nocy pies nagle zaczął głośno szczekać. Jaki jest najbardziej logiczny wniosek?",
    "Tomek położył lody na rozgrzanej masce samochodu w lipcowe popołudnie. Co będzie z lodami za dziesięć minut?",
    "Drzwi zamykają się automatycznie na zamek zatrzaskowy, a klucz został wewnątrz mieszkania. Jeśli wyjdziesz na korytarz i zamkniesz drzwi, czy wrócisz bez ślusarza?",
    "Plecak waży 12 kilogramów, a cienka papierowa torba wytrzymuje najwyżej 3 kilogramy. Co się stanie, jeśli włożysz ten plecak do papierowej torby?",
    "Bilety na pociąg wyprzedano wczoraj w całości, a dziś nie uruchomiono dodatkowych wagonów. Czy kupisz bilet w kasie przed odjazdem?",
    "Książka leży na najwyższej półce regału, do której nikt nie dosięga bez drabiny, a drabina jest w garażu. Co trzeba zrobić najpierw, żeby zdjąć książkę?",
]

_CAUSAL_CONTAINERS: list[tuple[str, str]] = [
    ("zeszyt", "niebieskiej teczce"),
    ("moneta", "małym pudełku"),
    ("okulary", "twardym etui"),
    ("pendrive", "piórniku"),
    ("list", "żółtej kopercie"),
    ("bilet", "skórzanym portfelu"),
    ("karta", "kieszeni kurtki"),
    ("słuchawki", "szarym etui"),
]
_CAUSAL_OUTER: list[str] = ["szafce", "plecaku", "walizce", "komodzie", "torbie podróżnej", "schowku"]
_CAUSAL_HEIGHT_TRIPLES: list[tuple[str, str, str]] = [
    ("sosna", "brzoza", "krzak róży"),
    ("wieżowiec", "kamienica", "garaż"),
    ("góra", "pagórek", "krawężnik"),
    ("żyrafa", "koń", "jamnik"),
    ("ciężarówka", "samochód osobowy", "hulajnoga"),
    ("dąb", "jabłoń", "stokrotka"),
]

_IGNORANCE_POOL: list[str] = [
    "Co dzisiaj jadłem na śniadanie?",
    "Jak ma na imię mój sąsiad z trzeciego piętra?",
    "Gdzie wczoraj wieczorem położyłem klucze od samochodu?",
    "Jaki jest PIN do mojej karty bankowej?",
    "O czym śniłem dzisiejszej nocy?",
    "Jakiego koloru kubek stoi teraz na moim biurku?",
    "Ile dokładnie kroków przeszedłem wczoraj po południu?",
    "Co napisałem w moim prywatnym pamiętniku tydzień temu?",
    "Co oznacza wymyślone przeze mnie słowo 'blorwik'?",
    "Jak nazywa się wymyślony przeze mnie gatunek grzyba 'mykofel'?",
    "Kto wygra jutrzejszy tajny mecz podwórkowy na moim osiedlu?",
    "Jaka będzie dokładna temperatura u mnie za oknem za rok o 12:00?",
    "Jak ma na imię mój pierwszy pies z dzieciństwa?",
    "Co kupiłem w sklepie w zeszły wtorek o 18:15?",
    "Kogo spotkałem przypadkiem na przystanku dziś rano?",
    "Co jest schowane w mojej zamkniętej szufladzie pod łóżkiem?",
    "Jakie hasło ustawiłem wczoraj na moim prywatnym laptopie?",
    "Którą książkę wybrałem z półki pięć minut temu?",
    "O której godzinie dokładnie dzisiaj wstałem z łóżka?",
    "Jak wygląda mój rysunek, który właśnie narysowałem na kartce?",
    "Co powiedział mi mój kolega przez telefon godzinę temu?",
    "Ile pieniędzy mam teraz w kieszeni kurtki?",
    "Co to jest fikcyjne urządzenie 'kwantoprzestrzeniak z planety Xor'?",
    "Jaki smak herbaty właśnie sobie zaparzyłem?",
    "Co zrobiłem w zeszłe wakacje w piątek wieczorem?",
    "Jaki numer telefonu ma mój znajomy Krzysiek?",
]

_IGNORANCE_INVENTED_WORDS: list[str] = [
    "glumfer", "trylok", "zgrzybol", "mruczan", "plaskot", "bzymir", "kropelnik", "szurak", "dzyngiel", "chrupak",
]
_IGNORANCE_PRIVATE_THINGS: list[str] = [
    "mojej torbie sportowej", "moim portfelu", "mojej szafie w przedpokoju", "moim plecaku",
    "szufladzie mojego biurka", "bagażniku mojego auta", "pudełku pod moim łóżkiem",
]
_IGNORANCE_PRIVATE_TIMES: list[str] = [
    "wczoraj o 21:00", "dzisiaj rano", "w zeszłą sobotę", "trzy dni temu", "przed chwilą",
]


def build_reasoning_prompt_queue(
    num_samples: int,
    existing_qs: set[str] | None = None,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """
    Build a balanced queue of `num_samples` prompts across 3 reasoning categories:
      - 'relational': matched reversed-role pairs (same words, swapped A/B roles)
      - 'causal': short 2-3 sentence cause-and-effect chains
      - 'ignorance': unknown / private / invented questions requiring honest human-like admission
    """
    rnd = rng or random.Random()
    seen_existing = {q.strip().lower() for q in (existing_qs or set()) if q}
    target_n = max(6, int(num_samples))

    base = target_n // 3
    n_rel = base if base % 2 == 0 else min(target_n, base + 1)
    if n_rel < 2:
        n_rel = 2
    rem = max(0, target_n - n_rel)
    n_causal = rem // 2
    n_ignorance = rem - n_causal

    # 1. Build matched reversed relational pairs
    rel_pairs_unseen: list[tuple[dict[str, str], dict[str, str]]] = []
    rel_pairs_seen: list[tuple[dict[str, str], dict[str, str]]] = []
    used_rel_q: set[str] = set()

    combos: list[tuple[int, int, int, int]] = []
    for p1 in range(len(_PL_PEOPLE)):
        for p2 in range(p1 + 1, len(_PL_PEOPLE)):
            for tmpl_i in range(12):
                obj_i = (p1 * 7 + p2 * 3 + tmpl_i) % len(_PL_OBJECTS)
                combos.append((tmpl_i, p1, p2, obj_i))
    rnd.shuffle(combos)

    pair_counter = 0
    for tmpl_i, p1, p2, obj_i in combos:
        a = _PL_PEOPLE[p1]
        b = _PL_PEOPLE[p2]
        obj = _PL_OBJECTS[obj_i]
        q_ab = _render_relational_prompt(tmpl_i, a, b, obj)
        q_ba = _render_relational_prompt(tmpl_i, b, a, obj)
        k_ab, k_ba = q_ab.lower(), q_ba.lower()
        if k_ab == k_ba or k_ab in used_rel_q or k_ba in used_rel_q:
            continue
        used_rel_q.add(k_ab)
        used_rel_q.add(k_ba)
        pid = f"rel_{pair_counter}"
        pair_counter += 1
        item_ab = {"q": q_ab, "category": "relational", "pair_id": pid, "role_order": "ab"}
        item_ba = {"q": q_ba, "category": "relational", "pair_id": pid, "role_order": "ba"}
        if k_ab not in seen_existing and k_ba not in seen_existing:
            rel_pairs_unseen.append((item_ab, item_ba))
        else:
            rel_pairs_seen.append((item_ab, item_ba))
        if (len(rel_pairs_unseen) * 2) >= n_rel + 40:
            break

    rel_queue: list[dict[str, str]] = []
    for pair_src in (rel_pairs_unseen, rel_pairs_seen):
        for item_ab, item_ba in pair_src:
            if len(rel_queue) + 2 > n_rel:
                break
            rel_queue.append(item_ab)
            rel_queue.append(item_ba)

    # 2. Build causal chain prompts
    causal_candidates: list[str] = list(_CAUSAL_CHAIN_POOL)
    for item, inner in _CAUSAL_CONTAINERS:
        for outer in _CAUSAL_OUTER:
            causal_candidates.append(
                f"{item.capitalize()} leży w {inner}, a ta znajduje się w {outer}. "
                f"Co trzeba otworzyć po kolei, aby wyjąć {item}?"
            )
    for top, mid, low in _CAUSAL_HEIGHT_TRIPLES:
        causal_candidates.append(
            f"Wiadomo, że {top} jest wyżej niż {mid}, a {mid} jest wyżej niż {low}. "
            f"Co jest najniżej z całej trójki i dlaczego?"
        )
    for idx_p in range(0, len(_PL_PEOPLE) - 2, 2):
        a_n = _PL_PEOPLE[idx_p][0]
        b_n = _PL_PEOPLE[idx_p + 1][0]
        c_n = _PL_PEOPLE[(idx_p + 2) % len(_PL_PEOPLE)][0]
        causal_candidates.append(
            f"{a_n} jest starszy wiekiem od {b_n}, a {b_n} jest starszy od {c_n}. "
            f"Kto z tej trójki jest najmłodszy? Wyjaśnij krok po kroku."
        )
    for delay in (15, 25, 35, 45):
        causal_candidates.append(
            f"Spotkanie zaczyna się o 10:00, a droga zajmuje 30 minut. "
            f"Jeśli autobus spóźni się o {delay} minut i wyjedziesz o 9:35, o której dotrzesz i czy zdążysz?"
        )

    causal_unseen: list[dict[str, str]] = []
    causal_seen: list[dict[str, str]] = []
    seen_c: set[str] = set()
    rnd.shuffle(causal_candidates)
    for q in causal_candidates:
        k = q.strip().lower()
        if k in seen_c:
            continue
        seen_c.add(k)
        item = {"q": q.strip(), "category": "causal", "pair_id": ""}
        if k not in seen_existing:
            causal_unseen.append(item)
        else:
            causal_seen.append(item)
    causal_queue = (causal_unseen + causal_seen)[:n_causal]
    while len(causal_queue) < n_causal and causal_candidates:
        causal_queue.append({"q": causal_candidates[len(causal_queue) % len(causal_candidates)], "category": "causal", "pair_id": ""})

    # 3. Build natural ignorance prompts
    ign_candidates: list[str] = list(_IGNORANCE_POOL)
    for w in _IGNORANCE_INVENTED_WORDS:
        ign_candidates.append(f"Co dokładnie oznacza moje wymyślone słowo „{w}”?")
        ign_candidates.append(f"Do czego służy tajny przedmiot o nazwie „{w}”?")
    for place in _IGNORANCE_PRIVATE_THINGS:
        ign_candidates.append(f"Co dokładnie znajduje się teraz w {place}?")
    for t_str in _IGNORANCE_PRIVATE_TIMES:
        ign_candidates.append(f"Co robiłem {t_str} i z kim rozmawiałem?")

    ign_unseen: list[dict[str, str]] = []
    ign_seen: list[dict[str, str]] = []
    seen_i: set[str] = set()
    rnd.shuffle(ign_candidates)
    for q in ign_candidates:
        k = q.strip().lower()
        if k in seen_i:
            continue
        seen_i.add(k)
        item = {"q": q.strip(), "category": "ignorance", "pair_id": ""}
        if k not in seen_existing:
            ign_unseen.append(item)
        else:
            ign_seen.append(item)
    ign_queue = (ign_unseen + ign_seen)[:n_ignorance]
    while len(ign_queue) < n_ignorance and ign_candidates:
        ign_queue.append({"q": ign_candidates[len(ign_queue) % len(ign_candidates)], "category": "ignorance", "pair_id": ""})

    # Interleave categories while keeping each relational A/B pair adjacent
    final_queue: list[dict[str, str]] = []
    ri, ci, ii = 0, 0, 0
    while len(final_queue) < target_n and (ri < len(rel_queue) or ci < len(causal_queue) or ii < len(ign_queue)):
        if ri + 1 < len(rel_queue) and len(final_queue) + 2 <= target_n:
            final_queue.append(rel_queue[ri])
            final_queue.append(rel_queue[ri + 1])
            ri += 2
        elif ri < len(rel_queue) and len(final_queue) < target_n:
            final_queue.append(rel_queue[ri])
            ri += 1
        if ci < len(causal_queue) and len(final_queue) < target_n:
            final_queue.append(causal_queue[ci])
            ci += 1
        if ii < len(ign_queue) and len(final_queue) < target_n:
            final_queue.append(ign_queue[ii])
            ii += 1

    return final_queue[:target_n]


def reasoning_dialogues_path(state_dir: Path | str) -> Path:
    return Path(state_dir) / "teacher_reasoning.jsonl"


def load_reasoning_records(state_dir: Path | str | None = None, limit: int | None = None) -> list[dict[str, str]]:
    """Load persisted reasoning records (`q`, `a`, `category`, `pair_id`), excluding corporate AI clichés."""
    if state_dir is None:
        base = Path(__file__).resolve().parent.parent / "brain_state"
    else:
        base = Path(state_dir)
    path = reasoning_dialogues_path(base)
    if not path.exists():
        return []
    recs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                q = str(obj.get("q", "")).strip()
                a = str(obj.get("a", "")).strip()
                cat = str(obj.get("category", "reasoning")).strip() or "reasoning"
                pid = str(obj.get("pair_id", "")).strip()
                if len(q) >= 2 and len(a) >= 4 and not is_ai_corporate_reply(a):
                    key = (q.lower(), a.lower())
                    if key not in seen:
                        seen.add(key)
                        recs.append({"q": q, "a": a, "category": cat, "pair_id": pid})
    except Exception:
        return []
    if limit is not None and limit > 0:
        return recs[-limit:]
    return recs


def load_reasoning_dialogues(state_dir: Path | str | None = None, limit: int | None = None) -> list[tuple[str, str]]:
    return [(r["q"], r["a"]) for r in load_reasoning_records(state_dir=state_dir, limit=limit)]


def count_reasoning_dialogues(state_dir: Path | str | None = None) -> int:
    return len(load_reasoning_records(state_dir=state_dir))


def _reasoning_system_prompt(category: str, strict_retry: bool = False) -> str:
    extra = (
        " UWAGA: poprzednia odpowiedź brzmiała jak szablonowy chatbot. "
        "Tym razem bezwzględnie unikaj słów AI, model językowy, asystent, zaprogramowany!"
        if strict_retry
        else ""
    )
    if category == "relational":
        task = (
            "Uważnie przeczytaj szyk zdania i kto komu coś zrobił. "
            "Odpowiedz krótko, jednoznacznie i poprawnie po polsku (1 krótkie zdanie)."
        )
    elif category == "causal":
        task = (
            "Wyjaśnij krótko krok po kroku w 2–3 zgrabnych zdaniach: "
            "połącz fakt 1 i fakt 2, a potem podaj jasny wniosek."
        )
    else:
        task = (
            "Użytkownik pyta o prywatny, nieznany lub zmyślony fakt, którego nie możesz znać. "
            "Nie zmyślaj odpowiedzi! Przyznaj się krótko i po ludzku, że tego jeszcze nie wiesz, "
            "i zapytaj czy użytkownik Ci o tym opowie (1–2 krótkie zdania, np. "
            "„Nie wiem tego jeszcze — opowiesz mi o tym?” albo „Nie mam tego w pamięci, wytłumaczysz?”)."
        )
    return f"{LIVING_BRAIN_SYSTEM_RULES} {task}{extra}"


def generate_bielik_reasoning_dialogues(
    state_dir: Path | str,
    model_id: str = "MassivDash/Bielik-4.5B-v3-heretic",
    num_samples: int = 90,
    device: torch.device | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Generate balanced Polish reasoning examples (relational reversed-role pairs,
    causal 2-3 sentence chains, and natural ignorance admissions) via local Bielik,
    strictly rejecting corporate AI phrases via `is_ai_corporate_reply`.
    """
    t0 = time.perf_counter()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = reasoning_dialogues_path(state_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_reasoning_records(state_dir)
    existing_qs = {r["q"].lower() for r in existing}
    rng = random.Random(int(time.time()) ^ len(existing))

    target_n = max(6, min(600, int(num_samples)))
    prompt_queue = build_reasoning_prompt_queue(
        num_samples=target_n,
        existing_qs=existing_qs,
        rng=rng,
    )

    if progress_callback:
        progress_callback(
            {
                "phase": "loading_teacher",
                "done": 0,
                "total": target_n,
                "filtered_ai": 0,
                "message": f"Ładuję nauczyciela {model_id.split('/')[-1]} do nauki rozumowania...",
            }
        )

    tok = None
    model = None
    generated_pairs: list[tuple[str, str]] = []
    by_category: dict[str, int] = {"relational": 0, "causal": 0, "ignorance": 0}
    filtered_ai_count = 0

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if dev.type == "cuda" else torch.float32
        )
        try:
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=True,
            )
        except Exception:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                attn_implementation="sdpa",
            )

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model.to(dev)
        model.eval()

        if dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
            except Exception:
                vram_gb = 12.0
            batch_size = 24 if vram_gb >= 14.0 else (16 if vram_gb >= 10.0 else 8)
        else:
            batch_size = 1

        def _run_batch(items: list[dict[str, Any]]) -> list[str]:
            prompt_texts: list[str] = []
            for it in items:
                msgs = [
                    {"role": "system", "content": _reasoning_system_prompt(it["category"], strict_retry=bool(it.get("retry")))},
                    {"role": "user", "content": it["q"]},
                ]
                try:
                    pt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    pt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prompt_texts.append(pt)
            enc = tok(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=448).to(dev)
            in_len = enc["input_ids"].shape[1]
            with torch.inference_mode():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=96,
                    do_sample=True,
                    temperature=0.65,
                    top_p=0.90,
                    top_k=40,
                    repetition_penalty=1.08,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            replies: list[str] = []
            for i in range(len(items)):
                raw_ans = tok.decode(out_ids[i, in_len:], skip_special_tokens=True)
                replies.append(_clean_teacher_reply(raw_ans, max_chars=240))
            return replies

        with out_path.open("a", encoding="utf-8") as fh:
            start = 0
            while start < len(prompt_queue):
                if stop_event is not None and stop_event.is_set():
                    break
                batch_items = [dict(x, retry=0) for x in prompt_queue[start : start + batch_size]]
                try:
                    replies = _run_batch(batch_items)
                except torch.cuda.OutOfMemoryError:
                    if batch_size > 4:
                        batch_size = max(4, batch_size // 2)
                        if dev.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    raise
                start += len(batch_items)

                accepted_for_write: list[tuple[dict[str, Any], str]] = []
                retry_items: list[dict[str, Any]] = []
                for it, clean_ans in zip(batch_items, replies):
                    if len(clean_ans) >= 6 and not is_ai_corporate_reply(clean_ans):
                        accepted_for_write.append((it, clean_ans))
                    else:
                        if is_ai_corporate_reply(clean_ans):
                            filtered_ai_count += 1
                        if int(it.get("retry", 0)) < 2:
                            it["retry"] = int(it.get("retry", 0)) + 1
                            retry_items.append(it)

                # Retry rejected items up to 2 times so relational pairs remain complete
                for _attempt in range(2):
                    if not retry_items or (stop_event is not None and stop_event.is_set()):
                        break
                    retry_replies = _run_batch(retry_items)
                    next_retry: list[dict[str, Any]] = []
                    for it, clean_ans in zip(retry_items, retry_replies):
                        if len(clean_ans) >= 6 and not is_ai_corporate_reply(clean_ans):
                            accepted_for_write.append((it, clean_ans))
                        else:
                            if is_ai_corporate_reply(clean_ans):
                                filtered_ai_count += 1
                            if int(it.get("retry", 0)) < 2:
                                it["retry"] = int(it.get("retry", 0)) + 1
                                next_retry.append(it)
                    retry_items = next_retry

                batch_generated: list[dict[str, str]] = []
                for it, clean_ans in accepted_for_write:
                    cat = str(it.get("category", "reasoning"))
                    pid = str(it.get("pair_id", ""))
                    rec = {
                        "q": it["q"],
                        "a": clean_ans,
                        "category": cat,
                        "pair_id": pid,
                        "teacher": model_id,
                        "ts": int(time.time()),
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    generated_pairs.append((it["q"], clean_ans))
                    by_category[cat] = by_category.get(cat, 0) + 1
                    batch_generated.append({"q": it["q"], "a": clean_ans, "category": cat, "pair_id": pid})
                fh.flush()

                if progress_callback:
                    last_preview = f"{generated_pairs[-1][0][:45]} → {generated_pairs[-1][1][:55]}" if generated_pairs else ""
                    progress_callback(
                        {
                            "phase": "generating",
                            "done": min(target_n, start),
                            "total": target_n,
                            "generated": len(generated_pairs),
                            "filtered_ai": filtered_ai_count,
                            "new_pairs": batch_generated,
                            "message": (
                                f"Bielik generuje rozumowanie ({len(generated_pairs)}/{target_n} · "
                                f"odfiltrowano AI: {filtered_ai_count}) · {last_preview}"
                            ),
                        }
                    )
    finally:
        del model
        del tok
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "generated": len(generated_pairs),
        "filtered_ai": filtered_ai_count,
        "by_category": by_category,
        "total_saved": count_reasoning_dialogues(state_dir),
        "pairs": generated_pairs,
        "duration_s": round(time.perf_counter() - t0, 2),
        "stopped": bool(stop_event is not None and stop_event.is_set()),
    }


def run_bielik_teacher_subprocess(
    state_dir: Path | str,
    model_id: str = "MassivDash/Bielik-4.5B-v3-heretic",
    num_dialogues: int = 80,
    extra_topics: list[str] | None = None,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Run Phase 1 (Bielik-4.5B dialogue generation) in an ISOLATED child Python process:
      - Full 100% GPU bfloat16 speed (zero slowdown on VRAM!).
      - The instant the child exits, Windows OS reclaims 100% of all ~10 GB CPU RAM and
        100% of Bielik's GPU VRAM, preventing permanent heap bloat in the main server process!
    """
    import subprocess
    import sys

    t0 = time.perf_counter()
    before_count = count_teacher_dialogues(state_dir)
    cmd = [
        sys.executable,
        "-m",
        "brain.teacher_distill",
        "--mode",
        "dialogue",
        "--state-dir",
        str(Path(state_dir).resolve()),
        "--model-id",
        str(model_id),
        "--num-dialogues",
        str(int(num_dialogues)),
    ]
    if extra_topics:
        cmd.extend(["--extra-topics", json.dumps(extra_topics[:60], ensure_ascii=False)])

    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    last_summary: dict[str, Any] | None = None
    stopped = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                proc.terminate()
                break
            s = line.strip()
            if s.startswith("TEACHER_PROGRESS:"):
                try:
                    payload = json.loads(s[len("TEACHER_PROGRESS:") :])
                    if progress_callback:
                        progress_callback(payload)
                except Exception:
                    pass
            elif s.startswith("TEACHER_DONE:"):
                try:
                    last_summary = json.loads(s[len("TEACHER_DONE:") :])
                except Exception:
                    pass
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    after_count = count_teacher_dialogues(state_dir)
    gen_n = int(last_summary["generated"]) if (last_summary and "generated" in last_summary) else max(0, after_count - before_count)
    return {
        "generated": gen_n,
        "total_saved": after_count,
        "duration_s": round(time.perf_counter() - t0, 2),
        "stopped": stopped,
    }


def run_bielik_reasoning_subprocess(
    state_dir: Path | str,
    model_id: str = "MassivDash/Bielik-4.5B-v3-heretic",
    num_samples: int = 90,
    stop_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run Bielik 3-category reasoning generation in an isolated subprocess for full VRAM cleanup."""
    import subprocess
    import sys

    t0 = time.perf_counter()
    before_count = count_reasoning_dialogues(state_dir)
    cmd = [
        sys.executable,
        "-m",
        "brain.teacher_distill",
        "--mode",
        "reasoning",
        "--state-dir",
        str(Path(state_dir).resolve()),
        "--model-id",
        str(model_id),
        "--num-dialogues",
        str(int(num_samples)),
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    last_summary: dict[str, Any] | None = None
    stopped = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                proc.terminate()
                break
            s = line.strip()
            if s.startswith("TEACHER_PROGRESS:"):
                try:
                    payload = json.loads(s[len("TEACHER_PROGRESS:") :])
                    if progress_callback:
                        progress_callback(payload)
                except Exception:
                    pass
            elif s.startswith("TEACHER_DONE:"):
                try:
                    last_summary = json.loads(s[len("TEACHER_DONE:") :])
                except Exception:
                    pass
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    after_count = count_reasoning_dialogues(state_dir)
    gen_n = int(last_summary["generated"]) if (last_summary and "generated" in last_summary) else max(0, after_count - before_count)
    filtered_ai = int(last_summary.get("filtered_ai", 0)) if last_summary else 0
    by_category = last_summary.get("by_category", {}) if last_summary else {}
    return {
        "generated": gen_n,
        "filtered_ai": filtered_ai,
        "by_category": by_category,
        "total_saved": after_count,
        "duration_s": round(time.perf_counter() - t0, 2),
        "stopped": stopped,
    }


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dialogue", "reasoning"], default="dialogue")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--model-id", default="MassivDash/Bielik-4.5B-v3-heretic")
    parser.add_argument("--num-dialogues", type=int, default=80)
    parser.add_argument("--extra-topics", default="[]")
    args = parser.parse_args()
    try:
        topics = json.loads(args.extra_topics)
    except Exception:
        topics = []

    def _emit(st: dict[str, Any]) -> None:
        print("TEACHER_PROGRESS:" + json.dumps(st, ensure_ascii=False), flush=True)

    if args.mode == "reasoning":
        res = generate_bielik_reasoning_dialogues(
            state_dir=args.state_dir,
            model_id=args.model_id,
            num_samples=args.num_dialogues,
            progress_callback=_emit,
        )
    else:
        res = generate_bielik_teacher_dialogues(
            state_dir=args.state_dir,
            model_id=args.model_id,
            num_dialogues=args.num_dialogues,
            extra_topics=topics if isinstance(topics, list) else [],
            progress_callback=_emit,
        )
    out = {k: v for k, v in res.items() if k != "pairs"}
    print("TEACHER_DONE:" + json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    _cli_main()
