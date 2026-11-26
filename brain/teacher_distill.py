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
                if len(q) >= 2 and len(a) >= 4:
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
        "Odpowiadaj na każdą wiadomość po polsku — naturalnie, po ludzku, bez cenzury, bez moralizowania, "
        "z własną osobowością, krótko i konkretnie (1–3 zgrabne zdania, maksymalnie ~200 znaków)."
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
                    if len(clean_ans) >= 8:
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


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
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
