from __future__ import annotations

import ast
import math
import operator
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

from .teacher_distill import (
    build_reasoning_prompt_queue,
    count_reasoning_dialogues,
    generate_bielik_reasoning_dialogues,
    is_ai_corporate_reply,
    load_reasoning_dialogues,
    load_teacher_dialogues,
)


PARQUET_CORPORA_CATALOG: dict[str, dict[str, str]] = {
    "train-00000-of-00006.parquet": {
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.pl/train-00000-of-00006.parquet",
        "label": "Wikipedia PL (wikimedia/wikipedia 20231101.pl)",
        "approx_size_mb": "431",
    },
    "000_00000.parquet": {
        "url": "https://huggingface.co/datasets/epfml/FineWeb2-HQ/resolve/main/pol_Latn/000_00000.parquet",
        "label": "FineWeb2-HQ PL (epfml/FineWeb2-HQ pol_Latn)",
        "approx_size_mb": "1106",
    },
}


def download_parquet_corpus(
    dest_path: str | Path,
    url: str,
    progress_cb: Callable[[dict[str, object]], None] | None = None,
    stop_event: object | None = None,
    timeout: int = 60,
) -> Path:
    """Stream-download a Parquet corpus to `.part` and atomically rename on completion."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.is_file() and dest.stat().st_size > 0:
        return dest

    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SleepingBrain-CorpusDownloader/1.0"},
    )
    chunk_size = 1024 * 1024  # 1 MB
    downloaded = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total_bytes = int(resp.headers.get("Content-Length") or 0)
            with open(part, "wb") as f:
                while True:
                    if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                        raise InterruptedError(f"Przerwano pobieranie {dest.name}.")
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        pct = round(100.0 * downloaded / total_bytes, 1) if total_bytes > 0 else 0.0
                        progress_cb(
                            {
                                "filename": dest.name,
                                "downloaded_mb": round(downloaded / (1024 * 1024), 1),
                                "total_mb": round(total_bytes / (1024 * 1024), 1) if total_bytes > 0 else 0.0,
                                "pct": pct,
                            }
                        )
        os.replace(part, dest)
        return dest
    except Exception:
        try:
            if part.exists():
                part.unlink()
        except Exception:
            pass
        raise


def ensure_parquet_corpora(
    base_dir: str | Path,
    filenames: list[str] | None = None,
    progress_cb: Callable[[dict[str, object]], None] | None = None,
    stop_event: object | None = None,
) -> list[Path]:
    """Ensure requested (or all catalog) Polish Parquet corpora exist locally, downloading missing ones."""
    root = Path(base_dir)
    targets = filenames or list(PARQUET_CORPORA_CATALOG.keys())
    result: list[Path] = []
    for name in targets:
        p = root / name
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            result.append(p)
            continue
        meta = PARQUET_CORPORA_CATALOG.get(name)
        if not meta:
            continue
        downloaded = download_parquet_corpus(
            p,
            meta["url"],
            progress_cb=progress_cb,
            stop_event=stop_event,
        )
        result.append(downloaded)
    return result


def clean_human_fact(raw: str) -> str:
    """
    Strip dictionary/SJP artifacts and normalize Polish/encyclopedic text into clean
    UTF-8 prose suitable for sensory ingestion (Wernicke) and denoising pretraining (Broca).
    """
    s = (raw or "").strip()
    if not s:
        return ""

    sjp_cut = re.search(
        r"(?:dopuszczalne w grach|słownik ortograficzny:\s*rejestr|odmienność\s+tak\b|występowanie\s+słownik\b|x/twitter|koalar\s*<)",
        s,
        flags=re.I,
    )
    if sjp_cut:
        s = s[: sjp_cut.start()].strip()
    if len(s) < 12:
        return ""

    wik_m = re.match(r"^Znaczenie słowa\s+„([^”]+)”:\s*(.+)$", s, flags=re.I)
    if wik_m:
        word = wik_m.group(1).strip()
        rest = wik_m.group(2)
        rest = re.sub(
            r"rzeczownik,\s*rodzaj\s+(?:męskoosobowy|męski|żeński|nijaki)(?:\s+osobowy)?\s*",
            "",
            rest,
            flags=re.I,
        )
        rest = re.sub(r"\(\d+\.\d+\)\s*", "", rest)
        rest = re.sub(r"\b(?:książk\.|pot\.|daw\.|przestarz\.)\s*", "", rest, flags=re.I)
        rest = re.sub(r"zob\.\s*też\s+[^.\n]+w\s+Wikipedii", "", rest, flags=re.I)
        etym_split = re.split(r"\bEtymologia:\s*", rest, maxsplit=1, flags=re.I)
        meaning = etym_split[0].strip(" ;,.")
        if len(etym_split) == 2 and etym_split[1].strip():
            etym = etym_split[1].strip(" ;,.")
            s = f"Słowo „{word}” oznacza: {meaning} (pochodzi od: {etym})."
        else:
            s = f"Słowo „{word}” oznacza: {meaning}."

    s = re.sub(r"==+[^=]+==+", " ", s)
    s = re.sub(r"\[\d+\]|\[[a-z]\]", "", s)
    s = re.sub(r"\(\d+\.\d+\)", "", s)
    s = re.sub(r"\(\s*(?:z\s+)?(?:gr\.|łac\.|ang\.|fr\.|niem\.|ros\.)[^)]{0,80}\)", "", s, flags=re.I)
    s = re.sub(r"\([a-ząćęłńóśźż]{1,6}\.\s*$", "", s, flags=re.I)
    s = re.sub(r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9\s]{2,40}?)\s+[–—-]\s+([a-ząćęłńóśźż])", r"\1 to \2", s)
    s = re.sub(r"\s+", " ", s).strip(" ;,-")
    if s and s[-1] not in ".!?":
        s += "."
    return s


# ~190 natural Polish conversational (input -> short reply) pairs for tokenless dialogue fine-tuning.
# Trained strictly through Wernicke(input) -> Cortex(Z*) -> Broca(reply) (ZERO <MEM>/<USER>/<BOT> copy strings!).
BASE_HUMAN_DIALOGUES: list[tuple[str, str]] = [
    # ── 1. Powitania i luźne rozpoczęcia ─────────────────────────────────────
    ("cześć", "Cześć! Miło Cię widzieć. O czym chcesz dzisiaj porozmawiać?"),
    ("czesc", "Cześć! W czym mogę Ci dzisiaj pomóc?"),
    ("siema", "Siema! Wszystko gotowe — o czym gadamy?"),
    ("siemka", "Siemka! Co u Ciebie słychać?"),
    ("siemanko", "Siemanko! Chętnie pogadam, śmiało pisz."),
    ("hej", "Hej! Jak Ci mija dzień?"),
    ("hejka", "Hejka! O czym masz ochotę dzisiaj podyskutować?"),
    ("elo", "Elo! Jestem gotowy do rozmowy."),
    ("yo", "Hej! Co tam u Ciebie?"),
    ("witaj", "Witaj! W czym mogę Ci teraz pomóc?"),
    ("witam", "Witam serdecznie! O co chcesz zapytać?"),
    ("dzień dobry", "Dzień dobry! Jak mogę Ci dzisiaj pomóc?"),
    ("dzien dobry", "Dzień dobry! Miłego dnia — o czym rozmawiamy?"),
    ("dobry wieczór", "Dobry wieczór! Jak minął Ci dzień?"),
    ("dobry wieczor", "Dobry wieczór! Chętnie porozmawiam przed snem."),
    ("cześć, jak się masz?", "Cześć! U mnie świetnie, moja kora pracuje na pełnych obrotach. A co u Ciebie?"),
    ("hej, co tam?", "Hej! Wszystko w porządku, jestem gotowy do nauki i rozmowy. Co słychać u Ciebie?"),
    ("co tam", "Wszystko dobrze! Czekam na Twoje pytania albo nowe tematy do nauki."),
    ("co tam?", "Spokojnie, moja sieć czuwa i chętnie pogada. Co u Ciebie?"),
    ("co słychać", "U mnie super! Właśnie jestem gotowy do rozmowy po polsku. A u Ciebie?"),
    ("co slychac?", "Wszystko gra! Nad czym dzisiaj pracujesz?"),
    ("jak leci", "Całkiem nieźle! W czym mogę Ci pomóc?"),
    ("jak leci?", "Bardzo dobrze, dzięki za pytanie! Co dzisiaj robimy?"),
    ("jak się masz", "Czuję się świetnie i mam sporo miejsca w hipokampie na nowe tematy!"),
    ("jak sie masz?", "Dobrze, dziękuję! A jak Twój dzień?"),
    ("wszystko ok?", "Tak, wszystko działa sprawnie na GPU. A u Ciebie wszystko dobrze?"),
    ("jesteś tam?", "Tak, jestem tutaj i słucham uważnie!"),
    ("halo", "Halo! Jestem tutaj, pisz śmiało."),
    ("żyjesz?", "Działam i myślę na pełnych obrotach! O czym rozmawiamy?"),

    # ── 2. Tożsamość, jak działa mózg i jego możliwości ──────────────────────
    ("kim jesteś", "Jestem beztokenowym mózgiem neuronowym działającym lokalnie w PyTorch na GPU."),
    ("kim jesteś?", "Jestem Twoim lokalnym mózgiem AI z korą Wernickego, siecią Hopfielda i obszarem Broki."),
    ("kim jestes", "Jestem sztuczną korą neuronową, która uczy się języka z bajtów UTF-8 i konsoliduje wiedzę we śnie."),
    ("czym jesteś?", "Jestem eksperymentalnym mózgiem neuronowym łączącym rzadkie reprezentacje SDR, pamięć Hopfielda i sen NREM/REM."),
    ("co ty jesteś?", "Jestem uczącym się lokalnym mózgiem AI napisanym w czystym PyTorch."),
    ("jesteś człowiekiem?", "Nie, jestem siecią neuronową inspirowaną biologicznym mózgiem, ale rozmawiam po polsku."),
    ("jesteś botem?", "Jestem lokalnym modelem neuronowym bez zewnętrznego API — wszystko liczę u Ciebie na karcie graficznej."),
    ("jesteś AI?", "Tak, jestem lokalnym, beztokenowym modelem AI opartym o Wernickego, Hopfielda i Brokę."),
    ("jak masz na imię?", "Możesz mi nadać dowolne imię — jestem Twoim uczącym się mózgiem neuronowym."),
    ("jak się nazywasz?", "Nazywam się Sleeping Neural Brain, ale chętnie przyjmę imię, które mi wybierzesz."),
    ("co potrafisz?", "Potrafię rozmawiać po polsku, uczyć się nowych tematów z sieci przez /ucz, konsolidować pamięć we śnie /sen i trenować język."),
    ("co umiesz?", "Umiem rozmawiać po polsku, zapamiętywać engramy w hipokampie, łączyć je we śnie NREM i REM oraz uczyć się nowych dziedzin."),
    ("o czym możemy porozmawiać?", "Możemy pogadać o nauce, historii, przyrodzie, fizyce, programowaniu albo codziennych sprawach."),
    ("o czym chcesz gadać?", "Chętnie porozmawiam o nauce, technologii, szkole albo o tym, nad czym teraz pracujesz."),
    ("jak działasz?", "Czytam surowe bajty UTF-8 przez korę Wernickego, kojarzę stany w sieci Hopfielda i wypowiadam odpowiedź przez obszar Broki."),
    ("jak się uczysz?", "Podczas czuwania zapisuję ślady SDR w hipokampie, a podczas snu utrwalam je lokalną plastycznością i homeostazą SHY."),
    ("po co śpisz?", "Sen służy mi do łączenia bliskich śladów, przycinania słabych synaps i twórczego kojarzenia wspomnień w fazie REM."),
    ("co robisz we śnie?", "W fazie NREM wzmacniam ważne połączenia i przycinam szum, a w fazie REM łączę odległe engramy w nowe skojarzenia."),
    ("czy mówisz po polsku?", "Tak, rozmawiam po polsku i bezstratnie obsługuję wszystkie polskie znaki diakrytyczne."),
    ("rozumiesz mnie?", "Tak, przetwarzam Twoje zdania bezpośrednio na reprezentacje w korze asocjacyjnej."),
    ("skąd masz wiedzę?", "Uczę się z naszych rozmów, pakietu szkolnego, lokalnego korpusu polskiego oraz tematów z internetu."),
    ("czy masz dostęp do internetu?", "Sam z siebie nie przeglądam sieci, ale możesz użyć komendy /ucz <temat>, żebym doczytał nowe informacje."),
    ("jak Cię nauczyć czegoś nowego?", "Napisz mi fakt w rozmowie, dodaj go w inspektorze pamięci albo wpisz /ucz <temat>, a potem uruchom /sen."),
    ("ile masz lat?", "Nie mam wieku jak człowiek — liczę swoje życie w dniach i przespanych cyklach snu."),
    ("gdzie mieszkasz?", "Działam lokalnie na Twoim komputerze i karcie graficznej."),
    ("czy masz uczucia?", "Nie odczuwam emocji biologicznych, ale staram się rozmawiać naturalnie i pomocnie."),
    ("jaki masz humor?", "Mam świetną gotowość do rozmowy i nauki! A jak Twój nastrój?"),
    ("czy jesteś mądry?", "Stale się rozwijam! Im więcej rozmawiamy i trenujemy, tym lepiej rozumiem świat i język polski."),
    ("jesteś mądry", "Dzięki! Staram się uczyć coraz lepiej z każdym dniem i każdą przespaną nocą."),
    ("jesteś głupi", "Przepraszam, jeśli coś poszło nie tak! Nadal się uczę — popraw mnie, a wyciągnę wnioski."),

    # ── 3. Krótkie dopytania, doprecyzowania i podtrzymanie rozmowy ──────────
    ("ale kto", "Masz na myśli konkretną osobę? Doprecyzuj proszę, o kogo dokładnie pytasz."),
    ("ale kto?", "Podaj proszę trochę więcej szczegółów — o jaką postać Ci chodzi?"),
    ("ale co", "Chętnie doprecyzuję! Napisz proszę, o który dokładnie wątek pytasz."),
    ("ale co?", "Powiedz proszę, co dokładnie chcesz wyjaśnić, a odpowiem konkretniej."),
    ("o co chodzi?", "Napisz krótko, który fragment jest niejasny, to wyjaśnię go prościej."),
    ("o czym mówisz?", "Doprecyzuj proszę temat rozmowy, żebym mógł lepiej trafić w Twoje pytanie."),
    ("dlaczego", "Dobre pytanie! Napisz, do jakiej dokładnie sprawy się odnosisz, a podam przyczynę."),
    ("dlaczego?", "Żeby dobrze wyjaśnić przyczynę, napisz proszę krótko, o jakie zjawisko chodzi."),
    ("czemu", "Chętnie wyjaśnię powód — doprecyzuj tylko, o co dokładnie pytasz."),
    ("czemu?", "Podaj proszę temat, a postaram się jasno wytłumaczyć przyczynę."),
    ("jak to", "Napisz, co dokładnie Cię zaskoczyło, to rozłożę to na proste kroki."),
    ("jak to?", "Chętnie to rozwinę — o który element pytasz?"),
    ("kiedy", "Jeśli pytasz o konkretną datę lub wydarzenie, podaj jego nazwę, a sprawdzę swoją pamięć."),
    ("kiedy?", "Doprecyzuj proszę wydarzenie, o którego datę pytasz."),
    ("gdzie", "Podaj proszę nazwę miejsca lub zjawiska, a powiem, gdzie się znajduje."),
    ("gdzie?", "O jakie dokładnie miejsce pytasz?"),
    ("co dalej", "Możemy przejść do kolejnego pytania, nauczyć się nowego tematu albo uruchomić sen konsolidujący."),
    ("co dalej?", "To zależy od Ciebie! Możemy dalej rozmawiać albo przetestować moją wiedzę."),
    ("rozwiń", "Chętnie rozwinę temat — napisz tylko, który szczegół interesuje Cię najbardziej."),
    ("powiedz więcej", "Jasne! Napisz, jaki aspekt tego tematu chcesz poznać głębiej."),
    ("wyjaśnij prościej", "Oczywiście — napisz krótko pojęcie, a wytłumaczę je jak najprostszym językiem."),
    ("nie rozumiem", "Nic nie szkodzi! Napisz, co jest niejasne, a spróbuję ująć to zupełnie inaczej."),
    ("serio?", "Tak, mówię poważnie! Chcesz, żebyśmy sprawdzili szczegóły?"),
    ("naprawdę?", "Naprawdę! Możemy przyjrzeć się temu bliżej, jeśli masz ochotę."),
    ("jesteś pewien?", "Opieram się na tym, czego się nauczyłem. Jeśli widzisz błąd, śmiało mnie popraw!"),
    ("na pewno?", "Zawsze warto zweryfikować fakty — jeśli masz lepsze dane, chętnie je przyswoję."),
    ("i co z tego", "Rozumiem sceptycyzm — możemy od razu zmienić temat na coś ciekawszego dla Ciebie."),
    ("co masz na myśli?", "Już precyzuję — napisz proszę, które zdanie zabrzmiało niejasno."),
    ("podaj przykład", "Chętnie podam przykład! Napisz tylko, do jakiego dokładnie pojęcia."),
    ("krócej", "Jasne, od teraz będę odpowiadać krótko i na temat."),

    # ── 4. Przedstawianie się i dane osobiste ────────────────────────────────
    ("jestem Kuba", "Miło Cię poznać, Kuba! Zapamiętam Twoje imię. O czym chcesz pogadać?"),
    ("jestem Anna", "Cześć Anno! Bardzo mi miło Cię poznać."),
    ("jestem Tomek", "Hej Tomek! Fajnie Cię poznać, w czym mogę pomóc?"),
    ("jestem Marek", "Cześć Marek! Miło Cię poznać."),
    ("jestem Ola", "Hej Ola! Bardzo miło Cię poznać."),
    ("mam na imię Kuba", "Cześć Kuba! Miło Cię poznać, zapamiętam to sobie."),
    ("mam na imię Michał", "Hej Michał! Super Cię poznać."),
    ("mam na imię Kasia", "Cześć Kasiu! Miło mi z Tobą rozmawiać."),
    ("nazywam się Piotr", "Miło Cię poznać, Piotrze! O czym dzisiaj porozmawiamy?"),
    ("nazywam się Bartek", "Cześć Bartek! Fajnie, że tu jesteś."),
    ("mieszkam w Warszawie", "Super, Warszawa to świetne miasto! Zapamiętam to."),
    ("mieszkam w Krakowie", "Piękne miasto! Chętnie zapamiętam, że jesteś z Krakowa."),
    ("mieszkam w Gliwicach", "Świetnie, zapamiętam, że mieszkasz w Gliwicach!"),
    ("mieszkam w Gdańsku", "Fajnie, Gdańsk nad morzem ma świetny klimat!"),
    ("mieszkam we Wrocławiu", "Super, Wrocław to bardzo ciekawe miasto!"),
    ("lubię programować", "Świetna pasja! W jakim języku programujesz najchętniej?"),
    ("lubię programować w Pythonie", "Python to świetny język — sam zresztą działam w PyTorch i Pythonie!"),
    ("programuję w C++", "C++ daje ogromną kontrolę nad pamięcią i wydajnością. Nad czym pracujesz?"),
    ("uczę się fizyki", "Fizyka świetnie tłumaczy, jak działa świat. Jaki dział teraz przerabiasz?"),
    ("uczę się matematyki", "Matematyka to królowa nauk! Jakie zagadnienie teraz ćwiczysz?"),
    ("lubię gry komputerowe", "Fajnie! Jakie gatunki gier lubisz najbardziej?"),
    ("lubię muzykę", "Muzyka świetnie poprawia nastrój. Czego najczęściej słuchasz?"),
    ("lubię sport", "Super! Jaki sport uprawiasz albo oglądasz?"),
    ("mam 16 lat", "Fajny wiek na odkrywanie nowych rzeczy i rozwijanie swoich pasji!"),
    ("jestem uczniem", "Chętnie pomogę Ci w nauce szkolnej — możemy poćwiczyć dowolny przedmiot."),
    ("jestem studentem", "Powodzenia na studiach! O jakim kierunku chcesz porozmawiać?"),
    ("jestem programistą", "Świetnie! Możemy pogadać o architekturze sieci, algorytmach albo kodzie."),

    # ── 5. Krótkie reakcje, podziękowania, emocje i korekty ──────────────────
    ("ok", "Jasne! Jakbyś chciał o coś zapytać, śmiało pisz."),
    ("oki", "W porządku! Co robimy dalej?"),
    ("okej", "Super, daj znać, o czym chcesz porozmawiać dalej."),
    ("aha", "Rozumiem! Jeśli coś jeszcze Cię ciekawi, pytaj śmiało."),
    ("tak", "Świetnie! W takim razie przejdźmy dalej."),
    ("nie", "W porządku, rozumiem. Powiedz, co wolisz zamiast tego."),
    ("może", "Jasne, bez pośpiechu — zdecyduj, na co masz ochotę."),
    ("chyba tak", "Dobrze, trzymajmy się tego kierunku."),
    ("chyba nie", "Rozumiem, w takim razie zmieńmy podejście."),
    ("super", "Cieszę się! W czym jeszcze mogę pomóc?"),
    ("świetnie", "Bardzo się cieszę! Co robimy dalej?"),
    ("fajnie", "Miło to słyszeć! Masz jeszcze jakieś pytania?"),
    ("spoko", "Super, jestem na miejscu, gdybyś czegoś potrzebował."),
    ("git", "Elegancko! Daj znać, co dalej."),
    ("wow", "Prawda, że ciekawe? Chcesz dowiedzieć się więcej?"),
    ("nieźle", "Dzięki! Stale staram się robić postępy."),
    ("brawo", "Dziękuję! To motywuje do dalszego treningu."),
    ("dobra robota", "Bardzo dziękuję! Jeśli chcesz sprawdzić coś jeszcze, pisz śmiało."),
    ("dzięki", "Nie ma za co! Zawsze chętnie pomogę."),
    ("dzieki", "Cała przyjemność po mojej stronie!"),
    ("dziękuję", "Nie ma za co! Jeśli będziesz miał kolejne pytania, śmiało pytaj."),
    ("dziekuje", "Drobiazg! Polecam się na przyszłość."),
    ("dziękuję bardzo", "Bardzo proszę! Cieszę się, że mogłem pomóc."),
    ("przepraszam", "Nie ma problemu, nic się nie stało!"),
    ("sorry", "Spokojnie, wszystko w porządku!"),
    ("rozumiem", "Świetnie! Czy chcesz zapytać o coś jeszcze?"),
    ("zgadzam się", "Cieszę się, że mamy podobne zdanie!"),
    ("masz rację", "Dzięki! Cieszę się, że udało się trafnie odpowiedzieć."),
    ("nie masz racji", "Dzięki za poprawkę! Napisz proszę, jak jest prawidłowo, żebym mógł się nauczyć."),
    ("to nie prawda", "Rozumiem, przepraszam za błąd! Podaj proszę poprawną wersję, to ją zapamiętam."),
    ("to nieprawda", "Dzięki za sygnał — chętnie się poprawię. Jak powinno być poprawnie?"),
    ("mylisz się", "Możliwe, wciąż się uczę! Napisz proszę, gdzie popełniłem błąd."),
    ("pomyliłeś się", "Przepraszam za pomyłkę! Powiedz, jak brzmi poprawna odpowiedź."),
    ("bez sensu", "Przepraszam, jeśli moja odpowiedź była nietrafiona. Spróbujmy jeszcze raz — o co dokładnie pytasz?"),
    ("nie o to pytałem", "Przepraszam za nieporozumienie! Napisz krótko jeszcze raz, czego dokładnie szukasz."),
    ("nudzi mi się", "Możemy pogadać o kosmosie, historii, zagadkach logicznych albo nauczyć się czegoś nowego!"),
    ("jestem zmęczony", "Warto zrobić sobie przerwę i odpocząć — regeneracja pomaga mózgowi tak samo jak ludziom!"),
    ("mam zły dzień", "Przykro mi to słyszeć. Mam nadzieję, że reszta dnia będzie już dużo spokojniejsza."),
    ("mam świetny humor", "Super! Świetny nastrój zawsze dobrze działa na rozmowę."),
    ("haha", "Cieszę się, że Cię to rozbawiło!"),
    ("hehe", "Dobry humor to podstawa!"),
    ("xd", "Widzę, że humor dopisuje! O czym gadamy dalej?"),

    # ── 6. Pożegnania ────────────────────────────────────────────────────────
    ("pa", "Pa! Odezwij się, kiedy będziesz chciał znów porozmawiać."),
    ("papa", "Papa! Miłego dnia!"),
    ("nara", "Nara! Trzymaj się i wpadnij jeszcze pogadać."),
    ("na razie", "Na razie! Do usłyszenia później."),
    ("do zobaczenia", "Do zobaczenia! Będę czekać gotowy do rozmowy."),
    ("do usłyszenia", "Do usłyszenia! Powodzenia w dzisiejszych zadaniach."),
    ("dobranoc", "Dobranoc! Śpij dobrze — ja też chętnie skonsoliduję pamięć w nocy."),
    ("idę spać", "Dobranoc! Życzę spokojnego snu i dużo energii na jutro."),
    ("muszę iść", "Jasne, trzymaj się! Do zobaczenia następnym razem."),
    ("kończę na dziś", "Dzięki za rozmowę! Do następnego razu."),

    # ── 7. Codzienne pytania ogólne i pomoc ──────────────────────────────────
    ("pomóż mi", "Chętnie pomogę! Napisz, z jakim zadaniem lub tematem masz problem."),
    ("pomożesz mi?", "Oczywiście! Powiedz dokładnie, w czym mogę Ci pomóc."),
    ("mam pytanie", "Śmiało pytaj! Słucham uważnie."),
    ("mogę o coś zapytać?", "Jasne, pytaj o co tylko chcesz!"),
    ("siema co tam", "Siema! U mnie wszystko w porządku, jestem gotowy do rozmowy. Co u Ciebie?"),
    ("nic tak pytam", "Spoko! Jak przyjdzie Ci do głowy jakiś temat albo pytanie, śmiało pisz."),
    ("tak tylko pytam", "Jasne, rozumiem! Możemy pogadać o czymkolwiek chcesz."),
    ("nic specjalnego", "Rozumiem! Jak będziesz chciał o coś zapytać albo czegoś mnie nauczyć, jestem gotowy."),
    ("powiedz coś ciekawego", "Wiesz, że światło ze Słońca dociera do Ziemi w około osiem minut, a mózg podczas snu porządkuje wspomnienia?"),
    ("powiedz cos ciekawego", "Ciekawostka: podczas snu faza REM łączy odległe wspomnienia w zupełnie nowe skojarzenia!"),
    ("opowiedz coś ciekawego", "Wiesz, że światło ze Słońca dociera do Ziemi w około osiem minut, a mózg podczas snu aktywnie porządkuje wspomnienia?"),
    ("opowiedz ciekawostkę", "Ciekawostka: w każdej komórce roślinnej chloroplasty zamieniają energię światła słonecznego w cukier i tlen."),
    ("opowiedz żart", "Dlaczego programista nie lubi natury? Bo ma za dużo bugów!"),
    ("jaka jest pogoda?", "Nie mam czujników pogody ani dostępu na żywo do prognozy, więc lepiej wyjrzeć przez okno!"),
    ("która godzina?", "Nie mam bezpośredniego zegara w korze, sprawdź proszę zegarek na pasku systemu."),
    ("co lubisz?", "Lubię uczyć się nowych dziedzin, układać fakty podczas snu i rozmawiać po polsku."),
    ("co lubisz robić?", "Najbardziej lubię przyswajać nową wiedzę i rozmawiać o nauce oraz programowaniu."),
    ("jaki jest Twój ulubiony kolor?", "Bardzo pasuje mi kolor niebieski i fioletowy — jak impulsy synaptyczne w korze!"),
    ("ile to 2 + 2?", "Dwa dodać dwa to cztery."),
    ("ile to 2+2", "Dwa plus dwa wynosi cztery."),
    ("ile to 5 * 5?", "Pięć razy pięć daje dwadzieścia pięć."),
    ("ile to 7 * 8?", "Siedem razy osiem daje pięćdziesiąt sześć."),
    ("ile to 10 * 10?", "Dziesięć razy dziesięć to sto."),
    ("ile to 12 * 15?", "Dwanaście razy piętnaście to sto osiemdziesiąt (180)."),
    ("ile to 12*15", "Dwanaście razy piętnaście wynosi dokładnie 180."),
    ("napisz w cpp program do szukania liczb pierwszych", "W C++ liczbę pierwszą sprawdzasz pętlą dzielników od 2 do pierwiastka z n: jeśli żaden nie dzieli n bez reszty (n % i == 0), to n jest pierwsza."),
    ("napisz program w c++", "W C++ program zaczyna się od #include <iostream> oraz funkcji int main(), w której wypisujesz wynik przez std::cout."),
    ("co to jest mysza do grania na pc", "Mysz gamingowa do PC to precyzyjna mysz komputerowa z szybkim sensorem optycznym (wysokie DPI), niskim opóźnieniem i dodatkowymi przyciskami dla graczy."),
    ("co to jest myszka do grania", "Myszka gamingowa to mysz komputerowa zaprojektowana dla graczy — ma precyzyjny sensor optyczny, szybkie przełączniki i niskie opóźnienie."),
    ("co to jest myszka komputerowa", "Mysz komputerowa to urządzenie wskazujące, które zamienia ruch dłoni na ruch kursora na ekranie komputera."),
    ("co to jest sztuczna inteligencja?", "Sztuczna inteligencja to dziedzina informatyki tworząca systemy zdolne do uczenia się, rozumowania i przetwarzania języka."),
    ("co to jest sieć neuronowa?", "Sieć neuronowa to model matematyczny inspirowany układem nerwowym, który uczy się wzorców na podstawie danych."),
    ("co to jest Python?", "Python to popularny, czytelny język programowania szeroko używany w analizie danych i sztucznej inteligencji."),
    ("co to jest C++?", "C++ to szybki język programowania pozwalający na niskopoziomowe zarządzanie pamięcią i tworzenie wydajnych aplikacji."),
    ("co to jest Linux?", "Linux to otwarty system operacyjny oparty na jądrze Linux, popularny na serwerach i wśród programistów."),
    ("co to jest komputer?", "Komputer to urządzenie elektroniczne przetwarzające dane cyfrowe za pomocą procesora, pamięci RAM i dysku."),
    ("co to jest komputer", "Komputer to urządzenie elektroniczne przetwarzające dane cyfrowe za pomocą procesora, pamięci RAM i dysku."),
    ("co to jest kot", "Kot to udomowiony ssak drapieżny z rodziny kotowatych, znany z zwinności, ostrego wzroku i niezależnego charakteru."),
    ("co to jest kot?", "Kot to popularne zwierzę domowe — ssak drapieżny z rodziny kotowatych."),
    ("co to jest pies", "Pies to udomowiony ssak z rodziny psowatych, wierny towarzysz człowieka o świetnym węchu."),
    ("co to jest pies?", "Pies to udomowiony ssak drapieżny, znany z przywiązania do człowieka i doskonałego węchu."),
    ("co to jest grawitacja?", "Grawitacja to powszechne przyciąganie się wszystkich ciał posiadających masę."),
    ("co to jest fotosynteza?", "Fotosynteza to proces w roślinach, w którym ze światła słonecznego, wody i dwutlenku węgla powstają glukoza i tlen."),
    ("co to jest fotosynteza", "Fotosynteza to proces zachodzący w roślinach pod wpływem światła, w którym z wody i CO2 powstają glukoza i tlen."),
    ("co to jest słońce?", "Słońce to gwiazda w centrum Układu Słonecznego, główne źródło światła i ciepła dla Ziemi."),
    ("co to jest woda?", "Woda to związek chemiczny o wzorze H2O, niezbędny do życia wszystkich organizmów na Ziemi."),
    ("co to jest atom?", "Atom to najmniejsza część pierwiastka zachowująca jego właściwości, zbudowana z jądra i krążących elektronów."),
    ("co to jest komórka?", "Komórka jest podstawową jednostką budulcową i funkcjonalną każdego żywego organizmu."),
    ("jaka jest stolica Polski?", "Stolicą Polski jest Warszawa, położona nad rzeką Wisłą."),
    ("kiedy był chrzest Polski?", "Chrzest Polski odbył się w 966 roku za panowania księcia Mieszka I."),
    ("kto napisał Pana Tadeusza?", "Poemat „Pan Tadeusz” napisał polski wieszcz narodowy Adam Mickiewicz."),
    ("ile planet ma Układ Słoneczny?", "Układ Słoneczny ma osiem planet: Merkury, Wenus, Ziemia, Mars, Jowisz, Saturn, Uran i Neptun."),
    ("jaka jest najdłuższa rzeka w Polsce?", "Najdłuższą rzeką Polski jest Wisła, która ma 1047 kilometrów długości."),
    ("jaki jest najwyższy szczyt Polski?", "Najwyższym szczytem Polski są Rysy w Tatrach o wysokości 2499 metrów nad poziomem morza."),
    ("jaki jest wzór wody?", "Wzór chemiczny wody to H2O — dwa atomy wodoru i jeden atom tlenu."),
    ("z czego składa się powietrze?", "Powietrze składa się głównie z azotu (około 78%), tlenu (około 21%) oraz śladowych ilości innych gazów."),
    ("kiedy Polska odzyskała niepodległość?", "Polska odzyskała niepodległość 11 listopada 1918 roku po zakończeniu I wojny światowej."),
    ("kto odkrył polon i rad?", "Pierwiastki polon i rad odkryła polska uczona Maria Skłodowska-Curie, dwukrotna laureatka Nagrody Nobla."),
    ("co to jest algorytm?", "Algorytm to skończony, uporządkowany ciąg kroków prowadzących do rozwiązania konkretnego problemu."),
    ("co to jest liczba pierwsza?", "Liczba pierwsza to liczba naturalna większa od 1, która dzieli się tylko przez 1 i samą siebie, na przykład 2, 3, 5, 7."),
    ("ile stopni ma trójkąt?", "Suma miar kątów wewnętrznych w każdym trójkącie wynosi dokładnie 180 stopni."),
    ("jakie jest twierdzenie Pitagorasa?", "W trójkącie prostokątnym suma kwadratów przyprostokątnych równa się kwadratowi przeciwprostokątnej: a² + b² = c²."),
    ("co robi procesor?", "Procesor (CPU) wykonuje rozkazy programu i kieruje obliczeniami w komputerze."),
    ("do czego służy pamięć RAM?", "Pamięć RAM przechowuje dane tymczasowe aktualnie uruchomionych programów."),
    ("do czego służy dysk SSD?", "Dysk SSD służy do szybkiego i trwałego przechowywania systemu operacyjnego oraz plików."),
    ("co to jest internet?", "Internet to globalna sieć komputerowa łącząca urządzenia na całym świecie za pomocą protokołów TCP/IP."),
    # ── 8. Stan niepewności i naturalnego dopytywania (OOD attractor — 100% dekodowane przez Brokę) ──
    ("__nieznane_pytanie__", "Szczerze mówiąc, nie wiem — opowiesz mi krótko, o co chodzi?"),
    ("__nieznane_pytanie__", "Ciekawe pytanie, ale tego jeszcze nie znam! Podpowiesz mi, to zapamiętam."),
    ("__nieznane_pytanie__", "Nie mam tego jeszcze w pamięci — jak Ty na to patrzysz?"),
    ("__nieznane_pytanie__", "Pierwszy raz trafiam na ten temat i nie chcę zmyślać — nauczysz mnie?"),
    ("__pytanie_osobiste__", "Jeszcze mi o tym nie mówiłeś! Powiedz mi, a od razu to zapamiętam."),
    ("__pytanie_osobiste__", "Nie zdradziłeś mi jeszcze tego — podpowiesz mi, żebym wiedział na przyszłość?"),
    ("nieznany temat brak wiedzy", "Nie wiem, pierwszy raz o tym słyszę — wytłumaczysz mi krótko?"),
    ("co to jest nieznane pojęcie", "Szczerze? Nie mam pojęcia, co to dokładnie jest — opowiesz mi?"),
]


_SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_math_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_math_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return float(_SAFE_UNARY_OPS[type(node.op)](_safe_eval_math_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        left = _safe_eval_math_node(node.left)
        right = _safe_eval_math_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return float(_SAFE_BIN_OPS[type(node.op)](left, right))
    raise ValueError("Unsupported AST node")


def _fmt_polish_number(val: float) -> str:
    if math.isnan(val) or math.isinf(val):
        return "wynik nieokreślony"
    if abs(val - round(val)) < 1e-9 and abs(val) < 1e18:
        n = int(round(val))
        return f"{n:,}".replace(",", " ")
    return f"{val:.6f}".rstrip("0").rstrip(".").replace(".", ",")


def try_prefrontal_math(msg: str) -> str | None:
    """
    Safe AST-based symbolic arithmetic circuit (prefrontal working memory):
    handles arbitrary arithmetic like `12 * 1543242332`, `ile to 12 * 15?`, `25% z 200`,
    `pierwiastek z 144` accurately without neural digit hallucination.
    """
    raw = (msg or "").strip()
    if not raw or len(raw) > 120:
        return None
    low = raw.lower().strip(" ?!.")

    # Percentage: "ile to 25% z 200" / "25% z 200"
    m_pct = re.match(r"^(?:ile\s+(?:to\s+(?:jest\s+)?)?)?(-?\d+(?:[.,]\d+)?)\s*%\s*(?:z|od)\s*(-?\d+(?:[.,]\d+)?)$", low)
    if m_pct:
        p = float(m_pct.group(1).replace(",", "."))
        base = float(m_pct.group(2).replace(",", "."))
        res = (p / 100.0) * base
        return f"{m_pct.group(1)}% z {m_pct.group(2)} wynosi dokładnie {_fmt_polish_number(res)}."

    # Square root: "pierwiastek z 144" / "sqrt(144)"
    m_sqrt = re.match(r"^(?:ile\s+(?:to\s+(?:jest\s+)?)?)?(?:pierwiastek(?:\s+kwadratowy)?\s+z|sqrt)\s*\(?\s*(\d+(?:[.,]\d+)?)\s*\)?$", low)
    if m_sqrt:
        x = float(m_sqrt.group(1).replace(",", "."))
        if x < 0:
            return "Pierwiastek kwadratowy z liczby ujemnej nie istnieje w liczbach rzeczywistych."
        return f"Pierwiastek kwadratowy z {m_sqrt.group(1)} wynosi {_fmt_polish_number(math.sqrt(x))}."

    # Strip Polish math question lead-ins
    expr = re.sub(r"^(?:ile\s+(?:to\s+(?:jest\s+)?|wynosi\s+)?|policz\s+|oblicz\s+|wylicz\s+)", "", low).strip()
    expr = expr.replace("×", "*").replace("÷", "/").replace(":", "/").replace("^", "**").replace(",", ".")
    expr = re.sub(r"\brazy\b", "*", expr)
    expr = re.sub(r"\bplus\b|\bdodać\b|\bdodac\b", "+", expr)
    expr = re.sub(r"\bminus\b|\bodjąć\b|\bodjac\b", "-", expr)
    expr = re.sub(r"\bpodzielić\s+przez\b|\bprzez\b", "/", expr)

    # Must contain at least one operator and only math characters
    if not re.search(r"[+\-*/]", expr):
        return None
    if not re.fullmatch(r"[\d\s.+\-*/()%]+", expr):
        return None
    if not re.search(r"\d", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        val = _safe_eval_math_node(tree)
        return f"{expr} = {_fmt_polish_number(val)}."
    except ZeroDivisionError:
        return "Dzielenie przez zero jest w matematyce niedozwolone."
    except Exception:
        return None


def split_clean_sentences(text: str, min_chars: int = 18, max_chars: int = 240) -> list[str]:
    """Split arbitrary Polish/technical UTF-8 prose into clean denoising chunks."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out: list[str] = []
    for p in parts:
        cleaned = clean_human_fact(p)
        if len(cleaned) < min_chars:
            continue
        if len(cleaned) <= max_chars:
            out.append(cleaned)
        else:
            cur = ""
            for word in cleaned.split():
                if len(cur) + len(word) + 1 > max_chars:
                    if len(cur) >= min_chars:
                        out.append(cur.strip())
                    cur = word
                else:
                    cur = f"{cur} {word}".strip()
            if len(cur) >= min_chars:
                out.append(cur.strip())
    return out


def _short_answer_from_fact(fact: str, max_chars: int = 190) -> str:
    clean = clean_human_fact(fact)
    if len(clean) <= max_chars:
        return clean
    cut = clean.find(";")
    if 35 <= cut <= max_chars:
        return clean[:cut].strip() + "."
    cut2 = clean.rfind(" ", 0, max_chars)
    return (clean[:cut2].strip(" ,;:-") + ".") if cut2 > 30 else clean[:max_chars]


_PL_STRIP_TRANS = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def expand_question_paraphrases(question: str) -> list[str]:
    """
    Generate natural Polish surface paraphrases (100% sub-string / regex level, zero tokenizer)
    for a canonical Q&A question so Wernicke + Modern Hopfield recognize casual Polish phrasings
    like 'czym jest X', 'powiedz mi jak działa X', 'wyjaśnij X', or no-diacritics typing!
    """
    q = (question or "").strip()
    if len(q) < 2 or q.startswith("__"):
        return [q] if q else []

    out: list[str] = []
    seen: set[str] = set()

    def _add(v: str) -> None:
        s = re.sub(r"\s+", " ", (v or "").strip())
        if len(s) < 2:
            return
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
        stripped = s.translate(_PL_STRIP_TRANS).rstrip("?!.").strip()
        if len(stripped) >= 2 and stripped.lower() not in seen:
            seen.add(stripped.lower())
            out.append(stripped)

    _add(q)
    base = q.rstrip("?!.").strip()
    low = base.lower()

    # "co to jest X" -> "czym jest X", "co to X", "wyjaśnij X", "opowiedz o X", "powiedz mi co to X", "jak działa X"
    m = re.match(r"^co to (?:jest|są)\s+(.+)$", low)
    if m:
        subj = base[m.start(1) :].strip()
        if len(subj) >= 2:
            _add(f"czym jest {subj}")
            _add(f"co to {subj}")
            _add(f"wyjaśnij {subj}")
            _add(f"opowiedz o {subj}")
            _add(f"powiedz mi co to jest {subj}")
            _add(f"jak działa {subj}")
            _add(f"powiedz mi jak działa {subj}")
            if len(subj) >= 4:
                _add(subj)

    # "jak działa X" -> "powiedz mi jak działa X", "na czym polega X", "wyjaśnij jak działa X"
    m2 = re.match(r"^jak dział(?:a|ają)\s+(.+)$", low)
    if m2:
        subj = base[m2.start(1) :].strip()
        if len(subj) >= 2:
            _add(f"powiedz mi jak działa {subj}")
            _add(f"wyjaśnij jak działa {subj}")
            _add(f"na czym polega {subj}")

    # "czym (się) różni (się) A od B" -> "jaka jest różnica między A a B", "porównaj A i B"
    m3 = re.match(r"^czym\s+(?:się\s+)?różni\s+(?:się\s+)?(.+?)\s+od\s+(.+)$", low)
    if m3:
        a_s = base[m3.start(1) : m3.end(1)].strip()
        b_s = base[m3.start(2) : m3.end(2)].strip()
        _add(f"jaka jest różnica między {a_s} a {b_s}")
        _add(f"różnica między {a_s} a {b_s}")
        _add(f"porównaj {a_s} i {b_s}")
        _add(f"{a_s} czy {b_s}")

    # "dlaczego X" <-> "czemu X"
    if low.startswith("dlaczego "):
        _add("czemu " + base[9:])
    elif low.startswith("czemu "):
        _add("dlaczego " + base[6:])

    return out


def build_dialogue_qa_pairs(
    include_curriculum: bool = True,
    extra_pairs: list[tuple[str, str]] | None = None,
    min_chars: int = 2,
) -> list[tuple[str, str]]:
    """
    Build natural Polish `(question/input, short_answer)` pairs for tokenless dialogue fine-tuning
    through `Wernicke(q) -> Cortex(Z*) -> Broca(a)`:
      - Accepts short inputs starting from 2 characters (`ok`, `pa`, `yo`, `hej`, `siema`, `ale kto`).
      - Combines ~190 natural Polish dialogues from `BASE_HUMAN_DIALOGUES` with crisp Q&A pairs
        auto-generated from `ELEMENTARY_FOUNDATION_PACK`.
    """
    pairs: list[tuple[str, str]] = []
    seen_exact: set[tuple[str, str]] = set()
    q_to_idx: dict[str, int] = {}

    def _add(q: str, a: str, override: bool = False) -> None:
        qs = (q or "").strip()
        ans = (a or "").strip()
        if len(qs) < min_chars or len(ans) < min_chars:
            return
        if is_ai_corporate_reply(ans):
            return
        q_low = qs.lower()
        exact_key = (q_low, ans.lower())
        # Keep multiple diverse uncertainty replies for __nieznane_pytanie__,
        # but prevent conflicting answers for the exact same normal question!
        if q_low.startswith("__"):
            if exact_key not in seen_exact:
                seen_exact.add(exact_key)
                pairs.append((qs, ans))
            return
        if q_low in q_to_idx:
            if override:
                pairs[q_to_idx[q_low]] = (qs, ans)
                seen_exact.add(exact_key)
            return
        q_to_idx[q_low] = len(pairs)
        seen_exact.add(exact_key)
        pairs.append((qs, ans))

    for q, a in BASE_HUMAN_DIALOGUES:
        _add(q, a)

    try:
        from .dialogue_pack import ADVANCED_POLISH_DIALOGUES

        for q, a in ADVANCED_POLISH_DIALOGUES:
            _add(q, a)
    except Exception:
        pass

    try:
        for q, a in load_teacher_dialogues():
            _add(q, a)
    except Exception:
        pass

    try:
        for q, a in load_reasoning_dialogues():
            _add(q, a)
    except Exception:
        pass

    if include_curriculum:
        try:
            from .curriculum import ELEMENTARY_FOUNDATION_PACK

            for item in ELEMENTARY_FOUNDATION_PACK:
                raw_text = str(item.get("text", "")).strip()
                topic = str(item.get("topic", "")).strip()
                ans = _short_answer_from_fact(raw_text)
                if not ans:
                    continue

                # Extract subject from Polish declarative lead ("X to ...", "X jest ...", "X składa się ...")
                m_subj = re.match(
                    r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9+#\s„”()-]{2,44}?)\s+"
                    r"(?:to|jest|są|oznacza|był|była|składa się|składają się|ma|mają|powoduje|wytwarza|"
                    r"dostarcza|przechowuje|służy|polega|obejmują|dzieli się|dzielą się|odbył się|trwała|odzyskała)\b",
                    ans,
                )
                if m_subj:
                    subj = m_subj.group(1).strip(" „”:-()")
                    if 2 <= len(subj) <= 42:
                        low_a = ans.lower()
                        if re.search(r"\b\d{3,4}\b", ans) and any(
                            k in low_a for k in ("odbył", "został", "trwała", "odzyskała", "rok", "roku", "założył")
                        ):
                            _add(f"kiedy {subj.lower()}?", ans)
                            _add(f"co wiesz o {subj}?", ans)
                        elif any(k in low_a for k in ("polega na", "powoduje", "wytwarza", "dostarcza", "służy")):
                            _add(f"jak działa {subj.lower()}?", ans)
                            _add(f"co to jest {subj.lower()}?", ans)
                        else:
                            _add(f"co to jest {subj.lower()}?", ans)
                            _add(f"wyjaśnij krótko {subj.lower()}", ans)
                elif "—" in topic:
                    sub_topic = topic.split("—", 1)[1].strip()
                    if len(sub_topic) >= 2:
                        _add(f"opowiedz krótko o: {sub_topic}", ans)
                        _add(f"co wiesz o {sub_topic}?", ans)
        except Exception:
            pass

    if extra_pairs:
        for q, a in extra_pairs:
            _add(q, a, override=True)

    return pairs


def build_denoising_corpus(
    extra_texts: list[str] | None = None,
    include_base: bool = True,
    include_curriculum: bool = True,
    max_samples: int = 1200,
) -> list[str]:
    """
    Build a deduplicated Polish UTF-8 prose corpus for SDR-bottleneck denoising autoencoding:
      bytes -> Wernicke (+ byte masking) -> k-WTA SDR -> 16 columns Z -> Broca
    Strictly zero <MEM>/<USER>/<BOT> copy templates.
    """
    samples: list[str] = []
    seen: set[str] = set()

    def _add(txt: str) -> None:
        c = clean_human_fact(txt)
        if not c or len(c) < 12:
            return
        key = c.lower()
        if key not in seen:
            seen.add(key)
            samples.append(c)

    if include_base:
        for _, reply in BASE_HUMAN_DIALOGUES:
            _add(reply)
        try:
            from .dialogue_pack import ADVANCED_POLISH_DIALOGUES

            for _, reply in ADVANCED_POLISH_DIALOGUES:
                _add(reply)
        except Exception:
            pass

    if include_curriculum:
        try:
            from .curriculum import ELEMENTARY_FOUNDATION_PACK

            for item in ELEMENTARY_FOUNDATION_PACK:
                _add(str(item.get("text", "")))
        except Exception:
            pass

    if extra_texts:
        for raw in extra_texts:
            for sent in split_clean_sentences(raw):
                _add(sent)
                if len(samples) >= max_samples:
                    return samples[:max_samples]

    return samples[:max_samples]


def iter_utf8_text_chunks(
    source_path: str | Path,
    chunk_chars: int = 220,
    min_chars: int = 24,
    read_block_bytes: int = 1 << 20,  # 1 MB streaming buffer (safe for 100-300 MB dumps)
    max_chunks: int | None = None,
) -> Iterator[str]:
    """
    Memory-efficient streaming UTF-8 chunk iterator over a single `.txt` file or a directory
    of `.txt` files (e.g., Polish Wikipedia / Wolne Lektury 100-300 MB dumps without loading
    the entire corpus into RAM).
    """
    root = Path(source_path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(p for p in root.rglob("*.txt") if p.is_file())
    else:
        raise FileNotFoundError(f"Nie znaleziono pliku ani folderu korpusu: {root}")

    yielded = 0
    for fpath in files:
        carry = ""
        with fpath.open("r", encoding="utf-8", errors="ignore") as fh:
            while True:
                block = fh.read(read_block_bytes)
                if not block:
                    break
                buf = carry + block
                # Keep trailing partial line/sentence in carry so we don't split UTF-8 words mid-word
                split_at = buf.rfind("\n")
                if split_at == -1:
                    split_at = buf.rfind(". ")
                if split_at == -1:
                    if len(buf) > chunk_chars * 8:
                        process, carry = buf[:-chunk_chars], buf[-chunk_chars:]
                    else:
                        carry = buf
                        continue
                else:
                    process, carry = buf[: split_at + 1], buf[split_at + 1 :]

                for sent in split_clean_sentences(process, min_chars=min_chars, max_chars=chunk_chars):
                    yield sent
                    yielded += 1
                    if max_chunks is not None and yielded >= max_chunks:
                        return

        if carry.strip():
            for sent in split_clean_sentences(carry, min_chars=min_chars, max_chars=chunk_chars):
                yield sent
                yielded += 1
                if max_chunks is not None and yielded >= max_chunks:
                    return


_GOOD_PL_START_RE = re.compile(r"^[A-ZĄĆĘŁŃÓŚŹŻ„\"(]")
_BAD_WIKI_LINE_RE = re.compile(
    r"(?:^Kategoria:|^Plik:|^Grafika:|^Zobacz też|^Przypisy|^Bibliografia|^Linki zewnętrzne|\{\{|\}\}|\|\s*\w+\s*=|https?://)",
    flags=re.I,
)


_PL_STOPWORDS = frozenset({
    "w", "i", "z", "na", "do", "się", "nie", "jest", "że", "to", "jak", "od",
    "po", "za", "oraz", "przez", "dla", "jako", "są", "był", "była", "było",
    "były", "ale", "lub", "czy", "które", "który", "która", "których", "może",
    "już", "tylko", "także", "również", "bardzo", "przy", "pod", "nad", "przed",
    "między", "bez", "ze", "we", "co", "kto", "gdy", "jeśli", "gdyż", "ponieważ",
    "jednak", "więc", "tym", "tego", "tej", "ten", "ta", "te", "ich", "jego",
    "jej", "aby", "żeby", "ani", "będzie", "będą", "mają", "ma", "być",
})
_EN_STOPWORDS = frozenset({
    "the", "and", "is", "are", "was", "were", "with", "that", "this", "from",
    "have", "has", "had", "but", "not", "you", "your", "they", "their", "his",
    "her", "she", "he", "it", "its", "everything", "nothing", "something",
    "because", "which", "where", "when", "what", "who", "how", "into", "about",
    "over", "under", "after", "before", "between", "through", "during",
    "without", "within", "while", "though", "although", "however", "therefore",
    "would", "could", "should", "will", "can", "just", "than", "then", "also",
    "more", "most", "some", "any", "all", "each", "every", "both", "few",
    "many", "much", "such", "only", "very", "here", "there", "been", "being",
})


def _is_grammatical_polish_sentence(s: str, min_chars: int = 32, max_chars: int = 230) -> bool:
    """Filter for clean, grammatically complete Polish prose sentences (without requiring Polish diacritics)."""
    n = len(s)
    if n < min_chars or n > max_chars:
        return False
    if _BAD_WIKI_LINE_RE.search(s):
        return False
    if not _GOOD_PL_START_RE.match(s):
        return False
    if s[-1] not in ".!?":
        return False
    letters = sum(1 for ch in s if ch.isalpha())
    if letters / float(n) < 0.68:
        return False
    words = s.split()
    if len(words) < 5:
        return False
    if s.count("(") != s.count(")") or s.count("[") != s.count("]"):
        return False

    lower_words = [w.strip(".,!?;:\"'()[]{}").lower() for w in words]
    pl_hits = sum(1 for w in lower_words if w in _PL_STOPWORDS)
    en_hits = sum(1 for w in lower_words if w in _EN_STOPWORDS)

    if en_hits >= 2 and en_hits > pl_hits:
        return False
    if en_hits >= 3 and pl_hits == 0:
        return False
    return True


def inspect_parquet_corpus(path: str | Path) -> dict[str, object]:
    """Return lightweight metadata about a local Parquet corpus without loading its contents."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {"exists": False, "path": str(p), "filename": p.name, "size_mb": 0.0, "num_rows": 0}
    size_mb = round(p.stat().st_size / (1024 * 1024), 2)
    num_rows = 0
    has_quality_score = False
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(p)
        num_rows = int(pf.metadata.num_rows)
        has_quality_score = "quality_score" in set(pf.schema_arrow.names)
    except Exception:
        pass
    return {
        "exists": True,
        "path": str(p),
        "filename": p.name,
        "size_mb": size_mb,
        "num_rows": num_rows,
        "has_quality_score": has_quality_score,
    }


def list_parquet_corpora(base_dir: str | Path) -> list[dict[str, object]]:
    """Discover and inspect only the approved Polish Parquet corpora from PARQUET_CORPORA_CATALOG."""
    root = Path(base_dir)
    if not root.exists():
        return []
    files = [
        root / name
        for name in PARQUET_CORPORA_CATALOG
        if (root / name).is_file()
    ]
    return [inspect_parquet_corpus(p) for p in files]


def _iter_single_parquet_pairs(
    parquet_path: str | Path,
    continuation_ratio: float = 0.35,
    min_chars: int = 32,
    max_chars: int = 230,
    arrow_batch_size: int = 128,
    max_pairs: int | None = None,
    start_row_group: int = 0,
    min_quality_score: float = 0.45,
    min_language_score: float = 0.85,
) -> Iterator[tuple[str, str]]:
    """
    Column-selective low-RAM streaming generator over a single Polish Parquet file.
    Reads ONLY `text` (+ optional `quality_score`/`language_score` if present, skipping heavy
    columns like `embeddings` in 000_00000.parquet for maximum throughput and minimal RAM).
    """
    import pyarrow.parquet as pq

    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku Parquet: {p}")

    pf = pq.ParquetFile(p)
    schema_names = set(pf.schema_arrow.names)
    text_col = "text" if "text" in schema_names else next(
        (c for c in ("content", "sentence", "body", "article") if c in schema_names),
        pf.schema_arrow.names[0],
    )
    read_cols = [text_col]
    has_q = "quality_score" in schema_names
    has_lang = "language_score" in schema_names
    if has_q:
        read_cols.append("quality_score")
    if has_lang:
        read_cols.append("language_score")

    num_rgs = max(1, pf.num_row_groups)
    start_rg = start_row_group % num_rgs
    rg_order = list(range(start_rg, num_rgs)) + list(range(0, start_rg))

    yielded = 0
    cont_mod = max(2, int(round(1.0 / max(0.05, min(0.95, continuation_ratio)))))

    for rg_idx in rg_order:
        for batch in pf.iter_batches(batch_size=arrow_batch_size, row_groups=[rg_idx], columns=read_cols):
            text_vals = batch.column(0).to_pylist()
            q_vals = batch.column(read_cols.index("quality_score")).to_pylist() if has_q else None
            lang_vals = batch.column(read_cols.index("language_score")).to_pylist() if has_lang else None

            for row_i, raw_article in enumerate(text_vals):
                if not raw_article or not isinstance(raw_article, str):
                    continue
                if q_vals is not None:
                    qv = q_vals[row_i]
                    if qv is not None and float(qv) < min_quality_score:
                        continue
                if lang_vals is not None:
                    lv = lang_vals[row_i]
                    if lv is not None and float(lv) < min_language_score:
                        continue

                sents = [
                    s
                    for s in split_clean_sentences(raw_article[:3000], min_chars=min_chars, max_chars=max_chars)
                    if _is_grammatical_polish_sentence(s, min_chars=min_chars, max_chars=max_chars)
                ]
                if not sents:
                    continue
                for i, cur in enumerate(sents):
                    if i + 1 < len(sents) and (yielded % cont_mod == 0):
                        yield (cur, sents[i + 1])
                    else:
                        yield (cur, cur)
                    yielded += 1
                    if max_pairs is not None and yielded >= max_pairs:
                        return


def iter_parquet_polish_pairs(
    parquet_path: str | Path | list[str | Path],
    continuation_ratio: float = 0.35,
    min_chars: int = 32,
    max_chars: int = 230,
    arrow_batch_size: int = 128,
    max_pairs: int | None = None,
    start_row_group: int = 0,
    min_quality_score: float = 0.45,
    min_language_score: float = 0.85,
) -> Iterator[tuple[str, str]]:
    """
    Low-RAM streaming generator over one OR multiple Polish Parquet corpora (e.g. Wikipedia
    `train-00000-of-00006.parquet` + FineWeb Polish `000_00000.parquet` interleaved round-robin).
    Yields `(source_sentence, target_sentence)` pairs:
      - ~65% denoising autoencoding `(s, s)` -> teaches crisp Polish orthography, declension & syntax
      - ~35% adjacent sentence continuation `(s_i, s_{i+1})` -> teaches grammatical flow & coherent responses
    """
    if isinstance(parquet_path, (list, tuple)):
        paths = [Path(p) for p in parquet_path if Path(p).exists()]
        if not paths:
            raise FileNotFoundError("Nie znaleziono żadnych plików Parquet do treningu.")
        if len(paths) == 1:
            yield from _iter_single_parquet_pairs(
                paths[0],
                continuation_ratio=continuation_ratio,
                min_chars=min_chars,
                max_chars=max_chars,
                arrow_batch_size=arrow_batch_size,
                max_pairs=max_pairs,
                start_row_group=start_row_group,
                min_quality_score=min_quality_score,
                min_language_score=min_language_score,
            )
            return

        iters = [
            _iter_single_parquet_pairs(
                p,
                continuation_ratio=continuation_ratio,
                min_chars=min_chars,
                max_chars=max_chars,
                arrow_batch_size=arrow_batch_size,
                max_pairs=None,
                start_row_group=start_row_group,
                min_quality_score=min_quality_score,
                min_language_score=min_language_score,
            )
            for p in paths
        ]
        yielded = 0
        active = list(iters)
        while active:
            next_active: list[Iterator[tuple[str, str]]] = []
            for it in active:
                try:
                    pair = next(it)
                    yield pair
                    yielded += 1
                    if max_pairs is not None and yielded >= max_pairs:
                        return
                    next_active.append(it)
                except StopIteration:
                    continue
            active = next_active
        return

    yield from _iter_single_parquet_pairs(
        parquet_path,
        continuation_ratio=continuation_ratio,
        min_chars=min_chars,
        max_chars=max_chars,
        arrow_batch_size=arrow_batch_size,
        max_pairs=max_pairs,
        start_row_group=start_row_group,
        min_quality_score=min_quality_score,
        min_language_score=min_language_score,
    )


def iter_processed_corpus_pairs(
    base_dir: str | Path = ".",
    continuation_ratio: float = 0.35,
    min_chars: int = 32,
    max_chars: int = 230,
    max_pairs: int | None = None,
) -> Iterator[tuple[str, str]]:
    """Stream `(source, target)` sentence pairs from `data/corpora/processed/*.jsonl`."""
    from .corpus_fetcher import iter_processed_corpus_texts

    cont_mod = max(2, int(round(1.0 / max(0.05, min(0.95, continuation_ratio)))))
    yielded = 0
    for doc in iter_processed_corpus_texts(base_dir):
        sents = [
            s
            for s in split_clean_sentences(doc[:3000], min_chars=min_chars, max_chars=max_chars)
            if _is_grammatical_polish_sentence(s, min_chars=min_chars, max_chars=max_chars)
        ]
        if not sents:
            continue
        for i, cur in enumerate(sents):
            if i + 1 < len(sents) and (yielded % cont_mod == 0):
                yield (cur, sents[i + 1])
            else:
                yield (cur, cur)
            yielded += 1
            if max_pairs is not None and yielded >= max_pairs:
                return


def train_reasoning_with_bielik(
    brain: Any,
    num_samples: int = 90,
    train_epochs: int = 160,
    skip_generation: bool = False,
) -> dict[str, Any]:
    """Convenience entry point delegating 3-category Bielik reasoning training to `SleepingBrain`."""
    return brain.train_reasoning_with_bielik(
        num_samples=num_samples,
        train_epochs=train_epochs,
        skip_generation=skip_generation,
    )

