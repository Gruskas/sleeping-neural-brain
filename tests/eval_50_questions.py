from __future__ import annotations

from pathlib import Path
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.agent import SleepingBrain
from brain.language_cortex import LanguageCortex

# 50 diverse sequential Polish conversation turns with STRICT semantic keyword verification!
# Format: (question, category, expected_any_keywords_lowercase)
QUESTIONS_50: list[tuple[str, str, list[str]]] = [
    # 1-10: Powitania, parafrazy potoczne, reakcje i empatia
    ("siema", "dialog", ["siema", "cześć", "hej", "gadamy", "gotowe"]),
    ("co tam sluchac u ciebie", "dialog", ["super", "kora", "u ciebie", "słychać", "pogadam"]),
    ("jak sie dzis czujesz", "dialog", ["świetnie", "gotowość", "dzień", "dobrze"]),
    ("kim jestes", "dialog", ["korą", "mózgiem", "neuron", "utf-8", "pytorch"]),
    ("czy masz tokeny", "dialog", ["bez token", "bajty", "utf-8", "nie"]),
    ("a co jak tego nie zrobie", "dialog", ["wybór", "strasznego", "temat", "problem"]),
    ("aha", "dialog", ["rozumiem", "pytaj", "ciekawi", "jasne"]),
    ("nudze sie", "dialog", ["kosmos", "zagadk", "/ucz", "nudzi"]),
    ("opowiedz mi cos smiesznego", "dialog", ["komputer", "wirus", "procesor", "lekarz", "żart"]),
    ("stresuje sie", "dialog", ["oddech", "stres", "krok", "spokoj"]),
    # 11-20: Fizyka, kosmos i matematyka (w tym parafrazy!)
    ("czym jest czarna dziura", "science", ["grawitac", "horyzont", "światł", "czarna dziura"]),
    ("dlaczego niebo jest niebieskie", "science", ["rayleigh", "niebiesk", "rozprasz", "błękit"]),
    ("czemu zachod slonca jest czerwony", "science", ["atmosfer", "czerwon", "pomarańcz", "rozprosz"]),
    ("co to jest teoria wzglednosci", "science", ["einstein", "czas", "przestrze", "grawitac"]),
    ("co oznacza e=mc2", "science", ["mas", "energi", "światł", "równoważ"]),
    ("co to jest splatanie kwantowe", "science", ["korelacj", "cząst", "pomiar", "odległoś"]),
    ("co to jest entropia", "science", ["nieuporządkow", "stanów", "entropi", "maleje"]),
    ("co to sa liczby pierwsze", "science", ["naturaln", "dzielą", "przez 1", "2, 3, 5"]),
    ("czym sie rozni korelacja od przyczynowosci", "science", ["korelacja", "powoduje", "czynnik", "dowodzi"]),
    ("dlaczego lod unosi sie na wodzie", "science", ["gęstość", "krystalicz", "wodorow", "lód"]),
    # 21-30: Informatyka, AI, programowanie i bezpieczeństwo
    ("powiedz mi jak dziala komputer", "tech", ["procesor", "ram", "elektronicz", "dane"]),
    ("czym rozni sie cpu od gpu", "tech", ["rdzeni", "równoleg", "sekwencyj", "gpu"]),
    ("porownaj ram i dysk", "tech", ["ulotna", "ssd", "trwale", "ram"]),
    ("co to sa wskazniki w c++", "tech", ["adres", "pamięci", "wskaźnik", "c++"]),
    ("co to jest rekurencja", "tech", ["wywołuje", "samą siebie", "bazow", "rekurencj"]),
    ("czym sie rozni http od https", "tech", ["szyfrowan", "tls", "podsłuch", "https"]),
    ("jak dziala internet", "tech", ["tcp/ip", "pakiet", "dns", "router"]),
    ("co to jest transformer w ai", "tech", ["uwagi", "attention", "sekwencj", "transformer"]),
    ("co to jest overfitting", "tech", ["przeuczeni", "szum", "pamięć", "nowymi danymi"]),
    ("jak zaczac programowac", "tech", ["python", "pętli", "projekt", "język"]),
    # 31-38: Biologia, neuronauka, psychologia i pytania kreatywne
    ("co to jest fotosynteza", "bio", ["roślin", "światł", "glukoz", "tlen"]),
    ("jak dziala ludzki mozg", "bio", ["miliard", "neuron", "synaps", "impuls"]),
    ("co to jest hipokamp", "bio", ["wspomn", "pamięci", "śnie", "hipokamp"]),
    ("czym sie rozni wirus od bakterii", "bio", ["komórk", "antybiotyk", "gospodarz", "wirus"]),
    ("co to jest efekt dunninga-krugera", "bio", ["przeceni", "kompetencj", "wiedzy", "dunning"]),
    ("co sie stanie gdy polaczymy kota i psa", "bio", ["chromosom", "skrzyżować", "rodzin", "zaprzyjaź"]),
    ("dlaczego sen jest wazny dla mozgu", "bio", ["utrwala", "synaptycz", "wspomn", "oczyszcz"]),
    ("co to jest dna", "bio", ["helisa", "genetyczn", "kwasu", "dna"]),
    # 39-44: Filozofia, ekonomia, historia i porady
    ("jaki jest sens zycia", "human", ["filozofi", "relacj", "rozwój", "sens"]),
    ("co to jest stoicyzm", "human", ["spokoj", "wpływ", "reakcj", "stoicyzm"]),
    ("co to jest brzytwa ockhama", "human", ["najprostsz", "wyjaśnień", "ockham", "faktów"]),
    ("co to jest inflacja", "human", ["wzrost", "cen", "siły nabywczej", "pieniądz"]),
    ("dlaczego wybuchla druga wojna swiatowa", "human", ["1939", "polsk", "niemiec", "hitler"]),
    ("jak pokonac prokrastynacje", "human", ["próg", "mikro-krok", "dwie minuty", "zacznij"]),
    # 45-47: Arytmetyka symboliczna + pamięć osobista
    ("ile to 12 * 1543242332", "math", ["18 518 907 984"]),
    ("okej to nadaje ci imie: Krzysztof", "memory", ["krzysztof"]),
    ("jak masz na imie", "memory", ["krzysztof"]),
    # 48-50: Auto-sieć (nowy fakt z Wikipedii!) + powrót do starej wiedzy + czysty OOD
    ("co to jest mitochondrium", "web", ["organell", "komórk", "atp", "energi", "mitochondri"]),
    ("co to jest kot", "recall_after_web", ["kot", "ssak", "drapieżn", "kotowat"]),
    ("xyzqwe 99999 blabla", "ood", ["nie wiem", "nie kojarzę", "nie mam pojęcia", "nie znam"]),
]


def main() -> int:
    t0 = time.perf_counter()
    brain = SleepingBrain(base_dir=".", load_language_mode="sync")
    # Rebuild paraphrase attractor bank cleanly with the latest expand_question_paraphrases()
    bank_n = brain.language_cortex.ensure_familiarity_bank(force_rebuild=True)
    brain.save_state()  # persist updated paraphrase attractor bank (zero weight change!)
    brain.save_state = lambda: None  # keep benchmark turns read-only on disk
    brain.auto_sleep = False
    brain.auto_web = True
    print(f"PARAPHRASE ATTRACTOR BANK READY: {bank_n} SDRs", flush=True)

    passed = 0
    failed_items: list[tuple[int, str, str, str]] = []

    for idx, (q, cat, expected_kws) in enumerate(QUESTIONS_50, start=1):
        res = brain.chat(q)
        ans = (res.get("reply") or "").strip()
        low_ans = ans.lower()
        qual = LanguageCortex.score_polish_quality(ans, creativity_level=brain.creativity_level)
        ok = True
        reason = ""

        if len(ans) < 8:
            ok = False
            reason = "za krótka odpowiedź"
        elif cat != "math" and qual < 0.65:
            ok = False
            reason = f"niska jakość języka ({qual:.2f})"
        elif expected_kws and not any(kw in low_ans for kw in expected_kws):
            ok = False
            reason = f"nie na temat (brak słów kluczowych: {expected_kws[:3]})"

        status = "OK" if ok else f"FAIL({reason})"
        if ok:
            passed += 1
        else:
            failed_items.append((idx, q, ans, reason))

        print(f"[{idx:02d}/50] ({cat:16s}) {status}\n  Q: {q}\n  A: {ans}", flush=True)

    elapsed = time.perf_counter() - t0
    print("=" * 72)
    print(f"WYNIK 50-PYTANIOWEGO MARATONU: {passed}/50 ({100.0*passed/50:.1f}%) w {elapsed:.1f}s")
    if failed_items:
        print("NIEUDANE:")
        for idx, q, ans, rsn in failed_items:
            print(f"  #{idx}: {q!r} -> {ans!r} [{rsn}]")
    return 0 if passed == 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
