from __future__ import annotations

import re
from typing import Any


def clean_human_fact(raw: str) -> str:
    """
    Strip dictionary/SJP artifacts and turn raw encyclopedic entries into clean,
    natural Polish prose suitable for training and conversation.
    """
    s = (raw or "").strip()
    if not s:
        return ""

    # Drop SJP Scrabble / spelling-register spam entirely or trim before it
    sjp_cut = re.search(
        r"(?:dopuszczalne w grach|słownik ortograficzny:\s*rejestr|odmienność\s+tak\b|występowanie\s+słownik\b|x/twitter|koalar\s*<)",
        s,
        flags=re.I,
    )
    if sjp_cut:
        s = s[: sjp_cut.start()].strip()
    if len(s) < 12:
        return ""

    # Turn Wiktionary headers into natural Polish sentences
    wik_m = re.match(
        r"^Znaczenie słowa\s+„([^”]+)”:\s*(.+)$",
        s,
        flags=re.I,
    )
    if wik_m:
        word = wik_m.group(1).strip()
        rest = wik_m.group(2)
        rest = re.sub(r"rzeczownik,\s*rodzaj\s+(?:męskoosobowy|męski|żeński|nijaki)(?:\s+osobowy)?\s*", "", rest, flags=re.I)
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
    s = re.sub(r"\([a-ząćęłńóśźż]{1,6}\.\s*$", "", s, flags=re.I)  # broken trailing '(łac.'
    # Convert encyclopedic "X – Y" into natural spoken "X to Y"
    s = re.sub(r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9\s]{2,40}?)\s+[–—-]\s+([a-ząćęłńóśźż])", r"\1 to \2", s)
    s = re.sub(r"\s+", " ", s).strip(" ;,-")
    if s and s[-1] not in ".!?":
        s += "."
    return s


# Natural Polish conversational dialogues used to train CorticalTransformerLM
# Format: (mem_hint, ctx_hint, user_msg, bot_reply)
BASE_HUMAN_DIALOGUES: list[tuple[str, str, str, str]] = [
    ("", "", "cześć", "Cześć! Miło Cię widzieć. O czym chcesz dzisiaj pogadać?"),
    ("", "", "hej, jak się masz?", "Hej! U mnie wszystko w porządku, jestem gotowy do rozmowy. A co u Ciebie?"),
    ("", "", "dzień dobry", "Dzień dobry! W czym mogę Ci dzisiaj pomóc?"),
    ("", "", "dobry wieczór", "Dobry wieczór! Jak minął Ci dzień?"),
    ("", "", "o czym możemy porozmawiać?", "Możemy pogadać właściwie o wszystkim — od codziennych spraw, przez zadania ze szkoły, aż po programowanie albo tematy z internetu."),
    ("", "", "o czy moge z toba rozmiac", "Możemy rozmawiać o czym tylko chcesz: o nauce, matematyce, przyrodzie, historii czy Twoich projektach. Jeśli czegoś jeszcze nie znam, mogę sam doczytać w sieci."),
    ("", "", "co potrafisz?", "Potrafię normalnie rozmawiać po polsku, zapamiętywać to, o czym rozmawiamy, liczyć zadania szkolne i uczyć się nowych tematów z sieci."),
    ("", "", "jak masz na imię?", "Jeszcze nie nadałeś mi imienia — jak chciałbyś mnie nazwać?"),
    ("Mózg ma na imię Krzysztof.", "", "jak masz na imię?", "Nazwałeś mnie Krzysztof i tego się trzymam!"),
    ("Użytkownik ma na imię Kuba.", "", "jak mam na imię?", "Masz na imię Kuba — pamiętam!"),
    ("", "", "nazywam się Kuba", "Miło Cię poznać, Kuba! Zapamiętam to sobie."),
    ("", "", "twoje imię to Krzysztof", "Jasne, od teraz mów mi Krzysztof! Fajne imię."),
    ("", "", "mieszkam w Gliwicach", "Super, zapamiętam że mieszkasz w Gliwicach."),
    ("", "", "lubię programować w Pythonie", "Świetnie, zapamiętam że lubisz Pythona — chętnie o nim pogadam."),
    ("", "", "dzięki!", "Nie ma za co! Jakbyś chciał o coś jeszcze zapytać, śmiało pisz."),
    ("", "", "super", "Cieszę się! Co robimy dalej?"),
    ("", "", "dobranoc", "Dobranoc! Odezwij się, jak znów będziesz chciał porozmawiać."),
    ("", "Rozmawialiśmy o fotosyntezie.", "ale co?", "Chodziło mi o fotosyntezę, o której przed chwilą wspominałem. Chcesz, żebym wyjaśnił to prościej?"),
    ("", "Rozmawialiśmy o Układzie Słonecznym.", "powiedz coś więcej", "Jasne! W Układzie Słonecznym wokół Słońca krąży osiem planet — od najbliższego Merkurego aż po odległego Neptuna."),
    ("", "", "ale to nie prawda", "Rozumiem, dzięki za poprawkę! Napisz mi proszę, jak jest poprawnie, to od razu zaktualizuję swoją pamięć."),
    ("", "", "pamiętasz co wcześniej pisaliśmy?", "Pewnie, pamiętam nasz ostatni wątek i wiadomości, które wymieniliśmy. Przypomnieć Ci szczegóły?"),
    ("", "", "co wcześniej pisaliśmy?", "Rozmawialiśmy przed chwilą o tematach z Twoich poprzednich wiadomości — chętnie do nich wrócę."),
    ("", "", "kto to był Tamerlan?", "Szczerze mówiąc nie mam jeszcze w pamięci żadnych faktów o Tamerlanie. Mogę to dla Ciebie sprawdzić, jeśli wpiszesz `/ucz Tamerlan`."),
    ("", "", "ile to jest 7 * 8?", "7 razy 8 daje 56."),
    ("", "", "ile to 25% z 200?", "25% z 200 to dokładnie 50."),
    ("", "", "pierwiastek z 144", "Pierwiastek kwadratowy z 144 wynosi 12."),
]


def _humanize_statement(fact: str) -> str:
    s = clean_human_fact(fact)
    if not s:
        return ""
    return s


def build_training_dialogues(
    memories: list[str],
    episodic_memories: list[str],
    chat_turns: list[dict[str, str]],
    user_profile: list[str],
    max_total: int = 420,
) -> list[str]:
    """
    Build natural Polish multi-turn training sequences (<MEM> ... <CTX> ... <USER> ... <BOT> ...)
    for self-training `CorticalTransformerLM` on GPU without any external models.
    """
    sequences: list[str] = []

    def add_seq(mem: str, ctx: str, usr: str, bot: str) -> None:
        m_part = f"<MEM> {mem.strip()}" if mem.strip() else "<MEM> brak"
        c_part = f"<CTX> {ctx.strip()}" if ctx.strip() else "<CTX> początek rozmowy"
        sequences.append(f"{m_part} {c_part} <USER> {usr.strip()} <BOT> {bot.strip()}")

    for mem, ctx, usr, bot in BASE_HUMAN_DIALOGUES:
        add_seq(mem, ctx, usr, bot)

    # Personal profile dialogues
    for p in user_profile[:8]:
        cp = clean_human_fact(p)
        if not cp:
            continue
        m_name = re.search(r"Użytkownik ma na imię\s+([^\s.,!?;]+)", cp, flags=re.I)
        if m_name:
            nm = m_name.group(1)
            add_seq(cp, "", "cześć, pamiętasz jak mam na imię?", f"Cześć {nm}! Jasne, że pamiętam Twoje imię.")
        b_name = re.search(r"(?:Mózg ma nadane przez użytkownika imię|Twoje imi[eę] to)\s+([^\s.,!?;]+)", cp, flags=re.I)
        if b_name:
            bn = b_name.group(1).capitalize()
            add_seq(cp, "", "jak masz na imię?", f"Mam na imię {bn} — tak mnie nazwałeś.")

    # Knowledge fact -> natural human Q&A + contextual follow-up turns
    seen_facts: set[str] = set()
    for raw in memories:
        clean = _humanize_statement(raw)
        if not clean or len(clean) < 18:
            continue
        key = clean.lower()
        if key in seen_facts:
            continue
        seen_facts.add(key)

        # Extract candidate subject from the beginning of the sentence
        lower_body = clean[0].lower() + clean[1:] if len(clean) > 1 and not clean[:2].isupper() else clean
        m_subj = re.match(r"^([A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9+#\s„”-]{2,42}?)\s+(?:to|jest|są|oznacza|był|była|składa się|ma)\b", clean)
        if m_subj:
            subj = m_subj.group(1).strip(" „”:-")
            if 2 <= len(subj) <= 36:
                add_seq(clean, "", f"co to jest {subj}?", f"Najprościej mówiąc: {lower_body}")
                add_seq(clean, "", f"co wiesz o {subj}?", f"Z tego co pamiętam, {lower_body}")
                add_seq(clean, f"Rozmawialiśmy o: {subj}.", "powiedz coś więcej na ten temat", f"Jeśli chodzi o {subj}, to {lower_body}")
        else:
            words = re.findall(r"[A-ZŁŚŻŹĆŃa-ząćęłńóśźż0-9+#]{4,}", clean)
            if words:
                hint = " ".join(words[:2])
                add_seq(clean, "", f"wyjaśnij krótko {hint}", f"W skrócie: {lower_body}")

        if len(sequences) >= max_total - 40:
            break

    # Episodic conversation memory dialogues
    if episodic_memories:
        ep_clean = [clean_human_fact(e) for e in episodic_memories[-4:] if clean_human_fact(e)]
        if ep_clean:
            summary = " ".join(ep_clean[:2])
            add_seq(summary, "", "o czym wcześniej pisaliśmy?", f"Pamiętam z naszych rozmów: {summary}")
            add_seq(summary, "", "co wcześniej pisałem?", f"Wcześniej wspominałeś między innymi: {ep_clean[0]}")

    # Real user-assistant conversation turns from chat history
    for i in range(len(chat_turns) - 1):
        t_u = chat_turns[i]
        t_b = chat_turns[i + 1]
        if t_u.get("role") == "user" and t_b.get("role") == "assistant":
            u_txt = t_u.get("content", "").strip()
            b_txt = clean_human_fact(t_b.get("content", ""))
            if 4 <= len(u_txt) <= 140 and 10 <= len(b_txt) <= 220:
                prev_ctx = chat_turns[i - 1].get("content", "")[:90] if i > 0 else ""
                add_seq("", prev_ctx, u_txt, b_txt)
        if len(sequences) >= max_total:
            break

    return sequences[:max_total]
