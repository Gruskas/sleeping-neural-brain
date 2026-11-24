from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from pathlib import Path

from brain.agent import SleepingBrain
from server import BrainServer


def run_cli(brain: SleepingBrain) -> None:
    print(f"=== SLEEPING NEURAL BRAIN — CZAT + NAUKA Z SIECI ({brain.gpu_info()['device_name']}) ===")
    print("Ładowanie Kory Językowej na GPU...")
    if not brain.language_cortex.load_sync():
        print("BŁĄD ładowania modelu:", brain.language_cortex.load_error)
        return
    print("Gotowe! Pisz normalnie po polsku. Komendy specjalne:")
    print("  /ucz <temat|URL> -> mózg sam bada temat lub link w sieci aż go opanuje")
    print("  /sen             -> uśpij mózg (NREM + trening sieci PyTorch na GPU + REM)")
    print("  /pamiec          -> pokaż hipokamp, pamięć trwałą, refleksje REM")
    print("  /rdzen           -> pokaż skondensowany rdzeń wiedzy NREM")
    print("  /dodaj <fakt>    -> dodaj fakt bezpośrednio do hipokampu")
    print("  /eksport         -> zapisz bazę wiedzy do brain_knowledge_export.json")
    print("  /status          -> energia, dzień, GPU, parametry sieci, limity pamięci")
    print("  /reset           -> wyczyść mózg od zera")
    print("  /quit            -> zapisz i wyjdź")

    try:
        while True:
            try:
                raw = input(f"\n[Dzień {brain.day} | Energia {brain.energy:.0f}% | {brain.state_mode}] Ty: ").strip()
            except (EOFError, KeyboardInterrupt):
                brain.save_state()
                break
            if not raw:
                continue
            low = raw.lower()
            if low in ("/quit", "/exit"):
                brain.save_state()
                print("Stan mózgu został zapisany.")
                break
            if low == "/reset":
                brain.reset_brain()
                print("Zresetowano mózg od zera.")
                continue
            if low == "/status":
                snap = brain.snapshot()
                lc = snap["language_cortex"]
                print(
                    f"Sieć: {lc['model_id']} ({lc['total_params']/1e6:.2f}M param., słownik {lc['vocab_used']}/{lc['vocab_size']}) | "
                    f"GPU: {snap['gpu']['device_name']} (VRAM: {snap['gpu']['vram_allocated_mb']} MB)"
                )
                print(f"Dzień: {snap['day']} | Noce: {snap['total_sleeps']} | Treningi snu NREM: {lc['lora_updates']}")
                print(
                    f"Pamięć: Hipokamp {snap['memory']['short_term_memory_count']}/{snap['limits']['short_term_memories']} | "
                    f"Kora {snap['memory']['semantic_memory_count']}/{snap['limits']['semantic_memories']} | "
                    f"REM {snap['memory']['rem_reflection_count']}"
                )
                continue
            if low == "/rdzen":
                core = brain.language_cortex.core_knowledge_summary
                print("\n--- RDZEŃ WIEDZY NREM ---")
                print(core if core else "(Pusty — powstaje podczas snu po zgromadzeniu nowych wspomnień.)")
                continue
            if low.startswith("/dodaj "):
                fact_text = raw[7:].strip()
                if fact_text:
                    try:
                        mem = brain.add_manual_memory(fact_text)
                        print(f"  [Hipokamp +1]: {mem['text']} (id={mem['id']})")
                    except Exception as exc:
                        print(f"[BŁĄD] {exc}")
                continue
            if low == "/eksport":
                out_path = brain.base_dir / "brain_knowledge_export.json"
                out_path.write_text(
                    json.dumps(brain.export_knowledge(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"Wyeksportowano wiedzę do: {out_path}")
                continue
            if low == "/pamiec":
                snap = brain.snapshot()
                m = snap["memory"]
                print(f"\n--- HIPOKAMP ({m['short_term_memory_count']} wspomnień przed snem) ---")
                for x in m["short_term_memories"][-15:]:
                    print(f"  • [{x['category']}] {x['text']}")
                print(f"--- KORA SEMANTYCZNA ({m['semantic_memory_count']} trwałych) ---")
                for x in m["semantic_memories"][:20]:
                    print(f"  • [{x['category']}] {x['text']} (siła {x['strength']})")
                print(f"--- REFLEKSJE REM ({m['rem_reflection_count']}) ---")
                for r in m["rem_reflections"][-10:]:
                    print(f"  ✨ {r['text']}")
                continue
            if low.startswith("/ucz "):
                topic = raw[5:].strip()
                if topic:
                    print(f"\n[🌐 NAUKA Z SIECI] Badam temat '{topic}' aż go opanuję...")
                    res = brain.study_topic_online(topic, consolidate_after=True)
                    print(
                        f"[GOTOWE] Opanowanie: {res['mastery']:.0f}% | Strony: {res['pages_read']} | "
                        f"Fakty: {res['facts_learned']} | Noce: {res['sleeps']}"
                    )
                    if res.get("summary"):
                        print(f"Podsumowanie: {res['summary']}")
                continue

            try:
                streamed_any = False
                learned_memories: list[dict[str, object]] = []
                done_reply = ""
                for ev in brain.chat_stream(raw):
                    ev_type = ev.get("type")
                    if ev_type == "meta":
                        learned_memories = list(ev.get("learned_memories", []))
                        rec = ev.get("recalled_memories", {})
                        sem_n = len(rec.get("semantic", []))
                        hip_n = len(rec.get("hippocampus", []))
                        if sem_n or hip_n:
                            print(f"  [Pamięć: kora={sem_n}, hipokamp={hip_n}]")
                        print("\nMózg: ", end="", flush=True)
                    elif ev_type == "token":
                        print(ev.get("delta", ""), end="", flush=True)
                        streamed_any = True
                    elif ev_type == "done":
                        done_reply = str(ev.get("reply", ""))
                        if not streamed_any and done_reply:
                            print(f"\nMózg: {done_reply}", end="", flush=True)
                        learned_memories = list(ev.get("learned_memories", learned_memories))
                print()
                if learned_memories:
                    print("  [Hipokamp zapamiętał]:", " | ".join(str(m["text"]) for m in learned_memories))
            except Exception as exc:
                print(f"[BŁĄD] {exc}")
    finally:
        brain.language_cortex.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bio-inspired Sleeping Neural Brain (Chat + Web Learning + GPU LoRA)")
    parser.add_argument("--cli", action="store_true", help="Uruchom czat w terminalu (CLI)")
    parser.add_argument("--port", type=int, default=8765, help="Port lokalnego dashboardu (domyślnie 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Nie otwieraj automatycznie przeglądarki")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    brain = SleepingBrain(base_dir=base_dir, load_language_mode="none" if args.cli else "async")

    if args.cli:
        run_cli(brain)
        return

    server = BrainServer(brain=brain, static_dir=base_dir / "static", port=args.port)
    if not args.no_browser:
        threading.Timer(0.9, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    server.start()


if __name__ == "__main__":
    main()
