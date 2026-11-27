# 🧠 Sleeping Neural Brain — Biologicznie Inspirowany Mózg AI (PyTorch GPU + CUDA)

Lokalny, eksperymentalny mózg kognitywny działający na GPU (CUDA) w PyTorch, łączący architekturę **bez tokenizera (surowe bajty UTF-8)** `Wernicke → Kora Hopfielda → Broca`, wielowarstwową pamięć hipokampową, sen **NREM/REM**, autonomiczne myślenie **DMN**, własne kernele **C++/CUDA** oraz opcjonalną destylację dialogów i rozumowania z lokalnego nauczyciela Bielik.

---

## Najważniejsze komponenty i funkcje

### 1. Architektura korowa bez tokenizera (`Wernicke → Hopfield Cortex → Broca`)
- **`WernickeByteEncoder`** (`brain/sensory_wernicke.py`): pracuje bezpośrednio na surowych bajtach UTF-8 (`vocab=260`, brak tokenizera BPE/subword), patche 4-bajtowe, RoPE, 6 warstw, 12 głowic, 16 kolumn korowych (`column_dim=768`).
- **Biologiczna Kora Skojarzeniowa** (`brain/neural_cortex.py`, `brain/fp8_synapse.py`):
  - `12 288` neuronów z rzadkim kodowaniem **k-WTA SDR** (`k=492`, ~4% aktywnych).
  - **Modern Continuous Hopfield** (`beta=12.0`, `4` kroki atraktora) + rekurencyjna macierz synaptyczna `W_rec` w natywnym **FP8 E4M3** (fallback `INT8`/`BF16`).
  - Lokalna plastyczność **Hebba / Oja**, **Predictive Coding** i homeostaza bez potrzeby każdorazowego pełnego retreningu.
- **`BrocaByteTransformer`** (`brain/broca_llm.py`, `brain/language_cortex.py`):
  - 10-warstwowy dekoder bajtowy z Cross-Attention wyłącznie na reprezentacje `Z*` / `Z_ctx` (limit odpowiedzi do `1024` bajtów).
  - Trzy tryby FFN (`broca_ffn_mode`):
    - `moe` *(domyślny)*: Sparse Mixture-of-Experts (`8` ekspertów, `top_k=2`).
    - `peer` (`brain/peer_memory.py`): Product-Key Micro-Experts z pamięcią podręczną Pinned RAM ↔ GPU VRAM (LRU).
    - `dense`: SwiGLU.
  - Wsparcie dla **CUDA Graphs** przy dekodowaniu oraz regularyzacja anty-overfitting (`label_smoothing`, `early_stop_loss`, replay korpusu Parquet).

### 2. Natywne przyspieszenie C++ / CUDA (`brain/csrc/`, `brain/cuda_ops.py`)
- Automatycznie kompilowane i cache'owane rozszerzenie `_brain_cuda_ext` (zoptymalizowane pod **RTX 5080 / Blackwell SM 12.0, CUDA 12.8**) z pełnym, automatycznym fallbackiem do czystego PyTorcha:
  - `fp8_synapse_kernels.cu`: sfuzowana kwantyzacja wierszowa FP8/INT8 (z zaokrąglaniem stochastycznym w rejestrach) oraz SHY pruning w miejscu.
  - `peer_kernels.cu` + `peer_cache.cpp`: kernele PEER i szybka obsługa cache na hoście.
  - **CUDA Graphs** w `BrocaByteTransformer`.

### 3. Living Brain, Sen NREM/REM i Autonomiczne Myślenie DMN (2-hop)
- **`LivingBrainEngine`** (`brain/living_brain.py`): wewnętrzna dynamika M1/M2/M4, neuromodulatory (**NA / DA / ACh**), poziom ciekawości oraz zegar działający w tle.
- **DMN — Default Mode Network** (`brain/agent.py`, `brain/hippocampus.py`): w trakcie bezczynności (domyślnie co `25 s`) lub ręcznie przyciskiem **„💭 Pomyśl teraz”** (`/api/dmn/think`) wyszukuje odległe mosty 2-hop (`A → B → C`) w pamięci trwałej i zapisuje nowe refleksje z podglądem na żywo w dashboardzie.
- **Sen NREM / REM** (`brain/sleep_engine.py`): konsolidacja pamięci przez powtórki NREM, osłabianie i przycinanie słabych synaps (**SHY**), scalanie duplikatów engramów oraz twórcze skojarzenia 2-hop w fazie REM.

### 4. Wielowarstwowa pamięć i rozmowa multi-turn
- **Pamięć robocza**: `64` SDR-y, `8` slotów PFC, okno rozmowy `128` wiadomości (`256` w buforze).
- **Hipokamp krótkotrwały**: do `50 000` engramów.
- **Kora Semantyczna (trwała na GPU)**: do `200 000` engramów + refleksje REM (do `2 000`).
- **System-2 i „słyszenie siebie”**: 3 równoległe ścieżki odpowiedzi (Precyzyjna, Kontekstowa, Kreatywna), weryfikacja szkicu przez Wernicke przed wysłaniem, wykrywanie *overconfident off-topic* / braku wiedzy (OOD) oraz ochrona small-talku przed przejęciem przez `auto_web`.

### 5. Pipeline „🚀 Auto-Trening” i Kurs Wiedzy i Łączenia Pojęć
- Konfigurowalny, wieloetapowy **Pipeline Auto-Treningu** (z presetami i bezpiecznym Stopem w trakcie oraz między etapami):
  1. **Korpusy Parquet**: gramatyka i polszczyzna (wczesne zatrzymanie na `target_loss`).
  2. **Bielik Dialogi** (`brain/teacher_distill.py`): generacja naturalnych dialogów w partiach ze zwalnianiem VRAM.
  3. **Bielik Rozumowanie**: relacje, logika i przyznawanie się do niewiedzy.
  4. **Wspólne dostrajanie `Wernicke + Broca`**: z replayem Parqueta przeciwko przeuczeniu i podglądem odpowiedzi na żywo w UI.
  5. **Kurs Wiedzy i Łączenia Pojęć** (`brain/curriculum.py`, `/api/study/course`): **60 modułów przekrojowych** uczących łączenia faktów między dziedzinami (fizyka, biologia, ekonomia, logika, technologia, psychologia itd.).
  6. **Sen NREM/REM**: konsolidacja świeżo nabytej wiedzy.
- **Autonomiczna nauka z sieci** (`brain/web_learner.py`): pobieranie haseł i URL-i (Wikipedia/Wikisłownik/DuckDuckGo), ekstrakcja faktów i utrwalanie w pamięci.

---

## Struktura projektu

- `brain/agent.py` — główna klasa `SleepingBrain`, pamięć robocza, czat, DMN, Auto-Trening.
- `brain/config.py` — parametry architektury, pamięci, treningu i CUDA.
- `brain/sensory_wernicke.py` — enkoder bajtowy UTF-8 (Wernicke).
- `brain/broca_llm.py` — dekoder bajtowy UTF-8 z MoE / PEER / CUDA Graphs (Broca).
- `brain/language_cortex.py` — wspólna pętla Wernicke ↔ Broca, System-2, samokrytyka i synteza 2-hop.
- `brain/neural_cortex.py`, `brain/fp8_synapse.py` — kora k-WTA SDR, Modern Hopfield i synapsy FP8/INT8.
- `brain/peer_memory.py` — pamięć PEER micro-experts z cache Pinned RAM ↔ VRAM.
- `brain/hippocampus.py` — banki engramów na GPU i wyszukiwanie mostów 2-hop.
- `brain/living_brain.py`, `brain/sleep_engine.py` — neuromodulacja, dynamika tła i cykl snu NREM/REM.
- `brain/curriculum.py`, `brain/teacher_distill.py`, `brain/web_learner.py` — kurs pojęć, destylacja Bielika i nauka z sieci.
- `brain/cuda_ops.py`, `brain/csrc/` — loader i źródła C++/CUDA.
- `server.py`, `static/index.html`, `run.py` — serwer HTTP/SSE, dashboard WWW oraz tryb CLI.
- `tests/` — testy jednostkowe i integracyjne.

---

## Uruchomienie i testy

- **Dashboard w przeglądarce:** kliknij dwukrotnie `start.bat` (`http://127.0.0.1:8765`) lub uruchom:
  ```powershell
  .\.venv\Scripts\python.exe run.py
  ```
- **Terminal (CLI):**
  ```powershell
  .\.venv\Scripts\python.exe run.py --cli
  ```
- **Wszystkie testy jednostkowe (`tests/test_*.py`):**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
  ```
- **Wybrane testy modułowe:**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest tests/test_cuda_kernels.py tests/test_chat_context_and_gating.py tests/test_dmn_and_course.py tests/test_training_pipeline.py -v
  ```
