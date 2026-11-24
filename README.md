# 🧠 Sleeping Neural Brain — W 100% Samowystarczalny Mózg AI (PyTorch GPU)

W pełni autonomiczny, lokalny mózg AI działający **w 100% samodzielnie w czystym PyTorch + NumPy na GPU (CUDA)** — **bez Ollama, bez HuggingFace i bez zewnętrznych modeli LLM**.

## Architektura i najważniejsze funkcje

- **100% własne sieci neuronowe trenowane na GPU:**
  - **`SubwordTokenizer`**: bezstratny tokenizator bajtowo-słownikowy (UTF-8 + dynamicznie rosnący polski/techniczny słownik do 16 384 tokenów).
  - **`CorticalSemanticEncoder`**: 256-wymiarowy koder semantyczny łączący deterministyczne cechy n-gramowe/rdzeniowe z trenowaną kontrastowo (InfoNCE) siecią rezydualną na GPU.
  - **`CorticalTransformerLM`**: 6-warstwowy przyczynowy Transformer na GPU (`bfloat16`) trenowany podczas snu oraz komendą `/wyksztalc` na generowanych samodzielnie dialogach konwersacyjnych (`brain/language_trainer.py`).
  - **`AssociativeConceptCortex`**: grafowa sieć pojęć (do **16 384** pojęć) chroniona przez *Elastic Weight Consolidation* (EWC) oraz 2-skokowe marzenia **REM**.
- **Ludzki styl rozmowy, kontekst poprzednich wiadomości i pamięć epizodyczna**: mózg śledzi zaimki i dopytania z poprzednich wiadomości, pamięta wcześniej przeprowadzone rozmowy (`"co wcześniej pisaliśmy"`), zwraca się naturalnie w 1./2. osobie i wykonuje obliczenia szkolne (działania, procenty, potęgi, pierwiastki).
- **Pakiet „Szkoła Podstawowa” jednym kliknięciem (`/szkola` / przycisk w dashboardzie)**: ~140 podstawowych pojęć i ~90 relacji grafu z 9 przedmiotów (matematyka, przyroda, geografia, fizyka, chemia, historia, język polski, informatyka, wiedza codzienna) + opcjonalny kurs 30 tematów z Wikipedii (`/szkola siec`).
- **Wielowarstwowa pamięć:**
  - **Hipokamp** (krótkotrwała): do **5 000** wspomnień + **5 000** powiązań grafu (automatyczna ekstrakcja faktów z rozmowy + ręczne dodawanie i usuwanie).
  - **Kora Semantyczna** (trwała): do **50 000** wspomnień z dopasowaniem semantycznym na GPU i fleksyjnym dopasowaniem polskich odmian.
  - **Rdzeń Wiedzy NREM** oraz **Refleksje REM** (do **500**).
- **Autonomiczna nauka z sieci (`/ucz <temat|URL>` lub pasek w dashboardzie):**
  1. Obsługuje hasła z literówkami oraz bezpośrednie adresy URL (`https://...`).
  2. Równolegle pobiera źródła (Wikisłownik PL, Wikipedia PL/EN, DuckDuckGo) i wydobywa definicje.
  3. Ocenia zdania na GPU za pomocą kodera semantycznego i ekstrahuje fakty oraz relacje grafu.
  4. Ocenia poziom opanowania (0–100%), doszukuje brakujące podtematy i konsoliduje wiedzę we śnie na GPU.

## Uruchomienie

- **Dashboard w przeglądarce:** kliknij dwukrotnie `start.bat` (`http://127.0.0.1:8765`).
- **Terminal (CLI):**
  ```powershell
  .\.venv\Scripts\python.exe run.py --cli
  ```
- **Testy jednostkowe i integracyjne (GPU + HTTP/SSE):**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
  ```
- **Pełny test regresyjny (GPU + nauka z sieci + trening snu):**
  ```powershell
  .\.venv\Scripts\python.exe test_brain.py
  ```
