# Jak uruchomić Sleeping Neural Brain

Instrukcja instalacji środowiska oraz uruchamiania projektu w trybie dashboardu webowego i terminala (CLI).

## 1. Wymagania

- **Python** 3.10+ (zalecane środowisko wirtualne `.venv`)
- **GPU NVIDIA z obsługą CUDA** (zalecane — projekt korzysta z PyTorch na GPU; dla nowszych kart np. RTX 50xx wymagany jest PyTorch z CUDA 12.8)

## 2. Instalacja

Sklonuj repozytorium i przejdź do katalogu projektu:

```powershell
git clone https://github.com/Gruskas/sleeping-neural-brain.git
cd sleeping-neural-brain
```

Utwórz środowisko wirtualne `.venv` i zainstaluj zależności:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Zainstaluj PyTorch z obsługą CUDA (przykład dla CUDA 12.8 z `requirements.txt`) oraz pozostałe pakiety:

```powershell
pip install torch numpy --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 3. Uruchomienie

### Tryb dashboardu w przeglądarce (zalecany)

Na Windowsie możesz kliknąć dwukrotnie plik `start.bat` albo uruchomić ręcznie:

```powershell
.\.venv\Scripts\python.exe run.py
```

Domyślnie serwer startuje pod adresem `http://127.0.0.1:8765` i automatycznie otwiera przeglądarkę.

Dostępne flagi:
- `--port <numer>` — zmiana portu serwera (domyślnie `8765`)
- `--no-browser` — uruchomienie serwera bez automatycznego otwierania przeglądarki

### Tryb terminala (CLI)

```powershell
.\.venv\Scripts\python.exe run.py --cli
```

Po uruchomieniu możesz rozmawiać po polsku lub używać komend specjalnych:
- `/ucz <temat|URL>` — autonomiczna nauka z sieci (Wikisłownik, Wikipedia, DuckDuckGo lub podany link) wraz z konsolidacją we śnie
- `/sen` — ręczne uśpienie mózgu (NREM + trening sieci PyTorch na GPU + REM)
- `/pamiec` — podgląd hipokampu, kory semantycznej i refleksji REM
- `/rdzen` — wyświetlenie skondensowanego rdzenia wiedzy NREM
- `/dodaj <fakt>` — ręczne dodanie faktu do hipokampu
- `/eksport` — eksport bazy wiedzy do pliku `brain_knowledge_export.json`
- `/status` — stan energii, dnia, GPU, parametrów sieci i limitów pamięci
- `/reset` — wyczyszczenie stanu mózgu od zera
- `/quit` — zapis stanu i wyjście

## 4. Testy

Testy jednostkowe i integracyjne (GPU + HTTP/SSE):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Pełny test regresyjny (GPU + nauka z sieci + trening snu):

```powershell
.\.venv\Scripts\python.exe test_brain.py
```
