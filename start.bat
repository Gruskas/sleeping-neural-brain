@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Uruchamianie Sleeping Neural Brain (PyTorch GPU)...

if exist ".venv\Scripts\python.exe" goto run

echo Nie znaleziono srodowiska .venv. Sprawdzanie Pythona...
where python >nul 2>&1
if errorlevel 1 (
    echo [BLAD] Nie znaleziono Pythona w PATH ani folderu .venv.
    echo Zainstaluj Python 3.10+ i uruchom ponownie start.bat.
    pause
    exit /b 1
)

echo Tworzenie wirtualnego srodowiska .venv...
python -m venv .venv
if errorlevel 1 (
    echo [BLAD] Nie udalo sie utworzyc .venv.
    pause
    exit /b 1
)

echo Instalacja PyTorch (CUDA 12.8) i zaleznosci...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install torch numpy --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo [UWAGA] Instalacja CUDA 12.8 nie powiodla sie, proba instalacji domyslnej z requirements.txt...
    ".venv\Scripts\pip.exe" install -r requirements.txt
) else (
    ".venv\Scripts\pip.exe" install -r requirements.txt
)
if errorlevel 1 (
    echo [BLAD] Nie udalo sie zainstalowac zaleznosci.
    pause
    exit /b 1
)

:run
".venv\Scripts\python.exe" run.py
pause
