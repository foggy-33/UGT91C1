@echo off
setlocal

REM ============================================================
REM  UGT91C1 mutant activity prediction platform - LAN launcher
REM  Double-click this file on the shared lab PC, then tell lab
REM  members the address printed below.
REM  NOTE: keep this file pure ASCII -- Chinese text in a .bat file
REM  corrupts the cmd parser and the script will not run.
REM  The first run downloads the ESM-2 weights (~2.5 GB) into
REM  models\torch_cache; later runs work offline.
REM ============================================================

cd /d "%~dp0"

REM ---- 1. Activate the Python environment: conda env ugt91c1, else .venv ----
call conda activate ugt91c1 2>nul
if errorlevel 1 (
    if exist ".venv\Scripts\activate.bat" (
        call ".venv\Scripts\activate.bat"
    ) else (
        echo [ERROR] No usable Python environment found.
        echo         Create conda env "ugt91c1", or a .venv in this folder.
        echo         See README section 17 for the exact steps.
        pause
        exit /b 1
    )
)

REM ---- 2. Dependency self-check ----
python -c "import streamlit, torch, xgboost, esm" 2>nul
if errorlevel 1 (
    echo [ERROR] Missing dependencies in the active environment.
    echo         Run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM ---- 3. Pin the ESM weight cache inside this project (download once) ----
set "TORCH_HOME=%CD%\models\torch_cache"

REM ---- 4. Environment info: shows whether CUDA is picked up ----
python -c "import torch;print('[ENV] torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

REM ---- 5. Print LAN addresses for lab members ----
echo.
echo [ADDRESS] Lab members can open any of these in a browser (port 8501):
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do echo            http://%%b:8501
)
echo            http://localhost:8501  ^(this machine only^)
echo.

REM ---- 6. Start the service on the LAN with the file watcher disabled ----
python -m streamlit run streamlit_app.py ^
    --server.address 0.0.0.0 ^
    --server.port 8501 ^
    --server.headless true ^
    --server.fileWatcherType none

echo.
echo [NOTE] The service has stopped. If it failed, read the error above.
pause
