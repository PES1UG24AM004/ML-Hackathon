@echo off
setlocal
cd /d "%~dp0"
echo ===============================================================================
echo            PathSnap - Map-First Map-Matching & Route Reconstruction
echo                                Application Launcher
echo ===============================================================================
echo.

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    .\.venv\Scripts\pip.exe install -r requirements.txt
)

:: Check if evaluator data is extracted
if not exist "evaluator_data\inputs" (
    if exist "%USERPROFILE%\Downloads\evaluator_test.zip" (
        echo [*] Auto-extracting evaluator dataset...
        .\.venv\Scripts\python.exe scripts\extract_evaluator_dataset.py
    )
)

echo [*] Starting PathSnap Web Application on http://localhost:8000...
start "" "http://localhost:8000"
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
