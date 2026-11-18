@echo off
setlocal enabledelayedexpansion
title PathSnap Demonstration Launcher
cd /d "%~dp0"

echo ===============================================================================
echo                PathSnap - Map-Matching and Route Reconstruction
echo                              Interactive Demo Launcher
echo ===============================================================================
echo.

:: 1. Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.10+ and re-run.
    pause
    exit /b 1
)

:: 2. Check Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [*] Installing dependencies from requirements.txt...
    .\.venv\Scripts\pip.exe install --upgrade pip
    .\.venv\Scripts\pip.exe install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    echo [*] Using existing virtual environment: .venv
)

echo.
echo Select Demonstration Option:
echo   [1] Start Web UI and REST API server (FastAPI at http://localhost:8000)
echo   [2] Run HMM Sequence Reconstruction on Student Test Trips
echo   [3] Generate Visualizations and HTML Consolidated Report
echo   [4] Run Unit / Smoke Tests
echo   [5] Exit
echo.
set /p OPTION="Enter choice [1-5] (default=1): "
if "%OPTION%"=="" set OPTION=1

if "%OPTION%"=="1" goto launch_api
if "%OPTION%"=="2" goto launch_hmm
if "%OPTION%"=="3" goto launch_viz
if "%OPTION%"=="4" goto launch_tests
if "%OPTION%"=="5" goto exit_script

echo Invalid choice. Defaulting to Web UI and REST API...
goto launch_api

:launch_api
echo.
echo [*] Starting PathSnap FastAPI server on http://localhost:8000...
echo [*] Opening browser to Web UI...
start "" "http://localhost:8000"
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
goto end

:launch_hmm
echo.
echo [*] Running HMM Viterbi Route Reconstruction across student test trips...
.\.venv\Scripts\python.exe src\hmm_viterbi.py
echo.
echo [*] HMM evaluation completed. Results saved to results\hmm_vs_classifier.csv
pause
goto end

:launch_viz
echo.
echo [*] Generating figures and Folium maps...
.\.venv\Scripts\python.exe src\visualizations.py
echo [*] Compiling consolidated HTML and PDF reports...
.\.venv\Scripts\python.exe src\build_report.py
echo [*] Reports generated:
echo     - results\consolidated_report.html
echo     - results\consolidated_report.pdf
start "" "results\consolidated_report.html"
pause
goto end

:launch_tests
echo.
echo [*] Running unit and pipeline tests...
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
pause
goto end

:exit_script
echo Exiting.
exit /b 0

:end
