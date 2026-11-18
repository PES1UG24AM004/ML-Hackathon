@echo off
setlocal
cd /d "%~dp0"
echo ===============================================================================
echo      PathSnap - Evaluator Scoring & Metric Compilation (Private Verification)
echo ===============================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment (.venv) not found.
    pause
    exit /b 1
)

:: Step 1: Verify predictions exist
if not exist "evaluator_outputs\predictions\prediction_summary.json" (
    echo [ERROR] Predictions not found! You must execute predictions first.
    echo [*] Running RUN_EVALUATOR_PREDICTIONS.bat now...
    call RUN_EVALUATOR_PREDICTIONS.bat
)

:: Step 2: Run separate scoring
echo [*] Scoring predictions against isolated private reference labels...
.\.venv\Scripts\python.exe scripts\score_evaluator_predictions.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Scoring pipeline failed.
    pause
    exit /b 1
)

echo.
echo [*] Evaluator scoring completed successfully.
echo [*] Summary report saved to: evaluator_outputs\reports\evaluator_final_report.md
echo [*] Metrics saved to: evaluator_outputs\metrics\
pause
exit /b 0
