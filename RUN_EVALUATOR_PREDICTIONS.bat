@echo off
setlocal
cd /d "%~dp0"
echo ===============================================================================
echo      PathSnap - Evaluator Dataset Prediction Pipeline (Isolated Execution)
echo ===============================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment (.venv) not found. Please run RUN_PATHSNAP.bat first.
    pause
    exit /b 1
)

:: Step 1: Ensure dataset is extracted and validated
echo [*] Checking evaluator inputs...
if not exist "evaluator_data\inputs" (
    echo [*] Extracting evaluator dataset from Downloads...
    .\.venv\Scripts\python.exe scripts\extract_evaluator_dataset.py
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to extract evaluator dataset.
        pause
        exit /b 1
    )
)

:: Step 2: Run predictions
echo [*] Running inference across all 10 evaluator trips...
.\.venv\Scripts\python.exe scripts\run_evaluator_predictions.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Evaluator prediction pipeline failed.
    pause
    exit /b 1
)

echo.
echo [*] Predictions completed successfully.
echo [*] Outputs saved to: evaluator_outputs\predictions\
pause
exit /b 0
