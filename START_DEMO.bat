@echo off
setlocal
cd /d "%~dp0"
echo =============================================
echo   PathSnap - SRNs first-round demo launcher
echo =============================================
set "PYTHON_CMD=python"
python -c "import numpy,scipy,matplotlib,folium,fastapi,uvicorn" >nul 2>&1
if errorlevel 1 (
  echo Required packages not found in system Python.
  if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || goto :error
  )
  if not exist ".venv\installed.ok" (
    echo Installing Python dependencies. Internet is needed for this first install unless cached.
    .venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
    .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
    echo ok>.venv\installed.ok
  )
  set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
  echo Using system Python because the required packages are already installed.
)
%PYTHON_CMD% scripts\extract_dataset.py || goto :error
if not exist "outputs\eda\summary.json" (
  echo Generating actual dataset exploration outputs...
  %PYTHON_CMD% scripts\run_eda.py || goto :error
)
if not exist "outputs\student_test_baseline_summary.csv" (
  echo Running the baseline on all 10 student-testing trips...
  %PYTHON_CMD% scripts\summarize_student_baseline.py || goto :error
)
echo.
echo Open http://127.0.0.1:8000 in your browser.
echo Keep this terminal open to show live API inference logs.
echo Press Ctrl+C to stop the server.
echo.
%PYTHON_CMD% -m uvicorn pathsnap.api:app --app-dir src --host 127.0.0.1 --port 8000 --access-log
:error
if errorlevel 1 pause
endlocal
