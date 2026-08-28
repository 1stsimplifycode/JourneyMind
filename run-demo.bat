@echo off
REM ===========================================================================
REM  JourneyMind - one-click demo launcher
REM
REM  Sets up whatever is missing, starts the server, waits until it is really
REM  answering, then opens the browser. Safe to run repeatedly.
REM
REM    run-demo.bat            start on port 8000
REM    run-demo.bat 8080       start on a different port
REM ===========================================================================
setlocal EnableExtensions EnableDelayedExpansion
pushd "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

set "VENV=.venv\Scripts\python.exe"
set "URL=http://localhost:%PORT%"

echo.
echo   JourneyMind
echo   A travel advisor that plans your whole trip - not just one ride.
echo   ---------------------------------------------------------------
echo.

REM --- 1. Python -----------------------------------------------------------
if not exist "%VENV%" (
    echo   [1/4] Creating the Python environment ^(first run only^)...
    where python >nul 2>&1
    if errorlevel 1 (
        echo.
        echo   ERROR: Python was not found on your PATH.
        echo          Install Python 3.12+ from https://python.org and re-run this file.
        echo.
        goto :fail
    )
    python -m venv .venv
    if errorlevel 1 goto :venvfail
    "%VENV%" -m pip install --upgrade pip --quiet
    echo         Installing dependencies...
    "%VENV%" -m pip install -r backend\requirements.txt --quiet
    if errorlevel 1 goto :depsfail
) else (
    echo   [1/4] Python environment found.
)

REM Make sure the environment is actually usable, not just present.
"%VENV%" -c "import fastapi, uvicorn, numpy" >nul 2>&1
if errorlevel 1 (
    echo         Repairing dependencies...
    "%VENV%" -m pip install -r backend\requirements.txt --quiet
    if errorlevel 1 goto :depsfail
)

REM --- 2. Frontend ---------------------------------------------------------
if exist "backend\app\static\index.html" (
    echo   [2/4] Web interface already built.
) else (
    echo   [2/4] Building the web interface ^(first run only^)...
    where npm >nul 2>&1
    if errorlevel 1 (
        echo.
        echo   WARNING: npm was not found, so the web interface cannot be built.
        echo            The API will still run at %URL%/api/docs
        echo            Install Node 20+ from https://nodejs.org for the full UI.
        echo.
    ) else (
        pushd frontend
        if not exist "node_modules" call npm install --silent
        call npm run build --silent
        popd
        if not exist "backend\app\static\index.html" (
            echo   WARNING: the frontend build did not produce index.html. Continuing API-only.
        )
    )
)

REM --- 3. Trained models + simulated mobility bundle -----------------------
if exist "models\gat_model.npz" (
    echo   [3/5] Travel-time model found.
) else (
    echo   [3/5] No travel-time weights - the service will fall back to the
    echo         historical-mean baseline and say so in every response.
    echo         Run: .venv\Scripts\python.exe scripts\train.py --all
)

if exist "models\reliability_model.npz" (
    echo   [4/5] Reliability model found.
) else (
    echo   [4/5] Building the simulated booking history and reliability model...
    "%VENV%" scripts\generate_mobility_data.py
    if errorlevel 1 goto :mobfail
    "%VENV%" scripts\train_reliability.py
    if errorlevel 1 (
        echo         WARNING: reliability training failed. The comparison view
        echo                  will use flat fallback rates and say so.
    )
)

REM --- 4. Free the port, then start ---------------------------------------
echo   [5/5] Starting the server on port %PORT% ...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:"LISTENING" ^| findstr /C:":%PORT% "') do (
    echo         Port %PORT% was busy - stopping process %%P
    taskkill /F /PID %%P >nul 2>&1
)

start "JourneyMind server (close this window to stop)" /D "%~dp0backend" ^
    "%~dp0%VENV%" -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%

REM --- wait until it actually answers, rather than guessing --------------
set "READY="
for /L %%i in (1,1,45) do (
    if not defined READY (
        curl.exe -s -o nul -m 3 "%URL%/health" >nul 2>&1
        if not errorlevel 1 set "READY=1"
        if not defined READY (
            <nul set /p "=."
            timeout /t 1 /nobreak >nul
        )
    )
)
echo.

if not defined READY (
    echo.
    echo   The server did not answer in time.
    echo   Look at the "JourneyMind server" window for the error, then try:
    echo       run-demo.bat 8080
    echo.
    goto :fail
)

echo.
echo   Ready.  Opening %URL%
echo.
echo   COMPARE    - what will this trip really cost, across every provider?
echo   PLAN       - the full multi-modal journey planner
echo   ENTERPRISE - the fleet dashboard ^(demo key: demo-analyst-key^)
echo.
echo   Ride-hailing fares, availability and cancellation rates are SIMULATED
echo   and labelled as such. No commercial provider API is contacted.
echo.
echo   API docs:  %URL%/api/docs
echo   To stop:   close the "JourneyMind server" window.
echo.
start "" "%URL%"
goto :done

:venvfail
echo.
echo   ERROR: could not create the Python environment ^(python -m venv .venv^).
goto :fail

:mobfail
echo.
echo   ERROR: could not build the simulated mobility bundle.
echo          Try manually:  .venv\Scripts\python.exe scripts\generate_mobility_data.py
goto :fail

:depsfail
echo.
echo   ERROR: dependency installation failed.
echo          Try manually:  .venv\Scripts\python.exe -m pip install -r backend\requirements.txt
goto :fail

:fail
popd
echo   Press any key to close...
pause >nul
endlocal
exit /b 1

:done
popd
endlocal
exit /b 0
