@echo off
setlocal EnableDelayedExpansion

rem -----------------------------------------------------------------------------
rem  Mid-span probe harness launcher
rem  - boots the FastAPI server with live telemetry
rem  - runs the configurable mid-span injector for ~45 minutes
rem  - logs probe metrics to logs\probe_runs\mid_span
rem -----------------------------------------------------------------------------

set "SCRIPT_HOME=%~dp0.."
for %%I in ("%SCRIPT_HOME%") do set "SCRIPT_HOME=%%~fI"
for %%I in ("%SCRIPT_HOME%\..") do set "ROOT=%%~fI"
pushd "%ROOT%"

rem ---- configuration knobs ---------------------------------------------------
set "SERVER_HOST=127.0.0.1"
set "SERVER_PORT=8000"

set "PROBE_PROFILES=instruct base"
set "MID_SPAN_MINUTES=45"
set "MID_SPAN_COOLDOWN=30"
set "MID_SPAN_TURNS=5"
set "MID_SPAN_LOG_DIR=logs\probe_runs\mid_span"

set "PYTHON=%ROOT%\.venv-win\Scripts\python.exe"
set "LOCK_FILE=logs\probe_runs\.probe.lock"

rem ---- summary prompt --------------------------------------------------------
echo.
echo  Mid-span probe plan
echo  -------------------
echo     Server host/port   : %SERVER_HOST%:%SERVER_PORT%
echo     Probe profiles     : %PROBE_PROFILES%
echo     Duration (minutes) : %MID_SPAN_MINUTES%
echo     Cooldown (seconds) : %MID_SPAN_COOLDOWN%
echo     Turns per profile  : %MID_SPAN_TURNS%
echo     Log directory      : %MID_SPAN_LOG_DIR%
echo.
set "TELEMETRY_CHOICE="
set /p "TELEMETRY_CHOICE=  Show live telemetry console? (Y/N, default N): "
if /I "%TELEMETRY_CHOICE%"=="Y" (
    set "LIVING_TELEMETRY_CONSOLE=1"
) else (
    set "LIVING_TELEMETRY_CONSOLE=0"
)
set "USER_DURATION="
set /p "USER_DURATION=  Override duration minutes (press Enter to keep %MID_SPAN_MINUTES%): "
if defined USER_DURATION (
    set "USER_DURATION=%USER_DURATION:"=%"
    for /f "usebackq tokens=* delims=" %%D in (`powershell -NoLogo -Command "$inputText = '%USER_DURATION%'; $match = [regex]::Match($inputText, '\\d+(\\.\\d+)?'); if ($match.Success) { $match.Value } else { '' }"`) do set "USER_DURATION=%%D"
    if defined USER_DURATION (
        set "MID_SPAN_MINUTES=%USER_DURATION%"
    ) else (
        echo     (Input ignored; keeping default duration.)
    )
)
echo     Using duration (minutes) : %MID_SPAN_MINUTES%
echo.
echo  Press any key to launch the server and begin the mid-span run...
pause >nul

rem ---- clean up stale lock ---------------------------------------------------
if exist "%LOCK_FILE%" (
    echo [run_mid_span] Removing stale lock file "%LOCK_FILE%"
    del "%LOCK_FILE%" >nul 2>&1
)

rem ---- start the server in a dedicated window --------------------------------
set "SERVER_CMD=set LIVING_AUTO_TELEMETRY=1 && set LIVING_TELEMETRY_CONSOLE=%LIVING_TELEMETRY_CONSOLE% && cd /d "%ROOT%" && "%PYTHON%" -m uvicorn main:app --host %SERVER_HOST% --port %SERVER_PORT%"
start "living-ai server" /min cmd /c "%SERVER_CMD%"
echo [run_mid_span] Server launching... waiting 10 seconds for warm-up.
timeout /t 10 /nobreak >nul

rem ---- run mid-span probe ----------------------------------------------------
echo [run_mid_span] Starting mid-span injector:
echo     "%PYTHON%" -m scripts.probes.mid_span_probes --duration-minutes %MID_SPAN_MINUTES% --cooldown-seconds %MID_SPAN_COOLDOWN% --interactions %MID_SPAN_TURNS% --log-dir "%MID_SPAN_LOG_DIR%" --profiles %PROBE_PROFILES% --lock-file "%LOCK_FILE%"
"%PYTHON%" -m scripts.probes.mid_span_probes --duration-minutes %MID_SPAN_MINUTES% --cooldown-seconds %MID_SPAN_COOLDOWN% --interactions %MID_SPAN_TURNS% --log-dir "%MID_SPAN_LOG_DIR%" --profiles %PROBE_PROFILES% --lock-file "%LOCK_FILE%"

rem ---- completion ------------------------------------------------------------
echo.
echo [run_mid_span] Mid-span probe complete. Server window remains open.
echo               Press any key after you shut it down (Ctrl+C in the server window).
pause >nul

popd
endlocal
