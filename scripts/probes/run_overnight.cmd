@echo off
setlocal EnableDelayedExpansion

rem -----------------------------------------------------------------------------
rem  Living AI overnight probe runner
rem  - launches the FastAPI server (with live telemetry window)
rem  - runs continuous probes with safety guards and per-batch logging
rem  - resets session state between probe cycles
rem -----------------------------------------------------------------------------

set "SCRIPT_HOME=%~dp0.."
for %%I in ("%SCRIPT_HOME%") do set "SCRIPT_HOME=%%~fI"
for %%I in ("%SCRIPT_HOME%\..") do set "ROOT=%%~fI"
pushd "%ROOT%"

rem ---- configuration knobs ---------------------------------------------------
set "SERVER_HOST=127.0.0.1"
set "SERVER_PORT=8000"

set "PROBE_PROFILES=instruct base"
set "PROBE_INTERVAL=120"
set "PROBE_IDLE=90"
set "PROBE_MAX_HOURS=8"
set "PROBE_UNTIL=05:30"
set "PROBE_LOG_DIR=logs\probe_runs\overnight"

set "PYTHON=%ROOT%\.venv-win\Scripts\python.exe"
set "LOCK_FILE=%PROBE_LOG_DIR%\.probe.lock"

rem ---- summary prompt --------------------------------------------------------
echo.
echo  Overnight probe plan
echo  --------------------
echo     Server host/port   : %SERVER_HOST%:%SERVER_PORT%
echo     Probe profiles     : %PROBE_PROFILES%
echo     Interval (sec)     : %PROBE_INTERVAL%
echo     Idle cool-down     : %PROBE_IDLE%
echo     Max hours cap      : %PROBE_MAX_HOURS%
echo     Stop by            : %PROBE_UNTIL%
echo     Log directory      : %PROBE_LOG_DIR%
echo.
set "TELEMETRY_CHOICE="
set /p "TELEMETRY_CHOICE=  Show live telemetry console? (Y/N, default N): "
if /I "%TELEMETRY_CHOICE%"=="Y" (
    set "LIVING_TELEMETRY_CONSOLE=1"
) else (
    set "LIVING_TELEMETRY_CONSOLE=0"
)
echo  Press any key to launch the server and begin the probe run...
pause >nul

rem ---- clean up stale lock ---------------------------------------------------
if exist "%LOCK_FILE%" (
    echo [run_overnight] Removing stale lock file "%LOCK_FILE%"
    del "%LOCK_FILE%" >nul 2>&1
)

rem ---- start the server in a dedicated window --------------------------------
set "SERVER_CMD=set LIVING_AUTO_TELEMETRY=1 && set LIVING_TELEMETRY_CONSOLE=%LIVING_TELEMETRY_CONSOLE% && cd /d ""%ROOT%"" && ""%PYTHON%"" -m uvicorn main:app --host %SERVER_HOST% --port %SERVER_PORT%"
start "living-ai server" /min cmd /c "%SERVER_CMD%"
echo [run_overnight] Server launching... waiting 15 seconds for warm-up.
timeout /t 15 /nobreak >nul

rem ---- build optional flags --------------------------------------------------
set "CMD_ARGS=--profiles %PROBE_PROFILES% --interval %PROBE_INTERVAL% --idle-seconds %PROBE_IDLE% --reset-session"
if not "%PROBE_MAX_HOURS%"=="0" (
    set "CMD_ARGS=!CMD_ARGS! --max-hours %PROBE_MAX_HOURS%"
)
if not "%PROBE_UNTIL%"=="" (
    set "CMD_ARGS=!CMD_ARGS! --until %PROBE_UNTIL%"
)

rem ---- run probes ------------------------------------------------------------
echo [run_overnight] Starting probe injector:
echo     "%PYTHON%" -m scripts.continuous_probes --log-dir "%PROBE_LOG_DIR%" !CMD_ARGS!
"%PYTHON%" -m scripts.continuous_probes --log-dir "%PROBE_LOG_DIR%" !CMD_ARGS!

rem ---- completion ------------------------------------------------------------
echo.
echo [run_overnight] Probe run complete. Server is still running in the other window.
echo                Press any key after you shut it down (Ctrl+C in the server window).
pause >nul

popd
endlocal
