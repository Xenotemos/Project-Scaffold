@echo off
setlocal EnableDelayedExpansion

rem -----------------------------------------------------------------------------
rem  Affect validation probe launcher
rem  - runs scripts.probes.affect_validation with convenient prompts
rem  - reuses the in-process FastAPI app (no external server needed)
rem  - ensures hormone tracing stays enabled for the session
rem -----------------------------------------------------------------------------

set "SCRIPT_HOME=%~dp0.."
for %%I in ("%SCRIPT_HOME%") do set "SCRIPT_HOME=%%~fI"
for %%I in ("%SCRIPT_HOME%\..") do set "ROOT=%%~fI"
pushd "%ROOT%"

set "PYTHON=%ROOT%\.venv-win\Scripts\python.exe"
if not exist "%PYTHON%" (
    set "PYTHON=python"
)

rem ---- defaults --------------------------------------------------------------
set "PROBE_PROFILES=instruct base"
set "DELTA_THRESHOLD=0.4"
set "LOG_PATH=logs\endocrine_turns.jsonl"
set "JSON_OUT=logs\probe_runs\affect_validation.json"
set "SCENARIO_INPUT="

echo.
echo  Affect validation plan
echo  ----------------------
echo     Profiles        : %PROBE_PROFILES%
echo     Delta threshold : %DELTA_THRESHOLD%
echo     Log path        : %LOG_PATH%
echo     JSON report     : %JSON_OUT%
echo     Scenarios       : all (default)
echo.
set "USER_INPUT="
set /p "USER_INPUT=  Override profiles (space separated)? (Enter keeps default): "
if defined USER_INPUT (
    set "USER_INPUT=%USER_INPUT:"=%"
    if defined USER_INPUT (
        set "PROBE_PROFILES=%USER_INPUT%"
    )
)
set "USER_INPUT="
set /p "USER_INPUT=  Override delta threshold? (Enter keeps %DELTA_THRESHOLD%): "
if defined USER_INPUT (
    set "USER_INPUT=%USER_INPUT:"=%"
    if defined USER_INPUT (
        set "DELTA_THRESHOLD=%USER_INPUT%"
    )
)
set "USER_INPUT="
set /p "USER_INPUT=  Override log path? (Enter keeps %LOG_PATH%): "
if defined USER_INPUT (
    set "USER_INPUT=%USER_INPUT:"=%"
    if defined USER_INPUT (
        set "LOG_PATH=%USER_INPUT%"
    )
)
set "USER_INPUT="
set /p "USER_INPUT=  Override JSON output path? (Enter keeps %JSON_OUT%): "
if defined USER_INPUT (
    set "USER_INPUT=%USER_INPUT:"=%"
    if defined USER_INPUT (
        set "JSON_OUT=%USER_INPUT%"
    )
)
set "USER_INPUT="
set /p "USER_INPUT=  Limit to specific scenarios (space/comma separated, Enter = all): "
if defined USER_INPUT (
    set "USER_INPUT=%USER_INPUT:"=%"
    set "SCENARIO_INPUT=!USER_INPUT:,= !"
)

echo.
echo     Using profiles        : %PROBE_PROFILES%
echo     Using delta threshold : %DELTA_THRESHOLD%
echo     Using log path        : %LOG_PATH%
echo     Using JSON output     : %JSON_OUT%
if defined SCENARIO_INPUT (
    echo     Scenarios          : %SCENARIO_INPUT%
) else (
    echo     Scenarios          : all defaults
)
echo.
echo  Press any key to launch the affect validation probe...
pause >nul

set "SCENARIO_ARGS="
if defined SCENARIO_INPUT (
    for %%S in (%SCENARIO_INPUT%) do (
        set "SCENARIO_ARGS=!SCENARIO_ARGS! --scenario %%~S"
    )
)

set "LIVING_HORMONE_TRACE=1"
echo [run_affect_validation] Running probe with hormone tracing forced on.
echo     "%PYTHON%" -m scripts.probes.affect_validation --profiles %PROBE_PROFILES% --delta-threshold %DELTA_THRESHOLD% --log-path "%LOG_PATH%" --json-out "%JSON_OUT%" %SCENARIO_ARGS%
"%PYTHON%" -m scripts.probes.affect_validation --profiles %PROBE_PROFILES% --delta-threshold %DELTA_THRESHOLD% --log-path "%LOG_PATH%" --json-out "%JSON_OUT%" %SCENARIO_ARGS%

echo.
echo [run_affect_validation] Completed. Summaries (if any) are in "%JSON_OUT%".
popd
endlocal
