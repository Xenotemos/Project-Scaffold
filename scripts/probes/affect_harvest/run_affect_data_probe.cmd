@echo off
setlocal EnableDelayedExpansion

rem -----------------------------------------------------------------------------
rem  Affect data probe launcher
rem  - wraps scripts.probes.affect_data_probe
rem  - prompts for target interaction count and confirms before running
rem -----------------------------------------------------------------------------

set "SCRIPT_HOME=%~dp0.."
for %%I in ("%SCRIPT_HOME%") do set "SCRIPT_HOME=%%~fI"
for %%I in ("%SCRIPT_HOME%\..") do set "ROOT=%%~fI"
pushd "%ROOT%"

set "PYTHON=%ROOT%\.venv-win\Scripts\python.exe"
if not exist "%PYTHON%" (
    set "PYTHON=python"
)

set "PROBE_PROFILES=instruct base"
set "COOLDOWN_SECONDS=8"
set "TARGET_TURNS="
set "RESET_SESSION=true"
set "SCENARIO_FILE=%ROOT%\scripts\probes\scenarios\affect_scenarios.json"
set "RUN_ID="

echo.
echo  Affect Data Probe
echo  -----------------
echo     Profiles       : %PROBE_PROFILES%
echo     Cooldown (sec) : %COOLDOWN_SECONDS%
echo     Scenario file  : %SCENARIO_FILE%
echo.
set "USER_INPUT="
set /p "USER_INPUT=  Override profiles (space separated)? (Enter keeps default): "
if defined USER_INPUT (
    set "USER_INPUT=%USER_INPUT:"=%"
    if defined USER_INPUT (
        set "PROBE_PROFILES=%USER_INPUT%"
    )
)
:PROMPT_RUNID
echo.
set "RUN_ID="
set /p "RUN_ID=  Run ID for this harvest (e.g., run-001): "
set "RUN_ID=%RUN_ID:"=%"
if not defined RUN_ID (
    echo      Please enter a run id.
    goto PROMPT_RUNID
)
:PROMPT_TARGET
echo.
set "TARGET_TURNS="
set /p "TARGET_TURNS=  Enter exact number of turns for this run (e.g., 2000-4000): "
set "TARGET_TURNS=%TARGET_TURNS:"=%"
if not defined TARGET_TURNS (
    echo      Please enter a numeric value greater than zero.
    goto PROMPT_TARGET
)
set "NONNUM="
for /f "delims=0123456789" %%A in ("%TARGET_TURNS%") do set "NONNUM=%%A"
if defined NONNUM (
    echo      Digits only, please.
    set "NONNUM="
    goto PROMPT_TARGET
)
if "%TARGET_TURNS%"=="0" (
    echo      Value must be greater than zero.
    goto PROMPT_TARGET
)
for /f "delims=" %%A in ("%TARGET_TURNS%") do set "TARGET_TURNS=%%~A"
set "USER_INPUT="
set /p "USER_INPUT=  Override cooldown seconds? (Enter keeps %COOLDOWN_SECONDS%): "
if defined USER_INPUT (
    set "USER_INPUT=%USER_INPUT:"=%"
    if defined USER_INPUT (
        set "COOLDOWN_SECONDS=%USER_INPUT%"
    )
)

echo.
echo     Using profiles  : %PROBE_PROFILES%
echo     Run ID         : %RUN_ID%
echo     Target turns   : %TARGET_TURNS%
echo     Cooldown       : %COOLDOWN_SECONDS%s
echo.
set /p "CONFIRM=Press Enter to launch the affect data probe (Ctrl+C to abort)..."
echo.

set "TARGET_ARG=--target-turns %TARGET_TURNS%"
set "RUN_ARG=--run-id %RUN_ID%"

echo [run_affect_data_probe] Starting probe...
echo     "%PYTHON%" -m scripts.probes.affect_data_probe --profiles %PROBE_PROFILES% --cooldown-seconds %COOLDOWN_SECONDS% --scenario-file "%SCENARIO_FILE%" --run-id %RUN_ID% --reset-session %TARGET_ARG%
"%PYTHON%" -m scripts.probes.affect_data_probe --profiles %PROBE_PROFILES% --cooldown-seconds %COOLDOWN_SECONDS% --scenario-file "%SCENARIO_FILE%" --run-id %RUN_ID% --reset-session %TARGET_ARG%

echo.
echo [run_affect_data_probe] Completed. Raw rows in logs\probe_runs\%RUN_ID%\affect_data.jsonl
popd
endlocal
