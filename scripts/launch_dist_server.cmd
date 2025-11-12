@echo off
setlocal

rem ------------------------------------------------------------------
rem  Launch the packaged Living AI server with configurable telemetry
rem  and llama log windows.
rem ------------------------------------------------------------------

set "SCRIPT_HOME=%~dp0"
for %%I in ("%SCRIPT_HOME%..") do set "ROOT=%%~fI"
set "DIST_DIR=%ROOT%\dist"

if not exist "%DIST_DIR%\living_ai_boot_attached.exe" (
    echo [launch_dist_server] Could not find living_ai_boot_attached.exe under %DIST_DIR%
    exit /b 1
)

echo.
echo  Dist Server Launcher
echo  --------------------
echo     Executable path : %DIST_DIR%
echo.
set "TELEMETRY_CHOICE="
set /p "TELEMETRY_CHOICE=  Show live telemetry console? (Y/N, default Y): "
if /I "%TELEMETRY_CHOICE%"=="N" (
    set "LIVING_TELEMETRY_CONSOLE=0"
) else (
    set "LIVING_TELEMETRY_CONSOLE=1"
)

set "LLAMA_CHOICE="
set /p "LLAMA_CHOICE=  Show llama log tail window? (Y/N, default N): "
if /I "%LLAMA_CHOICE%"=="Y" (
    set "LLAMA_LOG_WINDOW=1"
) else (
    set "LLAMA_LOG_WINDOW=0"
)

set "EXE_NAME=living_ai_boot_attached.exe"
set /p "EXE_SELECTION=  Launch attached (A) or detached (D)? [default A]: "
if /I "%EXE_SELECTION%"=="D" (
    set "EXE_NAME=living_ai_boot_detached.exe"
)

set "TARGET_EXE=%DIST_DIR%\%EXE_NAME%"

echo.
echo  Launching %EXE_NAME% with:
echo     LIVING_TELEMETRY_CONSOLE=%LIVING_TELEMETRY_CONSOLE%
echo     LLAMA_LOG_WINDOW=%LLAMA_LOG_WINDOW%
echo.

set "LIVING_TELEMETRY_CONSOLE=%LIVING_TELEMETRY_CONSOLE%"
set "LLAMA_LOG_WINDOW=%LLAMA_LOG_WINDOW%"
pushd "%DIST_DIR%"
start "living-ai server" "%EXE_NAME%"
popd

endlocal
