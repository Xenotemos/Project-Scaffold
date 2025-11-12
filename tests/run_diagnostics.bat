@echo off
pushd "%~dp0\.."
call scripts\run_diagnostics.bat
set "code=%ERRORLEVEL%"
popd
exit /b %code%

