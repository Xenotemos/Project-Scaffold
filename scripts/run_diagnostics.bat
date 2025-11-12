@echo off
pushd "%~dp0\.."
python scripts\diagnostics.py --format text --repair
set "code=%ERRORLEVEL%"
popd
echo.
pause
exit /b %code%

