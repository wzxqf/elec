@echo off
setlocal

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo run_all finished successfully.
) else (
    echo run_all failed with exit code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
