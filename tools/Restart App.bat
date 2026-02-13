@echo off
REM ============================================================================
REM Restart App - Kill old processes and start fresh
REM ============================================================================

title Restarting CatalogMatch...

cd /d "%~dp0"

echo.
echo ================================================================================
echo                    Restarting CatalogMatch
echo ================================================================================
echo.

echo [1/3] Killing old Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/3] Cleaning up lock files...
del /q backend\*.applock >nul 2>&1

echo [3/3] Starting app...
echo.
echo ================================================================================
echo                         Server Starting
echo ================================================================================
echo.

python main.py

pause
