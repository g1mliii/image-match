@echo off
REM ============================================================================
REM CatalogMatch - Launch App
REM ============================================================================

title CatalogMatch

cd /d "%~dp0"

echo.
echo ================================================================================
echo                    Starting CatalogMatch...
echo ================================================================================
echo.

python main.py

if %errorlevel% neq 0 (
    echo.
    echo ================================================================================
    echo                         Error Starting App
    echo ================================================================================
    echo.
    echo Common fixes:
    echo 1. Run SETUP.bat first to install dependencies
    echo 2. Make sure Python 3.12 is installed
    echo 3. Check if port 8000 is available
    echo.
    pause
)
