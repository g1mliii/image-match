@echo off
REM ============================================================================
REM CatalogMatch - One-Time Setup
REM ============================================================================

title CatalogMatch Setup

cd /d "%~dp0"

echo.
echo ================================================================================
echo                    CatalogMatch - Setup
echo ================================================================================
echo.
echo This will install all dependencies (takes ~2 minutes)
echo Internet connection required
echo.
pause

echo.
echo [INFO] Installing dependencies...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ================================================================================
    echo                         Setup Complete!
    echo ================================================================================
    echo.
    echo Next step: Double-click "RUN.bat" to start the app
    echo.
) else (
    echo.
    echo ================================================================================
    echo                         Setup Failed
    echo ================================================================================
    echo.
    echo Please check the error above and try again.
    echo.
)

pause
