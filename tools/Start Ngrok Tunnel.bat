@echo off
REM ============================================================================
REM CatalogMatch - Start ngrok Tunnel for Remote Mobile Access
REM ============================================================================

setlocal
cd /d "%~dp0\.."

set PORT=%~1
if "%PORT%"=="" set PORT=8000

where ngrok >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ngrok is not installed or not in PATH.
    echo Install from: https://ngrok.com/download
    echo Then run: ngrok config add-authtoken ^<YOUR_TOKEN^>
    echo.
    pause
    exit /b 1
)

echo.
echo =======================================================================
echo Starting ngrok tunnel for http://127.0.0.1:%PORT%
echo =======================================================================
echo.
echo Keep this window open while remote mobile access is needed.
echo After ngrok starts:
echo 1) In app click CONNECT PHONE
echo 2) Click AUTO NGROK
echo.

ngrok http http://127.0.0.1:%PORT%

endlocal
