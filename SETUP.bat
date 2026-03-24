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
echo This will:
echo   1. Install Python dependencies
echo   2. Setup GPU acceleration (auto-detects AMD/NVIDIA/Intel/Apple)
echo   3. Create a desktop shortcut
echo.
echo Internet connection required (takes ~3-5 minutes)
echo.
pause

REM ============================================================================
REM Step 1: Install base dependencies
REM ============================================================================
echo.
echo [STEP 1/3] Installing base dependencies...
echo.
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ================================================================================
    echo              Step 1 Failed - Could not install dependencies
    echo ================================================================================
    echo.
    echo Please check:
    echo   - Python 3.12 is installed (NOT 3.13+)
    echo   - You have internet connection
    echo   - Run this as administrator if needed
    echo.
    pause
    exit /b 1
)

echo.
echo [STEP 1/3] Base dependencies installed successfully!
echo.

REM ============================================================================
REM Step 2: GPU Setup (auto-detect and install)
REM ============================================================================
echo.
echo [STEP 2/3] Setting up GPU acceleration...
echo.
echo Detecting GPU type (AMD/NVIDIA/Intel/Apple)...
echo.
python gpu/setup_gpu.py

if %errorlevel% equ 0 (
    echo.
    echo [STEP 2/3] GPU setup complete!
) else (
    echo.
    echo [STEP 2/3] GPU setup had issues - app will still work in CPU mode.
    echo            You can re-run "python gpu/setup_gpu.py" later to retry.
)

echo.

REM ============================================================================
REM Step 3: Create Desktop Shortcut
REM ============================================================================
echo.
echo [STEP 3/3] Creating desktop shortcut...
echo.

set "APP_DIR=%~dp0"
REM Remove trailing backslash
if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"
set "BAT_FILE=%APP_DIR%\RUN.bat"
set "DESKTOP=%USERPROFILE%\Desktop"

powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%DESKTOP%\CatalogMatch.lnk'); $Shortcut.TargetPath = '%BAT_FILE%'; $Shortcut.WorkingDirectory = '%APP_DIR%'; $Shortcut.IconLocation = 'imageres.dll,3'; $Shortcut.Description = 'CatalogMatch - Product Matching System'; $Shortcut.Save()"

if %errorlevel% equ 0 (
    echo [STEP 3/3] Desktop shortcut created: %DESKTOP%\CatalogMatch.lnk
) else (
    echo [STEP 3/3] Could not create shortcut automatically.
    echo            You can run "tools\Create Desktop Shortcut.bat" manually.
)

echo.
echo ================================================================================
echo                         Setup Complete!
echo ================================================================================
echo.
echo To start the app:
echo   - Double-click "CatalogMatch" on your desktop
echo   - OR double-click "RUN.bat" in this folder
echo.

pause
