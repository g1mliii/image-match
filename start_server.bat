@echo off
REM Startup script for Product Matching System
REM Automatically detects GPU and uses correct Python version

echo ========================================
echo Product Matching System - Starting...
echo ========================================
echo.

REM Check if Python 3.12 is available (required for AMD ROCm)
py -3.12 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python 3.12 detected - checking for AMD GPU...
    
    REM Check if AMD GPU is present
    py -3.12 -c "import torch; exit(0 if torch.cuda.is_available() and 'AMD' in torch.cuda.get_device_name(0).upper() else 1)" >nul 2>&1
    if %errorlevel% equ 0 (
        echo AMD GPU detected! Using Python 3.12 for ROCm support
        echo.
        cd backend
        py -3.12 app.py
        goto :end
    )
)

REM Check if Python 3.13 is available
py -3.13 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Using Python 3.13
    echo.
    cd backend
    py -3.13 app.py
    goto :end
)

REM Fall back to default Python
echo Using default Python
echo.
cd backend
python app.py

:end
pause
