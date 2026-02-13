@echo off
REM ============================================================================
REM Delete Old/Corrupted Snapshot Files
REM ============================================================================

title Deleting Old Snapshots...

cd /d "%~dp0backend\catalogs"

echo.
echo ================================================================================
echo                    Delete Old/Corrupted Snapshots
echo ================================================================================
echo.
echo This will delete ALL existing snapshot files in backend\catalogs\
echo.
echo Current snapshots:
dir /b *.db 2>nul

echo.
echo ================================================================================
echo.

set /p confirm="Delete all snapshots? (y/n): "

if /i "%confirm%" neq "y" (
    echo.
    echo Cancelled.
    pause
    exit /b
)

echo.
echo [INFO] Deleting snapshot databases...
del /q *.db 2>nul

echo [OK] All snapshot databases deleted
echo.
echo [INFO] Cleaning up old directories...
for /d %%d in (*) do (
    echo   - Deleting %%d
    rd /s /q "%%d" 2>nul
)

echo.
echo ================================================================================
echo                         Cleanup Complete!
echo ================================================================================
echo.
echo All old snapshots have been deleted.
echo.
echo Next steps:
echo 1. Run "python main.py" to start the app
echo 2. Upload your CSVs and run matching
echo 3. Save new snapshot (will work now!)
echo.

pause
