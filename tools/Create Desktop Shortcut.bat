@echo off
REM ============================================================================
REM Create Desktop Shortcut for CatalogMatch
REM ============================================================================

title Creating Desktop Shortcut...

echo.
echo ================================================================================
echo                    Creating Desktop Shortcut
echo ================================================================================
echo.

REM Get current directory
set "SCRIPT_DIR=%~dp0"
set "BAT_FILE=%SCRIPT_DIR%Launch CatalogMatch.bat"

REM Get desktop path
set "DESKTOP=%USERPROFILE%\Desktop"

REM Create shortcut using PowerShell
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%DESKTOP%\CatalogMatch.lnk'); $Shortcut.TargetPath = '%BAT_FILE%'; $Shortcut.WorkingDirectory = '%SCRIPT_DIR%'; $Shortcut.IconLocation = 'imageres.dll,3'; $Shortcut.Description = 'CatalogMatch - Product Matching System'; $Shortcut.Save()"

if %errorlevel% equ 0 (
    echo.
    echo ================================================================================
    echo                         Success!
    echo ================================================================================
    echo.
    echo Desktop shortcut created: %DESKTOP%\CatalogMatch.lnk
    echo.
    echo You can now double-click "CatalogMatch" on your desktop to launch the app!
    echo.
) else (
    echo.
    echo ================================================================================
    echo                         Failed
    echo ================================================================================
    echo.
    echo Could not create desktop shortcut.
    echo Please create it manually:
    echo 1. Right-click "Launch CatalogMatch.bat"
    echo 2. Send to ^> Desktop (create shortcut)
    echo.
)

pause
