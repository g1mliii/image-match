@echo off
REM ============================================================================
REM Fix Database - Reinitialize Database Schema
REM ============================================================================

title Fixing Database...

cd /d "%~dp0"

echo.
echo ================================================================================
echo                    Fixing Database Schema
echo ================================================================================
echo.
echo This will reinitialize the database schema without deleting any data.
echo.

python -c "
import sys
sys.path.insert(0, 'backend')

from database import init_db
import os

print('[INFO] Initializing database schema...')
init_db()
print('[OK] Database schema initialized successfully!')
print()
print('[INFO] Verifying tables...')

import sqlite3
from path_manager import get_database_path

conn = sqlite3.connect(get_database_path())
cursor = conn.cursor()

# Check tables
cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\")
tables = [row[0] for row in cursor.fetchall()]

print(f'[OK] Found {len(tables)} tables:')
for table in tables:
    cursor.execute(f'SELECT COUNT(*) FROM {table}')
    count = cursor.fetchone()[0]
    print(f'  - {table}: {count} rows')

conn.close()

print()
print('[SUCCESS] Database is ready!')
"

if %errorlevel% equ 0 (
    echo.
    echo ================================================================================
    echo                         Database Fixed!
    echo ================================================================================
    echo.
    echo The database schema has been reinitialized.
    echo You can now run the app and save snapshots.
    echo.
) else (
    echo.
    echo ================================================================================
    echo                         Fix Failed
    echo ================================================================================
    echo.
    echo Could not fix the database. Please check the error above.
    echo.
)

pause
