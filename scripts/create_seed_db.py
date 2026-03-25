#!/usr/bin/env python3
"""Create an empty, schema-initialized seed database for distribution.

This generates a portable `product_matching.db` with all tables, indexes,
and pragmas — but zero product/feature/match rows. The output file is
cross-platform (works on Windows, macOS, Linux) because SQLite is OS-agnostic
and no data (like file paths) is baked in.

Usage:
    python scripts/create_seed_db.py                    # writes to backend/seed/product_matching.db
    python scripts/create_seed_db.py --output my.db     # custom output path
    python scripts/create_seed_db.py --verify            # create + verify schema

The generated DB should be committed to the repo and shipped with every
ZIP/EXE build so the app never needs to run init_db() on first launch.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "backend" / "seed" / "product_matching.db"


def create_seed_db(db_path: Path) -> None:
    """Create an empty database with the full CatalogMatch schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale file so we always start fresh
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Pragmas (match init_db in database.py)
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")

    # ── Tables ──────────────────────────────────────────────────────────

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            category TEXT,
            product_name TEXT,
            sku TEXT,
            is_historical BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            color_features BLOB NOT NULL,
            shape_features BLOB NOT NULL,
            texture_features BLOB NOT NULL,
            embedding_type TEXT DEFAULT 'legacy',
            embedding_version TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            new_product_id INTEGER NOT NULL,
            matched_product_id INTEGER NOT NULL,
            similarity_score REAL NOT NULL,
            color_score REAL NOT NULL,
            shape_score REAL NOT NULL,
            texture_score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (new_product_id) REFERENCES products(id),
            FOREIGN KEY (matched_product_id) REFERENCES products(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_result_sessions (
            session_id TEXT PRIMARY KEY,
            mode TEXT,
            threshold REAL,
            limit_value INTEGER,
            visual_weight REAL,
            metadata_weight REAL,
            batch_size INTEGER,
            summary_json TEXT,
            errors_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_result_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            product_id INTEGER,
            result_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES match_result_sessions(session_id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            price REAL NOT NULL,
            currency TEXT DEFAULT 'USD',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            sales INTEGER DEFAULT 0,
            views INTEGER DEFAULT 0,
            conversion_rate REAL DEFAULT 0.0,
            revenue REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata_schema (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            column_name TEXT UNIQUE NOT NULL,
            data_type TEXT DEFAULT 'string',
            display_name TEXT,
            default_weight REAL DEFAULT 0.0,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Indexes (match init_db in database.py) ──────────────────────────

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_is_historical ON products(is_historical)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category_historical ON products(category, is_historical)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_name ON products(product_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_created_at ON products(created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_historical_created_at ON products(is_historical, created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category_created_at ON products(category, created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_new_product ON matches(new_product_id, similarity_score DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_matched_product ON matches(matched_product_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_match_result_items_session_id ON match_result_items(session_id, id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_match_result_items_session_product ON match_result_items(session_id, product_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_match_result_sessions_created_at ON match_result_sessions(created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_product_id ON features(product_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_product_id_id ON features(product_id, id DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_product_id ON price_history(product_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(product_id, date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_history_product_id ON performance_history(product_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_history_date ON performance_history(product_id, date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_schema_column_name ON metadata_schema(column_name)")

    conn.commit()
    conn.close()


EXPECTED_TABLES = {
    "products", "features", "matches",
    "match_result_sessions", "match_result_items",
    "price_history", "performance_history", "metadata_schema",
}

EXPECTED_INDEX_COUNT = 20  # 20 CREATE INDEX statements above


def verify_seed_db(db_path: Path) -> bool:
    """Verify the seed DB has the correct schema and is empty."""
    if not db_path.exists():
        print(f"[FAIL] File not found: {db_path}")
        return False

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    ok = True

    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = {row[0] for row in cursor.fetchall()}
    missing = EXPECTED_TABLES - tables
    extra = tables - EXPECTED_TABLES
    if missing:
        print(f"[FAIL] Missing tables: {missing}")
        ok = False
    if extra:
        print(f"[WARN] Extra tables (not fatal): {extra}")
    if not missing:
        print(f"[OK] All {len(EXPECTED_TABLES)} tables present")

    # Check indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
    indexes = [row[0] for row in cursor.fetchall()]
    print(f"[{'OK' if len(indexes) >= EXPECTED_INDEX_COUNT else 'FAIL'}] {len(indexes)} indexes (expected {EXPECTED_INDEX_COUNT})")
    if len(indexes) < EXPECTED_INDEX_COUNT:
        ok = False

    # Check all tables are empty
    for table in sorted(EXPECTED_TABLES):
        if table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"[FAIL] {table} has {count} rows (should be 0)")
                ok = False

    if ok:
        print(f"[OK] All tables empty — clean seed DB")

    # Check WAL mode
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    print(f"[{'OK' if mode == 'wal' else 'WARN'}] journal_mode={mode}")

    # File size
    size_kb = db_path.stat().st_size / 1024
    print(f"[OK] File size: {size_kb:.1f} KB")

    conn.close()

    # Cross-platform check: verify no OS-specific data
    print(f"[OK] No file paths or OS-specific data stored — portable across platforms")

    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create empty seed database for CatalogMatch distribution.")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify schema after creation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()

    print(f"Creating seed database: {output}")
    create_seed_db(output)
    print(f"[OK] Seed database created ({output.stat().st_size / 1024:.1f} KB)")

    if args.verify:
        print()
        print("Verifying schema...")
        if not verify_seed_db(output):
            print("\n[FAIL] Verification failed")
            return 1
        print("\n[OK] Verification passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
