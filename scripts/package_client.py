#!/usr/bin/env python3
"""Create cross-platform CatalogMatch client ZIP bundles.

This script packages the repo contents used for local client testing into:
1. A timestamped archive, e.g. ``CatalogMatch_Client_20260324_143000.zip``
2. A latest archive, e.g. ``CatalogMatch_Client.zip``

It is OS-agnostic and avoids Finder/Explorer metadata such as ``__MACOSX``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


DEFAULT_BASENAME = "CatalogMatch_Client"


EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "release",
    "staging",
    "__MACOSX",
}

# Also exclude any directory whose name starts with ".venv" (e.g., .venv_test).
_EXCLUDED_DIR_PREFIXES = (".venv",)

EXCLUDED_DIR_PATHS = {
    "backend/uploads",
}

EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}

EXCLUDED_FILE_GLOBS = {
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.db-shm",
    "*.db-wal",
    "*.db-journal",
    "*.db.backup",
    "*.db.bak",
    "*.log",
    "CatalogMatch_Client.zip",
    "CatalogMatch_Client_*.zip",
}

# Database files that should NOT be shipped (user data / snapshots).
# The seed DB at backend/product_matching.db IS shipped.
EXCLUDED_FILE_PATHS = {
    "backend/product_matching.db.backup",
}


@dataclass(frozen=True)
class PackagePlan:
    source_root: Path
    output_dir: Path
    basename: str
    timestamped_zip: Path
    latest_zip: Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Create OS-agnostic client ZIP archives for local testing."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Project root to package. Defaults to the repo root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root,
        help="Directory where ZIP files should be written.",
    )
    parser.add_argument(
        "--basename",
        default=DEFAULT_BASENAME,
        help=f"Base archive name. Defaults to {DEFAULT_BASENAME}.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only write the non-timestamped latest ZIP.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be packaged without writing ZIP files.",
    )
    return parser.parse_args()


def build_plan(args: argparse.Namespace) -> PackagePlan:
    source_root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = args.basename

    return PackagePlan(
        source_root=source_root,
        output_dir=output_dir,
        basename=basename,
        timestamped_zip=output_dir / f"{basename}_{timestamp}.zip",
        latest_zip=output_dir / f"{basename}.zip",
    )


def normalized_rel_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def should_exclude_dir(rel_path: str, name: str) -> bool:
    if name in EXCLUDED_DIR_NAMES:
        return True
    if any(name.startswith(prefix) for prefix in _EXCLUDED_DIR_PREFIXES):
        return True
    return rel_path in EXCLUDED_DIR_PATHS


def should_exclude_file(rel_path: str, name: str) -> bool:
    if name in EXCLUDED_FILE_NAMES:
        return True
    if rel_path in EXCLUDED_FILE_PATHS:
        return True
    return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern) for pattern in EXCLUDED_FILE_GLOBS)


def iter_files(root: Path) -> Iterable[Path]:
    for current_root, dir_names, file_names in os.walk(root):
        current_path = Path(current_root)

        kept_dirs: list[str] = []
        for dir_name in sorted(dir_names):
            dir_path = current_path / dir_name
            rel_path = normalized_rel_path(dir_path, root)
            if not should_exclude_dir(rel_path, dir_name):
                kept_dirs.append(dir_name)
        dir_names[:] = kept_dirs

        for file_name in sorted(file_names):
            file_path = current_path / file_name
            rel_path = normalized_rel_path(file_path, root)
            if should_exclude_file(rel_path, file_name):
                continue
            yield file_path


def build_zip(source_root: Path, zip_path: Path, files: list[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for file_path in files:
            rel_path = normalized_rel_path(file_path, source_root)
            zip_info = ZipInfo.from_file(file_path, arcname=rel_path)
            zip_info.compress_type = ZIP_DEFLATED

            # Preserve Unix executable bits for shell scripts on macOS/Linux.
            mode = stat.S_IMODE(file_path.stat().st_mode)
            zip_info.external_attr = (mode & 0xFFFF) << 16

            with file_path.open("rb") as handle:
                archive.writestr(zip_info, handle.read())


def print_summary(plan: PackagePlan, files: list[Path]) -> None:
    total_size = sum(file_path.stat().st_size for file_path in files)
    size_mb = total_size / (1024 * 1024)

    print(f"Source root: {plan.source_root}")
    print(f"Files included: {len(files)}")
    print(f"Uncompressed size: {size_mb:.2f} MB")
    print("Key shipped database/config files:")

    for rel_path in (
        "backend/product_matching.db",
        "backend/catalogs/default-catalog.db",
        "backend/config/active_catalogs.json",
    ):
        full_path = plan.source_root / rel_path
        state = "included" if full_path in files else "missing"
        print(f"  - {rel_path}: {state}")


def generate_seed_db(source_root: Path) -> None:
    """Generate a fresh seed database before packaging.

    Runs ``scripts/create_seed_db.py`` to produce an empty, schema-initialized
    ``backend/product_matching.db`` so the ZIP ships a clean, portable DB
    with no user data or stale file paths.
    """
    import subprocess

    db_path = source_root / "backend" / "product_matching.db"
    seed_script = source_root / "scripts" / "create_seed_db.py"

    if not seed_script.exists():
        print(f"[WARNING] Seed script not found: {seed_script}")
        print("          Shipping existing DB as-is.")
        return

    print(f"Generating fresh seed database -> {db_path}")
    result = subprocess.run(
        [sys.executable, str(seed_script), "--output", str(db_path), "--verify"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] Seed DB generation failed:\n{result.stderr}")
        raise SystemExit(1)

    print(result.stdout.strip())
    print()


def main() -> int:
    args = parse_args()
    plan = build_plan(args)

    if not plan.source_root.exists():
        raise SystemExit(f"Project root does not exist: {plan.source_root}")

    # Generate a clean seed DB so we never ship user data.
    if not args.dry_run:
        generate_seed_db(plan.source_root)

    files = list(iter_files(plan.source_root))
    print_summary(plan, files)

    if args.dry_run:
        print("Dry run only. No ZIP files written.")
        return 0

    outputs = [plan.latest_zip] if args.latest_only else [plan.timestamped_zip, plan.latest_zip]

    for zip_path in outputs:
        if zip_path.exists():
            zip_path.unlink()
        build_zip(plan.source_root, zip_path, files)
        print(f"Wrote {zip_path}")

    if not args.latest_only:
        latest_size = plan.latest_zip.stat().st_size if plan.latest_zip.exists() else 0
        timestamp_size = plan.timestamped_zip.stat().st_size if plan.timestamped_zip.exists() else 0
        if latest_size != timestamp_size:
            print("Warning: latest and timestamped ZIP sizes differ.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
