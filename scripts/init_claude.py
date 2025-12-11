#!/usr/bin/env python3
"""
Generate consolidated Claude Code context from kiro specification files.

This script reads all kiro steering and spec files and generates a single
.claude/context.md file that provides complete project context for AI assistants.

Usage:
    python scripts/init_claude.py

The script will:
1. Read all files from .kiro/steering/ and .kiro/specs/
2. Generate .claude/context.md with consolidated project information
3. Include task status from tasks.md

Run this script whenever you update your kiro files to refresh the context.
"""

import os
from pathlib import Path
from datetime import datetime

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Source directories
KIRO_STEERING_DIR = PROJECT_ROOT / '.kiro' / 'steering'
KIRO_SPECS_DIR = PROJECT_ROOT / '.kiro' / 'specs' / 'product-matching-system'

# Output directory
CLAUDE_DIR = PROJECT_ROOT / '.claude'
OUTPUT_FILE = CLAUDE_DIR / 'context.md'


def read_file(file_path):
    """Read a file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return f"<!-- Could not read {file_path} -->"


def extract_incomplete_tasks(tasks_content):
    """Extract incomplete tasks from tasks.md."""
    lines = tasks_content.split('\n')
    incomplete_tasks = []

    for line in lines:
        # Find lines with [ ] (incomplete) but skip [x] (complete)
        if line.strip().startswith('- [ ]'):
            incomplete_tasks.append(line)

    return incomplete_tasks


def count_tasks(tasks_content):
    """Count complete and incomplete tasks."""
    lines = tasks_content.split('\n')
    complete = sum(1 for line in lines if line.strip().startswith('- [x]'))
    incomplete = sum(1 for line in lines if line.strip().startswith('- [ ]'))
    return complete, incomplete


def generate_context():
    """Generate the consolidated context.md file."""

    # Ensure output directory exists
    CLAUDE_DIR.mkdir(exist_ok=True)

    # Read all kiro files
    print("Reading kiro files...")

    product_md = read_file(KIRO_STEERING_DIR / 'product.md')
    tech_md = read_file(KIRO_STEERING_DIR / 'tech.md')
    structure_md = read_file(KIRO_STEERING_DIR / 'structure.md')
    requirements_md = read_file(KIRO_SPECS_DIR / 'requirements.md')
    design_md = read_file(KIRO_SPECS_DIR / 'design.md')
    tasks_md = read_file(KIRO_SPECS_DIR / 'tasks.md')

    # Extract task information
    complete_count, incomplete_count = count_tasks(tasks_md)
    incomplete_tasks = extract_incomplete_tasks(tasks_md)

    # Generate context file
    print(f"Generating {OUTPUT_FILE}...")

    context_content = f"""# CatalogMatch - Project Context for Claude Code

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Source:** Auto-generated from `.kiro/` specification files
**Regenerate:** Run `python scripts/init_claude.py` after updating kiro files

---

## Project Status

**Tasks Progress:** {complete_count} completed / {incomplete_count} remaining

---

{product_md}

---

{tech_md}

---

{structure_md}

---

## Requirements Summary

The full requirements document contains {len(requirements_md.split('###'))} detailed requirements with acceptance criteria.
Key requirements include:

- **Image Upload & Processing:** Support JPEG/PNG/WebP up to 10MB, extract CLIP embeddings
- **Visual Similarity Matching:** CLIP-based cosine similarity (0-100 score), category filtering
- **Batch Processing:** Up to 100 products at once with progress tracking
- **Three Matching Modes:**
  - Mode 1: Visual only (CLIP embeddings)
  - Mode 2: Metadata only (SKU, name, category)
  - Mode 3: Hybrid (weighted combination)
- **Desktop Application:** PyWebView wrapper, offline-first, local SQLite database
- **GPU Acceleration:** AMD ROCm, NVIDIA CUDA, Apple MPS, Intel GPU support
- **Real-World Data Handling:** Graceful handling of missing metadata, corrupted images, NULL fields

For complete requirements, see: `.kiro/specs/product-matching-system/requirements.md`

---

## Architecture & Design Summary

**Key Design Decisions:**

1. **Lightweight Desktop App:** PyWebView + Flask (no Electron, no build tools)
2. **Folder-Based Workflow:** Upload historical catalog folder → Upload new products → Match → View results
3. **CSV Metadata Linking:** Flexible linking strategies (filename=SKU, regex patterns, fuzzy matching)
4. **CLIP Embeddings:** ViT-B-32 model (~350MB), GPU-accelerated feature extraction
5. **FAISS Indexing:** Efficient similarity search for large catalogs (10K+ products)
6. **Nullable Database Schema:** Only image_path required, graceful handling of missing data
7. **Progressive Enhancement:** Works without GPU (CPU fallback), works without metadata (visual-only matching)

**Technology Stack:**
- Backend: Python 3.12 + Flask + SQLite
- AI/ML: PyTorch + sentence-transformers + CLIP + FAISS
- Frontend: Vanilla JS + Tailwind CSS (brutalist design)
- Desktop: PyWebView 4.4.1

For complete architecture details, see: `.kiro/specs/product-matching-system/design.md`

---

## Incomplete Tasks

The following tasks are still pending (from `tasks.md`):

"""

    # Add incomplete tasks
    if incomplete_tasks:
        for task in incomplete_tasks:
            context_content += f"{task}\n"
    else:
        context_content += "_No incomplete tasks - all features implemented!_\n"

    context_content += f"""

For complete task list with all details, see: `.kiro/specs/product-matching-system/tasks.md`

---

## Key Files & Locations

**Entry Points:**
- `main.py` - Desktop application launcher (PyWebView)
- `backend/app.py` - Flask backend server with REST API
- `start_server.bat/sh` - Platform-specific startup scripts

**Core Backend:**
- `backend/database.py` - SQLite operations (handles NULL/missing data)
- `backend/image_processing_clip.py` - CLIP embeddings (GPU-accelerated)
- `backend/product_matching.py` - Three matching modes (visual, metadata, hybrid)
- `backend/similarity.py` - Cosine similarity + FAISS indexing

**Frontend:**
- `backend/static/index.html` - Main application UI
- `backend/static/catalog-manager.html` - Catalog management tool
- `backend/static/csv-builder.html` - CSV builder tool

**GPU Acceleration:**
- `gpu/setup_gpu.py` - Auto-detects platform and installs PyTorch
- `gpu/check_gpu.py` - Quick GPU detection test
- `gpu/benchmark_gpu.py` - Performance comparison

**Testing:**
- `backend/tests/test_clip.py` - CLIP & GPU tests (33 tests)
- `backend/tests/test_matching.py` - Matching algorithm tests
- `backend/tests/test_database.py` - Database layer tests

---

## Common Commands

**Run the application:**
```bash
# Windows
start_server.bat

# macOS/Linux
./start_server.sh

# Manual start
cd backend && python app.py
```

**GPU setup:**
```bash
# Auto-detect and install GPU support
python gpu/setup_gpu.py

# Check GPU status
python gpu/check_gpu.py

# Benchmark GPU performance
python gpu/benchmark_gpu.py
```

**Testing:**
```bash
# Run all tests
pytest

# Run specific test file
pytest backend/tests/test_clip.py -v

# Run with coverage
pytest --cov=backend
```

**Database operations:**
```bash
# Reset database
python scripts/reset_db.py

# Verify dependencies
python scripts/verify_dependencies.py
```

---

## Critical Constraints & Gotchas

1. **AMD GPU (Windows):** MUST use Python 3.12 (ROCm requirement)
2. **sentence-transformers:** MUST BE <3.0.0 for AMD ROCm compatibility
3. **macOS Port:** Default port 5001 (port 5000 conflicts with AirPlay)
4. **Nullable Fields:** Only `image_path` is required in database, handle NULL gracefully
5. **Real-World Data:** Users upload messy data - missing categories, corrupted images, weird filenames
6. **Performance:** AMD/NVIDIA GPU: 150-300 imgs/sec, CPU: 5-20 imgs/sec
7. **FAISS Indexing:** Rebuild index when catalog changes, cache in memory for performance

---

## Recent Implementation Notes

**CLIP + FAISS Integration:**
- Mode 1 uses CLIP embeddings + FAISS for fast similarity search
- Features extracted on upload, cached in database (BLOB storage)
- FAISS index built in-memory, invalidated when catalog changes
- Parallel processing for batch uploads (multiprocessing)

**Mode 3 Optimizations:**
- Chunked streaming for large result sets (reduces RAM usage)
- Parallel matching with multiprocessing
- Index invalidation on catalog changes
- Session cleanup prevents memory leaks

**CSV Builder Enhancements:**
- Flexible metadata linking (filename=SKU, regex, fuzzy matching)
- Excel workflow: export template → edit in Excel → import back
- Handles missing data gracefully
- Auto-detect categories from folder structure

**Catalog Management:**
- Three modes: use existing / add to existing / replace catalog
- Database cleanup tools (clear by category, date, type)
- Catalog browser with filtering and search
- Test data cleanup utilities

---

## Working with This Codebase

**When adding new features:**
1. Check existing patterns in `structure.md` and `design.md`
2. Follow nullable schema - handle missing/NULL data gracefully
3. Add GPU acceleration where applicable (check `image_processing_clip.py`)
4. Update tests in `backend/tests/`
5. Use real-world data handling (corruption, missing fields, weird formats)

**When debugging:**
1. Check GPU status: `python gpu/check_gpu.py`
2. Verify dependencies: `python scripts/verify_dependencies.py`
3. Check database schema: `sqlite3 backend/product_matching.db .schema`
4. Run tests: `pytest -v`
5. Check logs in terminal (Flask debug mode enabled by default)

**When working on tasks:**
1. Reference `.kiro/specs/product-matching-system/tasks.md` for detailed requirements
2. Use TodoWrite to track progress during implementation
3. Update tasks.md after completing major features (optional)
4. Re-run `python scripts/init_claude.py` if kiro files are updated

---

**End of Context** - You now have complete project context. Ready to work!
"""

    # Write the output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(context_content)

    print(f"[OK] Generated {OUTPUT_FILE}")
    print(f"  - {complete_count} completed tasks")
    print(f"  - {incomplete_count} incomplete tasks")
    print(f"  - File size: {len(context_content)} characters")
    print("\nContext file ready for Claude Code!")


if __name__ == '__main__':
    generate_context()
