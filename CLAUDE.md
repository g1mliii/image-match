# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CatalogMatch is a desktop AI-powered visual product comparison application for inventory management. It uses CLIP embeddings to find visually similar products in catalogs.

**Tech Stack:** Python 3.12 (backend) + Flask + PyWebView (desktop) + SQLite + Vanilla JS frontend with Tailwind CSS and Lucide Icons

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start desktop app (main entry point)
python main.py

# Start Flask server only (development)
cd backend && python app.py

# GPU setup (auto-detects AMD/NVIDIA/Intel/Apple)
python gpu/setup_gpu.py

# Run all tests
pytest

# Run specific test file
pytest backend/tests/test_clip.py -v

# Run GPU tests only
pytest backend/tests/test_clip.py backend/tests/test_gpu_support.py -v

# GPU benchmark
python gpu/benchmark_gpu.py

# Full verification
python gpu/verify_setup.py

# Pre-download CLIP model (~350MB)
python scripts/download_clip_model.py
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  PyWebView (Desktop Container)                      │
│  ├─ Main Window (http://127.0.0.1:5000)            │
│  ├─ CSV Builder (child window)                     │
│  └─ Catalog Manager (child window)                 │
└──────────────┬──────────────────────────────────────┘
               │
        ┌──────▼────────┐
        │  Flask Server │
        │  (app.py)     │
        └──────┬────────┘
               │
      ┌────────┴───────┬──────────────┬────────────┐
      ▼                ▼              ▼            ▼
   SQLite DB      Image Processing   Matching    Snapshots
   (database.py)  (CLIP/Legacy)    (product_   (snapshot_
                  (image_         matching.py) manager.py)
                  processing_
                  clip.py)
```

### Key Components

| File | Purpose |
|------|---------|
| `main.py` | Desktop launcher (PyWebView), multi-window API |
| `backend/app.py` | Flask REST API (40+ endpoints), main business logic |
| `backend/database.py` | SQLite layer with connection pooling |
| `backend/image_processing_clip.py` | CLIP embeddings, GPU detection |
| `backend/product_matching.py` | Visual/metadata/hybrid matching algorithms |
| `backend/snapshot_manager.py` | Catalog snapshots, CSV storage, session management |
| `backend/feature_extraction_service.py` | Unified interface routing to CLIP or legacy |
| `backend/faiss_index.py` | FAISS integration for fast similarity search |
| `backend/path_manager.py` | Cross-platform path handling |
| `backend/static/app.js` | Main frontend logic (7000+ lines), IconManager |

### Multi-Window Communication

Child windows (CSV Builder, Catalog Manager) save data to `staging/` directory with manifest files. Main window polls for changes. No direct inter-window references (prevents crashes).

### Database Schema

- `products` - Core product data with image_path, category, sku, name, metadata (JSON)
- `features` - Visual embeddings (CLIP 512-dim or legacy color/shape/texture)
- `matches` - Similarity results with component scores
- `snapshot_metadata` - CSV content stored as TEXT columns inside snapshot DBs

### CSV Storage

CSV files are stored **inside** snapshot database files (`snapshot_metadata` table), not as separate files. When a snapshot is deleted, CSV data is deleted with it.

## Critical Constraints

- **Python 3.12 required** for AMD GPU support (ROCm compatibility)
- **sentence-transformers <3.0.0** for AMD ROCm (torch.distributed issue)
- **macOS port 5001** (port 5000 conflicts with AirPlay)
- **CLIP model ~350MB** cached in `~/.cache/clip-models/`

## GPU Support Matrix

| Platform | Requirements | Performance |
|----------|-------------|-------------|
| AMD (Windows) | ROCm 6.4+ + Python 3.12 | 150-300 img/s |
| NVIDIA | CUDA 12.4 | 150-300 img/s |
| Apple Silicon | MPS (built-in) | 150-300 img/s |
| Intel | intel-extension-for-pytorch | 30-80 img/s |
| CPU fallback | Any Python 3.8+ | 5-20 img/s |

## File Locations

```
Development:
├── backend/product_matching.db      # Main database
├── backend/catalogs/*.db            # Snapshots
├── backend/uploads/                 # Temporary images
├── backend/config/                  # active_catalogs.json, mobile_config.json
└── backend/logs/                    # Rotating logs (10MB max)

Bundled App (Windows):
%APPDATA%\ProductMatcher\
├── product_matching.db
├── catalogs/
├── uploads/
└── config/
```

## Icon System (Lucide)

`IconManager` in `app.js` (lines 56-165) handles icon initialization with:
- 50ms debounced re-initialization for batched updates
- Scoped initialization with `{ root: container }` for partial DOM updates

Must call `IconManager.reinit(container)` after dynamic content, modals, or status updates.

## Caching

- **CSV cache**: LRU (max 10 items) in memory
- **Category cache**: 5min TTL with thread-safe locks
- **Connection pool**: 3-5 reusable SQLite connections
- **CLIP model**: Singleton pattern, one instance per process

## API Patterns

- Matching: `POST /api/match/{visual,metadata,hybrid}` and batch variants
- Products: `POST /api/products/upload`, `GET /api/products`, `DELETE /api/products/:id`
- Catalogs: `POST /api/catalog/create`, `GET /api/catalogs`, `DELETE /api/catalog/:id`
- CSV: `POST /api/csv/parse`, `GET /api/csv/extract?type={historical,new}`
- System: `GET /api/gpu/status`, `GET /api/database/stats`

## Testing

Test files in `backend/tests/` (35+ files). Key test files:
- `test_clip.py` - CLIP embeddings and GPU (33 tests)
- `test_matching.py` - Matching algorithms
- `test_database.py` - Database operations
- `test_full_workflow_e2e.py` - End-to-end workflow
- `test_snapshot_e2e.py` - Snapshot management
