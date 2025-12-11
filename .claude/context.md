# CatalogMatch - Project Context for Claude Code

**Generated:** 2025-12-10 23:15:47
**Source:** Auto-generated from `.kiro/` specification files
**Regenerate:** Run `python scripts/init_claude.py` after updating kiro files

---

## Project Status

**Tasks Progress:** 20 completed / 15 remaining

---

# Product Overview

CatalogMatch is a desktop application for AI-powered visual product comparison and inventory management.

## Core Purpose
Help businesses compare new products against existing inventory using visual similarity analysis (color, shape, texture) to make smarter purchasing decisions and avoid duplicate inventory.

## Key Features
- **Visual Similarity Matching**: CLIP-based embeddings for accurate visual comparison
- **Category Filtering**: Match products within same category or across all categories
- **Batch Processing**: Handle up to 100 products at once
- **Offline Desktop App**: All processing happens locally, no cloud required
- **CSV Import/Export**: Easy integration with existing systems
- **GPU Acceleration**: Optional AMD/NVIDIA/Intel/Apple Silicon support for 3-30x speedup

## Target Users
- E-commerce inventory managers
- Retail stock managers
- Product catalog managers
- Businesses needing to identify duplicate or similar products

## Pricing Model
- Free: Up to 50 products
- Pro: $29 one-time payment for unlimited products

## Privacy & Security
- 100% offline processing
- No cloud storage or tracking
- Local SQLite database
- Data never leaves user's machine


---

# Technology Stack

## Backend
- **Python 3.12** (required for AMD GPU support; 3.8+ for others)
- **Flask 3.0.0** - Web framework with CORS support
- **SQLite** - Local database (no external DB server needed)
- **OpenCV** - Image processing
- **NumPy/SciPy** - Scientific computing
- **Pillow** - Image handling
- **PyWebView 4.4.1** - Desktop wrapper

## AI/ML
- **PyTorch 2.x** - Deep learning framework
  - AMD: PyTorch 2.8.0 + ROCm 6.4
  - NVIDIA: PyTorch 2.x + CUDA 12.4
  - Apple Silicon: PyTorch 2.x + MPS
  - CPU: PyTorch 2.x CPU-only
- **sentence-transformers <3.0.0** - CLIP model wrapper (version locked for AMD ROCm compatibility)
- **CLIP (ViT-B-32)** - Visual embeddings (~350MB model)

## Frontend
- **HTML/CSS/JavaScript** - Vanilla JS, no framework
- **Tailwind CSS** - Utility-first styling
- **Brutalist design** - Current UI style

## GPU Support
- **AMD ROCm 6.4** - Windows AMD GPU acceleration
- **NVIDIA CUDA 12.4** - NVIDIA GPU acceleration
- **Apple MPS** - Apple Silicon acceleration
- **Intel Extension for PyTorch** - Intel GPU acceleration (optional)

## Common Commands

### Installation
```bash
# Install base requirements
pip install -r requirements.txt

# Install GPU support (auto-detects platform)
python gpu/setup_gpu.py

# AMD GPU users (Windows) - must use Python 3.12
py -3.12 gpu/setup_gpu.py
```

### Running the Application
```bash
# Start server (Windows)
start_server.bat

# Start server (macOS/Linux)
./start_server.sh

# Manual start
cd backend
python app.py

# AMD GPU users (Windows)
cd backend
py -3.12 app.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest backend/tests/test_clip.py

# Run GPU tests only
pytest backend/tests/test_clip.py backend/tests/test_gpu_support.py backend/tests/test_amd_gpu.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=backend
```

### GPU Verification
```bash
# Quick GPU check
python gpu/check_gpu.py

# Performance benchmark
python gpu/benchmark_gpu.py

# Complete verification suite
python gpu/verify_setup.py
```

### Database Operations
```bash
# Reset database
python reset_db.py

# Verify dependencies
python verify_dependencies.py
```

### Development
```bash
# Flask runs in debug mode by default (auto-reload)
cd backend
python app.py

# Access application
# macOS: http://127.0.0.1:5001 (port 5001 to avoid AirPlay conflict)
# Windows/Linux: http://127.0.0.1:5000
```

## Critical Version Constraints

### Python Version
- **AMD GPU (Windows)**: MUST use Python 3.12 (ROCm requirement)
- **Others**: Python 3.8+ works

### sentence-transformers Version
- **MUST BE <3.0.0** for AMD ROCm compatibility
- Version 3.0+ requires `torch.distributed.is_initialized()` which is missing in ROCm Windows builds
- This constraint applies even with ROCm 7.1.1

### Port Configuration
- **macOS**: Default port 5001 (port 5000 conflicts with AirPlay Receiver)
- **Windows/Linux**: Default port 5000

## Performance Expectations
- **AMD/NVIDIA GPU**: 150-300 images/sec
- **Intel GPU**: 30-80 images/sec (3-5x faster than CPU)
- **Apple Silicon**: 50-150 images/sec
- **CPU only**: 5-20 images/sec


---

# Project Structure

## Directory Organization

```
image-match/
├── backend/                    # Backend application (Flask + Python)
│   ├── static/                # Frontend files (HTML/CSS/JS)
│   ├── uploads/               # User-uploaded product images
│   ├── tests/                 # All test files (pytest)
│   ├── docs/                  # Technical documentation
│   ├── config/                # Configuration files
│   ├── catalogs/              # Catalog database files
│   ├── app.py                 # Flask application entry point
│   ├── database.py            # SQLite database layer
│   ├── image_processing.py    # Legacy feature extraction
│   ├── image_processing_clip.py  # CLIP embeddings (GPU-accelerated)
│   ├── feature_extraction_service.py  # Unified feature extraction
│   ├── product_matching.py    # Matching algorithms
│   ├── similarity.py          # Similarity computation
│   ├── validation_utils.py    # Input validation utilities
│   └── snapshot_manager.py    # Catalog snapshot management
│
├── gpu/                        # GPU acceleration setup & tools
│   ├── setup_gpu.py           # Automated GPU setup (all platforms)
│   ├── check_gpu.py           # Quick GPU detection
│   ├── benchmark_gpu.py       # Performance benchmark
│   ├── verify_setup.py        # Complete verification suite
│   └── GPU_SETUP_GUIDE.md     # Comprehensive GPU guide
│
├── docs/                       # GitHub Pages website
│   ├── guides/                # User setup guides
│   │   ├── SETUP_SIMPLE.md
│   │   └── START_SERVER.md
│   ├── index.html             # Landing page
│   ├── download.html          # Download page
│   ├── pricing.html           # Pricing page
│   └── docs.html              # User documentation
│
├── scripts/                    # Utility scripts
│   ├── verify_dependencies.py # Dependency verification & auto-fix
│   ├── download_clip_model.py # Pre-download CLIP model
│   ├── reset_db.py            # Database reset utility
│   └── README.md              # Scripts documentation
│
├── sample-data/               # Sample CSV files for testing
├── main.py                    # Desktop application launcher (pywebview)
├── requirements.txt           # Main Python dependencies
├── pytest.ini                 # Pytest configuration
└── start_server.bat/sh        # Server startup scripts
```

## Key File Purposes

### Application Entry Points
- **`main.py`** - Desktop app launcher using pywebview
- **`backend/app.py`** - Flask backend server with REST API
- **`start_server.bat/sh`** - Platform-specific startup scripts

### Core Backend Modules
- **`database.py`** - SQLite operations, handles NULL/missing data gracefully
- **`image_processing_clip.py`** - CLIP-based visual embeddings (GPU-accelerated)
- **`feature_extraction_service.py`** - Unified interface for CLIP/legacy extraction
- **`product_matching.py`** - Three matching modes: visual, metadata, hybrid
- **`similarity.py`** - Cosine similarity computation on embeddings
- **`validation_utils.py`** - Input validation and sanitization

### GPU Acceleration
- **`gpu/setup_gpu.py`** - Auto-detects platform and installs correct PyTorch
- **`gpu/check_gpu.py`** - Quick GPU detection test
- **`gpu/benchmark_gpu.py`** - Performance comparison (CPU vs GPU)
- **`gpu/verify_setup.py`** - Runs full test suite for verification
- **`gpu/INTEL_GPU_SETUP.md`** - Intel GPU specific setup guide

### Utility Scripts
- **`scripts/verify_dependencies.py`** - Verifies dependencies and auto-fixes issues
- **`scripts/download_clip_model.py`** - Pre-downloads CLIP model for offline use
- **`scripts/reset_db.py`** - Resets database (deletes all data)

### Frontend
- **`backend/static/index.html`** - Main application UI (brutalist design)
- **`backend/static/app.js`** - Main application logic
- **`backend/static/catalog-manager.html`** - Catalog management tool
- **`backend/static/csv-builder.html`** - CSV builder tool

### Testing
- **`backend/tests/test_clip.py`** - CLIP & GPU tests (33 tests)
- **`backend/tests/test_gpu_support.py`** - Platform detection (3 tests)
- **`backend/tests/test_amd_gpu.py`** - AMD-specific tests (7 tests)
- **`backend/tests/test_matching.py`** - Matching algorithm tests
- **`backend/tests/test_database.py`** - Database layer tests

## Architecture Patterns

### Database Layer
- **Nullable fields by design** - Only `image_path` is required
- **Graceful handling of missing data** - Category, SKU, name can be NULL
- **Separate features table** - Products can exist without features initially
- **BLOB storage** - NumPy arrays serialized to bytes for feature vectors

### Feature Extraction
- **Unified interface** - `extract_features_unified()` handles CLIP or legacy
- **Automatic fallback** - CLIP → legacy if GPU unavailable
- **Embedding versioning** - Tracks embedding type and version in database
- **Lazy loading** - CLIP model loaded on first use, cached thereafter

### Matching Modes
1. **Visual matching** - CLIP embeddings + cosine similarity (requires features)
2. **Metadata matching** - SKU/name/category comparison (no features needed)
3. **Hybrid matching** - Weighted combination of visual + metadata

### Error Handling
- **Custom exception classes** - `ImageProcessingError`, `MatchingError`, etc.
- **Standardized error responses** - `create_error_response()` with error codes
- **Graceful degradation** - Continue processing even if some products fail
- **Detailed logging** - All errors logged with context

### GPU Support
- **Automatic detection** - Detects AMD/NVIDIA/Intel/Apple Silicon
- **Graceful fallback** - Falls back to CPU if GPU unavailable
- **Platform-specific optimizations** - Different PyTorch builds per platform
- **Performance monitoring** - Tracks throughput and reports to UI

## File Naming Conventions

### Python Modules
- **Snake case** - `image_processing_clip.py`, `product_matching.py`
- **Descriptive names** - Module name indicates purpose
- **No abbreviations** - Prefer clarity over brevity

### Test Files
- **Prefix with `test_`** - `test_clip.py`, `test_matching.py`
- **Match module name** - `test_database.py` tests `database.py`
- **Descriptive test names** - `test_clip_visual_matching_with_gpu()`

### Frontend Files
- **Kebab case** - `catalog-manager.html`, `csv-builder.js`
- **Descriptive names** - File name indicates purpose
- **Separate by concern** - HTML, CSS, JS in separate files

## Code Organization Principles

### Separation of Concerns
- **Database layer** - Pure SQLite operations, no business logic
- **Feature extraction** - Image processing only, no matching logic
- **Matching logic** - Uses features from extraction, no direct image access
- **API layer** - Request validation, error handling, response formatting

### Dependency Flow
```
app.py (API)
  ↓
product_matching.py (business logic)
  ↓
feature_extraction_service.py (unified interface)
  ↓
image_processing_clip.py OR image_processing.py
  ↓
database.py (data access)
```

### Configuration
- **Environment-based** - Different ports for macOS (5001) vs others (5000)
- **Platform detection** - Auto-detect GPU type and capabilities
- **Graceful defaults** - Sensible defaults for all optional parameters

## Important Paths

### Upload Directory
- **Location**: `backend/uploads/`
- **Naming**: `{timestamp}_{original_filename}`
- **Cleanup**: Manual cleanup via catalog manager

### Database Files
- **Main DB**: `backend/product_matching.db`
- **Catalog DBs**: `backend/catalogs/{catalog_name}.db`
- **Backup**: `backend/product_matching.db.backup`

### Model Cache
- **CLIP model**: `~/.cache/torch/sentence_transformers/`
- **Size**: ~350MB (downloaded on first use)
- **Shared**: Same cache across all Python environments

## Adding New Features

### New API Endpoint
1. Add route in `backend/app.py`
2. Add validation in `validation_utils.py`
3. Add business logic in appropriate module
4. Add tests in `backend/tests/`
5. Update frontend in `backend/static/`

### New Matching Algorithm
1. Add function in `product_matching.py`
2. Add similarity computation in `similarity.py`
3. Add database queries in `database.py` if needed
4. Add tests in `backend/tests/test_matching.py`
5. Expose via API in `backend/app.py`

### New GPU Platform
1. Add detection in `image_processing_clip.py`
2. Add setup logic in `gpu/setup_gpu.py`
3. Add tests in `backend/tests/test_gpu_support.py`
4. Update documentation in `gpu/GPU_SETUP_GUIDE.md`


---

## Requirements Summary

The full requirements document contains 23 detailed requirements with acceptance criteria.
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

- [ ] 25.1. GPU acceleration performance monitoring and metrics
- [ ] 25.2. GPU acceleration testing and validation
- [ ] 22.5. Set up distribution and analytics
- [ ] 23. Marketing and launch
  - [ ] 16.5. Create application icons for cross-platform packaging
  - [ ] 17. Package Windows executable with PyInstaller for public release whic users will download by clicking link on site and some solution to enable download without triggerig virus issue on chrome and other browsers
- [ ] 18. Package macOS application for public release whic users will download by clicking link on site and some solution to enable download without triggerig virus issue on chrome and other browsers
- [ ] 40. Implement auto-update mechanism for packaged application
- [ ] 31. Implement mobile photo upload with WiFi connection
- [ ] 31.1. Add password authentication for mobile access
- [ ] 31.2. Add mobile upload support to existing endpoints
- [ ] 31.3. Create desktop UI for mobile connection setup
- [ ] 31.4. Create mobile upload page (static/mobile-upload.html)
- [ ] 31.5. Add network discovery helpers
- [ ] 31.6. Test mobile upload functionality


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
