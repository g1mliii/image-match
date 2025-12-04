# Quick Navigation Guide

Quick reference for finding files in the reorganized workspace.

## 🚀 Getting Started

| What you need | Where to find it |
|---------------|------------------|
| Installation instructions | `INSTALLATION.md` |
| Quick start guide | `README.md` |
| Simple setup | `docs/guides/SETUP_SIMPLE.md` |
| Server startup | `docs/guides/START_SERVER.md` |

## 🔧 Running the Application

| Task | Command |
|------|---------|
| Start server (Windows) | `start_server.bat` |
| Start server (macOS/Linux) | `./start_server.sh` |
| Desktop app | `python main.py` |
| Backend only | `python backend/app.py` |

## 🛠️ Utility Scripts

All utility scripts are now in `scripts/`:

| Script | Purpose | Command |
|--------|---------|---------|
| Verify dependencies | Check & fix dependencies | `python scripts/verify_dependencies.py` |
| Download CLIP model | Pre-download model | `python scripts/download_clip_model.py` |
| Reset database | Clear all data | `python scripts/reset_db.py` |

## 🎮 GPU Setup

All GPU-related files are in `gpu/`:

| File | Purpose |
|------|---------|
| `gpu/setup_gpu.py` | Auto-setup for all platforms |
| `gpu/check_gpu.py` | Quick GPU detection |
| `gpu/benchmark_gpu.py` | Performance test |
| `gpu/GPU_SETUP_GUIDE.md` | Complete GPU guide |
| `gpu/INTEL_GPU_SETUP.md` | Intel-specific guide |

**Commands:**
```bash
python gpu/setup_gpu.py      # Setup GPU
python gpu/check_gpu.py      # Check GPU
python gpu/benchmark_gpu.py  # Benchmark
```

## 📚 Documentation

### User Documentation
- **Website**: `docs/` folder (GitHub Pages)
- **Setup Guides**: `docs/guides/`
  - `SETUP_SIMPLE.md` - Simple setup
  - `START_SERVER.md` - Server startup

### Developer Documentation
- **Backend docs**: `backend/docs/`
  - `CLIP_DEVELOPER_GUIDE.md` - CLIP integration
  - `DATABASE_DESIGN.md` - Database schema
  - `MATCHING_SERVICE.md` - Matching algorithms
  - `ERROR_HANDLING_IMPLEMENTATION.md` - Error handling
  - `PRICE_HISTORY_GUIDE.md` - Price history feature
  - `UI_FEATURES_GUIDE.md` - UI features

### Project Documentation
- **Root level**:
  - `README.md` - Project overview
  - `INSTALLATION.md` - Installation guide
  - `WORKSPACE_STRUCTURE.md` - Workspace organization
  - `QUICK_REFERENCE.md` - Quick reference
  - `CONTRIBUTING.md` - Contribution guidelines

## 🧪 Testing

| Task | Command |
|------|---------|
| Run all tests | `pytest` |
| Run specific test | `pytest backend/tests/test_clip.py` |
| Run with verbose | `pytest -v` |
| GPU tests only | `pytest backend/tests/test_clip.py backend/tests/test_gpu_support.py` |

## 📁 Key Directories

```
image-match/
├── backend/          # Backend application
│   ├── static/      # Frontend (HTML/CSS/JS)
│   ├── tests/       # All tests
│   ├── docs/        # Technical docs
│   └── *.py         # Backend modules
│
├── gpu/             # GPU setup & tools
├── docs/            # Website & user guides
│   └── guides/      # Setup guides
├── scripts/         # Utility scripts
├── sample-data/     # Sample CSV files
└── docs-backup/     # Archived docs
```

## 🔍 Finding Specific Files

### Configuration Files
- Python dependencies: `requirements.txt`
- Pytest config: `pytest.ini`
- Git ignore: `.gitignore`
- License: `LICENSE`

### Application Files
- Desktop launcher: `main.py`
- Flask backend: `backend/app.py`
- Database layer: `backend/database.py`
- CLIP processing: `backend/image_processing_clip.py`
- Matching logic: `backend/product_matching.py`

### Frontend Files
- Main UI: `backend/static/index.html`
- Main JS: `backend/static/app.js`
- Styles: `backend/static/styles.css`
- Catalog manager: `backend/static/catalog-manager.html`
- CSV builder: `backend/static/csv-builder.html`

## 📝 Common Tasks

### First Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python scripts/verify_dependencies.py

# 3. Setup GPU (optional)
python gpu/setup_gpu.py

# 4. Start server
start_server.bat  # Windows
./start_server.sh # macOS/Linux
```

### Development Workflow
```bash
# 1. Make changes to code
# 2. Run tests
pytest

# 3. Check for errors
python scripts/verify_dependencies.py

# 4. Start server
python backend/app.py
```

### Troubleshooting
```bash
# Check dependencies
python scripts/verify_dependencies.py

# Check GPU
python gpu/check_gpu.py

# Reset database
python scripts/reset_db.py

# View logs
# Check terminal output when running backend/app.py
```

## 🆘 Getting Help

| Issue | Where to look |
|-------|---------------|
| Installation problems | `INSTALLATION.md` |
| GPU not working | `gpu/GPU_SETUP_GUIDE.md` |
| Dependency errors | `scripts/README.md` |
| Database issues | `backend/docs/DATABASE_DESIGN.md` |
| API questions | `backend/docs/MATCHING_SERVICE.md` |
| General questions | `README.md` or `QUICK_REFERENCE.md` |

## 📊 Project Stats

- **Root files**: 13 (down from 20)
- **Main folders**: 7
- **Backend modules**: 10+
- **Test files**: 30+
- **Documentation files**: 20+
- **Lines of code**: 15,000+

---

**Last updated**: December 3, 2025 (after workspace cleanup)
