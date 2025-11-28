# Workspace Organization

Last Updated: November 23, 2025

## Documentation Structure

**Last Updated**: November 20, 2025

## 📁 Directory Structure

```
image-match/
├── .git/                           # Git repository
├── .github/                        # GitHub workflows and configs
├── .kiro/                          # Kiro IDE configuration
│   └── specs/                      # Feature specifications
│       └── product-matching-system/
│           ├── design.md           # System design document
│           ├── requirements.md     # Requirements specification
│           └── tasks.md            # Implementation tasks (✅ 25/26 complete)
├── .vscode/                        # VS Code settings
│
├── backend/                        # 🔧 Backend Application
│   ├── docs/                       # Backend-specific documentation
│   │   ├── DATABASE_DESIGN.md
│   │   ├── FUZZY_CATEGORY_MATCHING_SUMMARY.md
│   │   ├── IMAGE_PROCESSING_ERRORS.md
│   │   ├── IMPLEMENTATION_SUMMARY.md
│   │   ├── MATCHING_SERVICE.md
│   │   ├── PERFORMANCE_OPTIMIZATIONS.md
│   │   ├── REAL_WORLD_DATA_HANDLING.md
│   │   ├── SIMILARITY_ERROR_HANDLING.md
│   │   └── SKU_IMPLEMENTATION.md
│   ├── static/                     # Frontend assets
│   │   ├── app.js                  # Main JavaScript (with price & performance)
│   │   ├── index.html              # Main HTML (with CSV help modal)
│   │   └── styles.css              # Styles (with price & performance styles)
│   ├── tests/                      # ✅ Test Suite (18 test files)
│   │   ├── __init__.py
│   │   ├── example_similarity_usage.py
│   │   ├── README.md
│   │   ├── run_all_tests.py
│   │   ├── test_database.py
│   │   ├── test_feature_cache.py
│   │   ├── test_fuzzy_category_matching.py
│   │   ├── test_fuzzy_matching_integration.py
│   │   ├── test_image_processing.py
│   │   ├── test_matching.py
│   │   ├── test_performance_api.py        # ✅ NEW: Performance API tests
│   │   ├── test_performance_history.py    # ✅ NEW: Performance history tests
│   │   ├── test_performance_optimizations.py
│   │   ├── test_price_api.py              # ✅ NEW: Price API tests
│   │   ├── test_price_history.py          # ✅ NEW: Price history tests
│   │   ├── test_realworld_data.py
│   │   ├── test_similarity_simple.py
│   │   ├── test_similarity.py
│   │   └── test_sku_handling.py
│   ├── uploads/                    # Uploaded product images
│   ├── __pycache__/                # Python cache (gitignored)
│   ├── app.py                      # Flask application (with price & performance APIs)
│   ├── database.py                 # Database layer (with price_history & performance_history)
│   ├── feature_cache.py            # Feature caching
│   ├── image_processing.py         # Image processing
│   ├── product_matching.db         # SQLite database
│   ├── product_matching.py         # Matching logic
│   ├── requirements.txt            # Python dependencies
│   └── similarity.py               # Similarity computation
│
├── docs/                           # 🌐 Public Documentation (GitHub Pages)
│   ├── images/
│   ├── 404.html
│   ├── contact.html
│   ├── docs.html
│   ├── download.html
│   ├── index.html
│   ├── pricing.html
│   ├── privacy.html
│   ├── robots.txt
│   ├── scripts.js
│   ├── sitemap.xml
│   ├── styles.css
│   ├── terms.html
│   └── _config.yml
│
├── docs-backup/                    # 📦 Backup of Old Docs
│   ├── DEPLOYMENT.md
│   ├── ICON-REPLACEMENT-GUIDE.md
│   ├── index.old.html
│   ├── README.md
│   ├── scripts.old.js
│   ├── SECURITY.md
│   └── styles.old.css
│
├── docs-implementation/            # 📚 Implementation Documentation
│   ├── IMPLEMENTATION_COMPLETE.md          # Price history completion
│   ├── PERFORMANCE_HISTORY_COMPLETE.md     # Performance history completion
│   ├── PERFORMANCE_HISTORY_TESTING.md      # Performance testing results
│   ├── PRICE_HISTORY_GUIDE.md              # User guide for price history
│   ├── PRICE_HISTORY_IMPLEMENTATION_SUMMARY.md
│   ├── TASK_25_COMPLETE.md                 # Task 25 completion summary
│   ├── TESTING_RESULTS.md                  # Price history testing
│   └── UI_FEATURES_GUIDE.md                # UI features walkthrough
│
├── node_modules/                   # Node dependencies (if any)
│
├── sample-data/                    # 📊 Sample Data Files
│   └── sample_product_data.csv     # Complete sample with price & performance
│
├── .gitignore                      # Git ignore rules
├── CONTRIBUTING.md                 # Contribution guidelines
├── ERROR_HANDLING_IMPLEMENTATION.md # Error handling documentation
├── LICENSE                         # MIT License
├── main.py                         # Desktop launcher (pywebview)
├── QUICK_REFERENCE.md              # Quick reference guide
├── README.md                       # Main project README
├── requirements.txt                # Root Python dependencies
├── SETUP_SIMPLE.md                 # Simple setup guide
└── WORKSPACE_ORGANIZATION.md       # This file
```

---

## 📊 Statistics

### Code Files
- **Backend Python**: 7 modules
- **Frontend**: 3 files (HTML, CSS, JS)
- **Tests**: 18 test files
- **Total Lines of Code**: ~15,000+

### Documentation
- **Implementation Docs**: 8 files
- **Backend Docs**: 9 files
- **Public Docs**: 11 files
- **Root Docs**: 6 files
- **Total**: 34 documentation files

### Features Implemented
- ✅ Image-based product matching
- ✅ Category filtering
- ✅ Fuzzy category matching
- ✅ SKU handling
- ✅ Real-world data handling
- ✅ Price history tracking (Task 24)
- ✅ Performance history tracking (Task 25)
- ⏳ CSV Builder UI (Task 26 - pending)

---

## 🎯 Quick Access

### Start the Application
```bash
python backend/app.py
```
**Access at**: http://127.0.0.1:5000

### Run All Tests
```bash
python backend/tests/run_all_tests.py
```

### Run Specific Tests
```bash
# Price history
python backend/tests/test_price_history.py
python backend/tests/test_price_api.py

# Performance history
python backend/tests/test_performance_history.py
python backend/tests/test_performance_api.py

# Database
python backend/tests/test_database.py

# Matching
python backend/tests/test_matching.py
```

### View Documentation
- **User Guides**: `docs-implementation/`
- **Backend Docs**: `backend/docs/`
- **Public Site**: `docs/`
- **Sample Data**: `sample-data/`

---

## 🗂️ File Organization Principles

### Backend Code (`backend/`)
- **Core modules**: Main application logic
- **Tests**: All in `tests/` subdirectory
- **Documentation**: All in `docs/` subdirectory
- **Static assets**: All in `static/` subdirectory

### Documentation
- **Public docs**: `docs/` (GitHub Pages website)
- **Implementation docs**: `docs-implementation/` (technical details, testing)
- **Backend docs**: `backend/docs/` (backend-specific documentation)
- **Backup docs**: `docs-backup/` (old versions, archived)
- **Root docs**: High-level guides (README, SETUP, CONTRIBUTING)

### Sample Data
- **Sample files**: `sample-data/` (CSV templates, example data)
- **Naming**: Descriptive names (e.g., `sample_product_data.csv`)

### Configuration
- **IDE configs**: `.kiro/`, `.vscode/`
- **Git config**: `.git/`, `.gitignore`
- **GitHub config**: `.github/`

---

## 🧹 Recent Cleanup (Nov 20, 2025)

### Files Removed
- ❌ `ORGANIZATION_SUMMARY.md` (redundant)
- ❌ `PROJECT_STRUCTURE.md` (redundant)
- ❌ `backend/tests/test_db.py` (old duplicate)
- ❌ `backend/tests/test_real_world_data.py` (duplicate)
- ❌ `backend/tests/test_error_handling.html` (not needed)
- ❌ `sample-data/sample_with_price_history.csv` (superseded)

### Files Moved
- ✅ `IMPLEMENTATION_COMPLETE.md` → `docs-implementation/`
- ✅ `TASK_25_COMPLETE.md` → `docs-implementation/`

### Files Renamed
- ✅ `sample_with_price_and_performance.csv` → `sample_product_data.csv`

### Result
- **Cleaner root directory**
- **Better organized documentation**
- **No duplicate files**
- **Clear file naming**

---

## 📝 Maintenance Guidelines

### Adding New Features
1. Update specs in `.kiro/specs/product-matching-system/tasks.md`
2. Implement in `backend/`
3. Add tests in `backend/tests/`
4. Document in `docs-implementation/`
5. Update this file

### Adding New Tests
1. Create test file in `backend/tests/`
2. Follow naming: `test_<feature>.py`
3. Update `run_all_tests.py` if needed
4. Document test results in `docs-implementation/`

### Adding Documentation
1. **Implementation docs** → `docs-implementation/`
2. **Backend docs** → `backend/docs/`
3. **User-facing docs** → `docs/` (public site)
4. **Root docs** → Only high-level guides

### Cleaning Up
```bash
# Remove uploaded images
rm -rf backend/uploads/*

# Remove database (will be recreated)
rm backend/product_matching.db

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

---

## 🎯 Current Status

### Completed Tasks (25/26)
- ✅ Tasks 1-24: Core functionality
- ✅ Task 25: Performance history tracking
- ⏳ Task 26: CSV Builder UI (next)

### Test Coverage
- **Unit Tests**: 50+ tests
- **API Tests**: 20+ tests
- **Integration Tests**: 10+ tests
- **Total**: 80+ tests
- **Pass Rate**: 100% ✅

### Code Quality
- **Syntax Errors**: 0 ✅
- **Linting**: Clean ✅
- **Type Hints**: Partial
- **Documentation**: Complete ✅

### Server Status
- **Running**: ✅ http://127.0.0.1:5001
- **Database**: ✅ Initialized with all tables
- **Features**: ✅ All working

---

## 🚀 Next Steps

1. **Task 26**: Implement CSV Builder UI
2. **Testing**: Continue comprehensive testing
3. **Documentation**: Keep docs updated
4. **Optimization**: Monitor performance
5. **Deployment**: Prepare for production

---

## 📞 Support

### For Development
- Check `backend/docs/` for backend documentation
- Check `docs-implementation/` for implementation details
- Run tests to verify functionality

### For Users
- Check `README.md` for overview
- Check `SETUP_SIMPLE.md` for setup
- Check `QUICK_REFERENCE.md` for quick help
- Check `docs-implementation/PRICE_HISTORY_GUIDE.md` for price history
- Check `docs-implementation/UI_FEATURES_GUIDE.md` for UI features

---

## ✅ Workspace Status

**CLEAN AND ORGANIZED** ✅

- ✅ No duplicate files
- ✅ Clear directory structure
- ✅ Logical file organization
- ✅ Comprehensive documentation
- ✅ All tests passing
- ✅ Server running
- ✅ Ready for development

**Last cleanup**: November 20, 2025


### Root Directory Files

**Installation & Setup:**
- `INSTALLATION.md` - Complete installation guide for all platforms
- `START_SERVER_README.md` - Server startup guide
- `start_server.bat` - Windows startup script (auto-detects GPU)
- `start_server.sh` - macOS/Linux startup script
- `requirements.txt` - Main Python requirements

**Documentation:**
- `README.md` - Project overview and quick start
- `QUICK_REFERENCE.md` - Quick reference guide
- `SETUP_SIMPLE.md` - Simple setup guide
- `CONTRIBUTING.md` - Contribution guidelines
- `WORKSPACE_ORGANIZATION.md` - This file
- `WORKSPACE_STRUCTURE.md` - Detailed workspace structure

### Documentation Folders

**`docs/` - Website & User Documentation**
- `index.html`, `download.html`, `docs.html` - Website pages
- `guides/` - User guides directory
  - `README.md` - Documentation index
  - `CLIP_USER_GUIDE.md` - Link to CLIP guide

**`backend/docs/` - Developer Documentation**
- `CLIP_DEVELOPER_GUIDE.md` - CLIP integration technical guide
- `DATABASE_DESIGN.md` - Database schema and design
- `MATCHING_SERVICE.md` - Matching algorithm details
- `PERFORMANCE_OPTIMIZATIONS.md` - Performance tips
- `IMAGE_PROCESSING_ERRORS.md` - Error handling
- `SKU_IMPLEMENTATION.md` - SKU handling
- `FUZZY_CATEGORY_MATCHING_SUMMARY.md` - Category matching
- `REAL_WORLD_DATA_HANDLING.md` - Data handling
- `SIMILARITY_ERROR_HANDLING.md` - Similarity computation

**`gpu/` - GPU Acceleration Documentation**
- `GPU_SETUP_GUIDE.md` - Complete GPU setup guide
- `GPU_STATUS.md` - GPU status and compatibility
- `MEMORY_MANAGEMENT.md` - GPU memory management
- `WORKFLOW_INTEGRATION.md` - GPU workflow integration
- `README.md` - GPU folder overview

**`docs-implementation/` - Implementation Guides**
- `ERROR_HANDLING_IMPLEMENTATION.md` - Error handling
- `PRICE_HISTORY_GUIDE.md` - Price history feature
- `UI_FEATURES_GUIDE.md` - UI features

**`docs-backup/` - Archived Documentation**
- Old documentation and backup files

### Spec Files

**`.kiro/specs/product-matching-system/`**
- `requirements.md` - Feature requirements
- `design.md` - System design document
- `tasks.md` - Implementation tasks
- `clip-implementation-tasks.md` - CLIP-specific tasks

## Quick Navigation

### For Users
1. Start here: `README.md`
2. Install: `INSTALLATION.md`
3. Run: `start_server.bat` (Windows) or `start_server.sh` (macOS/Linux)
4. GPU Setup: `gpu/GPU_SETUP_GUIDE.md`
5. Online docs: https://g1mliii.github.io/image-match/docs.html

### For Developers
1. Architecture: `backend/docs/CLIP_DEVELOPER_GUIDE.md`
2. Database: `backend/docs/DATABASE_DESIGN.md`
3. Matching: `backend/docs/MATCHING_SERVICE.md`
4. Specs: `.kiro/specs/product-matching-system/`

### For GPU Setup
1. Quick start: `gpu/README.md`
2. Detailed guide: `gpu/GPU_SETUP_GUIDE.md`
3. Check GPU: `python gpu/check_gpu.py`
4. Benchmark: `python gpu/benchmark_gpu.py`

## File Organization Principles

1. **Root level** - Installation, setup, and startup files
2. **docs/** - User-facing documentation and website
3. **backend/docs/** - Technical/developer documentation
4. **gpu/** - GPU-specific documentation and scripts
5. **specs/** - Requirements, design, and task specifications

## Recent Additions (Task 26.7)

### New Files Created
- `INSTALLATION.md` - Comprehensive installation guide
- `START_SERVER_README.md` - Server startup guide
- `start_server.bat` - Windows startup script
- `start_server.sh` - macOS/Linux startup script
- `backend/docs/CLIP_DEVELOPER_GUIDE.md` - CLIP technical guide
- `docs/guides/README.md` - Documentation index
- `docs/guides/CLIP_USER_GUIDE.md` - CLIP user guide link

### Updated Files
- `docs/download.html` - Added GPU requirements section
- `docs/index.html` - Updated features with GPU info
- `docs/docs.html` - Added CLIP FAQ and GPU troubleshooting
- `backend/requirements.txt` - Updated with GPU requirements
- `requirements.txt` - Updated with installation instructions
- `README.md` - Added documentation section
- `backend/static/index.html` - Added GPU status indicator
- `backend/static/styles.css` - Added GPU status styling
- `backend/static/app.js` - Added GPU status JavaScript
- `backend/app.py` - Added `/api/gpu/status` endpoint
- `backend/image_processing_clip.py` - Added `get_device_info()` function

## Maintenance

This file should be updated when:
- New documentation files are added
- Documentation structure changes
- Major features are added
- File organization changes
