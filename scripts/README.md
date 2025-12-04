# Utility Scripts

This folder contains utility scripts for setup, verification, and maintenance.

## Scripts

### `verify_dependencies.py`
Verifies all dependencies are installed correctly and auto-fixes common issues.

**Usage:**
```bash
python scripts/verify_dependencies.py

# AMD GPU users (Windows)
py -3.12 scripts/verify_dependencies.py
```

**What it checks:**
- Python 3.12 version (required for AMD ROCm)
- sentence-transformers < 3.0.0 (critical for AMD compatibility)
- All Flask, PyTorch, OpenCV dependencies
- GPU support (AMD/NVIDIA/Apple Silicon)

**Auto-fixes:**
- Downgrades sentence-transformers if >= 3.0.0
- Installs missing dependencies
- Fixes version conflicts

### `download_clip_model.py`
Pre-downloads the CLIP model (~350MB) before first run.

**Usage:**
```bash
python scripts/download_clip_model.py
```

**Why use this:**
- Avoids download delay on first app launch
- Useful for offline environments (download once, then disconnect)
- Verifies model downloads correctly

### `reset_db.py`
Resets the database by clearing all products, features, and matches.

**Usage:**
```bash
python scripts/reset_db.py
```

**Warning:** This deletes ALL data! Use with caution.

## When to Use These Scripts

### After Installation
```bash
# 1. Verify dependencies
python scripts/verify_dependencies.py

# 2. Pre-download CLIP model (optional)
python scripts/download_clip_model.py

# 3. Start the app
python backend/app.py
```

### Troubleshooting
```bash
# Check dependencies
python scripts/verify_dependencies.py

# Reset database if corrupted
python scripts/reset_db.py
```

### Before Deployment
```bash
# Verify everything is working
python scripts/verify_dependencies.py

# Pre-download model for offline use
python scripts/download_clip_model.py
```

## Related Documentation

- **Installation Guide**: `../INSTALLATION.md`
- **GPU Setup**: `../gpu/GPU_SETUP_GUIDE.md`
- **Quick Reference**: `../QUICK_REFERENCE.md`
