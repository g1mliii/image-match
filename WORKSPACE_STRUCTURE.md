# Workspace Structure

## Overview

This document describes the organization of the Product Matching System codebase.

## Directory Structure

```
image-match/
├── backend/                    # Backend application code
│   ├── tests/                 # All test files
│   │   ├── test_clip.py              # CLIP/GPU tests (33 tests)
│   │   ├── test_gpu_support.py       # GPU platform tests (3 tests)
│   │   ├── test_amd_gpu.py           # AMD-specific tests (7 tests)
│   │   ├── test_clip_realworld.py    # Real-world scenario tests (12 tests)
│   │   └── ...                       # Other feature tests
│   ├── static/                # Frontend files (HTML/CSS/JS)
│   ├── uploads/               # Uploaded images
│   ├── app.py                 # Flask application
│   ├── database.py            # Database layer
│   ├── image_processing.py    # Legacy feature extraction
│   ├── image_processing_clip.py  # CLIP embeddings (GPU-accelerated)
│   ├── product_matching.py    # Matching logic
│   ├── similarity.py          # Similarity computation
│   └── requirements.txt       # Backend dependencies
│
├── gpu/                        # GPU acceleration setup & tools
│   ├── setup_gpu.py           # Automated GPU setup script
│   ├── check_gpu.py           # Quick GPU detection
│   ├── benchmark_gpu.py       # Performance benchmark
│   ├── verify_setup.py        # Complete verification suite
│   ├── GPU_SETUP_GUIDE.md     # Comprehensive setup guide
│   ├── GPU_STATUS.md          # Current GPU support status
│   ├── requirements_gpu.txt   # GPU-specific dependencies
│   └── README.md              # GPU folder overview
│
├── docs/                       # GitHub Pages website
│   ├── index.html             # Landing page
│   ├── download.html          # Download page
│   ├── pricing.html           # Pricing page
│   ├── docs.html              # Documentation
│   └── ...                    # Other website files
│
├── sample-data/               # Sample images for testing
│
├── main.py                    # Desktop application launcher
├── requirements.txt           # Main dependencies
├── pytest.ini                 # Pytest configuration
├── README.md                  # Main project README
└── ...                        # Other config files
```

## Key Files

### Application Entry Points
- **`main.py`** - Desktop application launcher (pywebview)
- **`backend/app.py`** - Flask backend server

### GPU Acceleration
- **`gpu/setup_gpu.py`** - One-command GPU setup for all platforms
- **`gpu/check_gpu.py`** - Quick GPU detection check
- **`gpu/benchmark_gpu.py`** - Performance benchmark (CPU vs GPU)
- **`gpu/verify_setup.py`** - Complete verification (runs all tests)

### Core Functionality
- **`backend/image_processing_clip.py`** - CLIP embeddings (GPU-accelerated, cross-platform)
- **`backend/product_matching.py`** - Matching logic (CLIP visual + optional metadata/category)
- **`backend/similarity.py`** - Similarity computation (cosine similarity on CLIP embeddings)
- **`backend/database.py`** - SQLite database layer

### Testing
- **`backend/tests/test_clip.py`** - CLIP & GPU tests (33 tests)
- **`backend/tests/test_gpu_support.py`** - Platform detection (3 tests)
- **`backend/tests/test_amd_gpu.py`** - AMD-specific tests (7 tests)
- **`backend/tests/test_clip_realworld.py`** - Real-world scenarios (12 tests)

## File Organization Principles

### 1. GPU Files → `gpu/` Folder
All GPU-related setup, testing, and documentation is in the `gpu/` folder:
- Setup scripts
- Verification tools
- Benchmarks
- Documentation

**Why?** Keeps GPU acceleration separate from core application logic.

### 2. Tests → `backend/tests/` Folder
All test files are in `backend/tests/`:
- Unit tests
- Integration tests
- Performance tests
- Real-world scenario tests

**Why?** Standard Python project structure, easy to run with pytest.

### 3. Documentation → Multiple Locations
- **Root**: Main README, quick reference
- **`gpu/`**: GPU-specific documentation
- **`docs/`**: GitHub Pages website

**Why?** Documentation close to relevant code, website separate.

### 4. Backend → `backend/` Folder
All backend code in one place:
- Flask application
- Image processing
- Database layer
- Static files (HTML/CSS/JS)

**Why?** Clear separation of backend and frontend concerns.

## Running the Application

### Development Mode
```bash
# Start backend server
python backend/app.py

# Or use desktop launcher
python main.py
```

### GPU Setup
```bash
cd gpu
python setup_gpu.py
```

### Testing
```bash
# All tests
pytest

# GPU tests only
pytest backend/tests/test_clip.py backend/tests/test_gpu_support.py backend/tests/test_amd_gpu.py

# Quick verification
cd gpu
python verify_setup.py
```

### Benchmarking
```bash
cd gpu
python benchmark_gpu.py
```

## Dependencies

### Main Dependencies (`requirements.txt`)
- Flask - Web framework
- OpenCV - Image processing
- NumPy - Numerical computing
- Pillow - Image handling
- pywebview - Desktop wrapper

### Backend Dependencies (`backend/requirements.txt`)
- Flask-CORS - CORS support
- scikit-image - Advanced image processing
- scipy - Scientific computing
- sentence-transformers - CLIP models

### GPU Dependencies (`gpu/requirements_gpu.txt`)
- torch - PyTorch (GPU-accelerated)
- torchvision - Computer vision models
- torchaudio - Audio processing (dependency)

## Platform-Specific Notes

### Windows
- Main development platform
- AMD GPU support via ROCm 6.4 (Python 3.12 required)
- NVIDIA GPU support via CUDA 12.4
- Desktop app uses pywebview

### macOS
- Apple Silicon (M1/M2/M3/M4/M5) supported via MPS
- No additional GPU setup needed
- Desktop app uses pywebview

### Linux
- AMD GPU support via ROCm 6.2+
- NVIDIA GPU support via CUDA 12.4
- Better ROCm support than Windows

## Recent Changes

### 2025-11-23: GPU Folder Organization
- Created `gpu/` folder for all GPU-related files
- Moved setup scripts, benchmarks, and documentation
- Updated paths in scripts to work from new location
- Fixed deprecation warnings in PyTorch autocast API

### 2025-11-23: AMD GPU Support
- Added full AMD Radeon RX 9070 XT support
- Implemented Python 3.12 requirement check
- Added sentence-transformers < 3.0.0 compatibility
- Documented ROCm HIP SDK installation process
- All 55 GPU tests passing

### 2025-11-23: Comprehensive Testing
- CLIP visual matching: ✅ Working
- Category filtering: ✅ Working
- GPU acceleration: ✅ Working (183.7 img/s on AMD RX 9070 XT)
- CPU fallback: ✅ Working
- Batch processing: ✅ Working (10.5x speedup)
- Cross-platform GPU support: ✅ AMD, NVIDIA, Intel, Apple Silicon

## Contributing

See `CONTRIBUTING.md` for contribution guidelines.

## License

See `LICENSE` for license information.
