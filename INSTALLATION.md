# Installation Guide

### Step 1: Install Python

**AMD GPU Users (Windows):**
- **MUST use Python 3.12** (ROCm requirement)
- Download from: https://www.python.org/downloads/release/python-3120/

**NVIDIA GPU / Apple Silicon / CPU Users:**
- Python 3.8 or later works
- Download from: https://www.python.org/downloads/

### Step 2: Install Base Requirements

```bash
pip install -r requirements.txt
```

This installs Flask, OpenCV, NumPy, and other core dependencies.

### Step 3: Install GPU Support (Optional but Recommended)

**AMD GPU (Windows with Python 3.12):**
```bash
py -3.12 gpu/setup_gpu.py
```

**NVIDIA GPU / Apple Silicon:**
```bash
python gpu/setup_gpu.py
```

**What this does:**
- Detects your GPU automatically
- Installs correct PyTorch version:
  - AMD: PyTorch 2.8.0 + ROCm 6.4 (~780MB)
  - NVIDIA: PyTorch 2.x + CUDA 12.4
  - Apple Silicon: PyTorch 2.x + MPS
- Installs sentence-transformers < 3.0.0 (AMD compatibility)
- Verifies installation and runs benchmark

### Step 4: Start the Server

**Windows:**
```bash
start_server.bat
```

**macOS/Linux:**
```bash
./start_server.sh
```

Or manually:
```bash
cd backend
python app.py
```

For AMD GPU users with Python 3.12:
```bash
cd backend
py -3.12 app.py
```

### Step 5: Open Browser

Navigate to: 
- **macOS:** http://127.0.0.1:5001 (port 5001 to avoid AirPlay Receiver conflict)
- **Windows/Linux:** http://127.0.0.1:5000

Check the GPU status indicator in the top right:
- **AMD GPU Active** - ROCm working
- **NVIDIA GPU Active** - CUDA working
- **Apple Silicon Active** - MPS working
- **CPU Mode** - No GPU detected

---

## Detailed Installation

### Requirements by Platform

#### AMD GPU (Windows)

**Requirements:**
- Python 3.12 (exactly - not 3.11 or 3.13)
- AMD Radeon RX 6000/7000/9000 series
- ROCm HIP SDK 6.4
- Windows 10/11 or Windows Server 2022

**Installation:**
1. Install Python 3.12
2. Install ROCm HIP SDK 6.4 from: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
3. Restart computer
4. Run: `py -3.12 gpu/setup_gpu.py`

**What gets installed:**
- PyTorch 2.8.0 with ROCm 6.4
- sentence-transformers 2.7.0 (< 3.0.0 for ROCm compatibility)
- All dependencies

**Why Python 3.12?**
AMD only provides PyTorch ROCm wheels for Python 3.12 on Windows.

**Why sentence-transformers < 3.0.0?**
Version 3.0+ requires `torch.distributed.is_initialized()` which is missing in AMD's ROCm build for Windows.

#### NVIDIA GPU (Windows/Linux)

**Requirements:**
- Python 3.8 or later
- NVIDIA GPU with CUDA support
- NVIDIA drivers (usually pre-installed)

**Installation:**
1. Install Python 3.8+
2. Run: `python gpu/setup_gpu.py`

**What gets installed:**
- PyTorch 2.x with CUDA 12.4
- sentence-transformers (latest compatible)
- All dependencies

#### Apple Silicon (macOS)

**Requirements:**
- Python 3.12 recommended (3.8+ works)
- M1/M2/M3/M4/M5 chip
- macOS 10.15 or later

**Installation:**
1. Install Python 3.12 from python.org (recommended) or `brew install python@3.12`
2. Run: `pip install -r requirements.txt`

**What gets installed:**
- PyTorch 2.x with MPS (Metal Performance Shaders)
- sentence-transformers (latest compatible)
- All dependencies

**macOS-Specific Notes:**
- Port 5001 used by default (port 5000 conflicts with AirPlay Receiver)
- GPU acceleration works automatically - no drivers needed
- Expected performance: 50-150 images/sec
- Use python.org Python to avoid Homebrew's virtual environment requirement

#### CPU Only (Any Platform)

**Requirements:**
- Python 3.8 or later

**Installation:**
1. Install Python 3.8+
2. Run: `pip install -r requirements.txt`

This installs CPU-only versions of all dependencies.

---

## Verification

### Check GPU Detection

```bash
python gpu/check_gpu.py
```

**Expected output (AMD GPU):**
```
PyTorch version: 2.8.0a0+gitfc14c65
CUDA available: True
Device count: 1
Device name: AMD Radeon RX 9070 XT
Device memory: 15.92 GB
```

**Expected output (NVIDIA GPU):**
```
PyTorch version: 2.7.0+cu124
CUDA available: True
Device count: 1
Device name: NVIDIA GeForce RTX 3060
Device memory: 12.00 GB
```

**Expected output (Apple Silicon):**
```
PyTorch version: 2.7.0
MPS available: True
GPU: Apple Silicon
```

**Expected output (CPU):**
```
PyTorch version: 2.7.0
CUDA available: False
No GPU detected
```

### Run Benchmark

```bash
python gpu/benchmark_gpu.py
```

This tests CLIP performance and shows throughput (images/sec).

---

## Troubleshooting

### AMD GPU Not Detected

**Problem:** GPU shows as CPU mode even though you have AMD GPU

**Solutions:**
1. Verify Python version: `python --version` (must be 3.12.x)
2. Check ROCm installation: `dir "C:\Program Files\AMD\ROCm"`
3. Restart computer after ROCm installation
4. Run: `py -3.12 gpu/setup_gpu.py`
5. Verify: `py -3.12 gpu/check_gpu.py`

### Wrong Python Version

**Problem:** Server starts with wrong Python version

**Solution:** Use the startup scripts:
- Windows: `start_server.bat` (auto-detects GPU and uses correct Python)
- macOS/Linux: `./start_server.sh`

Or manually specify Python version:
```bash
cd backend
py -3.12 app.py  # For AMD GPU
```

### sentence-transformers Version Conflict

**Problem:** Error about `torch.distributed.is_initialized()`

**Solution:** Downgrade sentence-transformers:
```bash
pip install "sentence-transformers<3.0.0"
```

This is automatically done by `gpu/setup_gpu.py` for AMD users.

### CLIP Model Download Fails

**Problem:** Model download times out or fails

**Solutions:**
1. Check internet connection
2. Ensure 1GB free disk space
3. Try manual download:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('clip-ViT-B-32')
   ```
4. Model is cached in: `~/.cache/torch/sentence_transformers/`

### Port 5000 Already in Use

**Problem:** "Address already in use" error

**Solutions:**

**macOS Users:**
- The app automatically uses port 5001 to avoid AirPlay Receiver conflict
- Open http://127.0.0.1:5001 instead
- To use port 5000: System Settings → General → AirDrop & Handoff → Turn off "AirPlay Receiver"

**Windows/Linux Users:**
1. Stop other Flask apps
2. Change port in `backend/app.py`:
   ```python
   app.run(debug=True, port=5001)  # Use different port
   ```

### Homebrew Python Issues (macOS)

**Problem:** `error: externally-managed-environment` when installing packages

**Solution:** Use Python from python.org instead of Homebrew, OR use `--break-system-packages` flag:
```bash
pip install -r requirements.txt --break-system-packages
```

**Recommended:** Download Python 3.12 from python.org to avoid this issue entirely.

---

## Package Structure

```
product-matching-system/
├── requirements.txt              # Main requirements (CPU-only)
├── start_server.bat             # Windows startup script
├── start_server.sh              # macOS/Linux startup script
├── INSTALLATION.md              # This file
├── START_SERVER_README.md       # Server startup guide
│
├── backend/
│   ├── app.py                   # Flask server
│   ├── requirements.txt         # Backend requirements
│   ├── image_processing_clip.py # CLIP integration
│   └── ...
│
├── gpu/
│   ├── setup_gpu.py            # GPU setup script
│   ├── check_gpu.py            # GPU detection test
│   ├── benchmark_gpu.py        # Performance benchmark
│   ├── requirements_gpu.txt    # GPU-specific requirements
│   ├── GPU_SETUP_GUIDE.md      # Detailed GPU guide
│   └── ...
│
└── docs/
    ├── CLIP_USER_GUIDE.md      # User guide for CLIP
    └── ...
```

---

## Development Setup

For development with hot-reload:

```bash
cd backend
python app.py
```

Flask runs in debug mode by default, so changes to Python files will auto-reload.

For frontend changes (HTML/CSS/JS), just refresh the browser.

---

## Production Deployment

For production, use a proper WSGI server:

```bash
pip install gunicorn
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Or use the packaged executables (coming soon):
- Windows: `CatalogMatch-Windows-1.0.0.exe`
- macOS: `CatalogMatch-macOS-1.0.0.dmg`

---

## Support

**Documentation:**
- GPU Setup: `gpu/GPU_SETUP_GUIDE.md`
- CLIP Guide: `docs/CLIP_USER_GUIDE.md`
- Developer Guide: `backend/docs/CLIP_DEVELOPER_GUIDE.md`

**Issues:**
- GitHub: https://github.com/g1mliii/image-match/issues

**Quick Help:**
```bash
python gpu/check_gpu.py      # Check GPU detection
python gpu/benchmark_gpu.py  # Test performance
python gpu/verify_setup.py   # Complete verification
```
