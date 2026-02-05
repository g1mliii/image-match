# GPU Setup Guide

Complete guide for setting up GPU acceleration for the Product Matching System.

## Quick Start

### For AMD GPU Users (Recommended: AMD Adrenaline 26.1.1+)

1. **Install AMD Adrenaline driver with bundled PyTorch:**
   - Download from: https://www.amd.com/en/support/amd-radeon-software
   - Run installer (driver 26.1.1 or later)
   - PyTorch is automatically installed

2. **Install Python 3.12:**
   ```bash
   py -3.12 --version  # Check if already installed
   ```

3. **Run setup script:**
   ```bash
   py -3.12 setup_gpu.py
   ```

### For All Other GPU Users (NVIDIA/Intel/Apple Silicon)

```bash
python setup_gpu.py
```

### What the Setup Script Does

- Detects your GPU type (AMD/NVIDIA/Apple Silicon/CPU)
- Checks Python version compatibility
- **For AMD with Adrenaline:** Verifies PyTorch is already installed and compatible
- **For others:** Installs PyTorch with appropriate GPU support
- Installs all project dependencies
- Verifies GPU is working properly
- Runs performance benchmark

---

## System Requirements

### AMD GPUs (Windows)

**Requirements:**
- **Python 3.12** (Required - ROCm wheels only available for 3.12)
- **AMD Radeon RX 6000/7000/9000 series** (ROCm compatible)
- **Windows 10/11** or **Windows Server 2022**
- **AMD Adrenaline Driver 26.1.1+** (Bundles PyTorch + ROCm) OR **ROCm HIP SDK 6.4/7.x** (Manual)

**RECOMMENDED Installation Method - AMD Adrenaline (Easiest):**

1. **Install Python 3.12** (if not already installed):
   ```bash
   # Download from https://www.python.org/downloads/
   # Or check if already installed:
   py -3.12 --version
   ```

2. **Install AMD Adrenaline Driver 26.1.1 or later**:
   - Download from: https://www.amd.com/en/support/amd-radeon-software
   - Run installer and follow setup steps
   - **Important:** PyTorch will be automatically bundled and installed with ROCm
   - Restart computer after installation

3. **Run setup script with Python 3.12**:
   ```bash
   py -3.12 setup_gpu.py
   ```
   The script will detect PyTorch is already installed via Adrenaline and verify compatibility.

**What Gets Installed:**
- PyTorch 2.9.1 with ROCm bundled via AMD Adrenaline
- sentence-transformers 2.7.0 (ROCm compatible version, auto-verified)
- All project dependencies

---

**ALTERNATIVE Installation Method - Manual HIP SDK:**

If you prefer not to use AMD Adrenaline, you can install ROCm HIP SDK manually:

1. **Install Python 3.12** (if not already installed):
   ```bash
   py -3.12 --version
   ```

2. **Install ROCm HIP SDK**:
   - Download from: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
   - OR: https://github.com/ROCm/rocm-install-on-windows/releases
   - Choose ROCm 6.4.4 (stable, recommended) or ROCm 7.1.1 (latest)
   - Run installer as Administrator
   - Select components:
     - HIP Core (Required)
     - Libraries (Required)
     - Runtime Compiler (Required)
   - Restart computer after installation

3. **Run setup script with Python 3.12**:
   ```bash
   py -3.12 setup_gpu.py
   ```
   The script will install PyTorch with manual ROCm wheels.

**What Gets Installed (Manual Method):**
- PyTorch 2.8.0 with ROCm 6.4.4 support (~780MB)
- sentence-transformers 2.7.0 (ROCm compatible version)
- All project dependencies

**Performance:**
- **Batch Processing**: ~193 images/sec (AMD Radeon RX 9070 XT)
- **Single Image**: ~19 images/sec
- **Speedup**: 10x faster than sequential processing

---

### NVIDIA GPUs (Windows/Linux)

**Requirements:**
- **Python 3.8+** (Any recent version works)
- **NVIDIA GPU** with CUDA support
- **NVIDIA Drivers** (Usually already installed)

**Installation Steps:**

1. **Run setup script**:
   ```bash
   python setup_gpu.py
   ```

2. **If drivers missing**, download from:
   - https://www.nvidia.com/download/index.aspx

**What Gets Installed:**
- PyTorch with CUDA 12.4 support
- sentence-transformers (latest version)
- All project dependencies

**Performance:**
- **Expected**: 100-300 images/sec (depending on GPU)

---

### Intel GPUs (Arc/Iris Xe/UHD Graphics)

**Requirements:**
- **Python 3.8+** (Any recent version works)
- **Intel Arc** (discrete GPU) or **Intel Iris Xe/UHD** (11th gen+ integrated)
- **Windows 10/11** or **Linux**
- **Intel Graphics Drivers** (Usually already installed)

**Installation Steps:**

1. **Run setup script**:
   ```bash
   python setup_gpu.py
   ```
   
   The script will automatically detect Intel GPU and install Intel Extension.

2. **Manual installation** (if needed):
   ```bash
   pip install intel-extension-for-pytorch
   ```

**What Gets Installed:**
- PyTorch (CPU version)
- Intel Extension for PyTorch (enables GPU acceleration)

**Performance:**
- **Intel Arc A770**: 60-80 images/sec
- **Intel Iris Xe**: 30-50 images/sec  
- **Intel UHD (11th gen+)**: 20-40 images/sec
- **Speedup**: 3-5x faster than CPU only

**Note:** Intel Extension is optional. If not installed, system automatically falls back to CPU mode.

---

### Apple Silicon (M1/M2/M3/M4/M5)

**Requirements:**
- **Python 3.8+** (Any recent version works)
- **macOS** with Apple Silicon chip
- **No additional drivers needed!**

**Installation Steps:**

1. **Run setup script**:
   ```bash
   python setup_gpu.py
   ```

**What Gets Installed:**
- PyTorch with MPS (Metal Performance Shaders) support
- sentence-transformers (latest version)
- All project dependencies

**Performance:**
- **Expected**: 50-150 images/sec (depending on chip)

---

### CPU Mode (Fallback)

**Requirements:**
- **Python 3.8+**
- **No GPU required**

**Installation Steps:**

1. **Run setup script**:
   ```bash
   python setup_gpu.py
   ```

**Performance:**
- **Expected**: 5-20 images/sec (depending on CPU)
- Still works great for small catalogs!

---

## Verification

### Check GPU Detection

```bash
python check_gpu.py
```

**Expected Output (AMD GPU):**
```
PyTorch version: 2.8.0a0+gitfc14c65
CUDA available: True
Device count: 1
Device name: AMD Radeon RX 9070 XT
Device memory: 15.92 GB
```

**Expected Output (NVIDIA GPU):**
```
PyTorch version: 2.7.0+cu124
CUDA available: True
Device count: 1
Device name: NVIDIA GeForce RTX 4090
Device memory: 24.00 GB
```

**Expected Output (Apple Silicon):**
```
PyTorch version: 2.7.0
MPS available: True
GPU: Apple Silicon
```

**Expected Output (CPU):**
```
PyTorch version: 2.7.0
CUDA available: False
No GPU detected
```

---

### Run Benchmark

```bash
python benchmark_gpu.py
```

**Sample Output:**
```
================================================================================
GPU Benchmark - CLIP Embeddings
================================================================================

Current Device: cuda
GPU: AMD Radeon RX 9070 XT
VRAM: 15.92 GB
ROCm Version: 6.4.50101-9a6572ae7

Single Image:
  Time: 0.053s
  Throughput: 18.9 images/sec

Batch Processing:
  Time per image: 0.005s
  Throughput: 193.0 images/sec
  Speedup vs single: 10.2x

EXCELLENT (High-end GPU)
```

---

### Run Tests

```bash
# All GPU tests
python -m pytest backend/tests/test_clip.py backend/tests/test_gpu_support.py backend/tests/test_amd_gpu.py backend/tests/test_clip_realworld.py -v

# Quick test
python -m pytest backend/tests/test_clip.py::TestGPUDetection::test_detect_device -v
```

**Expected:** All tests should pass

---

## Troubleshooting

### AMD GPU Not Detected

**Problem:** `CUDA available: False` even after installing ROCm/Adrenaline

**Solutions:**

1. **Check Python version** (Required: 3.12.x):
   ```bash
   python --version  # Must be 3.12.x
   ```
   If not 3.12, run: `py -3.12 setup_gpu.py`

2. **Verify AMD Adrenaline/ROCm installation**:
   ```bash
   dir "C:\Program Files\AMD\ROCm"
   ```
   Should show ROCm version (6.4.x or 7.x)

   OR check AMD Adrenaline:
   - Open AMD Adrenaline app
   - Check that driver version is 26.1.1 or later
   - PyTorch should be listed as installed

3. **Restart computer** after installing AMD Adrenaline or ROCm HIP SDK
   - System needs to reload GPU drivers and HIP runtime

4. **Verify PyTorch installation**:
   ```bash
   py -3.12 -m pip show torch
   ```
   Should show `rocmsdk` or `rocm` in version string

5. **Reinstall if needed**:
   ```bash
   py -3.12 -m pip uninstall torch torchvision torchaudio -y
   py -3.12 setup_gpu.py
   ```

6. **For AMD Adrenaline users**: Ensure driver 26.1.1+ is installed
   - Download from: https://www.amd.com/en/support/amd-radeon-software
   - PyTorch is automatically bundled

---

### NVIDIA GPU Not Detected

**Problem:** `CUDA available: False`

**Solutions:**
1. **Check drivers**:
   ```bash
   nvidia-smi
   ```
   Should show GPU info

2. **Update drivers** from: https://www.nvidia.com/download/index.aspx

3. **Reinstall PyTorch**:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   python setup_gpu.py
   ```

---

### Slow Performance

**Problem:** GPU detected but performance is slow

**Solutions:**
1. **Check GPU usage** during processing
2. **Update drivers** to latest version
3. **Check thermal throttling** (GPU temperature)
4. **Close background applications** using GPU
5. **For AMD**: ROCm on Windows may have optimization issues - this is normal

---

### sentence-transformers Error

**Problem:** `AttributeError: module 'torch.distributed' has no attribute 'is_initialized'`

**Solution:** Downgrade sentence-transformers (AMD ROCm only):
```bash
py -3.12 -m pip install "sentence-transformers<3.0.0"
```

This is automatically done by `setup_gpu.py` for AMD GPUs.

---

## Performance Comparison

| GPU Type | Throughput | Rating | Notes |
|----------|-----------|--------|-------|
| **AMD RX 9070 XT** | 193 img/s | Excellent | ROCm 6.4, Python 3.12 |
| **NVIDIA RTX 4090** | 250-300 img/s | Excellent | CUDA 12.4 |
| **NVIDIA RTX 3080** | 150-200 img/s | Excellent | CUDA 12.4 |
| **Apple M3 Max** | 100-150 img/s | Very Good | MPS |
| **Apple M2** | 50-100 img/s | Very Good | MPS |
| **AMD RX 6700 XT** | 80-120 img/s | Very Good | ROCm 6.4 |
| **NVIDIA RTX 3060** | 80-120 img/s | Very Good | CUDA 12.4 |
| **Intel Arc A770** | 60-80 img/s | Good | Intel Extension |
| **Intel Iris Xe** | 30-50 img/s | Good | Intel Extension |
| **CPU (Ryzen 9)** | 10-20 img/s | Acceptable | No GPU |
| **CPU (Intel i7)** | 5-15 img/s | Slow | No GPU |

---

## Technical Details

### AMD ROCm on Windows

**Installation Methods:**

**1. AMD Adrenaline (Recommended - New in 26.1.1+)**
- Download from: https://www.amd.com/en/support/amd-radeon-software
- PyTorch automatically bundled and installed with ROCm
- Easier updates and simpler setup process
- Automatic dependency management

**2. Manual HIP SDK Installation**
- Download from: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
- Requires separate PyTorch installation via setup script
- More control over specific versions

**Why Python 3.12?**
- AMD only provides PyTorch wheels for Python 3.12 on Windows
- Python 3.13+ not yet supported (as of ROCm 7.1)
- Consistent across all platforms (NVIDIA, Apple Silicon, CPU)

**Why sentence-transformers < 3.0.0?**
- Newer versions require `torch.distributed.is_initialized()`
- This function is missing in AMD's ROCm build for Windows
- Version 2.7.0 works perfectly with ROCm
- Setup script automatically verifies and installs compatible version

**DLLs Installed (via ROCm):**
- `amdhip64_6.dll` or `amdhip64_7.dll` - HIP runtime
- `amd_comgr_2.dll` - Code object manager
- `hiprt0200564.dll` - Ray tracing (optional)

**Compatibility:**
- ROCm 6.x is NOT backward-compatible with 7.x
- Supported GPUs: RX 6000/7000/9000 series
- Requires Windows 10 1903+ or Windows 11
- AMD Adrenaline driver 26.1.1 or later for bundled PyTorch

---

### NVIDIA CUDA

**CUDA Version:** 12.4 (included in PyTorch)

**No separate CUDA installation needed!** PyTorch includes CUDA libraries.

---

### Apple MPS

**MPS (Metal Performance Shaders)** is built into macOS.

**No drivers needed!** Works out of the box on Apple Silicon.

---

## FAQ

**Q: Do I need a GPU?**
A: No! CPU mode works great for small catalogs (< 1000 products).

**Q: Which GPU is best?**
A: Any modern GPU works. NVIDIA has best support, AMD works well with ROCm 6.4, Apple Silicon is excellent, Intel GPUs provide good speedup for office computers.

**Q: Can I use AMD GPU on Linux?**
A: Yes! ROCm has better Linux support. Use: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`

**Q: Why is my AMD GPU slower than expected?**
A: ROCm on Windows is still maturing. Performance is good but may not match NVIDIA CUDA yet.

**Q: Can I switch between CPU and GPU?**
A: Yes! The application automatically detects and uses the best available device.

**Q: How much VRAM do I need?**
A: 4GB minimum, 8GB+ recommended for large batches.

---

## Support

**Issues?** Check:
1. Python version (3.12 for AMD, 3.8+ for others)
2. Drivers installed and up to date
3. Computer restarted after driver installation
4. Run: `python check_gpu.py` to verify detection
5. Run: `python benchmark_gpu.py` to test performance

**Still stuck?** Open an issue with:
- Output of `python check_gpu.py`
- Output of `python setup_gpu.py`
- Your GPU model and OS version
