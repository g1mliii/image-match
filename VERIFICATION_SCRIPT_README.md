# Dependency Verification Script

**File:** `verify_dependencies.py`  
**Purpose:** Verify and auto-fix critical dependency issues

---

## 🎯 What It Checks

### 1. ✅ Python Version (CRITICAL)
- **Requirement:** Python 3.12.x
- **Why:** Required for AMD ROCm GPU support on Windows
- **Impact:** Application won't work with other versions
- **Auto-fix:** No (must install Python 3.12 manually)

### 2. ✅ sentence-transformers Version (CRITICAL)
- **Requirement:** < 3.0.0 (must be 2.7.0 - 2.99.x)
- **Why:** Version 3.0+ breaks AMD ROCm GPU support
- **Impact:** AMD GPU users will get runtime errors
- **Auto-fix:** Yes (automatically downgrades if needed)

### 3. ✅ Flask and Web Framework
- **Checks:** Flask, Flask-CORS, Werkzeug
- **Why:** Required for web application
- **Impact:** App won't start without these
- **Auto-fix:** Yes (installs if missing)

### 4. ✅ All Dependencies
- **Checks:** NumPy, SciPy, OpenCV, scikit-image, Pillow, PyTorch, pytest
- **Why:** Required for image processing and testing
- **Impact:** Various features won't work
- **Auto-fix:** Yes (installs if missing)

### 5. ✅ AMD ROCm Compatibility (Windows Only)
- **Checks:** Python 3.12 + AMD GPU + PyTorch ROCm
- **Why:** AMD GPUs require specific setup on Windows
- **Impact:** AMD GPU won't work without correct setup
- **Auto-fix:** Partial (guides user to run gpu/setup_gpu.py)

### 6. ✅ GPU Support
- **Checks:** NVIDIA CUDA, AMD ROCm, Apple Silicon MPS
- **Why:** Verifies GPU acceleration is available
- **Impact:** Performance (22-55x slower without GPU)
- **Auto-fix:** No (guides user to run gpu/setup_gpu.py)

---

## 🚀 How to Use

### Basic Usage
```bash
# After installing requirements
pip install -r requirements.txt

# Run verification
python verify_dependencies.py
```

### With Python 3.12 (Recommended)
```bash
# Windows
py -3.12 verify_dependencies.py

# macOS/Linux
python3.12 verify_dependencies.py
```

### Auto-Fix Mode
```bash
# Run and answer 'y' when prompted
python verify_dependencies.py
# When asked: "Would you like to automatically fix these issues? (y/n):"
# Type: y
```

---

## 📊 Example Output

### All Checks Pass ✅
```
================================================================================
Dependency Verification and Auto-Fix
================================================================================

STEP 1: Python Version Check
================================================================================
  Current: Python 3.12.2
  Required: Python 3.12.x
  ✓ PASS: Python 3.12.2 detected

STEP 2: sentence-transformers Version Check (CRITICAL)
================================================================================
  Installed: sentence-transformers 2.7.0
  Required: < 3.0.0
  ✓ PASS: sentence-transformers 2.7.0 is < 3.0.0
  ✓ Compatible with AMD ROCm

STEP 3: All Dependencies Check
================================================================================
  ✓ Flask                        3.0.0
  ✓ Flask-CORS                   4.0.0
  ✓ Werkzeug                     3.0.0
  ✓ NumPy                        1.26.2
  ✓ SciPy                        1.11.3
  ✓ OpenCV                       4.8.1.78
  ✓ scikit-image                 0.22.0
  ✓ Pillow                       10.1.0
  ✓ PyTorch                      2.0.0
  ✓ sentence-transformers        2.7.0 (>=2.7.0, <3.0.0)
  ✓ pytest                       7.0.0

[SUCCESS] All dependencies are correctly installed!

STEP 5: GPU Support Verification
================================================================================
  PyTorch version: 2.9.1
  CUDA available: False
  MPS available: True
  GPU: Apple Silicon (MPS)

  ✓ GPU acceleration available

================================================================================
Verification Summary
================================================================================

  Python 3.12: ✓ PASS
  sentence-transformers < 3.0.0: ✓ PASS
  Flask: ✓ PASS
  GPU Support: ✓ AVAILABLE

✅ All checks passed!

Next steps:
  1. For GPU acceleration: python gpu/setup_gpu.py
  2. Run tests: python backend/tests/run_comprehensive_gpu_tests.py
  3. Start app: python backend/app.py
```

### Critical Issue Found ❌
```
================================================================================
STEP 2: sentence-transformers Version Check (CRITICAL)
================================================================================
  Installed: sentence-transformers 5.1.2
  Required: < 3.0.0

  ✗ CRITICAL FAIL: sentence-transformers 5.1.2 is >= 3.0.0

  [WHY THIS IS CRITICAL]
    • Version 3.0+ breaks AMD ROCm GPU support on Windows
    • Requires torch.distributed.is_initialized() which is missing in ROCm
    • Will cause runtime errors on AMD GPUs

  [IMPACT]
    • AMD GPU users: Application will crash
    • NVIDIA/Apple Silicon: May work but untested

  [MUST FIX] This will be automatically fixed if you choose 'y'

================================================================================
Issues Found
================================================================================

[CRITICAL ISSUES]
  ✗ sentence-transformers >= 3.0.0 (breaks AMD ROCm)

================================================================================
Would you like to automatically fix these issues? (y/n): y

================================================================================
Auto-Fix in Progress
================================================================================

================================================================================
Fixing sentence-transformers Version
================================================================================
  [CRITICAL] sentence-transformers 5.1.2 is >= 3.0.0
  [ACTION] Downgrading to < 3.0.0 for AMD ROCm compatibility...

  [1/2] Uninstalling sentence-transformers 5.1.2...
  [2/2] Installing sentence-transformers < 3.0.0...
  ✓ Successfully downgraded to sentence-transformers 2.7.0
  ✓ Now compatible with AMD ROCm

[SUCCESS] All issues fixed!
```

---

## 🔧 What Gets Auto-Fixed

### Automatically Fixed
- ✅ sentence-transformers >= 3.0.0 → Downgrades to < 3.0.0
- ✅ Missing Flask → Installs Flask
- ✅ Missing dependencies → Installs them
- ✅ Wrong versions → Reinstalls correct versions

### Requires Manual Action
- ❌ Python version (must install Python 3.12)
- ❌ GPU drivers (must install NVIDIA/AMD drivers)
- ❌ ROCm SDK (must install AMD HIP SDK on Windows)

---

## ⚠️ Critical Warnings

### sentence-transformers Version
**DO NOT upgrade to 3.0+!**

If you accidentally upgrade:
```bash
# This will break AMD ROCm support
pip install sentence-transformers --upgrade  # ❌ DON'T DO THIS

# Fix it with:
python verify_dependencies.py  # Auto-fixes to < 3.0.0
```

### Python Version
**Must use Python 3.12!**

Other versions won't work:
- Python 3.11 or older: AMD ROCm won't work on Windows
- Python 3.13 or newer: Not tested, may have compatibility issues

---

## 📝 Integration with Installation

### Recommended Installation Flow
```bash
# Step 1: Install Python 3.12
# Download from: https://www.python.org/downloads/

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Verify and auto-fix
python verify_dependencies.py
# Answer 'y' to auto-fix any issues

# Step 4: Setup GPU (optional but recommended)
python gpu/setup_gpu.py

# Step 5: Run tests
python backend/tests/run_comprehensive_gpu_tests.py

# Step 6: Start app
python backend/app.py
```

---

## 🎯 When to Run This Script

### Always Run After:
1. ✅ Fresh installation (`pip install -r requirements.txt`)
2. ✅ Upgrading dependencies (`pip install --upgrade`)
3. ✅ Switching Python versions
4. ✅ Before packaging the application
5. ✅ When troubleshooting issues

### Optional But Recommended:
- Before running tests
- Before starting the application
- After system updates
- When setting up on a new machine

---

## 💡 Troubleshooting

### "Python 3.12 is required"
**Solution:** Install Python 3.12 from python.org

### "sentence-transformers >= 3.0.0"
**Solution:** Run the script and answer 'y' to auto-fix

### "PyTorch not installed"
**Solution:** Run `python gpu/setup_gpu.py`

### "GPU not detected"
**Solution:** 
1. Install GPU drivers (NVIDIA/AMD)
2. Run `python gpu/setup_gpu.py`
3. For AMD on Windows: Install HIP SDK

---

## 🔗 Related Files

- `requirements.txt` - Dependency specifications
- `gpu/setup_gpu.py` - GPU setup script
- `backend/tests/run_comprehensive_gpu_tests.py` - Test suite
- `PYTHON_VERSION_DECISION.md` - Why Python 3.12

---

## ✅ Summary

**This script ensures:**
1. Python 3.12 is being used
2. sentence-transformers < 3.0.0 (critical for AMD ROCm)
3. Flask and all dependencies are installed
4. Versions are correct
5. GPU support is available (if applicable)

**Run it after every installation to ensure everything is correct!**
