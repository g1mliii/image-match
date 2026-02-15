# Final Packaging Plan - Single Launcher + 4 GPU Variants

## Summary

Users download **ONE tiny launcher** (~6MB). On first run:
1. Launcher auto-detects GPU using `detect_gpu()` from `gpu/setup_gpu.py`
2. Downloads correct payload from GitHub (app_nvidia.zip, app_amd.zip, app_intel.zip, or app_cpu.zip)
3. Extracts and launches the app
4. Subsequent runs: instant launch

## What Gets Built (4 Payloads)

Each payload is a complete standalone app with:
- Correct PyTorch version for that GPU type
- **ALL dependencies from requirements.txt** (Flask, numpy, opencv, PIL, webview, faiss, sentence-transformers <3.0.0, etc.)
- CLIP model cache bundled

| Payload | PyTorch | Dependencies | Size | Target Users |
|---------|---------|--------------|------|--------------|
| **app_nvidia.zip** | CUDA 12.4 | requirements.txt (all) | ~600MB | NVIDIA GPU users (~70%) |
| **app_amd.zip** | CPU only* | requirements.txt (all, includes sentence-transformers <3.0.0) | ~450MB | AMD GPU users (AMD driver bundles PyTorch) |
| **app_intel.zip** | CPU + intel-extension-for-pytorch | requirements.txt (all) + intel-extension | ~500MB | Intel Arc/Iris/UHD users |
| **app_cpu.zip** | CPU only | requirements.txt (all) | ~400MB | Universal fallback |

*AMD Note: AMD Adrenaline driver (26.1.1+) bundles PyTorch with ROCm. We ship CPU version in app_amd.zip but app will use GPU PyTorch if AMD driver has it installed.

## Key Points

✅ **Python 3.12 is bundled** - PyInstaller includes Python interpreter automatically
✅ **All dependencies bundled** - Everything from requirements.txt:
   - Flask, numpy, scipy, opencv, PIL, Pillow, scikit-image
   - webview, faiss-cpu, psutil, fuzzywuzzy, python-Levenshtein
   - **sentence-transformers <3.0.0** (CRITICAL for AMD compatibility)
✅ **PyTorch is GPU-specific** - Each build uses correct PyTorch version
✅ **CPU fallback** - Each payload can degrade to CPU if GPU unavailable
✅ **GPU auto-detection** - Uses existing `detect_gpu()` function from `gpu/setup_gpu.py`
✅ **CLIP model bundled** - Each payload includes cached model (~350MB)
✅ **intel-extension optional** - Intel build tries to use it, falls back to CPU
✅ **Remote access ready** - ngrok auto-start is supported when ngrok is available

⚠ **ngrok redistribution requires policy check** - if we bundle ngrok binaries, distribution model must follow ngrok Terms

## Summary: What Gets Packaged

| Component | Bundle? | Size | Notes |
|-----------|---------|------|-------|
| **Python 3.12** | ✅ Yes | Included | PyInstaller bundles this automatically |
| **PyTorch** | ✅ Yes (GPU-specific) | 150-800MB per build | Different for each GPU type (NVIDIA/AMD/Intel/CPU) |
| **Dependencies** | ✅ Yes (all) | ~200MB | Flask, numpy, opencv, PIL, scipy, faiss, etc. |
| **CLIP Model** | ✅ Yes (identical) | ~350MB | Same for all 4 builds, no GPU variants |
| **App Code** | ✅ Yes | ~50MB | Backend, frontend, static files |
| **Total per payload** | | **600MB (NVIDIA/Intel) to 400MB (CPU)** | Includes PyTorch + Dependencies + CLIP |

---

## ngrok Remote Access Packaging Plan (Windows + macOS)

### Decision Gate (Required Before Release)

Choose one model and document it in release notes:

1. **External ngrok install (recommended for first release)**
   - Do not bundle ngrok binary.
   - App auto-starts ngrok if `ngrok` is found in PATH.
   - User performs one-time in-app setup: `CONNECT PHONE` -> `SETUP TOKEN`.
2. **Bundled ngrok agent (later release)**
   - Ship ngrok binaries inside app package for each OS.
   - Use bundled binary path first, then fallback to PATH.
   - Requires internal legal/compliance check against ngrok Terms for redistribution model.

### Streamlined User Experience Target

1. Install app
2. Open `CONNECT PHONE`
3. Paste token once and click `SETUP TOKEN`
4. Click `AUTO NGROK` (or app auto-starts on next launch)
5. Share URL + PIN with authorized user

### Platform Packaging Targets

- **Windows**
  - Optional bundled binary path: `third_party/ngrok/windows/ngrok.exe`
  - Launcher output: `ProductMatcher_Setup.exe`
- **macOS**
  - Optional bundled binary path: `third_party/ngrok/macos/ngrok`
  - Launcher output: `ProductMatcher_Setup.app` (or signed `.dmg` installer)
  - Ensure executable bit: `chmod +x third_party/ngrok/macos/ngrok`
  - Ensure codesign/notarization if bundled in distributed app

### Runtime Resolution Order (for bundled mode)

When app starts, resolve ngrok in this order:

1. `NGROK_PATH` environment variable (override)
2. Bundled platform binary inside app payload
3. System PATH (`ngrok`)

### Bundled Mode: PyInstaller Inputs

If bundled mode is chosen, include ngrok binary in platform build inputs:

```python
# Windows build
binaries=[
    ('third_party/ngrok/windows/ngrok.exe', 'third_party/ngrok/windows'),
]

# macOS build
binaries=[
    ('third_party/ngrok/macos/ngrok', 'third_party/ngrok/macos'),
]
```

Runtime should compute bundled path from `sys._MEIPASS` (PyInstaller extraction dir) when available.

### Operational Ports (No Conflict)

- App server: `127.0.0.1:8000`
- ngrok local API: `127.0.0.1:4040`
- Public ngrok URL forwards to app port `8000`

---

## CLIP Model Bundling (Not a PyTorch Issue)

**IMPORTANT:** CLIP model is **NOT the same as PyTorch** - it's a data file, not a compiled package.

- **CLIP is:** Pre-trained model weights (~350MB) downloaded by sentence-transformers
- **CLIP location:** `~/.cache/clip-models/` on user's machine
- **CLIP in packaging:** We **WILL bundle** CLIP in all 4 payloads
- **Why:** CLIP is identical on all systems (NVIDIA/AMD/Intel/CPU) - no GPU-specific variants
- **Result:** All 4 payloads include CLIP, users get instant matching on first run (no download)

**Build process:**
1. Run `python scripts/download_clip_model.py` ONCE (before building any payload)
2. Downloads CLIP to `~/.cache/clip-models/` (~350MB, 1-2 min)
3. PyInstaller includes this cache in each payload:
   ```python
   datas=[
       ('backend/static', 'backend/static'),
       (os.path.expanduser('~/.cache/clip-models'), '.cache/clip-models'),  # CLIP bundled
   ]
   ```
4. Each payload (app_nvidia.zip, app_amd.zip, etc.) will contain CLIP model
5. No download needed on first run - instant matching

**File sizes with CLIP:**
- Without CLIP: Each payload ~250-300MB
- With CLIP: Each payload ~550-600MB
- Total disk usage: ~2GB for all 4 payloads

---

## Phase 1: Build 4 Payloads

You must build these manually on your development machine. Between each build, uninstall the previous PyTorch.

### Build 1: NVIDIA Payload (app_nvidia.zip)

```bash
# 1. Clean environment
pip uninstall torch torchvision torchaudio -y

# 2. Install NVIDIA PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install all other dependencies
pip install -r requirements.txt

# 4. Build with PyInstaller
pyinstaller --onedir --clean product-matcher.spec

# 5. Package
cd dist/
zip -r ../app_nvidia.zip ProductMatcher/
cd ..
```

**Result:** `app_nvidia.zip` (~600MB)

### Build 2: AMD Payload (app_amd.zip)

```bash
# 1. Clean environment
pip uninstall torch torchvision torchaudio -y

# 2. Install CPU PyTorch (AMD driver bundles ROCm version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install ALL dependencies from requirements.txt
# CRITICAL: sentence-transformers <3.0.0 is in requirements.txt
pip install -r requirements.txt

# 4. Build with PyInstaller
pyinstaller --onedir --clean product-matcher.spec

# 5. Package
cd dist/
zip -r ../app_amd.zip ProductMatcher/
cd ..

# 6. Clean up before next build
rm -rf dist/ build/
```

**Result:** `app_amd.zip` (~450MB)

**Note:** App will auto-detect if AMD driver has ROCm PyTorch and use it. Otherwise falls back to CPU.

### Build 3: Intel Payload (app_intel.zip)

```bash
# 1. Clean environment
pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y

# 2. Install CPU PyTorch
pip install torch torchvision torchaudio

# 3. Install Intel Extension (enables Arc/Iris/UHD GPU)
pip install intel-extension-for-pytorch

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Build with PyInstaller (CRITICAL: --collect-all for Intel DLLs)
pyinstaller --onedir --clean --collect-all intel_extension_for_pytorch product-matcher.spec

# 6. Package
cd dist/
zip -r ../app_intel.zip ProductMatcher/
cd ..

# 7. Clean up before next build
rm -rf dist/ build/
```

**Result:** `app_intel.zip` (~450MB)

**Note:** App tries to use Intel extension. If unavailable, falls back to CPU.

### Build 4: CPU Payload (app_cpu.zip)

```bash
# 1. Clean environment
pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y

# 2. Install CPU PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install all other dependencies
pip install -r requirements.txt

# 4. Build with PyInstaller
pyinstaller --onedir --clean product-matcher.spec

# 5. Package
cd dist/
zip -r ../app_cpu.zip ProductMatcher/
cd ..
```

**Result:** `app_cpu.zip` (~400MB)

---

## Phase 2: Create Automated Build Script

**CRITICAL FIRST STEP:** Download CLIP model before building anything!

```bash
# MUST do this first - downloads CLIP once (~350MB, 1-2 minutes)
python scripts/download_clip_model.py
```

This creates `~/.cache/clip-models/` which will be bundled in all 4 payloads.

Then create `build_all_payloads.bat` to automate all 4 builds:

```batch
@echo off
echo Building all 4 GPU variants...
echo This will take 60-90 minutes and create 4 zip files
echo.

REM CRITICAL: CLIP model must be downloaded first!
REM Run: python scripts\download_clip_model.py
if not exist "%USERPROFILE%\.cache\clip-models" (
    echo ERROR: CLIP model not found!
    echo Run first: python scripts\download_clip_model.py
    pause
    exit /b 1
)

REM Clean any previous builds
if exist dist\ rmdir /s /q dist
if exist build\ rmdir /s /q build

REM Build 1: NVIDIA
echo.
echo ========================================
echo Building NVIDIA variant (CUDA 12.4)...
echo ========================================
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo Installing all dependencies from requirements.txt...
pip install -r requirements.txt
echo Building with PyInstaller...
pyinstaller --onedir --clean product-matcher.spec
cd dist
echo Creating zip file...
powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::CreateFromDirectory('ProductMatcher', '..\app_nvidia.zip')"
cd ..
rmdir /s /q dist build

REM Build 2: AMD
echo.
echo ========================================
echo Building AMD variant (CPU fallback)...
echo ========================================
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo Installing all dependencies from requirements.txt...
pip install -r requirements.txt
echo Building with PyInstaller...
pyinstaller --onedir --clean product-matcher.spec
cd dist
echo Creating zip file...
powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::CreateFromDirectory('ProductMatcher', '..\app_amd.zip')"
cd ..
rmdir /s /q dist build

REM Build 3: Intel
echo.
echo ========================================
echo Building Intel variant (Arc/Iris/UHD)...
echo ========================================
pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y
pip install torch torchvision torchaudio
pip install intel-extension-for-pytorch
echo Installing all dependencies from requirements.txt...
pip install -r requirements.txt
echo Building with PyInstaller...
pyinstaller --onedir --clean --collect-all intel_extension_for_pytorch product-matcher.spec
cd dist
echo Creating zip file...
powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::CreateFromDirectory('ProductMatcher', '..\app_intel.zip')"
cd ..
rmdir /s /q dist build

REM Build 4: CPU
echo.
echo ========================================
echo Building CPU variant (universal)...
echo ========================================
pip uninstall torch torchvision torchaudio intel-extension-for-pytorch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo Installing all dependencies from requirements.txt...
pip install -r requirements.txt
echo Building with PyInstaller...
pyinstaller --onedir --clean product-matcher.spec
cd dist
echo Creating zip file...
powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::CreateFromDirectory('ProductMatcher', '..\app_cpu.zip')"
cd ..
rmdir /s /q dist build

echo.
echo ========================================
echo All builds complete!
echo ========================================
echo.
echo Created files:
dir app_*.zip
echo.
echo Next steps:
echo 1. Test each zip file
echo 2. Upload to GitHub Release
echo 3. Build launcher: pyinstaller launcher.py --onefile --name ProductMatcher_Setup
echo 4. Share ProductMatcher_Setup.exe with users
echo.
pause
```

---

## Phase 3: Create Universal Launcher

Create `launcher.py` - the ONLY file users download (~6MB):

```python
import sys
import os
import platform
import subprocess
import urllib.request
import json
import zipfile
import shutil
import time

# --- CONFIGURATION ---
GITHUB_USER = "YourUsername"            # REPLACE WITH YOUR GITHUB USERNAME
GITHUB_REPO = "YourRepoName"            # REPLACE WITH YOUR GITHUB REPO
APP_EXE_NAME = "ProductMatcher.exe"     # Name of exe inside the zip
INSTALL_DIR = "bin"                     # Folder where app is installed
# ---------------------

def get_gpu_type():
    """
    Detects GPU and returns: 'nvidia', 'amd', 'intel', or 'cpu'

    Reuses logic from gpu/setup_gpu.py detect_gpu() function
    """
    system = platform.system()
    print("Detecting GPU...")

    if system == "Windows":
        try:
            # Check Video Controllers using WMI
            cmd = 'wmic path win32_videocontroller get name'
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().lower()

            # Check in order of preference
            if 'nvidia' in result or 'geforce' in result or 'rtx' in result or 'quadro' in result:
                print("✓ NVIDIA GPU detected")
                return 'nvidia'
            elif 'amd' in result or 'radeon' in result:
                print("✓ AMD GPU detected")
                return 'amd'
            elif 'intel' in result and ('arc' in result or 'iris' in result or 'uhd' in result):
                print("✓ Intel Arc/Iris/UHD GPU detected")
                return 'intel'

        except Exception as e:
            print(f"GPU detection failed: {e}")

    elif system == "Darwin":  # macOS
        if platform.machine() == "arm64":
            print("✓ Apple Silicon detected")
            return 'apple'  # Note: Not currently built, would need separate Mac builds

    # Fallback
    print("✓ No dedicated GPU found - using CPU")
    return 'cpu'

def get_latest_release_url(gpu_type):
    """
    Finds the correct zip URL from GitHub releases.
    Looks for app_nvidia.zip, app_amd.zip, etc.
    """
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/releases/latest"
    target_name = f"app_{gpu_type}.zip"

    print(f"Checking GitHub for {target_name}...")

    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'ProductMatcher-Launcher'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())

        # Find the asset matching our GPU type
        for asset in data.get('assets', []):
            if asset['name'] == target_name:
                return asset['browser_download_url']

        print(f"ERROR: Release found, but {target_name} is missing!")
        return None

    except Exception as e:
        print(f"GitHub check failed: {e}")
        return None

def download_file(url, filename):
    """Download file with progress bar"""
    print(f"Downloading ({os.path.getsize(filename) / 1024 / 1024 / 1024:.1f}GB)...")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ProductMatcher-Launcher'})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 8192 * 4
            downloaded = 0

            with open(filename, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)

                    # Progress bar
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        filled = int(40 * percent / 100)
                        bar = '█' * filled + '-' * (40 - filled)
                        sys.stdout.write(f"\r[{bar}] {percent:.1f}%")
                        sys.stdout.flush()

        print("\n✓ Download complete")
        return True

    except Exception as e:
        print(f"\n✗ Download error: {e}")
        return False

def install_update(zip_path):
    """Extract zip to install directory"""
    print("Installing...")

    try:
        # 1. Remove old install if exists
        if os.path.exists(INSTALL_DIR):
            try:
                shutil.rmtree(INSTALL_DIR)
            except:
                print("✗ Error: Could not remove old version. Is the app running?")
                return False

        # 2. Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("temp_extract")

        # 3. Move extracted folder to bin
        # Handle if zip has nested ProductMatcher folder
        source_dir = "temp_extract"
        if os.path.exists("temp_extract/ProductMatcher"):
            source_dir = "temp_extract/ProductMatcher"

        shutil.move(source_dir, INSTALL_DIR)
        shutil.rmtree("temp_extract", ignore_errors=True)
        os.remove(zip_path)

        print("✓ Installation complete")
        return True

    except Exception as e:
        print(f"✗ Installation error: {e}")
        return False

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 50)
    print("ProductMatcher Launcher")
    print("=" * 50)
    print()

    exe_path = os.path.join(INSTALL_DIR, APP_EXE_NAME)

    # Check if already installed
    if os.path.exists(exe_path):
        print(f"✓ App installed at: {INSTALL_DIR}")
        print()
        print("Launching...")

        try:
            if platform.system() == 'Windows':
                # DETACHED_PROCESS (0x00000008) allows launcher to close
                subprocess.Popen([exe_path], creationflags=0x00000008)
            else:
                subprocess.Popen([exe_path])

            time.sleep(1)  # Brief pause to ensure handoff

        except Exception as e:
            print(f"✗ Launch failed: {e}")
            input("Press Enter to exit...")

        return

    # Not installed - download and install
    print("App not installed. Starting first-time setup...")
    print()

    # Detect GPU
    gpu_type = get_gpu_type()
    print()

    # Get download URL from GitHub
    url = get_latest_release_url(gpu_type)
    if not url:
        print()
        print("✗ Could not find download. Check your internet connection.")
        input("Press Enter to exit...")
        return

    # Download
    print()
    zip_name = "update_pkg.zip"
    if not download_file(url, zip_name):
        print()
        print("✗ Download failed.")
        input("Press Enter to exit...")
        return

    # Install
    print()
    if not install_update(zip_name):
        print()
        print("✗ Installation failed.")
        input("Press Enter to exit...")
        return

    # Launch
    print()
    print("Launching ProductMatcher...")
    try:
        if platform.system() == 'Windows':
            subprocess.Popen([exe_path], creationflags=0x00000008)
        else:
            subprocess.Popen([exe_path])

        time.sleep(1)

    except Exception as e:
        print(f"✗ Launch failed: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
```

---

### Phase 3A: Cross-Platform Launcher Outputs

- **Windows release artifact:** `ProductMatcher_Setup.exe`
- **macOS release artifact:** `ProductMatcher_Setup.app` (preferred) or `ProductMatcher_Setup.dmg`
- Both launchers should preserve same first-run flow:
  - detect environment
  - install/unpack payload
  - run app
  - support ngrok auto-start behavior

---

## Phase 4: Build Launcher Executable

```bash
# Install PyInstaller if needed
pip install pyinstaller

# Build launcher (only needs Python standard library)
pyinstaller launcher.py --onefile --name ProductMatcher_Setup --clean

# Result: dist/ProductMatcher_Setup.exe (~6MB)
```

---

## Phase 5: Upload to GitHub

1. **Create GitHub Release:**
   - Tag: `v1.0.0` (or your version)
   - Title: "Product Matcher v1.0"

2. **Upload files:**
   - `app_nvidia.zip` (~600MB)
   - `app_amd.zip` (~400MB)
   - `app_intel.zip` (~450MB)
   - `app_cpu.zip` (~400MB)

3. **Share launcher:**
   - Users download: `ProductMatcher_Setup.exe` (~6MB)

---

## How It Works for Users

1. **Download** `ProductMatcher_Setup.exe` (6MB)
2. **Run** `ProductMatcher_Setup.exe`
3. Launcher **detects GPU** (NVIDIA/AMD/Intel/CPU)
4. Launcher **downloads** correct app from GitHub (~400-600MB)
5. Launcher **extracts** app to `bin/` directory
6. Launcher **launches** ProductMatcher.exe
7. **Subsequent runs:** Direct launch (app already installed)

---

## PyInstaller Spec File (product-matcher.spec)

```python
# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

home_dir = str(Path.home())
CLIP_CACHE_PATH = os.path.join(home_dir, '.cache', 'clip-models')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('backend/static', 'backend/static'),
        ('first_run_setup.py', '.'),  # Optional: GPU setup on first run
        # CRITICAL: Bundle CLIP model (~350MB) - downloaded at ~/.cache/clip-models/
        # Run first: python scripts/download_clip_model.py
        # This ensures instant matching on first app run (no model download needed)
        (os.path.expanduser('~/.cache/clip-models'), '.cache/clip-models'),
    ],
    hiddenimports=[
        # Backend modules
        'backend.app',
        'backend.database',
        'backend.config',
        'backend.similarity',
        'backend.image_processing',
        'backend.image_processing_clip',
        'backend.product_matching',
        'backend.hybrid_matching',
        'backend.feature_extraction_service',
        'backend.feature_cache',
        'backend.faiss_index',
        'backend.snapshot_manager',
        'backend.matching_utils',
        'backend.validation_utils',

        # Web Framework (from requirements.txt)
        'flask',
        'flask_cors',
        'werkzeug',
        'urllib3',

        # Core Scientific (from requirements.txt)
        'numpy',
        'scipy',
        'scipy.sparse',
        'scipy.spatial',

        # Image Processing (from requirements.txt)
        'cv2',
        'skimage',
        'PIL',
        'PIL.Image',

        # Deep Learning (from requirements.txt)
        'torch',
        'torch._C',
        'torch.distributed',
        'torchvision',
        'torchaudio',

        # CLIP Model (from requirements.txt - CRITICAL: <3.0.0)
        'sentence_transformers',
        'sentence_transformers.models',
        'transformers',

        # Fast Search (from requirements.txt)
        'faiss',

        # Desktop (from requirements.txt)
        'webview',

        # Utilities (from requirements.txt)
        'psutil',
        'fuzzywuzzy',
        'Levenshtein',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['pytest', 'unittest', 'test', 'tests'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ProductMatcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ProductMatcher'
)
```

---

## Testing Checklist

### Pre-Build Phase (CRITICAL)
- [ ] Download CLIP model: `python scripts/download_clip_model.py`
  - This creates `~/.cache/clip-models/` (~350MB, takes 1-2 minutes)
  - Check it exists: `~/.cache/clip-models/models--sentence-transformers--clip-ViT-B-32/`
  - This MUST be done before any builds start

### Build Phase
- [ ] Run: `build_all_payloads.bat`
- [ ] Verify 4 zip files created with CLIP bundled:
  - `app_nvidia.zip` (~600MB - includes CLIP)
  - `app_amd.zip` (~450MB - includes CLIP)
  - `app_intel.zip` (~500MB - includes CLIP)
  - `app_cpu.zip` (~400MB - includes CLIP)
- [ ] Build launcher: `pyinstaller launcher.py --onefile --name ProductMatcher_Setup`
- [ ] Verify `dist/ProductMatcher_Setup.exe` (~6MB)
- [ ] Decide ngrok distribution mode (`external` or `bundled`) and document decision
- [ ] If bundled mode: stage ngrok binaries for both Windows/macOS and validate executable permissions

### Test Each Payload
For each zip file (app_nvidia.zip, app_amd.zip, etc.):
1. Extract manually
2. Run `ProductMatcher.exe` (should work immediately)
3. Test GPU detection (check console logs)
4. Test workflow: upload images → run matching → export CSV

### Test Launcher
1. Delete `bin/` directory (simulate first-time user)
2. Run `ProductMatcher_Setup.exe`
3. Wait for download and extraction
4. App should launch automatically
5. Run again - should launch instantly (already installed)

### Test ngrok Remote Access
1. Launch app and confirm no crash if ngrok is missing
2. In app: `CONNECT PHONE` -> paste token -> `SETUP TOKEN`
3. Confirm tunnel starts immediately (no restart required)
4. Confirm subsequent app launch auto-starts tunnel
5. In app: `CONNECT PHONE` -> `AUTO NGROK`
6. Confirm remote URL is saved and QR uses remote URL
7. Confirm ports: app `8000`, ngrok API `4040`, no conflicts
8. Repeat on Windows and macOS builds

---

## Success Criteria

✅ Single launcher file (~6MB) to distribute
✅ Auto-detects GPU on first run
✅ Downloads correct variant automatically
✅ Extracts and launches instantly
✅ All 4 variants work independently
✅ CPU fallback available for all variants
✅ Python 3.12 bundled (PyInstaller does this)
✅ All dependencies bundled
✅ CLIP model included (instant first use)
✅ Subsequent launches instant
✅ ngrok path defined for release model (external or bundled)
✅ Remote mobile flow tested on Windows + macOS
