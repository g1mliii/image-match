"""
GPU Setup for Product Matching System
Automatically installs PyTorch + GPU drivers (NVIDIA/AMD/Apple Silicon)

Based on:
- AMD ROCm: Bundled with AMD Adrenaline driver (26.1.1+)
- PyTorch: https://pytorch.org/get-started/locally/
"""

import subprocess
import sys
import os
import platform
import urllib.request


# Use the current Python interpreter for all subprocess calls so that
# pip/python commands target the correct environment (venv or system).
_PYTHON = sys.executable
_PIP = f'"{_PYTHON}" -m pip'


def run_cmd(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_nvidia_drivers():
    """Check if NVIDIA drivers are installed"""
    system = platform.system()
    
    if system == "Windows":
        # Check for nvidia-smi
        success, stdout, _ = run_cmd("nvidia-smi --version")
        if success:
            return True, "NVIDIA drivers installed"
        
        # Check for NVIDIA DLLs
        nvidia_paths = [
            "C:\\Windows\\System32\\nvcuda.dll",
            "C:\\Program Files\\NVIDIA Corporation"
        ]
        for path in nvidia_paths:
            if os.path.exists(path):
                return True, "NVIDIA drivers detected"
        
        return False, "NVIDIA drivers not found"
    
    elif system == "Linux":
        success, stdout, _ = run_cmd("nvidia-smi --version")
        return success, "NVIDIA drivers installed" if success else "NVIDIA drivers not found"
    
    return True, "Not applicable"


def detect_gpu():
    """Detect GPU type"""
    system = platform.system()
    
    if system == "Windows":
        # Check for AMD/NVIDIA/Intel GPU
        success, stdout, _ = run_cmd('powershell "Get-WmiObject Win32_VideoController | Select-Object Name"')
        if success:
            if 'AMD' in stdout or 'Radeon' in stdout:
                for line in stdout.split('\n'):
                    if 'AMD' in line or 'Radeon' in line:
                        return 'amd', line.strip()
            elif 'NVIDIA' in stdout or 'GeForce' in stdout or 'RTX' in stdout or 'Quadro' in stdout:
                for line in stdout.split('\n'):
                    if 'NVIDIA' in line or 'GeForce' in line or 'RTX' in line or 'Quadro' in line:
                        return 'nvidia', line.strip()
            elif 'Intel' in stdout and ('Arc' in stdout or 'Iris' in stdout or 'UHD' in stdout):
                for line in stdout.split('\n'):
                    if 'Intel' in line and ('Arc' in line or 'Iris' in line or 'UHD' in line):
                        return 'intel', line.strip()
    
    elif system == "Darwin":  # macOS
        if platform.machine() == "arm64":
            return 'apple', 'Apple Silicon (M1/M2/M3/M4/M5)'
    
    elif system == "Linux":
        # Check for NVIDIA
        success, stdout, _ = run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader")
        if success and stdout.strip():
            return 'nvidia', stdout.strip()
        
        # Check for AMD
        success, stdout, _ = run_cmd("lspci | grep -i 'vga\\|3d\\|display'")
        if success and ('AMD' in stdout or 'Radeon' in stdout):
            return 'amd', 'AMD GPU'
        
        # Check for Intel
        if success and 'Intel' in stdout:
            return 'intel', 'Intel GPU'
    
    return 'cpu', 'No GPU detected'


def install_dependencies():
    """Install all required Python dependencies"""
    print("\n" + "="*80)
    print("Installing Python Dependencies")
    print("="*80)

    # Upgrade pip first
    print("\n[INFO] Upgrading pip to latest version...")
    run_cmd(f"{_PIP} install --upgrade pip")

    # Use consolidated requirements.txt in project root
    # Resolve relative to this script's location so it works regardless of CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(script_dir, "..", "requirements.txt")

    if os.path.exists(req_file):
        print(f"\n[INFO] Installing from {req_file}...")
        success, stdout, stderr = run_cmd(f'{_PIP} install -r "{req_file}"')
        if success:
            print(f"[OK] Installed dependencies from {req_file}")
        else:
            print(f"[WARNING] Some dependencies failed: {stderr}")
            print(f"[INFO] This is normal - PyTorch will be installed separately for GPU support")
    else:
        print(f"[ERROR] Requirements file not found: {req_file}")
        print(f"[INFO] Make sure you're running this from the gpu/ directory")
        return False
    
    return True


def install_pytorch(gpu_type):
    """Install PyTorch with correct GPU support"""
    print("\n" + "="*80)
    print("Installing PyTorch")
    print("="*80)

    # Check if PyTorch is already installed (e.g., via AMD Adrenaline)
    pytorch_check = run_cmd(f"{_PIP} show torch")
    if pytorch_check[0]:
        existing_version = pytorch_check[1]
        if 'rocmsdk' in existing_version or 'rocm' in existing_version:
            print("\n[OK] PyTorch with AMD ROCm already installed via AMD Adrenaline!")
            print("[INFO] Skipping PyTorch installation")

            # Verify sentence-transformers compatibility
            print("\n[INFO] Verifying sentence-transformers compatibility...")
            success_st, _, stderr_st = run_cmd(f'{_PIP} install "sentence-transformers>=2.7.0,<3.0.0"')

            if success_st:
                print("[OK] sentence-transformers < 3.0.0 installed")
            else:
                print(f"[WARNING] Failed to install sentence-transformers: {stderr_st}")

            return True

    # Uninstall existing
    print("\n[1/3] Removing existing PyTorch...")
    run_cmd(f"{_PIP} uninstall torch torchvision torchaudio -y")

    # Install based on GPU type
    print(f"\n[2/3] Installing PyTorch for {gpu_type.upper()}...")

    if gpu_type == 'nvidia':
        cmd = f"{_PIP} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    elif gpu_type == 'amd':
        if platform.system() == "Linux":
            cmd = f"{_PIP} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
        else:  # Windows
            # AMD ROCm PyTorch for Windows requires Python 3.12
            # NOTE: This check is redundant (main() already checked) but kept as safety net
            if sys.version_info.minor == 12:
                # AMD Adrenaline driver (26.1.1+) should have already installed PyTorch+ROCm.
                # If we got here, it means ROCm was detected but PyTorch wasn't already caught
                # by the early-return check above. Try the official PyTorch ROCm index.
                print("\n[INFO] Installing PyTorch with ROCm support for AMD GPU...")
                print("[INFO] If this fails, install AMD Adrenaline driver (26.1.1+) which")
                print("       bundles PyTorch+ROCm automatically.")
                cmd = f"{_PIP} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
            else:
                # This should never happen (main() already checked), but handle it anyway
                print(f"\n[ERROR] AMD ROCm PyTorch requires Python 3.12, but you have Python {sys.version_info.major}.{sys.version_info.minor}")
                print("[ERROR] This should have been caught earlier. Please report this bug.")
                print("[INFO] Installing CPU version of PyTorch as fallback...")
                cmd = f"{_PIP} install torch torchvision torchaudio"
    elif gpu_type == 'apple':
        cmd = f"{_PIP} install torch torchvision torchaudio"
    elif gpu_type == 'intel':
        # Intel GPU - install PyTorch first, then Intel extension
        print("\n[INFO] Installing PyTorch for Intel GPU...")
        cmd = f"{_PIP} install torch torchvision torchaudio"
        success, stdout, stderr = run_cmd(cmd)

        if success:
            print("[OK] PyTorch installed")
            print("\n[INFO] Installing Intel Extension for PyTorch...")
            print("[INFO] This enables GPU acceleration on Intel Arc/Iris/UHD GPUs...")

            # Try multiple package names (Intel keeps changing them)
            intel_packages = [
                "intel-extension-for-pytorch",  # Current name
                "intel_extension_for_pytorch",  # Alternative
            ]

            intel_installed = False
            for pkg in intel_packages:
                print(f"[INFO] Trying package: {pkg}")
                intel_cmd = f'{_PIP} install {pkg}'
                intel_success, intel_stdout, intel_stderr = run_cmd(intel_cmd)

                if intel_success:
                    print(f"[OK] Intel Extension installed ({pkg}) - GPU acceleration enabled!")
                    print("[INFO] Expected speedup: 3-5x faster than CPU")
                    intel_installed = True
                    break

            if not intel_installed:
                print("[WARNING] Intel Extension installation failed")
                print("[INFO] This is common on some Windows configurations")
                print("[INFO] Your app will work fine in CPU mode:")
                print("       • CPU: 10-20 images/sec (perfectly usable)")
                print("       • Still much faster than manual matching!")

            # Install other dependencies (pywebview, Flask, etc.)
            print("\n[INFO] Installing other dependencies...")
            install_dependencies()

            return True  # PyTorch still works, just without Intel GPU
        else:
            print(f"[ERROR] PyTorch installation failed: {stderr}")
            return False
    else:  # CPU
        cmd = f"{_PIP} install torch torchvision torchaudio"
    
    success, stdout, stderr = run_cmd(cmd)
    
    if not success:
        print(f"[ERROR] Installation failed: {stderr}")
        return False
    
    print("[OK] PyTorch installed")
    
    # Install other dependencies
    print("\n[3/3] Installing other dependencies...")
    install_dependencies()
    
    return True


def check_rocm_installed():
    """Check if ROCm is already installed (via Adrenaline driver or standalone HIP SDK)"""
    # Check for ROCm installed via AMD Adrenaline driver (26.1.1+)
    # Adrenaline installs PyTorch with ROCm bundled directly into the Python environment
    pytorch_check = run_cmd(f"{_PIP} show torch")
    if pytorch_check[0]:
        version_info = pytorch_check[1]
        if 'rocm' in version_info.lower():
            return True, "AMD Adrenaline (PyTorch with ROCm bundled)"

    # Check for standalone HIP SDK paths (legacy)
    rocm_paths = [
        "C:\\Program Files\\AMD\\ROCm",
        "C:\\Program Files\\AMD\\ROCm\\bin",
    ]

    for path in rocm_paths:
        if os.path.exists(path):
            return True, path

    # Check for HIP DLLs on PATH (installed by either Adrenaline or HIP SDK)
    hip_dlls = ["amdhip64_6.dll", "amdhip64_7.dll", "amd_comgr_2.dll"]
    system_path = os.environ.get('PATH', '').split(';')

    for dll in hip_dlls:
        for path in system_path:
            dll_path = os.path.join(path, dll)
            if os.path.exists(dll_path):
                return True, path

    return False, None


def install_rocm_windows():
    """Guide user through ROCm installation for Windows via AMD Adrenaline driver"""
    print("\n" + "="*80)
    print("AMD GPU Support Setup - ROCm via AMD Adrenaline Driver")
    print("="*80)

    # Check if already installed
    is_installed, install_path = check_rocm_installed()
    if is_installed:
        print(f"\n[OK] ROCm already available: {install_path}")
        response = input("\nView setup instructions anyway? (y/n): ")
        if response.lower() != 'y':
            print("[INFO] Skipping ROCm setup")
            return

    print("\nYour AMD GPU needs ROCm + PyTorch for GPU acceleration.")
    print("\n" + "="*80)
    print("Install AMD Adrenaline Driver (26.1.1 or later)")
    print("="*80)
    print("\nAMD Adrenaline driver bundles PyTorch + ROCm automatically.")
    print("This is the easiest and recommended way to get GPU acceleration.\n")
    print("Steps:")
    print("  1. Download AMD Adrenaline from:")
    print("     https://www.amd.com/en/support/amd-radeon-software")
    print("  2. Run the installer (version 26.1.1 or later required)")
    print("  3. Follow the setup wizard - ROCm is included automatically")
    print("  4. Restart your computer after installation")
    print("  5. Run this script again to verify and install remaining dependencies")

    print("\n" + "-"*80)
    print("Important Notes:")
    print("-"*80)
    print("  - Adrenaline 26.1.1+ installs PyTorch with ROCm bundled")
    print("  - Python 3.12 is required for AMD ROCm compatibility")
    print("  - No separate HIP SDK download needed (it was required before)")
    print("  - Supported: Windows 10/11 with AMD Radeon GPUs")
    print("  - Limitation: sentence-transformers must be < 3.0.0 (no torch.distributed)")
    print("-"*80)

    try:
        import webbrowser
        print("\n[INFO] Opening AMD Adrenaline download page...")
        webbrowser.open("https://www.amd.com/en/support/amd-radeon-software")
    except Exception:
        print("\n[INFO] Visit: https://www.amd.com/en/support/amd-radeon-software")

    print("\n[INFO] After installing Adrenaline and restarting, run this script again.")


def verify_gpu():
    """Verify GPU detection"""
    print("\n" + "="*80)
    print("Verifying GPU Detection")
    print("="*80)
    
    verify_script = """
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f"MPS available: True")
    print(f"GPU: Apple Silicon")
else:
    print(f"Mode: CPU")
"""
    
    success, stdout, stderr = run_cmd(f'"{_PYTHON}" -c "{verify_script}"')
    
    if success:
        print("\n" + stdout)
        
        if "CUDA available: True" in stdout or "MPS available: True" in stdout:
            print("\n[SUCCESS] GPU acceleration is working!")
            return True
        else:
            print("\n[INFO] Running in CPU mode (42 img/s - still excellent!)")
            return True
    else:
        print(f"\n[ERROR] Verification failed: {stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.12 (required for all platforms)"""
    major = sys.version_info.major
    minor = sys.version_info.minor
    
    print(f"\nPython Version: {major}.{minor}.{sys.version_info.micro}")
    print(f"Required Version: 3.12.x")
    
    # Enforce Python 3.12 for all platforms
    if minor != 12:
        print(f"\n[ERROR] Python 3.12 is required, but you're using Python {major}.{minor}")
        print("\n[WHY?] Python 3.12 ensures compatibility with:")
        print("  • AMD ROCm GPU support via Adrenaline driver (Windows)")
        print("  • NVIDIA CUDA GPU support")
        print("  • Apple Silicon MPS support")
        print("  • Consistent behavior across all platforms")
        
        # Check if Python 3.12 is available
        if platform.system() == "Windows":
            print("\n[INFO] Checking if Python 3.12 is installed...")
            success, stdout, _ = run_cmd("py -3.12 --version")
            if success and "3.12" in stdout:
                print("[OK] Python 3.12 is installed!")
                print("\n[ACTION REQUIRED] Run this script with Python 3.12:")
                print("  py -3.12 gpu/setup_gpu.py")
                print("\nOr install dependencies with Python 3.12:")
                print("  py -3.12 -m pip install -r requirements.txt")
            else:
                print("[WARNING] Python 3.12 not found")
                print("\n[ACTION REQUIRED] Install Python 3.12:")
                print("  1. Download from: https://www.python.org/downloads/")
                print("  2. Install Python 3.12.x")
                print("  3. Run: py -3.12 gpu/setup_gpu.py")
        else:
            print("\n[ACTION REQUIRED] Install Python 3.12:")
            print("  1. Download from: https://www.python.org/downloads/")
            print("  2. Install Python 3.12.x")
            print("  3. Run this script again with Python 3.12")
        
        return False
    
    print("[OK] Python 3.12 detected - compatible with all GPU types!")
    return True


def main():
    """Main installation flow"""
    print("="*80)
    print("GPU Setup for Product Matching System")
    print("="*80)
    
    # Check Python version first
    if not check_python_version():
        return False
    
    # Detect GPU
    print("\nDetecting GPU...")
    gpu_type, gpu_name = detect_gpu()
    
    print(f"\nGPU Type: {gpu_type.upper()}")
    print(f"GPU Name: {gpu_name}")
    
    # Check NVIDIA drivers if NVIDIA GPU
    if gpu_type == 'nvidia':
        drivers_installed, driver_msg = check_nvidia_drivers()
        print(f"Driver Status: {driver_msg}")
        
        if not drivers_installed:
            print("\n" + "="*80)
            print("NVIDIA Drivers Required")
            print("="*80)
            print("\n[INFO] NVIDIA GPU detected but drivers not installed.")
            print("\nDownload NVIDIA drivers from:")
            print("https://www.nvidia.com/download/index.aspx")
            print("\nAfter installing drivers, run this script again.")
            
            response = input("\nOpen NVIDIA driver download page? (y/n): ")
            if response.lower() == 'y':
                import webbrowser
                webbrowser.open("https://www.nvidia.com/download/index.aspx")
            
            return True
        else:
            print("[OK] NVIDIA drivers are installed")
    
    # Apple Silicon - always ready
    elif gpu_type == 'apple':
        print("[OK] Apple Silicon GPU ready (no drivers needed)")
    
    # AMD Windows - check ROCm (installed via Adrenaline driver)
    if gpu_type == 'amd' and platform.system() == "Windows":
        is_installed, install_path = check_rocm_installed()

        if not is_installed:
            print("\n[WARNING] AMD GPU detected but ROCm/PyTorch not found.")
            print("[INFO] ROCm is bundled with AMD Adrenaline driver (26.1.1+).")
            install_rocm_windows()
            print("\n[INFO] Please install AMD Adrenaline driver and restart, then run this script again.")
            return True
        else:
            print(f"\n[OK] ROCm detected: {install_path}")
    
    # Install PyTorch
    if not install_pytorch(gpu_type):
        return False
    
    # Verify
    if not verify_gpu():
        return False
    
    # Download CLIP model
    print("\n" + "="*80)
    print("Downloading CLIP Model")
    print("="*80)

    try:
        print("\n[INFO] Pre-downloading CLIP model to cache...")
        download_script = os.path.join(os.path.dirname(__file__), "..", "scripts", "download_clip_model.py")
        success, stdout, stderr = run_cmd(f'"{_PYTHON}" "{download_script}"')
        if success:
            print("[OK] CLIP model downloaded and cached")
        else:
            print(f"[WARNING] CLIP model download failed: {stderr}")
            print("[INFO] Model will download automatically on first app launch")
    except Exception as e:
        print(f"[WARNING] Could not pre-download CLIP model: {e}")
        print("[INFO] Model will download automatically on first app launch")
    
    # Run benchmark
    print("\n" + "="*80)
    print("Running Quick Benchmark")
    print("="*80)
    
    try:
        print("\n[INFO] Testing GPU performance...")
        benchmark_path = os.path.join(os.path.dirname(__file__), "benchmark_gpu.py")
        run_cmd(f'"{_PYTHON}" "{benchmark_path}"')
    except Exception as e:
        print(f"[WARNING] Benchmark failed: {e}")
    
    print("\n" + "="*80)
    print("Setup Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run tests: cd .. && python -m pytest backend/tests/test_clip.py -v")
    print("2. Check GPU: python check_gpu.py")
    print("3. Benchmark: python benchmark_gpu.py")
    print("4. Verify: python verify_setup.py")
    print("5. Start using the application: cd .. && python main.py")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
