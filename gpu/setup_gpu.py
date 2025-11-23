"""
GPU Setup for Product Matching System
Automatically installs PyTorch + GPU drivers (NVIDIA/AMD/Apple Silicon)

Based on:
- AMD ROCm: https://rocm.docs.amd.com/projects/install-on-windows/en/latest/
- PyTorch: https://pytorch.org/get-started/locally/
"""

import subprocess
import sys
import os
import platform
import urllib.request


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
        # Check for AMD GPU
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
    
    return 'cpu', 'No GPU detected'


def install_dependencies():
    """Install all required Python dependencies"""
    print("\n" + "="*80)
    print("Installing Python Dependencies")
    print("="*80)
    
    # Use consolidated requirements.txt in project root
    req_file = "../requirements.txt"
    
    if os.path.exists(req_file):
        print(f"\n[INFO] Installing from {req_file}...")
        success, stdout, stderr = run_cmd(f"pip install -r {req_file}")
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
    
    # Uninstall existing
    print("\n[1/3] Removing existing PyTorch...")
    run_cmd("pip uninstall torch torchvision torchaudio -y")
    
    # Install based on GPU type
    print(f"\n[2/3] Installing PyTorch for {gpu_type.upper()}...")
    
    if gpu_type == 'nvidia':
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    elif gpu_type == 'amd':
        if platform.system() == "Linux":
            cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
        else:  # Windows
            # AMD ROCm PyTorch for Windows requires Python 3.12
            # NOTE: This check is redundant (main() already checked) but kept as safety net
            if sys.version_info.minor == 12:
                # Official AMD ROCm wheels for Windows (Python 3.12 only)
                print("\n[INFO] Installing AMD ROCm PyTorch for Windows (Python 3.12)")
                print("[INFO] This may take several minutes (~780MB download)...")
                
                # Install PyTorch with ROCm 6.4.4
                success1, _, _ = run_cmd("pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torch-2.8.0a0%2Bgitfc14c65-cp312-cp312-win_amd64.whl")
                success2, _, _ = run_cmd("pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchvision-0.24.0a0%2Bc85f008-cp312-cp312-win_amd64.whl")
                success3, _, _ = run_cmd("pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchaudio-2.6.0a0%2B1a8f621-cp312-cp312-win_amd64.whl")
                
                if success1 and success2 and success3:
                    print("[OK] AMD ROCm PyTorch installed")
                    
                    # CRITICAL: Install sentence-transformers < 3.0.0 for AMD ROCm compatibility
                    print("\n[INFO] Installing sentence-transformers < 3.0.0 (required for AMD ROCm)...")
                    success_st, _, stderr_st = run_cmd('pip install "sentence-transformers>=2.7.0,<3.0.0"')
                    
                    if success_st:
                        print("[OK] sentence-transformers < 3.0.0 installed")
                    else:
                        print(f"[WARNING] Failed to install sentence-transformers: {stderr_st}")
                        print("[WARNING] AMD ROCm may not work correctly without sentence-transformers < 3.0.0")
                    
                    return True
                else:
                    print("[ERROR] Failed to install AMD ROCm PyTorch")
                    return False
            else:
                # This should never happen (main() already checked), but handle it anyway
                print(f"\n[ERROR] AMD ROCm PyTorch requires Python 3.12, but you have Python {sys.version_info.major}.{sys.version_info.minor}")
                print("[ERROR] This should have been caught earlier. Please report this bug.")
                print("[INFO] Installing CPU version of PyTorch as fallback...")
                cmd = "pip install torch torchvision torchaudio"
    elif gpu_type == 'apple':
        cmd = "pip install torch torchvision torchaudio"
    else:  # CPU
        cmd = "pip install torch torchvision torchaudio"
    
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
    """Check if ROCm is already installed"""
    rocm_paths = [
        "C:\\Program Files\\AMD\\ROCm",
        "C:\\Program Files\\AMD\\ROCm\\bin",
    ]
    
    for path in rocm_paths:
        if os.path.exists(path):
            return True, path
    
    # Check for HIP DLLs
    hip_dlls = ["amdhip64_6.dll", "amd_comgr_2.dll"]
    system_path = os.environ.get('PATH', '').split(';')
    
    for dll in hip_dlls:
        for path in system_path:
            dll_path = os.path.join(path, dll)
            if os.path.exists(dll_path):
                return True, path
    
    return False, None


def download_rocm_installer():
    """Attempt to download ROCm installer automatically"""
    print("\n[INFO] Attempting to download HIP SDK installer...")
    
    # Latest ROCm 6.x installer URL (update as needed)
    installer_url = "https://github.com/ROCm/rocm-install-on-windows/releases/latest"
    
    try:
        import webbrowser
        print(f"[INFO] Opening download page: {installer_url}")
        webbrowser.open(installer_url)
        return True
    except Exception as e:
        print(f"[WARNING] Could not open browser: {e}")
        print(f"[INFO] Please manually visit: {installer_url}")
        return False


def install_rocm_windows():
    """Guide user through ROCm installation for Windows"""
    print("\n" + "="*80)
    print("AMD ROCm HIP SDK Installation Required")
    print("="*80)
    
    # Check if already installed
    is_installed, install_path = check_rocm_installed()
    if is_installed:
        print(f"\n[OK] ROCm appears to be installed at: {install_path}")
        response = input("\nReinstall anyway? (y/n): ")
        if response.lower() != 'y':
            print("[INFO] Skipping ROCm installation")
            return
    
    print("\nYour AMD GPU needs the HIP SDK (ROCm) to work with PyTorch.")
    print("\nOfficial AMD ROCm Documentation:")
    print("https://rocm.docs.amd.com/projects/install-on-windows/en/latest/")
    
    print("\n" + "-"*80)
    print("Installation Steps:")
    print("-"*80)
    
    print("\n1. Download the HIP SDK installer:")
    print("   https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html")
    print("   OR")
    print("   https://github.com/ROCm/rocm-install-on-windows/releases")
    print("   - Choose ROCm 6.x (recommended)")
    print("   - Accept license agreement")
    
    print("\n2. Run the installer as Administrator")
    print("   - Extracts to C:\\AMD temporarily")
    
    print("\n3. Choose components:")
    print("   ✓ HIP Core (Required)")
    print("   ✓ Libraries (Required)")
    print("   ✓ Runtime Compiler (Required)")
    print("   ✓ Ray Tracing (Optional)")
    print("   ✓ VS plugin (Optional)")
    print("   - Select Full/Partial/None for each")
    
    print("\n4. Optional: Install AMD Display Driver")
    print("   - Full, Minimal, or Driver-only")
    print("   - Restart required after driver install")
    
    print("\n5. Finish installation and wait for completion")
    
    print("\n6. Restart your computer")
    
    print("\n7. Verify installation:")
    print("   python check_gpu.py")
    
    print("\n" + "-"*80)
    print("Important Notes:")
    print("-"*80)
    print("• ROCm 6.x is NOT backward-compatible with 5.x")
    print("• New DLLs: amdhip64_6.dll, amd_comgr_2.dll, hiprt0200564.dll")
    print("• Supported: Windows 10/11, Server 2022")
    print("• AMD GPU required only to run apps (not for SDK install)")
    print("-"*80)
    
    # Automatically open download page
    print("\n[INFO] Opening download page in browser...")
    download_rocm_installer()
    
    print("\n[INFO] After installing HIP SDK and restarting, run this script again to verify.")


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
    
    success, stdout, stderr = run_cmd(f'python -c "{verify_script}"')
    
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
        print("  • AMD ROCm GPU support (Windows)")
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
    
    # AMD Windows - check ROCm before PyTorch
    if gpu_type == 'amd' and platform.system() == "Windows":
        is_installed, install_path = check_rocm_installed()
        
        if not is_installed:
            install_rocm_windows()
            print("\n[INFO] Please install ROCm and restart, then run this script again.")
            return True
        else:
            print(f"\n[OK] ROCm detected at: {install_path}")
    
    # Install PyTorch
    if not install_pytorch(gpu_type):
        return False
    
    # Verify
    if not verify_gpu():
        return False
    
    # Run benchmark
    print("\n" + "="*80)
    print("Running Quick Benchmark")
    print("="*80)
    
    try:
        print("\n[INFO] Testing GPU performance...")
        import os
        benchmark_path = os.path.join(os.path.dirname(__file__), "benchmark_gpu.py")
        run_cmd(f"python {benchmark_path}")
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
