#!/usr/bin/env python3
"""
Dependency Verification and Auto-Fix Script

This script verifies that all dependencies are installed with correct versions
and automatically fixes any issues (like sentence-transformers > 3.0.0).

Run after: pip install -r requirements.txt
"""

import sys
import subprocess
import importlib.metadata
from packaging import version


def run_cmd(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def get_installed_version(package_name):
    """Get installed version of a package"""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python_version():
    """Check if Python 3.12 is being used"""
    print("\n" + "="*80)
    print("Checking Python Version")
    print("="*80)
    
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro
    
    print(f"  Current: Python {major}.{minor}.{micro}")
    print(f"  Required: Python 3.12.x")
    
    # Check if it's Python 3
    if major != 3:
        print(f"\n  ✗ FAIL: Python 3 is required, but you're using Python {major}")
        return False
    
    # Check if it's Python 3.12
    if minor != 12:
        print(f"\n  ✗ FAIL: Python 3.12 is required, but you're using Python {major}.{minor}")
        print(f"\n  [WHY?] Python 3.12 is required for:")
        print(f"    • AMD ROCm GPU support (Windows)")
        print(f"    • Consistent behavior across all platforms")
        print(f"    • Maximum compatibility")
        print(f"\n  [ACTION REQUIRED] Install Python 3.12:")
        print(f"    1. Download from: https://www.python.org/downloads/")
        print(f"    2. Install Python 3.12.x (latest 3.12 version)")
        print(f"    3. Run: python3.12 verify_dependencies.py")
        
        # Check if Python 3.12 is available on Windows
        import platform
        if platform.system() == "Windows":
            print(f"\n  [INFO] Checking if Python 3.12 is already installed...")
            success, stdout, _ = run_cmd("py -3.12 --version")
            if success and "3.12" in stdout:
                print(f"  [OK] Python 3.12 is installed!")
                print(f"  [ACTION] Run this script with: py -3.12 verify_dependencies.py")
        
        return False
    
    print(f"  ✓ PASS: Python 3.12.{micro} detected")
    
    # Warn if using very old 3.12 version
    if micro < 2:
        print(f"  [WARNING] Python 3.12.{micro} is quite old")
        print(f"  [RECOMMENDATION] Update to latest Python 3.12.x for bug fixes")
    
    return True


def check_package_version(package_name, display_name, min_version=None, max_version=None, exact_version=None):
    """Check if package is installed with correct version"""
    installed = get_installed_version(package_name)
    
    if installed is None:
        print(f"  ✗ {display_name:30} NOT INSTALLED")
        return False, None
    
    installed_ver = version.parse(installed)
    
    # Check exact version
    if exact_version:
        required_ver = version.parse(exact_version)
        if installed_ver == required_ver:
            print(f"  ✓ {display_name:30} {installed} (exact match)")
            return True, installed
        else:
            print(f"  ✗ {display_name:30} {installed} (need {exact_version})")
            return False, installed
    
    # Check version range
    if min_version:
        min_ver = version.parse(min_version)
        if installed_ver < min_ver:
            print(f"  ✗ {display_name:30} {installed} (need >={min_version})")
            return False, installed
    
    if max_version:
        max_ver = version.parse(max_version)
        if installed_ver >= max_ver:
            print(f"  ✗ {display_name:30} {installed} (need <{max_version})")
            return False, installed
    
    # Version is OK
    version_str = f"{installed}"
    if min_version and max_version:
        version_str += f" (>={min_version}, <{max_version})"
    elif min_version:
        version_str += f" (>={min_version})"
    elif max_version:
        version_str += f" (<{max_version})"
    
    print(f"  ✓ {display_name:30} {version_str}")
    return True, installed


def check_all_dependencies():
    """Check all required dependencies"""
    print("\n" + "="*80)
    print("Checking Dependencies")
    print("="*80)
    
    issues = []
    
    # Define required packages with version constraints
    packages = [
        # Web Framework
        ("flask", "Flask", "3.0.0", None, None),
        ("flask-cors", "Flask-CORS", "6.0.0", None, None),
        ("werkzeug", "Werkzeug", "3.1.4", None, None),
        
        # Core Scientific Computing
        ("numpy", "NumPy", "1.26.2", None, None),
        ("scipy", "SciPy", "1.11.3", None, None),
        
        # Image Processing
        ("opencv-python", "OpenCV", "4.8.1.78", None, None),
        ("scikit-image", "scikit-image", "0.22.0", None, None),
        ("Pillow", "Pillow", "10.1.0", None, None),
        
        # Deep Learning
        ("torch", "PyTorch", "2.0.0", None, None),
        
        # CRITICAL: sentence-transformers must be < 3.0.0 for AMD ROCm
        ("sentence-transformers", "sentence-transformers", "2.7.0", "3.0.0", None),
        
        # Fast Similarity Search (for large catalogs 1000+ products)
        ("faiss-cpu", "FAISS (CPU)", "1.7.4", None, None),
        
        # System Monitoring (for memory leak detection)
        ("psutil", "psutil", "5.9.0", None, None),
        
        # Testing
        ("pytest", "pytest", "7.0.0", None, None),
    ]
    
    for pkg_name, display_name, min_ver, max_ver, exact_ver in packages:
        ok, installed_ver = check_package_version(pkg_name, display_name, min_ver, max_ver, exact_ver)
        if not ok:
            issues.append((pkg_name, display_name, min_ver, max_ver, exact_ver, installed_ver))
    
    return issues


def check_sentence_transformers_critical():
    """Critical check for sentence-transformers version (MUST be < 3.0.0)"""
    print("\n" + "="*80)
    print("CRITICAL: Checking sentence-transformers Version")
    print("="*80)
    
    installed = get_installed_version("sentence-transformers")
    
    if installed is None:
        print("  [INFO] sentence-transformers not installed yet")
        return None  # Not installed yet, will be checked later
    
    installed_ver = version.parse(installed)
    max_ver = version.parse("3.0.0")
    
    print(f"  Installed: sentence-transformers {installed}")
    print(f"  Required: < 3.0.0")
    
    if installed_ver >= max_ver:
        print(f"\n  ✗ CRITICAL FAIL: sentence-transformers {installed} is >= 3.0.0")
        print(f"\n  [WHY THIS IS CRITICAL]")
        print(f"    • Version 3.0+ breaks AMD ROCm GPU support on Windows")
        print(f"    • Requires torch.distributed.is_initialized() which is missing in ROCm")
        print(f"    • Will cause runtime errors on AMD GPUs")
        print(f"\n  [IMPACT]")
        print(f"    • AMD GPU users: Application will crash")
        print(f"    • NVIDIA/Apple Silicon: May work but untested")
        print(f"\n  [MUST FIX] This will be automatically fixed if you choose 'y'")
        return False
    else:
        print(f"  ✓ PASS: sentence-transformers {installed} is < 3.0.0")
        print(f"  ✓ Compatible with AMD ROCm")
        return True


def fix_sentence_transformers():
    """Fix sentence-transformers version if it's >= 3.0.0"""
    print("\n" + "="*80)
    print("Fixing sentence-transformers Version")
    print("="*80)
    
    installed = get_installed_version("sentence-transformers")
    if installed is None:
        print("  [INFO] sentence-transformers not installed, installing correct version...")
        success, stdout, stderr = run_cmd('pip install "sentence-transformers>=2.7.0,<3.0.0"')
        if success:
            new_version = get_installed_version("sentence-transformers")
            print(f"  ✓ Installed sentence-transformers {new_version}")
            return True
        else:
            print(f"  ✗ Failed to install: {stderr}")
            return False
    
    installed_ver = version.parse(installed)
    max_ver = version.parse("3.0.0")
    
    if installed_ver >= max_ver:
        print(f"  [CRITICAL] sentence-transformers {installed} is >= 3.0.0")
        print(f"  [ACTION] Downgrading to < 3.0.0 for AMD ROCm compatibility...")
        
        # Uninstall current version
        print(f"\n  [1/2] Uninstalling sentence-transformers {installed}...")
        success, _, _ = run_cmd("pip uninstall sentence-transformers -y")
        
        if not success:
            print(f"  ✗ Failed to uninstall")
            return False
        
        # Install correct version
        print(f"  [2/2] Installing sentence-transformers < 3.0.0...")
        success, stdout, stderr = run_cmd('pip install "sentence-transformers>=2.7.0,<3.0.0"')
        
        if success:
            new_version = get_installed_version("sentence-transformers")
            print(f"  ✓ Successfully downgraded to sentence-transformers {new_version}")
            print(f"  ✓ Now compatible with AMD ROCm")
            return True
        else:
            print(f"  ✗ Failed to install: {stderr}")
            return False
    else:
        print(f"  ✓ sentence-transformers {installed} is already < 3.0.0")
        return True


def fix_issues(issues):
    """Attempt to fix dependency issues"""
    if not issues:
        return True
    
    print("\n" + "="*80)
    print("Fixing Dependency Issues")
    print("="*80)
    
    # Special handling for sentence-transformers
    sentence_transformers_issue = None
    other_issues = []
    
    for issue in issues:
        pkg_name, display_name, min_ver, max_ver, exact_ver, installed_ver = issue
        if pkg_name == "sentence-transformers":
            sentence_transformers_issue = issue
        else:
            other_issues.append(issue)
    
    # Fix sentence-transformers first (critical for AMD ROCm)
    if sentence_transformers_issue:
        if not fix_sentence_transformers():
            print("\n  ✗ Failed to fix sentence-transformers")
            return False
    
    # Fix other issues
    for pkg_name, display_name, min_ver, max_ver, exact_ver, installed_ver in other_issues:
        print(f"\n  [INFO] Fixing {display_name}...")
        
        if installed_ver is None:
            # Package not installed
            if exact_ver:
                cmd = f'pip install "{pkg_name}=={exact_ver}"'
            elif min_ver and max_ver:
                cmd = f'pip install "{pkg_name}>={min_ver},<{max_ver}"'
            elif min_ver:
                cmd = f'pip install "{pkg_name}>={min_ver}"'
            else:
                cmd = f'pip install "{pkg_name}"'
        else:
            # Package installed but wrong version
            if exact_ver:
                cmd = f'pip install "{pkg_name}=={exact_ver}" --force-reinstall'
            elif min_ver and max_ver:
                cmd = f'pip install "{pkg_name}>={min_ver},<{max_ver}" --force-reinstall'
            elif min_ver:
                cmd = f'pip install "{pkg_name}>={min_ver}" --force-reinstall'
            else:
                cmd = f'pip install "{pkg_name}" --force-reinstall'
        
        print(f"  Running: {cmd}")
        success, stdout, stderr = run_cmd(cmd)
        
        if success:
            new_version = get_installed_version(pkg_name)
            print(f"  ✓ Fixed {display_name} (now {new_version})")
        else:
            print(f"  ✗ Failed to fix {display_name}: {stderr}")
            return False
    
    return True


def check_amd_rocm_compatibility():
    """Check AMD ROCm compatibility (Windows + Python 3.12 required)"""
    import platform
    
    if platform.system() != "Windows":
        return True  # ROCm compatibility only matters on Windows
    
    print("\n" + "="*80)
    print("Checking AMD ROCm Compatibility (Windows)")
    print("="*80)
    
    # Check Python version
    if sys.version_info.minor != 12:
        print(f"  ✗ FAIL: AMD ROCm requires Python 3.12 on Windows")
        print(f"  Current: Python {sys.version_info.major}.{sys.version_info.minor}")
        return False
    
    print(f"  ✓ Python 3.12 detected (required for AMD ROCm)")
    
    # Check if AMD GPU is present
    success, stdout, _ = run_cmd('powershell "Get-WmiObject Win32_VideoController | Select-Object Name"')
    
    has_amd = False
    if success and ('AMD' in stdout or 'Radeon' in stdout):
        has_amd = True
        print(f"  [INFO] AMD GPU detected")
        
        # Check if PyTorch is installed
        try:
            import torch
            pytorch_version = torch.__version__
            print(f"  PyTorch version: {pytorch_version}")
            
            # Check if it's ROCm build
            if 'rocm' in pytorch_version.lower() or torch.cuda.is_available():
                print(f"  ✓ PyTorch with GPU support detected")
                
                # Verify CUDA/ROCm is available
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"  GPU: {gpu_name}")
                    
                    if 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper():
                        print(f"  ✓ AMD GPU accessible via ROCm")
                    else:
                        print(f"  [WARNING] GPU detected but may not be AMD")
                else:
                    print(f"  [WARNING] PyTorch installed but GPU not accessible")
                    print(f"  [INFO] You may need to run: python gpu/setup_gpu.py")
            else:
                print(f"  [WARNING] PyTorch CPU version detected")
                print(f"  [INFO] For AMD GPU support, run: python gpu/setup_gpu.py")
        except ImportError:
            print(f"  [INFO] PyTorch not installed yet")
            print(f"  [INFO] Run: python gpu/setup_gpu.py to install PyTorch with ROCm")
    else:
        print(f"  [INFO] No AMD GPU detected (ROCm not needed)")
    
    return True


def verify_gpu_support():
    """Verify GPU support is working"""
    print("\n" + "="*80)
    print("Verifying GPU Support")
    print("="*80)
    
    try:
        import torch
        
        print(f"  PyTorch version: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name}")
            
            # Detect AMD vs NVIDIA
            if 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper():
                print(f"  Type: AMD (ROCm)")
                
                # Verify Python 3.12 on Windows
                import platform
                if platform.system() == "Windows":
                    if sys.version_info.minor != 12:
                        print(f"  ✗ WARNING: AMD ROCm requires Python 3.12 on Windows")
                        print(f"  Current: Python {sys.version_info.major}.{sys.version_info.minor}")
                        return False
                    else:
                        print(f"  ✓ Python 3.12 confirmed (required for AMD ROCm)")
                        
            elif 'NVIDIA' in gpu_name.upper() or 'GEFORCE' in gpu_name.upper():
                print(f"  Type: NVIDIA (CUDA)")
            else:
                print(f"  Type: Unknown CUDA device")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
            print(f"  MPS available: {mps_available}")
            
            if mps_available:
                print(f"  GPU: Apple Silicon (MPS)")
        
        # Summary
        if cuda_available or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print(f"\n  ✓ GPU acceleration available")
            return True
        else:
            print(f"\n  [INFO] No GPU detected, will use CPU")
            return True
            
    except ImportError:
        print(f"  [INFO] PyTorch not installed yet")
        print(f"  [INFO] Run: python gpu/setup_gpu.py to install PyTorch")
        return True  # Not an error, just not installed yet
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        return False


def main():
    """Main verification flow"""
    print("="*80)
    print("Dependency Verification and Auto-Fix")
    print("="*80)
    print("\nThis script verifies all dependencies are correctly installed")
    print("and automatically fixes critical issues:")
    print("  • Python 3.12 version check")
    print("  • sentence-transformers < 3.0.0 (CRITICAL for AMD ROCm)")
    print("  • Flask and all dependencies")
    print("  • GPU support verification")
    
    all_ok = True
    critical_issues = []
    
    # Step 1: Check Python version (CRITICAL)
    print("\n" + "="*80)
    print("STEP 1: Python Version Check")
    print("="*80)
    
    if not check_python_version():
        critical_issues.append("Python 3.12 required")
        all_ok = False
        print("\n[CRITICAL] Python 3.12 is required. Please install it and try again.")
        print("\n[STOP] Cannot continue without Python 3.12")
        return False
    
    # Step 2: Check sentence-transformers version (CRITICAL)
    print("\n" + "="*80)
    print("STEP 2: sentence-transformers Version Check (CRITICAL)")
    print("="*80)
    
    st_check = check_sentence_transformers_critical()
    if st_check is False:
        critical_issues.append("sentence-transformers >= 3.0.0 (breaks AMD ROCm)")
        all_ok = False
    
    # Step 3: Check all dependencies
    print("\n" + "="*80)
    print("STEP 3: All Dependencies Check")
    print("="*80)
    
    issues = check_all_dependencies()
    
    # Step 4: Check AMD ROCm compatibility (Windows only)
    import platform
    if platform.system() == "Windows":
        print("\n" + "="*80)
        print("STEP 4: AMD ROCm Compatibility Check (Windows)")
        print("="*80)
        check_amd_rocm_compatibility()
    
    # Step 5: Verify GPU support
    print("\n" + "="*80)
    print("STEP 5: GPU Support Verification")
    print("="*80)
    
    gpu_ok = verify_gpu_support()
    
    # Handle issues
    if critical_issues or issues:
        print("\n" + "="*80)
        print("Issues Found")
        print("="*80)
        
        if critical_issues:
            print("\n[CRITICAL ISSUES]")
            for issue in critical_issues:
                print(f"  ✗ {issue}")
        
        if issues:
            print(f"\n[DEPENDENCY ISSUES] Found {len(issues)} issue(s)")
        
        # Ask user if they want to auto-fix
        print("\n" + "="*80)
        print("Would you like to automatically fix these issues? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            print("\n" + "="*80)
            print("Auto-Fix in Progress")
            print("="*80)
            
            if fix_issues(issues):
                print("\n[SUCCESS] All issues fixed!")
                
                # Re-check dependencies
                print("\n" + "="*80)
                print("Re-checking Dependencies")
                print("="*80)
                
                issues = check_all_dependencies()
                st_check = check_sentence_transformers_critical()
                
                if issues or st_check is False:
                    print(f"\n[WARNING] Still have issues")
                    all_ok = False
                else:
                    print("\n[SUCCESS] All dependencies verified!")
                    all_ok = True
            else:
                print("\n[ERROR] Failed to fix some issues")
                all_ok = False
        else:
            print("\n[INFO] Skipping auto-fix. Please fix manually:")
            print("\n  pip install -r requirements.txt --force-reinstall")
            all_ok = False
    else:
        print("\n[SUCCESS] All dependencies are correctly installed!")
    
    # Final summary
    print("\n" + "="*80)
    print("Verification Summary")
    print("="*80)
    
    print(f"\n  Python 3.12: {'✓ PASS' if sys.version_info.minor == 12 else '✗ FAIL'}")
    
    st_version = get_installed_version("sentence-transformers")
    if st_version:
        st_ok = version.parse(st_version) < version.parse("3.0.0")
        print(f"  sentence-transformers < 3.0.0: {'✓ PASS' if st_ok else '✗ FAIL'}")
    else:
        print(f"  sentence-transformers: [NOT INSTALLED]")
    
    flask_version = get_installed_version("flask")
    print(f"  Flask: {'✓ PASS' if flask_version else '✗ FAIL'}")
    
    print(f"  GPU Support: {'✓ AVAILABLE' if gpu_ok else '[INFO] CPU mode'}")
    
    if all_ok:
        print("\n✅ All checks passed!")
        print("\nNext steps:")
        print("  1. For GPU acceleration: python gpu/setup_gpu.py")
        print("  2. Run tests: python backend/tests/run_comprehensive_gpu_tests.py")
        print("  3. Start app: python backend/app.py")
        return True
    else:
        print("\n❌ Some checks failed")
        print("\nPlease fix the issues above and run this script again:")
        print("  python verify_dependencies.py")
        return False


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
