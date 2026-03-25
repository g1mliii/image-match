"""
Verify Setup Script
Quick verification that GPU acceleration is working correctly
"""

import sys
import subprocess
from pathlib import Path


# Resolve paths once so they work regardless of CWD or spaces in path.
_PYTHON = sys.executable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def run_command(cmd, description, cwd=None):
    """Run a command and report results"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=cwd,
            encoding="utf-8",
            errors="replace",
        )
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("[ERROR] Command timed out")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def main():
    """Run all verification checks"""
    print("="*80)
    print("Product Matching System - Setup Verification")
    print("="*80)

    results = {}

    # 1. Check Python version
    print(f"\nPython Version: {sys.version}")
    results['python'] = True

    check_gpu_path = str(_SCRIPT_DIR / "check_gpu.py")

    # 2. Check GPU detection
    results['gpu'] = run_command(
        [_PYTHON, check_gpu_path],
        "1. GPU Detection"
    )

    # 3. Run quick CLIP test
    results['clip'] = run_command(
        [_PYTHON, "-m", "pytest", "backend/tests/test_clip.py::TestGPUDetection::test_detect_device", "-v"],
        "2. CLIP Model Test",
        cwd=str(_PROJECT_ROOT),
    )

    # 4. Run GPU support test
    results['gpu_support'] = run_command(
        [_PYTHON, "-m", "pytest", "backend/tests/test_gpu_support.py::TestGPUPlatformSupport::test_gpu_detection_comprehensive", "-v"],
        "3. GPU Support Test",
        cwd=str(_PROJECT_ROOT),
    )

    # 5. Run Mode 1 test
    results['mode1'] = run_command(
        [_PYTHON, "-m", "pytest", "backend/tests/test_clip.py::TestMode1Matching::test_mode1_basic_matching", "-v"],
        "4. Mode 1 (Visual) Matching Test",
        cwd=str(_PROJECT_ROOT),
    )

    # 6. Run Mode 3 test
    results['mode3'] = run_command(
        [_PYTHON, "-m", "pytest", "backend/tests/test_clip.py::TestMode3HybridMatching::test_mode3_visual_plus_category", "-v"],
        "5. Mode 3 (Hybrid) Matching Test",
        cwd=str(_PROJECT_ROOT),
    )

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    all_passed = True
    for test, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} - {test.replace('_', ' ').title()}")
        if not passed:
            all_passed = False

    print("\n" + "="*80)

    if all_passed:
        print("ALL CHECKS PASSED!")
        print("\nYour system is ready to use GPU acceleration.")
        print("\nNext steps:")
        print("1. Run benchmark: python benchmark_gpu.py")
        print("2. Start the application: cd .. && python main.py")
    else:
        print("SOME CHECKS FAILED")
        print("\nPlease review the errors above and:")
        print("1. Check GPU_SETUP_GUIDE.md for troubleshooting")
        print("2. Run: python setup_gpu.py")
        print("3. Ensure all dependencies are installed")
        print("4. For AMD GPU: Ensure Python 3.12 is being used")

    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
