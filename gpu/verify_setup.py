"""
Verify Setup Script
Quick verification that GPU acceleration is working correctly
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
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
    
    # Get script directory
    script_dir = Path(__file__).parent
    check_gpu_path = script_dir / "check_gpu.py"
    
    # 2. Check GPU detection
    results['gpu'] = run_command(
        f"python {check_gpu_path}",
        "1. GPU Detection"
    )
    
    # 3. Run quick CLIP test
    results['clip'] = run_command(
        "cd .. && python -m pytest backend/tests/test_clip.py::TestGPUDetection::test_detect_device -v",
        "2. CLIP Model Test"
    )
    
    # 4. Run GPU support test
    results['gpu_support'] = run_command(
        "cd .. && python -m pytest backend/tests/test_gpu_support.py::TestGPUPlatformSupport::test_gpu_detection_comprehensive -v",
        "3. GPU Support Test"
    )
    
    # 5. Run Mode 1 test
    results['mode1'] = run_command(
        "cd .. && python -m pytest backend/tests/test_clip.py::TestMode1Matching::test_mode1_basic_matching -v",
        "4. Mode 1 (Visual) Matching Test"
    )
    
    # 6. Run Mode 3 test
    results['mode3'] = run_command(
        "cd .. && python -m pytest backend/tests/test_clip.py::TestMode3HybridMatching::test_mode3_visual_plus_category -v",
        "5. Mode 3 (Hybrid) Matching Test"
    )
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test.replace('_', ' ').title()}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("\nYour system is ready to use GPU acceleration.")
        print("\nNext steps:")
        print("1. Run benchmark: python benchmark_gpu.py")
        print("2. Start the application: cd .. && python main.py")
    else:
        print("⚠️  SOME CHECKS FAILED")
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
