#!/usr/bin/env python3
"""
Comprehensive GPU Acceleration Test Suite
Tests GPU acceleration across all platforms and modes
"""

import sys
import os
import subprocess
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def print_section(title):
    """Print formatted section"""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)

def run_test(name, command):
    """Run a test command and return success status"""
    print(f"\n▶ Running: {name}")
    print(f"  Command: {command}")
    
    start = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"  ✓ PASSED ({elapsed:.2f}s)")
        return True
    else:
        print(f"  ✗ FAILED ({elapsed:.2f}s)")
        if result.stdout:
            print(f"\nStdout:\n{result.stdout}")
        if result.stderr:
            print(f"\nStderr:\n{result.stderr}")
        return False

def main():
    """Run comprehensive GPU tests"""
    print_header("Comprehensive GPU Acceleration Test Suite")
    
    # System info
    print_section("System Information")
    try:
        from image_processing_clip import detect_device, get_device_info, TORCH_AVAILABLE
        import torch
        
        if TORCH_AVAILABLE:
            device = detect_device()
            info = get_device_info()
            
            print(f"  PyTorch Version: {torch.__version__}")
            print(f"  Device: {device}")
            print(f"  GPU Name: {info.get('gpu_name', 'N/A')}")
            print(f"  VRAM: {info.get('vram_gb', 'N/A')} GB")
            
            if device == 'cuda':
                print(f"  CUDA Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"  CUDA Version: {torch.version.cuda}")
                    print(f"  GPU Count: {torch.cuda.device_count()}")
            elif device == 'mps':
                print(f"  MPS Available: {torch.backends.mps.is_available()}")
                print(f"  MPS Built: {torch.backends.mps.is_built()}")
        else:
            print("  ⚠ PyTorch not available")
    except Exception as e:
        print(f"  ⚠ Error getting system info: {e}")
    
    # Test suite
    tests = []
    results = []
    
    print_section("Test Suite")
    
    # 1. CLIP Unit Tests
    tests.append(("CLIP Unit Tests", "python3 backend/tests/test_clip.py -v -m 'not slow' --tb=short -q"))
    
    # 2. Performance Benchmarks
    tests.append(("Performance Benchmarks", "python3 -m pytest backend/tests/test_clip.py::TestPerformance -v -s --tb=short -q"))
    
    # 3. GPU Platform Tests
    tests.append(("GPU Platform Tests", "python3 backend/tests/test_gpu_support.py -v --tb=short -q"))
    
    # 4. Mode 1 & Mode 3 Tests
    tests.append(("Mode 1 & Mode 3 Tests", "python3 backend/tests/test_clip.py -v -k 'Mode1 or Mode3' --tb=short -q"))
    
    # 5. GPU Memory Tests (if CUDA available)
    if TORCH_AVAILABLE and torch.cuda.is_available():
        tests.append(("GPU Memory Tests", "python3 backend/tests/test_gpu_memory.py -v --tb=short -q"))
    
    # Run all tests
    for name, command in tests:
        success = run_test(name, command)
        results.append((name, success))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status:12} {name}")
    
    print(f"\n  Total: {len(results)} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    # Performance summary
    print_section("Performance Summary")
    try:
        from image_processing_clip import extract_clip_embedding, batch_extract_clip_embeddings
        import tempfile
        import numpy as np
        from PIL import Image
        
        # Create test image
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img, 'RGB')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        pil_img.save(temp_file.name)
        temp_file.close()
        
        # Single image benchmark
        extract_clip_embedding(temp_file.name)  # Warm up
        start = time.time()
        for _ in range(10):
            extract_clip_embedding(temp_file.name)
        single_time = (time.time() - start) / 10
        
        print(f"  Single Image: {single_time:.3f}s per image ({1/single_time:.1f} img/s)")
        
        # Batch benchmark
        images = [temp_file.name] * 50
        start = time.time()
        batch_extract_clip_embeddings(images, batch_size=32)
        batch_time = time.time() - start
        
        print(f"  Batch (50 imgs): {batch_time:.3f}s ({50/batch_time:.1f} img/s)")
        
        # Cleanup
        os.unlink(temp_file.name)
        
    except Exception as e:
        print(f"  ⚠ Could not run performance benchmark: {e}")
    
    # Final status
    print_header("Final Status")
    
    if failed == 0:
        print("  ✅ ALL TESTS PASSED")
        print("  GPU acceleration is working correctly!")
        return 0
    else:
        print(f"  ❌ {failed} TEST(S) FAILED")
        print("  Please review the failures above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
