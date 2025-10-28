"""
Run all backend tests.

This script runs all test files in sequence and reports results.
"""

import sys
import os
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# List of test files to run
TEST_FILES = [
    'test_database.py',
    'test_sku_handling.py',
    'test_image_processing.py',
    'test_feature_cache.py',
    'test_similarity_simple.py',
]

def run_test(test_file):
    """Run a single test file"""
    print(f"\n{'=' * 70}")
    print(f"Running: {test_file}")
    print('=' * 70)
    
    test_path = os.path.join(os.path.dirname(__file__), test_file)
    
    try:
        result = subprocess.run(
            [sys.executable, test_path],
            capture_output=False,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False

def main():
    """Run all tests and report results"""
    print("=" * 70)
    print("RUNNING ALL BACKEND TESTS")
    print("=" * 70)
    
    results = {}
    
    for test_file in TEST_FILES:
        success = run_test(test_file)
        results[test_file] = success
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for success in results.values() if success)
    failed = len(results) - passed
    
    for test_file, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status:12} {test_file}")
    
    print("=" * 70)
    print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
