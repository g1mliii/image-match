"""
Simple test script to verify similarity computation works correctly.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import similarity functions
from similarity import (
    compute_color_similarity,
    compute_shape_similarity,
    compute_texture_similarity,
    compute_combined_similarity,
    compute_all_similarities,
    batch_compute_similarities
)


def test_color_similarity():
    """Test color similarity computation"""
    print("Testing color similarity...")
    
    # Test 1: Identical features should give 100% similarity
    features = np.random.rand(256).astype(np.float32)
    features /= features.sum()
    similarity = compute_color_similarity(features, features)
    assert abs(similarity - 100.0) < 0.01, f"Expected 100, got {similarity}"
    print("  ✓ Identical features: 100% similarity")
    
    # Test 2: Non-overlapping histograms should give 0% similarity
    features1 = np.zeros(256, dtype=np.float32)
    features1[:128] = 1.0
    features1 /= features1.sum()
    
    features2 = np.zeros(256, dtype=np.float32)
    features2[128:] = 1.0
    features2 /= features2.sum()
    
    similarity = compute_color_similarity(features1, features2)
    assert abs(similarity - 0.0) < 0.01, f"Expected 0, got {similarity}"
    print("  ✓ Non-overlapping features: 0% similarity")
    
    # Test 3: Score range
    for _ in range(10):
        f1 = np.random.rand(256).astype(np.float32)
        f2 = np.random.rand(256).astype(np.float32)
        sim = compute_color_similarity(f1, f2)
        assert 0 <= sim <= 100, f"Score out of range: {sim}"
    print("  ✓ All scores in 0-100 range")
    
    print("✓ Color similarity tests passed\n")


def test_shape_similarity():
    """Test shape similarity computation"""
    print("Testing shape similarity...")
    
    # Test 1: Identical features should give 100% similarity
    features = np.random.rand(7).astype(np.float32)
    similarity = compute_shape_similarity(features, features)
    assert abs(similarity - 100.0) < 0.01, f"Expected 100, got {similarity}"
    print("  ✓ Identical features: 100% similarity")
    
    # Test 2: Very different features should give low similarity
    features1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    features2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float32)
    similarity = compute_shape_similarity(features1, features2)
    assert similarity < 10, f"Expected low similarity, got {similarity}"
    print("  ✓ Very different features: low similarity")
    
    # Test 3: Score range
    for _ in range(10):
        f1 = np.random.rand(7).astype(np.float32) * 10
        f2 = np.random.rand(7).astype(np.float32) * 10
        sim = compute_shape_similarity(f1, f2)
        assert 0 <= sim <= 100, f"Score out of range: {sim}"
    print("  ✓ All scores in 0-100 range")
    
    print("✓ Shape similarity tests passed\n")


def test_texture_similarity():
    """Test texture similarity computation"""
    print("Testing texture similarity...")
    
    # Test 1: Identical features should give 100% similarity
    features = np.random.rand(256).astype(np.float32)
    features /= features.sum()
    similarity = compute_texture_similarity(features, features)
    assert abs(similarity - 100.0) < 0.01, f"Expected 100, got {similarity}"
    print("  ✓ Identical features: 100% similarity")
    
    # Test 2: Very different features should give low similarity
    features1 = np.zeros(256, dtype=np.float32)
    features1[:50] = 1.0
    features1 /= features1.sum()
    
    features2 = np.zeros(256, dtype=np.float32)
    features2[200:] = 1.0
    features2 /= features2.sum()
    
    similarity = compute_texture_similarity(features1, features2)
    assert similarity < 20, f"Expected low similarity, got {similarity}"
    print("  ✓ Very different features: low similarity")
    
    # Test 3: Score range
    for _ in range(10):
        f1 = np.random.rand(256).astype(np.float32)
        f2 = np.random.rand(256).astype(np.float32)
        sim = compute_texture_similarity(f1, f2)
        assert 0 <= sim <= 100, f"Score out of range: {sim}"
    print("  ✓ All scores in 0-100 range")
    
    print("✓ Texture similarity tests passed\n")


def test_combined_similarity():
    """Test combined similarity computation"""
    print("Testing combined similarity...")
    
    # Test 1: Default weights
    color_sim = 80.0
    shape_sim = 60.0
    texture_sim = 70.0
    
    combined = compute_combined_similarity(color_sim, shape_sim, texture_sim)
    expected = 0.5 * 80 + 0.3 * 60 + 0.2 * 70  # 72.0
    assert abs(combined - expected) < 0.01, f"Expected {expected}, got {combined}"
    print(f"  ✓ Default weights: {combined:.1f}% (expected {expected:.1f}%)")
    
    # Test 2: Custom weights
    combined = compute_combined_similarity(
        color_sim, shape_sim, texture_sim,
        color_weight=0.6, shape_weight=0.3, texture_weight=0.1
    )
    expected = 0.6 * 80 + 0.3 * 60 + 0.1 * 70  # 73.0
    assert abs(combined - expected) < 0.01, f"Expected {expected}, got {combined}"
    print(f"  ✓ Custom weights: {combined:.1f}% (expected {expected:.1f}%)")
    
    # Test 3: Score range
    for _ in range(10):
        c = np.random.rand() * 100
        s = np.random.rand() * 100
        t = np.random.rand() * 100
        combined = compute_combined_similarity(c, s, t)
        assert 0 <= combined <= 100, f"Score out of range: {combined}"
    print("  ✓ All scores in 0-100 range")
    
    print("✓ Combined similarity tests passed\n")


def test_compute_all_similarities():
    """Test compute_all_similarities function"""
    print("Testing compute_all_similarities...")
    
    # Create test features
    features1 = {
        'color_features': np.random.rand(256).astype(np.float32),
        'shape_features': np.random.rand(7).astype(np.float32),
        'texture_features': np.random.rand(256).astype(np.float32)
    }
    
    features2 = {
        'color_features': np.random.rand(256).astype(np.float32),
        'shape_features': np.random.rand(7).astype(np.float32),
        'texture_features': np.random.rand(256).astype(np.float32)
    }
    
    result = compute_all_similarities(features1, features2)
    
    # Check all keys present
    assert 'color_similarity' in result
    assert 'shape_similarity' in result
    assert 'texture_similarity' in result
    assert 'combined_similarity' in result
    print("  ✓ All similarity keys present")
    
    # Check all scores in valid range
    assert 0 <= result['color_similarity'] <= 100
    assert 0 <= result['shape_similarity'] <= 100
    assert 0 <= result['texture_similarity'] <= 100
    assert 0 <= result['combined_similarity'] <= 100
    print("  ✓ All scores in valid range")
    
    # Test with identical features
    result = compute_all_similarities(features1, features1)
    assert abs(result['color_similarity'] - 100.0) < 0.01
    assert abs(result['shape_similarity'] - 100.0) < 0.01
    assert abs(result['texture_similarity'] - 100.0) < 0.01
    assert abs(result['combined_similarity'] - 100.0) < 0.01
    print("  ✓ Identical features give 100% similarity")
    
    print("✓ compute_all_similarities tests passed\n")


def test_batch_compute_similarities():
    """Test batch_compute_similarities function"""
    print("Testing batch_compute_similarities...")
    
    query_features = {
        'color_features': np.random.rand(256).astype(np.float32),
        'shape_features': np.random.rand(7).astype(np.float32),
        'texture_features': np.random.rand(256).astype(np.float32)
    }
    
    candidates = [
        {
            'color_features': np.random.rand(256).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        }
        for _ in range(5)
    ]
    
    results = batch_compute_similarities(query_features, candidates)
    
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    print("  ✓ Correct number of results")
    
    for i, result in enumerate(results):
        assert 'combined_similarity' in result
        assert 0 <= result['combined_similarity'] <= 100
    print("  ✓ All results have valid scores")
    
    print("✓ batch_compute_similarities tests passed\n")


def test_error_handling():
    """Test error handling for real-world data issues"""
    print("Testing error handling for real-world data...")
    
    from similarity import (
        InvalidFeatureError,
        FeatureDimensionError,
        validate_feature_array,
        safe_normalize_histogram
    )
    
    # Test 1: NaN values
    try:
        features_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        validate_feature_array(features_with_nan, 7, "Test")
        assert False, "Should have raised InvalidFeatureError for NaN"
    except InvalidFeatureError as e:
        assert "NaN" in e.message
        print("  ✓ NaN values detected and rejected")
    
    # Test 2: Inf values
    try:
        features_with_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        validate_feature_array(features_with_inf, 7, "Test")
        assert False, "Should have raised InvalidFeatureError for Inf"
    except InvalidFeatureError as e:
        assert "infinite" in e.message
        print("  ✓ Inf values detected and rejected")
    
    # Test 3: Wrong dimensions
    try:
        features_wrong_dim = np.random.rand(10).astype(np.float32)
        validate_feature_array(features_wrong_dim, 7, "Test")
        assert False, "Should have raised FeatureDimensionError"
    except FeatureDimensionError as e:
        assert "wrong dimensions" in e.message
        print("  ✓ Wrong dimensions detected and rejected")
    
    # Test 4: Empty array
    try:
        features_empty = np.array([], dtype=np.float32)
        validate_feature_array(features_empty, 7, "Test")
        assert False, "Should have raised InvalidFeatureError for empty array"
    except InvalidFeatureError as e:
        assert "empty" in e.message
        print("  ✓ Empty arrays detected and rejected")
    
    # Test 5: None value
    try:
        validate_feature_array(None, 7, "Test")
        assert False, "Should have raised InvalidFeatureError for None"
    except InvalidFeatureError as e:
        assert "None" in e.message
        print("  ✓ None values detected and rejected")
    
    # Test 6: Zero-sum histogram normalization
    zero_hist = np.zeros(256, dtype=np.float32)
    normalized = safe_normalize_histogram(zero_hist)
    assert np.allclose(normalized.sum(), 1.0), "Normalized histogram should sum to 1"
    assert np.all(normalized > 0), "All bins should be positive"
    print("  ✓ Zero-sum histograms handled gracefully")
    
    # Test 7: Batch processing with errors
    query_features = {
        'color_features': np.random.rand(256).astype(np.float32),
        'shape_features': np.random.rand(7).astype(np.float32),
        'texture_features': np.random.rand(256).astype(np.float32)
    }
    
    candidates = [
        {
            'color_features': np.random.rand(256).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        },
        {
            # Invalid candidate - wrong dimensions
            'color_features': np.random.rand(100).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        },
        {
            'color_features': np.random.rand(256).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        }
    ]
    
    results = batch_compute_similarities(query_features, candidates, skip_errors=True)
    
    assert len(results) == 3, "Should return results for all candidates"
    assert 'error' not in results[0], "First candidate should succeed"
    assert 'error' in results[1], "Second candidate should have error"
    assert 'error_code' in results[1], "Error should include error code"
    assert 'error' not in results[2], "Third candidate should succeed"
    print("  ✓ Batch processing handles errors gracefully")
    
    print("✓ Error handling tests passed\n")


def test_edge_cases():
    """Test edge cases with real-world data"""
    print("Testing edge cases...")
    
    # Test 1: All-zero features (can happen with blank images)
    features1 = {
        'color_features': np.zeros(256, dtype=np.float32),
        'shape_features': np.ones(7, dtype=np.float32),
        'texture_features': np.zeros(256, dtype=np.float32)
    }
    
    features2 = {
        'color_features': np.random.rand(256).astype(np.float32),
        'shape_features': np.random.rand(7).astype(np.float32),
        'texture_features': np.random.rand(256).astype(np.float32)
    }
    
    # Should handle zero histograms gracefully
    result = compute_all_similarities(features1, features2)
    assert 0 <= result['combined_similarity'] <= 100
    print("  ✓ Zero histograms handled gracefully")
    
    # Test 2: Very small values (numerical precision issues)
    features1 = {
        'color_features': np.ones(256, dtype=np.float32) * 1e-10,
        'shape_features': np.ones(7, dtype=np.float32) * 1e-10,
        'texture_features': np.ones(256, dtype=np.float32) * 1e-10
    }
    
    features2 = {
        'color_features': np.ones(256, dtype=np.float32) * 1e-10,
        'shape_features': np.ones(7, dtype=np.float32) * 1e-10,
        'texture_features': np.ones(256, dtype=np.float32) * 1e-10
    }
    
    result = compute_all_similarities(features1, features2)
    assert 0 <= result['combined_similarity'] <= 100
    print("  ✓ Very small values handled correctly")
    
    # Test 3: Very large values
    features1 = {
        'color_features': np.ones(256, dtype=np.float32) * 1e6,
        'shape_features': np.ones(7, dtype=np.float32) * 100,
        'texture_features': np.ones(256, dtype=np.float32) * 1e6
    }
    
    features2 = {
        'color_features': np.ones(256, dtype=np.float32) * 1e6,
        'shape_features': np.ones(7, dtype=np.float32) * 100,
        'texture_features': np.ones(256, dtype=np.float32) * 1e6
    }
    
    result = compute_all_similarities(features1, features2)
    assert 0 <= result['combined_similarity'] <= 100
    print("  ✓ Very large values handled correctly")
    
    print("✓ Edge case tests passed\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running Similarity Computation Tests")
    print("=" * 60 + "\n")
    
    try:
        test_color_similarity()
        test_shape_similarity()
        test_texture_similarity()
        test_combined_similarity()
        test_compute_all_similarities()
        test_batch_compute_similarities()
        test_error_handling()
        test_edge_cases()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
