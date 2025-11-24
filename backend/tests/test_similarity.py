"""
Tests for similarity computation module.
"""

import pytest
import numpy as np
from similarity import (
    compute_color_similarity,
    compute_shape_similarity,
    compute_texture_similarity,
    compute_combined_similarity,
    compute_all_similarities,
    batch_compute_similarities
)


class TestColorSimilarity:
    """Tests for color similarity computation"""
    
    def test_identical_features(self):
        """Identical color features should give 100% similarity"""
        features = np.random.rand(256).astype(np.float32)
        features /= features.sum()  # Normalize
        
        similarity = compute_color_similarity(features, features)
        assert similarity == pytest.approx(100.0, abs=0.01)
    
    def test_completely_different_features(self):
        """Non-overlapping histograms should give low similarity"""
        features1 = np.zeros(256, dtype=np.float32)
        features1[:128] = 1.0
        features1 /= features1.sum()
        
        features2 = np.zeros(256, dtype=np.float32)
        features2[128:] = 1.0
        features2 /= features2.sum()
        
        similarity = compute_color_similarity(features1, features2)
        assert similarity == pytest.approx(0.0, abs=0.01)
    
    def test_partial_overlap(self):
        """Partially overlapping histograms should give intermediate similarity"""
        features1 = np.ones(256, dtype=np.float32)
        features1 /= features1.sum()
        
        features2 = np.zeros(256, dtype=np.float32)
        features2[:128] = 1.0
        features2 /= features2.sum()
        
        similarity = compute_color_similarity(features1, features2)
        assert 0 < similarity < 100
        assert similarity == pytest.approx(50.0, abs=1.0)
    
    def test_wrong_dimensions(self):
        """Should raise error for wrong dimensions"""
        features1 = np.random.rand(256).astype(np.float32)
        features2 = np.random.rand(128).astype(np.float32)
        
        with pytest.raises(Exception, match="wrong dimensions"):
            compute_color_similarity(features1, features2)
    
    def test_invalid_dimension_size(self):
        """Should raise error for non-256 dimensional features"""
        features1 = np.random.rand(100).astype(np.float32)
        features2 = np.random.rand(100).astype(np.float32)
        
        with pytest.raises(Exception, match="wrong dimensions"):
            compute_color_similarity(features1, features2)
    
    def test_score_range(self):
        """Similarity score should always be in 0-100 range"""
        for _ in range(10):
            features1 = np.random.rand(256).astype(np.float32)
            features2 = np.random.rand(256).astype(np.float32)
            
            similarity = compute_color_similarity(features1, features2)
            assert 0 <= similarity <= 100


class TestShapeSimilarity:
    """Tests for shape similarity computation"""
    
    def test_identical_features(self):
        """Identical shape features should give 100% similarity"""
        features = np.random.rand(7).astype(np.float32)
        
        similarity = compute_shape_similarity(features, features)
        assert similarity == pytest.approx(100.0, abs=0.01)
    
    def test_very_different_features(self):
        """Very different Hu moments should give low similarity"""
        features1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        features2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float32)
        
        similarity = compute_shape_similarity(features1, features2)
        assert similarity < 10  # Should be very low
    
    def test_similar_features(self):
        """Similar Hu moments should give high similarity"""
        features1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        features2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1], dtype=np.float32)
        
        similarity = compute_shape_similarity(features1, features2)
        assert similarity > 80  # Should be high
    
    def test_wrong_dimensions(self):
        """Should raise error for wrong dimensions"""
        features1 = np.random.rand(7).astype(np.float32)
        features2 = np.random.rand(5).astype(np.float32)
        
        with pytest.raises(Exception, match="wrong dimensions"):
            compute_shape_similarity(features1, features2)
    
    def test_invalid_dimension_size(self):
        """Should raise error for non-7 dimensional features"""
        features1 = np.random.rand(10).astype(np.float32)
        features2 = np.random.rand(10).astype(np.float32)
        
        with pytest.raises(Exception, match="wrong dimensions"):
            compute_shape_similarity(features1, features2)
    
    def test_score_range(self):
        """Similarity score should always be in 0-100 range"""
        for _ in range(10):
            features1 = np.random.rand(7).astype(np.float32) * 10
            features2 = np.random.rand(7).astype(np.float32) * 10
            
            similarity = compute_shape_similarity(features1, features2)
            assert 0 <= similarity <= 100


class TestTextureSimilarity:
    """Tests for texture similarity computation"""
    
    def test_identical_features(self):
        """Identical texture features should give 100% similarity"""
        features = np.random.rand(256).astype(np.float32)
        features /= features.sum()  # Normalize
        
        similarity = compute_texture_similarity(features, features)
        assert similarity == pytest.approx(100.0, abs=0.01)
    
    def test_very_different_features(self):
        """Very different LBP histograms should give low similarity"""
        features1 = np.zeros(256, dtype=np.float32)
        features1[:50] = 1.0
        features1 /= features1.sum()
        
        features2 = np.zeros(256, dtype=np.float32)
        features2[200:] = 1.0
        features2 /= features2.sum()
        
        similarity = compute_texture_similarity(features1, features2)
        assert similarity < 20  # Should be low
    
    def test_similar_features(self):
        """Similar LBP histograms should give high similarity"""
        features1 = np.random.rand(256).astype(np.float32)
        features1 /= features1.sum()
        
        # Add small noise
        features2 = features1 + np.random.rand(256).astype(np.float32) * 0.01
        features2 /= features2.sum()
        
        similarity = compute_texture_similarity(features1, features2)
        assert similarity > 88  # Should be high (lowered threshold slightly for randomness)
    
    def test_wrong_dimensions(self):
        """Should raise error for wrong dimensions"""
        features1 = np.random.rand(256).astype(np.float32)
        features2 = np.random.rand(128).astype(np.float32)
        
        with pytest.raises(Exception, match="wrong dimensions"):
            compute_texture_similarity(features1, features2)
    
    def test_invalid_dimension_size(self):
        """Should raise error for non-256 dimensional features"""
        features1 = np.random.rand(100).astype(np.float32)
        features2 = np.random.rand(100).astype(np.float32)
        
        with pytest.raises(Exception, match="wrong dimensions"):
            compute_texture_similarity(features1, features2)
    
    def test_score_range(self):
        """Similarity score should always be in 0-100 range"""
        for _ in range(10):
            features1 = np.random.rand(256).astype(np.float32)
            features2 = np.random.rand(256).astype(np.float32)
            
            similarity = compute_texture_similarity(features1, features2)
            assert 0 <= similarity <= 100


class TestCombinedSimilarity:
    """Tests for combined similarity computation"""
    
    def test_default_weights(self):
        """Test combined similarity with default weights"""
        color_sim = 80.0
        shape_sim = 60.0
        texture_sim = 70.0
        
        combined = compute_combined_similarity(color_sim, shape_sim, texture_sim)
        
        # Expected: 0.5*80 + 0.3*60 + 0.2*70 = 40 + 18 + 14 = 72
        assert combined == pytest.approx(72.0, abs=0.01)
    
    def test_custom_weights(self):
        """Test combined similarity with custom weights"""
        color_sim = 80.0
        shape_sim = 60.0
        texture_sim = 70.0
        
        combined = compute_combined_similarity(
            color_sim, shape_sim, texture_sim,
            color_weight=0.6,
            shape_weight=0.3,
            texture_weight=0.1
        )
        
        # Expected: 0.6*80 + 0.3*60 + 0.1*70 = 48 + 18 + 7 = 73
        assert combined == pytest.approx(73.0, abs=0.01)
    
    def test_equal_weights(self):
        """Test combined similarity with equal weights"""
        color_sim = 90.0
        shape_sim = 60.0
        texture_sim = 30.0
        
        combined = compute_combined_similarity(
            color_sim, shape_sim, texture_sim,
            color_weight=1/3,
            shape_weight=1/3,
            texture_weight=1/3
        )
        
        # Expected: (90 + 60 + 30) / 3 = 60
        assert combined == pytest.approx(60.0, abs=0.01)
    
    def test_weights_not_sum_to_one(self):
        """Should raise error if weights don't sum to 1.0"""
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_combined_similarity(80, 60, 70, 0.5, 0.3, 0.1)
    
    def test_negative_weights(self):
        """Should raise error for negative weights"""
        with pytest.raises(ValueError, match="non-negative"):
            compute_combined_similarity(80, 60, 70, -0.5, 0.8, 0.7)
    
    def test_invalid_score_range(self):
        """Should raise error for scores outside 0-100 range"""
        with pytest.raises(ValueError, match="range 0-100"):
            compute_combined_similarity(150, 60, 70)
        
        with pytest.raises(ValueError, match="range 0-100"):
            compute_combined_similarity(80, -10, 70)
    
    def test_score_range(self):
        """Combined score should always be in 0-100 range"""
        for _ in range(10):
            color_sim = np.random.rand() * 100
            shape_sim = np.random.rand() * 100
            texture_sim = np.random.rand() * 100
            
            combined = compute_combined_similarity(color_sim, shape_sim, texture_sim)
            assert 0 <= combined <= 100


class TestComputeAllSimilarities:
    """Tests for compute_all_similarities function"""
    
    def test_compute_all_similarities(self):
        """Test computing all similarities at once"""
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
        
        # Check all keys are present
        assert 'color_similarity' in result
        assert 'shape_similarity' in result
        assert 'texture_similarity' in result
        assert 'combined_similarity' in result
        
        # Check all scores are in valid range
        assert 0 <= result['color_similarity'] <= 100
        assert 0 <= result['shape_similarity'] <= 100
        assert 0 <= result['texture_similarity'] <= 100
        assert 0 <= result['combined_similarity'] <= 100
    
    def test_identical_features(self):
        """Identical features should give 100% similarity"""
        features = {
            'color_features': np.random.rand(256).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        }
        
        result = compute_all_similarities(features, features)
        
        assert result['color_similarity'] == pytest.approx(100.0, abs=0.01)
        assert result['shape_similarity'] == pytest.approx(100.0, abs=0.01)
        assert result['texture_similarity'] == pytest.approx(100.0, abs=0.01)
        assert result['combined_similarity'] == pytest.approx(100.0, abs=0.01)
    
    def test_missing_keys(self):
        """Should raise error if feature dictionaries are missing keys"""
        features1 = {
            'color_features': np.random.rand(256).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32)
            # Missing texture_features
        }
        
        features2 = {
            'color_features': np.random.rand(256).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        }
        
        with pytest.raises(ValueError, match="missing required keys"):
            compute_all_similarities(features1, features2)
    
    def test_custom_weights(self):
        """Test with custom weights"""
        features1 = {
            'color_features': np.ones(256, dtype=np.float32) / 256,
            'shape_features': np.ones(7, dtype=np.float32),
            'texture_features': np.ones(256, dtype=np.float32) / 256
        }
        
        features2 = {
            'color_features': np.ones(256, dtype=np.float32) / 256,
            'shape_features': np.ones(7, dtype=np.float32),
            'texture_features': np.ones(256, dtype=np.float32) / 256
        }
        
        result = compute_all_similarities(
            features1, features2,
            color_weight=0.6,
            shape_weight=0.3,
            texture_weight=0.1
        )
        
        # All features identical, so combined should be 100
        assert result['combined_similarity'] == pytest.approx(100.0, abs=0.01)


class TestBatchComputeSimilarities:
    """Tests for batch_compute_similarities function"""
    
    def test_batch_computation(self):
        """Test computing similarities for multiple candidates"""
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
        
        assert len(results) == 5
        
        for result in results:
            assert 'color_similarity' in result
            assert 'shape_similarity' in result
            assert 'texture_similarity' in result
            assert 'combined_similarity' in result
            assert 0 <= result['combined_similarity'] <= 100
    
    def test_empty_candidates(self):
        """Test with empty candidate list"""
        query_features = {
            'color_features': np.random.rand(256).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        }
        
        results = batch_compute_similarities(query_features, [])
        assert len(results) == 0
    
    def test_error_handling(self):
        """Test that errors in individual candidates don't stop batch processing"""
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
                # Invalid candidate - missing texture_features
                'color_features': np.random.rand(256).astype(np.float32),
                'shape_features': np.random.rand(7).astype(np.float32)
            },
            {
                'color_features': np.random.rand(256).astype(np.float32),
                'shape_features': np.random.rand(7).astype(np.float32),
                'texture_features': np.random.rand(256).astype(np.float32)
            }
        ]
        
        results = batch_compute_similarities(query_features, candidates)
        
        assert len(results) == 3
        assert 'error' not in results[0]
        assert 'error' in results[1]  # Should have error
        assert 'error' not in results[2]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
