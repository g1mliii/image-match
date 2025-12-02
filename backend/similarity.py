"""
Similarity computation module for product matching.

This module implements various similarity metrics for comparing visual features:
- Color similarity using histogram intersection
- Shape similarity using Euclidean distance on Hu moments
- Texture similarity using chi-square distance on LBP histograms
- Combined similarity scoring with configurable weights

All similarity scores are normalized to 0-100 range where:
- 100 = identical features
- 0 = completely different features

Error Handling:
- Handles NaN/Inf values in feature vectors
- Handles empty or zero-sum histograms
- Validates feature dimensions and types
- Provides meaningful error messages for debugging
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class SimilarityComputationError(Exception):
    """Base exception for similarity computation errors"""
    def __init__(self, message: str, error_code: str, suggestion: str = None):
        self.message = message
        self.error_code = error_code
        self.suggestion = suggestion
        super().__init__(self.message)
    
    def to_dict(self):
        """Convert error to dictionary for API responses"""
        return {
            'error': self.message,
            'error_code': self.error_code,
            'suggestion': self.suggestion
        }


class InvalidFeatureError(SimilarityComputationError):
    """Raised when feature vectors are invalid or corrupted"""
    def __init__(self, message: str, suggestion: str = None):
        super().__init__(
            message,
            'INVALID_FEATURES',
            suggestion or 'Feature vectors may be corrupted. Try re-extracting features from the image.'
        )


class FeatureDimensionError(SimilarityComputationError):
    """Raised when feature dimensions don't match expected values"""
    def __init__(self, message: str, suggestion: str = None):
        super().__init__(
            message,
            'DIMENSION_MISMATCH',
            suggestion or 'Feature vectors have incompatible dimensions. Ensure features were extracted correctly.'
        )


def validate_feature_array(features: np.ndarray, expected_dim: int, feature_name: str) -> None:
    """
    Validate feature array for common real-world data issues.
    
    Args:
        features: Feature array to validate
        expected_dim: Expected number of dimensions
        feature_name: Name of feature type for error messages
    
    Raises:
        InvalidFeatureError: If features contain NaN, Inf, or are empty
        FeatureDimensionError: If dimensions don't match expected
    """
    # Check if array is None or empty
    if features is None:
        raise InvalidFeatureError(
            f"{feature_name} features are None",
            "Ensure features were extracted successfully before computing similarity."
        )
    
    if not isinstance(features, np.ndarray):
        raise InvalidFeatureError(
            f"{feature_name} features must be numpy array, got {type(features)}",
            "Ensure features are in correct format."
        )
    
    if features.size == 0:
        raise InvalidFeatureError(
            f"{feature_name} features are empty",
            "Feature extraction may have failed. Check the input image."
        )
    
    # Check dimensions
    if len(features) != expected_dim:
        raise FeatureDimensionError(
            f"{feature_name} features have wrong dimensions. Expected {expected_dim}, got {len(features)}",
            f"Ensure {feature_name} features are extracted correctly."
        )
    
    # Check for NaN values
    if np.any(np.isnan(features)):
        raise InvalidFeatureError(
            f"{feature_name} features contain NaN values",
            "Feature extraction produced invalid values. Try re-processing the image."
        )
    
    # Check for Inf values
    if np.any(np.isinf(features)):
        raise InvalidFeatureError(
            f"{feature_name} features contain infinite values",
            "Feature extraction produced invalid values. Try re-processing the image."
        )
    
    # Check for all zeros (can happen with corrupted images)
    if np.all(features == 0):
        warnings.warn(f"{feature_name} features are all zeros. This may indicate a problem with feature extraction.")


def safe_normalize_histogram(hist: np.ndarray) -> np.ndarray:
    """
    Safely normalize histogram, handling edge cases.
    
    Args:
        hist: Histogram to normalize
    
    Returns:
        Normalized histogram (sums to 1)
    """
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    
    # Handle zero-sum histogram (can happen with blank images)
    if hist_sum == 0 or np.isnan(hist_sum) or np.isinf(hist_sum):
        # Return uniform distribution
        return np.ones_like(hist) / len(hist)
    
    return hist / hist_sum


def compute_color_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compute color similarity using histogram intersection.
    
    Histogram intersection measures the overlap between two normalized histograms.
    It's particularly effective for color histograms as it's robust to partial occlusions
    and changes in image scale.
    
    Args:
        features1: First color feature vector (256-dimensional HSV histogram)
        features2: Second color feature vector (256-dimensional HSV histogram)
    
    Returns:
        Similarity score in range 0-100, where 100 is identical
    
    Raises:
        InvalidFeatureError: If features contain NaN, Inf, or are empty
        FeatureDimensionError: If feature dimensions are incorrect
    """
    # Validate inputs
    validate_feature_array(features1, 256, "Color")
    validate_feature_array(features2, 256, "Color")
    
    if features1.shape != features2.shape:
        raise FeatureDimensionError(
            f"Feature vectors must have same shape. "
            f"Got {features1.shape} and {features2.shape}"
        )
    
    # Safely normalize histograms (handles zero-sum and edge cases)
    features1 = safe_normalize_histogram(features1)
    features2 = safe_normalize_histogram(features2)
    
    # Compute histogram intersection
    # Intersection = sum of minimum values at each bin
    intersection = np.minimum(features1, features2).sum()
    
    # Convert to 0-100 scale
    # Intersection is already in [0, 1] range for normalized histograms
    similarity = intersection * 100.0
    
    # Ensure result is in valid range (handle any numerical issues)
    if np.isnan(similarity) or np.isinf(similarity):
        warnings.warn("Color similarity computation produced NaN/Inf, returning 0")
        return 0.0
    
    similarity = np.clip(similarity, 0.0, 100.0)
    
    return float(similarity)


def compute_shape_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compute shape similarity using Euclidean distance on Hu moments.
    
    Hu moments are scale, rotation, and translation invariant shape descriptors.
    We use Euclidean distance and convert it to a similarity score.
    
    Args:
        features1: First shape feature vector (7-dimensional Hu moments)
        features2: Second shape feature vector (7-dimensional Hu moments)
    
    Returns:
        Similarity score in range 0-100, where 100 is identical
    
    Raises:
        InvalidFeatureError: If features contain NaN, Inf, or are empty
        FeatureDimensionError: If feature dimensions are incorrect
    """
    # Validate inputs
    validate_feature_array(features1, 7, "Shape")
    validate_feature_array(features2, 7, "Shape")
    
    if features1.shape != features2.shape:
        raise FeatureDimensionError(
            f"Feature vectors must have same shape. "
            f"Got {features1.shape} and {features2.shape}"
        )
    
    # Convert to float64 for precision
    features1 = features1.astype(np.float64)
    features2 = features2.astype(np.float64)
    
    # Compute Euclidean distance
    distance = np.linalg.norm(features1 - features2)
    
    # Check for invalid distance
    if np.isnan(distance) or np.isinf(distance):
        warnings.warn("Shape distance computation produced NaN/Inf, returning 0 similarity")
        return 0.0
    
    # Convert distance to similarity score
    # Use exponential decay: similarity = 100 * exp(-distance / scale)
    # Scale factor chosen empirically to give good discrimination
    # Typical Hu moment distances range from 0 to ~10 for different shapes
    scale = 2.0
    similarity = 100.0 * np.exp(-distance / scale)
    
    # Ensure result is in valid range
    if np.isnan(similarity) or np.isinf(similarity):
        warnings.warn("Shape similarity computation produced NaN/Inf, returning 0")
        return 0.0
    
    similarity = np.clip(similarity, 0.0, 100.0)
    
    return float(similarity)


def compute_texture_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compute texture similarity using chi-square distance on LBP histograms.
    
    Chi-square distance is effective for comparing histograms as it accounts for
    the statistical significance of differences between bins.
    
    Args:
        features1: First texture feature vector (256-dimensional LBP histogram)
        features2: Second texture feature vector (256-dimensional LBP histogram)
    
    Returns:
        Similarity score in range 0-100, where 100 is identical
    
    Raises:
        InvalidFeatureError: If features contain NaN, Inf, or are empty
        FeatureDimensionError: If feature dimensions are incorrect
    """
    # Validate inputs
    validate_feature_array(features1, 256, "Texture")
    validate_feature_array(features2, 256, "Texture")
    
    if features1.shape != features2.shape:
        raise FeatureDimensionError(
            f"Feature vectors must have same shape. "
            f"Got {features1.shape} and {features2.shape}"
        )
    
    # Safely normalize histograms (handles zero-sum and edge cases)
    features1 = safe_normalize_histogram(features1)
    features2 = safe_normalize_histogram(features2)
    
    # Compute chi-square distance
    # chi^2 = 0.5 * sum((h1[i] - h2[i])^2 / (h1[i] + h2[i]))
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    denominator = features1 + features2 + epsilon
    numerator = (features1 - features2) ** 2
    chi_square = 0.5 * np.sum(numerator / denominator)
    
    # Check for invalid chi-square value
    if np.isnan(chi_square) or np.isinf(chi_square):
        warnings.warn("Texture chi-square computation produced NaN/Inf, returning 0 similarity")
        return 0.0
    
    # Convert chi-square distance to similarity score
    # Use exponential decay: similarity = 100 * exp(-chi_square / scale)
    # Scale factor chosen empirically
    # Typical chi-square distances range from 0 to ~2 for different textures
    scale = 0.5
    similarity = 100.0 * np.exp(-chi_square / scale)
    
    # Ensure result is in valid range
    if np.isnan(similarity) or np.isinf(similarity):
        warnings.warn("Texture similarity computation produced NaN/Inf, returning 0")
        return 0.0
    
    similarity = np.clip(similarity, 0.0, 100.0)
    
    return float(similarity)


def compute_combined_similarity(
    color_sim: float,
    shape_sim: float,
    texture_sim: float,
    color_weight: float = 0.5,
    shape_weight: float = 0.3,
    texture_weight: float = 0.2
) -> float:
    """
    Combine individual similarity scores with configurable weights.
    
    The default weights prioritize color (0.5) over shape (0.3) and texture (0.2),
    as color is often the most discriminative feature for product matching.
    
    Args:
        color_sim: Color similarity score (0-100)
        shape_sim: Shape similarity score (0-100)
        texture_sim: Texture similarity score (0-100)
        color_weight: Weight for color similarity (default: 0.5)
        shape_weight: Weight for shape similarity (default: 0.3)
        texture_weight: Weight for texture similarity (default: 0.2)
    
    Returns:
        Combined similarity score in range 0-100
    
    Raises:
        ValueError: If weights don't sum to 1.0 or scores are out of range
        InvalidFeatureError: If scores contain NaN or Inf
    """
    # Check for NaN/Inf in scores
    for score, name in [(color_sim, 'color'), (shape_sim, 'shape'), (texture_sim, 'texture')]:
        if np.isnan(score) or np.isinf(score):
            raise InvalidFeatureError(
                f"{name} similarity score is NaN or Inf: {score}",
                "Individual similarity computation may have failed. Check feature quality."
            )
    
    # Validate weights
    total_weight = color_weight + shape_weight + texture_weight
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        raise ValueError(
            f"Weights must sum to 1.0, got {total_weight}. "
            f"Weights: color={color_weight}, shape={shape_weight}, texture={texture_weight}"
        )
    
    if color_weight < 0 or shape_weight < 0 or texture_weight < 0:
        raise ValueError("Weights must be non-negative")
    
    # Validate similarity scores
    for score, name in [(color_sim, 'color'), (shape_sim, 'shape'), (texture_sim, 'texture')]:
        if not 0 <= score <= 100:
            raise ValueError(f"{name} similarity score must be in range 0-100, got {score}")
    
    # Compute weighted average
    combined = (
        color_weight * color_sim +
        shape_weight * shape_sim +
        texture_weight * texture_sim
    )
    
    # Ensure result is in valid range (should already be, but just in case)
    if np.isnan(combined) or np.isinf(combined):
        warnings.warn("Combined similarity computation produced NaN/Inf, returning 0")
        return 0.0
    
    combined = np.clip(combined, 0.0, 100.0)
    
    return float(combined)


def compute_all_similarities(
    features1: Dict[str, np.ndarray],
    features2: Dict[str, np.ndarray],
    color_weight: float = 0.5,
    shape_weight: float = 0.3,
    texture_weight: float = 0.2
) -> Dict[str, float]:
    """
    Compute all similarity scores between two feature sets.
    
    This is a convenience function that computes color, shape, texture,
    and combined similarity scores in one call.
    
    Args:
        features1: First feature dictionary with keys:
                  'color_features', 'shape_features', 'texture_features'
        features2: Second feature dictionary with same keys
        color_weight: Weight for color similarity (default: 0.5)
        shape_weight: Weight for shape similarity (default: 0.3)
        texture_weight: Weight for texture similarity (default: 0.2)
    
    Returns:
        Dictionary with keys:
        - 'color_similarity': Color similarity score (0-100)
        - 'shape_similarity': Shape similarity score (0-100)
        - 'texture_similarity': Texture similarity score (0-100)
        - 'combined_similarity': Combined similarity score (0-100)
    
    Raises:
        ValueError: If feature dictionaries are missing required keys
        InvalidFeatureError: If features are corrupted or invalid
        FeatureDimensionError: If feature dimensions are incorrect
    """
    # Validate feature dictionaries
    required_keys = {'color_features', 'shape_features', 'texture_features'}
    
    if not isinstance(features1, dict):
        raise ValueError(f"features1 must be a dictionary, got {type(features1)}")
    
    if not isinstance(features2, dict):
        raise ValueError(f"features2 must be a dictionary, got {type(features2)}")
    
    if not required_keys.issubset(features1.keys()):
        missing = required_keys - set(features1.keys())
        raise ValueError(f"features1 missing required keys: {missing}")
    
    if not required_keys.issubset(features2.keys()):
        missing = required_keys - set(features2.keys())
        raise ValueError(f"features2 missing required keys: {missing}")
    
    # Compute individual similarities with error handling
    try:
        color_sim = compute_color_similarity(
            features1['color_features'],
            features2['color_features']
        )
    except (InvalidFeatureError, FeatureDimensionError) as e:
        # Re-raise with more context
        raise InvalidFeatureError(
            f"Failed to compute color similarity: {e.message}",
            e.suggestion
        )
    
    try:
        shape_sim = compute_shape_similarity(
            features1['shape_features'],
            features2['shape_features']
        )
    except (InvalidFeatureError, FeatureDimensionError) as e:
        raise InvalidFeatureError(
            f"Failed to compute shape similarity: {e.message}",
            e.suggestion
        )
    
    try:
        texture_sim = compute_texture_similarity(
            features1['texture_features'],
            features2['texture_features']
        )
    except (InvalidFeatureError, FeatureDimensionError) as e:
        raise InvalidFeatureError(
            f"Failed to compute texture similarity: {e.message}",
            e.suggestion
        )
    
    # Compute combined similarity
    try:
        combined_sim = compute_combined_similarity(
            color_sim,
            shape_sim,
            texture_sim,
            color_weight,
            shape_weight,
            texture_weight
        )
    except (InvalidFeatureError, ValueError) as e:
        raise InvalidFeatureError(
            f"Failed to compute combined similarity: {str(e)}",
            "Check that individual similarity scores are valid."
        )
    
    return {
        'color_similarity': color_sim,
        'shape_similarity': shape_sim,
        'texture_similarity': texture_sim,
        'combined_similarity': combined_sim
    }


def _compute_similarity_worker(args):
    """Worker function for multiprocessing similarity computation
    
    Args:
        args: Tuple of (query_features, candidate_features, color_weight, shape_weight, texture_weight)
    
    Returns:
        Dictionary with similarity scores or error info
    """
    query_features, candidate_features, color_weight, shape_weight, texture_weight = args
    
    try:
        return compute_all_similarities(
            query_features,
            candidate_features,
            color_weight,
            shape_weight,
            texture_weight
        )
    except (InvalidFeatureError, FeatureDimensionError) as e:
        return {
            'color_similarity': 0.0,
            'shape_similarity': 0.0,
            'texture_similarity': 0.0,
            'combined_similarity': 0.0,
            'error': e.message,
            'error_code': e.error_code,
            'suggestion': e.suggestion
        }
    except Exception as e:
        return {
            'color_similarity': 0.0,
            'shape_similarity': 0.0,
            'texture_similarity': 0.0,
            'combined_similarity': 0.0,
            'error': str(e),
            'error_code': 'UNKNOWN_ERROR'
        }


def batch_compute_similarities(
    query_features: Dict[str, np.ndarray],
    candidate_features_list: List[Dict[str, np.ndarray]],
    color_weight: float = 0.5,
    shape_weight: float = 0.3,
    texture_weight: float = 0.2,
    skip_errors: bool = True,
    use_multiprocessing: bool = None,
    max_workers: int = None
) -> List[Dict[str, float]]:
    """
    Compute similarities between one query and multiple candidates.
    
    This is optimized for the common use case of comparing one new product
    against many historical products. Handles real-world data issues gracefully.
    
    PERFORMANCE OPTIMIZATIONS:
    - For large catalogs (>1000 products), uses multiprocessing for CPU parallelization
    - For smaller catalogs, uses sequential processing (less overhead)
    - Numpy vectorization for fast computation
    
    Args:
        query_features: Query product features
        candidate_features_list: List of candidate product features
        color_weight: Weight for color similarity (default: 0.5)
        shape_weight: Weight for shape similarity (default: 0.3)
        texture_weight: Weight for texture similarity (default: 0.2)
        skip_errors: If True, continue processing on errors and return error info.
                    If False, raise exception on first error (default: True)
        use_multiprocessing: Force multiprocessing on/off. If None, auto-detect based on catalog size
        max_workers: Number of worker processes (default: cpu_count - 1)
    
    Returns:
        List of similarity dictionaries, one for each candidate.
        On error (if skip_errors=True), includes 'error' and 'error_code' keys.
    
    Raises:
        ValueError: If feature dictionaries are invalid (only if skip_errors=False)
        InvalidFeatureError: If features are corrupted (only if skip_errors=False)
    """
    # Determine if we should use multiprocessing
    # Only beneficial for large catalogs (>1000 products)
    if use_multiprocessing is None:
        use_multiprocessing = len(candidate_features_list) > 1000
    
    # Sequential processing for small catalogs or if multiprocessing disabled
    if not use_multiprocessing or len(candidate_features_list) < 100:
        results = []
        
        for i, candidate_features in enumerate(candidate_features_list):
            try:
                similarities = compute_all_similarities(
                    query_features,
                    candidate_features,
                    color_weight,
                    shape_weight,
                    texture_weight
                )
                results.append(similarities)
            except (InvalidFeatureError, FeatureDimensionError) as e:
                if not skip_errors:
                    raise
                
                # Add error result with details
                warnings.warn(f"Failed to compute similarity for candidate {i}: {e.message}")
                results.append({
                    'color_similarity': 0.0,
                    'shape_similarity': 0.0,
                    'texture_similarity': 0.0,
                    'combined_similarity': 0.0,
                    'error': e.message,
                    'error_code': e.error_code,
                    'suggestion': e.suggestion
                })
            except Exception as e:
                if not skip_errors:
                    raise
                
                # Add generic error result
                warnings.warn(f"Unexpected error computing similarity for candidate {i}: {str(e)}")
                results.append({
                    'color_similarity': 0.0,
                    'shape_similarity': 0.0,
                    'texture_similarity': 0.0,
                    'combined_similarity': 0.0,
                    'error': str(e),
                    'error_code': 'UNKNOWN_ERROR'
                })
        
        return results
    
    # Multiprocessing for large catalogs
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Determine number of workers
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        max_workers = max(1, cpu_count - 1)
    
    # Prepare arguments for workers
    worker_args = [
        (query_features, candidate_features, color_weight, shape_weight, texture_weight)
        for candidate_features in candidate_features_list
    ]
    
    results = []
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(_compute_similarity_worker, args) for args in worker_args]
            
            # Collect results in order
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    if not skip_errors:
                        raise
                    
                    warnings.warn(f"Process failed: {str(e)}")
                    results.append({
                        'color_similarity': 0.0,
                        'shape_similarity': 0.0,
                        'texture_similarity': 0.0,
                        'combined_similarity': 0.0,
                        'error': str(e),
                        'error_code': 'PROCESS_ERROR'
                    })
    
    except Exception as e:
        if not skip_errors:
            raise
        
        warnings.warn(f"Multiprocessing failed, falling back to sequential: {str(e)}")
        # Fallback to sequential processing
        return batch_compute_similarities(
            query_features,
            candidate_features_list,
            color_weight,
            shape_weight,
            texture_weight,
            skip_errors,
            use_multiprocessing=False
        )
    
    return results
