# Similarity Computation Error Handling

This document describes the comprehensive error handling implemented in the similarity computation module to handle real-world data issues.

## Custom Exception Classes

### `SimilarityComputationError` (Base Class)
Base exception for all similarity computation errors with structured error information:
- `message`: Human-readable error description
- `error_code`: Machine-readable error code
- `suggestion`: Actionable suggestion for fixing the issue
- `to_dict()`: Convert to dictionary for API responses

### `InvalidFeatureError`
Raised when feature vectors are invalid or corrupted:
- NaN values in features
- Infinite values in features
- Empty feature arrays
- None values
- Wrong data types

**Error Code**: `INVALID_FEATURES`

### `FeatureDimensionError`
Raised when feature dimensions don't match expected values:
- Wrong number of dimensions
- Mismatched shapes between feature vectors

**Error Code**: `DIMENSION_MISMATCH`

## Validation Functions

### `validate_feature_array(features, expected_dim, feature_name)`
Comprehensive validation for feature arrays:
- ✓ Checks for None values
- ✓ Validates numpy array type
- ✓ Checks for empty arrays
- ✓ Validates dimensions match expected size
- ✓ Detects NaN values
- ✓ Detects Inf values
- ✓ Warns about all-zero features (blank images)

### `safe_normalize_histogram(hist)`
Safely normalizes histograms with edge case handling:
- ✓ Handles zero-sum histograms (returns uniform distribution)
- ✓ Handles NaN/Inf sums
- ✓ Uses float64 for numerical precision
- ✓ Always returns valid normalized histogram

## Error Handling in Similarity Functions

### `compute_color_similarity()`
- Validates both input feature arrays
- Safely normalizes histograms (handles zero-sum)
- Checks for NaN/Inf in final result
- Returns 0.0 if computation produces invalid values

### `compute_shape_similarity()`
- Validates both input feature arrays
- Checks for NaN/Inf in distance computation
- Checks for NaN/Inf in final similarity score
- Returns 0.0 if computation produces invalid values

### `compute_texture_similarity()`
- Validates both input feature arrays
- Safely normalizes histograms (handles zero-sum)
- Checks for NaN/Inf in chi-square distance
- Checks for NaN/Inf in final similarity score
- Returns 0.0 if computation produces invalid values

### `compute_combined_similarity()`
- Validates individual similarity scores for NaN/Inf
- Validates weight constraints (sum to 1.0, non-negative)
- Validates score ranges (0-100)
- Checks for NaN/Inf in final combined score
- Returns 0.0 if computation produces invalid values

### `compute_all_similarities()`
- Validates feature dictionary structure
- Validates feature dictionary types
- Wraps individual similarity computations with error handling
- Provides detailed error context for debugging
- Re-raises errors with additional context

### `batch_compute_similarities()`
- New parameter: `skip_errors` (default: True)
- Continues processing on errors when `skip_errors=True`
- Returns error information in result dictionary:
  - `error`: Error message
  - `error_code`: Machine-readable error code
  - `suggestion`: How to fix the issue
- Sets all similarity scores to 0.0 for failed candidates
- Logs warnings for failed candidates
- Can raise exceptions if `skip_errors=False`

## Real-World Data Issues Handled

### 1. Corrupted Feature Data
**Issue**: Features extracted from corrupted images may contain NaN or Inf values

**Handling**:
- Detected by `validate_feature_array()`
- Raises `InvalidFeatureError` with clear message
- Suggestion: "Try re-processing the image"

### 2. Blank or Uniform Images
**Issue**: Blank images produce all-zero histograms (zero-sum)

**Handling**:
- Detected by `safe_normalize_histogram()`
- Returns uniform distribution instead of division by zero
- Warning logged for debugging
- Computation continues with valid values

### 3. Dimension Mismatches
**Issue**: Features from different extraction versions or corrupted database

**Handling**:
- Detected by `validate_feature_array()`
- Raises `FeatureDimensionError` with expected vs actual dimensions
- Suggestion: "Ensure features were extracted correctly"

### 4. Missing or None Features
**Issue**: Database query returns None or missing features

**Handling**:
- Detected by `validate_feature_array()`
- Raises `InvalidFeatureError` immediately
- Suggestion: "Ensure features were extracted successfully"

### 5. Numerical Precision Issues
**Issue**: Very large or very small values causing overflow/underflow

**Handling**:
- Uses float64 for all computations
- Clips final results to valid range [0, 100]
- Checks for NaN/Inf after each computation step
- Returns 0.0 as safe fallback

### 6. Batch Processing Failures
**Issue**: One corrupted product shouldn't stop entire batch

**Handling**:
- `skip_errors=True` allows processing to continue
- Failed products get error information in results
- Warnings logged for debugging
- Successful products processed normally

## Testing

Comprehensive test coverage in `test_similarity_simple.py`:

### Error Handling Tests
- ✓ NaN values detection
- ✓ Inf values detection
- ✓ Wrong dimensions detection
- ✓ Empty arrays detection
- ✓ None values detection
- ✓ Zero-sum histogram handling
- ✓ Batch processing with errors

### Edge Case Tests
- ✓ All-zero features (blank images)
- ✓ Very small values (numerical precision)
- ✓ Very large values (overflow prevention)

## Usage Examples

### Basic Usage with Error Handling
```python
from similarity import compute_all_similarities, InvalidFeatureError

try:
    similarities = compute_all_similarities(features1, features2)
    print(f"Similarity: {similarities['combined_similarity']:.1f}%")
except InvalidFeatureError as e:
    print(f"Error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
    # Log error for debugging
    error_info = e.to_dict()
```

### Batch Processing with Error Handling
```python
from similarity import batch_compute_similarities

# Process all candidates, skip errors
results = batch_compute_similarities(
    query_features,
    candidate_features_list,
    skip_errors=True  # Continue on errors
)

# Filter out failed candidates
successful_results = [r for r in results if 'error' not in r]
failed_results = [r for r in results if 'error' in r]

print(f"Successful: {len(successful_results)}")
print(f"Failed: {len(failed_results)}")

# Log failures for debugging
for i, result in enumerate(failed_results):
    print(f"Candidate {i}: {result['error']}")
    print(f"  Code: {result['error_code']}")
    print(f"  Suggestion: {result['suggestion']}")
```

### Strict Mode (Raise on First Error)
```python
# Raise exception on first error
try:
    results = batch_compute_similarities(
        query_features,
        candidate_features_list,
        skip_errors=False  # Raise on errors
    )
except InvalidFeatureError as e:
    print(f"Batch processing failed: {e.message}")
    # Handle error appropriately
```

## Integration with Other Modules

The error handling integrates seamlessly with:

1. **Image Processing Module** (`image_processing.py`)
   - Uses similar error class structure
   - Consistent error codes and suggestions
   - Compatible exception handling patterns

2. **Feature Cache Module** (`feature_cache.py`)
   - Can catch and handle similarity errors
   - Logs errors for corrupted cached features
   - Triggers re-extraction if needed

3. **API Layer** (future)
   - `to_dict()` method provides API-ready error responses
   - Error codes enable client-side error handling
   - Suggestions help users fix issues

## Best Practices

1. **Always validate features before similarity computation**
   - Use `validate_feature_array()` for custom validation
   - Let similarity functions handle validation automatically

2. **Use batch processing with skip_errors=True for production**
   - Prevents one bad product from stopping entire batch
   - Provides error information for debugging
   - Maintains system reliability

3. **Log warnings and errors for debugging**
   - Warnings indicate potential data quality issues
   - Error logs help identify systematic problems
   - Monitor error rates to detect issues early

4. **Handle zero-sum histograms gracefully**
   - `safe_normalize_histogram()` handles this automatically
   - Blank images get uniform distribution
   - Computation continues with valid values

5. **Check for NaN/Inf at each computation step**
   - Prevents error propagation
   - Provides clear error messages
   - Returns safe fallback values (0.0)
