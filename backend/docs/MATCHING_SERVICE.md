# Product Matching Service Documentation

## Overview

The Product Matching Service implements intelligent product matching with category filtering and comprehensive real-world data handling. It compares new products against a historical catalog using visual similarity analysis.

**Key Feature: Production-Ready Error Handling**

This service is designed to handle messy real-world data gracefully:
- Corrupted or missing image features (NaN, Inf, empty arrays)
- Missing metadata fields (NULL names, SKUs, categories)
- Wrong data formatting and type mismatches
- Unopenable or corrupted images
- Database errors and connection issues
- Mixed good and bad data in the same catalog
- **No file size limits** - handles large images gracefully

## Features

### Core Functionality

1. **Category-Based Filtering**
   - Matches products within the same category
   - Handles NULL/missing categories gracefully
   - Fallback to match against all categories when needed

2. **Robust Error Handling**
   - Gracefully handles missing or corrupted features
   - Skips problematic products and continues processing
   - Provides detailed error information and suggestions

3. **Flexible Matching Options**
   - Configurable similarity threshold (0-100)
   - Result limiting (top N matches)
   - Customizable feature weights (color, shape, texture)
   - Option to match against all categories

4. **Duplicate Detection**
   - Automatically flags matches with score > 90 as potential duplicates
   - Useful for identifying duplicate products in catalog

5. **Comprehensive Reporting**
   - Detailed match results with similarity scores
   - Breakdown by feature type (color, shape, texture)
   - Warnings for data quality issues
   - Error details for failed matches

## API Reference

### `find_matches()`

Main function for finding similar products.

```python
def find_matches(
    product_id: int,
    threshold: float = 0.0,
    limit: int = 10,
    match_against_all: bool = False,
    include_uncategorized: bool = True,
    color_weight: float = 0.5,
    shape_weight: float = 0.3,
    texture_weight: float = 0.2,
    store_matches: bool = True
) -> Dict[str, Any]
```

**Parameters:**
- `product_id`: ID of the product to match
- `threshold`: Minimum similarity score (0-100) to include in results
- `limit`: Maximum number of matches to return
- `match_against_all`: If True, match against all categories
- `include_uncategorized`: If True, include products with NULL category
- `color_weight`: Weight for color similarity (default: 0.5)
- `shape_weight`: Weight for shape similarity (default: 0.3)
- `texture_weight`: Weight for texture similarity (default: 0.2)
- `store_matches`: If True, store results in database

**Returns:**
```python
{
    'matches': [
        {
            'product_id': int,
            'image_path': str,
            'category': str or None,
            'product_name': str or None,
            'sku': str or None,
            'similarity_score': float,  # 0-100
            'color_score': float,
            'shape_score': float,
            'texture_score': float,
            'is_potential_duplicate': bool,  # True if score > 90
            'created_at': str
        },
        ...
    ],
    'total_candidates': int,
    'successful_matches': int,
    'failed_matches': int,
    'filtered_by_threshold': int,
    'threshold_used': float,
    'limit_used': int,
    'category_filter': str or None,
    'matched_against_all_categories': bool,
    'warnings': [str, ...],
    'errors': [
        {
            'product_id': int,
            'error': str,
            'error_code': str,
            'suggestion': str
        },
        ...
    ] or None
}
```

**Raises:**
- `ProductNotFoundError`: Product doesn't exist
- `MissingFeaturesError`: Product doesn't have features extracted
- `EmptyCatalogError`: No historical products exist for matching
- `AllMatchesFailedError`: All similarity computations failed

**Example:**
```python
from product_matching import find_matches

# Basic matching
result = find_matches(product_id=123)

# With threshold and limit
result = find_matches(
    product_id=123,
    threshold=50.0,  # Only matches with score >= 50
    limit=5          # Top 5 matches
)

# Match against all categories
result = find_matches(
    product_id=123,
    match_against_all=True
)

# Custom feature weights
result = find_matches(
    product_id=123,
    color_weight=0.6,
    shape_weight=0.3,
    texture_weight=0.1
)
```

### `batch_find_matches()`

Find matches for multiple products in batch.

```python
def batch_find_matches(
    product_ids: List[int],
    threshold: float = 0.0,
    limit: int = 10,
    match_against_all: bool = False,
    include_uncategorized: bool = True,
    color_weight: float = 0.5,
    shape_weight: float = 0.3,
    texture_weight: float = 0.2,
    store_matches: bool = True,
    continue_on_error: bool = True
) -> Dict[str, Any]
```

**Parameters:**
Same as `find_matches()`, plus:
- `product_ids`: List of product IDs to match
- `continue_on_error`: If True, continue processing on errors

**Returns:**
```python
{
    'results': [
        {
            'product_id': int,
            'status': 'success' or 'failed',
            'match_count': int,
            'matches': [...],  # Same format as find_matches()
            'warnings': [str, ...]
        },
        ...
    ],
    'summary': {
        'total_products': int,
        'successful': int,
        'failed': int,
        'success_rate': float  # Percentage
    },
    'errors': [
        {
            'product_id': int,
            'status': 'failed',
            'error': str,
            'error_code': str,
            'suggestion': str
        },
        ...
    ] or None
}
```

**Example:**
```python
from product_matching import batch_find_matches

# Batch match multiple products
result = batch_find_matches(
    product_ids=[123, 124, 125],
    threshold=50.0,
    limit=10
)

print(f"Success rate: {result['summary']['success_rate']}%")
for item in result['results']:
    if item['status'] == 'success':
        print(f"Product {item['product_id']}: {item['match_count']} matches")
```

### `get_match_statistics()`

Get statistics about matches for a product.

```python
def get_match_statistics(product_id: int) -> Dict[str, Any]
```

**Returns:**
```python
{
    'product_id': int,
    'total_matches': int,
    'has_matches': bool,
    'highest_score': float,
    'lowest_score': float,
    'average_score': float,
    'potential_duplicates': int,  # Matches with score > 90
    'high_similarity': int,       # Matches with score > 70
    'medium_similarity': int,     # Matches with 50 <= score <= 70
    'low_similarity': int         # Matches with score < 50
}
```

**Example:**
```python
from product_matching import get_match_statistics

stats = get_match_statistics(123)
print(f"Average similarity: {stats['average_score']:.2f}")
print(f"Potential duplicates: {stats['potential_duplicates']}")
```

## Error Handling

### Exception Hierarchy

```
MatchingError (base)
├── ProductNotFoundError
├── MissingFeaturesError
├── EmptyCatalogError
└── AllMatchesFailedError
```

### Error Codes

- `PRODUCT_NOT_FOUND`: Product doesn't exist in database
- `MISSING_FEATURES`: Product doesn't have features extracted
- `EMPTY_CATALOG`: No historical products available for matching
- `ALL_MATCHES_FAILED`: All similarity computations failed
- `INVALID_FEATURES`: Feature vectors are corrupted or invalid
- `DIMENSION_MISMATCH`: Feature dimensions don't match expected values

### Error Response Format

All errors include:
```python
{
    'error': str,           # Human-readable error message
    'error_code': str,      # Machine-readable error code
    'suggestion': str       # Actionable suggestion for fixing the issue
}
```

## Real-World Data Handling

### NULL/Missing Categories

The service handles products without categories in multiple ways:

1. **Automatic Fallback**: Products with NULL category automatically match against all categories
2. **Explicit Control**: Use `match_against_all=True` to force matching against all categories
3. **Include/Exclude**: Use `include_uncategorized` to control whether NULL category products are included
4. **Category Normalization**: Handles variations like "unknown", "uncategorized", "N/A"

### Corrupted Features

When features are corrupted or missing:

1. **Skip and Continue**: Problematic products are skipped by default (`skip_invalid_products=True`)
2. **Detailed Validation**: Checks for NaN, Inf, empty arrays, wrong dimensions
3. **Detailed Logging**: Warnings logged for each skipped product with specific issue
4. **Error Reporting**: Detailed error information with error codes and suggestions
5. **Partial Results**: Returns successful matches even if some fail
6. **Data Quality Metrics**: Tracks and reports types of issues encountered

### Missing Metadata

Products with missing metadata are handled gracefully:

1. **Optional Fields**: product_name, sku, and category are optional
2. **NULL Handling**: Safe access to all fields with proper NULL checks
3. **Metadata Tracking**: Flags products with missing fields in results
4. **Continues Matching**: Missing metadata doesn't prevent similarity computation

### Database Errors

Robust error handling for database issues:

1. **Connection Errors**: Catches and reports database connection failures
2. **Query Errors**: Handles malformed queries and constraint violations
3. **Transaction Safety**: Uses context managers for safe transactions
4. **Graceful Degradation**: Continues processing other products on individual failures

### Image Processing Issues

Handles various image-related problems:

1. **No File Size Limits**: Removed 10MB limit, handles large images gracefully
2. **Corrupted Images**: Detects and reports unopenable or corrupted image files
3. **Invalid Formats**: Validates image format before processing
4. **Empty Files**: Detects and reports zero-byte files
5. **Feature Extraction Failures**: Handles failures in color, shape, or texture extraction

### Edge Cases

The service handles various edge cases:

1. **Empty Catalog**: Raises `EmptyCatalogError` with helpful suggestion
2. **All Matches Fail**: Raises `AllMatchesFailedError` if no similarities can be computed
3. **Self-Matching**: Automatically skips matching a product against itself
4. **Zero Results**: Returns empty list with appropriate warnings
5. **Mixed Data Quality**: Processes good products even when others fail

## Performance Considerations

### Optimization Tips

1. **Use Threshold**: Set appropriate threshold to reduce result set size
2. **Limit Results**: Use `limit` parameter to control result count
3. **Category Filtering**: Matching within category is faster than matching all
4. **Batch Processing**: Use `batch_find_matches()` for multiple products
5. **Store Matches**: Set `store_matches=False` if you don't need to persist results

### Scalability

- Handles catalogs with 10,000+ products
- Efficient database queries with proper indexing
- Graceful degradation with corrupted data
- Parallel processing support in batch mode

## Data Quality Monitoring

### Data Quality Metrics

Every match result includes comprehensive data quality information:

```python
{
    'data_quality_issues': {
        'missing_features': int,      # Products without feature vectors
        'corrupted_features': int,    # Products with NaN/Inf in features
        'missing_metadata': int,      # Products with NULL name/SKU
        'invalid_categories': int,    # Products with invalid categories
        'computation_errors': int     # Failed similarity computations
    },
    'data_quality_summary': {
        'total_issues': int,          # Sum of all issues
        'success_rate': float,        # Percentage of successful matches
        'has_data_quality_issues': bool
    }
}
```

### Warnings

The service provides warnings for:
- Products with NULL categories
- Products with missing metadata (name, SKU)
- Normalized category names
- Filtered results
- Corrupted feature vectors
- Products without features

### Error Tracking

Detailed error information for each failed product:

```python
{
    'product_id': int,
    'error': str,              # Human-readable error message
    'error_code': str,         # Machine-readable code
    'suggestion': str          # Actionable fix suggestion
}
```

Error codes include:
- `MISSING_FEATURES`: Product has no feature vectors
- `CORRUPTED_FEATURES`: Features contain NaN/Inf
- `PRODUCT_NOT_FOUND`: Product doesn't exist
- `DATABASE_ERROR`: Database query failed
- `UNKNOWN_ERROR`: Unexpected error occurred

### Recommendations

Monitor the following metrics:
- `data_quality_summary.success_rate`: Should be > 80% for healthy data
- `data_quality_issues.corrupted_features`: High count indicates image processing issues
- `data_quality_issues.missing_metadata`: Indicates incomplete product data
- `failed_matches` count: Should be < 20% of total candidates
- `warnings` list length: Review for patterns
- Error codes in `errors` list: Identify systematic issues

### Data Quality Best Practices

1. **Regular Monitoring**: Check data quality metrics after each batch
2. **Feature Re-extraction**: Re-process products with corrupted features
3. **Metadata Completion**: Fill in missing product names and SKUs
4. **Category Validation**: Ensure consistent category naming
5. **Image Quality**: Verify images are not corrupted before upload
6. **Batch Testing**: Test with small batches before large-scale processing

## Integration Examples

### Flask API Integration

```python
from flask import Flask, request, jsonify
from product_matching import find_matches, ProductNotFoundError, MissingFeaturesError

app = Flask(__name__)

@app.route('/api/products/match', methods=['POST'])
def match_products():
    data = request.json
    product_id = data.get('product_id')
    threshold = data.get('threshold', 0.0)
    limit = data.get('limit', 10)
    
    try:
        result = find_matches(
            product_id=product_id,
            threshold=threshold,
            limit=limit
        )
        return jsonify(result), 200
        
    except ProductNotFoundError as e:
        return jsonify(e.to_dict()), 404
        
    except MissingFeaturesError as e:
        return jsonify(e.to_dict()), 400
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500
```

### Command Line Usage

```python
import sys
from product_matching import find_matches

if __name__ == '__main__':
    product_id = int(sys.argv[1])
    
    result = find_matches(product_id=product_id, threshold=50.0, limit=5)
    
    print(f"Found {len(result['matches'])} matches")
    for match in result['matches']:
        print(f"  {match['product_id']}: {match['similarity_score']:.2f}")
```

## Requirements Mapping

This implementation satisfies the following requirements:

- **Requirement 3.1**: Category filtering with exact matching
- **Requirement 3.2**: Case-insensitive category comparison
- **Requirement 4.1**: Ranked results by similarity score
- **Requirement 5.1**: Configurable similarity threshold
- **Requirement 5.2**: Threshold filtering
- **Requirement 5.3**: Duplicate detection (score > 90)

## Testing

Run the test suite:

```bash
python backend/test_matching.py
```

Tests cover:
- Basic matching functionality
- NULL category handling
- Missing features handling
- Threshold filtering
- Duplicate detection
- Batch matching
- Empty catalog handling
- Match statistics

## Logging

The service uses Python's logging module:

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Or for debugging
logging.basicConfig(level=logging.DEBUG)
```

Log levels:
- `INFO`: Normal operations (matching started, completed, etc.)
- `WARNING`: Data quality issues (missing features, NULL categories, etc.)
- `ERROR`: Errors (product not found, all matches failed, etc.)

## Future Enhancements

Potential improvements:
1. Caching of similarity computations
2. Parallel processing for large catalogs
3. Machine learning-based feature weighting
4. Category inference for uncategorized products
5. Performance metrics and monitoring
6. A/B testing of different similarity algorithms
