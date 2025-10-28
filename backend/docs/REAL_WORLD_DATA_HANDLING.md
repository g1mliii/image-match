# Real-World Data Handling Strategy

This document describes the comprehensive strategy for handling messy, incomplete, and invalid real-world data throughout the product matching system.

## Philosophy

Real-world data is messy. Products may have:
- Missing or NULL metadata (name, SKU, category)
- Corrupted or invalid images
- Duplicate or conflicting information
- Invalid formats or values
- Incomplete feature extraction

The system is designed to be **resilient** and **graceful** in handling these issues:
1. **Never crash** - Always handle errors gracefully
2. **Partial success** - Process what we can, report what failed
3. **Clear feedback** - Tell users exactly what went wrong and how to fix it
4. **Data quality monitoring** - Track and report data quality issues
5. **Recovery options** - Provide ways to fix or retry failed operations

## Database Schema Design

### Products Table
```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,              -- REQUIRED: Only required field
    category TEXT,                         -- OPTIONAL: Can be NULL
    product_name TEXT,                     -- OPTIONAL: Can be NULL
    sku TEXT,                              -- OPTIONAL: Can be NULL, duplicates allowed
    is_historical BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT                          -- OPTIONAL: JSON for additional data
)
```

**Design Decisions:**
- Only `image_path` is required (we need an image to match)
- All other fields are optional to handle incomplete data
- `sku` allows duplicates (real-world SKUs may be reused or incorrect)
- `metadata` field for extensibility (store any additional data as JSON)
- No foreign key constraints that could block inserts

### Features Table
```sql
CREATE TABLE features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    color_features BLOB NOT NULL,
    shape_features BLOB NOT NULL,
    texture_features BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id)
)
```

**Design Decisions:**
- Features may not exist for a product (extraction failed)
- Features can be re-extracted if corrupted
- Separate table allows re-extraction without affecting product data

## Field-Specific Handling

### Product Name
**Issues:**
- Missing/NULL
- Empty string
- Very long names
- Special characters
- Duplicate names

**Handling:**
- Allow NULL values
- Display as "Unnamed Product" in UI
- Trim whitespace on input
- Validate max length (255 characters)
- No uniqueness constraint (duplicates allowed)
- Sanitize for display (escape HTML)

**Validation:**
```python
def validate_product_name(name: Optional[str]) -> Optional[str]:
    if name is None or name.strip() == '':
        return None
    name = name.strip()
    if len(name) > 255:
        raise ValueError("Product name too long (max 255 characters)")
    return name
```

### SKU (Stock Keeping Unit)
**Issues:**
- Missing/NULL
- Invalid format
- Duplicates
- Inconsistent formatting (spaces, case, special chars)
- Very long SKUs

**Handling:**
- Allow NULL values
- Display as "No SKU" in UI
- Validate format: alphanumeric, hyphens, underscores only
- Max length: 50 characters
- Allow duplicates (warn user but don't block)
- Case-insensitive comparison for duplicate detection
- Trim whitespace and normalize format

**Validation:**
```python
import re

def validate_sku(sku: Optional[str]) -> Optional[str]:
    if sku is None or sku.strip() == '':
        return None
    
    sku = sku.strip().upper()  # Normalize to uppercase
    
    # Validate format: alphanumeric, hyphens, underscores
    if not re.match(r'^[A-Z0-9\-_]+$', sku):
        raise ValueError(
            "Invalid SKU format. Use only letters, numbers, hyphens, and underscores."
        )
    
    if len(sku) > 50:
        raise ValueError("SKU too long (max 50 characters)")
    
    return sku

def check_duplicate_sku(sku: str, exclude_product_id: Optional[int] = None) -> List[int]:
    """
    Check if SKU already exists in database.
    Returns list of product IDs with matching SKU.
    """
    # Implementation in database.py
    pass
```

**UI Display:**
- Show SKU prominently in product cards
- Highlight duplicate SKUs with warning icon
- Show "No SKU" for NULL values
- Truncate very long SKUs with tooltip

### Category
**Issues:**
- Missing/NULL
- Typos and inconsistencies
- Case sensitivity
- Empty string vs NULL
- Categories that don't exist

**Handling:**
- Allow NULL values
- Display as "Uncategorized" in UI
- Case-insensitive matching for filtering
- Trim whitespace on input
- Validate against existing categories (warn if new)
- Default to NULL if not provided (not "unknown" string)

**Matching Logic:**
- Products with NULL category can match against all categories OR
- Products with NULL category only match other NULL category products (configurable)
- Category filter includes "Uncategorized" option

**Validation:**
```python
def validate_category(category: Optional[str]) -> Optional[str]:
    if category is None or category.strip() == '':
        return None
    
    category = category.strip()
    
    if len(category) > 100:
        raise ValueError("Category name too long (max 100 characters)")
    
    return category

def get_category_for_matching(category: Optional[str], match_null_with_all: bool = False) -> Optional[str]:
    """
    Get category for matching logic.
    
    Args:
        category: Product category (may be NULL)
        match_null_with_all: If True, NULL category matches all categories
    
    Returns:
        Category for filtering, or None to match all
    """
    if category is None and match_null_with_all:
        return None  # Match all categories
    return category
```

### Image Path
**Issues:**
- File doesn't exist
- File moved/deleted
- Corrupted file
- Invalid format
- File too large
- Network path issues

**Handling:**
- Validate file exists before processing
- Validate file format (JPEG, PNG, WebP)
- Validate file size (max 10MB)
- Handle corrupted files with clear error messages
- Store relative paths (not absolute)
- Validate path on retrieval

**Error Handling:**
```python
from image_processing import (
    InvalidImageFormatError,
    CorruptedImageError,
    ImageTooSmallError,
    ImageProcessingFailedError
)

def process_image_with_error_handling(image_path: str) -> Dict[str, Any]:
    """
    Process image with comprehensive error handling.
    
    Returns:
        {
            'success': bool,
            'features': dict (if success),
            'error': str (if failure),
            'error_code': str (if failure),
            'suggestion': str (if failure)
        }
    """
    try:
        features = extract_all_features(image_path)
        return {
            'success': True,
            'features': features
        }
    except InvalidImageFormatError as e:
        return {
            'success': False,
            'error': e.message,
            'error_code': e.error_code,
            'suggestion': e.suggestion
        }
    except CorruptedImageError as e:
        return {
            'success': False,
            'error': e.message,
            'error_code': e.error_code,
            'suggestion': e.suggestion
        }
    except ImageTooSmallError as e:
        return {
            'success': False,
            'error': e.message,
            'error_code': e.error_code,
            'suggestion': e.suggestion
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'error_code': 'UNKNOWN_ERROR',
            'suggestion': 'Please try a different image or contact support.'
        }
```

## Feature Extraction Handling

### Missing Features
**Issues:**
- Features not extracted yet
- Feature extraction failed
- Features corrupted in database

**Handling:**
- Check if features exist before matching
- Provide option to re-extract features
- Skip products with missing features in matching (log warning)
- Show feature extraction status in UI
- Batch re-extraction for failed products

### Corrupted Features
**Issues:**
- NaN or Inf values in feature vectors
- Wrong dimensions
- Empty arrays
- Database corruption

**Handling:**
- Validate features before similarity computation
- Use `validate_feature_array()` from similarity module
- Skip corrupted features in batch processing
- Log errors with product ID for debugging
- Provide re-extraction option

**Example:**
```python
from similarity import InvalidFeatureError, FeatureDimensionError

def compute_similarity_with_error_handling(features1, features2):
    """
    Compute similarity with error handling.
    
    Returns:
        {
            'success': bool,
            'similarity': float (if success),
            'error': str (if failure),
            'error_code': str (if failure)
        }
    """
    try:
        result = compute_all_similarities(features1, features2)
        return {
            'success': True,
            'similarity': result['combined_similarity'],
            'breakdown': {
                'color': result['color_similarity'],
                'shape': result['shape_similarity'],
                'texture': result['texture_similarity']
            }
        }
    except (InvalidFeatureError, FeatureDimensionError) as e:
        return {
            'success': False,
            'error': e.message,
            'error_code': e.error_code,
            'suggestion': e.suggestion
        }
```

## Matching Service Error Handling

### Empty Historical Catalog
**Issue:** No historical products to match against

**Handling:**
- Return empty results with clear message
- Suggest adding historical products
- Don't treat as error (valid state)

### All Products Filtered Out
**Issue:** Category filter excludes all products

**Handling:**
- Return empty results with explanation
- Show filter settings in message
- Suggest adjusting filters

### All Similarity Computations Fail
**Issue:** Every product has corrupted features

**Handling:**
- Return error with count of failed products
- Suggest batch re-extraction
- Log detailed errors for debugging

### Partial Failures
**Issue:** Some products match successfully, others fail

**Handling:**
- Return successful matches
- Include error summary in response
- Provide detailed error list
- Don't block successful results

**Example Response:**
```json
{
  "matches": [
    {
      "product_id": 123,
      "similarity": 85.5,
      "product_name": "Blue Widget",
      "sku": "SKU-123",
      "category": "Widgets"
    }
  ],
  "total_matches": 1,
  "total_candidates": 10,
  "failed_candidates": 9,
  "errors": [
    {
      "product_id": 124,
      "error": "Features contain NaN values",
      "error_code": "INVALID_FEATURES"
    }
  ],
  "warnings": [
    "9 products skipped due to corrupted features. Consider re-extracting features."
  ]
}
```

## Batch Processing Error Handling

### Isolation
- Each file processed independently
- One failure doesn't stop batch
- Track success/failure per file

### Resource Management
- Monitor memory usage
- Throttle if resources low
- Handle disk space errors

### Progress Tracking
- Show current file
- Show overall progress
- Allow pause/cancel
- Estimate time remaining

### Error Recovery
- Retry failed files
- Track retry attempts
- Export error details

## UI Error Display

### Error Message Format
```typescript
interface ErrorDisplay {
  title: string;           // Short error title
  message: string;         // Detailed error message
  errorCode: string;       // Machine-readable code
  suggestion: string;      // How to fix
  actions: Action[];       // Available actions (retry, cancel, etc.)
}
```

### Error Severity Levels
1. **Error** (red) - Operation failed, user action required
2. **Warning** (yellow) - Operation succeeded with issues
3. **Info** (blue) - Informational message

### User Actions
- **Retry** - Try operation again
- **Skip** - Continue without this item
- **Cancel** - Stop entire operation
- **View Details** - Show technical details
- **Report** - Send error report

## Data Quality Monitoring

### Metrics to Track
- Products with missing names
- Products with missing SKUs
- Products with missing categories
- Products with failed feature extraction
- Duplicate SKUs
- Corrupted features
- Failed similarity computations

### Dashboard Display
```
Data Quality Summary
━━━━━━━━━━━━━━━━━━━━
Total Products: 1,234

Metadata Completeness:
  ✓ With Name:     1,100 (89%)
  ✗ Missing Name:    134 (11%)
  
  ✓ With SKU:        980 (79%)
  ✗ Missing SKU:     254 (21%)
  
  ✓ With Category: 1,150 (93%)
  ✗ Missing Category: 84 (7%)

Feature Extraction:
  ✓ Success:       1,200 (97%)
  ✗ Failed:           34 (3%)

Data Issues:
  ⚠ Duplicate SKUs:   12
  ⚠ Corrupted Features: 5

Recommendations:
  • Re-extract features for 34 failed products
  • Review 12 products with duplicate SKUs
  • Add categories to 84 uncategorized products
```

## Best Practices

### For Developers
1. **Always validate input** - Never trust user input
2. **Handle NULL gracefully** - NULL is valid for optional fields
3. **Provide clear errors** - Include error code, message, and suggestion
4. **Log everything** - Log errors with context for debugging
5. **Test with bad data** - Test with NULL, empty, invalid, corrupted data
6. **Isolate failures** - One bad record shouldn't stop batch processing
7. **Monitor data quality** - Track and report data quality metrics

### For Users
1. **Provide as much metadata as possible** - More data = better matching
2. **Use consistent SKU format** - Makes duplicate detection easier
3. **Categorize products** - Improves matching accuracy
4. **Review data quality dashboard** - Fix issues proactively
5. **Re-extract failed features** - Improves matching coverage
6. **Report persistent errors** - Help improve the system

## Testing Strategy

### Test Cases for Real-World Data
1. **NULL values** - Test all optional fields with NULL
2. **Empty strings** - Test with empty strings vs NULL
3. **Invalid formats** - Test with invalid SKU, category formats
4. **Corrupted images** - Test with corrupted, truncated, invalid images
5. **Missing files** - Test with non-existent image paths
6. **Duplicate data** - Test with duplicate SKUs, names
7. **Very long values** - Test with max length + 1
8. **Special characters** - Test with Unicode, HTML, SQL injection attempts
9. **Batch failures** - Test batch with mix of valid/invalid data
10. **Database errors** - Test with connection failures, constraint violations

### Example Test
```python
def test_product_with_minimal_data():
    """Test creating product with only required field (image_path)"""
    product = {
        'image_path': 'test.jpg',
        'product_name': None,
        'sku': None,
        'category': None
    }
    
    result = create_product(product)
    
    assert result['success'] == True
    assert result['product_id'] is not None
    
    # Verify NULL fields stored correctly
    retrieved = get_product(result['product_id'])
    assert retrieved['product_name'] is None
    assert retrieved['sku'] is None
    assert retrieved['category'] is None
```

## Migration Strategy

If you have existing data that doesn't follow these conventions:

1. **Audit existing data** - Identify data quality issues
2. **Clean data** - Fix obvious issues (trim whitespace, normalize SKUs)
3. **Migrate in batches** - Don't try to fix everything at once
4. **Preserve original data** - Keep backup before cleaning
5. **Document changes** - Track what was changed and why
6. **Validate after migration** - Ensure data still works
7. **Monitor for issues** - Watch for problems after migration
