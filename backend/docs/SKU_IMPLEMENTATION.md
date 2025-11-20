# SKU Implementation Summary

## Overview

SKU (Stock Keeping Unit) support has been fully implemented in Catalog Match with comprehensive real-world data handling. SKUs are **optional** metadata that can help identify and track products, but are not required for the system to function.

## Database Schema

The `sku` field is part of the `products` table:

```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,              -- REQUIRED
    category TEXT,                         -- OPTIONAL
    product_name TEXT,                     -- OPTIONAL
    sku TEXT,                              -- OPTIONAL (this field)
    is_historical BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
)
```

**Key Design Decisions:**
- SKU is **optional** (can be NULL)
- **No uniqueness constraint** - duplicates are allowed (real-world data may have duplicate SKUs)
- **No foreign keys** - SKU is just a text field for identification
- **Case-insensitive** search and comparison by default
- **Max length: 50 characters**

## SKU Format Rules

### Valid Format
- Alphanumeric characters: `A-Z`, `a-z`, `0-9`
- Hyphens: `-`
- Underscores: `_`
- Max length: 50 characters

### Examples of Valid SKUs
```
SKU-123
PROD_ABC
12345
ABC-123-XYZ
ITEM_2024_001
```

### Invalid SKUs
```
SKU 123        # Contains space
SKU@123        # Contains @
SKU#123        # Contains #
SKU.123        # Contains period
```

## Normalization

SKUs are normalized for consistent storage and comparison:

1. **Trim whitespace** - Leading/trailing spaces removed
2. **Convert to uppercase** - `sku-123` becomes `SKU-123`
3. **Empty string → NULL** - Empty strings treated as NULL

**Example:**
```python
normalize_sku("  sku-123  ")  # Returns: "SKU-123"
normalize_sku("")             # Returns: None
normalize_sku(None)           # Returns: None
```

## Core Functions

### Validation
```python
def validate_sku_format(sku: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate SKU format.
    
    Returns:
        (True, None) if valid
        (False, error_message) if invalid
    """
```

### Normalization
```python
def normalize_sku(sku: Optional[str]) -> Optional[str]:
    """
    Normalize SKU for consistent storage.
    
    Returns normalized SKU or None.
    """
```

### Search and Retrieval
```python
def get_products_by_sku(sku: str, case_sensitive: bool = False) -> List[sqlite3.Row]:
    """
    Get all products with a specific SKU.
    May return multiple products if duplicates exist.
    """

def check_sku_exists(sku: str, exclude_product_id: Optional[int] = None, 
                     case_sensitive: bool = False) -> bool:
    """
    Check if SKU already exists in database.
    Useful for duplicate warnings.
    """

def search_products(query: str, search_fields: List[str] = None, 
                   category: Optional[str] = None,
                   is_historical: Optional[bool] = None,
                   limit: int = 100) -> List[sqlite3.Row]:
    """
    Search products across multiple fields including SKU.
    """
```

### Data Quality
```python
def get_duplicate_skus() -> List[Dict[str, Any]]:
    """
    Get all SKUs that appear multiple times.
    Returns list with 'sku' and 'count' keys.
    """

def get_products_without_sku(is_historical: Optional[bool] = None) -> List[sqlite3.Row]:
    """
    Get all products missing SKU information.
    """

def get_data_quality_stats() -> Dict[str, Any]:
    """
    Get comprehensive data quality statistics including SKU completeness.
    """
```

## Real-World Data Handling

### Duplicate SKUs
**Issue:** Multiple products may have the same SKU due to:
- Data entry errors
- Intentional reuse
- Different product variants
- Import from multiple sources

**Handling:**
- Duplicates are **allowed** (no uniqueness constraint)
- `get_duplicate_skus()` identifies duplicates for review
- UI shows warning icon for duplicate SKUs
- Users can search and view all products with same SKU

**Example:**
```python
# Both inserts succeed
insert_product('image1.jpg', 'Electronics', 'Product 1', 'SKU-001')
insert_product('image2.jpg', 'Electronics', 'Product 2', 'SKU-001')

# Find duplicates
duplicates = get_duplicate_skus()
# Returns: [{'sku': 'SKU-001', 'count': 2}]
```

### Missing SKUs
**Issue:** Many products may not have SKUs

**Handling:**
- NULL is valid and expected
- UI shows "No SKU" or "-" for NULL values
- `get_products_without_sku()` identifies products needing SKUs
- Data quality dashboard shows SKU completeness percentage

### Invalid Format
**Issue:** SKUs may have invalid characters or be too long

**Handling:**
- Validate before insert using `validate_sku_format()`
- Show clear error message with format requirements
- Suggest corrections (e.g., "Remove spaces and special characters")

**Example:**
```python
is_valid, error = validate_sku_format("SKU 123")
# Returns: (False, "Invalid SKU format. Use only letters, numbers, hyphens, and underscores.")
```

### Inconsistent Formatting
**Issue:** Same SKU entered with different cases or whitespace

**Handling:**
- Normalize before storage using `normalize_sku()`
- Case-insensitive search by default
- Consistent uppercase storage

**Example:**
```python
# These are treated as the same SKU
normalize_sku("sku-123")      # Returns: "SKU-123"
normalize_sku("SKU-123")      # Returns: "SKU-123"
normalize_sku("  SKU-123  ")  # Returns: "SKU-123"
```

## Integration with Matching

### SKU in Matching Logic
- **SKUs are NOT used for visual matching** - Only image features are used
- SKUs are **metadata** for identification and tracking
- SKUs are displayed in match results for easy identification
- Duplicate SKUs are highlighted in comparison view

### Match Results Display
```python
def get_matches_for_product(new_product_id: int, limit: int = 10) -> List[sqlite3.Row]:
    """
    Get match results including SKU information.
    
    Returns matches with:
    - similarity_score
    - product_name
    - sku (may be NULL)
    - category
    - image_path
    """
```

### Comparison View
When comparing two products:
- Show both SKUs side-by-side
- Highlight if SKUs match (potential duplicate)
- Highlight if SKUs differ (different products)
- Show "No SKU" if NULL

## Usage Examples

### Basic Usage
```python
from database import insert_product, normalize_sku, validate_sku_format

# Validate and normalize SKU before insert
sku = "  sku-123  "
is_valid, error = validate_sku_format(sku)

if is_valid:
    normalized_sku = normalize_sku(sku)
    product_id = insert_product(
        image_path='product.jpg',
        category='Electronics',
        product_name='Laptop',
        sku=normalized_sku
    )
else:
    print(f"Invalid SKU: {error}")
```

### Check for Duplicates
```python
from database import check_sku_exists, get_products_by_sku

sku = "SKU-123"

if check_sku_exists(sku):
    # Get all products with this SKU
    products = get_products_by_sku(sku)
    print(f"Warning: {len(products)} products already have SKU {sku}")
    
    # Show user the existing products
    for product in products:
        print(f"  - {product['product_name']} (ID: {product['id']})")
    
    # Ask user if they want to proceed
    # (duplicates are allowed but user should be aware)
```

### Search by SKU
```python
from database import search_products

# Search for products with SKU containing "LAP"
results = search_products('LAP', search_fields=['sku'])

# Search across name and SKU
results = search_products('Laptop', search_fields=['product_name', 'sku'])

# Search with category filter
results = search_products('001', category='Electronics')
```

### Data Quality Monitoring
```python
from database import get_data_quality_stats, get_duplicate_skus, get_products_without_sku

# Get overall stats
stats = get_data_quality_stats()
print(f"SKU Completeness: {stats['completeness']['sku']}%")
print(f"Products without SKU: {stats['missing_sku']}")
print(f"Duplicate SKUs: {stats['duplicate_skus']}")

# Get specific issues
duplicates = get_duplicate_skus()
for dup in duplicates:
    print(f"SKU {dup['sku']} appears {dup['count']} times")

no_sku = get_products_without_sku(is_historical=True)
print(f"{len(no_sku)} historical products need SKUs")
```

## UI Integration Guidelines

### Upload Form
```typescript
interface ProductUploadForm {
  image: File;              // Required
  category?: string;        // Optional
  productName?: string;     // Optional
  sku?: string;             // Optional
}

// Validate SKU on input
function validateSKU(sku: string): ValidationResult {
  // Call backend validation endpoint
  // Show real-time feedback
  // Warn if duplicate exists
}

// Normalize SKU before submit
function normalizeSKU(sku: string): string {
  return sku.trim().toUpperCase();
}
```

### Display in Product Cards
```typescript
function ProductCard({ product }) {
  return (
    <div>
      <img src={product.image_path} />
      <h3>{product.product_name || "Unnamed Product"}</h3>
      <div className="sku">
        {product.sku ? (
          <>
            <span>SKU: {product.sku}</span>
            {isDuplicate(product.sku) && <WarningIcon />}
          </>
        ) : (
          <span className="no-sku">No SKU</span>
        )}
      </div>
      <div className="category">
        {product.category || "Uncategorized"}
      </div>
    </div>
  );
}
```

### Comparison View
```typescript
function ComparisonView({ product1, product2 }) {
  const skusMatch = product1.sku && product2.sku && 
                    product1.sku.toUpperCase() === product2.sku.toUpperCase();
  
  return (
    <div className="comparison">
      <div className="product">
        <div className={skusMatch ? "sku-match" : "sku"}>
          SKU: {product1.sku || "No SKU"}
        </div>
      </div>
      <div className="product">
        <div className={skusMatch ? "sku-match" : "sku"}>
          SKU: {product2.sku || "No SKU"}
        </div>
      </div>
      {skusMatch && (
        <div className="warning">
          ⚠️ Both products have the same SKU - possible duplicate
        </div>
      )}
    </div>
  );
}
```

## Testing

Comprehensive tests in `test_sku_handling.py`:

✓ SKU format validation
✓ SKU normalization
✓ Storage and retrieval
✓ Duplicate detection
✓ Search functionality
✓ Data quality statistics
✓ Real-world scenarios (minimal data, duplicates, etc.)

Run tests:
```bash
python backend/test_sku_handling.py
```

## Migration Notes

If you have existing data without SKUs:

1. **All existing products will have NULL SKU** - This is valid and expected
2. **No data migration needed** - NULL is a valid state
3. **Gradually add SKUs** - Add SKUs as you process products
4. **Monitor completeness** - Use `get_data_quality_stats()` to track progress
5. **No breaking changes** - System works with or without SKUs

## Best Practices

### For Developers
1. **Always validate** - Use `validate_sku_format()` before insert
2. **Always normalize** - Use `normalize_sku()` for consistent storage
3. **Handle NULL gracefully** - NULL is valid, don't treat as error
4. **Warn on duplicates** - Use `check_sku_exists()` to warn users
5. **Case-insensitive by default** - Use case-insensitive search/comparison
6. **Monitor data quality** - Track SKU completeness and duplicates

### For Users
1. **Use consistent format** - Stick to one format (e.g., "PROD-001")
2. **Avoid duplicates** - Check if SKU exists before adding
3. **Add SKUs when possible** - Helps with product identification
4. **Review duplicates** - Check duplicate SKU report regularly
5. **Don't rely on SKUs for matching** - Visual matching is primary method

## Future Enhancements

Potential improvements:
- Auto-generate SKUs for products without them
- SKU format templates per category
- Bulk SKU import from CSV
- SKU validation rules per category
- Automatic duplicate resolution suggestions
- SKU-based product linking (variants, versions)
