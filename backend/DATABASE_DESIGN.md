# Database Design - Product Matching System

## Overview
The database layer is designed to handle real-world messy data where product information may be incomplete or missing.

## Schema Design

### Products Table
```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,           -- REQUIRED: Only field that must be present
    category TEXT,                      -- OPTIONAL: Can be NULL if unknown
    product_name TEXT,                  -- OPTIONAL: Can be NULL
    sku TEXT,                          -- OPTIONAL: Can be NULL
    is_historical BOOLEAN DEFAULT 0,    -- Flag for catalog vs new products
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT                       -- OPTIONAL: JSON for additional fields
)
```

**Design Rationale:**
- Only `image_path` is required - we can always extract features from an image
- `category` is optional because it may need to be inferred from image analysis
- Other fields (name, SKU) are optional as they may not be available for all products
- `metadata` field allows storing additional unstructured data as JSON

### Features Table
```sql
CREATE TABLE features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    color_features BLOB NOT NULL,      -- Numpy array serialized to bytes
    shape_features BLOB NOT NULL,      -- Numpy array serialized to bytes
    texture_features BLOB NOT NULL,    -- Numpy array serialized to bytes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id)
)
```

**Design Rationale:**
- Feature vectors stored as BLOBs using numpy serialization
- All three feature types required when features exist
- Separate table allows products without features (pending extraction)

### Matches Table
```sql
CREATE TABLE matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    new_product_id INTEGER NOT NULL,
    matched_product_id INTEGER NOT NULL,
    similarity_score REAL NOT NULL,
    color_score REAL NOT NULL,
    shape_score REAL NOT NULL,
    texture_score REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (new_product_id) REFERENCES products(id),
    FOREIGN KEY (matched_product_id) REFERENCES products(id)
)
```

## Handling Missing Data

### Category Handling
The system provides multiple ways to work with missing categories:

1. **Insert without category:**
   ```python
   product_id = insert_product(image_path="/path/to/image.jpg")
   ```

2. **Query uncategorized products:**
   ```python
   uncategorized = get_products_by_category(None)
   ```

3. **Include uncategorized in category queries:**
   ```python
   products = get_products_by_category("dinnerware", include_uncategorized=True)
   ```

4. **Bulk categorize products:**
   ```python
   bulk_update_category([id1, id2, id3], "dinnerware")
   ```

### Feature Extraction Workflow
Products can exist without features initially:

1. **Find products needing feature extraction:**
   ```python
   products = get_products_without_features()
   ```

2. **Extract and store features:**
   ```python
   for product in products:
       features = extract_features(product['image_path'])
       insert_features(product['id'], features['color'], features['shape'], features['texture'])
   ```

### Data Quality Monitoring
Track incomplete products:

```python
incomplete = get_incomplete_products()
for product in incomplete:
    if product['missing_category']:
        # Trigger category inference
    if product['missing_features']:
        # Trigger feature extraction
    if product['missing_name'] or product['missing_sku']:
        # Flag for manual data entry
```

## Indexes

Performance indexes are created on commonly queried fields:

- `idx_products_category` - Fast category filtering
- `idx_products_is_historical` - Fast historical/new product filtering
- `idx_products_category_historical` - Combined category + historical queries
- `idx_matches_new_product` - Fast match retrieval sorted by score

**Note:** Indexes on nullable columns (like category) still work efficiently - NULL values are indexed.

## API Functions

### Product CRUD
- `insert_product()` - Only image_path required
- `get_product_by_id()` - Retrieve single product
- `update_product()` - Update any fields dynamically
- `delete_product()` - Cascading delete of features and matches
- `get_products_by_category()` - Filter by category (or NULL)
- `count_products()` - Count with optional filters

### Feature Operations
- `insert_features()` - Store numpy arrays as BLOBs
- `get_features_by_product_id()` - Retrieve as numpy arrays
- `update_features()` - Update specific feature vectors
- `get_all_features_by_category()` - Batch retrieval for matching

### Match Operations
- `insert_match()` - Store match results
- `get_matches_for_product()` - Retrieve ranked matches
- `delete_matches_for_product()` - Clean up matches

### Utility Functions
- `get_products_without_category()` - Find uncategorized products
- `get_products_without_features()` - Find products needing feature extraction
- `get_incomplete_products()` - Data quality report
- `get_all_categories()` - List unique categories
- `bulk_update_category()` - Batch categorization

## Usage Examples

### Scenario 1: New Product with Unknown Category
```python
# Insert product with minimal info
product_id = insert_product(image_path="/uploads/new_product.jpg")

# Extract features
features = extract_features_from_image("/uploads/new_product.jpg")
insert_features(product_id, features['color'], features['shape'], features['texture'])

# Match against all historical products (regardless of category)
historical_features = get_all_features_by_category(category=None, is_historical=True)
matches = find_best_matches(features, historical_features)

# Infer category from top matches
inferred_category = matches[0]['category']
update_product(product_id, category=inferred_category)
```

### Scenario 2: Batch Processing Uncategorized Products
```python
# Find all uncategorized products
uncategorized = get_products_without_category(is_historical=True)

# Group by inferred category
category_groups = {}
for product in uncategorized:
    inferred_cat = infer_category_from_image(product['image_path'])
    if inferred_cat not in category_groups:
        category_groups[inferred_cat] = []
    category_groups[inferred_cat].append(product['id'])

# Bulk update
for category, product_ids in category_groups.items():
    bulk_update_category(product_ids, category)
```

### Scenario 3: Matching with Flexible Category Handling
```python
# Try matching within category first
if product_category:
    historical_features = get_all_features_by_category(
        category=product_category, 
        is_historical=True,
        include_uncategorized=True  # Include products that might belong to this category
    )
else:
    # No category - match against everything
    historical_features = get_all_features_by_category(
        category=None, 
        is_historical=True
    )
```

## Best Practices

1. **Always handle NULL categories** - Use `include_uncategorized` parameter when appropriate
2. **Check for missing features** - Use `get_products_without_features()` before matching
3. **Monitor data quality** - Regularly check `get_incomplete_products()`
4. **Use bulk operations** - `bulk_update_category()` for batch processing
5. **Store metadata as JSON** - Use metadata field for flexible additional data
