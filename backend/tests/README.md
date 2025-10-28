# Backend Tests

This directory contains all test files for the product matching system backend.

## Test Files

### Database Tests
- **test_database.py** - Tests for database operations (CRUD, queries, indexes)
- **test_db.py** - Additional database tests
- **test_sku_handling.py** - Tests for SKU validation, normalization, and search

### Image Processing Tests
- **test_image_processing.py** - Tests for image preprocessing and feature extraction
- **test_feature_cache.py** - Tests for feature caching mechanism

### Similarity Computation Tests
- **test_similarity.py** - Comprehensive pytest-based similarity tests
- **test_similarity_simple.py** - Simple standalone similarity tests (no pytest required)

### Examples
- **example_similarity_usage.py** - Working examples demonstrating error handling patterns

## Running Tests

### Run All Tests (if pytest is installed)
```bash
cd backend
python -m pytest tests/ -v
```

### Run Individual Test Files
```bash
# Database tests
python tests/test_database.py
python tests/test_sku_handling.py

# Image processing tests
python tests/test_image_processing.py
python tests/test_feature_cache.py

# Similarity tests
python tests/test_similarity_simple.py
```

### Run Specific Test
```bash
# With pytest
python -m pytest tests/test_similarity.py::TestColorSimilarity -v

# Without pytest (for simple tests)
python tests/test_similarity_simple.py
```

## Test Coverage

### Database Layer (✓ Complete)
- Product CRUD operations
- Feature storage and retrieval
- Category filtering
- SKU validation and search
- Duplicate detection
- Data quality monitoring
- Real-world data handling (NULL values, missing fields)

### Image Processing (✓ Complete)
- Image validation (format, size, corruption)
- Feature extraction (color, shape, texture)
- Error handling (corrupted images, invalid formats)
- Edge cases (blank images, very small images)

### Feature Caching (✓ Complete)
- Cache hit/miss scenarios
- Feature re-extraction
- Memory cache management
- Database cache integration

### Similarity Computation (✓ Complete)
- Color similarity (histogram intersection)
- Shape similarity (Euclidean distance)
- Texture similarity (chi-square distance)
- Combined similarity (weighted average)
- Error handling (NaN, Inf, wrong dimensions)
- Batch processing with error isolation
- Edge cases (zero histograms, extreme values)

### SKU Handling (✓ Complete)
- Format validation
- Normalization
- Duplicate detection
- Search functionality
- Data quality statistics

## Test Data

Test files create temporary databases and test images as needed:
- `test_*.db` - Temporary test databases (cleaned up after tests)
- Test images are generated programmatically or use minimal fixtures

## Best Practices

1. **Isolation** - Each test file cleans up after itself
2. **Independence** - Tests can run in any order
3. **Real-world scenarios** - Tests include edge cases and error conditions
4. **Clear assertions** - Test failures provide clear error messages
5. **Documentation** - Each test has descriptive docstrings

## Adding New Tests

When adding new tests:

1. Create test file with `test_` prefix
2. Add docstrings explaining what is being tested
3. Include both success and failure cases
4. Test edge cases and error handling
5. Clean up any test data/files created
6. Update this README with test coverage info

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- No external dependencies beyond requirements.txt
- Fast execution (< 1 minute for full suite)
- Clear pass/fail indicators
- Detailed error messages for debugging
