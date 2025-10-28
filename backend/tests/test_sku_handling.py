"""
Test SKU handling with real-world data scenarios.
"""

import os
import sys

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import (
    init_db,
    insert_product,
    get_products_by_sku,
    check_sku_exists,
    get_duplicate_skus,
    get_products_without_sku,
    validate_sku_format,
    normalize_sku,
    get_data_quality_stats,
    search_products
)

# Use test database
import database
database.DB_PATH = 'test_sku.db'


def setup_test_db():
    """Create fresh test database"""
    if os.path.exists('test_sku.db'):
        os.remove('test_sku.db')
    init_db()


def teardown_test_db():
    """Remove test database"""
    if os.path.exists('test_sku.db'):
        os.remove('test_sku.db')


def test_sku_validation():
    """Test SKU format validation"""
    print("Testing SKU validation...")
    
    # Valid SKUs
    valid_skus = [
        None,  # NULL is valid
        "",  # Empty string is valid (treated as NULL)
        "SKU-123",
        "PROD_ABC",
        "12345",
        "ABC-123-XYZ",
        "a" * 50  # Max length
    ]
    
    for sku in valid_skus:
        is_valid, error = validate_sku_format(sku)
        assert is_valid, f"Expected {sku!r} to be valid, got error: {error}"
    
    print("  ✓ Valid SKUs accepted")
    
    # Invalid SKUs
    invalid_skus = [
        ("a" * 51, "too long"),  # Too long
        ("SKU 123", "spaces"),  # Contains space
        ("SKU@123", "special char"),  # Invalid character
        ("SKU#123", "special char"),  # Invalid character
    ]
    
    for sku, reason in invalid_skus:
        is_valid, error = validate_sku_format(sku)
        assert not is_valid, f"Expected {sku!r} to be invalid ({reason})"
        assert error is not None
    
    print("  ✓ Invalid SKUs rejected")
    print("✓ SKU validation tests passed\n")


def test_sku_normalization():
    """Test SKU normalization"""
    print("Testing SKU normalization...")
    
    test_cases = [
        (None, None),
        ("", None),
        ("  ", None),
        ("sku-123", "SKU-123"),
        ("  SKU-123  ", "SKU-123"),
        ("abc", "ABC"),
        ("Prod_123", "PROD_123")
    ]
    
    for input_sku, expected in test_cases:
        result = normalize_sku(input_sku)
        assert result == expected, f"normalize_sku({input_sku!r}) = {result!r}, expected {expected!r}"
    
    print("  ✓ SKU normalization works correctly")
    print("✓ SKU normalization tests passed\n")


def test_sku_storage_and_retrieval():
    """Test storing and retrieving products with SKUs"""
    print("Testing SKU storage and retrieval...")
    
    setup_test_db()
    
    try:
        # Insert products with various SKU scenarios
        product1 = insert_product('image1.jpg', 'Electronics', 'Product 1', 'SKU-001')
        product2 = insert_product('image2.jpg', 'Electronics', 'Product 2', 'SKU-002')
        product3 = insert_product('image3.jpg', 'Electronics', 'Product 3', None)  # No SKU
        product4 = insert_product('image4.jpg', 'Electronics', 'Product 4', 'sku-001')  # Duplicate (different case)
        
        print("  ✓ Products inserted with various SKU scenarios")
        
        # Test get_products_by_sku (case-insensitive)
        results = get_products_by_sku('SKU-001', case_sensitive=False)
        assert len(results) == 2, f"Expected 2 products with SKU-001, got {len(results)}"
        print("  ✓ Case-insensitive SKU search works")
        
        # Test get_products_by_sku (case-sensitive)
        results = get_products_by_sku('SKU-001', case_sensitive=True)
        assert len(results) == 1, f"Expected 1 product with exact SKU-001, got {len(results)}"
        print("  ✓ Case-sensitive SKU search works")
        
        # Test check_sku_exists
        assert check_sku_exists('SKU-001') == True
        assert check_sku_exists('SKU-999') == False
        assert check_sku_exists('sku-001', case_sensitive=False) == True
        print("  ✓ SKU existence check works")
        
        # Test get_duplicate_skus
        duplicates = get_duplicate_skus()
        assert len(duplicates) == 1, f"Expected 1 duplicate SKU, got {len(duplicates)}"
        assert duplicates[0]['count'] == 2
        print("  ✓ Duplicate SKU detection works")
        
        # Test get_products_without_sku
        no_sku = get_products_without_sku()
        assert len(no_sku) == 1, f"Expected 1 product without SKU, got {len(no_sku)}"
        print("  ✓ Finding products without SKU works")
        
        print("✓ SKU storage and retrieval tests passed\n")
    
    finally:
        teardown_test_db()


def test_sku_search():
    """Test searching products by SKU"""
    print("Testing SKU search...")
    
    setup_test_db()
    
    try:
        # Insert test products
        insert_product('image1.jpg', 'Electronics', 'Laptop', 'LAP-001')
        insert_product('image2.jpg', 'Electronics', 'Laptop Pro', 'LAP-002')
        insert_product('image3.jpg', 'Electronics', 'Desktop', 'DESK-001')
        insert_product('image4.jpg', 'Furniture', 'Desk', 'FURN-001')
        
        # Search by SKU
        results = search_products('LAP', search_fields=['sku'])
        assert len(results) == 2, f"Expected 2 products with 'LAP' in SKU, got {len(results)}"
        print("  ✓ SKU search works")
        
        # Search by name
        results = search_products('Laptop', search_fields=['product_name'])
        assert len(results) == 2, f"Expected 2 products with 'Laptop' in name, got {len(results)}"
        print("  ✓ Name search works")
        
        # Search across multiple fields
        results = search_products('Desk', search_fields=['product_name', 'sku'])
        assert len(results) == 2, f"Expected 2 products with 'Desk' in name or SKU, got {len(results)}"
        print("  ✓ Multi-field search works")
        
        # Search with category filter
        results = search_products('001', category='Electronics')
        assert len(results) == 2, f"Expected 2 Electronics products with '001', got {len(results)}"
        print("  ✓ Search with category filter works")
        
        print("✓ SKU search tests passed\n")
    
    finally:
        teardown_test_db()


def test_data_quality_stats():
    """Test data quality statistics"""
    print("Testing data quality statistics...")
    
    setup_test_db()
    
    try:
        # Insert products with various completeness levels
        insert_product('image1.jpg', 'Electronics', 'Product 1', 'SKU-001', is_historical=True)
        insert_product('image2.jpg', 'Electronics', None, 'SKU-002', is_historical=True)  # No name
        insert_product('image3.jpg', None, 'Product 3', None, is_historical=True)  # No category, no SKU
        insert_product('image4.jpg', None, None, None, is_historical=False)  # Minimal data
        
        stats = get_data_quality_stats()
        
        assert stats['total_products'] == 4
        assert stats['historical_products'] == 3
        assert stats['new_products'] == 1
        assert stats['missing_name'] == 2
        assert stats['missing_sku'] == 2
        assert stats['missing_category'] == 2
        assert stats['missing_features'] == 4  # No features extracted yet
        
        print("  ✓ Data quality stats calculated correctly")
        
        # Check completeness percentages
        assert stats['completeness']['name'] == 50.0
        assert stats['completeness']['sku'] == 50.0
        assert stats['completeness']['category'] == 50.0
        assert stats['completeness']['features'] == 0.0
        
        print("  ✓ Completeness percentages correct")
        print("✓ Data quality stats tests passed\n")
    
    finally:
        teardown_test_db()


def test_real_world_scenarios():
    """Test real-world data scenarios"""
    print("Testing real-world scenarios...")
    
    setup_test_db()
    
    try:
        # Scenario 1: Product with minimal data (only image)
        product1 = insert_product('image1.jpg')
        assert product1 > 0
        print("  ✓ Product with only image path can be inserted")
        
        # Scenario 2: Product with duplicate SKU (should be allowed)
        product2 = insert_product('image2.jpg', 'Electronics', 'Product 2', 'DUP-001')
        product3 = insert_product('image3.jpg', 'Electronics', 'Product 3', 'DUP-001')
        assert product2 > 0 and product3 > 0
        print("  ✓ Duplicate SKUs are allowed")
        
        # Scenario 3: Product with inconsistent SKU formatting
        product4 = insert_product('image4.jpg', 'Electronics', 'Product 4', '  sku-123  ')
        results = get_products_by_sku('SKU-123', case_sensitive=False)
        # Note: Database stores as-is, normalization should be done before insert
        print("  ✓ SKU with whitespace stored (normalization should be done before insert)")
        
        # Scenario 4: Search for products without metadata
        no_sku = get_products_without_sku()
        assert len(no_sku) >= 1
        print("  ✓ Can find products missing SKU")
        
        print("✓ Real-world scenario tests passed\n")
    
    finally:
        teardown_test_db()


def main():
    """Run all tests"""
    print("=" * 60)
    print("SKU Handling Tests")
    print("=" * 60 + "\n")
    
    try:
        test_sku_validation()
        test_sku_normalization()
        test_sku_storage_and_retrieval()
        test_sku_search()
        test_data_quality_stats()
        test_real_world_scenarios()
        
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
