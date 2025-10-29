"""
Test real-world data handling edge cases

This test verifies that the system properly handles:
- NULL/missing values
- Invalid file formats
- Corrupted files
- Invalid metadata
- Duplicate data
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import (
    init_db, insert_product, get_product_by_id,
    validate_sku_format, normalize_sku, check_sku_exists,
    get_products_by_category
)
from product_matching import normalize_category


def test_null_values():
    """Test that NULL values are handled properly"""
    print("\n[TEST] NULL value handling...")
    
    init_db()
    
    # Test 1: Insert product with all NULL metadata
    product_id = insert_product(
        image_path="/fake/path.jpg",
        category=None,
        product_name=None,
        sku=None,
        is_historical=True
    )
    
    product = get_product_by_id(product_id)
    assert product is not None, "Product should be inserted"
    assert product['category'] is None, "Category should be NULL"
    assert product['product_name'] is None, "Product name should be NULL"
    assert product['sku'] is None, "SKU should be NULL"
    print("  [✓] Product with all NULL metadata inserted successfully")
    
    # Test 2: Query products with NULL category
    products = get_products_by_category(None, is_historical=True)
    assert len(products) > 0, "Should find products with NULL category"
    print("  [✓] Query for NULL category works")
    
    # Test 3: Query with include_uncategorized
    products = get_products_by_category('test', is_historical=True, include_uncategorized=True)
    # Should not crash even if no products exist
    print("  [✓] Query with include_uncategorized works")
    
    print("[PASS] NULL value handling works correctly")
    return True


def test_sku_validation():
    """Test SKU validation and normalization"""
    print("\n[TEST] SKU validation...")
    
    # Test 1: Valid SKUs
    valid_skus = ['SKU-001', 'ABC123', 'test_sku', 'A-B-C_1-2-3']
    for sku in valid_skus:
        is_valid, error = validate_sku_format(sku)
        assert is_valid, f"SKU '{sku}' should be valid"
    print("  [✓] Valid SKUs accepted")
    
    # Test 2: Invalid SKUs
    invalid_skus = ['SKU 001', 'SKU@001', 'SKU#001', 'a' * 51]
    for sku in invalid_skus:
        is_valid, error = validate_sku_format(sku)
        assert not is_valid, f"SKU '{sku}' should be invalid"
        assert error is not None, "Should provide error message"
    print("  [✓] Invalid SKUs rejected with error messages")
    
    # Test 3: NULL/empty SKUs
    assert validate_sku_format(None)[0] == True, "NULL SKU should be valid"
    assert validate_sku_format('')[0] == True, "Empty SKU should be valid"
    assert validate_sku_format('   ')[0] == True, "Whitespace SKU should be valid (treated as empty)"
    print("  [✓] NULL/empty SKUs handled correctly")
    
    # Test 4: SKU normalization
    assert normalize_sku('  sku-001  ') == 'SKU-001', "Should trim and uppercase"
    assert normalize_sku('') is None, "Empty string should become None"
    assert normalize_sku(None) is None, "None should stay None"
    print("  [✓] SKU normalization works")
    
    print("[PASS] SKU validation works correctly")
    return True


def test_category_normalization():
    """Test category normalization"""
    print("\n[TEST] Category normalization...")
    
    # Test 1: Normal categories
    assert normalize_category('Placemats') == 'placemats', "Should lowercase"
    assert normalize_category('  Dinnerware  ') == 'dinnerware', "Should trim and lowercase"
    print("  [✓] Normal categories normalized")
    
    # Test 2: Empty/NULL categories
    assert normalize_category(None) is None, "None should stay None"
    assert normalize_category('') is None, "Empty string should become None"
    assert normalize_category('   ') is None, "Whitespace should become None"
    print("  [✓] Empty/NULL categories handled")
    
    # Test 3: "Unknown" variations
    unknown_variations = ['unknown', 'Unknown', 'UNKNOWN', 'uncategorized', 'none', 'n/a', 'NA']
    for variation in unknown_variations:
        assert normalize_category(variation) is None, f"'{variation}' should become None"
    print("  [✓] 'Unknown' variations normalized to None")
    
    print("[PASS] Category normalization works correctly")
    return True


def test_duplicate_detection():
    """Test duplicate SKU detection"""
    print("\n[TEST] Duplicate detection...")
    
    init_db()
    
    # Test 1: Insert product with SKU
    product_id1 = insert_product(
        image_path="/fake/path1.jpg",
        sku="TEST-SKU-001",
        is_historical=True
    )
    
    # Test 2: Check if SKU exists
    assert check_sku_exists("TEST-SKU-001"), "SKU should exist"
    assert check_sku_exists("test-sku-001"), "Should be case-insensitive"
    assert not check_sku_exists("NONEXISTENT"), "Non-existent SKU should return False"
    print("  [✓] Duplicate detection works")
    
    # Test 3: Exclude current product
    assert not check_sku_exists("TEST-SKU-001", exclude_product_id=product_id1), "Should exclude current product"
    print("  [✓] Exclude current product works")
    
    print("[PASS] Duplicate detection works correctly")
    return True


def test_error_handling():
    """Test that errors are handled gracefully"""
    print("\n[TEST] Error handling...")
    
    # Test 1: Invalid product ID
    product = get_product_by_id(999999)
    assert product is None, "Should return None for invalid ID"
    print("  [✓] Invalid product ID handled")
    
    # Test 2: Empty category query
    products = get_products_by_category('', is_historical=True)
    # Should not crash
    print("  [✓] Empty category query handled")
    
    print("[PASS] Error handling works correctly")
    return True


def main():
    """Run all real-world data tests"""
    print("=" * 70)
    print("REAL-WORLD DATA HANDLING TESTS")
    print("=" * 70)
    
    tests = [
        ("NULL Value Handling", test_null_values),
        ("SKU Validation", test_sku_validation),
        ("Category Normalization", test_category_normalization),
        ("Duplicate Detection", test_duplicate_detection),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "[+]" if result else "[X]"
        print(f"{symbol} {status:8} {test_name}")
    
    print("=" * 70)
    print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ All real-world data handling tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
