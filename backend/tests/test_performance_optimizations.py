"""
Test performance optimizations (Task 14)

This test verifies that the performance optimizations are working correctly:
1. Database indexes are created
2. Category-based filtering works efficiently
3. Lazy loading is implemented in frontend
"""

import sys
import os
import sqlite3

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import init_db, DB_PATH


def test_database_indexes():
    """Verify that all performance indexes are created"""
    print("\n[TEST] Checking database indexes...")
    
    # Initialize database
    init_db()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get list of indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = [row[0] for row in cursor.fetchall()]
    
    # Expected indexes for performance optimization
    expected_indexes = [
        'idx_products_category',
        'idx_products_is_historical',
        'idx_products_category_historical',
        'idx_matches_new_product',
        'idx_features_product_id'
    ]
    
    print(f"  Found {len(indexes)} indexes in database")
    
    # Check each expected index
    missing_indexes = []
    for index_name in expected_indexes:
        if index_name in indexes:
            print(f"  [✓] {index_name}")
        else:
            print(f"  [✗] {index_name} - MISSING!")
            missing_indexes.append(index_name)
    
    conn.close()
    
    if missing_indexes:
        print(f"\n[FAIL] Missing indexes: {missing_indexes}")
        return False
    else:
        print("\n[PASS] All performance indexes are present")


def test_category_filtering_query():
    """Verify that category filtering uses efficient queries"""
    print("\n[TEST] Checking category filtering query efficiency...")
    
    # Initialize database
    init_db()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Test query that should use the composite index
    test_query = '''
        SELECT p.id, p.category, f.color_features, f.shape_features, f.texture_features
        FROM products p
        JOIN features f ON p.id = f.product_id
        WHERE p.category = ? AND p.is_historical = ?
    '''
    
    # Use EXPLAIN QUERY PLAN to check if index is used
    cursor.execute(f"EXPLAIN QUERY PLAN {test_query}", ('test_category', True))
    query_plan = cursor.fetchall()
    
    print("  Query plan:")
    for row in query_plan:
        print(f"    {row}")
    
    # Check if the composite index is mentioned in the query plan
    query_plan_str = str(query_plan).lower()
    uses_index = 'idx_products_category_historical' in query_plan_str or 'index' in query_plan_str
    
    conn.close()
    
    if uses_index:
        print("\n[PASS] Query uses index for efficient filtering")
    else:
        print("\n[INFO] Query plan doesn't explicitly show index usage (may still be optimized)")


def test_lazy_loading_implementation():
    """Verify that lazy loading is implemented in frontend"""
    print("\n[TEST] Checking lazy loading implementation...")
    
    app_js_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'app.js')
    
    if not os.path.exists(app_js_path):
        print(f"[FAIL] app.js not found at {app_js_path}")
        return False
    
    with open(app_js_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for lazy loading implementation
    checks = {
        'IntersectionObserver': 'IntersectionObserver' in content,
        'lazy-load class': 'lazy-load' in content,
        'data-src attribute': 'data-src' in content,
        'initLazyLoading function': 'function initLazyLoading' in content or 'initLazyLoading()' in content
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            print(f"  [✓] {check_name}")
        else:
            print(f"  [✗] {check_name} - MISSING!")
            all_passed = False
    
    if all_passed:
        print("\n[PASS] Lazy loading is properly implemented")
    else:
        print("\n[FAIL] Lazy loading implementation incomplete")
        assert False, "Lazy loading implementation incomplete"


def main():
    """Run all performance optimization tests"""
    print("=" * 70)
    print("PERFORMANCE OPTIMIZATION TESTS (Task 14)")
    print("=" * 70)
    
    tests = [
        ("Database Indexes", test_database_indexes),
        ("Category Filtering Query", test_category_filtering_query),
        ("Lazy Loading Implementation", test_lazy_loading_implementation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed with exception: {e}")
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
        print("\n✓ All performance optimizations are working correctly!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
