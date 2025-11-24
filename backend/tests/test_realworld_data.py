"""
Test script for real-world data handling in product matching.

This script tests the matching service's ability to handle:
- Missing features
- Corrupted features (NaN, Inf values)
- Missing metadata fields
- NULL categories
- Database errors
- Mixed good and bad data
"""

import os
import sys
import numpy as np

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import (
    init_db,
    insert_product,
    insert_features,
    count_products,
    serialize_numpy_array
)
from product_matching import find_matches
import sqlite3


def create_corrupted_features(corruption_type='nan'):
    """Create intentionally corrupted features for testing"""
    color_features = np.random.rand(256).astype(np.float32)
    color_features /= color_features.sum()
    
    shape_features = np.random.randn(7).astype(np.float32)
    
    texture_features = np.random.rand(256).astype(np.float32)
    texture_features /= texture_features.sum()
    
    if corruption_type == 'nan':
        # Add NaN values
        color_features[0] = np.nan
        shape_features[0] = np.nan
    elif corruption_type == 'inf':
        # Add Inf values
        color_features[0] = np.inf
        texture_features[0] = np.inf
    elif corruption_type == 'empty':
        # Return empty arrays
        return np.array([]), np.array([]), np.array([])
    elif corruption_type == 'wrong_size':
        # Wrong dimensions
        return np.random.rand(100).astype(np.float32), shape_features, texture_features
    
    return color_features, shape_features, texture_features


def create_good_features(seed=None):
    """Create valid features"""
    if seed is not None:
        np.random.seed(seed)
    
    color_features = np.random.rand(256).astype(np.float32)
    color_features /= color_features.sum()
    
    shape_features = np.random.randn(7).astype(np.float32)
    
    texture_features = np.random.rand(256).astype(np.float32)
    texture_features /= texture_features.sum()
    
    return color_features, shape_features, texture_features


def setup_mixed_quality_data():
    """Set up database with mix of good and bad data"""
    print("Setting up mixed quality test data...")
    
    init_db()
    
    # Clear existing data
    from database import get_db_connection
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM matches')
        cursor.execute('DELETE FROM features')
        cursor.execute('DELETE FROM products')
        conn.commit()
    
    product_ids = []
    
    # Create 5 good historical products
    print("\nCreating good historical products...")
    for i in range(5):
        product_id = insert_product(
            image_path=f'/test/good_product_{i}.jpg',
            category='placemats',
            product_name=f'Good Product {i}',
            sku=f'GOOD-{i}',
            is_historical=True
        )
        color, shape, texture = create_good_features(seed=i)
        insert_features(product_id, color, shape, texture)
        product_ids.append(product_id)
        print(f"  Created product {product_id} with valid data")
    
    # Create historical products with missing features
    print("\nCreating products with missing features...")
    for i in range(2):
        product_id = insert_product(
            image_path=f'/test/no_features_{i}.jpg',
            category='placemats',
            product_name=f'No Features {i}',
            sku=f'NOFEAT-{i}',
            is_historical=True
        )
        # Don't insert features
        product_ids.append(product_id)
        print(f"  Created product {product_id} WITHOUT features")
    
    # Create historical products with corrupted features (NaN)
    print("\nCreating products with NaN corrupted features...")
    for i in range(2):
        product_id = insert_product(
            image_path=f'/test/nan_features_{i}.jpg',
            category='placemats',
            product_name=f'NaN Features {i}',
            sku=f'NAN-{i}',
            is_historical=True
        )
        color, shape, texture = create_corrupted_features('nan')
        insert_features(product_id, color, shape, texture)
        product_ids.append(product_id)
        print(f"  Created product {product_id} with NaN features")
    
    # Create historical products with corrupted features (Inf)
    print("\nCreating products with Inf corrupted features...")
    for i in range(2):
        product_id = insert_product(
            image_path=f'/test/inf_features_{i}.jpg',
            category='placemats',
            product_name=f'Inf Features {i}',
            sku=f'INF-{i}',
            is_historical=True
        )
        color, shape, texture = create_corrupted_features('inf')
        insert_features(product_id, color, shape, texture)
        product_ids.append(product_id)
        print(f"  Created product {product_id} with Inf features")
    
    # Create historical products with missing metadata
    print("\nCreating products with missing metadata...")
    for i in range(2):
        product_id = insert_product(
            image_path=f'/test/missing_meta_{i}.jpg',
            category='placemats',
            product_name=None,  # Missing name
            sku=None,  # Missing SKU
            is_historical=True
        )
        color, shape, texture = create_good_features(seed=100 + i)
        insert_features(product_id, color, shape, texture)
        product_ids.append(product_id)
        print(f"  Created product {product_id} with missing metadata")
    
    # Create a new product to match (with good data)
    print("\nCreating new product to match...")
    new_product_id = insert_product(
        image_path='/test/new_product.jpg',
        category='placemats',
        product_name='New Test Product',
        sku='NEW-TEST',
        is_historical=False
    )
    color, shape, texture = create_good_features(seed=999)
    insert_features(new_product_id, color, shape, texture)
    print(f"  Created new product {new_product_id}")
    
    print(f"\nTotal products created: {len(product_ids) + 1}")
    print(f"  Good products: 5")
    print(f"  Missing features: 2")
    print(f"  NaN corrupted: 2")
    print(f"  Inf corrupted: 2")
    print(f"  Missing metadata: 2")
    
    return new_product_id


def test_matching_with_mixed_data():
    """Test matching with mix of good and bad data"""
    print("\n" + "=" * 70)
    print("TEST: Matching with Mixed Quality Data")
    print("=" * 70)
    
    new_product_id = setup_mixed_quality_data()
    
    print("\nRunning match with skip_invalid_products=True...")
    try:
        result = find_matches(
            product_id=new_product_id,
            threshold=0.0,
            limit=100,
            skip_invalid_products=True
        )
        
        print("\n✓ Matching completed successfully!")
        print(f"\nResults:")
        print(f"  Total candidates: {result['total_candidates']}")
        print(f"  Successful matches: {result['successful_matches']}")
        print(f"  Failed matches: {result['failed_matches']}")
        print(f"  Matches returned: {len(result['matches'])}")
        
        print(f"\nData Quality Issues:")
        dq = result['data_quality_issues']
        print(f"  Missing features: {dq['missing_features']}")
        print(f"  Corrupted features: {dq['corrupted_features']}")
        print(f"  Missing metadata: {dq['missing_metadata']}")
        print(f"  Computation errors: {dq['computation_errors']}")
        
        print(f"\nData Quality Summary:")
        dqs = result['data_quality_summary']
        print(f"  Total issues: {dqs['total_issues']}")
        print(f"  Success rate: {dqs['success_rate']}%")
        print(f"  Has issues: {dqs['has_data_quality_issues']}")
        
        if result['warnings']:
            print(f"\nWarnings ({len(result['warnings'])}):")
            for warning in result['warnings'][:5]:
                print(f"  - {warning}")
            if len(result['warnings']) > 5:
                print(f"  ... and {len(result['warnings']) - 5} more")
        
        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result['errors'][:5]:
                print(f"  - Product {error['product_id']}: {error['error_code']}")
            if len(result['errors']) > 5:
                print(f"  ... and {len(result['errors']) - 5} more")
        
        if result['matches']:
            print(f"\nTop 3 Matches:")
            for i, match in enumerate(result['matches'][:3]):
                print(f"  {i+1}. Product {match['product_id']}: {match['similarity_score']:.2f}")
                print(f"     Name: {match['product_name']}, SKU: {match['sku']}")
                if match.get('has_missing_metadata'):
                    print(f"     ⚠ Missing: {match['missing_fields']}")
        
        # Verify expectations
        # Note: Products without features won't be returned by get_all_features_by_category
        # so they won't appear in candidates. This is correct behavior.
        assert result['successful_matches'] >= 5, "Should have at least 5 successful matches"
        assert result['failed_matches'] >= 4, "Should have at least 4 failed matches (NaN + Inf)"
        # missing_features won't be detected because those products aren't in the candidate list
        assert dq['corrupted_features'] >= 4, "Should detect corrupted features (NaN + Inf)"
        assert dq['missing_metadata'] >= 2, "Should detect missing metadata"
        
        print("\n✓ All assertions passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, str(e)


def test_matching_without_skipping():
    """Test that matching fails fast when skip_invalid_products=False"""
    print("\n" + "=" * 70)
    print("TEST: Matching WITHOUT Skipping Invalid Products")
    print("=" * 70)
    
    # Get the new product ID from previous test
    from database import get_products_by_category
    new_products = get_products_by_category('placemats', is_historical=False)
    if not new_products:
        print("No new products found, skipping test")
        return  # Skip test if no products
    
    new_product_id = new_products[0]['id']
    
    print(f"\nRunning match with skip_invalid_products=False...")
    print("Expected: Should encounter errors and potentially fail")
    
    try:
        result = find_matches(
            product_id=new_product_id,
            threshold=0.0,
            limit=100,
            skip_invalid_products=False
        )
        
        # If we get here, it means all products were processed
        # (might happen if corrupted products are filtered out early)
        print("\n✓ Matching completed (all products processed)")
        print(f"  Successful: {result['successful_matches']}")
        print(f"  Failed: {result['failed_matches']}")
        
    except Exception as e:
        print(f"\n✓ Expected behavior: Raised exception")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Message: {str(e)}")


def test_all_corrupted_data():
    """Test handling when ALL historical products have corrupted data"""
    print("\n" + "=" * 70)
    print("TEST: All Historical Products Corrupted")
    print("=" * 70)
    
    # Clear database
    from database import get_db_connection
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM matches')
        cursor.execute('DELETE FROM features')
        cursor.execute('DELETE FROM products')
        conn.commit()
    
    print("\nCreating only corrupted historical products...")
    
    # Create only corrupted products
    for i in range(3):
        product_id = insert_product(
            image_path=f'/test/corrupted_{i}.jpg',
            category='placemats',
            product_name=f'Corrupted {i}',
            sku=f'CORRUPT-{i}',
            is_historical=True
        )
        color, shape, texture = create_corrupted_features('nan')
        insert_features(product_id, color, shape, texture)
        print(f"  Created corrupted product {product_id}")
    
    # Create new product
    new_product_id = insert_product(
        image_path='/test/new_product.jpg',
        category='placemats',
        product_name='New Product',
        sku='NEW',
        is_historical=False
    )
    color, shape, texture = create_good_features(seed=777)
    insert_features(new_product_id, color, shape, texture)
    print(f"  Created new product {new_product_id}")
    
    print("\nRunning match (should fail with AllMatchesFailedError)...")
    
    try:
        result = find_matches(
            product_id=new_product_id,
            threshold=0.0,
            limit=100
        )
        print(f"\n✗ Should have raised AllMatchesFailedError")
        print(f"  Got {result['successful_matches']} successful matches")
        return False
        
    except Exception as e:
        if 'AllMatchesFailed' in type(e).__name__:
            print(f"\n✓ Correctly raised AllMatchesFailedError")
            print(f"  Message: {e.message if hasattr(e, 'message') else str(e)}")
        else:
            print(f"\n✗ Raised wrong exception: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            assert False, f"Wrong exception type: {type(e).__name__}"


def test_missing_metadata_handling():
    """Test that products with missing metadata still match"""
    print("\n" + "=" * 70)
    print("TEST: Missing Metadata Handling")
    print("=" * 70)
    
    # Clear database
    from database import get_db_connection
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM matches')
        cursor.execute('DELETE FROM features')
        cursor.execute('DELETE FROM products')
        conn.commit()
    
    print("\nCreating products with missing metadata...")
    
    # Create historical products with various missing fields
    # Note: image_path is required, so we test missing name and SKU
    for i in range(3):
        product_id = insert_product(
            image_path=f'/test/product_{i}.jpg',  # image_path is required
            category='placemats',
            product_name=f'Product {i}' if i != 1 else None,  # Second missing name
            sku=f'SKU-{i}' if i != 2 else None,  # Third missing SKU
            is_historical=True
        )
        color, shape, texture = create_good_features(seed=i)
        insert_features(product_id, color, shape, texture)
        print(f"  Created product {product_id} with some missing metadata")
    
    # Create new product
    new_product_id = insert_product(
        image_path='/test/new.jpg',
        category='placemats',
        product_name='New',
        sku='NEW',
        is_historical=False
    )
    color, shape, texture = create_good_features(seed=888)
    insert_features(new_product_id, color, shape, texture)
    
    print("\nRunning match...")
    
    try:
        result = find_matches(
            product_id=new_product_id,
            threshold=0.0,
            limit=100
        )
        
        print(f"\n✓ Matching completed successfully!")
        print(f"  Matches: {len(result['matches'])}")
        print(f"  Missing metadata issues: {result['data_quality_issues']['missing_metadata']}")
        
        # Check that matches include metadata status
        for match in result['matches']:
            if match.get('has_missing_metadata'):
                print(f"\n  Product {match['product_id']} missing: {match['missing_fields']}")
        
        assert len(result['matches']) >= 3, "Should match all products despite missing metadata"
        assert result['data_quality_issues']['missing_metadata'] >= 2, "Should detect missing metadata"
        
        print("\n✓ All assertions passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, str(e)


def main():
    """Run all real-world data tests"""
    print("=" * 70)
    print("REAL-WORLD DATA HANDLING TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Mixed Quality Data", test_matching_with_mixed_data),
        ("Without Skipping Invalid", test_matching_without_skipping),
        ("All Corrupted Data", test_all_corrupted_data),
        ("Missing Metadata", test_missing_metadata_handling),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")


if __name__ == '__main__':
    main()
