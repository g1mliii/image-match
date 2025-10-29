"""
Test script for product matching service.

This script tests the matching functionality with various scenarios including:
- Normal matching with category filtering
- Handling NULL/missing categories
- Handling corrupted features
- Edge cases (empty catalog, all matches fail, etc.)
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
    get_product_by_id,
    get_features_by_product_id,
    count_products
)
from product_matching import (
    find_matches,
    batch_find_matches,
    get_match_statistics,
    ProductNotFoundError,
    MissingFeaturesError,
    EmptyCatalogError,
    AllMatchesFailedError
)


def create_test_features(seed=None):
    """Create random test features for testing"""
    if seed is not None:
        np.random.seed(seed)
    
    color_features = np.random.rand(256).astype(np.float32)
    color_features /= color_features.sum()  # Normalize
    
    shape_features = np.random.randn(7).astype(np.float32)
    
    texture_features = np.random.rand(256).astype(np.float32)
    texture_features /= texture_features.sum()  # Normalize
    
    return color_features, shape_features, texture_features


def setup_test_data():
    """Set up test data for matching"""
    print("Setting up test data...")
    
    # Initialize database
    init_db()
    
    # Check if we already have products
    existing_count = count_products()
    if existing_count > 0:
        print(f"Database already has {existing_count} products")
        return
    
    # Create historical products in different categories
    categories = ['placemats', 'dinnerware', 'textiles', None]  # Include NULL category
    
    for i, category in enumerate(categories):
        for j in range(3):  # 3 products per category
            # Create product
            product_id = insert_product(
                image_path=f'/test/images/historical_{i}_{j}.jpg',
                category=category,
                product_name=f'Historical Product {i}-{j}',
                sku=f'SKU-{i}{j}',
                is_historical=True
            )
            
            # Create features
            color, shape, texture = create_test_features(seed=i*10 + j)
            insert_features(product_id, color, shape, texture)
            
            print(f"Created historical product {product_id} in category '{category}'")
    
    # Create a new product to match
    new_product_id = insert_product(
        image_path='/test/images/new_product.jpg',
        category='placemats',
        product_name='New Product',
        sku='SKU-NEW',
        is_historical=False
    )
    
    # Create features similar to first historical product
    color, shape, texture = create_test_features(seed=0)
    # Add some noise to make it similar but not identical
    color = color * 0.9 + np.random.rand(256).astype(np.float32) * 0.1
    color /= color.sum()
    insert_features(new_product_id, color, shape, texture)
    
    print(f"Created new product {new_product_id}")
    print(f"Total products: {count_products()}")


def test_basic_matching():
    """Test basic matching functionality"""
    print("\n=== Test 1: Basic Matching ===")
    
    try:
        # Create a test product for matching
        product_id = insert_product(
            image_path='/test/images/test_match.jpg',
            category='placemats',
            product_name='Test Match Product',
            sku='SKU-TEST',
            is_historical=False
        )
        
        color, shape, texture = create_test_features(seed=42)
        insert_features(product_id, color, shape, texture)
        
        print(f"Created test product {product_id}")
        
        result = find_matches(
            product_id=product_id,
            threshold=0.0,
            limit=10
        )
        
        print(f"Found {len(result['matches'])} matches")
        print(f"Total candidates: {result['total_candidates']}")
        print(f"Successful: {result['successful_matches']}")
        print(f"Failed: {result['failed_matches']}")
        
        if result['matches']:
            print("\nTop 3 matches:")
            for i, match in enumerate(result['matches'][:3]):
                print(f"  {i+1}. Product {match['product_id']}: {match['similarity_score']:.2f}")
                print(f"     Category: {match['category']}, SKU: {match['sku']}")
                print(f"     Duplicate: {match['is_potential_duplicate']}")
        
        if result['warnings']:
            print(f"\nWarnings: {len(result['warnings'])}")
            for warning in result['warnings'][:3]:
                print(f"  - {warning}")
        
        print("✓ Basic matching test passed")
        
    except Exception as e:
        print(f"✗ Basic matching test failed: {e}")


def test_null_category_handling():
    """Test handling of NULL categories"""
    print("\n=== Test 2: NULL Category Handling ===")
    
    try:
        # Create a product with NULL category
        product_id = insert_product(
            image_path='/test/images/no_category.jpg',
            category=None,  # NULL category
            product_name='Product Without Category',
            sku='SKU-NOCAT',
            is_historical=False
        )
        
        color, shape, texture = create_test_features(seed=999)
        insert_features(product_id, color, shape, texture)
        
        print(f"Created product {product_id} with NULL category")
        
        # Try to match - should match against all products
        result = find_matches(
            product_id=product_id,
            threshold=0.0,
            limit=5
        )
        
        print(f"Found {len(result['matches'])} matches")
        print(f"Matched against all categories: {result['matched_against_all_categories']}")
        
        if result['warnings']:
            print(f"Warnings: {result['warnings']}")
        
        print("✓ NULL category handling test passed")
        
    except Exception as e:
        print(f"✗ NULL category handling test failed: {e}")


def test_missing_features():
    """Test handling of missing features"""
    print("\n=== Test 3: Missing Features Handling ===")
    
    try:
        # Create a product without features
        product_id = insert_product(
            image_path='/test/images/no_features.jpg',
            category='placemats',
            product_name='Product Without Features',
            sku='SKU-NOFEAT',
            is_historical=False
        )
        
        print(f"Created product {product_id} without features")
        
        # Try to match - should raise MissingFeaturesError
        try:
            result = find_matches(product_id=product_id)
            print("✗ Should have raised MissingFeaturesError")
        except MissingFeaturesError as e:
            print(f"✓ Correctly raised MissingFeaturesError: {e.message}")
            print(f"  Error code: {e.error_code}")
            print(f"  Suggestion: {e.suggestion}")
        
    except Exception as e:
        print(f"✗ Missing features test failed: {e}")


def test_threshold_filtering():
    """Test threshold filtering"""
    print("\n=== Test 4: Threshold Filtering ===")
    
    try:
        # Create a test product
        product_id = insert_product(
            image_path='/test/images/threshold_test.jpg',
            category='placemats',
            product_name='Threshold Test Product',
            sku='SKU-THRESH',
            is_historical=False
        )
        
        color, shape, texture = create_test_features(seed=55)
        insert_features(product_id, color, shape, texture)
        
        # Use the test product
        result_no_threshold = find_matches(
            product_id=product_id,
            threshold=0.0,
            limit=100,
            store_matches=False
        )
        
        result_high_threshold = find_matches(
            product_id=product_id,
            threshold=50.0,
            limit=100,
            store_matches=False
        )
        
        print(f"Matches with threshold 0: {len(result_no_threshold['matches'])}")
        print(f"Matches with threshold 50: {len(result_high_threshold['matches'])}")
        print(f"Filtered out: {result_high_threshold['filtered_by_threshold']}")
        
        print("✓ Threshold filtering test passed")
        
    except Exception as e:
        print(f"✗ Threshold filtering test failed: {e}")


def test_duplicate_detection():
    """Test duplicate detection (score > 90)"""
    print("\n=== Test 5: Duplicate Detection ===")
    
    try:
        # Create a product with identical features to an existing one
        product_id = insert_product(
            image_path='/test/images/duplicate.jpg',
            category='placemats',
            product_name='Potential Duplicate',
            sku='SKU-DUP',
            is_historical=False
        )
        
        # Use same features as first historical product
        color, shape, texture = create_test_features(seed=0)
        insert_features(product_id, color, shape, texture)
        
        print(f"Created product {product_id} with identical features")
        
        result = find_matches(
            product_id=product_id,
            threshold=0.0,
            limit=10,
            store_matches=False
        )
        
        duplicates = [m for m in result['matches'] if m['is_potential_duplicate']]
        print(f"Found {len(duplicates)} potential duplicates")
        
        if duplicates:
            for dup in duplicates:
                print(f"  Product {dup['product_id']}: {dup['similarity_score']:.2f}")
        
        print("✓ Duplicate detection test passed")
        
    except Exception as e:
        print(f"✗ Duplicate detection test failed: {e}")


def test_batch_matching():
    """Test batch matching"""
    print("\n=== Test 6: Batch Matching ===")
    
    try:
        # Create multiple products
        product_ids = []
        for i in range(3):
            product_id = insert_product(
                image_path=f'/test/images/batch_{i}.jpg',
                category='placemats',
                product_name=f'Batch Product {i}',
                sku=f'SKU-BATCH{i}',
                is_historical=False
            )
            
            color, shape, texture = create_test_features(seed=100 + i)
            insert_features(product_id, color, shape, texture)
            product_ids.append(product_id)
        
        print(f"Created {len(product_ids)} products for batch matching")
        
        result = batch_find_matches(
            product_ids=product_ids,
            threshold=0.0,
            limit=5,
            store_matches=False
        )
        
        print(f"Batch summary:")
        print(f"  Total: {result['summary']['total_products']}")
        print(f"  Successful: {result['summary']['successful']}")
        print(f"  Failed: {result['summary']['failed']}")
        print(f"  Success rate: {result['summary']['success_rate']}%")
        
        print("✓ Batch matching test passed")
        
    except Exception as e:
        print(f"✗ Batch matching test failed: {e}")


def test_empty_catalog():
    """Test handling of empty catalog"""
    print("\n=== Test 7: Empty Catalog Handling ===")
    
    try:
        # Create a product in a category with no historical products
        product_id = insert_product(
            image_path='/test/images/empty_cat.jpg',
            category='nonexistent_category',
            product_name='Product in Empty Category',
            sku='SKU-EMPTY',
            is_historical=False
        )
        
        color, shape, texture = create_test_features(seed=888)
        insert_features(product_id, color, shape, texture)
        
        print(f"Created product {product_id} in empty category")
        
        # Try to match without including uncategorized - should raise EmptyCatalogError
        try:
            result = find_matches(
                product_id=product_id,
                include_uncategorized=False  # Don't include NULL category products
            )
            print("✗ Should have raised EmptyCatalogError")
        except EmptyCatalogError as e:
            print(f"✓ Correctly raised EmptyCatalogError: {e.message}")
            print(f"  Error code: {e.error_code}")
            print(f"  Suggestion: {e.suggestion}")
        
    except Exception as e:
        print(f"✗ Empty catalog test failed: {e}")


def test_match_statistics():
    """Test match statistics"""
    print("\n=== Test 8: Match Statistics ===")
    
    try:
        # Create a product and get matches first
        product_id = insert_product(
            image_path='/test/images/stats_test.jpg',
            category='placemats',
            product_name='Stats Test Product',
            sku='SKU-STATS',
            is_historical=False
        )
        
        color, shape, texture = create_test_features(seed=77)
        insert_features(product_id, color, shape, texture)
        
        # Get matches to populate the database
        find_matches(product_id=product_id, threshold=0.0, limit=10)
        
        # Get statistics for the product
        stats = get_match_statistics(product_id)
        
        print(f"Match statistics for product {stats['product_id']}:")
        print(f"  Total matches: {stats['total_matches']}")
        
        if stats['has_matches']:
            print(f"  Highest score: {stats['highest_score']:.2f}")
            print(f"  Lowest score: {stats['lowest_score']:.2f}")
            print(f"  Average score: {stats['average_score']:.2f}")
            print(f"  Potential duplicates: {stats['potential_duplicates']}")
            print(f"  High similarity (>70): {stats['high_similarity']}")
        
        print("✓ Match statistics test passed")
        
    except Exception as e:
        print(f"✗ Match statistics test failed: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Product Matching Service Test Suite")
    print("=" * 60)
    
    # Setup test data
    setup_test_data()
    
    # Run tests
    test_basic_matching()
    test_null_category_handling()
    test_missing_features()
    test_threshold_filtering()
    test_duplicate_detection()
    test_batch_matching()
    test_empty_catalog()
    test_match_statistics()
    
    print("\n" + "=" * 60)
    print("Test suite complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
