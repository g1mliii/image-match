"""
Test script for database layer implementation
Tests CRUD operations, feature storage/retrieval, and numpy serialization
"""
import os
import sys
import numpy as np
import pytest

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import (
    init_db, 
    insert_product, 
    get_product_by_id,
    update_product,
    delete_product,
    get_products_by_category,
    count_products,
    insert_features,
    get_features_by_product_id,
    update_features,
    delete_features,
    get_all_features_by_category,
    insert_match,
    get_matches_for_product,
    delete_matches_for_product,
    DB_PATH
)

def cleanup_test_db():
    """Remove test database if it exists"""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database at {DB_PATH}")

def test_database_initialization():
    """Test database initialization"""
    print("\n=== Testing Database Initialization ===")
    init_db()
    assert os.path.exists(DB_PATH), "Database file should exist"
    print("✓ Database initialized successfully")

@pytest.fixture
def product_id():
    """Fixture that creates a test product and returns its ID"""
    product_id = insert_product(
        image_path="/path/to/image.jpg",
        category="placemats",
        product_name="Test Placemat",
        sku="PM-001",
        is_historical=True,
        metadata='{"color": "blue"}'
    )
    return product_id

@pytest.fixture
def product_id_with_features():
    """Fixture that creates a test product with features and returns its ID"""
    product_id = insert_product(
        image_path="/path/to/image_with_features.jpg",
        category="placemats",
        product_name="Test Placemat With Features",
        sku="PM-002",
        is_historical=True,
        metadata='{"color": "blue"}'
    )
    # Add features for the product (needed for deletion test)
    color_features = np.random.rand(256).astype(np.float32)
    shape_features = np.random.rand(7).astype(np.float32)
    texture_features = np.random.rand(256).astype(np.float32)
    insert_features(product_id, color_features, shape_features, texture_features)
    return product_id

@pytest.fixture
def new_product_id():
    """Fixture that creates a new product for matching tests"""
    product_id = insert_product(
        image_path="/path/to/new.jpg",
        category="textiles",
        product_name="New Product",
        sku="NEW-001",
        is_historical=False
    )
    return product_id

@pytest.fixture
def matched_product_ids():
    """Fixture that creates matched products for testing"""
    ids = []
    for i in range(3):
        product_id = insert_product(
            image_path=f"/path/to/matched{i}.jpg",
            category="textiles",
            product_name=f"Matched Product {i}",
            sku=f"MATCH-{i:03d}",
            is_historical=True
        )
        ids.append(product_id)
    return ids

def test_product_crud():
    """Test product CRUD operations"""
    print("\n=== Testing Product CRUD Operations ===")
    
    # Create with full data
    product_id = insert_product(
        image_path="/path/to/image.jpg",
        category="placemats",
        product_name="Test Placemat",
        sku="PM-001",
        is_historical=True,
        metadata='{"color": "blue"}'
    )
    print(f"✓ Created product with ID: {product_id}")
    
    # Create with minimal data (only image_path required)
    minimal_product_id = insert_product(
        image_path="/path/to/minimal.jpg",
        is_historical=False
    )
    print(f"✓ Created minimal product with ID: {minimal_product_id} (no category, name, or SKU)")
    
    # Read
    product = get_product_by_id(product_id)
    assert product is not None, "Product should exist"
    assert product['category'] == "placemats", "Category should match"
    assert product['product_name'] == "Test Placemat", "Name should match"
    print(f"✓ Retrieved product: {dict(product)}")
    
    # Read minimal product
    minimal_product = get_product_by_id(minimal_product_id)
    assert minimal_product is not None, "Minimal product should exist"
    assert minimal_product['category'] is None, "Category should be NULL"
    assert minimal_product['product_name'] is None, "Name should be NULL"
    print(f"✓ Retrieved minimal product with NULL fields")
    
    # Update
    success = update_product(
        product_id,
        product_name="Updated Placemat",
        sku="PM-002"
    )
    assert success, "Update should succeed"
    updated_product = get_product_by_id(product_id)
    assert updated_product['product_name'] == "Updated Placemat", "Name should be updated"
    assert updated_product['sku'] == "PM-002", "SKU should be updated"
    print("✓ Updated product successfully")
    
    # Test category filtering
    products = get_products_by_category("placemats", is_historical=True)
    assert len(products) > 0, "Should find products in category"
    print(f"✓ Found {len(products)} products in 'placemats' category")
    
    # Test uncategorized products
    uncategorized = get_products_by_category(None)
    assert len(uncategorized) > 0, "Should find uncategorized products"
    print(f"✓ Found {len(uncategorized)} uncategorized products")
    
    # Test count
    count = count_products(category="placemats", is_historical=True)
    assert count > 0, "Should count products"
    print(f"✓ Counted {count} historical products in 'placemats' category")

def test_feature_operations(product_id):
    """Test feature storage and retrieval with numpy arrays"""
    print("\n=== Testing Feature Operations ===")
    
    # Create test feature vectors
    color_features = np.random.rand(256).astype(np.float32)
    shape_features = np.random.rand(7).astype(np.float32)
    texture_features = np.random.rand(256).astype(np.float32)
    
    print(f"✓ Created test feature vectors:")
    print(f"  - Color: shape {color_features.shape}, dtype {color_features.dtype}")
    print(f"  - Shape: shape {shape_features.shape}, dtype {shape_features.dtype}")
    print(f"  - Texture: shape {texture_features.shape}, dtype {texture_features.dtype}")
    
    # Insert features
    feature_id = insert_features(product_id, color_features, shape_features, texture_features)
    print(f"✓ Inserted features with ID: {feature_id}")
    
    # Retrieve features
    retrieved_features = get_features_by_product_id(product_id)
    assert retrieved_features is not None, "Features should exist"
    
    # Verify numpy arrays are correctly deserialized
    assert np.allclose(retrieved_features['color_features'], color_features), "Color features should match"
    assert np.allclose(retrieved_features['shape_features'], shape_features), "Shape features should match"
    assert np.allclose(retrieved_features['texture_features'], texture_features), "Texture features should match"
    print("✓ Retrieved and verified feature vectors match original")
    
    # Update features
    new_color_features = np.random.rand(256).astype(np.float32)
    success = update_features(product_id, color_features=new_color_features)
    assert success, "Update should succeed"
    
    updated_features = get_features_by_product_id(product_id)
    assert np.allclose(updated_features['color_features'], new_color_features), "Updated color features should match"
    assert np.allclose(updated_features['shape_features'], shape_features), "Shape features should remain unchanged"
    print("✓ Updated color features successfully")

def test_category_features():
    """Test retrieving all features by category"""
    print("\n=== Testing Category Feature Retrieval ===")
    
    # Create multiple products with features - some with category, some without
    product_ids = []
    for i in range(3):
        # First two have category, third one doesn't
        category = "dinnerware" if i < 2 else None
        pid = insert_product(
            image_path=f"/path/to/image{i}.jpg",
            category=category,
            product_name=f"Test Product {i}" if i < 2 else None,
            is_historical=True
        )
        product_ids.append(pid)
        
        # Add features
        color = np.random.rand(256).astype(np.float32)
        shape = np.random.rand(7).astype(np.float32)
        texture = np.random.rand(256).astype(np.float32)
        insert_features(pid, color, shape, texture)
    
    print(f"✓ Created {len(product_ids)} products with features (1 without category)")
    
    # Retrieve all features for category (excluding uncategorized)
    category_features = get_all_features_by_category("dinnerware", is_historical=True)
    assert len(category_features) == 2, "Should retrieve only categorized features"
    print(f"✓ Retrieved {len(category_features)} feature sets for 'dinnerware' category")
    
    # Retrieve including uncategorized
    all_features = get_all_features_by_category("dinnerware", is_historical=True, include_uncategorized=True)
    assert len(all_features) >= 3, "Should retrieve all features including uncategorized"
    print(f"✓ Retrieved {len(all_features)} feature sets including uncategorized products")
    
    # Retrieve all features regardless of category
    all_products_features = get_all_features_by_category(category=None, is_historical=True)
    assert len(all_products_features) >= 3, "Should retrieve all features"
    print(f"✓ Retrieved {len(all_products_features)} feature sets for all categories")
    
    for prod_id, features in category_features:
        assert 'color_features' in features, "Should have color features"
        assert 'shape_features' in features, "Should have shape features"
        assert 'texture_features' in features, "Should have texture features"
        assert 'category' in features, "Should have category field"
        assert features['color_features'].shape == (256,), "Color features should have correct shape"
        assert features['shape_features'].shape == (7,), "Shape features should have correct shape"
        assert features['texture_features'].shape == (256,), "Texture features should have correct shape"

def test_match_operations(new_product_id, matched_product_ids):
    """Test match storage and retrieval"""
    print("\n=== Testing Match Operations ===")
    
    # Insert matches
    for i, matched_id in enumerate(matched_product_ids[:3]):
        similarity_score = 85.5 - (i * 10)
        color_score = 90.0 - (i * 5)
        shape_score = 80.0 - (i * 10)
        texture_score = 85.0 - (i * 15)
        
        match_id = insert_match(
            new_product_id,
            matched_id,
            similarity_score,
            color_score,
            shape_score,
            texture_score
        )
        print(f"✓ Inserted match {i+1} with ID: {match_id}, score: {similarity_score}")
    
    # Retrieve matches
    matches = get_matches_for_product(new_product_id, limit=10)
    assert len(matches) == 3, "Should retrieve all matches"
    
    # Verify matches are sorted by similarity score
    scores = [match['similarity_score'] for match in matches]
    assert scores == sorted(scores, reverse=True), "Matches should be sorted by score descending"
    print(f"✓ Retrieved {len(matches)} matches, sorted by similarity score")
    
    # Test delete matches
    success = delete_matches_for_product(new_product_id)
    assert success, "Delete should succeed"
    
    matches_after_delete = get_matches_for_product(new_product_id)
    assert len(matches_after_delete) == 0, "All matches should be deleted"
    print("✓ Deleted all matches successfully")

def test_product_deletion(product_id_with_features):
    """Test product deletion with cascading"""
    print("\n=== Testing Product Deletion ===")
    
    # Verify product and features exist
    product = get_product_by_id(product_id_with_features)
    assert product is not None, "Product should exist before deletion"
    
    features = get_features_by_product_id(product_id_with_features)
    assert features is not None, "Features should exist before deletion"
    
    # Delete product
    success = delete_product(product_id_with_features)
    assert success, "Deletion should succeed"
    
    # Verify product is deleted
    deleted_product = get_product_by_id(product_id_with_features)
    assert deleted_product is None, "Product should not exist after deletion"
    
    # Verify features are deleted
    deleted_features = get_features_by_product_id(product_id_with_features)
    assert deleted_features is None, "Features should not exist after deletion"
    
    print("✓ Product and associated features deleted successfully")

def test_missing_data_utilities():
    """Test utility functions for handling missing/incomplete data"""
    print("\n=== Testing Missing Data Utilities ===")
    
    from database import (
        get_products_without_category,
        get_products_without_features,
        get_incomplete_products,
        get_all_categories,
        bulk_update_category
    )
    
    # Create products with various missing fields
    complete_id = insert_product(
        image_path="/complete.jpg",
        category="glassware",
        product_name="Complete Product",
        sku="GL-001",
        is_historical=True
    )
    
    no_category_id = insert_product(
        image_path="/no_category.jpg",
        product_name="No Category Product",
        sku="NC-001",
        is_historical=True
    )
    
    no_features_id = insert_product(
        image_path="/no_features.jpg",
        category="silverware",
        product_name="No Features Product",
        is_historical=True
    )
    
    # Add features to some products
    color = np.random.rand(256).astype(np.float32)
    shape = np.random.rand(7).astype(np.float32)
    texture = np.random.rand(256).astype(np.float32)
    insert_features(complete_id, color, shape, texture)
    
    print("✓ Created test products with various missing fields")
    
    # Test get_products_without_category
    uncategorized = get_products_without_category(is_historical=True)
    assert len(uncategorized) > 0, "Should find uncategorized products"
    assert any(p['id'] == no_category_id for p in uncategorized), "Should include no_category product"
    print(f"✓ Found {len(uncategorized)} products without category")
    
    # Test get_products_without_features
    no_features = get_products_without_features()
    assert len(no_features) > 0, "Should find products without features"
    assert any(p['id'] == no_features_id for p in no_features), "Should include no_features product"
    print(f"✓ Found {len(no_features)} products without features")
    
    # Test get_incomplete_products
    incomplete = get_incomplete_products()
    assert len(incomplete) > 0, "Should find incomplete products"
    
    for product in incomplete:
        if product['id'] == no_category_id:
            assert product['missing_category'], "Should flag missing category"
            print(f"✓ Correctly identified product {product['id']} missing category")
        if product['id'] == no_features_id:
            assert product['missing_features'], "Should flag missing features"
            print(f"✓ Correctly identified product {product['id']} missing features")
    
    # Test get_all_categories
    categories = get_all_categories()
    assert len(categories) > 0, "Should find categories"
    assert "glassware" in categories, "Should include glassware"
    assert "silverware" in categories, "Should include silverware"
    print(f"✓ Found {len(categories)} unique categories: {categories}")
    
    # Test bulk_update_category
    updated_count = bulk_update_category([no_category_id], "tableware")
    assert updated_count == 1, "Should update one product"
    
    updated_product = get_product_by_id(no_category_id)
    assert updated_product['category'] == "tableware", "Category should be updated"
    print(f"✓ Bulk updated {updated_count} product(s) to 'tableware' category")

def run_all_tests():
    """Run all database tests"""
    print("=" * 60)
    print("Starting Database Layer Tests")
    print("=" * 60)
    
    try:
        # Clean up and initialize
        cleanup_test_db()
        test_database_initialization()
        
        # Test product CRUD
        product_id = test_product_crud()
        
        # Test feature operations
        feature_id = test_feature_operations(product_id)
        
        # Test category features
        dinnerware_product_ids = test_category_features()
        
        # Test match operations
        test_match_operations(product_id, dinnerware_product_ids)
        
        # Test missing data utilities
        test_missing_data_utilities()
        
        # Test deletion
        test_product_deletion(product_id)
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
