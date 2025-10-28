"""
Integration test for feature caching functionality.
"""

import os
import sys
import numpy as np
import tempfile
from PIL import Image

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import init_db, insert_product, get_product_by_id, delete_product
from feature_cache import (
    get_feature_cache,
    extract_and_cache_features,
    batch_extract_features
)
from image_processing import ImageProcessingError


def create_test_image():
    """Create a test image file"""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[0:256, 0:256] = [255, 0, 0]  # Red
    img[0:256, 256:512] = [0, 255, 0]  # Green
    img[256:512, 0:256] = [0, 0, 255]  # Blue
    img[256:512, 256:512] = [255, 255, 0]  # Yellow
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_path = temp_file.name
    temp_file.close()
    
    pil_img = Image.fromarray(img, 'RGB')
    pil_img.save(temp_path, format='PNG')
    
    return temp_path


def test_feature_caching():
    """Test feature extraction and caching"""
    print("Testing feature caching...")
    
    # Initialize database
    init_db()
    
    # Create test image
    test_img = create_test_image()
    
    try:
        # Insert product
        product_id = insert_product(
            image_path=test_img,
            category='test_category',
            product_name='Test Product',
            is_historical=True
        )
        print(f"✓ Created test product with ID: {product_id}")
        
        # First extraction - should extract and cache
        features1, was_cached1 = extract_and_cache_features(product_id, test_img)
        assert not was_cached1, "First extraction should not be cached"
        assert 'color_features' in features1
        assert 'shape_features' in features1
        assert 'texture_features' in features1
        assert len(features1['color_features']) == 256
        assert len(features1['shape_features']) == 7
        assert len(features1['texture_features']) == 256
        print("✓ Features extracted and cached successfully")
        print(f"  - Color features: {len(features1['color_features'])} dimensions")
        print(f"  - Shape features: {len(features1['shape_features'])} dimensions")
        print(f"  - Texture features: {len(features1['texture_features'])} dimensions")
        
        # Second extraction - should retrieve from cache
        features2, was_cached2 = extract_and_cache_features(product_id, test_img)
        assert was_cached2, "Second extraction should be cached"
        assert np.array_equal(features1['color_features'], features2['color_features'])
        assert np.array_equal(features1['shape_features'], features2['shape_features'])
        assert np.array_equal(features1['texture_features'], features2['texture_features'])
        print("✓ Features retrieved from cache successfully")
        
        # Test memory cache
        cache = get_feature_cache()
        stats = cache.get_cache_stats()
        print(f"✓ Cache stats: {stats['memory_cache_size']}/{stats['max_memory_cache_size']} items")
        
        # Cleanup
        delete_product(product_id)
        print("✓ Test product deleted")
        
    finally:
        os.unlink(test_img)
    
    print("✓ Feature caching tests passed\n")


def test_batch_feature_extraction():
    """Test batch feature extraction"""
    print("Testing batch feature extraction...")
    
    # Initialize database
    init_db()
    
    # Create multiple test images
    test_images = []
    product_ids = []
    
    try:
        for i in range(3):
            img_path = create_test_image()
            test_images.append(img_path)
            
            product_id = insert_product(
                image_path=img_path,
                category=f'category_{i}',
                product_name=f'Product {i}',
                is_historical=True
            )
            product_ids.append(product_id)
        
        print(f"✓ Created {len(product_ids)} test products")
        
        # Batch extract features
        results = batch_extract_features(product_ids)
        
        # Verify results
        assert len(results) == len(product_ids), f"Expected {len(product_ids)} results, got {len(results)}"
        
        success_count = sum(1 for r in results.values() if r['success'])
        print(f"✓ Batch extraction completed: {success_count}/{len(product_ids)} successful")
        
        for product_id, result in results.items():
            if result['success']:
                features = result['features']
                assert len(features['color_features']) == 256
                assert len(features['shape_features']) == 7
                assert len(features['texture_features']) == 256
                print(f"  - Product {product_id}: ✓")
            else:
                print(f"  - Product {product_id}: ✗ {result['error']}")
        
        # Cleanup
        for product_id in product_ids:
            delete_product(product_id)
        
    finally:
        for img_path in test_images:
            if os.path.exists(img_path):
                os.unlink(img_path)
    
    print("✓ Batch feature extraction tests passed\n")


def test_error_handling_with_cache():
    """Test error handling in caching system"""
    print("Testing error handling with cache...")
    
    # Initialize database
    init_db()
    
    # Create corrupted image file
    corrupted_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    corrupted_file.write(b'Not an image')
    corrupted_file.close()
    
    try:
        # Insert product with corrupted image
        product_id = insert_product(
            image_path=corrupted_file.name,
            category='test',
            is_historical=True
        )
        
        # Try to extract features - should fail gracefully
        try:
            features, was_cached = extract_and_cache_features(product_id, corrupted_file.name)
            assert False, "Should have raised ImageProcessingError"
        except ImageProcessingError as e:
            print(f"✓ Error caught correctly: {e.error_code}")
            print(f"  Message: {e.message}")
            print(f"  Suggestion: {e.suggestion}")
        
        # Cleanup
        delete_product(product_id)
        
    finally:
        os.unlink(corrupted_file.name)
    
    print("✓ Error handling tests passed\n")


def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("Running Feature Cache Integration Tests")
    print("=" * 60 + "\n")
    
    try:
        test_feature_caching()
        test_batch_feature_extraction()
        test_error_handling_with_cache()
        
        print("=" * 60)
        print("ALL INTEGRATION TESTS PASSED ✓")
        print("=" * 60)
        return True
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
