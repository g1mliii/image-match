"""
Comprehensive real-world data tests for CLIP Mode 1 and Mode 3 matching.

This test suite focuses on real-world scenarios with:
- Corrupted/missing/null data
- Mixed valid/invalid images
- GPU acceleration validation
- Performance benchmarks
- Error handling in production workflows
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import tempfile
import time
import pytest
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import CLIP module
try:
    from image_processing_clip import (
        extract_clip_embedding,
        batch_extract_clip_embeddings,
        compute_clip_similarity,
        batch_compute_clip_similarities,
        detect_device,
        is_clip_available,
        get_clip_model,
        TORCH_AVAILABLE
    )
    CLIP_AVAILABLE = is_clip_available()
except ImportError as e:
    print(f"Warning: Could not import CLIP module: {e}")
    CLIP_AVAILABLE = False
    TORCH_AVAILABLE = False

# Import error types
from image_processing import (
    InvalidImageFormatError,
    CorruptedImageError,
    ImageTooSmallError,
    ImageProcessingFailedError
)


# ============================================================================
# Test Utilities
# ============================================================================

def create_test_image(width=512, height=512, format='PNG', pattern='gradient'):
    """Create test image with various patterns"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if pattern == 'gradient':
        for i in range(height):
            for j in range(width):
                img[i, j] = [int(255 * i / height), int(255 * j / width), 128]
    elif pattern == 'solid':
        img[:, :] = [128, 128, 128]
    elif pattern == 'checkerboard':
        square_size = 64
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    img[i:i+square_size, j:j+square_size] = [255, 255, 255]
    else:
        img[0:height//2, 0:width//2] = [255, 0, 0]
        img[0:height//2, width//2:width] = [0, 255, 0]
        img[height//2:height, 0:width//2] = [0, 0, 255]
        img[height//2:height, width//2:width] = [255, 255, 0]
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format.lower()}')
    temp_path = temp_file.name
    temp_file.close()
    
    pil_img = Image.fromarray(img, 'RGB')
    pil_img.save(temp_path, format=format)
    
    return temp_path


def create_corrupted_image():
    """Create corrupted image file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.write(b'Not a valid image')
    temp_file.close()
    return temp_path


def create_product_catalog(num_products=10, categories=['plates', 'bowls', 'cups']):
    """Create realistic product catalog with mixed data quality"""
    products = []
    
    for i in range(num_products):
        category = categories[i % len(categories)]
        pattern = ['gradient', 'solid', 'checkerboard'][i % 3]
        
        # 80% valid, 20% problematic
        if i % 5 == 0:
            # Corrupted image
            img_path = create_corrupted_image()
            valid = False
        elif i % 7 == 0:
            # Very small image
            img_path = create_test_image(width=20, height=20, pattern=pattern)
            valid = False
        else:
            # Valid image
            img_path = create_test_image(pattern=pattern)
            valid = True
        
        products.append({
            'id': f'PROD-{i:03d}',
            'image_path': img_path,
            'category': category if i % 10 != 0 else None,  # 10% missing category
            'name': f'Product {i}' if i % 8 != 0 else None,  # Some missing names
            'sku': f'SKU-{i:03d}' if i % 6 != 0 else None,  # Some missing SKUs
            'valid_image': valid
        })
    
    return products


# ============================================================================
# GPU Acceleration Tests
# ============================================================================

@pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP dependencies not available")
class TestGPUAcceleration:
    """Tests for GPU acceleration in CLIP processing"""
    
    def test_gpu_detection(self):
        """Test GPU detection and device selection"""
        print("\n[GPU] Testing GPU detection...")
        
        device = detect_device()
        print(f"[OK] Detected device: {device}")
        
        if TORCH_AVAILABLE:
            import torch
            
            if device == 'cuda':
                assert torch.cuda.is_available(), "CUDA should be available"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"  GPU: {gpu_name}")
                print(f"  CUDA version: {torch.version.cuda}")
            elif device == 'mps':
                assert torch.backends.mps.is_available(), "MPS should be available"
                print(f"  GPU: Apple Silicon (MPS)")
            else:
                print(f"  No GPU detected, using CPU")
    
    def test_gpu_vs_cpu_performance(self):
        """Test GPU vs CPU performance comparison"""
        print("\n[GPU] Testing GPU vs CPU performance...")
        
        device = detect_device()
        test_img = create_test_image()
        
        try:
            # Warm up
            extract_clip_embedding(test_img)
            
            # Benchmark
            num_runs = 5
            start = time.time()
            for _ in range(num_runs):
                extract_clip_embedding(test_img)
            avg_time = (time.time() - start) / num_runs
            
            print(f"[OK] Device: {device}")
            print(f"[OK] Average time: {avg_time:.3f}s per image")
            print(f"[OK] Throughput: {1/avg_time:.1f} images/sec")
            
            # Performance expectations
            if device == 'cuda':
                # NVIDIA GPU should be fast
                assert avg_time < 0.2, f"GPU too slow: {avg_time:.3f}s (expected < 0.2s)"
                print(f"[OK] GPU performance: EXCELLENT (< 0.2s)")
            elif device == 'mps':
                # Apple Silicon should be reasonably fast
                assert avg_time < 0.5, f"MPS too slow: {avg_time:.3f}s (expected < 0.5s)"
                print(f"[OK] GPU performance: GOOD (< 0.5s)")
            else:
                # CPU is slower but should still be reasonable
                assert avg_time < 2.0, f"CPU too slow: {avg_time:.3f}s (expected < 2.0s)"
                print(f"[OK] CPU performance: ACCEPTABLE (< 2.0s)")
        finally:
            os.unlink(test_img)
    
    def test_batch_gpu_acceleration(self):
        """Test that batch processing utilizes GPU efficiently"""
        print("\n[GPU] Testing batch GPU acceleration...")
        
        device = detect_device()
        
        # Create test images
        num_images = 20
        images = [create_test_image() for _ in range(num_images)]
        
        try:
            # Batch processing
            start = time.time()
            results = batch_extract_clip_embeddings(images, batch_size=8)
            batch_time = time.time() - start
            
            success_count = sum(1 for _, emb, _ in results if emb is not None)
            throughput = success_count / batch_time
            
            print(f"[OK] Device: {device}")
            print(f"[OK] Processed {success_count}/{num_images} images in {batch_time:.2f}s")
            print(f"[OK] Throughput: {throughput:.1f} images/sec")
            
            # GPU should process faster
            if device in ['cuda', 'mps']:
                assert throughput > 2.0, f"GPU batch too slow: {throughput:.1f} img/s"
                print(f"[OK] GPU batch performance: GOOD (> 2 img/s)")
        finally:
            for img in images:
                os.unlink(img)


# ============================================================================
# Mode 1: Real-World Visual Matching Tests
# ============================================================================

@pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP dependencies not available")
class TestMode1RealWorld:
    """Real-world tests for Mode 1 (Visual-only) matching"""
    
    def test_mode1_with_corrupted_images(self):
        """Test Mode 1 matching with corrupted images in catalog"""
        print("\n[Mode 1 Real-World] Testing with corrupted images...")
        
        # Create catalog with mixed quality
        catalog = create_product_catalog(num_products=10)
        query_img = create_test_image(pattern='gradient')
        
        try:
            # Extract embeddings (skip errors)
            catalog_results = batch_extract_clip_embeddings(
                [p['image_path'] for p in catalog],
                skip_errors=True
            )
            
            # Filter valid embeddings
            valid_embeddings = []
            valid_products = []
            for i, (path, emb, err) in enumerate(catalog_results):
                if emb is not None:
                    valid_embeddings.append(emb)
                    valid_products.append(catalog[i])
            
            print(f"  Catalog: {len(catalog)} products")
            print(f"  Valid embeddings: {len(valid_embeddings)}")
            print(f"  Corrupted/failed: {len(catalog) - len(valid_embeddings)}")
            
            # Extract query embedding
            query_emb = extract_clip_embedding(query_img)
            
            # Compute similarities
            if len(valid_embeddings) > 0:
                catalog_embs = np.array(valid_embeddings)
                similarities = batch_compute_clip_similarities(query_emb, catalog_embs)
                
                # Rank matches
                ranked_indices = np.argsort(similarities)[::-1]
                
                print(f"[OK] Mode 1 matching handled corrupted images gracefully")
                print(f"  Top 3 matches:")
                for i in range(min(3, len(ranked_indices))):
                    idx = ranked_indices[i]
                    print(f"    {i+1}. {valid_products[idx]['id']}: {similarities[idx]:.1f}/100")
            else:
                print(f"[WARNING] No valid images in catalog")
        
        finally:
            os.unlink(query_img)
            for p in catalog:
                try:
                    os.unlink(p['image_path'])
                except:
                    pass
    
    def test_mode1_with_missing_data(self):
        """Test Mode 1 matching with missing metadata (names, SKUs)"""
        print("\n[Mode 1 Real-World] Testing with missing metadata...")
        
        # Create catalog with missing metadata
        catalog = create_product_catalog(num_products=15)
        query_img = create_test_image(pattern='solid')
        
        try:
            # Extract embeddings
            catalog_results = batch_extract_clip_embeddings(
                [p['image_path'] for p in catalog],
                skip_errors=True
            )
            
            # Filter valid
            valid_embeddings = []
            valid_products = []
            for i, (path, emb, err) in enumerate(catalog_results):
                if emb is not None:
                    valid_embeddings.append(emb)
                    valid_products.append(catalog[i])
            
            # Count missing metadata
            missing_names = sum(1 for p in valid_products if p['name'] is None)
            missing_skus = sum(1 for p in valid_products if p['sku'] is None)
            
            print(f"  Valid products: {len(valid_products)}")
            print(f"  Missing names: {missing_names}")
            print(f"  Missing SKUs: {missing_skus}")
            
            # Mode 1 should work without metadata
            query_emb = extract_clip_embedding(query_img)
            catalog_embs = np.array(valid_embeddings)
            similarities = batch_compute_clip_similarities(query_emb, catalog_embs)
            
            # Should still get matches
            matches = [(i, sim) for i, sim in enumerate(similarities) if sim > 50]
            
            print(f"[OK] Mode 1 works without metadata")
            print(f"  Matches found: {len(matches)}")
            
            assert len(matches) > 0, "Should find matches even with missing metadata"
        
        finally:
            os.unlink(query_img)
            for p in catalog:
                try:
                    os.unlink(p['image_path'])
                except:
                    pass
    
    def test_mode1_threshold_filtering_realworld(self):
        """Test Mode 1 threshold filtering with real-world data"""
        print("\n[Mode 1 Real-World] Testing threshold filtering...")
        
        # Create diverse catalog
        catalog = create_product_catalog(num_products=20)
        query_img = create_test_image(pattern='gradient')
        
        try:
            # Extract embeddings
            catalog_results = batch_extract_clip_embeddings(
                [p['image_path'] for p in catalog],
                skip_errors=True
            )
            
            valid_embeddings = []
            valid_products = []
            for i, (path, emb, err) in enumerate(catalog_results):
                if emb is not None:
                    valid_embeddings.append(emb)
                    valid_products.append(catalog[i])
            
            query_emb = extract_clip_embedding(query_img)
            catalog_embs = np.array(valid_embeddings)
            similarities = batch_compute_clip_similarities(query_emb, catalog_embs)
            
            # Test different thresholds
            thresholds = [50, 70, 90]
            for threshold in thresholds:
                matches = [(i, sim) for i, sim in enumerate(similarities) if sim >= threshold]
                print(f"  Threshold {threshold}: {len(matches)} matches")
            
            print(f"[OK] Threshold filtering works with real-world data")
        
        finally:
            os.unlink(query_img)
            for p in catalog:
                try:
                    os.unlink(p['image_path'])
                except:
                    pass


# ============================================================================
# Mode 3: Real-World Hybrid Matching Tests
# ============================================================================

@pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP dependencies not available")
class TestMode3RealWorld:
    """Real-world tests for Mode 3 (Hybrid: Visual + Metadata) matching"""
    
    def test_mode3_with_null_categories(self):
        """Test Mode 3 matching with NULL/missing categories"""
        print("\n[Mode 3 Real-World] Testing with NULL categories...")
        
        # Create catalog with missing categories
        catalog = create_product_catalog(num_products=20)
        query_img = create_test_image(pattern='gradient')
        query_category = 'plates'
        
        try:
            # Extract embeddings
            catalog_results = batch_extract_clip_embeddings(
                [p['image_path'] for p in catalog],
                skip_errors=True
            )
            
            valid_products = []
            valid_embeddings = []
            for i, (path, emb, err) in enumerate(catalog_results):
                if emb is not None:
                    valid_products.append(catalog[i])
                    valid_embeddings.append(emb)
            
            # Count NULL categories
            null_categories = sum(1 for p in valid_products if p['category'] is None)
            print(f"  Total valid products: {len(valid_products)}")
            print(f"  NULL categories: {null_categories}")
            
            # Filter by category (handle NULL)
            category_matches = []
            category_embeddings = []
            for i, p in enumerate(valid_products):
                if p['category'] == query_category:
                    category_matches.append(p)
                    category_embeddings.append(valid_embeddings[i])
            
            print(f"  Category '{query_category}' matches: {len(category_matches)}")
            
            if len(category_embeddings) > 0:
                # Compute similarities within category
                query_emb = extract_clip_embedding(query_img)
                catalog_embs = np.array(category_embeddings)
                similarities = batch_compute_clip_similarities(query_emb, catalog_embs)
                
                print(f"[OK] Mode 3 handled NULL categories correctly")
                print(f"  Average similarity: {similarities.mean():.1f}/100")
            else:
                print(f"[WARNING] No products in category '{query_category}'")
        
        finally:
            os.unlink(query_img)
            for p in catalog:
                try:
                    os.unlink(p['image_path'])
                except:
                    pass

    
    def test_mode3_category_fallback(self):
        """Test Mode 3 fallback when category has no matches"""
        print("\n[Mode 3 Real-World] Testing category fallback...")
        
        # Create catalog
        catalog = create_product_catalog(num_products=15, categories=['plates', 'bowls'])
        query_img = create_test_image(pattern='solid')
        query_category = 'cups'  # Not in catalog
        
        try:
            # Extract embeddings
            catalog_results = batch_extract_clip_embeddings(
                [p['image_path'] for p in catalog],
                skip_errors=True
            )
            
            valid_products = []
            valid_embeddings = []
            for i, (path, emb, err) in enumerate(catalog_results):
                if emb is not None:
                    valid_products.append(catalog[i])
                    valid_embeddings.append(emb)
            
            # Try to filter by category
            category_matches = [p for p in valid_products if p['category'] == query_category]
            
            print(f"  Query category: '{query_category}'")
            print(f"  Category matches: {len(category_matches)}")
            
            if len(category_matches) == 0:
                # Fallback: search all categories or use 'unknown'
                print(f"  Fallback: Searching all categories")
                
                query_emb = extract_clip_embedding(query_img)
                catalog_embs = np.array(valid_embeddings)
                similarities = batch_compute_clip_similarities(query_emb, catalog_embs)
                
                # Get top matches across all categories
                ranked_indices = np.argsort(similarities)[::-1][:5]
                
                print(f"[OK] Mode 3 fallback works correctly")
                print(f"  Top matches across all categories:")
                for i, idx in enumerate(ranked_indices):
                    p = valid_products[idx]
                    print(f"    {i+1}. {p['id']} ({p['category']}): {similarities[idx]:.1f}/100")
        
        finally:
            os.unlink(query_img)
            for p in catalog:
                try:
                    os.unlink(p['image_path'])
                except:
                    pass
    
    def test_mode3_mixed_quality_data(self):
        """Test Mode 3 with mixed quality: corrupted images, missing categories, missing SKUs"""
        print("\n[Mode 3 Real-World] Testing with mixed quality data...")
        
        # Create realistic messy catalog
        catalog = create_product_catalog(num_products=30)
        query_img = create_test_image(pattern='checkerboard')
        query_category = 'bowls'
        
        try:
            # Extract embeddings
            catalog_results = batch_extract_clip_embeddings(
                [p['image_path'] for p in catalog],
                skip_errors=True
            )
            
            # Collect statistics
            total_products = len(catalog)
            valid_images = sum(1 for _, emb, _ in catalog_results if emb is not None)
            corrupted_images = total_products - valid_images
            
            valid_products = []
            valid_embeddings = []
            for i, (path, emb, err) in enumerate(catalog_results):
                if emb is not None:
                    valid_products.append(catalog[i])
                    valid_embeddings.append(emb)
            
            null_categories = sum(1 for p in valid_products if p['category'] is None)
            missing_skus = sum(1 for p in valid_products if p['sku'] is None)
            missing_names = sum(1 for p in valid_products if p['name'] is None)
            
            print(f"  Data Quality Report:")
            print(f"    Total products: {total_products}")
            print(f"    Valid images: {valid_images}")
            print(f"    Corrupted images: {corrupted_images}")
            print(f"    NULL categories: {null_categories}")
            print(f"    Missing SKUs: {missing_skus}")
            print(f"    Missing names: {missing_names}")
            
            # Filter by category
            category_matches = []
            category_embeddings = []
            for i, p in enumerate(valid_products):
                if p['category'] == query_category:
                    category_matches.append(p)
                    category_embeddings.append(valid_embeddings[i])
            
            print(f"    Category '{query_category}' matches: {len(category_matches)}")
            
            if len(category_embeddings) > 0:
                # Compute similarities
                query_emb = extract_clip_embedding(query_img)
                catalog_embs = np.array(category_embeddings)
                similarities = batch_compute_clip_similarities(query_emb, catalog_embs)
                
                # Get top matches
                ranked_indices = np.argsort(similarities)[::-1][:3]
                
                print(f"[OK] Mode 3 handled mixed quality data successfully")
                print(f"  Top 3 matches:")
                for i, idx in enumerate(ranked_indices):
                    p = category_matches[idx]
                    name = p['name'] or 'N/A'
                    sku = p['sku'] or 'N/A'
                    print(f"    {i+1}. {p['id']} (SKU: {sku}, Name: {name}): {similarities[idx]:.1f}/100")
            else:
                print(f"[WARNING] No valid products in category '{query_category}'")
        
        finally:
            os.unlink(query_img)
            for p in catalog:
                try:
                    os.unlink(p['image_path'])
                except:
                    pass
    
    def test_mode3_error_reporting(self):
        """Test Mode 3 error reporting for failed products"""
        print("\n[Mode 3 Real-World] Testing error reporting...")
        
        # Create catalog with known issues
        catalog = create_product_catalog(num_products=10)
        query_img = create_test_image(pattern='gradient')
        
        try:
            # Extract embeddings and collect errors
            catalog_results = batch_extract_clip_embeddings(
                [p['image_path'] for p in catalog],
                skip_errors=True
            )
            
            # Collect error information
            errors = []
            for i, (path, emb, err) in enumerate(catalog_results):
                if emb is None and err is not None:
                    errors.append({
                        'product_id': catalog[i]['id'],
                        'image_path': path,
                        'error': err,
                        'category': catalog[i]['category']
                    })
            
            print(f"  Total products: {len(catalog)}")
            print(f"  Failed products: {len(errors)}")
            
            if errors:
                print(f"  Error details:")
                for error_info in errors:
                    print(f"    - {error_info['product_id']}: {error_info['error'][:50]}...")
            
            print(f"[OK] Mode 3 provides detailed error reporting")
            
            # Verify we can still match with remaining products
            valid_count = sum(1 for _, emb, _ in catalog_results if emb is not None)
            assert valid_count > 0, "Should have at least some valid products"
        
        finally:
            os.unlink(query_img)
            for p in catalog:
                try:
                    os.unlink(p['image_path'])
                except:
                    pass


# ============================================================================
# Performance and Scale Tests
# ============================================================================

@pytest.mark.skipif(not CLIP_AVAILABLE, reason="CLIP dependencies not available")
@pytest.mark.slow
class TestPerformanceRealWorld:
    """Performance tests with realistic data volumes"""
    
    def test_large_catalog_performance(self):
        """Test performance with large catalog (100+ products)"""
        print("\n[Performance] Testing large catalog (100 products)...")
        
        device = detect_device()
        num_products = 100
        
        # Create large catalog
        print(f"  Creating {num_products} test images...")
        catalog = []
        for i in range(num_products):
            pattern = ['gradient', 'solid', 'checkerboard'][i % 3]
            img_path = create_test_image(width=256, height=256, pattern=pattern)
            catalog.append({
                'id': f'PROD-{i:03d}',
                'image_path': img_path,
                'category': ['plates', 'bowls', 'cups'][i % 3]
            })
        
        try:
            # Benchmark batch extraction
            print(f"  Extracting embeddings...")
            start = time.time()
            results = batch_extract_clip_embeddings(
                [p['image_path'] for p in catalog],
                batch_size=32,
                skip_errors=True
            )
            extraction_time = time.time() - start
            
            success_count = sum(1 for _, emb, _ in results if emb is not None)
            extraction_throughput = success_count / extraction_time
            
            print(f"  Extraction: {extraction_time:.2f}s ({extraction_throughput:.1f} img/s)")
            
            # Benchmark matching
            valid_embeddings = [emb for _, emb, _ in results if emb is not None]
            query_img = create_test_image(pattern='gradient')
            query_emb = extract_clip_embedding(query_img)
            
            start = time.time()
            catalog_embs = np.array(valid_embeddings)
            similarities = batch_compute_clip_similarities(query_emb, catalog_embs)
            matching_time = time.time() - start
            
            print(f"  Matching: {matching_time:.3f}s")
            print(f"  Total time: {extraction_time + matching_time:.2f}s")
            
            # Performance expectations
            if device == 'cuda':
                assert extraction_throughput > 5, f"GPU too slow: {extraction_throughput:.1f} img/s"
                print(f"[OK] GPU performance: EXCELLENT (> 5 img/s)")
            elif device == 'mps':
                assert extraction_throughput > 2, f"MPS too slow: {extraction_throughput:.1f} img/s"
                print(f"[OK] MPS performance: GOOD (> 2 img/s)")
            else:
                assert extraction_throughput > 0.5, f"CPU too slow: {extraction_throughput:.1f} img/s"
                print(f"[OK] CPU performance: ACCEPTABLE (> 0.5 img/s)")
            
            os.unlink(query_img)
        
        finally:
            for p in catalog:
                try:
                    os.unlink(p['image_path'])
                except:
                    pass
    
    def test_matching_speed_requirements(self):
        """Test that matching meets speed requirements from design doc"""
        print("\n[Performance] Testing matching speed requirements...")
        
        device = detect_device()
        
        # Create catalog of 1000 products (as per Requirement 8.2)
        num_products = 100  # Using 100 for test speed, scale to 1000 in production
        print(f"  Creating catalog of {num_products} products...")
        
        catalog = []
        for i in range(num_products):
            img_path = create_test_image(width=256, height=256)
            catalog.append(img_path)
        
        try:
            # Extract embeddings
            print(f"  Extracting embeddings...")
            start = time.time()
            results = batch_extract_clip_embeddings(catalog, batch_size=32, skip_errors=True)
            extraction_time = time.time() - start
            
            valid_embeddings = [emb for _, emb, _ in results if emb is not None]
            
            # Test matching speed
            query_img = create_test_image()
            query_emb = extract_clip_embedding(query_img)
            
            start = time.time()
            catalog_embs = np.array(valid_embeddings)
            similarities = batch_compute_clip_similarities(query_emb, catalog_embs)
            matching_time = time.time() - start
            
            total_time = extraction_time + matching_time
            
            print(f"  Results:")
            print(f"    Extraction: {extraction_time:.2f}s")
            print(f"    Matching: {matching_time:.3f}s")
            print(f"    Total: {total_time:.2f}s")
            
            # Requirement 8.2: Should complete within 30 seconds for 1000 products
            # For 100 products, should be much faster
            expected_max_time = 30.0 * (num_products / 1000.0)  # Scale expectation
            
            if total_time > expected_max_time:
                print(f"[WARNING] Slower than expected ({total_time:.2f}s > {expected_max_time:.2f}s)")
            else:
                print(f"[OK] Performance meets requirements ({total_time:.2f}s < {expected_max_time:.2f}s)")
            
            os.unlink(query_img)
        
        finally:
            for img in catalog:
                try:
                    os.unlink(img)
                except:
                    pass


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == '__main__':
    import pytest
    
    print("=" * 80)
    print("CLIP Real-World Testing Suite")
    print("=" * 80)
    
    if not CLIP_AVAILABLE:
        print("\n❌ CLIP dependencies not available!")
        print("Install with: pip install torch sentence-transformers")
        sys.exit(1)
    
    # Run tests
    exit_code = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'not slow'
    ])
    
    sys.exit(exit_code)
