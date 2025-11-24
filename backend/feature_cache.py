"""
Feature caching module to avoid recomputation of image features.

This module provides caching functionality that:
1. Checks if features already exist in the database
2. Extracts features only if not cached
3. Stores features in the database for future use
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import os
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from database import (
    get_features_by_product_id,
    insert_features,
    update_features,
    get_product_by_id,
    get_all_features_by_category
)
from image_processing import (
    extract_all_features,
    ImageProcessingError
)

# Configure logging
logger = logging.getLogger(__name__)


class FeatureCache:
    """
    Feature cache manager that handles feature extraction and storage.
    """
    
    def __init__(self):
        """Initialize feature cache"""
        self.memory_cache = {}  # In-memory cache for frequently accessed features
        self.max_memory_cache_size = 100  # Maximum number of products to cache in memory
    
    def get_or_extract_features(self, product_id: int, image_path: str, 
                               force_recompute: bool = False) -> Dict[str, np.ndarray]:
        """
        Get features from cache or extract them if not cached.
        Uses unified feature extraction (includes CLIP if available).
        
        Args:
            product_id: Product ID
            image_path: Path to product image
            force_recompute: If True, recompute features even if cached
        
        Returns:
            Dictionary with 'color_features', 'shape_features', 'texture_features'
        
        Raises:
            ImageProcessingError: If feature extraction fails
        """
        # Check memory cache first (fastest)
        if not force_recompute and product_id in self.memory_cache:
            return self.memory_cache[product_id]
        
        # Check database cache (fast)
        if not force_recompute:
            db_features = get_features_by_product_id(product_id)
            if db_features:
                # Add to memory cache
                self._add_to_memory_cache(product_id, db_features)
                return db_features
        
        # Features not cached or force recompute - extract from image
        try:
            # Use unified feature extraction (includes CLIP if available)
            from feature_extraction_service import extract_features_unified
            
            features, embedding_type, embedding_version = extract_features_unified(image_path)
            
            # Store in database with embedding info
            if force_recompute:
                # Update existing features
                update_features(
                    product_id,
                    color_features=features['color_features'],
                    shape_features=features['shape_features'],
                    texture_features=features['texture_features'],
                    embedding_type=embedding_type,
                    embedding_version=embedding_version
                )
            else:
                # Insert new features
                insert_features(
                    product_id,
                    color_features=features['color_features'],
                    shape_features=features['shape_features'],
                    texture_features=features['texture_features'],
                    embedding_type=embedding_type,
                    embedding_version=embedding_version
                )
            
            # Add to memory cache
            self._add_to_memory_cache(product_id, features)
            
            return features
        
        except ImageProcessingError:
            raise
        except Exception as e:
            raise Exception(f"Failed to get or extract features for product {product_id}: {str(e)}")
    
    def _add_to_memory_cache(self, product_id: int, features: Dict[str, np.ndarray]):
        """
        Add features to in-memory cache with LRU eviction.
        
        Args:
            product_id: Product ID
            features: Feature dictionary
        """
        # If cache is full, remove oldest entry (simple FIFO for now)
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # Remove first item (oldest)
            first_key = next(iter(self.memory_cache))
            del self.memory_cache[first_key]
        
        self.memory_cache[product_id] = features
    
    def clear_memory_cache(self):
        """Clear in-memory cache"""
        self.memory_cache.clear()
    
    def remove_from_cache(self, product_id: int):
        """
        Remove product features from memory cache.
        
        Args:
            product_id: Product ID to remove
        """
        if product_id in self.memory_cache:
            del self.memory_cache[product_id]
    
    def preload_features(self, product_ids: List[int], max_workers: Optional[int] = None):
        """
        Preload features for multiple products into memory cache with parallel loading.
        
        Useful for batch operations where you know which products
        will be accessed. Uses multithreading for faster loading.
        
        Args:
            product_ids: List of product IDs to preload
            max_workers: Number of parallel workers (default: cpu_count + 4)
        """
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        # Filter out already cached products
        products_to_load = [pid for pid in product_ids if pid not in self.memory_cache]
        
        if not products_to_load:
            return
        
        logger.info(f"Preloading {len(products_to_load)} products into cache with {max_workers} workers")
        
        def load_single_product(product_id: int):
            try:
                features = get_features_by_product_id(product_id)
                if features:
                    return (product_id, features)
            except Exception as e:
                logger.warning(f"Failed to preload product {product_id}: {e}")
            return None
        
        # Load in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_single_product, pid) for pid in products_to_load]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    product_id, features = result
                    self._add_to_memory_cache(product_id, features)
        
        logger.info(f"Preloading complete: {len(self.memory_cache)} products in cache")
    
    def preload_catalog(self, category: Optional[str] = None, is_historical: bool = True, 
                       max_workers: Optional[int] = None):
        """
        Preload entire catalog (or category) into memory cache.
        
        Optimized for matching sessions where you'll be matching multiple products
        against the same catalog. Loads all features in parallel.
        
        Args:
            category: Category to preload (None = all categories)
            is_historical: Load historical products (default: True)
            max_workers: Number of parallel workers (default: cpu_count + 4)
        """
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        logger.info(f"Preloading catalog into cache (category: {category or 'all'}, workers: {max_workers})")
        
        # Get all features from database (already optimized with indexes)
        catalog_features = get_all_features_by_category(
            category=category,
            is_historical=is_historical,
            include_uncategorized=True
        )
        
        # Add to memory cache
        loaded_count = 0
        for product_id, features in catalog_features:
            if product_id not in self.memory_cache:
                self._add_to_memory_cache(product_id, features)
                loaded_count += 1
        
        logger.info(f"Catalog preloading complete: {loaded_count} new products loaded, {len(self.memory_cache)} total in cache")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'memory_cache_size': len(self.memory_cache),
            'max_memory_cache_size': self.max_memory_cache_size
        }


# Global feature cache instance
_global_cache = None


def get_feature_cache() -> FeatureCache:
    """
    Get global feature cache instance (singleton pattern).
    
    Returns:
        FeatureCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = FeatureCache()
    return _global_cache


def extract_and_cache_features(product_id: int, image_path: str, 
                               force_recompute: bool = False) -> Tuple[Dict[str, np.ndarray], bool]:
    """
    Extract features and cache them, or retrieve from cache.
    
    This is a convenience function that uses the global cache instance.
    
    Args:
        product_id: Product ID
        image_path: Path to product image
        force_recompute: If True, recompute features even if cached
    
    Returns:
        Tuple of (features_dict, was_cached)
        - features_dict: Dictionary with feature arrays
        - was_cached: True if features were retrieved from cache, False if newly extracted
    
    Raises:
        ImageProcessingError: If feature extraction fails
    """
    cache = get_feature_cache()
    
    # Check if features exist in database
    was_cached = False
    if not force_recompute:
        existing_features = get_features_by_product_id(product_id)
        if existing_features:
            was_cached = True
    
    features = cache.get_or_extract_features(product_id, image_path, force_recompute)
    
    return features, was_cached


def batch_extract_features(product_ids: List[int], max_workers: Optional[int] = None) -> Dict[int, Dict[str, any]]:
    """
    Extract features for multiple products in batch with parallel processing.
    
    This function processes products that don't have cached features yet.
    Uses multithreading for faster batch processing of images.
    
    Args:
        product_ids: List of product IDs to process
        max_workers: Maximum number of parallel workers (default: cpu_count + 4)
    
    Returns:
        Dictionary mapping product_id to result:
        - 'success': bool
        - 'features': feature dict (if successful)
        - 'error': error message (if failed)
        - 'error_code': error code (if failed)
    """
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    cache = get_feature_cache()
    results = {}
    
    logger.info(f"Batch extracting features for {len(product_ids)} products with {max_workers} workers")
    
    def process_single_product(product_id: int) -> Tuple[int, Dict[str, any]]:
        """Process a single product and return (product_id, result)"""
        try:
            # Get product info
            product = get_product_by_id(product_id)
            if not product:
                return (product_id, {
                    'success': False,
                    'error': f'Product {product_id} not found',
                    'error_code': 'PRODUCT_NOT_FOUND'
                })
            
            image_path = product['image_path']
            
            # Check if image file exists
            if not os.path.exists(image_path):
                return (product_id, {
                    'success': False,
                    'error': f'Image file not found: {image_path}',
                    'error_code': 'IMAGE_FILE_NOT_FOUND'
                })
            
            # Extract features
            features = cache.get_or_extract_features(product_id, image_path)
            
            return (product_id, {
                'success': True,
                'features': features
            })
        
        except ImageProcessingError as e:
            return (product_id, {
                'success': False,
                'error': e.message,
                'error_code': e.error_code,
                'suggestion': e.suggestion
            })
        except Exception as e:
            logger.error(f"Unexpected error processing product {product_id}: {e}")
            return (product_id, {
                'success': False,
                'error': str(e),
                'error_code': 'UNKNOWN_ERROR'
            })
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_product, pid) for pid in product_ids]
        
        for future in as_completed(futures):
            product_id, result = future.result()
            results[product_id] = result
    
    success_count = sum(1 for r in results.values() if r['success'])
    logger.info(f"Batch extraction complete: {success_count}/{len(product_ids)} successful")
    
    return results


def clear_all_caches():
    """Clear all caches (useful for testing or memory management)"""
    cache = get_feature_cache()
    cache.clear_memory_cache()
