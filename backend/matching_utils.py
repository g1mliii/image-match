"""
Matching utilities - Extracted common patterns from product_matching.py
"""

import numpy as np
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

REQUIRED_FEATURES = ['color_features', 'shape_features', 'texture_features']


def validate_feature_dict(features: Optional[Dict], product_id: int, feature_name: str = "features") -> tuple[bool, Optional[str]]:
    """
    Validate a feature dictionary has all required features with valid data.
    
    Args:
        features: Feature dictionary to validate
        product_id: Product ID for error messages
        feature_name: Name for error messages (e.g., "query features", "candidate features")
    
    Returns:
        (is_valid, error_message) tuple
    """
    if not features:
        return False, f"Product {product_id} has no {feature_name}"
    
    # Check if all required feature types exist
    missing_types = [ft for ft in REQUIRED_FEATURES if ft not in features]
    if missing_types:
        return False, f"Product {product_id} missing feature types: {missing_types}"
    
    # Validate each feature array
    for feature_type in REQUIRED_FEATURES:
        feature_array = features[feature_type]
        
        # Check if feature exists and is valid
        if feature_array is None:
            return False, f"Product {product_id} has NULL {feature_type}"
        
        if not isinstance(feature_array, np.ndarray):
            return False, f"Product {product_id} {feature_type} is not numpy array: {type(feature_array)}"
        
        if feature_array.size == 0:
            return False, f"Product {product_id} {feature_type} is empty"
        
        # Check for NaN or Inf values
        if np.any(np.isnan(feature_array)):
            return False, f"Product {product_id} {feature_type} contains NaN values"
        
        if np.any(np.isinf(feature_array)):
            return False, f"Product {product_id} {feature_type} contains Inf values"
    
    return True, None


def validate_candidate_features_quick(features: Optional[Dict]) -> bool:
    """
    Quick validation for candidate features (returns bool only).
    Used in loops where we just need to skip invalid candidates.
    
    Supports both legacy features and CLIP embeddings:
    - Legacy: color_features, shape_features, texture_features (all non-empty)
    - CLIP: color_features (512-dim), shape_features (empty), texture_features (empty)
    
    Args:
        features: Feature dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not features:
        return False
    
    # Check if this is CLIP mode (embedding_type='clip')
    is_clip = features.get('embedding_type') == 'clip'
    
    if is_clip:
        # CLIP mode: only validate color_features (which contains the CLIP embedding)
        if 'color_features' not in features:
            return False
        
        color_features = features['color_features']
        if color_features is None or not isinstance(color_features, np.ndarray):
            return False
        
        # CLIP embeddings should be 512-dimensional
        if color_features.size != 512:
            return False
        
        if np.any(np.isnan(color_features)) or np.any(np.isinf(color_features)):
            return False
        
        return True
    else:
        # Legacy mode: validate all three feature types
        for feature_type in REQUIRED_FEATURES:
            if feature_type not in features:
                return False
            
            feature_array = features[feature_type]
            if feature_array is None or not isinstance(feature_array, np.ndarray) or feature_array.size == 0:
                return False
            
            if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                return False
        
        return True


def safe_get_metadata(product: Dict, field: str, default: Any = None) -> Any:
    """
    Safely extract metadata field from product dict (handles sqlite3.Row and dict).
    
    Args:
        product: Product dictionary or sqlite3.Row
        field: Field name to extract
        default: Default value if field missing or None
    
    Returns:
        Field value or default
    """
    try:
        value = product[field]
        return value if value is not None else default
    except (KeyError, TypeError):
        return default


def track_missing_metadata(product: Dict, data_quality_issues: Dict) -> list:
    """
    Track missing metadata fields and update data quality counters.
    
    Args:
        product: Product dictionary
        data_quality_issues: Data quality tracking dict to update
    
    Returns:
        List of missing field names
    """
    missing_fields = []
    
    if not safe_get_metadata(product, 'image_path'):
        missing_fields.append('image_path')
    
    if not safe_get_metadata(product, 'product_name'):
        missing_fields.append('product_name')
        data_quality_issues['missing_name'] = data_quality_issues.get('missing_name', 0) + 1
    
    if not safe_get_metadata(product, 'sku'):
        missing_fields.append('sku')
        data_quality_issues['missing_sku'] = data_quality_issues.get('missing_sku', 0) + 1
    
    if not safe_get_metadata(product, 'category'):
        data_quality_issues['missing_category'] = data_quality_issues.get('missing_category', 0) + 1
    
    if missing_fields:
        data_quality_issues['missing_metadata'] = data_quality_issues.get('missing_metadata', 0) + 1
    
    return missing_fields


def create_match_result(
    candidate_id: int,
    candidate_product: Dict,
    similarities: Dict,
    missing_fields: list = None
) -> Dict:
    """
    Create a standardized match result dictionary.
    
    Args:
        candidate_id: Candidate product ID
        candidate_product: Candidate product data
        similarities: Similarity scores dict
        missing_fields: List of missing metadata fields
    
    Returns:
        Match result dictionary
    """
    # Apply penalty for uncategorized matches (lower confidence)
    # Uncategorized products get a 10% penalty to signal lower confidence
    similarity_score = similarities['combined_similarity']
    category = safe_get_metadata(candidate_product, 'category')
    is_uncategorized = category is None or category == ''
    
    if is_uncategorized:
        # Apply 10% penalty (multiply by 0.90)
        similarity_score = similarity_score * 0.90
        # Ensure it doesn't go below 0
        similarity_score = max(0, similarity_score)
    
    return {
        'product_id': candidate_id,
        'image_path': safe_get_metadata(candidate_product, 'image_path', ''),
        'category': safe_get_metadata(candidate_product, 'category'),
        'product_name': safe_get_metadata(candidate_product, 'product_name'),
        'sku': safe_get_metadata(candidate_product, 'sku'),
        'similarity_score': similarity_score,
        'color_score': similarities['color_similarity'],
        'shape_score': similarities['shape_similarity'],
        'texture_score': similarities['texture_similarity'],
        'is_potential_duplicate': similarity_score > 90,
        'is_uncategorized': is_uncategorized,  # Flag for UI display
        'created_at': safe_get_metadata(candidate_product, 'created_at', ''),
        'has_missing_metadata': len(missing_fields) > 0 if missing_fields else False,
        'missing_fields': missing_fields if missing_fields else None
    }


def get_product_metadata(product_id: int, logger) -> tuple[Any, Any]:
    """
    Get price and performance metadata for a product.
    
    Args:
        product_id: Product ID
        logger: Logger instance
    
    Returns:
        (price, performance_dict) tuple
    """
    from database import get_price_history, get_performance_history
    
    # Get price
    price = None
    try:
        price_history = get_price_history(product_id, limit=1)
        if price_history:
            price = price_history[0]['price']
    except Exception as e:
        logger.warning(f"Could not get price for product {product_id}: {e}")
    
    # Get performance
    performance = None
    try:
        perf_history = get_performance_history(product_id, limit=1)
        if perf_history:
            performance = {
                'sales': perf_history[0]['sales'],
                'views': perf_history[0]['views'],
                'conversion_rate': perf_history[0]['conversion_rate'],
                'revenue': perf_history[0]['revenue']
            }
    except Exception as e:
        logger.warning(f"Could not get performance for product {product_id}: {e}")
    
    return price, performance


def batch_fetch_metadata(logger) -> tuple[Dict, Dict]:
    """
    Batch fetch price and performance data for all products.
    
    Args:
        logger: Logger instance
    
    Returns:
        (price_lookup, performance_lookup) tuple of dicts
    """
    from database import get_products_with_price_history, get_products_with_performance_history
    
    logger.info("Batch fetching metadata...")
    
    price_lookup = {}
    performance_lookup = {}
    
    try:
        products_with_prices = get_products_with_price_history()
        for p in products_with_prices:
            if p['product_id'] not in price_lookup:
                price_lookup[p['product_id']] = p['price']
    except Exception as e:
        logger.warning(f"Could not batch fetch prices: {e}")
    
    try:
        products_with_perf = get_products_with_performance_history()
        for p in products_with_perf:
            if p['product_id'] not in performance_lookup:
                performance_lookup[p['product_id']] = {
                    'sales': p['sales'],
                    'views': p['views'],
                    'conversion_rate': p['conversion_rate'],
                    'revenue': p['revenue']
                }
    except Exception as e:
        logger.warning(f"Could not batch fetch performance: {e}")
    
    logger.info(f"Cached {len(price_lookup)} prices, {len(performance_lookup)} performance records")
    
    return price_lookup, performance_lookup
