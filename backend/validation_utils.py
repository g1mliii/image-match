"""
Validation utilities - Extracted common validation patterns from app.py
"""

def validate_threshold(threshold, default=0.0):
    """Validate threshold parameter (0-100)"""
    try:
        threshold = float(threshold)
        if not 0 <= threshold <= 100:
            return None, f'threshold must be between 0 and 100, got {threshold}'
        return threshold, None
    except (ValueError, TypeError):
        return None, f'threshold must be a number, got {threshold}'


def validate_limit(limit, default=10, max_limit=100):
    """Validate limit parameter"""
    try:
        limit = int(limit)
        if limit < 0:
            return None, f'limit must be non-negative, got {limit}'
        if limit > max_limit:
            limit = max_limit
        return limit, None
    except (ValueError, TypeError):
        return None, f'limit must be an integer, got {limit}'


def validate_weights(color_weight, shape_weight, texture_weight):
    """Validate similarity weights sum to 1.0"""
    try:
        color_weight = float(color_weight)
        shape_weight = float(shape_weight)
        texture_weight = float(texture_weight)
        
        total_weight = color_weight + shape_weight + texture_weight
        if not (0.99 <= total_weight <= 1.01):
            return None, f'Weights must sum to 1.0, got {total_weight:.3f}'
        
        if color_weight < 0 or shape_weight < 0 or texture_weight < 0:
            return None, 'All weights must be non-negative'
        
        return (color_weight, shape_weight, texture_weight), None
    except (ValueError, TypeError):
        return None, 'Weights must be numbers'


def validate_product_ids(product_ids, max_count=100):
    """Validate array of product IDs"""
    if not isinstance(product_ids, list):
        return None, f'product_ids must be an array, got {type(product_ids).__name__}'
    
    if len(product_ids) == 0:
        return None, 'product_ids array is empty'
    
    if len(product_ids) > max_count:
        return None, f'Too many products ({len(product_ids)}), max {max_count}'
    
    invalid_ids = []
    for pid in product_ids:
        try:
            int(pid)
        except (ValueError, TypeError):
            invalid_ids.append(pid)
    
    if invalid_ids:
        return None, f'Invalid product IDs: {invalid_ids}'
    
    return [int(pid) for pid in product_ids], None
