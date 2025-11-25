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


# ============ Catalog Management Validation ============

import re

def validate_category(category, max_length=100):
    """Validate category string
    
    Args:
        category: Category string to validate
        max_length: Maximum allowed length
    
    Returns:
        Tuple of (validated_category, error_message)
        - (category, None) if valid
        - (None, error_message) if invalid
    """
    if category is None:
        return None, None  # NULL is valid
    
    if not isinstance(category, str):
        return None, f'category must be a string, got {type(category).__name__}'
    
    category = category.strip()
    
    if category == '':
        return None, None  # Empty string becomes NULL
    
    if len(category) > max_length:
        return None, f'category too long (max {max_length} characters)'
    
    # Check for invalid characters (allow alphanumeric, spaces, hyphens, underscores)
    if not re.match(r'^[\w\s\-]+$', category, re.UNICODE):
        return None, 'category contains invalid characters (use letters, numbers, spaces, hyphens, underscores)'
    
    return category, None


def validate_product_name(name, max_length=200):
    """Validate product name string
    
    Args:
        name: Product name to validate
        max_length: Maximum allowed length
    
    Returns:
        Tuple of (validated_name, error_message)
    """
    if name is None:
        return None, None  # NULL is valid
    
    if not isinstance(name, str):
        return None, f'product_name must be a string, got {type(name).__name__}'
    
    name = name.strip()
    
    if name == '':
        return None, None  # Empty string becomes NULL
    
    if len(name) > max_length:
        return None, f'product_name too long (max {max_length} characters)'
    
    # Basic XSS prevention - strip HTML tags
    if '<' in name or '>' in name:
        name = re.sub(r'<[^>]*>', '', name)
    
    return name, None


def validate_sku(sku, max_length=50):
    """Validate SKU string
    
    Args:
        sku: SKU to validate
        max_length: Maximum allowed length
    
    Returns:
        Tuple of (validated_sku, error_message)
    """
    if sku is None:
        return None, None  # NULL is valid
    
    if not isinstance(sku, str):
        return None, f'sku must be a string, got {type(sku).__name__}'
    
    sku = sku.strip().upper()  # Normalize to uppercase
    
    if sku == '':
        return None, None  # Empty string becomes NULL
    
    if len(sku) > max_length:
        return None, f'sku too long (max {max_length} characters)'
    
    # SKU format: alphanumeric, hyphens, underscores only
    if not re.match(r'^[A-Z0-9\-_]+$', sku):
        return None, 'sku must contain only letters, numbers, hyphens, and underscores'
    
    return sku, None


def validate_cleanup_type(cleanup_type):
    """Validate cleanup type parameter
    
    Args:
        cleanup_type: Type of cleanup operation
    
    Returns:
        Tuple of (validated_type, error_message)
    """
    valid_types = ['all', 'historical', 'new', 'matches']
    
    if cleanup_type not in valid_types:
        return None, f'type must be one of: {", ".join(valid_types)}'
    
    return cleanup_type, None


def validate_days(days, min_days=1, max_days=3650):
    """Validate days parameter for date-based cleanup
    
    Args:
        days: Number of days
        min_days: Minimum allowed value
        max_days: Maximum allowed value (default 10 years)
    
    Returns:
        Tuple of (validated_days, error_message)
    """
    try:
        days = int(days)
        if days < min_days:
            return None, f'days must be at least {min_days}'
        if days > max_days:
            return None, f'days cannot exceed {max_days}'
        return days, None
    except (ValueError, TypeError):
        return None, f'days must be an integer, got {days}'


def validate_categories_list(categories):
    """Validate list of categories for bulk operations
    
    Args:
        categories: List of category strings
    
    Returns:
        Tuple of (validated_categories, error_message)
    """
    if not isinstance(categories, list):
        return None, f'categories must be an array, got {type(categories).__name__}'
    
    if len(categories) == 0:
        return None, 'categories array is empty'
    
    validated = []
    for cat in categories:
        if cat is None or cat == '':
            validated.append(None)  # NULL category
        elif isinstance(cat, str):
            validated.append(cat.strip())
        else:
            return None, f'invalid category value: {cat}'
    
    return validated, None


def validate_page_params(page, limit, max_limit=1000):
    """Validate pagination parameters
    
    Args:
        page: Page number (1-indexed)
        limit: Items per page
        max_limit: Maximum allowed limit
    
    Returns:
        Tuple of ((page, limit), error_message)
    """
    try:
        page = int(page)
        if page < 1:
            page = 1
    except (ValueError, TypeError):
        page = 1
    
    try:
        limit = int(limit)
        if limit < 1:
            limit = 50
        if limit > max_limit:
            limit = max_limit
    except (ValueError, TypeError):
        limit = 50
    
    return (page, limit), None


def sanitize_search_query(query, max_length=200):
    """Sanitize search query string
    
    Args:
        query: Search query string
        max_length: Maximum allowed length
    
    Returns:
        Sanitized query string
    """
    if query is None:
        return ''
    
    if not isinstance(query, str):
        return ''
    
    query = query.strip()
    
    if len(query) > max_length:
        query = query[:max_length]
    
    # Remove SQL injection attempts
    query = re.sub(r'[;\'"\\]', '', query)
    
    return query
