"""
Product Matching Service

This module implements the core matching logic with category filtering and
comprehensive real-world data handling. It handles:
- Category-based filtering (with NULL/missing category support)
- Graceful handling of missing or corrupted features
- Fallback logic for products without categories
- Ranking and threshold filtering
- Duplicate detection
- Edge case handling (empty catalog, all filtered out, etc.)
- Detailed error reporting and logging

Requirements: 3.1, 3.2, 4.1, 5.1, 5.2, 5.3
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import warnings
import logging
from datetime import datetime

from database import (
    get_product_by_id,
    get_features_by_product_id,
    get_all_features_by_category,
    insert_match,
    get_products_by_category
)
from similarity import (
    compute_all_similarities,
    batch_compute_similarities,
    InvalidFeatureError,
    FeatureDimensionError,
    SimilarityComputationError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchingError(Exception):
    """Base exception for matching errors"""
    def __init__(self, message: str, error_code: str, suggestion: str = None):
        self.message = message
        self.error_code = error_code
        self.suggestion = suggestion
        super().__init__(self.message)
    
    def to_dict(self):
        """Convert error to dictionary for API responses"""
        return {
            'error': self.message,
            'error_code': self.error_code,
            'suggestion': self.suggestion
        }


class ProductNotFoundError(MatchingError):
    """Raised when product doesn't exist"""
    def __init__(self, product_id: int):
        super().__init__(
            f"Product with ID {product_id} not found",
            'PRODUCT_NOT_FOUND',
            'Ensure the product ID is correct and the product exists in the database.'
        )


class MissingFeaturesError(MatchingError):
    """Raised when product doesn't have features extracted"""
    def __init__(self, product_id: int):
        super().__init__(
            f"Product {product_id} does not have features extracted",
            'MISSING_FEATURES',
            'Extract features from the product image before attempting to match.'
        )


class EmptyCatalogError(MatchingError):
    """Raised when no historical products exist for matching"""
    def __init__(self, category: Optional[str] = None):
        if category:
            message = f"No historical products found in category '{category}'"
            suggestion = f"Add historical products to the '{category}' category or try a different category."
        else:
            message = "No historical products found in the catalog"
            suggestion = "Add historical products to the catalog before attempting to match."
        
        super().__init__(message, 'EMPTY_CATALOG', suggestion)


class AllMatchesFailedError(MatchingError):
    """Raised when all similarity computations fail"""
    def __init__(self):
        super().__init__(
            "All similarity computations failed",
            'ALL_MATCHES_FAILED',
            'Check data quality of historical products. Features may be corrupted.'
        )


def normalize_category(category: Optional[str]) -> Optional[str]:
    """
    Normalize category string for consistent matching.
    
    Handles:
    - Case insensitivity (Placemats → placemats)
    - Whitespace trimming
    - Empty strings → None
    - "Unknown" variations → None
    
    Args:
        category: Category string (can be None)
    
    Returns:
        Normalized category or None
    """
    if category is None:
        return None
    
    # Trim whitespace and convert to lowercase for case-insensitive matching
    category = category.strip().lower()
    
    # Empty string becomes None
    if category == '':
        return None
    
    # Handle common variations of "unknown" or "uncategorized"
    if category in ['unknown', 'uncategorized', 'none', 'n/a', 'na']:
        return None
    
    return category


def fuzzy_match_category(input_category: str, available_categories: List[str], threshold: int = 2) -> Optional[str]:
    """
    Find the best matching category using fuzzy string matching.
    
    Handles common issues:
    - Misspellings: "placemat" → "placemats", "dinerware" → "dinnerware"
    - Capitalization: "PlaceMats" → "placemats"
    - Pluralization: "placemat" → "placemats"
    - Extra spaces: "place mats" → "placemats"
    
    Uses Levenshtein distance (edit distance) to find closest match.
    
    Args:
        input_category: Category to match (already normalized)
        available_categories: List of valid categories in database
        threshold: Maximum edit distance to consider a match (default: 2)
    
    Returns:
        Best matching category or None if no good match found
    
    Examples:
        >>> fuzzy_match_category("placemat", ["placemats", "dinnerware"])
        "placemats"
        >>> fuzzy_match_category("dinerware", ["placemats", "dinnerware"])
        "dinnerware"
        >>> fuzzy_match_category("xyz", ["placemats", "dinnerware"], threshold=2)
        None
    """
    if not input_category or not available_categories:
        return None
    
    # Normalize input
    input_normalized = input_category.lower().strip()
    
    # Remove spaces and hyphens for comparison (handles "place mats" vs "placemats")
    input_compact = input_normalized.replace(' ', '').replace('-', '')
    
    best_match = None
    best_distance = float('inf')
    
    for category in available_categories:
        category_normalized = category.lower().strip()
        category_compact = category_normalized.replace(' ', '').replace('-', '')
        
        # Exact match (after normalization)
        if input_compact == category_compact:
            return category
        
        # Calculate Levenshtein distance
        distance = levenshtein_distance(input_compact, category_compact)
        
        if distance < best_distance:
            best_distance = distance
            best_match = category
    
    # Only return match if within threshold
    if best_distance <= threshold:
        logger.info(f"Fuzzy matched '{input_category}' to '{best_match}' (distance: {best_distance})")
        return best_match
    
    return None


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance (edit distance) between two strings.
    
    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string
    into another.
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Edit distance (0 = identical, higher = more different)
    
    Examples:
        >>> levenshtein_distance("placemat", "placemats")
        1  # One insertion
        >>> levenshtein_distance("dinerware", "dinnerware")
        1  # One insertion
        >>> levenshtein_distance("cat", "dog")
        3  # Three substitutions
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def find_matches(
    product_id: int,
    threshold: float = 0.0,
    limit: int = 10,
    match_against_all: bool = False,
    include_uncategorized: bool = True,
    color_weight: float = 0.5,
    shape_weight: float = 0.3,
    texture_weight: float = 0.2,
    store_matches: bool = True,
    skip_invalid_products: bool = True
) -> Dict[str, Any]:
    """
    Find similar products in the historical catalog with comprehensive error handling.
    
    This is the main matching function that implements all requirements with robust
    real-world data handling:
    - Category filtering with NULL handling
    - Graceful handling of corrupted/missing features
    - Handles unopenable images and corrupted data
    - Handles missing fields and wrong formatting
    - Fallback logic for products without category
    - Ranking by similarity score
    - Threshold filtering
    - Result limiting
    - Duplicate detection (score > 90)
    - Edge case handling
    - Detailed error reporting and logging
    
    Args:
        product_id: ID of the new product to match
        threshold: Minimum similarity score (0-100) to include in results
        limit: Maximum number of matches to return
        match_against_all: If True, match against all categories (ignores product category)
        include_uncategorized: If True, include products with NULL category in matching
        color_weight: Weight for color similarity (default: 0.5)
        shape_weight: Weight for shape similarity (default: 0.3)
        texture_weight: Weight for texture similarity (default: 0.2)
        store_matches: If True, store match results in database
        skip_invalid_products: If True, skip products with data issues and continue
    
    Returns:
        Dictionary with:
        - 'matches': List of match results
        - 'total_candidates': Total number of products considered
        - 'successful_matches': Number of successful similarity computations
        - 'failed_matches': Number of failed similarity computations
        - 'filtered_by_threshold': Number of matches filtered out by threshold
        - 'warnings': List of warning messages
        - 'errors': List of error details for failed matches
        - 'data_quality_issues': Summary of data quality problems encountered
    
    Raises:
        ProductNotFoundError: If product doesn't exist
        MissingFeaturesError: If product doesn't have features
        EmptyCatalogError: If no historical products exist
        AllMatchesFailedError: If all similarity computations fail
    
    Requirements: 3.1, 3.2, 4.1, 5.1, 5.2, 5.3
    """
    warnings_list = []
    errors_list = []
    data_quality_issues = {
        'missing_features': 0,
        'corrupted_features': 0,
        'missing_metadata': 0,
        'invalid_categories': 0,
        'computation_errors': 0
    }
    
    # Step 1: Validate product exists
    logger.info(f"Finding matches for product {product_id}")
    product = get_product_by_id(product_id)
    
    if not product:
        logger.error(f"Product {product_id} not found")
        raise ProductNotFoundError(product_id)
    
    # Step 2: Get product features with comprehensive validation
    try:
        query_features = get_features_by_product_id(product_id)
    except Exception as e:
        logger.error(f"Database error retrieving features for product {product_id}: {e}")
        raise MissingFeaturesError(product_id)
    
    if not query_features:
        logger.error(f"Product {product_id} has no features extracted")
        raise MissingFeaturesError(product_id)
    
    # Validate query features with detailed error reporting
    try:
        # Check if all required feature types exist
        required_features = ['color_features', 'shape_features', 'texture_features']
        missing_types = [ft for ft in required_features if ft not in query_features]
        
        if missing_types:
            logger.error(f"Product {product_id} missing feature types: {missing_types}")
            raise MissingFeaturesError(product_id)
        
        # Validate each feature array
        for feature_type in required_features:
            features = query_features[feature_type]
            
            # Check if feature exists and is valid
            if features is None:
                logger.error(f"Product {product_id} has NULL {feature_type}")
                raise MissingFeaturesError(product_id)
            
            if not isinstance(features, np.ndarray):
                logger.error(f"Product {product_id} {feature_type} is not numpy array: {type(features)}")
                raise MissingFeaturesError(product_id)
            
            if features.size == 0:
                logger.error(f"Product {product_id} {feature_type} is empty")
                raise MissingFeaturesError(product_id)
            
            # Check for NaN or Inf values
            if np.any(np.isnan(features)):
                logger.error(f"Product {product_id} {feature_type} contains NaN values")
                raise MissingFeaturesError(product_id)
            
            if np.any(np.isinf(features)):
                logger.error(f"Product {product_id} {feature_type} contains Inf values")
                raise MissingFeaturesError(product_id)
        
        logger.info(f"Product {product_id} features validated successfully")
        
    except MissingFeaturesError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error validating features for product {product_id}: {e}")
        raise MissingFeaturesError(product_id)
    
    # Step 3: Determine category for filtering with fuzzy matching
    from database import get_all_categories
    
    product_category = product['category']
    normalized_category = normalize_category(product_category)
    
    # Try fuzzy matching if category doesn't exist exactly
    if normalized_category is not None and not match_against_all:
        available_categories = get_all_categories()
        
        # Check if category exists exactly (case-insensitive)
        category_exists = any(cat.lower() == normalized_category.lower() for cat in available_categories)
        
        if not category_exists and available_categories:
            # Try fuzzy matching for misspellings
            fuzzy_match = fuzzy_match_category(normalized_category, available_categories, threshold=2)
            
            if fuzzy_match:
                warnings_list.append(
                    f"Category '{product_category}' not found. Using similar category '{fuzzy_match}' instead."
                )
                logger.info(f"Fuzzy matched category '{normalized_category}' to '{fuzzy_match}'")
                normalized_category = normalize_category(fuzzy_match)
            else:
                warnings_list.append(
                    f"Category '{product_category}' not found in catalog. Available categories: {', '.join(available_categories[:5])}{'...' if len(available_categories) > 5 else ''}"
                )
                logger.warning(f"No fuzzy match found for category '{normalized_category}'")
    
    if normalized_category is None:
        if product_category is not None:
            warnings_list.append(f"Product category '{product_category}' normalized to NULL")
        
        logger.warning(f"Product {product_id} has no valid category")
        
        if not match_against_all:
            warnings_list.append(
                "Product has no category. Matching against all historical products."
            )
            match_against_all = True
    
    # Step 4: Get candidate products from historical catalog
    # PERFORMANCE OPTIMIZATION (Task 14): Category filtering happens at database level
    # Uses composite index (category, is_historical) for efficient retrieval
    # This ensures we only load features for products in the target category,
    # avoiding unnecessary similarity computations on irrelevant products
    logger.info(f"Fetching historical products for matching (category: {normalized_category}, match_all: {match_against_all})")
    
    if match_against_all:
        # Match against all historical products regardless of category
        candidate_features = get_all_features_by_category(
            category=None,
            is_historical=True,
            include_uncategorized=True
        )
    else:
        # Match only within same category (OPTIMIZED: filters at DB level)
        candidate_features = get_all_features_by_category(
            category=normalized_category,
            is_historical=True,
            include_uncategorized=include_uncategorized
        )
    
    # Step 5: Handle empty catalog
    if not candidate_features:
        logger.warning(f"No historical products found for matching")
        raise EmptyCatalogError(normalized_category if not match_against_all else None)
    
    logger.info(f"Found {len(candidate_features)} candidate products")
    
    # Step 6: Compute similarities with comprehensive error handling
    matches = []
    successful_count = 0
    failed_count = 0
    
    for candidate_id, candidate_feature_dict in candidate_features:
        # Skip matching against self
        if candidate_id == product_id:
            continue
        
        try:
            # Validate candidate features exist
            if not candidate_feature_dict:
                logger.warning(f"Product {candidate_id} has no features, skipping")
                warnings_list.append(f"Product {candidate_id} has no features")
                failed_count += 1
                data_quality_issues['missing_features'] += 1
                errors_list.append({
                    'product_id': candidate_id,
                    'error': 'Missing features',
                    'error_code': 'MISSING_FEATURES'
                })
                if not skip_invalid_products:
                    raise MissingFeaturesError(candidate_id)
                continue
            
            # Validate candidate feature arrays
            required_features = ['color_features', 'shape_features', 'texture_features']
            has_invalid_features = False
            
            for feature_type in required_features:
                if feature_type not in candidate_feature_dict:
                    logger.warning(f"Product {candidate_id} missing {feature_type}")
                    has_invalid_features = True
                    break
                
                features = candidate_feature_dict[feature_type]
                
                # Check for None, empty, or invalid arrays
                if features is None or not isinstance(features, np.ndarray) or features.size == 0:
                    logger.warning(f"Product {candidate_id} has invalid {feature_type}")
                    has_invalid_features = True
                    break
                
                # Check for NaN or Inf
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    logger.warning(f"Product {candidate_id} has corrupted {feature_type} (NaN/Inf)")
                    has_invalid_features = True
                    break
            
            if has_invalid_features:
                warnings_list.append(f"Product {candidate_id} has corrupted features")
                failed_count += 1
                data_quality_issues['corrupted_features'] += 1
                errors_list.append({
                    'product_id': candidate_id,
                    'error': 'Corrupted or invalid features',
                    'error_code': 'CORRUPTED_FEATURES'
                })
                if not skip_invalid_products:
                    raise InvalidFeatureError(f"Product {candidate_id} has corrupted features")
                continue
            
            # Compute similarity with error handling
            try:
                similarities = compute_all_similarities(
                    query_features,
                    candidate_feature_dict,
                    color_weight=color_weight,
                    shape_weight=shape_weight,
                    texture_weight=texture_weight
                )
            except (InvalidFeatureError, FeatureDimensionError) as e:
                logger.warning(f"Similarity computation failed for product {candidate_id}: {e.message}")
                warnings_list.append(f"Product {candidate_id}: {e.message}")
                failed_count += 1
                data_quality_issues['computation_errors'] += 1
                errors_list.append({
                    'product_id': candidate_id,
                    'error': e.message,
                    'error_code': e.error_code,
                    'suggestion': e.suggestion
                })
                if not skip_invalid_products:
                    raise
                continue
            
            # Get candidate product details with error handling
            try:
                candidate_product = get_product_by_id(candidate_id)
            except Exception as e:
                logger.error(f"Database error retrieving product {candidate_id}: {e}")
                warnings_list.append(f"Product {candidate_id}: Database error")
                failed_count += 1
                errors_list.append({
                    'product_id': candidate_id,
                    'error': f'Database error: {str(e)}',
                    'error_code': 'DATABASE_ERROR'
                })
                if not skip_invalid_products:
                    raise
                continue
            
            if not candidate_product:
                logger.warning(f"Product {candidate_id} not found in database, skipping")
                warnings_list.append(f"Product {candidate_id} not found in database")
                failed_count += 1
                errors_list.append({
                    'product_id': candidate_id,
                    'error': 'Product not found',
                    'error_code': 'PRODUCT_NOT_FOUND'
                })
                if not skip_invalid_products:
                    raise ProductNotFoundError(candidate_id)
                continue
            
            # Handle missing metadata gracefully
            # sqlite3.Row objects support dict-like access with [] but not .get()
            missing_fields = []
            try:
                image_path = candidate_product['image_path']
                if not image_path:
                    missing_fields.append('image_path')
            except (KeyError, TypeError):
                missing_fields.append('image_path')
                image_path = ''
            
            try:
                product_name = candidate_product['product_name']
                if not product_name:
                    missing_fields.append('product_name')
            except (KeyError, TypeError):
                missing_fields.append('product_name')
                product_name = None
            
            try:
                sku = candidate_product['sku']
                if not sku:
                    missing_fields.append('sku')
            except (KeyError, TypeError):
                missing_fields.append('sku')
                sku = None
            
            try:
                category = candidate_product['category']
            except (KeyError, TypeError):
                category = None
            
            try:
                created_at = candidate_product['created_at']
            except (KeyError, TypeError):
                created_at = ''
            
            if missing_fields:
                logger.info(f"Product {candidate_id} missing metadata: {missing_fields}")
                data_quality_issues['missing_metadata'] += 1
            
            # Create match result with safe field access
            match_result = {
                'product_id': candidate_id,
                'image_path': image_path,
                'category': category,
                'product_name': product_name,
                'sku': sku,
                'similarity_score': similarities['combined_similarity'],
                'color_score': similarities['color_similarity'],
                'shape_score': similarities['shape_similarity'],
                'texture_score': similarities['texture_similarity'],
                'is_potential_duplicate': similarities['combined_similarity'] > 90,
                'created_at': created_at,
                'has_missing_metadata': len(missing_fields) > 0,
                'missing_fields': missing_fields if missing_fields else None
            }
            
            matches.append(match_result)
            successful_count += 1
            
        except (InvalidFeatureError, FeatureDimensionError, ProductNotFoundError, MissingFeaturesError) as e:
            # These are already logged above, just re-raise if not skipping
            if not skip_invalid_products:
                raise
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error processing product {candidate_id}: {e}", exc_info=True)
            warnings_list.append(f"Product {candidate_id}: Unexpected error - {str(e)}")
            failed_count += 1
            data_quality_issues['computation_errors'] += 1
            errors_list.append({
                'product_id': candidate_id,
                'error': str(e),
                'error_code': 'UNKNOWN_ERROR',
                'suggestion': 'Check product data integrity and try re-extracting features'
            })
            if not skip_invalid_products:
                raise
    
    # Step 7: Check if we have any successful matches
    if successful_count == 0:
        logger.error("All similarity computations failed")
        raise AllMatchesFailedError()
    
    # Log data quality issues
    if failed_count > 0:
        logger.warning(f"Data quality issues: {failed_count} products had corrupted or missing features")
    
    # Step 8: Sort matches by similarity score (descending)
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Step 9: Apply threshold filtering
    filtered_count = 0
    if threshold > 0:
        original_count = len(matches)
        matches = [m for m in matches if m['similarity_score'] >= threshold]
        filtered_count = original_count - len(matches)
        
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} matches below threshold {threshold}")
    
    # Step 10: Apply result limit
    if limit > 0 and len(matches) > limit:
        matches = matches[:limit]
        logger.info(f"Limited results to top {limit} matches")
    
    # Step 11: Store matches in database (optional)
    if store_matches and matches:
        try:
            for match in matches:
                insert_match(
                    new_product_id=product_id,
                    matched_product_id=match['product_id'],
                    similarity_score=match['similarity_score'],
                    color_score=match['color_score'],
                    shape_score=match['shape_score'],
                    texture_score=match['texture_score']
                )
            logger.info(f"Stored {len(matches)} matches in database")
        except Exception as e:
            logger.error(f"Failed to store matches in database: {e}")
            warnings_list.append(f"Failed to store matches: {str(e)}")
    
    # Step 12: Prepare comprehensive response with data quality information
    result = {
        'matches': matches,
        'total_candidates': len(candidate_features),
        'successful_matches': successful_count,
        'failed_matches': failed_count,
        'filtered_by_threshold': filtered_count,
        'threshold_used': threshold,
        'limit_used': limit,
        'category_filter': normalized_category if not match_against_all else None,
        'matched_against_all_categories': match_against_all,
        'warnings': warnings_list,
        'errors': errors_list if errors_list else None,
        'data_quality_issues': data_quality_issues,
        'data_quality_summary': {
            'total_issues': sum(data_quality_issues.values()),
            'success_rate': round(successful_count / len(candidate_features) * 100, 1) if candidate_features else 0,
            'has_data_quality_issues': sum(data_quality_issues.values()) > 0
        }
    }
    
    # Log summary
    logger.info(
        f"Matching complete: {len(matches)} matches returned, "
        f"{successful_count} successful, {failed_count} failed, "
        f"{sum(data_quality_issues.values())} data quality issues"
    )
    
    if data_quality_issues['missing_features'] > 0:
        logger.warning(f"Data quality: {data_quality_issues['missing_features']} products missing features")
    if data_quality_issues['corrupted_features'] > 0:
        logger.warning(f"Data quality: {data_quality_issues['corrupted_features']} products with corrupted features")
    if data_quality_issues['missing_metadata'] > 0:
        logger.info(f"Data quality: {data_quality_issues['missing_metadata']} products missing metadata")
    
    return result


def batch_find_matches(
    product_ids: List[int],
    threshold: float = 0.0,
    limit: int = 10,
    match_against_all: bool = False,
    include_uncategorized: bool = True,
    color_weight: float = 0.5,
    shape_weight: float = 0.3,
    texture_weight: float = 0.2,
    store_matches: bool = True,
    continue_on_error: bool = True,
    skip_invalid_products: bool = True
) -> Dict[str, Any]:
    """
    Find matches for multiple products in batch.
    
    This function processes multiple products and isolates errors so that
    one failure doesn't stop the entire batch. Handles real-world data issues
    gracefully across all products.
    
    Args:
        product_ids: List of product IDs to match
        threshold: Minimum similarity score (0-100)
        limit: Maximum matches per product
        match_against_all: Match against all categories
        include_uncategorized: Include NULL category products
        color_weight: Weight for color similarity
        shape_weight: Weight for shape similarity
        texture_weight: Weight for texture similarity
        store_matches: Store results in database
        continue_on_error: If True, continue processing on errors
        skip_invalid_products: If True, skip products with data issues
    
    Returns:
        Dictionary with:
        - 'results': List of match results per product
        - 'summary': Summary statistics including data quality metrics
        - 'errors': List of products that failed
    
    Requirements: 6.1, 6.2, 6.3
    """
    logger.info(f"Batch matching {len(product_ids)} products")
    
    results = []
    errors = []
    successful = 0
    failed = 0
    
    for product_id in product_ids:
        try:
            match_result = find_matches(
                product_id=product_id,
                threshold=threshold,
                limit=limit,
                match_against_all=match_against_all,
                include_uncategorized=include_uncategorized,
                color_weight=color_weight,
                shape_weight=shape_weight,
                texture_weight=texture_weight,
                store_matches=store_matches,
                skip_invalid_products=skip_invalid_products
            )
            
            results.append({
                'product_id': product_id,
                'status': 'success',
                'match_count': len(match_result['matches']),
                'matches': match_result['matches'],
                'warnings': match_result['warnings'],
                'data_quality_issues': match_result.get('data_quality_issues', {}),
                'data_quality_summary': match_result.get('data_quality_summary', {})
            })
            
            successful += 1
            
        except (ProductNotFoundError, MissingFeaturesError, EmptyCatalogError, AllMatchesFailedError) as e:
            logger.error(f"Failed to match product {product_id}: {e.message}")
            
            error_info = {
                'product_id': product_id,
                'status': 'failed',
                'error': e.message,
                'error_code': e.error_code,
                'suggestion': e.suggestion
            }
            
            errors.append(error_info)
            results.append(error_info)
            failed += 1
            
            if not continue_on_error:
                break
                
        except Exception as e:
            logger.error(f"Unexpected error matching product {product_id}: {e}")
            
            error_info = {
                'product_id': product_id,
                'status': 'failed',
                'error': str(e),
                'error_code': 'UNKNOWN_ERROR'
            }
            
            errors.append(error_info)
            results.append(error_info)
            failed += 1
            
            if not continue_on_error:
                break
    
    summary = {
        'total_products': len(product_ids),
        'successful': successful,
        'failed': failed,
        'success_rate': round(successful / len(product_ids) * 100, 1) if product_ids else 0
    }
    
    logger.info(f"Batch matching complete: {successful} successful, {failed} failed")
    
    return {
        'results': results,
        'summary': summary,
        'errors': errors if errors else None
    }


def get_match_statistics(product_id: int) -> Dict[str, Any]:
    """
    Get statistics about matches for a product.
    
    Args:
        product_id: Product ID
    
    Returns:
        Dictionary with match statistics
    """
    from database import get_matches_for_product
    
    matches = get_matches_for_product(product_id, limit=1000)
    
    if not matches:
        return {
            'product_id': product_id,
            'total_matches': 0,
            'has_matches': False
        }
    
    scores = [m['similarity_score'] for m in matches]
    
    return {
        'product_id': product_id,
        'total_matches': len(matches),
        'has_matches': True,
        'highest_score': max(scores),
        'lowest_score': min(scores),
        'average_score': sum(scores) / len(scores),
        'potential_duplicates': len([s for s in scores if s > 90]),
        'high_similarity': len([s for s in scores if s > 70]),
        'medium_similarity': len([s for s in scores if 50 <= s <= 70]),
        'low_similarity': len([s for s in scores if s < 50])
    }
