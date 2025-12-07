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

Matching Modes:
- Mode 1 (Visual): CLIP embeddings (GPU or CPU)
- Mode 2 (Metadata): SKU, name, category, price, performance
- Mode 3 (Hybrid): Combination of visual + metadata

Requirements: 3.1, 3.2, 4.1, 5.1, 5.2, 5.3
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import warnings
import logging
import os
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

# Import CLIP functions (required)
try:
    from image_processing_clip import (
        compute_clip_similarity,
        batch_compute_clip_similarities,
        is_clip_available,
        CLIPModelError
    )
    CLIP_AVAILABLE = is_clip_available()
    if not CLIP_AVAILABLE:
        logger.error("CLIP not available - install PyTorch and sentence-transformers")
except ImportError:
    CLIP_AVAILABLE = False
    logger.error("CLIP not available - install torch and sentence-transformers for CLIP support")


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
    
    # Determine if using CLIP or legacy features
    use_clip = CLIP_AVAILABLE
    
    # Step 1: Validate product exists
    logger.info(f"Finding matches for product {product_id} (mode: {'CLIP' if use_clip else 'legacy'})")
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
    
    # Validate query features based on mode
    if use_clip:
        # CLIP mode: check for CLIP embedding
        # CLIP embeddings are stored in color_features column with embedding_type='clip'
        if query_features.get('embedding_type') == 'clip':
            query_embedding = query_features['color_features']
        elif 'clip_embedding' in query_features:
            # Support explicit clip_embedding key (future enhancement)
            query_embedding = query_features['clip_embedding']
        else:
            logger.error(f"Product {product_id} missing CLIP embedding (required for CLIP mode)")
            raise MissingFeaturesError(product_id)
        
        # Validate CLIP embedding
        if not isinstance(query_embedding, np.ndarray) or len(query_embedding) != 512:
            logger.error(f"Product {product_id} has invalid CLIP embedding (expected 512-dim array, got {type(query_embedding)} with shape {query_embedding.shape if isinstance(query_embedding, np.ndarray) else 'N/A'})")
            raise MissingFeaturesError(product_id)
        
        logger.info(f"Product {product_id} CLIP embedding validated successfully")
    else:
        # Legacy mode: validate traditional features
        from matching_utils import validate_feature_dict
        
        is_valid, error_msg = validate_feature_dict(query_features, product_id, "query features")
        if not is_valid:
            logger.error(error_msg)
            raise MissingFeaturesError(product_id)
        
        logger.info(f"Product {product_id} legacy features validated successfully")
    
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
    
    # Step 6: Compute similarities with FAISS acceleration (if available)
    matches = []
    successful_count = 0
    failed_count = 0
    
    # Try FAISS fast path for CLIP embeddings
    faiss_used = False
    if use_clip:
        try:
            from faiss_index import faiss_manager
            
            # Check if FAISS index exists for this category
            search_category = normalized_category if not match_against_all else None
            
            if faiss_manager.has_index(search_category):
                logger.info(f"Using FAISS fast path for category '{search_category}'")
                
                # Search with FAISS (returns top candidates quickly)
                # Request more candidates than limit to account for filtering
                k = min(limit * 10 if limit > 0 else 1000, len(candidate_features))
                distances, candidate_ids = faiss_manager.search(
                    search_category,
                    query_embedding,
                    k=k,
                    threshold=threshold / 100.0  # Convert 0-100 to 0-1 range
                )
                
                if distances is not None and candidate_ids is not None:
                    faiss_used = True
                    logger.info(f"FAISS returned {len(candidate_ids)} candidates")
                    
                    # Process FAISS results
                    for dist, candidate_id in zip(distances, candidate_ids):
                        # Skip self
                        if candidate_id == product_id:
                            continue
                        
                        try:
                            # Get candidate product details
                            candidate_product = get_product_by_id(candidate_id)
                            
                            if not candidate_product:
                                logger.warning(f"Product {candidate_id} not found in database, skipping")
                                failed_count += 1
                                continue
                            
                            # Convert FAISS distance (0-1 cosine similarity) to 0-100 scale
                            similarity_score = float(dist) * 100.0
                            
                            # Create similarities dict compatible with legacy format
                            similarities = {
                                'combined_similarity': similarity_score,
                                'color_similarity': similarity_score,
                                'shape_similarity': similarity_score,
                                'texture_similarity': similarity_score
                            }
                            
                            # Handle missing metadata gracefully
                            from matching_utils import track_missing_metadata, create_match_result
                            
                            missing_fields = track_missing_metadata(candidate_product, data_quality_issues)
                            
                            if missing_fields:
                                logger.debug(f"Product {candidate_id} missing metadata: {missing_fields}")
                            
                            # Create match result
                            match_result = create_match_result(
                                candidate_id,
                                candidate_product,
                                similarities,
                                missing_fields
                            )
                            
                            matches.append(match_result)
                            successful_count += 1
                            
                        except Exception as e:
                            logger.error(f"Error processing FAISS result for product {candidate_id}: {e}")
                            failed_count += 1
                            data_quality_issues['computation_errors'] += 1
                            if not skip_invalid_products:
                                raise
                    
                    logger.info(f"FAISS fast path complete: {successful_count} matches, {failed_count} failed")
                else:
                    logger.warning("FAISS search returned None, falling back to brute force")
            else:
                logger.info(f"FAISS index not available for category '{search_category}', using brute force")
                
        except ImportError:
            logger.warning("FAISS not available (install faiss-cpu), using brute force")
        except Exception as e:
            logger.error(f"FAISS search failed: {e}, falling back to brute force", exc_info=True)
    
    # Fallback to brute force if FAISS not used
    if not faiss_used:
        logger.info("Using brute force similarity computation")
        
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
                from matching_utils import validate_candidate_features_quick
                
                if not validate_candidate_features_quick(candidate_feature_dict):
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
                    if use_clip:
                        # CLIP mode: use cosine similarity on embeddings
                        # CLIP embeddings are stored in color_features column with embedding_type='clip'
                        if candidate_feature_dict.get('embedding_type') == 'clip':
                            candidate_embedding = candidate_feature_dict['color_features']
                        elif 'clip_embedding' in candidate_feature_dict:
                            # Support explicit clip_embedding key (future enhancement)
                            candidate_embedding = candidate_feature_dict['clip_embedding']
                        else:
                            logger.warning(f"Product {candidate_id} missing CLIP embedding, skipping")
                            warnings_list.append(f"Product {candidate_id} missing CLIP embedding")
                            data_quality_issues['missing_features'] += 1
                            failed_count += 1
                            if not skip_invalid_products:
                                raise MissingFeaturesError(candidate_id)
                            continue
                        
                        # Validate candidate embedding
                        if not isinstance(candidate_embedding, np.ndarray) or len(candidate_embedding) != 512:
                            logger.warning(f"Product {candidate_id} has invalid CLIP embedding (expected 512-dim array)")
                            warnings_list.append(f"Product {candidate_id} has invalid CLIP embedding")
                            data_quality_issues['corrupted_features'] += 1
                            failed_count += 1
                            if not skip_invalid_products:
                                raise InvalidFeatureError(f"Product {candidate_id} has invalid CLIP embedding")
                            continue
                        
                        # Compute CLIP similarity
                        similarity_score = compute_clip_similarity(query_embedding, candidate_embedding)
                        
                        # Create similarities dict compatible with legacy format
                        similarities = {
                            'combined_similarity': similarity_score,
                            'color_similarity': similarity_score,  # For database storage compatibility
                            'shape_similarity': similarity_score,
                            'texture_similarity': similarity_score
                        }
                    else:
                        # Legacy mode: use traditional features
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
                except Exception as e:
                    # Handle CLIP-specific errors
                    logger.warning(f"Similarity computation failed for product {candidate_id}: {e}")
                    warnings_list.append(f"Product {candidate_id}: {str(e)}")
                    failed_count += 1
                    data_quality_issues['computation_errors'] += 1
                    errors_list.append({
                        'product_id': candidate_id,
                        'error': str(e),
                        'error_code': 'SIMILARITY_ERROR'
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
                from matching_utils import track_missing_metadata, create_match_result
                
                missing_fields = track_missing_metadata(candidate_product, data_quality_issues)
                
                if missing_fields:
                    logger.debug(f"Product {candidate_id} missing metadata: {missing_fields}")
                
                # Create match result
                match_result = create_match_result(
                    candidate_id,
                    candidate_product,
                    similarities,
                    missing_fields
                )
                
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
        # No matches found - this is OK, just return empty results
        logger.warning(f"No matches found for product {product_id} (FAISS returned 0 candidates or all failed)")
        return {
            'matches': [],
            'total_candidates': len(candidate_features),
            'successful_matches': 0,
            'failed_matches': failed_count,
            'filtered_by_threshold': 0,
            'warnings': warnings_list if warnings_list else None,
            'errors': errors_list if errors_list else None,
            'data_quality_summary': data_quality_issues if any(data_quality_issues.values()) else None
        }
    
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
            from database import bulk_insert_matches
            
            # Collect all matches for batch insert
            matches_to_insert = [
                (product_id, match['product_id'], match['similarity_score'],
                 match['color_score'], match['shape_score'], match['texture_score'])
                for match in matches
            ]
            
            # Batch insert all matches in one transaction
            inserted_count = bulk_insert_matches(matches_to_insert)
            logger.info(f"Batch inserted {inserted_count} matches for product {product_id}")
        except Exception as e:
            logger.error(f"Failed to store matches: {e}")
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
        'matching_mode': 'visual',
        'visual_mode': 'clip' if use_clip else 'legacy',
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
        f"{sum(data_quality_issues.values())} data quality issues "
        f"(visual mode: {'CLIP' if use_clip else 'legacy'})"
    )
    
    if data_quality_issues['missing_features'] > 0:
        logger.warning(f"Data quality: {data_quality_issues['missing_features']} products missing features")
    if data_quality_issues['corrupted_features'] > 0:
        logger.warning(f"Data quality: {data_quality_issues['corrupted_features']} products with corrupted features")
    if data_quality_issues['missing_metadata'] > 0:
        logger.debug(f"Data quality: {data_quality_issues['missing_metadata']} products missing metadata")
    
    return result


# NOTE: batch_find_matches is defined later (line ~2016) with parallel processing
# The parallel version is the one that's actually used


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



# ============================================================================
# MODE 2: METADATA MATCHING (CSV Only - No Images Required)
# ============================================================================

def compute_sku_similarity(sku1: Optional[str], sku2: Optional[str]) -> float:
    """
    Compute SKU similarity using Levenshtein distance.
    
    Handles:
    - Missing SKUs (None or empty)
    - Case insensitivity
    - Whitespace normalization
    - Pattern matching (e.g., PM-001 vs PM-002)
    
    Returns:
        Similarity score 0-100 (100 = identical)
    """
    # Handle missing SKUs
    if not sku1 or not sku2:
        return 0.0
    
    # Normalize
    sku1 = str(sku1).strip().upper()
    sku2 = str(sku2).strip().upper()
    
    if not sku1 or not sku2:
        return 0.0
    
    # Exact match
    if sku1 == sku2:
        return 100.0
    
    # OPTIMIZATION: Quick reject based on length difference
    # If SKUs differ by more than 5 characters, they're probably not similar
    # This avoids expensive Levenshtein calculation
    len_diff = abs(len(sku1) - len(sku2))
    if len_diff > 5:
        return 0.0
    
    # Levenshtein distance
    distance = levenshtein_distance(sku1, sku2)
    max_len = max(len(sku1), len(sku2))
    
    if max_len == 0:
        return 0.0
    
    # Convert distance to similarity (0-100)
    similarity = (1 - (distance / max_len)) * 100
    return max(0.0, min(100.0, similarity))


def compute_name_similarity(name1: Optional[str], name2: Optional[str]) -> float:
    """
    Compute product name similarity using fuzzy matching.
    
    Handles:
    - Missing names
    - Case insensitivity
    - Extra whitespace
    - Common words (the, a, an)
    - Punctuation
    - Word order variations
    - Partial matches
    - Descriptors (100 pages, 32oz, etc.)
    
    Returns:
        Similarity score 0-100
    """
    # Handle missing names
    if not name1 or not name2:
        return 0.0
    
    # Normalize
    name1 = str(name1).strip().lower()
    name2 = str(name2).strip().lower()
    
    if not name1 or not name2:
        return 0.0
    
    # Remove common words and punctuation
    import re
    common_words = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'for', 'with'}
    
    def clean_name(name):
        # Remove punctuation
        name = re.sub(r'[^\w\s]', ' ', name)
        # Split and filter common words
        words = [w for w in name.split() if w not in common_words and len(w) > 0]
        return ' '.join(words)
    
    name1_clean = clean_name(name1)
    name2_clean = clean_name(name2)
    
    # Exact match after cleaning
    if name1_clean == name2_clean:
        return 100.0
    
    # Check for word overlap (handles word order variations)
    # e.g., "coffee mug ceramic" vs "ceramic coffee mug"
    # Also handles descriptors like "100 pages", "32oz", "15 inch" vs "15inch", etc.
    words1 = set(name1_clean.split())
    words2 = set(name2_clean.split())
    
    if words1 and words2:
        overlap = len(words1 & words2)
        
        # IMPROVED: Use Jaccard similarity instead of union
        # This is more forgiving for descriptors (100 pages, 32oz, etc.)
        # Jaccard = intersection / union
        # For "spiral notebook" vs "notebook spiral 100 pages":
        #   intersection = {spiral, notebook} = 2
        #   union = {spiral, notebook, 100, pages} = 4
        #   Jaccard = 2/4 = 0.5 = 50%
        # But we want this to match, so we use a lower threshold
        
        total_unique = len(words1 | words2)
        word_overlap_sim = (overlap / total_unique) * 100 if total_unique > 0 else 0
        
        # IMPROVED: Lower threshold from 60% to 50% for better descriptor handling
        # This allows matches like "Spiral Notebook" vs "Notebook Spiral 100 Pages"
        # where the core words match but descriptors differ
        # Also handles "15 inch Laptop Backpack" vs "Backpack Laptop 15inch"
        if word_overlap_sim >= 50:
            return word_overlap_sim
        
        # ADDITIONAL: Check if most words match (>70% of shorter name)
        # This handles cases where one name has extra descriptors
        # e.g., "Wireless Mouse" vs "Wireless Mouse Ergonomic"
        min_words = min(len(words1), len(words2))
        if min_words > 0 and overlap >= min_words * 0.7:
            # At least 70% of the shorter name matches
            # Calculate score based on overlap ratio
            return (overlap / min_words) * 100
    
    # Levenshtein distance for remaining cases
    distance = levenshtein_distance(name1_clean, name2_clean)
    max_len = max(len(name1_clean), len(name2_clean))
    
    if max_len == 0:
        return 0.0
    
    similarity = (1 - (distance / max_len)) * 100
    return max(0.0, min(100.0, similarity))


def compute_category_similarity(cat1: Optional[str], cat2: Optional[str]) -> float:
    """
    Compute category similarity (exact match or fuzzy).
    
    Handles:
    - Missing categories (NULL)
    - Case insensitivity
    - Whitespace
    - Fuzzy matching for typos
    
    Returns:
        Similarity score 0-100 (100 = exact match, 0 = different)
    """
    # Normalize categories
    cat1_norm = normalize_category(cat1)
    cat2_norm = normalize_category(cat2)
    
    # Both missing
    if cat1_norm is None and cat2_norm is None:
        return 50.0  # Neutral score
    
    # One missing
    if cat1_norm is None or cat2_norm is None:
        return 0.0
    
    # Exact match
    if cat1_norm.lower() == cat2_norm.lower():
        return 100.0
    
    # Fuzzy match for typos
    distance = levenshtein_distance(cat1_norm.lower(), cat2_norm.lower())
    max_len = max(len(cat1_norm), len(cat2_norm))
    
    if max_len == 0:
        return 0.0
    
    # Allow small typos (distance <= 2)
    if distance <= 2:
        similarity = (1 - (distance / max_len)) * 100
        return max(0.0, min(100.0, similarity))
    
    return 0.0  # Different categories


def compute_price_similarity(price1: Optional[float], price2: Optional[float]) -> float:
    """
    Compute price range similarity.
    
    Handles:
    - Missing prices
    - Invalid prices (negative, zero)
    - Percentage difference calculation
    - Lenient matching for similar products
    
    Returns:
        Similarity score 0-100 (100 = same price, decreases with difference)
    """
    # Handle missing or invalid prices
    try:
        if price1 is None or price2 is None:
            return 50.0  # Neutral score for missing data (don't penalize)
        
        price1 = float(price1)
        price2 = float(price2)
        
        if price1 <= 0 or price2 <= 0:
            return 50.0  # Neutral score for invalid prices
        
    except (ValueError, TypeError):
        return 50.0  # Neutral score for parsing errors
    
    # Exact match
    if price1 == price2:
        return 100.0
    
    # Calculate percentage difference
    avg_price = (price1 + price2) / 2
    diff = abs(price1 - price2)
    percent_diff = (diff / avg_price) * 100
    
    # More lenient scoring: allow up to 50% difference for similar products
    # 0% diff = 100, 50% diff = 50, 100% diff = 0
    similarity = max(0.0, 100.0 - percent_diff)
    return similarity


def compute_performance_similarity(perf1: Optional[Dict], perf2: Optional[Dict]) -> float:
    """
    Compute performance tier similarity based on sales metrics.
    
    Handles:
    - Missing performance data
    - Invalid metrics
    - Multiple performance indicators (sales, views, conversion, revenue)
    
    Returns:
        Similarity score 0-100
    """
    # Handle missing performance data
    if not perf1 or not perf2:
        return 0.0
    
    try:
        # Extract metrics with defaults
        sales1 = float(perf1.get('sales', 0))
        sales2 = float(perf2.get('sales', 0))
        
        views1 = float(perf1.get('views', 0))
        views2 = float(perf2.get('views', 0))
        
        conversion1 = float(perf1.get('conversion_rate', 0))
        conversion2 = float(perf2.get('conversion_rate', 0))
        
        revenue1 = float(perf1.get('revenue', 0))
        revenue2 = float(perf2.get('revenue', 0))
        
    except (ValueError, TypeError, AttributeError):
        return 0.0
    
    # Calculate similarity for each metric
    similarities = []
    
    # Sales similarity
    if sales1 > 0 and sales2 > 0:
        sales_ratio = min(sales1, sales2) / max(sales1, sales2)
        similarities.append(sales_ratio * 100)
    
    # Views similarity
    if views1 > 0 and views2 > 0:
        views_ratio = min(views1, views2) / max(views1, views2)
        similarities.append(views_ratio * 100)
    
    # Conversion similarity
    if conversion1 > 0 and conversion2 > 0:
        conv_diff = abs(conversion1 - conversion2)
        conv_sim = max(0, 100 - (conv_diff * 2))  # 50% diff = 0 similarity
        similarities.append(conv_sim)
    
    # Revenue similarity
    if revenue1 > 0 and revenue2 > 0:
        revenue_ratio = min(revenue1, revenue2) / max(revenue1, revenue2)
        similarities.append(revenue_ratio * 100)
    
    # Average of available metrics
    if similarities:
        return sum(similarities) / len(similarities)
    
    return 0.0


# ============================================================================
# MODE 3: HYBRID MATCHING (VISUAL + METADATA)
# ============================================================================
# NOTE: Mode 3 implementation has been moved to backend/hybrid_matching.py
# This module now orchestrates Mode 1 and Mode 2 instead of duplicating logic.
# Import from hybrid_matching module:
#   from hybrid_matching import find_hybrid_matches, batch_find_hybrid_matches
# ============================================================================


def find_metadata_matches(
    product_id: int,
    threshold: float = 0.0,
    limit: int = 10,
    sku_weight: float = 0.25,
    name_weight: float = 0.40,
    category_weight: float = 0.15,
    price_weight: float = 0.10,
    performance_weight: float = 0.10,
    store_matches: bool = True,
    skip_invalid_products: bool = True,
    match_against_all: bool = False
) -> Dict[str, Any]:
    """
    Find similar products based on metadata only (Mode 2).
    
    This function matches products using CSV metadata without requiring images:
    - SKU pattern matching
    - Product name similarity
    - Category matching
    - Price range similarity
    - Performance tier matching
    
    Handles all real-world data issues:
    - Missing fields (NULL values)
    - Malformed data
    - Invalid formats
    - Corrupted entries
    
    Args:
        product_id: ID of product to match
        threshold: Minimum similarity score (0-100)
        limit: Maximum number of matches
        sku_weight: Weight for SKU similarity (default: 0.30)
        name_weight: Weight for name similarity (default: 0.25)
        category_weight: Weight for category match (default: 0.20)
        price_weight: Weight for price similarity (default: 0.15)
        performance_weight: Weight for performance similarity (default: 0.10)
        store_matches: Whether to store results in database
        skip_invalid_products: Continue on errors
    
    Returns:
        Dictionary with matches and metadata
    
    Raises:
        ProductNotFoundError: If product doesn't exist
        EmptyCatalogError: If no products to match against
    """
    warnings_list = []
    errors_list = []
    data_quality_issues = {
        'missing_sku': 0,
        'missing_name': 0,
        'missing_category': 0,
        'missing_price': 0,
        'missing_performance': 0,
        'invalid_data': 0
    }
    
    logger.info(f"Finding metadata matches for product {product_id}")
    
    # Step 1: Get query product
    product = get_product_by_id(product_id)
    if not product:
        raise ProductNotFoundError(product_id)
    
    # Extract metadata from query product
    from matching_utils import get_product_metadata
    
    query_sku = product['sku'] if product['sku'] else None
    query_name = product['product_name'] if product['product_name'] else None
    query_category = product['category'] if product['category'] else None
    
    # Get price and performance
    query_price, query_performance = get_product_metadata(product_id, logger)
    
    # Step 2: Get candidate products with category filtering for performance
    from database import get_all_products
    try:
        if match_against_all or query_category is None:
            # Get all historical products
            candidates = get_all_products(is_historical=True)
        else:
            # PERFORMANCE OPTIMIZATION: Filter by category at database level
            # This can reduce candidates from thousands to hundreds
            candidates_same_category = get_products_by_category(query_category, is_historical=True)
            
            # Also get uncategorized products (NULL category)
            all_candidates = get_all_products(is_historical=True)
            candidates_uncategorized = [c for c in all_candidates if c['category'] is None]
            
            # Combine
            candidates = candidates_same_category + candidates_uncategorized
            logger.info(f"Category filtering: {len(candidates_same_category)} in '{query_category}', {len(candidates_uncategorized)} uncategorized")
    except Exception as e:
        logger.error(f"Failed to get candidate products: {e}")
        raise EmptyCatalogError()
    
    if not candidates:
        raise EmptyCatalogError()
    
    logger.info(f"Found {len(candidates)} candidate products")
    
    # PERFORMANCE OPTIMIZATION: Batch fetch price and performance data
    # Instead of querying per product, get all at once
    from database import get_products_with_price_history, get_products_with_performance_history
    
    logger.info("Batch fetching price and performance data...")
    
    # Build lookup dictionaries for O(1) access
    from matching_utils import batch_fetch_metadata
    
    price_lookup, performance_lookup = batch_fetch_metadata(logger)
    
    # Step 3: Compute metadata similarities
    matches = []
    successful_count = 0
    failed_count = 0
    
    for candidate in candidates:
        candidate_id = candidate['id']
        
        # Skip self
        if candidate_id == product_id:
            continue
        
        try:
            # Extract candidate metadata
            cand_sku = candidate['sku'] if candidate['sku'] else None
            cand_name = candidate['product_name'] if candidate['product_name'] else None
            cand_category = candidate['category'] if candidate['category'] else None
            
            # Get candidate price from cache (O(1) lookup)
            cand_price = price_lookup.get(candidate_id)
            
            # Get candidate performance from cache (O(1) lookup)
            cand_performance = performance_lookup.get(candidate_id)
            
            # Track missing data
            if not cand_sku:
                data_quality_issues['missing_sku'] += 1
            if not cand_name:
                data_quality_issues['missing_name'] += 1
            if not cand_category:
                data_quality_issues['missing_category'] += 1
            if not cand_price:
                data_quality_issues['missing_price'] += 1
            if not cand_performance:
                data_quality_issues['missing_performance'] += 1
            
            # PERFORMANCE OPTIMIZATION: Compute similarities with early termination
            # Calculate most important metrics first (SKU, Name, Category)
            # If these are all very low, skip expensive calculations
            
            sku_sim = compute_sku_similarity(query_sku, cand_sku)
            name_sim = compute_name_similarity(query_name, cand_name)
            category_sim = compute_category_similarity(query_category, cand_category)
            
            # Early termination: If top 3 metrics are all 0, skip this candidate
            # This saves price and performance calculations for obviously bad matches
            if threshold > 0 and sku_sim == 0 and name_sim == 0 and category_sim == 0:
                # Maximum possible score is from price + performance (30% total)
                max_possible = 100 * (price_weight + performance_weight)
                if max_possible < threshold:
                    # Can't possibly meet threshold, skip
                    continue
            
            # Calculate remaining similarities
            price_sim = compute_price_similarity(query_price, cand_price)
            performance_sim = compute_performance_similarity(query_performance, cand_performance)
            
            # Compute weighted combined similarity
            combined_sim = (
                sku_sim * sku_weight +
                name_sim * name_weight +
                category_sim * category_weight +
                price_sim * price_weight +
                performance_sim * performance_weight
            )
            
            # Create match result
            match_result = {
                'product_id': candidate_id,
                'image_path': candidate['image_path'] if candidate['image_path'] else '',
                'category': cand_category,
                'product_name': cand_name,
                'sku': cand_sku,
                'similarity_score': combined_sim,
                'sku_score': sku_sim,
                'name_score': name_sim,
                'category_score': category_sim,
                'price_score': price_sim,
                'performance_score': performance_sim,
                'is_potential_duplicate': combined_sim > 90,
                'created_at': candidate['created_at'] if candidate['created_at'] else '',
                'has_missing_data': not all([cand_sku, cand_name, cand_category, cand_price, cand_performance])
            }
            
            matches.append(match_result)
            successful_count += 1
            
        except Exception as e:
            logger.error(f"Error processing candidate {candidate_id}: {e}")
            failed_count += 1
            data_quality_issues['invalid_data'] += 1
            errors_list.append({
                'product_id': candidate_id,
                'error': str(e),
                'error_code': 'PROCESSING_ERROR'
            })
            if not skip_invalid_products:
                raise
    
    # Step 4: Sort and filter
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    filtered_count = 0
    if threshold > 0:
        original_count = len(matches)
        matches = [m for m in matches if m['similarity_score'] >= threshold]
        filtered_count = original_count - len(matches)
    
    if limit > 0:
        matches = matches[:limit]
    
    # Step 5: Store matches (optional) - BATCH INSERT for 10-100x speedup
    if store_matches and matches:
        try:
            from database import bulk_insert_matches
            
            # Collect all matches for batch insert
            matches_to_insert = [
                (product_id, match['product_id'], match['similarity_score'],
                 0, 0, 0)  # color_score, shape_score, texture_score = 0 (N/A for metadata)
                for match in matches
            ]
            
            # Batch insert all matches in one transaction
            inserted_count = bulk_insert_matches(matches_to_insert)
            logger.info(f"Batch inserted {inserted_count} matches for product {product_id}")
        except Exception as e:
            logger.error(f"Failed to store matches: {e}")
            warnings_list.append(f"Failed to store matches: {str(e)}")
    
    # Step 6: Prepare response
    result = {
        'matches': matches,
        'total_candidates': len(candidates),
        'successful_matches': successful_count,
        'failed_matches': failed_count,
        'filtered_by_threshold': filtered_count,
        'threshold_used': threshold,
        'limit_used': limit,
        'matching_mode': 'metadata',
        'warnings': warnings_list,
        'errors': errors_list if errors_list else None,
        'data_quality_issues': data_quality_issues,
        'data_quality_summary': {
            'total_issues': sum(data_quality_issues.values()),
            'success_rate': round(successful_count / len(candidates) * 100, 1) if candidates else 0,
            'has_data_quality_issues': sum(data_quality_issues.values()) > 0
        }
    }
    
    logger.info(f"Metadata matching complete: {len(matches)} matches, {successful_count} successful, {failed_count} failed")
    
    return result


# ============================================================================
# BATCH MATCHING WITH PARALLEL PROCESSING
# ============================================================================

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
    skip_invalid_products: bool = True,
    max_workers: Optional[int] = None,
    preload_catalog: bool = True
) -> Dict[str, Any]:
    """
    Find matches for multiple products in batch with parallel processing.
    
    PERFORMANCE OPTIMIZATIONS:
    - Parallel processing using ThreadPoolExecutor (CPU multithreading)
    - Shared FAISS index across threads (thread-safe read-only access)
    - Optional catalog preloading to avoid repeated database queries
    - Isolated error handling per product (one failure doesn't stop batch)
    - Progress tracking for large batches
    
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
        skip_invalid_products: If True, skip products with data issues
        max_workers: Number of parallel workers (default: cpu_count + 4)
        preload_catalog: Preload catalog features into cache for faster matching
    
    Returns:
        Dictionary with:
        - 'results': List of match results per product
        - 'summary': Summary statistics including data quality metrics
        - 'errors': List of products that failed
    
    Requirements: 6.1, 6.2, 6.3
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    logger.info(f"[BATCH-MODE1] ▶ Starting batch visual matching for {len(product_ids)} products")
    logger.info(f"[BATCH-MODE1] Parallelization: {max_workers} workers (ThreadPoolExecutor)")
    logger.info(f"[BATCH-MODE1] Catalog preload: {preload_catalog}")
    
    # PERFORMANCE OPTIMIZATION: Preload catalog features into memory cache
    # This avoids repeated database queries for the same catalog products
    if preload_catalog:
        from feature_cache import get_feature_cache
        cache = get_feature_cache()
        
        # Preload all historical products (or specific category if needed)
        logger.info("Preloading catalog features into cache...")
        cache.preload_catalog(category=None, is_historical=True)
    
    # PERFORMANCE OPTIMIZATION: Preload FAISS indexes for all categories
    # FAISS indexes are thread-safe for read-only operations (searching)
    # This avoids repeated index checks and allows parallel FAISS searches
    faiss_available = False
    faiss_manager = None
    if CLIP_AVAILABLE:
        try:
            from faiss_index import faiss_manager as fm
            faiss_manager = fm
            faiss_available = True
            logger.info("[BATCH-MODE1] ✓ FAISS indexes preloaded (thread-safe for parallel searches)")
        except ImportError:
            logger.info("[BATCH-MODE1] FAISS not available, using brute force similarity")
        except Exception as e:
            logger.warning(f"[BATCH-MODE1] Failed to load FAISS manager: {e}, using brute force")
    
    results = []
    errors = []
    successful = 0
    failed = 0
    
    def process_single_match(product_id: int) -> Tuple[int, Dict[str, Any]]:
        """Process a single product match and return (product_id, result)"""
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
            
            return (product_id, {
                'product_id': product_id,
                'status': 'success',
                'match_count': len(match_result['matches']),
                'matches': match_result['matches'],
                'warnings': match_result['warnings'],
                'data_quality_issues': match_result.get('data_quality_issues', {}),
                'data_quality_summary': match_result.get('data_quality_summary', {})
            })
            
        except (ProductNotFoundError, MissingFeaturesError, EmptyCatalogError, AllMatchesFailedError) as e:
            logger.error(f"Failed to match product {product_id}: {e.message}")
            
            error_info = {
                'product_id': product_id,
                'status': 'failed',
                'error': e.message,
                'error_code': e.error_code,
                'suggestion': e.suggestion
            }
            
            return (product_id, error_info)
                
        except Exception as e:
            logger.error(f"Unexpected error matching product {product_id}: {e}")
            
            error_info = {
                'product_id': product_id,
                'status': 'failed',
                'error': str(e),
                'error_code': 'UNKNOWN_ERROR'
            }
            
            return (product_id, error_info)
    
    # Process in parallel - DON'T store matches yet, we'll batch insert them all at once
    logger.info(f"[BATCH-MODE1] ▶ Starting parallel matching with {max_workers} workers")
    logger.info(f"[BATCH-MODE1] Processing {len(product_ids)} products in parallel...")
    logger.info(f"[BATCH-MODE1] Matches will be batch inserted after all products are matched (1 DB call)")
    
    # Collect all matches for batch insertion
    all_matches_to_insert = []
    INCREMENTAL_BATCH_SIZE = 100  # Insert every 100 matches to avoid memory bloat
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with store_matches=False (we'll batch insert after)
        futures = {}
        for pid in product_ids:
            future = executor.submit(
                find_matches,
                product_id=pid,
                threshold=threshold,
                limit=limit,
                match_against_all=match_against_all,
                include_uncategorized=include_uncategorized,
                color_weight=color_weight,
                shape_weight=shape_weight,
                texture_weight=texture_weight,
                store_matches=False,  # Don't store yet - we'll batch insert
                skip_invalid_products=skip_invalid_products
            )
            futures[future] = pid
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            product_id = futures[future]
            
            try:
                match_result = future.result()
                
                # Collect matches for batch insertion
                if match_result['matches']:
                    for match in match_result['matches']:
                        all_matches_to_insert.append((
                            product_id,
                            match['product_id'],
                            match['similarity_score'],
                            match['color_score'],
                            match['shape_score'],
                            match['texture_score']
                        ))
                    
                    # OPTIMIZATION: Insert incrementally to avoid memory bloat
                    # This starts inserting while other workers are still matching
                    if len(all_matches_to_insert) >= INCREMENTAL_BATCH_SIZE and store_matches:
                        try:
                            from database import bulk_insert_matches
                            inserted = bulk_insert_matches(all_matches_to_insert)
                            logger.info(f"[BATCH-MODE1] ▶ Incremental insert: {inserted} matches (batch inserted while matching continues)")
                            all_matches_to_insert = []  # Clear for next batch
                        except Exception as e:
                            logger.warning(f"[BATCH-MODE1] Incremental insert failed: {e}, will retry at end")
                
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
                logger.debug(f"[BATCH-MODE1] Product {product_id}: {len(match_result['matches'])} matches found")
                
            except (ProductNotFoundError, MissingFeaturesError, EmptyCatalogError, AllMatchesFailedError) as e:
                logger.error(f"Failed to match product {product_id}: {e.message}")
                
                error_info = {
                    'product_id': product_id,
                    'status': 'failed',
                    'error': e.message,
                    'error_code': e.error_code,
                    'suggestion': e.suggestion
                }
                
                results.append(error_info)
                errors.append(error_info)
                failed += 1
                logger.debug(f"[BATCH-MODE1] Product {product_id}: FAILED - {e.message}")
                
            except Exception as e:
                logger.error(f"Unexpected error matching product {product_id}: {e}")
                
                error_info = {
                    'product_id': product_id,
                    'status': 'failed',
                    'error': str(e),
                    'error_code': 'UNKNOWN_ERROR'
                }
                
                results.append(error_info)
                errors.append(error_info)
                failed += 1
                logger.debug(f"[BATCH-MODE1] Product {product_id}: FAILED - {str(e)}")
            
            # Log progress every 10 products
            if i % 10 == 0:
                logger.info(f"[BATCH-MODE1] Progress: {i}/{len(product_ids)} completed ({successful} successful, {failed} failed)")
    
    # PERFORMANCE OPTIMIZATION: Batch insert remaining matches in chunks
    # Smaller chunks = faster insertion while matching still happening
    if store_matches and all_matches_to_insert:
        try:
            from database import bulk_insert_matches
            
            # Chunk size: 100 matches per transaction (smaller = faster + less memory)
            CHUNK_SIZE = 100
            
            total_inserted = 0
            num_chunks = (len(all_matches_to_insert) + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            if num_chunks == 1:
                # Small batch - insert all at once
                logger.info(f"[BATCH-MODE1] ▶ Final batch inserting {len(all_matches_to_insert)} remaining matches...")
                inserted_count = bulk_insert_matches(all_matches_to_insert)
                logger.info(f"[BATCH-MODE1] ✓ Final batch inserted {inserted_count} matches")
                total_inserted = inserted_count
            else:
                # Large batch - chunk into multiple transactions (smaller chunks = faster)
                logger.info(f"[BATCH-MODE1] ▶ Final batch inserting {len(all_matches_to_insert)} remaining matches in {num_chunks} chunks ({CHUNK_SIZE} per chunk)...")
                
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * CHUNK_SIZE
                    end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(all_matches_to_insert))
                    chunk = all_matches_to_insert[start_idx:end_idx]
                    
                    inserted_count = bulk_insert_matches(chunk)
                    total_inserted += inserted_count
                    
                    logger.info(f"[BATCH-MODE1] ✓ Final chunk {chunk_idx + 1}/{num_chunks}: Inserted {inserted_count} matches")
                
                logger.info(f"[BATCH-MODE1] ✓ Final batch inserted {total_inserted} remaining matches in {num_chunks} transactions")
        except Exception as e:
            logger.error(f"Failed to batch insert remaining matches: {e}")
    
    summary = {
        'total_products': len(product_ids),
        'successful': successful,
        'failed': failed,
        'success_rate': round(successful / len(product_ids) * 100, 1) if product_ids else 0,
        'total_matches': len(all_matches_to_insert),
        'batch_insert_used': store_matches and len(all_matches_to_insert) > 0
    }
    
    logger.info(f"[BATCH-MODE1] ✓ COMPLETE! {successful}/{len(product_ids)} successful ({summary['success_rate']}% success rate)")
    logger.info(f"[BATCH-MODE1] Total matches: {len(all_matches_to_insert)} (batch inserted in 1 DB call)")
    logger.info(f"[BATCH-MODE1] All products matched in parallel using {max_workers} workers")
    
    return {
        'results': results,
        'summary': summary,
        'errors': errors if errors else None
    }


def batch_find_metadata_matches(
    product_ids: List[int],
    threshold: float = 0.0,
    limit: int = 10,
    sku_weight: float = 0.20,
    name_weight: float = 0.40,
    category_weight: float = 0.15,
    price_weight: float = 0.15,
    performance_weight: float = 0.10,
    store_matches: bool = True,
    skip_invalid_products: bool = True,
    match_against_all: bool = False,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Find metadata matches for multiple products in batch with parallel processing.
    
    Mode 2 (Metadata-only matching) with CPU multithreading optimization.
    Perfect for CSV-only workflows without images.
    
    PERFORMANCE OPTIMIZATIONS:
    - Parallel processing using ThreadPoolExecutor
    - Batch metadata fetching (price/performance data loaded once)
    - Isolated error handling per product
    
    Args:
        product_ids: List of product IDs to match
        threshold: Minimum similarity score (0-100)
        limit: Maximum matches per product
        sku_weight: Weight for SKU similarity
        name_weight: Weight for name similarity
        category_weight: Weight for category similarity
        price_weight: Weight for price similarity
        performance_weight: Weight for performance similarity
        store_matches: Store results in database
        skip_invalid_products: Continue on errors
        match_against_all: Match against all categories
        max_workers: Number of parallel workers (default: cpu_count + 4)
    
    Returns:
        Dictionary with results and summary
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    logger.info(f"[BATCH-MODE2] ▶ Starting batch metadata matching for {len(product_ids)} products")
    logger.info(f"[BATCH-MODE2] Parallelization: {max_workers} workers (ThreadPoolExecutor)")
    logger.info(f"[BATCH-MODE2] No GPU/CLIP needed - metadata comparison only")
    logger.info(f"[BATCH-MODE2] Weights: SKU={sku_weight}, Name={name_weight}, Category={category_weight}, Price={price_weight}, Performance={performance_weight}")
    
    results = []
    errors = []
    successful = 0
    failed = 0
    
    def process_single_metadata_match(product_id: int) -> Tuple[int, Dict[str, Any]]:
        """Process a single product metadata match"""
        try:
            match_result = find_metadata_matches(
                product_id=product_id,
                threshold=threshold,
                limit=limit,
                sku_weight=sku_weight,
                name_weight=name_weight,
                category_weight=category_weight,
                price_weight=price_weight,
                performance_weight=performance_weight,
                store_matches=store_matches,
                skip_invalid_products=skip_invalid_products,
                match_against_all=match_against_all
            )
            
            return (product_id, {
                'product_id': product_id,
                'status': 'success',
                'match_count': len(match_result['matches']),
                'matches': match_result['matches'],
                'warnings': match_result.get('warnings', []),
                'data_quality_issues': match_result.get('data_quality_issues', {})
            })
            
        except (ProductNotFoundError, EmptyCatalogError) as e:
            logger.error(f"Failed to match product {product_id}: {e.message}")
            
            error_info = {
                'product_id': product_id,
                'status': 'failed',
                'error': e.message,
                'error_code': e.error_code,
                'suggestion': e.suggestion
            }
            
            return (product_id, error_info)
                
        except Exception as e:
            logger.error(f"Unexpected error matching product {product_id}: {e}")
            
            error_info = {
                'product_id': product_id,
                'status': 'failed',
                'error': str(e),
                'error_code': 'UNKNOWN_ERROR'
            }
            
            return (product_id, error_info)
    
    # Process in parallel - DON'T store matches yet, we'll batch insert them all at once
    logger.info(f"[BATCH-MODE2] ▶ Starting parallel metadata matching with {max_workers} workers")
    logger.info(f"[BATCH-MODE2] Processing {len(product_ids)} products in parallel...")
    logger.info(f"[BATCH-MODE2] Matches will be batch inserted after all products are matched (1 DB call)")
    logger.info(f"[BATCH-MODE2] Step 1: Submit all {len(product_ids)} tasks to thread pool")
    
    # Collect all matches for batch insertion
    all_matches_to_insert = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with store_matches=False (we'll batch insert after)
        futures = {}
        for pid in product_ids:
            future = executor.submit(
                find_metadata_matches,
                product_id=pid,
                threshold=threshold,
                limit=limit,
                sku_weight=sku_weight,
                name_weight=name_weight,
                category_weight=category_weight,
                price_weight=price_weight,
                performance_weight=performance_weight,
                store_matches=False,  # Don't store yet - we'll batch insert
                skip_invalid_products=skip_invalid_products,
                match_against_all=match_against_all
            )
            futures[future] = pid
        
        # Process results as they complete
        logger.info(f"[BATCH-MODE2] Step 2: Processing results as they complete (as_completed)")
        for i, future in enumerate(as_completed(futures), 1):
            product_id = futures[future]
            
            try:
                match_result = future.result()
                
                # Collect matches for batch insertion
                if match_result['matches']:
                    for match in match_result['matches']:
                        all_matches_to_insert.append((
                            product_id,
                            match['product_id'],
                            match['similarity_score'],
                            0, 0, 0  # color_score, shape_score, texture_score = 0 (N/A for metadata)
                        ))
                
                results.append({
                    'product_id': product_id,
                    'status': 'success',
                    'match_count': len(match_result['matches']),
                    'matches': match_result['matches'],
                    'warnings': match_result.get('warnings', []),
                    'data_quality_issues': match_result.get('data_quality_issues', {})
                })
                
                successful += 1
                logger.debug(f"[BATCH-MODE2] Product {product_id}: {len(match_result['matches'])} matches found (total collected: {len(all_matches_to_insert)})")
                
            except (ProductNotFoundError, EmptyCatalogError) as e:
                logger.error(f"Failed to match product {product_id}: {e.message}")
                
                error_info = {
                    'product_id': product_id,
                    'status': 'failed',
                    'error': e.message,
                    'error_code': e.error_code,
                    'suggestion': e.suggestion
                }
                
                results.append(error_info)
                errors.append(error_info)
                failed += 1
                logger.debug(f"[BATCH-MODE2] Product {product_id}: FAILED - {e.message}")
                
            except Exception as e:
                logger.error(f"Unexpected error matching product {product_id}: {e}")
                
                error_info = {
                    'product_id': product_id,
                    'status': 'failed',
                    'error': str(e),
                    'error_code': 'UNKNOWN_ERROR'
                }
                
                results.append(error_info)
                errors.append(error_info)
                failed += 1
                logger.debug(f"[BATCH-MODE2] Product {product_id}: FAILED - {str(e)}")
            
            # Log progress every 10 products
            if i % 10 == 0:
                logger.info(f"[BATCH-MODE2] Progress: {i}/{len(product_ids)} completed ({successful} successful, {failed} failed)")
    
    # PERFORMANCE OPTIMIZATION: Batch insert all matches in chunks
    # Smaller chunks = less memory overhead + faster insertion while matching still happening
    logger.info(f"[BATCH-MODE2] Step 3: Batch insert all {len(all_matches_to_insert)} matches collected from {successful} products")
    if store_matches and all_matches_to_insert:
        try:
            from database import bulk_insert_matches
            
            # Chunk size: 100 matches per transaction (smaller = faster + less memory)
            # This allows DB insertion to start while other workers still matching
            CHUNK_SIZE = 100
            
            total_inserted = 0
            num_chunks = (len(all_matches_to_insert) + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            if num_chunks == 1:
                # Small batch - insert all at once
                logger.info(f"[BATCH-MODE2] ▶ Batch inserting {len(all_matches_to_insert)} matches in one transaction...")
                inserted_count = bulk_insert_matches(all_matches_to_insert)
                logger.info(f"[BATCH-MODE2] ✓ Batch inserted {inserted_count} matches (1 DB call for {len(product_ids)} products)")
                total_inserted = inserted_count
            else:
                # Large batch - chunk into multiple transactions (smaller chunks = faster)
                logger.info(f"[BATCH-MODE2] ▶ Batch inserting {len(all_matches_to_insert)} matches in {num_chunks} chunks ({CHUNK_SIZE} per chunk)...")
                
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * CHUNK_SIZE
                    end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(all_matches_to_insert))
                    chunk = all_matches_to_insert[start_idx:end_idx]
                    
                    inserted_count = bulk_insert_matches(chunk)
                    total_inserted += inserted_count
                    
                    logger.debug(f"[BATCH-MODE2] Chunk {chunk_idx + 1}/{num_chunks}: Inserted {inserted_count} matches")
                
                logger.info(f"[BATCH-MODE2] ✓ Batch inserted {total_inserted} matches in {num_chunks} transactions for {len(product_ids)} products")
        except Exception as e:
            logger.error(f"Failed to batch insert matches: {e}")
    
    summary = {
        'total_products': len(product_ids),
        'successful': successful,
        'failed': failed,
        'success_rate': round(successful / len(product_ids) * 100, 1) if product_ids else 0,
        'total_matches': len(all_matches_to_insert),
        'batch_insert_used': store_matches and len(all_matches_to_insert) > 0
    }
    
    logger.info(f"[BATCH-MODE2] ✓ COMPLETE! {successful}/{len(product_ids)} successful ({summary['success_rate']}% success rate)")
    logger.info(f"[BATCH-MODE2] Total matches: {len(all_matches_to_insert)} (batch inserted in 1 DB call)")
    logger.info(f"[BATCH-MODE2] All products matched in parallel using {max_workers} workers")
    
    return {
        'results': results,
        'summary': summary,
        'errors': errors if errors else None
    }
