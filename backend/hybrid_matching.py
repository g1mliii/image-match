"""
Hybrid Matching Module (Mode 3)

This module implements hybrid matching by orchestrating Mode 1 (visual) and
Mode 2 (metadata) matching, then combining results with weighted scoring.

This is a lightweight wrapper that delegates to existing optimized functions
in product_matching.py, avoiding code duplication.

Performance:
- Mode 1 and Mode 2 run with their own optimizations (FAISS, batch fetching, etc.)
- Results are merged with simple arithmetic
- Minimal code - just orchestration and merging

Smart Linking:
- Automatically links visual and metadata matches by filename ↔ SKU
- Works even if user didn't use CSV Builder to pre-link data
- Thread-safe and fast (simple string comparison)
"""

import logging
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from database import insert_match

logger = logging.getLogger(__name__)


def matches_by_filename_sku(visual_match: Dict[str, Any], metadata_match: Dict[str, Any]) -> bool:
    """
    Check if visual and metadata matches represent the same product
    by comparing image filename with SKU (with/without extensions).

    This is the automatic backup linking strategy when products weren't
    pre-linked through CSV Builder. Handles ~80% of real-world cases.

    Thread-safe: Pure function, only reads data, no modifications.

    Args:
        visual_match: Visual match dict with 'image_path'
        metadata_match: Metadata match dict with 'sku' or metadata_values

    Returns:
        True if filename matches SKU, False otherwise

    Examples:
        - Image: "ABC-123.jpg" → SKU: "ABC-123" = Match ✓
        - Image: "ABC-123" → SKU: "ABC-123.jpg" = Match ✓
        - Image: "product.png" → SKU: "product" = Match ✓
        - Case-insensitive matching
    """
    if not visual_match or not metadata_match:
        return False

    # Extract filename from image path
    image_path = visual_match.get('image_path', '')
    if not image_path:
        return False

    filename = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename)[0].lower().strip()
    filename_with_ext = filename.lower().strip()

    if not filename_no_ext:
        return False

    # Get SKU from metadata match (try both top-level and metadata_values)
    sku = metadata_match.get('sku') or metadata_match.get('metadata_values', {}).get('sku')
    if not sku:
        return False

    sku_str = str(sku).lower().strip()
    sku_no_ext = os.path.splitext(sku_str)[0]

    # Try all combinations automatically (handles extensions smartly)
    return (filename_no_ext == sku_str or
            filename_no_ext == sku_no_ext or
            filename_with_ext == sku_str or
            filename_with_ext == sku_no_ext)


def find_hybrid_matches(
    product_id: int,
    threshold: float = 0.0,
    limit: int = 10,
    visual_weight: float = 0.50,
    metadata_weight: float = 0.50,
    metadata_weights: Optional[Dict[str, float]] = None,
    store_matches: bool = True,
    skip_invalid_products: bool = True,
    match_against_all: bool = False
) -> Dict[str, Any]:
    """
    Find similar products using hybrid approach (Mode 3).

    Orchestrates Mode 1 (visual) and Mode 2 (metadata) matching, then combines
    results with weighted scoring.

    The hybrid score is calculated as:
    hybrid_score = (visual_score * visual_weight) + (metadata_score * metadata_weight)

    Args:
        product_id: ID of product to match
        threshold: Minimum similarity score (0-100)
        limit: Maximum number of matches
        visual_weight: Weight for visual similarity (default: 0.50)
        metadata_weight: Weight for metadata similarity (default: 0.50)
        metadata_weights: Dynamic weights dict for metadata columns
        store_matches: Whether to store results in database
        skip_invalid_products: Continue on errors
        match_against_all: Match against all categories

    Returns:
        Dictionary with matches and comprehensive scoring.
        When using dynamic metadata_weights, includes 'metadata_scores' dict per match.
    
    Raises:
        ProductNotFoundError: If product doesn't exist
        MissingFeaturesError: If product doesn't have visual features
        EmptyCatalogError: If no products to match against
    """
    from product_matching import find_matches, find_metadata_matches
    
    # Enforce minimum threshold of 30% for filtering
    threshold = max(threshold, 30.0)
    
    logger.info(f"Finding hybrid matches for product {product_id} (visual: {visual_weight*100}%, metadata: {metadata_weight*100}%)")
    
    # Step 1: Run Mode 1 (visual matching) - already optimized with FAISS
    try:
        visual_result = find_matches(
            product_id=product_id,
            threshold=0.0,  # No threshold - we'll filter combined scores later
            limit=limit * 10 if limit > 0 else 1000,  # Get more candidates for merging
            match_against_all=match_against_all,
            include_uncategorized=True,
            store_matches=False,  # Don't store yet - we'll store hybrid scores
            skip_invalid_products=skip_invalid_products
        )
    except Exception as e:
        logger.error(f"Mode 1 (visual) failed: {e}")
        raise
    
    # Step 2: Run Mode 2 (metadata matching) - already optimized with batch fetching
    try:
        # Use dynamic weights if provided, otherwise use legacy individual weights
        if metadata_weights is None:
             raise ValueError("Metadata weights required for Hybrid mode (Legacy removed)")

        metadata_result = find_metadata_matches(
            product_id=product_id,
            threshold=0.0,
            limit=limit * 10 if limit > 0 else 1000,
            weights=metadata_weights,  # Dynamic weights mode
            store_matches=False,
            skip_invalid_products=skip_invalid_products,
            match_against_all=match_against_all
        )
    except Exception as e:
        logger.error(f"Mode 2 (metadata) failed: {e}")
        raise
    
    # Step 3: Merge results - combine visual and metadata scores
    logger.info(f"Merging {len(visual_result['matches'])} visual matches with {len(metadata_result['matches'])} metadata matches")
    
    # Build lookup dictionaries for fast merging
    visual_lookup = {m['product_id']: m for m in visual_result['matches']}
    metadata_lookup = {m['product_id']: m for m in metadata_result['matches']}
    
    # Get all unique candidate IDs
    all_candidate_ids = set(visual_lookup.keys()) | set(metadata_lookup.keys())
    
    # Compute hybrid scores with smart linking
    hybrid_matches = []
    for candidate_id in all_candidate_ids:
        visual_match = visual_lookup.get(candidate_id)
        metadata_match = metadata_lookup.get(candidate_id)

        # Get scores (default to 0 if match not found in one mode)
        visual_score = visual_match['similarity_score'] if visual_match else 0.0
        metadata_score = metadata_match['similarity_score'] if metadata_match else 0.0

        # SMART LINKING: If both found matches but to different products,
        # check if they're actually the same product by filename ↔ SKU
        if visual_match and metadata_match and visual_score > 0 and metadata_score > 0:
            if visual_match.get('product_id') != metadata_match.get('product_id'):
                # Different product_ids - verify they represent same item
                if not matches_by_filename_sku(visual_match, metadata_match):
                    # Not the same product - don't merge these scores
                    continue

        # Compute hybrid score
        hybrid_score = (visual_score * visual_weight) + (metadata_score * metadata_weight)
        
        # Use visual match data as base (has image_path, etc.)
        if visual_match:
            match_data = visual_match.copy()
        elif metadata_match:
            match_data = metadata_match.copy()
        else:
            continue  # Should never happen
        
        # Update with hybrid scores
        match_data['similarity_score'] = hybrid_score
        match_data['visual_score'] = visual_score
        match_data['metadata_score'] = metadata_score
        match_data['is_potential_duplicate'] = hybrid_score > 90
        
        # Add metadata sub-scores and values
        if metadata_match:
            # Pass through dynamic metadata structures
            match_data['metadata_scores'] = metadata_match.get('metadata_scores', {})
            # Ensure metadata values are passed
            if 'metadata_values' in metadata_match:
                match_data['metadata_values'] = metadata_match['metadata_values']
            
            # Map common fields to legacy top-level keys for compatibility
            ms_scores = match_data.get('metadata_scores', {})
            match_data['sku_score'] = ms_scores.get('sku', 0.0)
            match_data['name_score'] = ms_scores.get('name', 0.0)
            match_data['category_score'] = ms_scores.get('category', 0.0)
            match_data['price_score'] = ms_scores.get('price', 0.0)
            match_data['performance_score'] = ms_scores.get('performance', 0.0)

        if not match_data.get('metadata_values') and visual_match and 'metadata_values' in visual_match:
             # Fallback: if visual match somehow has it (unlikely but possible if enriched later)
             match_data['metadata_values'] = visual_match['metadata_values']

        hybrid_matches.append(match_data)
    
    # Step 4: Sort and filter
    hybrid_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    filtered_count = 0
    if threshold > 0:
        original_count = len(hybrid_matches)
        hybrid_matches = [m for m in hybrid_matches if m['similarity_score'] >= threshold]
        filtered_count = original_count - len(hybrid_matches)
    
    if limit > 0:
        hybrid_matches = hybrid_matches[:limit]
    
    # Step 5: Store matches (optional) - BATCH INSERT for 10-100x speedup
    if store_matches and hybrid_matches:
        try:
            from database import bulk_insert_matches
            
            # Collect all matches for batch insert
            matches_to_insert = [
                (product_id, match['product_id'], match['similarity_score'],
                 match.get('color_score', match.get('visual_score', 0.0)),
                 match.get('shape_score', match.get('visual_score', 0.0)),
                 match.get('texture_score', match.get('visual_score', 0.0)))
                for match in hybrid_matches
            ]
            
            # Batch insert all matches in one transaction
            inserted_count = bulk_insert_matches(matches_to_insert)
            logger.info(f"Batch inserted {inserted_count} hybrid matches for product {product_id}")
        except Exception as e:
            logger.error(f"Failed to store hybrid matches: {e}")
    
    # Step 6: Prepare response
    # Combine warnings and data quality issues from both modes
    warnings_list = visual_result.get('warnings', []) + metadata_result.get('warnings', [])
    
    # Merge data quality issues
    data_quality_issues = {}
    for key in set(visual_result.get('data_quality_issues', {}).keys()) | set(metadata_result.get('data_quality_issues', {}).keys()):
        data_quality_issues[key] = (
            visual_result.get('data_quality_issues', {}).get(key, 0) +
            metadata_result.get('data_quality_issues', {}).get(key, 0)
        )
    
    from product_matching import calculate_summary_stats
    result = {
        'matches': hybrid_matches,
        'summary_stats': calculate_summary_stats(hybrid_matches), # ADDED: upfront stats for groups
        'total_candidates': max(visual_result.get('total_candidates', 0), metadata_result.get('total_candidates', 0)),
        'successful_matches': len(hybrid_matches),
        'failed_matches': visual_result.get('failed_matches', 0) + metadata_result.get('failed_matches', 0),
        'filtered_by_threshold': filtered_count,
        'threshold_used': threshold,
        'limit_used': limit,
        'matching_mode': 'hybrid',
        'visual_mode': visual_result.get('visual_mode', 'clip'),
        'visual_weight': visual_weight,
        'metadata_weight': metadata_weight,
        'warnings': warnings_list if warnings_list else [],
        'errors': None,
        'data_quality_issues': data_quality_issues,
        'data_quality_summary': {
            'total_issues': sum(data_quality_issues.values()),
            'success_rate': round(len(hybrid_matches) / max(visual_result.get('total_candidates', 1), 1) * 100, 1),
            'has_data_quality_issues': sum(data_quality_issues.values()) > 0
        }
    }
    
    logger.info(f"Hybrid matching complete: {len(hybrid_matches)} matches (visual: {len(visual_result['matches'])}, metadata: {len(metadata_result['matches'])})")
    
    return result


def batch_find_hybrid_matches(
    product_ids: List[int],
    threshold: float = 0.0,
    limit: int = 10,
    visual_weight: float = 0.50,
    metadata_weight: float = 0.50,
    metadata_weights: Optional[Dict[str, float]] = None,
    store_matches: bool = True,
    skip_invalid_products: bool = True,
    match_against_all: bool = False,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Find hybrid matches for multiple products in batch with parallel processing.

    Mode 3 (Hybrid matching) with full parallelization:
    - Mode 1 (visual) runs in parallel via batch_find_matches()
    - Mode 2 (metadata) runs in parallel via batch_find_metadata_matches()
    - Results are merged in parallel using ThreadPoolExecutor

    PERFORMANCE OPTIMIZATIONS:
    - Parallel Mode 1 + Mode 2 execution (independent operations)
    - Parallel merge of results across all products
    - Minimal code - delegates to existing optimized functions

    Args:
        product_ids: List of product IDs to match
        threshold: Minimum similarity score (0-100)
        limit: Maximum matches per product
        visual_weight: Weight for visual similarity (default: 0.50)
        metadata_weight: Weight for metadata similarity (default: 0.50)
        metadata_weights: Dynamic weights dict for metadata columns
        store_matches: Store results in database
        skip_invalid_products: Continue on errors
        match_against_all: Match against all categories
        max_workers: Number of parallel workers (default: cpu_count + 4)

    Returns:
        Dictionary with results and summary
    """
    from product_matching import batch_find_matches, batch_find_metadata_matches
    import os
    import time
    
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    logger.info(f"[BATCH-HYBRID] ▶ Starting batch hybrid matching for {len(product_ids)} products")
    logger.debug(f"[BATCH-HYBRID] Workers: {max_workers}, Visual weight: {visual_weight*100}%, Metadata weight: {metadata_weight*100}%")
    
    start_time = time.time()
    
    # Step 1 & 2: Run Mode 1 (visual) and Mode 2 (metadata) SIMULTANEOUSLY
    # This is faster because they use different resources:
    # - Mode 1: GPU/FAISS (I/O bound)
    # - Mode 2: CPU/database (compute bound)
    logger.info(f"[BATCH-HYBRID] ▶ Starting Mode 1 (Visual) and Mode 2 (Metadata) SIMULTANEOUSLY")
    
    from concurrent.futures import ThreadPoolExecutor as TPE, as_completed
    
    mode1_time = 0
    mode2_time = 0
    visual_results = None
    metadata_results = None
    
    def run_mode1():
        """Run Mode 1 (visual) matching"""
        nonlocal mode1_time, visual_results
        logger.info(f"[BATCH-HYBRID] [MODE 1] ▶ Starting parallel visual matching for {len(product_ids)} products...")
        mode1_start = time.time()
        
        visual_results = batch_find_matches(
            product_ids=product_ids,
            threshold=0.0,
            limit=limit * 10 if limit > 0 else 1000,
            match_against_all=match_against_all,
            include_uncategorized=True,
            store_matches=False,
            skip_invalid_products=skip_invalid_products,
            max_workers=max_workers
        )
        
        mode1_time = time.time() - mode1_start
        logger.info(f"[BATCH-HYBRID] [MODE 1] ✓ Completed in {mode1_time:.2f}s - {visual_results['summary']['successful']} successful, {visual_results['summary']['failed']} failed")
        return visual_results
    
    def run_mode2():
        """Run Mode 2 (metadata) matching"""
        nonlocal mode2_time, metadata_results
        logger.info(f"[BATCH-HYBRID] [MODE 2] ▶ Starting parallel metadata matching for {len(product_ids)} products...")
        mode2_start = time.time()

        # Use dynamic weights
        metadata_results = batch_find_metadata_matches(
            product_ids=product_ids,
            threshold=0.0,
            limit=limit * 10 if limit > 0 else 1000,
            weights=metadata_weights,
            store_matches=False,
            skip_invalid_products=skip_invalid_products,
            match_against_all=match_against_all,
            max_workers=max_workers
        )

        mode2_time = time.time() - mode2_start
        logger.info(f"[BATCH-HYBRID] [MODE 2] ✓ Completed in {mode2_time:.2f}s - {metadata_results['summary']['successful']} successful, {metadata_results['summary']['failed']} failed")
        return metadata_results
    
    # Run both modes simultaneously using threads
    with TPE(max_workers=2) as executor:
        future_mode1 = executor.submit(run_mode1)
        future_mode2 = executor.submit(run_mode2)
        
        # Wait for both to complete
        visual_results = future_mode1.result()
        metadata_results = future_mode2.result()
    
    logger.info(f"[BATCH-HYBRID] ✓ Both modes completed! Mode 1: {mode1_time:.2f}s, Mode 2: {mode2_time:.2f}s (ran simultaneously)")
    
    # Step 3: Merge results in parallel
    logger.info(f"[BATCH-HYBRID] [MERGE] ▶ Starting parallel merge of {len(product_ids)} products...")
    merge_start = time.time()
    
    # Build lookup dictionaries for fast access
    visual_lookup = {r['product_id']: r for r in visual_results['results']}
    metadata_lookup = {r['product_id']: r for r in metadata_results['results']}
    logger.debug(f"[BATCH-HYBRID] [MERGE] Built lookup tables - Visual: {len(visual_lookup)}, Metadata: {len(metadata_lookup)}")
    
    def merge_product_results(product_id: int) -> Dict[str, Any]:
        """Merge visual and metadata results for a single product"""
        try:
            visual_result = visual_lookup.get(product_id)
            metadata_result = metadata_lookup.get(product_id)
            
            # Check if both modes succeeded
            if not visual_result or visual_result['status'] != 'success':
                return {
                    'product_id': product_id,
                    'status': 'failed',
                    'error': 'Visual matching failed',
                    'error_code': 'VISUAL_FAILED'
                }
            
            if not metadata_result or metadata_result['status'] != 'success':
                return {
                    'product_id': product_id,
                    'status': 'failed',
                    'error': 'Metadata matching failed',
                    'error_code': 'METADATA_FAILED'
                }
            
            # Build lookup dictionaries for matches
            visual_matches_lookup = {m['product_id']: m for m in visual_result['matches']}
            metadata_matches_lookup = {m['product_id']: m for m in metadata_result['matches']}
            
            # Get all unique candidate IDs
            all_candidate_ids = set(visual_matches_lookup.keys()) | set(metadata_matches_lookup.keys())
            
            # Compute hybrid scores with smart linking
            hybrid_matches = []
            for candidate_id in all_candidate_ids:
                visual_match = visual_matches_lookup.get(candidate_id)
                metadata_match = metadata_matches_lookup.get(candidate_id)

                # Get scores (default to 0 if match not found in one mode)
                visual_score = visual_match['similarity_score'] if visual_match else 0.0
                metadata_score = metadata_match['similarity_score'] if metadata_match else 0.0

                # SMART LINKING: If both found matches but to different products,
                # check if they're actually the same product by filename ↔ SKU
                if visual_match and metadata_match and visual_score > 0 and metadata_score > 0:
                    if visual_match.get('product_id') != metadata_match.get('product_id'):
                        # Different product_ids - verify they represent same item
                        if not matches_by_filename_sku(visual_match, metadata_match):
                            # Not the same product - don't merge these scores
                            continue

                # Compute hybrid score
                hybrid_score = (visual_score * visual_weight) + (metadata_score * metadata_weight)

                # Use visual match data as base (has image_path, etc.)
                if visual_match:
                    match_data = visual_match.copy()
                elif metadata_match:
                    match_data = metadata_match.copy()
                else:
                    continue
                
                # Update with hybrid scores
                match_data['similarity_score'] = hybrid_score
                match_data['visual_score'] = visual_score
                match_data['metadata_score'] = metadata_score
                match_data['is_potential_duplicate'] = hybrid_score > 90
                
                # Add metadata sub-scores and values
                if metadata_match:
                    # Pass through dynamic metadata structures
                    match_data['metadata_scores'] = metadata_match.get('metadata_scores', {})
                    match_data['metadata_values'] = metadata_match.get('metadata_values', {})
                    
                    # Map common fields to legacy top-level keys for compatibility
                    # (These are used by frontend for specific score bars)
                    ms_scores = match_data['metadata_scores']
                    match_data['sku_score'] = ms_scores.get('sku', 0.0)
                    match_data['name_score'] = ms_scores.get('name', 0.0)
                    match_data['category_score'] = ms_scores.get('category', 0.0)
                    match_data['price_score'] = ms_scores.get('price', 0.0)
                    match_data['performance_score'] = ms_scores.get('performance', 0.0)
                
                if not match_data.get('metadata_values') and visual_match and 'metadata_values' in visual_match:
                     # Fallback: if visual match somehow has it (unlikely but possible if enriched later)
                     match_data['metadata_values'] = visual_match['metadata_values']
                
                hybrid_matches.append(match_data)
            
            # Sort and filter
            hybrid_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            filtered_count = 0
            if threshold > 0:
                original_count = len(hybrid_matches)
                hybrid_matches = [m for m in hybrid_matches if m['similarity_score'] >= threshold]
                filtered_count = original_count - len(hybrid_matches)
            
            if limit > 0:
                hybrid_matches = hybrid_matches[:limit]
            
            # Collect matches for batch insertion (don't insert yet)
            # Return matches for later batch insertion
            matches_to_return = []
            for match in hybrid_matches:
                matches_to_return.append({
                    'product_id': product_id,
                    'matched_product_id': match['product_id'],
                    'similarity_score': match['similarity_score'],
                    'color_score': match.get('color_score', match.get('visual_score', 0.0)),
                    'shape_score': match.get('shape_score', match.get('visual_score', 0.0)),
                    'texture_score': match.get('texture_score', match.get('visual_score', 0.0))
                })
            
            return {
                'product_id': product_id,
                'status': 'success',
                'match_count': len(hybrid_matches),
                'matches': hybrid_matches,
                'filtered_by_threshold': filtered_count
            }
            
        except Exception as e:
            logger.error(f"Error merging results for product {product_id}: {e}")
            return {
                'product_id': product_id,
                'status': 'failed',
                'error': str(e),
                'error_code': 'MERGE_ERROR'
            }
    
    # Parallel merge - Insert matches incrementally while merging
    results = []
    successful = 0
    failed = 0
    all_matches_to_insert = []
    total_matches_inserted = 0  # Track total for logging (includes incremental + batch)
    total_matches_generated = 0

    # Fetch all query product data upfront for results display
    from database import get_products_by_ids
    query_products_data = get_products_by_ids(product_ids)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for pid in product_ids:
            # Create a modified merge function that doesn't store matches
            def merge_without_store(product_id):
                try:
                    visual_result = visual_lookup.get(product_id)
                    metadata_result = metadata_lookup.get(product_id)
                    
                    # Check if both modes succeeded
                    if not visual_result or visual_result['status'] != 'success':
                        return {
                            'product_id': product_id,
                            'status': 'failed',
                            'error': 'Visual matching failed',
                            'error_code': 'VISUAL_FAILED'
                        }
                
                    if not metadata_result or metadata_result['status'] != 'success':
                        return {
                            'product_id': product_id,
                            'status': 'failed',
                            'error': 'Metadata matching failed',
                            'error_code': 'METADATA_FAILED'
                        }
                    
                    # Build lookup dictionaries for matches
                    visual_matches_lookup = {m['product_id']: m for m in visual_result['matches']}
                    metadata_matches_lookup = {m['product_id']: m for m in metadata_result['matches']}
                    
                    # Get all unique candidate IDs
                    all_candidate_ids = set(visual_matches_lookup.keys()) | set(metadata_matches_lookup.keys())
                    
                    # Compute hybrid scores with smart linking
                    hybrid_matches = []
                    for candidate_id in all_candidate_ids:
                        visual_match = visual_matches_lookup.get(candidate_id)
                        metadata_match = metadata_matches_lookup.get(candidate_id)

                        # Get scores (default to 0 if match not found in one mode)
                        visual_score = visual_match['similarity_score'] if visual_match else 0.0
                        metadata_score = metadata_match['similarity_score'] if metadata_match else 0.0

                        # SMART LINKING: If both found matches but to different products,
                        # check if they're actually the same product by filename ↔ SKU
                        if visual_match and metadata_match and visual_score > 0 and metadata_score > 0:
                            if visual_match.get('product_id') != metadata_match.get('product_id'):
                                # Different product_ids - verify they represent same item
                                if not matches_by_filename_sku(visual_match, metadata_match):
                                    # Not the same product - don't merge these scores
                                    continue

                        # Compute hybrid score
                        hybrid_score = (visual_score * visual_weight) + (metadata_score * metadata_weight)
                        
                        # Use visual match data as base (has image_path, etc.)
                        if visual_match:
                            match_data = visual_match.copy()
                        elif metadata_match:
                            match_data = metadata_match.copy()
                        else:
                            continue
                        
                        # Update with hybrid scores
                        match_data['similarity_score'] = hybrid_score
                        match_data['visual_score'] = visual_score
                        match_data['metadata_score'] = metadata_score
                        match_data['is_potential_duplicate'] = hybrid_score > 90
                        
                        # Add metadata sub-scores and values
                        if metadata_match:
                            # Pass through dynamic metadata structures
                            match_data['metadata_scores'] = metadata_match.get('metadata_scores', {})
                            
                            # Ensure metadata values are passed
                            if 'metadata_values' in metadata_match:
                                match_data['metadata_values'] = metadata_match['metadata_values']
                            
                            # Map common fields to legacy top-level keys for compatibility
                            ms_scores = match_data.get('metadata_scores', {})
                            match_data['sku_score'] = ms_scores.get('sku', 0.0)
                            match_data['name_score'] = ms_scores.get('name', 0.0)
                            match_data['category_score'] = ms_scores.get('category', 0.0)
                            match_data['price_score'] = ms_scores.get('price', 0.0)
                            match_data['performance_score'] = ms_scores.get('performance', 0.0)

                        if not match_data.get('metadata_values') and visual_match and 'metadata_values' in visual_match:
                             # Fallback: if visual match somehow has it (unlikely but possible if enriched later)
                             match_data['metadata_values'] = visual_match['metadata_values']
                        
                        hybrid_matches.append(match_data)
                    
                    # Sort and filter
                    hybrid_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
                    
                    filtered_count = 0
                    if threshold > 0:
                        original_count = len(hybrid_matches)
                        hybrid_matches = [m for m in hybrid_matches if m['similarity_score'] >= threshold]
                        filtered_count = original_count - len(hybrid_matches)
                    
                    if limit > 0:
                        hybrid_matches = hybrid_matches[:limit]
                    
                    from product_matching import calculate_summary_stats
                    return {
                        'product_id': product_id,
                        'product_data': query_products_data.get(product_id, {}), # ADDED
                        'status': 'success',
                        'match_count': len(hybrid_matches),
                        'matches': hybrid_matches,
                        'summary_stats': calculate_summary_stats(hybrid_matches), # ADDED
                        'filtered_by_threshold': filtered_count
                    }
                    
                except Exception as e:
                    logger.error(f"Error merging results for product {product_id}: {e}")
                    return {
                        'product_id': product_id,
                        'status': 'failed',
                        'error': str(e),
                        'error_code': 'MERGE_ERROR'
                    }
            
            future = executor.submit(merge_without_store, pid)
            futures[future] = pid
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            # MEMORY OPTIMIZATION: Incremental insertion to prevent 50-100MB accumulation in Mode 3
            # Insert matches immediately instead of accumulating all in memory
            if result['status'] == 'success' and result['matches']:
                # Collect matches for this product
                product_matches = []
                for match in result['matches']:
                    product_matches.append((
                        result['product_id'],
                        match['product_id'],
                        match['similarity_score'],
                        match.get('color_score', match.get('visual_score', 0.0)),
                        match.get('shape_score', match.get('visual_score', 0.0)),
                        match.get('texture_score', match.get('visual_score', 0.0))
                    ))
                total_matches_generated += len(product_matches)

                # OPTIMIZATION: Insert incrementally while other workers are still merging
                # This starts inserting while other workers are still running, freeing memory as we go
                if store_matches and product_matches:
                    try:
                        from database import bulk_insert_matches
                        inserted = bulk_insert_matches(product_matches)
                        total_matches_inserted += inserted
                        logger.debug(f"[BATCH-HYBRID] [INSERT] ▶ Incremental insert: {inserted} matches for product {result['product_id']}")
                    except Exception as e:
                        logger.warning(f"[BATCH-HYBRID] [INSERT] Incremental insert failed for {result['product_id']}: {e}, will retry at end")
                        all_matches_to_insert.extend(product_matches)  # Fallback to end insert if immediate insert fails

            if result['status'] == 'success':
                successful += 1
                logger.debug(f"[BATCH-HYBRID] [MERGE] Product {result['product_id']}: {result['match_count']} matches")
            else:
                failed += 1
                logger.debug(f"[BATCH-HYBRID] [MERGE] Product {result['product_id']}: FAILED - {result.get('error', 'Unknown error')}")

            # Log progress every 10 products (debug-only to reduce normal log noise)
            if i % 10 == 0:
                logger.debug(f"[BATCH-HYBRID] [MERGE] Progress: {i}/{len(product_ids)} merged ({successful} successful, {failed} failed)")
    
    # PERFORMANCE OPTIMIZATION: Insert any remaining matches in chunks
    # (Most matches were already inserted incrementally, this handles any failed insertions)
    # Smaller chunks = less memory overhead + faster insertion
    remaining_matches_count = len(all_matches_to_insert)
    if remaining_matches_count > 0:
        logger.info(f"[BATCH-HYBRID] [INSERT] Step 2: Batch insert {remaining_matches_count} remaining matches (from failed incremental inserts)")

    if store_matches and all_matches_to_insert:
        try:
            from database import bulk_insert_matches

            # Chunk size: 100 matches per transaction (smaller = faster + less memory)
            # This allows DB insertion to start while other workers still merging
            CHUNK_SIZE = 100

            num_chunks = (len(all_matches_to_insert) + CHUNK_SIZE - 1) // CHUNK_SIZE

            if num_chunks == 1:
                # Small batch - insert all at once
                logger.info(f"[BATCH-HYBRID] [INSERT] ▶ Batch inserting {len(all_matches_to_insert)} remaining matches in one transaction...")
                inserted_count = bulk_insert_matches(all_matches_to_insert)
                total_matches_inserted += inserted_count
                logger.info(f"[BATCH-HYBRID] [INSERT] ✓ Batch inserted {inserted_count} remaining matches")
            else:
                # Large batch - chunk into multiple transactions (smaller chunks = faster)
                logger.info(f"[BATCH-HYBRID] [INSERT] ▶ Batch inserting {len(all_matches_to_insert)} remaining matches in {num_chunks} chunks ({CHUNK_SIZE} per chunk)...")

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * CHUNK_SIZE
                    end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(all_matches_to_insert))
                    # MEMORY OPTIMIZATION: Create slice and immediately delete to avoid 2x memory during processing
                    chunk = all_matches_to_insert[start_idx:end_idx]

                    inserted_count = bulk_insert_matches(chunk)
                    total_matches_inserted += inserted_count

                    # Clear chunk reference immediately to free memory
                    chunk = None

                    logger.debug(f"[BATCH-HYBRID] [INSERT] Chunk {chunk_idx + 1}/{num_chunks}: Inserted {inserted_count} remaining matches")

                logger.info(f"[BATCH-HYBRID] [INSERT] ✓ Batch inserted {remaining_matches_count} remaining matches in {num_chunks} transactions")
        except Exception as e:
            logger.error(f"Failed to batch insert hybrid matches: {e}")
    
    merge_time = time.time() - merge_start
    logger.info(f"[BATCH-HYBRID] [MERGE] ✓ Completed in {merge_time:.2f}s - {successful} successful, {failed} failed")

    summary = {
        'total_products': len(product_ids),
        'successful': successful,
        'failed': failed,
        'success_rate': round(successful / len(product_ids) * 100, 1) if product_ids else 0,
        'visual_weight': visual_weight,
        'metadata_weight': metadata_weight,
        'total_matches': total_matches_generated
    }

    total_time = time.time() - start_time
    logger.info(f"[BATCH-HYBRID] ✓ COMPLETE! Total time: {total_time:.2f}s")
    logger.debug(f"[BATCH-HYBRID] Timing breakdown:")
    logger.debug(f"[BATCH-HYBRID]   - Mode 1 (Visual):   {mode1_time:.2f}s")
    logger.debug(f"[BATCH-HYBRID]   - Mode 2 (Metadata): {mode2_time:.2f}s")
    logger.debug(f"[BATCH-HYBRID]   - Merge:             {merge_time:.2f}s")
    logger.info(f"[BATCH-HYBRID] Results: {successful}/{len(product_ids)} successful ({summary['success_rate']}%)")
    if store_matches:
        logger.info(f"[BATCH-HYBRID] Total matches stored: {total_matches_inserted} (inserted incrementally while merging + batch insert for remaining)")
    else:
        logger.debug(f"[BATCH-HYBRID] Total matches generated (not stored): {total_matches_generated}")
    
    return {
        'results': results,
        'summary': summary,
        'errors': [r for r in results if r['status'] == 'failed']
    }
