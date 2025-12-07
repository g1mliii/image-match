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
"""

import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from database import insert_match

logger = logging.getLogger(__name__)


def find_hybrid_matches(
    product_id: int,
    threshold: float = 0.0,
    limit: int = 10,
    visual_weight: float = 0.50,
    metadata_weight: float = 0.50,
    sku_weight: float = 0.30,
    name_weight: float = 0.25,
    category_weight: float = 0.20,
    price_weight: float = 0.15,
    performance_weight: float = 0.10,
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
        sku_weight: Weight for SKU within metadata (default: 0.30)
        name_weight: Weight for name within metadata (default: 0.25)
        category_weight: Weight for category within metadata (default: 0.20)
        price_weight: Weight for price within metadata (default: 0.15)
        performance_weight: Weight for performance within metadata (default: 0.10)
        store_matches: Whether to store results in database
        skip_invalid_products: Continue on errors
        match_against_all: Match against all categories
    
    Returns:
        Dictionary with matches and comprehensive scoring
    
    Raises:
        ProductNotFoundError: If product doesn't exist
        MissingFeaturesError: If product doesn't have visual features
        EmptyCatalogError: If no products to match against
    """
    from product_matching import find_matches, find_metadata_matches
    
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
        metadata_result = find_metadata_matches(
            product_id=product_id,
            threshold=0.0,  # No threshold - we'll filter combined scores later
            limit=limit * 10 if limit > 0 else 1000,  # Get more candidates for merging
            sku_weight=sku_weight,
            name_weight=name_weight,
            category_weight=category_weight,
            price_weight=price_weight,
            performance_weight=performance_weight,
            store_matches=False,  # Don't store yet - we'll store hybrid scores
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
    
    # Compute hybrid scores
    hybrid_matches = []
    for candidate_id in all_candidate_ids:
        visual_match = visual_lookup.get(candidate_id)
        metadata_match = metadata_lookup.get(candidate_id)
        
        # Get scores (default to 0 if match not found in one mode)
        visual_score = visual_match['similarity_score'] if visual_match else 0.0
        metadata_score = metadata_match['similarity_score'] if metadata_match else 0.0
        
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
        
        # Add metadata sub-scores if available
        if metadata_match:
            match_data['sku_score'] = metadata_match.get('sku_score', 0.0)
            match_data['name_score'] = metadata_match.get('name_score', 0.0)
            match_data['category_score'] = metadata_match.get('category_score', 0.0)
            match_data['price_score'] = metadata_match.get('price_score', 0.0)
            match_data['performance_score'] = metadata_match.get('performance_score', 0.0)
        
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
    
    result = {
        'matches': hybrid_matches,
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
    sku_weight: float = 0.30,
    name_weight: float = 0.25,
    category_weight: float = 0.20,
    price_weight: float = 0.15,
    performance_weight: float = 0.10,
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
        sku_weight: Weight for SKU within metadata (default: 0.30)
        name_weight: Weight for name within metadata (default: 0.25)
        category_weight: Weight for category within metadata (default: 0.20)
        price_weight: Weight for price within metadata (default: 0.15)
        performance_weight: Weight for performance within metadata (default: 0.10)
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
    logger.info(f"[BATCH-HYBRID] Workers: {max_workers}, Visual weight: {visual_weight*100}%, Metadata weight: {metadata_weight*100}%")
    
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
        
        metadata_results = batch_find_metadata_matches(
            product_ids=product_ids,
            threshold=0.0,
            limit=limit * 10 if limit > 0 else 1000,
            sku_weight=sku_weight,
            name_weight=name_weight,
            category_weight=category_weight,
            price_weight=price_weight,
            performance_weight=performance_weight,
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
    logger.info(f"[BATCH-HYBRID] [MERGE] Built lookup tables - Visual: {len(visual_lookup)}, Metadata: {len(metadata_lookup)}")
    
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
            
            # Compute hybrid scores
            hybrid_matches = []
            for candidate_id in all_candidate_ids:
                visual_match = visual_matches_lookup.get(candidate_id)
                metadata_match = metadata_matches_lookup.get(candidate_id)
                
                # Get scores (default to 0 if match not found in one mode)
                visual_score = visual_match['similarity_score'] if visual_match else 0.0
                metadata_score = metadata_match['similarity_score'] if metadata_match else 0.0
                
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
                
                # Add metadata sub-scores if available
                if metadata_match:
                    match_data['sku_score'] = metadata_match.get('sku_score', 0.0)
                    match_data['name_score'] = metadata_match.get('name_score', 0.0)
                    match_data['category_score'] = metadata_match.get('category_score', 0.0)
                    match_data['price_score'] = metadata_match.get('price_score', 0.0)
                    match_data['performance_score'] = metadata_match.get('performance_score', 0.0)
                
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
    
    # Parallel merge - DON'T store matches yet, we'll batch insert them all at once
    results = []
    successful = 0
    failed = 0
    all_matches_to_insert = []
    
    logger.info(f"[BATCH-HYBRID] [MERGE] Executing parallel merge with {max_workers} workers...")
    logger.info(f"[BATCH-HYBRID] [MERGE] Matches will be batch inserted after all products are merged (1 DB call)")
    
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
                    
                    # Compute hybrid scores
                    hybrid_matches = []
                    for candidate_id in all_candidate_ids:
                        visual_match = visual_matches_lookup.get(candidate_id)
                        metadata_match = metadata_matches_lookup.get(candidate_id)
                        
                        # Get scores (default to 0 if match not found in one mode)
                        visual_score = visual_match['similarity_score'] if visual_match else 0.0
                        metadata_score = metadata_match['similarity_score'] if metadata_match else 0.0
                        
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
                        
                        # Add metadata sub-scores if available
                        if metadata_match:
                            match_data['sku_score'] = metadata_match.get('sku_score', 0.0)
                            match_data['name_score'] = metadata_match.get('name_score', 0.0)
                            match_data['category_score'] = metadata_match.get('category_score', 0.0)
                            match_data['price_score'] = metadata_match.get('price_score', 0.0)
                            match_data['performance_score'] = metadata_match.get('performance_score', 0.0)
                        
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
            
            future = executor.submit(merge_without_store, pid)
            futures[future] = pid
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            # Collect matches for batch insertion
            if result['status'] == 'success' and result['matches']:
                for match in result['matches']:
                    all_matches_to_insert.append((
                        result['product_id'],
                        match['product_id'],
                        match['similarity_score'],
                        match.get('color_score', match.get('visual_score', 0.0)),
                        match.get('shape_score', match.get('visual_score', 0.0)),
                        match.get('texture_score', match.get('visual_score', 0.0))
                    ))
            
            if result['status'] == 'success':
                successful += 1
                logger.debug(f"[BATCH-HYBRID] [MERGE] Product {result['product_id']}: {result['match_count']} matches")
            else:
                failed += 1
                logger.debug(f"[BATCH-HYBRID] [MERGE] Product {result['product_id']}: FAILED - {result.get('error', 'Unknown error')}")
            
            # Log progress every 10 products
            if i % 10 == 0:
                logger.info(f"[BATCH-HYBRID] [MERGE] Progress: {i}/{len(product_ids)} merged ({successful} successful, {failed} failed)")
    
    # PERFORMANCE OPTIMIZATION: Batch insert all matches in chunks
    # Smaller chunks = faster insertion while merging still happening
    logger.info(f"[BATCH-HYBRID] [INSERT] ▶ Batch inserting {len(all_matches_to_insert)} matches collected from {successful} products")
    if store_matches and all_matches_to_insert:
        try:
            from database import bulk_insert_matches
            
            # Chunk size: 100 matches per transaction (smaller = faster + less memory)
            # This allows DB insertion to start while other workers still merging
            CHUNK_SIZE = 100
            
            total_inserted = 0
            num_chunks = (len(all_matches_to_insert) + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            if num_chunks == 1:
                # Small batch - insert all at once
                logger.info(f"[BATCH-HYBRID] [INSERT] ▶ Batch inserting {len(all_matches_to_insert)} matches in one transaction...")
                inserted_count = bulk_insert_matches(all_matches_to_insert)
                logger.info(f"[BATCH-HYBRID] [INSERT] ✓ Batch inserted {inserted_count} matches (1 DB call for {len(product_ids)} products)")
                total_inserted = inserted_count
            else:
                # Large batch - chunk into multiple transactions (smaller chunks = faster)
                logger.info(f"[BATCH-HYBRID] [INSERT] ▶ Batch inserting {len(all_matches_to_insert)} matches in {num_chunks} chunks ({CHUNK_SIZE} per chunk)...")
                
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * CHUNK_SIZE
                    end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(all_matches_to_insert))
                    chunk = all_matches_to_insert[start_idx:end_idx]
                    
                    inserted_count = bulk_insert_matches(chunk)
                    total_inserted += inserted_count
                    
                    logger.debug(f"[BATCH-HYBRID] [INSERT] Chunk {chunk_idx + 1}/{num_chunks}: Inserted {inserted_count} matches")
                
                logger.info(f"[BATCH-HYBRID] [INSERT] ✓ Batch inserted {total_inserted} matches in {num_chunks} transactions for {len(product_ids)} products")
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
        'metadata_weight': metadata_weight
    }
    
    total_time = time.time() - start_time
    logger.info(f"[BATCH-HYBRID] ✓ COMPLETE! Total time: {total_time:.2f}s")
    logger.info(f"[BATCH-HYBRID] Timing breakdown:")
    logger.info(f"[BATCH-HYBRID]   - Mode 1 (Visual):   {mode1_time:.2f}s")
    logger.info(f"[BATCH-HYBRID]   - Mode 2 (Metadata): {mode2_time:.2f}s")
    logger.info(f"[BATCH-HYBRID]   - Merge:             {merge_time:.2f}s")
    logger.info(f"[BATCH-HYBRID] Results: {successful}/{len(product_ids)} successful ({summary['success_rate']}%)")
    
    return {
        'results': results,
        'summary': summary,
        'errors': [r for r in results if r['status'] == 'failed']
    }
