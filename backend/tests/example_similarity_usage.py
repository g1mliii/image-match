"""
Example usage of similarity computation with error handling.

This demonstrates how to use the similarity module in a real-world scenario
with proper error handling for messy data.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from similarity import (
    compute_all_similarities,
    batch_compute_similarities,
    InvalidFeatureError,
    FeatureDimensionError
)
from image_processing import extract_all_features, ImageProcessingError
from feature_cache import get_feature_cache


def compare_two_products(product1_path: str, product2_path: str):
    """
    Compare two products and return similarity scores.
    
    Demonstrates basic usage with error handling.
    """
    try:
        # Extract features from both images
        print(f"Extracting features from {product1_path}...")
        features1 = extract_all_features(product1_path)
        
        print(f"Extracting features from {product2_path}...")
        features2 = extract_all_features(product2_path)
        
        # Compute similarities
        print("Computing similarities...")
        similarities = compute_all_similarities(features1, features2)
        
        # Display results
        print("\nSimilarity Results:")
        print(f"  Color:    {similarities['color_similarity']:.1f}%")
        print(f"  Shape:    {similarities['shape_similarity']:.1f}%")
        print(f"  Texture:  {similarities['texture_similarity']:.1f}%")
        print(f"  Combined: {similarities['combined_similarity']:.1f}%")
        
        return similarities
    
    except ImageProcessingError as e:
        print(f"\nImage processing error: {e.message}")
        print(f"Error code: {e.error_code}")
        print(f"Suggestion: {e.suggestion}")
        return None
    
    except (InvalidFeatureError, FeatureDimensionError) as e:
        print(f"\nSimilarity computation error: {e.message}")
        print(f"Error code: {e.error_code}")
        print(f"Suggestion: {e.suggestion}")
        return None
    
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return None


def find_similar_products(new_product_id: int, historical_product_ids: list):
    """
    Find similar products from a list of historical products.
    
    Demonstrates batch processing with error handling.
    """
    cache = get_feature_cache()
    
    try:
        # Get features for new product
        print(f"Loading features for new product {new_product_id}...")
        from database import get_product_by_id
        
        new_product = get_product_by_id(new_product_id)
        if not new_product:
            print(f"Product {new_product_id} not found")
            return []
        
        new_features = cache.get_or_extract_features(
            new_product_id,
            new_product['image_path']
        )
        
        # Get features for all historical products
        print(f"Loading features for {len(historical_product_ids)} historical products...")
        historical_features = []
        valid_product_ids = []
        
        for hist_id in historical_product_ids:
            try:
                hist_product = get_product_by_id(hist_id)
                if hist_product:
                    hist_features = cache.get_or_extract_features(
                        hist_id,
                        hist_product['image_path']
                    )
                    historical_features.append(hist_features)
                    valid_product_ids.append(hist_id)
            except Exception as e:
                print(f"  Warning: Failed to load product {hist_id}: {str(e)}")
                continue
        
        print(f"Successfully loaded {len(historical_features)} products")
        
        # Compute similarities in batch
        print("Computing similarities...")
        results = batch_compute_similarities(
            new_features,
            historical_features,
            skip_errors=True  # Continue on errors
        )
        
        # Combine results with product IDs
        matches = []
        for i, result in enumerate(results):
            if 'error' in result:
                print(f"  Warning: Failed to compute similarity for product {valid_product_ids[i]}")
                print(f"    Error: {result['error']}")
            else:
                matches.append({
                    'product_id': valid_product_ids[i],
                    'similarity': result['combined_similarity'],
                    'color_similarity': result['color_similarity'],
                    'shape_similarity': result['shape_similarity'],
                    'texture_similarity': result['texture_similarity']
                })
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Display top matches
        print(f"\nTop {min(5, len(matches))} matches:")
        for i, match in enumerate(matches[:5], 1):
            print(f"  {i}. Product {match['product_id']}: {match['similarity']:.1f}%")
            print(f"     (Color: {match['color_similarity']:.1f}%, "
                  f"Shape: {match['shape_similarity']:.1f}%, "
                  f"Texture: {match['texture_similarity']:.1f}%)")
        
        return matches
    
    except ImageProcessingError as e:
        print(f"\nImage processing error: {e.message}")
        print(f"Suggestion: {e.suggestion}")
        return []
    
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def handle_corrupted_features_example():
    """
    Example of handling corrupted features gracefully.
    
    Demonstrates error handling for real-world data issues.
    """
    print("Example: Handling corrupted features\n")
    
    # Simulate corrupted features
    good_features = {
        'color_features': np.random.rand(256).astype(np.float32),
        'shape_features': np.random.rand(7).astype(np.float32),
        'texture_features': np.random.rand(256).astype(np.float32)
    }
    
    corrupted_features = {
        'color_features': np.array([np.nan] * 256, dtype=np.float32),  # NaN values
        'shape_features': np.random.rand(7).astype(np.float32),
        'texture_features': np.random.rand(256).astype(np.float32)
    }
    
    try:
        print("Attempting to compute similarity with corrupted features...")
        similarities = compute_all_similarities(good_features, corrupted_features)
        print(f"Similarity: {similarities['combined_similarity']:.1f}%")
    
    except InvalidFeatureError as e:
        print(f"✓ Error caught successfully!")
        print(f"  Message: {e.message}")
        print(f"  Code: {e.error_code}")
        print(f"  Suggestion: {e.suggestion}")
        
        # In production, you might:
        # 1. Log the error
        # 2. Mark the product for re-processing
        # 3. Notify administrators
        # 4. Return a default similarity score
        print("\nIn production, we would:")
        print("  1. Log this error for debugging")
        print("  2. Mark product for feature re-extraction")
        print("  3. Continue processing other products")


def batch_processing_with_mixed_quality_example():
    """
    Example of batch processing with mixed quality data.
    
    Demonstrates how batch processing handles errors gracefully.
    """
    print("\nExample: Batch processing with mixed quality data\n")
    
    # Create query features
    query_features = {
        'color_features': np.random.rand(256).astype(np.float32),
        'shape_features': np.random.rand(7).astype(np.float32),
        'texture_features': np.random.rand(256).astype(np.float32)
    }
    
    # Create candidate features with various issues
    candidates = [
        # Good candidate
        {
            'color_features': np.random.rand(256).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        },
        # Corrupted candidate (NaN values)
        {
            'color_features': np.array([np.nan] * 256, dtype=np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        },
        # Good candidate
        {
            'color_features': np.random.rand(256).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        },
        # Wrong dimensions
        {
            'color_features': np.random.rand(100).astype(np.float32),  # Wrong size
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        },
        # Good candidate
        {
            'color_features': np.random.rand(256).astype(np.float32),
            'shape_features': np.random.rand(7).astype(np.float32),
            'texture_features': np.random.rand(256).astype(np.float32)
        }
    ]
    
    print(f"Processing {len(candidates)} candidates...")
    results = batch_compute_similarities(query_features, candidates, skip_errors=True)
    
    # Analyze results
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\n✓ Successfully processed: {len(successful)}/{len(candidates)}")
    print(f"✗ Failed: {len(failed)}/{len(candidates)}")
    
    if failed:
        print("\nFailed candidates:")
        for i, result in enumerate(results):
            if 'error' in result:
                print(f"  Candidate {i}:")
                print(f"    Error: {result['error']}")
                print(f"    Code: {result['error_code']}")
    
    if successful:
        print("\nSuccessful candidates:")
        for i, result in enumerate(results):
            if 'error' not in result:
                print(f"  Candidate {i}: {result['combined_similarity']:.1f}% similarity")


if __name__ == '__main__':
    print("=" * 70)
    print("Similarity Computation - Usage Examples")
    print("=" * 70 + "\n")
    
    # Example 1: Handling corrupted features
    handle_corrupted_features_example()
    
    # Example 2: Batch processing with mixed quality
    batch_processing_with_mixed_quality_example()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
