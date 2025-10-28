import numpy as np
from scipy.spatial.distance import euclidean, cdist

def compute_color_similarity(features1, features2):
    """
    Compute color similarity using histogram intersection.
    Returns: Similarity score 0-100
    """
    intersection = np.minimum(features1, features2).sum()
    similarity = intersection * 100
    return float(similarity)

def compute_shape_similarity(features1, features2):
    """
    Compute shape similarity using Euclidean distance on Hu moments.
    Returns: Similarity score 0-100
    """
    distance = euclidean(features1, features2)
    # Convert distance to similarity (inverse relationship)
    similarity = 100 / (1 + distance)
    return float(similarity)

def compute_texture_similarity(features1, features2):
    """
    Compute texture similarity using chi-square distance on LBP histograms.
    Returns: Similarity score 0-100
    """
    # Chi-square distance
    chi_square = np.sum((features1 - features2) ** 2 / (features1 + features2 + 1e-10))
    # Convert to similarity
    similarity = 100 / (1 + chi_square)
    return float(similarity)

def compute_combined_similarity(color_sim, shape_sim, texture_sim, 
                               color_weight=0.5, shape_weight=0.3, texture_weight=0.2):
    """
    Combine individual similarity scores with weights.
    Default weights: color=0.5, shape=0.3, texture=0.2
    Returns: Combined similarity score 0-100
    """
    combined = (color_sim * color_weight + 
                shape_sim * shape_weight + 
                texture_sim * texture_weight)
    return float(combined)

def find_matches(new_product_features, historical_products_features, 
                threshold=0.0, limit=10):
    """
    Find and rank similar products.
    Returns: List of match results with scores
    """
    matches = []
    
    for hist_id, hist_features in historical_products_features.items():
        color_sim = compute_color_similarity(
            new_product_features['color'], 
            hist_features['color']
        )
        shape_sim = compute_shape_similarity(
            new_product_features['shape'], 
            hist_features['shape']
        )
        texture_sim = compute_texture_similarity(
            new_product_features['texture'], 
            hist_features['texture']
        )
        
        combined_sim = compute_combined_similarity(color_sim, shape_sim, texture_sim)
        
        if combined_sim >= threshold:
            matches.append({
                'product_id': hist_id,
                'similarity_score': combined_sim,
                'color_score': color_sim,
                'shape_score': shape_sim,
                'texture_score': texture_sim,
                'is_potential_duplicate': combined_sim > 90
            })
    
    # Sort by similarity score descending
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Return top N matches
    return matches[:limit]
