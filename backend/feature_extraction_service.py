"""
Feature Extraction Service

This module provides CLIP-based feature extraction for product matching.
CLIP works on both GPU (fast) and CPU (slower but still accurate).

All extraction now uses batch processing internally for consistency and performance.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import CLIP (required)
try:
    from image_processing_clip import (
        batch_extract_clip_embeddings,
        is_clip_available,
        load_clip_config,
        CLIPModelError,
        CLIPModelDownloadError
    )
    CLIP_AVAILABLE = is_clip_available()
except ImportError:
    CLIP_AVAILABLE = False
    logger.error("CLIP not available - install PyTorch and sentence-transformers")

# Import error types for compatibility
from image_processing import (
    ImageProcessingError,
    InvalidImageFormatError,
    CorruptedImageError,
    ImageTooSmallError,
    ImageProcessingFailedError
)


def extract_features_unified(image_path: str) -> Tuple[Dict[str, np.ndarray], str, Optional[str]]:
    """Extract features using CLIP (GPU or CPU) - delegates to batch extraction
    
    This function delegates to batch_extract_clip_embeddings() for consistency.
    For multiple images, use batch_extract_features_unified() instead for better performance.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (features_dict, embedding_type, embedding_version)
        - features_dict: Dictionary with CLIP embedding
        - embedding_type: Always 'clip'
        - embedding_version: Model version string (e.g., 'clip-ViT-B-32')
    
    Raises:
        CLIPModelError: If CLIP is not available
        ImageProcessingError: For image processing errors
    """
    if not CLIP_AVAILABLE:
        raise CLIPModelError(
            "CLIP is required but not available",
            "Install PyTorch and sentence-transformers: pip install torch sentence-transformers"
        )
    
    logger.info(f"[EXTRACT-SINGLE] ▶ Delegating to batch_extract_clip_embeddings (batch_size=1)")
    logger.info(f"[EXTRACT-SINGLE] This uses GPU batch processing even for single images")
    
    # Use batch extraction even for single image (simpler, consistent, GPU-optimized)
    results = batch_extract_clip_embeddings(
        image_paths=[image_path],
        batch_size=1,
        skip_errors=False,
        use_amp=True,
        auto_adjust_batch=True
    )
    
    if not results or results[0][1] is None:
        error_msg = results[0][2] if results and results[0][2] else "Unknown error"
        raise ImageProcessingFailedError(f"CLIP extraction failed: {error_msg}")
    
    clip_embedding = results[0][1]
    logger.info(f"[EXTRACT-SINGLE] ✓ CLIP embedding extracted: {len(clip_embedding)} dimensions")
    
    # Create features dict compatible with database
    features_dict = {
        'color_features': clip_embedding,
        'shape_features': np.array([], dtype=np.float32),
        'texture_features': np.array([], dtype=np.float32)
    }
    
    return features_dict, 'clip', 'clip-ViT-B-32'


def batch_extract_features_unified(image_paths: List[str]) -> List[Tuple[str, Optional[Dict[str, np.ndarray]], Optional[str], Optional[str], Optional[str]]]:
    """Extract features from multiple images using batch CLIP extraction (GPU-optimized)
    
    This is the preferred method for processing multiple images as it:
    - Uses GPU batch processing for maximum throughput
    - Processes all images in parallel on GPU
    - Automatically adjusts batch size based on available memory
    
    Args:
        image_paths: List of paths to image files
    
    Returns:
        List of tuples: (image_path, features_dict, embedding_type, embedding_version, error_msg)
        - image_path: Original path
        - features_dict: Dictionary with CLIP embedding (None if failed)
        - embedding_type: Always 'clip' (or None if failed)
        - embedding_version: Model version string (or None if failed)
        - error_msg: Error message if extraction failed (None if success)
    """
    if not CLIP_AVAILABLE:
        # Return errors for all images
        return [
            (path, None, None, None, "CLIP not available - install PyTorch and sentence-transformers")
            for path in image_paths
        ]
    
    logger.info(f"[BATCH-EXTRACT] Starting batch CLIP extraction for {len(image_paths)} images")
    
    # Extract CLIP embeddings in batch (GPU-optimized)
    results = batch_extract_clip_embeddings(
        image_paths=image_paths,
        batch_size=32,  # Let auto_adjust_batch optimize this
        skip_errors=True,  # Continue processing even if some images fail
        use_amp=True,  # Use automatic mixed precision for faster GPU inference
        auto_adjust_batch=True  # Automatically adjust batch size based on GPU memory
    )
    
    # Convert to unified format
    unified_results = []
    for path, clip_embedding, error_msg in results:
        if clip_embedding is not None:
            # Success - create features dict
            features_dict = {
                'color_features': clip_embedding,
                'shape_features': np.array([], dtype=np.float32),
                'texture_features': np.array([], dtype=np.float32)
            }
            unified_results.append((path, features_dict, 'clip', 'clip-ViT-B-32', None))
        else:
            # Failed - return error
            unified_results.append((path, None, None, None, error_msg or "Unknown error"))
    
    success_count = sum(1 for _, features, _, _, _ in unified_results if features is not None)
    logger.info(f"[BATCH-EXTRACT] Completed: {success_count}/{len(image_paths)} successful")
    
    return unified_results


def get_feature_extraction_info() -> Dict[str, Any]:
    """Get information about current feature extraction configuration
    
    Returns:
        Dictionary with configuration info
    """
    info = {
        'clip_available': CLIP_AVAILABLE,
        'current_method': 'clip'  # Always CLIP now
    }
    
    if CLIP_AVAILABLE:
        try:
            config = load_clip_config()
            info['clip_config'] = {
                'model_name': config.get('model_name', 'clip-ViT-B-32')
            }
        except:
            pass
    
    return info


def validate_features_for_matching(features_dict: Dict[str, Any], 
                                   embedding_type: str) -> Tuple[bool, Optional[str]]:
    """Validate that features are suitable for matching
    
    Args:
        features_dict: Dictionary with feature arrays
        embedding_type: Should always be 'clip'
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Only CLIP embeddings are supported now
    if embedding_type != 'clip':
        return False, f"Unsupported embedding type: {embedding_type}. Only CLIP is supported."
    
    # Validate CLIP embedding
    if 'color_features' not in features_dict:
        return False, "Missing CLIP embedding (color_features)"
    
    embedding = features_dict['color_features']
    
    if not isinstance(embedding, np.ndarray):
        return False, f"CLIP embedding must be numpy array, got {type(embedding)}"
    
    if len(embedding) != 512:
        return False, f"CLIP embedding must be 512-dimensional, got {len(embedding)}"
    
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        return False, "CLIP embedding contains NaN or Inf values"
    
    return True, None
