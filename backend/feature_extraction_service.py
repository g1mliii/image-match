"""
Feature Extraction Service

This module provides CLIP-based feature extraction for product matching.
CLIP works on both GPU (fast) and CPU (slower but still accurate).

No legacy fallback - CLIP only for simplicity and better accuracy.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import CLIP (required)
try:
    from image_processing_clip import (
        extract_clip_embedding,
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
    """Extract features using CLIP (GPU or CPU)
    
    This is the main entry point for feature extraction.
    Always uses CLIP - no legacy fallback.
    
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
    
    # Always use CLIP
    return _extract_clip_features(image_path)


def _extract_clip_features(image_path: str) -> Tuple[Dict[str, np.ndarray], str, str]:
    """Extract CLIP embeddings
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (features_dict, 'clip', model_version)
    
    Raises:
        CLIPModelError: If CLIP extraction fails
        ImageProcessingError: If image is invalid
    """
    try:
        # Load config to get model name
        config = load_clip_config()
        model_name = config.get('model_name', 'clip-ViT-B-32')
        
        # Extract CLIP embedding
        logger.info(f"Extracting CLIP embedding from {image_path} using {model_name}")
        clip_embedding = extract_clip_embedding(
            image_path,
            model_name=model_name,
            normalize=True,  # Pre-normalize for faster similarity computation
            use_amp=True     # Use automatic mixed precision for faster GPU inference
        )
        
        # CLIP embeddings are stored in color_features column for backward compatibility
        # shape_features and texture_features are set to empty arrays
        features_dict = {
            'color_features': clip_embedding,
            'shape_features': np.array([], dtype=np.float32),
            'texture_features': np.array([], dtype=np.float32)
        }
        
        logger.info(f"CLIP embedding extracted successfully: {len(clip_embedding)} dimensions")
        return features_dict, 'clip', model_name
    
    except (CLIPModelError, CLIPModelDownloadError) as e:
        logger.error(f"CLIP extraction failed: {e.message}")
        raise
    
    except (InvalidImageFormatError, CorruptedImageError, ImageTooSmallError, ImageProcessingFailedError):
        # Re-raise image errors (these are not CLIP-specific)
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during CLIP extraction: {e}", exc_info=True)
        raise ImageProcessingFailedError(
            f"CLIP extraction failed: {str(e)}",
            "Check CLIP setup and ensure PyTorch is installed correctly"
        )


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
