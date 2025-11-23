"""
Feature Extraction Service

This module provides a unified interface for feature extraction that supports
both CLIP embeddings and legacy features (color/shape/texture).

It automatically detects which method to use based on configuration and
availability of CLIP dependencies.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import CLIP
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
    logger.info("CLIP not available - using legacy features only")

# Import legacy feature extraction
from image_processing import (
    extract_all_features,
    ImageProcessingError,
    InvalidImageFormatError,
    CorruptedImageError,
    ImageTooSmallError,
    ImageProcessingFailedError
)


def should_use_clip() -> bool:
    """Determine if CLIP should be used for feature extraction
    
    Returns:
        True if CLIP is available and enabled in config
    """
    if not CLIP_AVAILABLE:
        return False
    
    try:
        config = load_clip_config()
        return config.get('use_clip', True)
    except:
        return False


def extract_features_unified(image_path: str, 
                            force_clip: bool = False,
                            force_legacy: bool = False) -> Tuple[Dict[str, np.ndarray], str, Optional[str]]:
    """Extract features using CLIP or legacy method
    
    This is the main entry point for feature extraction. It automatically
    selects the best method based on configuration and availability.
    
    Args:
        image_path: Path to image file
        force_clip: Force use of CLIP (raises error if unavailable)
        force_legacy: Force use of legacy features
    
    Returns:
        Tuple of (features_dict, embedding_type, embedding_version)
        - features_dict: Dictionary with feature arrays
        - embedding_type: 'clip' or 'legacy'
        - embedding_version: Model version string (e.g., 'clip-ViT-B-32') or None
    
    Raises:
        ImageProcessingError subclasses for various error conditions
    """
    # Determine which method to use
    use_clip = False
    
    if force_legacy:
        use_clip = False
    elif force_clip:
        if not CLIP_AVAILABLE:
            raise CLIPModelError(
                "CLIP forced but not available",
                "Install PyTorch and sentence-transformers: pip install torch sentence-transformers"
            )
        use_clip = True
    else:
        use_clip = should_use_clip()
    
    # Extract features
    if use_clip:
        return _extract_clip_features(image_path)
    else:
        return _extract_legacy_features(image_path)


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
        
        # Check if fallback is enabled
        config = load_clip_config()
        if config.get('fallback_to_legacy', True):
            logger.warning("Falling back to legacy features")
            return _extract_legacy_features(image_path)
        else:
            raise
    
    except (InvalidImageFormatError, CorruptedImageError, ImageTooSmallError, ImageProcessingFailedError):
        # Re-raise image errors (these are not CLIP-specific)
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during CLIP extraction: {e}", exc_info=True)
        
        # Check if fallback is enabled
        try:
            config = load_clip_config()
            if config.get('fallback_to_legacy', True):
                logger.warning("Falling back to legacy features due to unexpected error")
                return _extract_legacy_features(image_path)
        except:
            pass
        
        raise ImageProcessingFailedError(
            f"CLIP extraction failed: {str(e)}",
            "Try using legacy features or check CLIP setup"
        )


def _extract_legacy_features(image_path: str) -> Tuple[Dict[str, np.ndarray], str, None]:
    """Extract legacy features (color/shape/texture)
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (features_dict, 'legacy', None)
    
    Raises:
        ImageProcessingError: If feature extraction fails
    """
    logger.info(f"Extracting legacy features from {image_path}")
    features = extract_all_features(image_path)
    
    logger.info("Legacy features extracted successfully")
    return features, 'legacy', None


def get_feature_extraction_info() -> Dict[str, Any]:
    """Get information about current feature extraction configuration
    
    Returns:
        Dictionary with configuration info
    """
    info = {
        'clip_available': CLIP_AVAILABLE,
        'current_method': 'clip' if should_use_clip() else 'legacy'
    }
    
    if CLIP_AVAILABLE:
        try:
            config = load_clip_config()
            info['clip_config'] = {
                'enabled': config.get('use_clip', True),
                'model_name': config.get('model_name', 'clip-ViT-B-32'),
                'fallback_to_legacy': config.get('fallback_to_legacy', True)
            }
        except:
            pass
    
    return info


def validate_features_for_matching(features_dict: Dict[str, Any], 
                                   embedding_type: str) -> Tuple[bool, Optional[str]]:
    """Validate that features are suitable for matching
    
    Args:
        features_dict: Dictionary with feature arrays
        embedding_type: 'clip' or 'legacy'
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if embedding_type == 'clip':
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
    
    elif embedding_type == 'legacy':
        # Validate legacy features
        required_keys = ['color_features', 'shape_features', 'texture_features']
        
        for key in required_keys:
            if key not in features_dict:
                return False, f"Missing {key}"
            
            if not isinstance(features_dict[key], np.ndarray):
                return False, f"{key} must be numpy array"
        
        # Check dimensions
        if len(features_dict['color_features']) != 256:
            return False, f"color_features must be 256-dimensional"
        
        if len(features_dict['shape_features']) != 7:
            return False, f"shape_features must be 7-dimensional"
        
        if len(features_dict['texture_features']) != 256:
            return False, f"texture_features must be 256-dimensional"
        
        return True, None
    
    else:
        return False, f"Unknown embedding type: {embedding_type}"
