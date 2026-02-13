"""
Feature Extraction Service

This module provides CLIP-based feature extraction for product matching.
It supports automatic profile selection for large catalogs and slow hardware.
"""

import os
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List

# Get logger (will inherit UTF-8 configuration from root logger in app.py)
logger = logging.getLogger(__name__)

# Import CLIP (required)
try:
    from image_processing_clip import (
        batch_extract_clip_embeddings,
        is_clip_available,
        load_clip_config,
        detect_device,
        AVAILABLE_MODELS,
        CLIPModelError,
        CLIPModelDownloadError
    )
    CLIP_AVAILABLE = is_clip_available()
except ImportError:
    CLIP_AVAILABLE = False
    AVAILABLE_MODELS = {}
    logger.error("CLIP not available - install PyTorch and sentence-transformers")

# Import error types for compatibility
from image_processing import (
    ImageProcessingError,
    InvalidImageFormatError,
    CorruptedImageError,
    ImageTooSmallError,
    ImageProcessingFailedError
)


SUPPORTED_PROCESSING_PROFILES = {'auto', 'balanced', 'fast'}
FAST_PROFILE_MIN_FILES = 5000
SLOW_CPU_CORES_THRESHOLD = 4
FAST_PREPROCESS_MAX_DIM = 768
_DETECTED_DEVICE_CACHE: Optional[str] = None


def _get_detected_device_cached() -> Tuple[str, Optional[str]]:
    """Detect compute device once per process to reduce repeated probing/log spam."""
    global _DETECTED_DEVICE_CACHE

    if _DETECTED_DEVICE_CACHE:
        return _DETECTED_DEVICE_CACHE, None

    if not CLIP_AVAILABLE:
        _DETECTED_DEVICE_CACHE = 'cpu'
        return _DETECTED_DEVICE_CACHE, None

    try:
        _DETECTED_DEVICE_CACHE = detect_device()
        return _DETECTED_DEVICE_CACHE, None
    except Exception as e:
        logger.debug(f"Failed to auto-detect device for profile resolution: {e}")
        return 'cpu', str(e)


def _resolve_processing_profile(
    image_count: int,
    processing_profile: str = 'auto',
    operation_total_files: Optional[int] = None
) -> Dict[str, Any]:
    """Resolve extraction profile based on request, hardware, and workload size."""
    requested_profile = (processing_profile or 'auto').strip().lower()
    if requested_profile not in SUPPORTED_PROCESSING_PROFILES:
        requested_profile = 'auto'

    total_files = operation_total_files if isinstance(operation_total_files, int) and operation_total_files > 0 else image_count
    total_files = max(total_files, image_count)

    cpu_cores = os.cpu_count() or 1
    detected_device, device_error = _get_detected_device_cached()

    is_cpu_only = detected_device == 'cpu'
    slow_cpu = is_cpu_only and cpu_cores <= SLOW_CPU_CORES_THRESHOLD
    huge_catalog = total_files >= FAST_PROFILE_MIN_FILES

    reasons = []
    if requested_profile == 'auto':
        if huge_catalog:
            resolved_profile = 'fast'
            reasons.append(f"large catalog ({total_files} files)")
        elif slow_cpu:
            resolved_profile = 'fast'
            reasons.append(f"slow CPU ({cpu_cores} cores, CPU mode)")
        else:
            resolved_profile = 'balanced'
            reasons.append("normal workload")
    elif requested_profile == 'fast':
        resolved_profile = 'fast'
        reasons.append("requested fast profile")
    else:
        resolved_profile = 'balanced'
        reasons.append("requested balanced profile")

    clip_config = load_clip_config() if CLIP_AVAILABLE else {'model_name': 'clip-ViT-B-32'}
    preferred_model = clip_config.get('model_name', 'clip-ViT-B-32')

    # Fast profile always uses the fastest supported 512-dim CLIP variant for compatibility.
    clip_model_name = 'clip-ViT-B-32' if resolved_profile == 'fast' else preferred_model
    if clip_model_name not in AVAILABLE_MODELS:
        clip_model_name = 'clip-ViT-B-32'
        reasons.append("unsupported model configured, falling back to clip-ViT-B-32")

    model_meta = AVAILABLE_MODELS.get(clip_model_name, {})
    if model_meta.get('embedding_dim', 512) != 512:
        clip_model_name = 'clip-ViT-B-32'
        reasons.append("configured model embedding dimension is not 512, falling back to clip-ViT-B-32")

    disable_multiprocessing = False

    return {
        'requested_profile': requested_profile,
        'profile_used': resolved_profile,
        'clip_model_name': clip_model_name,
        'fast_preprocess': resolved_profile == 'fast',
        'preprocess_max_dim': FAST_PREPROCESS_MAX_DIM if resolved_profile == 'fast' else None,
        'disable_multiprocessing': disable_multiprocessing,
        'detected_device': detected_device,
        'cpu_cores': cpu_cores,
        'operation_total_files': total_files,
        'reason': '; '.join(reasons),
        'device_detection_error': device_error
    }


def extract_features_unified(image_path: str) -> Tuple[Dict[str, np.ndarray], str, Optional[str]]:
    """Extract features for a single image using balanced CLIP settings."""
    if not CLIP_AVAILABLE:
        raise CLIPModelError(
            "CLIP is required but not available",
            "Install PyTorch and sentence-transformers: pip install torch sentence-transformers"
        )

    profile = _resolve_processing_profile(
        image_count=1,
        processing_profile='balanced',
        operation_total_files=1
    )

    results = batch_extract_clip_embeddings(
        image_paths=[image_path],
        model_name=profile['clip_model_name'],
        batch_size=1,
        skip_errors=False,
        use_amp=True,
        auto_adjust_batch=True,
        use_multiprocessing=False,
        fast_preprocess=profile['fast_preprocess'],
        preprocess_max_dim=profile['preprocess_max_dim']
    )

    if not results or results[0][1] is None:
        error_msg = results[0][2] if results and results[0][2] else "Unknown error"
        raise ImageProcessingFailedError(f"CLIP extraction failed: {error_msg}")

    clip_embedding = results[0][1]
    features_dict = {
        'color_features': clip_embedding,
        'shape_features': np.array([], dtype=np.float32),
        'texture_features': np.array([], dtype=np.float32)
    }

    return features_dict, 'clip', profile['clip_model_name']


def batch_extract_features_unified(
    image_paths: List[str],
    processing_profile: str = 'auto',
    operation_total_files: Optional[int] = None
) -> Tuple[List[Tuple[str, Optional[Dict[str, np.ndarray]], Optional[str], Optional[str], Optional[str]]], Dict[str, Any]]:
    """Extract features from multiple images using CLIP batch extraction with auto profile selection."""
    profile_info = _resolve_processing_profile(
        image_count=len(image_paths),
        processing_profile=processing_profile,
        operation_total_files=operation_total_files
    )

    if not CLIP_AVAILABLE:
        errors = [
            (path, None, None, None, "CLIP not available - install PyTorch and sentence-transformers")
            for path in image_paths
        ]
        profile_info['reason'] = f"{profile_info.get('reason', '')}; CLIP unavailable".strip('; ')
        return errors, profile_info

    logger.info(
        f"[BATCH-EXTRACT] Starting CLIP extraction for {len(image_paths)} images "
        f"(profile={profile_info['profile_used']}, model={profile_info['clip_model_name']}, "
        f"device={profile_info['detected_device']}, total_files={profile_info['operation_total_files']})"
    )

    results = batch_extract_clip_embeddings(
        image_paths=image_paths,
        model_name=profile_info['clip_model_name'],
        batch_size=32,
        skip_errors=True,
        use_amp=True,
        auto_adjust_batch=True,
        use_multiprocessing=None,
        fast_preprocess=profile_info['fast_preprocess'],
        preprocess_max_dim=profile_info['preprocess_max_dim']
    )

    unified_results = []
    embedding_version = profile_info['clip_model_name']
    for path, clip_embedding, error_msg in results:
        if clip_embedding is not None:
            features_dict = {
                'color_features': clip_embedding,
                'shape_features': np.array([], dtype=np.float32),
                'texture_features': np.array([], dtype=np.float32)
            }
            unified_results.append((path, features_dict, 'clip', embedding_version, None))
        else:
            unified_results.append((path, None, None, None, error_msg or "Unknown error"))

    success_count = sum(1 for _, features, _, _, _ in unified_results if features is not None)
    logger.info(
        f"[BATCH-EXTRACT] Completed: {success_count}/{len(image_paths)} successful "
        f"(profile={profile_info['profile_used']}, model={embedding_version})"
    )

    return unified_results, profile_info


def get_feature_extraction_info() -> Dict[str, Any]:
    """Get information about current feature extraction configuration."""
    info = {
        'clip_available': CLIP_AVAILABLE,
        'current_method': 'clip'
    }

    if CLIP_AVAILABLE:
        try:
            config = load_clip_config()
            info['clip_config'] = {
                'model_name': config.get('model_name', 'clip-ViT-B-32')
            }
        except Exception:
            pass

    return info


def validate_features_for_matching(features_dict: Dict[str, Any],
                                   embedding_type: str) -> Tuple[bool, Optional[str]]:
    """Validate that features are suitable for matching."""
    if embedding_type != 'clip':
        return False, f"Unsupported embedding type: {embedding_type}. Only CLIP is supported."

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
