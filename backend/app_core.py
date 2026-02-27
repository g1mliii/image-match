import os
import sys

if sys.platform == 'win32':
    os.environ['PYTHONUTF8'] = '1'
    # Reconfigure stdout/stderr to use UTF-8 with error replacement
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import shutil
import re
import json
import io
import logging
import time
import threading
import secrets
import ipaddress
import subprocess
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any, Tuple
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from datetime import datetime

# Set PyTorch environment variables BEFORE importing torch/CLIP
# This fixes GPU memory fragmentation issues with AMD ROCm
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from flask import Flask, request, jsonify, send_file, send_from_directory, Response, stream_with_context

from database import (
    init_db, insert_product, bulk_insert_products, get_product_by_id, get_features_by_product_id,
    insert_features, validate_sku_format, normalize_sku, check_sku_exists,
    insert_price_history, bulk_insert_price_history, get_price_history,
    get_price_statistics, link_price_history, get_products_with_price_history,
    insert_performance_history, bulk_insert_performance_history, bulk_insert_performance_history_batch, get_performance_history,
    get_performance_statistics, link_performance_history, get_products_with_performance_history,
    get_catalog_stats, get_products_paginated, get_all_categories, update_product,
    delete_product, bulk_delete_products, bulk_update_products, clear_products_by_type,
    clear_all_matches, clear_products_by_categories, clear_products_by_date,
    vacuum_database, clear_uploaded_images, stream_catalog_csv, delete_features,
    get_db_connection,
    get_products_by_ids,
    create_match_result_session, store_match_result_session_items, get_match_result_session_page,
    clear_all_match_result_sessions, cleanup_stale_match_result_sessions,
    # Dynamic metadata schema functions
    save_metadata_schema, get_metadata_schema, clear_metadata_schema,
    update_product_metadata, get_product_metadata
)
from image_processing import (
    validate_image_file,
    ImageProcessingError, InvalidImageFormatError, CorruptedImageError,
    ImageTooSmallError, ImageProcessingFailedError
)
from feature_extraction_service import (
    extract_features_unified,
    get_feature_extraction_info
)
from product_matching import (
    find_matches,
    find_metadata_matches,
    batch_find_matches,
    batch_find_metadata_matches,
    normalize_category,
    MatchingError, ProductNotFoundError, MissingFeaturesError,
    EmptyCatalogError, AllMatchesFailedError
)
from hybrid_matching import (
    find_hybrid_matches,
    batch_find_hybrid_matches
)
from validation_utils import (
    validate_category, validate_product_name, validate_sku,
    validate_cleanup_type, validate_days, validate_categories_list,
    validate_page_params, sanitize_search_query, validate_product_ids
)
from path_manager import get_uploads_dir, get_backend_dir

# Get backend directory first (needed for logging setup)
BACKEND_DIR = get_backend_dir()

# Configure logging with file rotation to prevent unbounded growth
from logging.handlers import RotatingFileHandler
import os

LOGS_DIR = os.path.join(BACKEND_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure root logger so all child loggers inherit UTF-8 handlers
# This ensures snapshot_manager, database, and all other modules work correctly
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid duplicates
root_logger.handlers.clear()

# Create formatter
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler with rotation (max 10MB per file, keep 5 backups = 50MB total)
log_file = os.path.join(LOGS_DIR, 'application.log')
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,  # Keep 5 rotated backups
    encoding='utf-8'  # Use UTF-8 for file as well
)
file_handler.setFormatter(log_format)
root_logger.addHandler(file_handler)

# Console handler for stdout with UTF-8 encoding
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_format)
# Force UTF-8 encoding for console to handle Unicode characters (▶, ✓, etc.)
if hasattr(console_handler.stream, 'reconfigure'):
    console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
root_logger.addHandler(console_handler)

# Create module logger (will inherit root logger's handlers)
logger = logging.getLogger(__name__)

# Suppress werkzeug HTTP request logs (too verbose)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Suppress noisy SWIG DeprecationWarnings from faiss/other C-extensions
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)

# App Lock and Crash Detection
# (BACKEND_DIR already defined above for logging setup)

# Import debug mode configuration (centralized to avoid circular imports)
from config import is_debug_mode, DEBUG_MODE

APP_LOCK_FILE = os.path.join(BACKEND_DIR, '.app.lock')

# Reduce noisy INFO logs in normal mode while preserving warnings/errors.
if not DEBUG_MODE:
    logging.getLogger('product_matching').setLevel(logging.WARNING)
    logging.getLogger('hybrid_matching').setLevel(logging.WARNING)
    logging.getLogger('feature_cache').setLevel(logging.WARNING)

def create_app_lock():
    """Create lock file to detect crashes on next startup"""
    try:
        with open(APP_LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        logger.info("Created app lock file (crash detection enabled)")
    except Exception as e:
        logger.warning(f"Failed to create app lock file: {e}")

def remove_app_lock():
    """Remove lock file on clean shutdown"""
    try:
        if os.path.exists(APP_LOCK_FILE):
            os.remove(APP_LOCK_FILE)
            logger.info("✓ App lock file removed (clean shutdown)")
        else:
            logger.debug("App lock file already removed or did not exist")
    except Exception as e:
        logger.warning(f"Failed to remove app lock file: {e}")

def detect_crash():
    """Check if app crashed on last run"""
    return os.path.exists(APP_LOCK_FILE)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Disable debug mode to prevent Flask from reloading and reloading CLIP model
# This was causing GPU memory fragmentation with multiple model loads
app.config['ENV'] = 'production'
app.debug = False

# Ensure Flask's logger uses the root logger's UTF-8 handlers (no duplicates)
app.logger.handlers.clear()
app.logger.propagate = True  # Propagate to root logger
app.logger.setLevel(logging.INFO)

# Configuration
app.config['UPLOAD_FOLDER'] = get_uploads_dir()
# 500MB max content length - prevents accidental DoS from oversized uploads
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
MAX_UPLOAD_FILES_PER_OPERATION = 50000
SUPPORTED_PROCESSING_PROFILES = {'auto', 'balanced', 'fast'}

# Ensure upload directory exists (get_uploads_dir() already does this, but keeping for clarity)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.after_request
def set_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    return response

# Initialize database
init_db()

# Crash detection and app lock
crash_detected = detect_crash()
if crash_detected:
    logger.warning("Previous app crash detected (lock file found)")
create_app_lock()

# Clean up expired session autosaves
try:
    from snapshot_manager import cleanup_expired_session_autosaves
    cleanup_expired_session_autosaves()
except Exception as e:
    logger.warning(f"Failed to cleanup expired session autosaves: {e}")

# Clean up stale match-result sessions (>24h old) left over from crashes
try:
    expired = cleanup_stale_match_result_sessions(max_age_hours=24)
    if expired > 0:
        logger.info(f"Cleaned up {expired} stale match-result session(s) on startup")
except Exception as e:
    logger.warning(f"Failed to cleanup stale match-result sessions: {e}")

# Pre-load CLIP model on startup (download if needed)
logger.info("Initializing CLIP model...")
try:
    from image_processing_clip import is_clip_available, get_clip_model
    if is_clip_available():
        # This will download the model if not cached (~350MB, one-time)
        model = get_clip_model()
        logger.info("CLIP model loaded successfully")
    else:
        logger.warning("CLIP not available, will use legacy feature extraction")
except Exception as e:
    logger.warning(f"Could not pre-load CLIP model: {e}. Will use legacy feature extraction.")

# Build FAISS indexes on startup for fast similarity search
logger.info("Building FAISS indexes for fast similarity search...")
try:
    from database import rebuild_all_faiss_indexes
    stats = rebuild_all_faiss_indexes()
    if 'error' in stats:
        logger.warning(f"FAISS indexes not built: {stats.get('error')} - {stats.get('suggestion', '')}")
        logger.info("Similarity search will use brute force (slower for large catalogs)")
    else:
        logger.info(f"FAISS indexes built: {stats['categories_processed']} categories, {stats['total_products_indexed']} products")
        if stats['failed_categories']:
            logger.warning(f"Failed to build indexes for: {stats['failed_categories']}")
except Exception as e:
    logger.warning(f"Could not build FAISS indexes: {e}. Similarity search will use brute force.")


from collections import OrderedDict
_csv_cache = OrderedDict()
_csv_cache_max_size = 10  # Max 10 cached CSVs
_CSV_CACHE_MAX_BYTES = 100 * 1024 * 1024  # 100MB total cache size limit
_csv_cache_sizes = {}  # Track byte size per entry

def cache_csv_data(cache_key, csv_data):
    """Add CSV data to cache with LRU eviction and byte-size limit.

    MEMORY SAFETY: Enforces both count and byte limits to prevent
    10 large CSVs (50k products each ~10MB) from using 100MB+ RAM.
    """
    global _csv_cache
    import sys
    entry_size = sys.getsizeof(csv_data) if csv_data else 0

    # Evict until we're under both count and byte limits
    total_bytes = sum(_csv_cache_sizes.values())
    while (_csv_cache and
           (len(_csv_cache) >= _csv_cache_max_size or
            total_bytes + entry_size > _CSV_CACHE_MAX_BYTES)):
        oldest_key, _ = _csv_cache.popitem(last=False)
        evicted_size = _csv_cache_sizes.pop(oldest_key, 0)
        total_bytes -= evicted_size
        logger.debug(f"CSV cache evicted: {oldest_key} (~{evicted_size // 1024}KB)")

    _csv_cache[cache_key] = csv_data
    _csv_cache_sizes[cache_key] = entry_size
    _csv_cache.move_to_end(cache_key)  # Mark as recently used

def get_cached_csv(cache_key):
    """Get CSV from cache and mark as recently used"""
    global _csv_cache
    if cache_key in _csv_cache:
        _csv_cache.move_to_end(cache_key)  # Mark as recently used
        return _csv_cache[cache_key]
    return None

def invalidate_csv_cache():
    """Invalidate CSV cache (call when products are modified or snapshot changes)"""
    global _csv_cache
    _csv_cache.clear()
    _csv_cache_sizes.clear()
    logger.debug("CSV cache invalidated")


_catalog_categories_cache = {}
_catalog_categories_cache_ttl = 300  # 5 minutes (longer than main DB since snapshots are static)
_catalog_categories_cache_lock = threading.Lock()

def get_cached_catalog_categories(catalog_id: str) -> Optional[List[str]]:
    """Get cached categories for a catalog snapshot"""
    with _catalog_categories_cache_lock:
        if catalog_id in _catalog_categories_cache:
            categories, timestamp = _catalog_categories_cache[catalog_id]
            if time.time() - timestamp < _catalog_categories_cache_ttl:
                logger.debug(f"[CATEGORY-CACHE] Hit for {catalog_id}: {len(categories)} categories")
                return categories
            else:
                # Expired
                del _catalog_categories_cache[catalog_id]
                logger.debug(f"[CATEGORY-CACHE] Expired for {catalog_id}")
        return None

def cache_catalog_categories(catalog_id: str, categories: List[str]):
    """Cache categories for a catalog snapshot"""
    with _catalog_categories_cache_lock:
        _catalog_categories_cache[catalog_id] = (categories, time.time())
        logger.debug(f"[CATEGORY-CACHE] Cached {len(categories)} categories for {catalog_id} (TTL: {_catalog_categories_cache_ttl}s)")

def invalidate_catalog_categories_cache(catalog_id: Optional[str] = None):
    """Invalidate category cache (call when catalog is modified or on snapshot change)"""
    global _catalog_categories_cache
    with _catalog_categories_cache_lock:
        if catalog_id:
            _catalog_categories_cache.pop(catalog_id, None)
            logger.debug(f"[CATEGORY-CACHE] Invalidated {catalog_id}")
        else:
            _catalog_categories_cache.clear()
            logger.debug("[CATEGORY-CACHE] Invalidated all catalogs")



def cleanup_on_shutdown():
    """
    Explicit cleanup function to call on application shutdown.
    This should be called from main.py when the desktop app closes.

    Cleans up:
    - CLIP model cache (350MB+)
    - FAISS indexes (500MB+)
    - Database connections
    - Old uploaded images (30+ days)
    - Logging handlers
    - Flask app context
    - GPU memory
    - Garbage collection
    """
    logger.info("Starting application shutdown cleanup...")
    try:
        # Clear matches from the database table
        from database import clear_all_matches, invalidate_faiss_index
        deleted = clear_all_matches()
        
        # Invalidate FAISS to ensure next run starts fresh
        invalidate_faiss_index(None)
        
        logger.info(f"✓ Session cleanup: Deleted {deleted} matches from database")
    except Exception as e:
        logger.warning(f"Failed to clear session matches: {e}")
    # --- END OF NEW BLOCK ---
    
    try:
        # Clear CLIP model cache (350MB+ memory)
        from image_processing_clip import clear_clip_model_cache
        clear_clip_model_cache()
        logger.info("✓ CLIP model cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear CLIP model cache: {e}")

    try:
        # Clear FAISS indexes from memory (500MB+)
        from faiss_index import faiss_manager
        faiss_manager.clear_all_indexes()
    except Exception as e:
        logger.warning(f"Failed to clear FAISS indexes: {e}")

    try:
        # Clear feature cache singleton to release preloaded embeddings
        from feature_cache import clear_all_caches
        clear_all_caches()
        logger.info("✓ Feature cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear feature cache: {e}")

    try:
        # Clear in-process CSV/category caches
        invalidate_csv_cache()
        invalidate_catalog_categories_cache()
        logger.info("✓ CSV and category caches cleared")
    except Exception as e:
        logger.warning(f"Failed to clear in-process caches: {e}")

    try:
        # Stop backend-managed ngrok process to avoid orphan background tunnel.
        stopped, stop_error = stop_ngrok_tunnel_for_backend()
        if stopped:
            logger.info("✓ Backend ngrok tunnel stopped")
        elif stop_error and 'No app-managed ngrok process' not in str(stop_error):
            logger.warning(f"Failed to stop backend ngrok tunnel: {stop_error}")
    except Exception as e:
        logger.warning(f"Failed to stop backend ngrok tunnel: {e}")

    try:
        # Close all database connections
        from database import close_all_db_connections
        close_all_db_connections()
    except Exception as e:
        logger.warning(f"Failed to close database connections: {e}")

    try:
        # Clean up old uploaded images (older than 30 days)
        from database import cleanup_old_uploaded_images
        cleanup_old_uploaded_images(days_retention=30)
    except Exception as e:
        logger.warning(f"Failed to cleanup old uploads: {e}")

    try:
        # Close all logging handlers to release file handles
        for handler in logging.root.handlers[:]:
            try:
                handler.close()
                logging.root.removeHandler(handler)
            except Exception as e:
                logger.warning(f"Failed to close logging handler: {e}")
        logger.info("✓ Logging handlers closed")
    except Exception as e:
        logger.warning(f"Failed to close logging handlers: {e}")

    try:
        # Clear CSV builder staging dict (prevent memory leak from orphaned windows)
        cleaned_count = len(csv_builder_staging)
        csv_builder_staging.clear()
        if cleaned_count > 0:
            logger.info(f"✓ Cleared CSV builder staging ({cleaned_count} orphaned entries)")
        else:
            logger.debug("CSV builder staging already empty")
    except Exception as e:
        logger.warning(f"Failed to clear CSV builder staging: {e}")

    try:
        # Clear Flask app context and thread-local storage
        try:
            # Pop any active Flask app contexts using internal stack
            from flask import _app_ctx_stack
            popped_count = 0
            while _app_ctx_stack.top is not None:
                _app_ctx_stack.pop()
                popped_count += 1
            if popped_count > 0:
                logger.debug(f"✓ Popped {popped_count} Flask app context(s)")
            else:
                logger.debug("No active Flask app contexts to pop")
        except Exception as e:
            logger.debug(f"Flask context cleanup skipped: {e}")
            pass

        try:
            # Clear Flask g object (thread-local request data)
            from flask import g
            if hasattr(g, '__dict__'):
                g.__dict__.clear()
            logger.debug("Cleared Flask g object")
        except Exception:
            pass

        logger.info("✓ Flask app context cleared")
    except Exception as e:
        logger.warning(f"Failed to clear Flask context: {e}")

    try:
        # Force garbage collection
        import gc
        collected = gc.collect()
        logger.info(f"✓ Garbage collection freed {collected} objects")
    except Exception as e:
        logger.warning(f"Failed to run garbage collection: {e}")

    try:
        # Clear CUDA/GPU cache if available
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ CUDA cache cleared")
    except Exception:
        pass

    try:
        # Remove app lock file (signal clean shutdown)
        remove_app_lock()
    except Exception as e:
        logger.warning(f"Failed to remove app lock: {e}")

    try:
        # Clear uploads folder as requested (temp folder)
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            # Count files and calculate size before removal
            files_count = 0
            total_size = 0
            try:
                for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    if os.path.isfile(filepath):
                        files_count += 1
                        total_size += os.path.getsize(filepath)
            except Exception as e:
                logger.warning(f"Error counting uploads folder contents: {e}")

            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            space_mb = round(total_size / (1024 * 1024), 2)
            logger.info(f"✓ Removed uploads folder ({files_count} files, {space_mb}MB freed)")
            # Recreate empty folder for next time
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        else:
            logger.debug("Uploads folder does not exist, skipping cleanup")
    except Exception as e:
        logger.warning(f"Failed to remove uploads folder: {e}")

    logger.info("Application cleanup complete")

# Supported image formats
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


_CATEGORY_METADATA_KEYS = {'category', 'product_category', 'productcategory'}
_CATEGORY_METADATA_HINTS = ('category', 'product_category', 'productcategory')


def _normalize_metadata_key(key: Any) -> str:
    """Normalize metadata keys for robust matching."""
    return str(key).strip().lower().replace('-', '_').replace(' ', '_')


def _extract_category_from_metadata_payload(metadata_payload: Any) -> Optional[str]:
    """
    Extract a normalized category value from metadata payload.

    PERFORMANCE:
    - Fast exits for non-dict/non-JSON strings.
    - JSON parse is only attempted when category hints exist in the payload text.

    SAFETY:
    - Returns None for empty/unknown category variants after normalization.
    """
    metadata_obj: Optional[Dict[str, Any]] = None

    if isinstance(metadata_payload, dict):
        metadata_obj = metadata_payload
    elif isinstance(metadata_payload, str):
        raw = metadata_payload.strip()
        if not raw:
            return None

        raw_lower = raw.lower()
        if not any(hint in raw_lower for hint in _CATEGORY_METADATA_HINTS):
            return None

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                metadata_obj = parsed
            else:
                return None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    else:
        return None

    if not metadata_obj:
        return None

    # Rule 1: If explicit "category" exists but is empty/unknown, keep uncategorized.
    # This preserves intentional blank category behavior.
    for key, value in metadata_obj.items():
        if _normalize_metadata_key(key) != 'category':
            continue
        category_raw = '' if value is None else str(value).strip()
        if not category_raw:
            return None
        return normalize_category(category_raw)

    # Rule 2: Backfill from alias keys only when explicit "category" is absent.
    for key, value in metadata_obj.items():
        key_norm = _normalize_metadata_key(key)
        if key_norm not in _CATEGORY_METADATA_KEYS or key_norm == 'category':
            continue

        category_raw = '' if value is None else str(value).strip()
        if not category_raw:
            continue

        normalized = normalize_category(category_raw)
        if normalized is not None:
            return normalized

    return None

def create_error_response(error_code, message, suggestion=None, details=None, status_code=400):
    """Create standardized error response"""
    response = {
        'error': message,
        'error_code': error_code
    }
    if suggestion:
        response['suggestion'] = suggestion
    if details:
        response['details'] = details
    
    logger.error(f"Error {error_code}: {message}")
    return jsonify(response), status_code

@app.route('/')
def index():
    """Serve the main application (brutalist design)"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/gradient')
def gradient():
    """Serve the old gradient version (archived)"""
    return send_from_directory(os.path.join(app.static_folder, 'old-gradient-ui'), 'index.html')

@app.route('/csv-builder')
def csv_builder():
    """Serve the CSV builder tool"""
    return send_from_directory(app.static_folder, 'csv-builder.html')

# CSV Builder staging (for passing data between windows)
csv_builder_staging = {}

@app.route('/api/csv-builder/stage', methods=['POST'])
def stage_csv_builder_data():
    """Stage file data for CSV builder window"""
    try:
        data = request.get_json()
        window_id = data.get('window_id')
        file_data = data.get('file_data', [])
        section = data.get('section', 'historical')

        if not window_id:
            return jsonify({'error': 'window_id is required'}), 400

        # Store data with timestamp for cleanup
        csv_builder_staging[window_id] = {
            'file_data': file_data,
            'section': section,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Staged {len(file_data)} files for CSV builder window {window_id}")
        return jsonify({'success': True, 'file_count': len(file_data)})

    except Exception as e:
        logger.error(f"Error staging CSV builder data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/csv-builder/get-staged/<window_id>', methods=['GET'])
def get_staged_csv_builder_data(window_id):
    """Retrieve staged file data for CSV builder window"""
    try:
        if window_id not in csv_builder_staging:
            return jsonify({'error': 'No staged data found for this window'}), 404

        data = csv_builder_staging[window_id]

        # Clean up after retrieval
        del csv_builder_staging[window_id]

        logger.info(f"Retrieved staged data for window {window_id}: {len(data['file_data'])} files")
        return jsonify({
            'file_data': data['file_data'],
            'section': data['section']
        })

    except Exception as e:
        logger.error(f"Error retrieving staged CSV builder data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/catalog-manager')
def catalog_manager():
    """Serve the Catalog Manager tool"""
    return send_from_directory(app.static_folder, 'catalog-manager.html')

@app.route('/mobile')
@app.route('/m')
@app.route('/phone')
@app.route('/upload')
def mobile_upload():
    """Serve the Mobile Upload page

    Accessible via multiple easy-to-type URLs:
    - /mobile (original)
    - /m (shortest)
    - /phone (memorable)
    - /upload (intuitive)
    """
    return send_from_directory(app.static_folder, 'mobile-upload.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'})

@app.route('/api/gpu/status', methods=['GET'])
def get_gpu_status():
    
    try:
        from image_processing_clip import (
            get_device_info,
            is_clip_available,
            TORCH_AVAILABLE
        )
        
        if not TORCH_AVAILABLE:
            return jsonify({
                'available': False,
                'device': 'cpu',
                'gpu_name': None,
                'throughput': '5-20',
                'first_run': False,
                'error': 'PyTorch not installed'
            })
        
        device_info = get_device_info()
        clip_available = is_clip_available()
        
        # Estimate throughput based on device
        throughput = '5-20'  # CPU default
        if device_info['device'] == 'cuda':
            throughput = '150-300'  # NVIDIA GPU
        elif device_info['device'] == 'rocm':
            throughput = '150-200'  # AMD GPU
        elif device_info['device'] == 'mps':
            throughput = '50-150'  # Apple Silicon
        
        # Check if this is first run (model not cached)
        first_run = False
        try:
            from pathlib import Path
            cache_dir = Path.home() / '.cache' / 'torch' / 'sentence_transformers'
            first_run = not cache_dir.exists() or not any(cache_dir.glob('*clip*'))
        except Exception:
            pass
        
        return jsonify({
            'available': device_info['device'] != 'cpu',
            'device': device_info['device'],
            'gpu_name': device_info.get('gpu_name'),
            'vram': device_info.get('vram_gb'),
            'throughput': throughput,
            'first_run': first_run,
            'clip_available': clip_available,
            'error': None
        })
        
    except Exception as e:
        logger.error(f"Error getting GPU status: {e}")
        return jsonify({
            'available': False,
            'device': 'cpu',
            'gpu_name': None,
            'throughput': '5-20',
            'first_run': False,
            'error': str(e)
        })

# Mobile routes moved to dedicated module to keep this file focused.
try:
    from .routes.mobile_routes import (
        mobile_bp,
        configure_mobile_routes,
        stop_ngrok_tunnel_for_backend,
    )  # type: ignore
except ImportError:
    from routes.mobile_routes import (
        mobile_bp,
        configure_mobile_routes,
        stop_ngrok_tunnel_for_backend,
    )  # type: ignore

configure_mobile_routes(
    logger=logger,
    create_error_response=create_error_response,
    invalidate_csv_cache=invalidate_csv_cache,
    invalidate_catalog_categories_cache=invalidate_catalog_categories_cache,
    BACKEND_DIR=BACKEND_DIR,
    get_product_by_id=get_product_by_id,
    get_product_metadata=get_product_metadata,
    insert_product=insert_product,
    insert_features=insert_features,
    extract_features_unified=extract_features_unified,
    validate_category=validate_category,
    validate_product_name=validate_product_name,
    validate_sku=validate_sku,
    allowed_file=allowed_file,
)
if 'mobile_routes' not in app.blueprints:
    app.register_blueprint(mobile_bp)

@app.route('/api/products/<int:product_id>/image', methods=['GET'])
def get_product_image(product_id):
    """Get product image by ID"""
    try:
        product = get_product_by_id(product_id)
        if not product:
            return create_error_response(
                'PRODUCT_NOT_FOUND',
                f'Product with ID {product_id} not found',
                status_code=404
            )
        
        image_path = product['image_path']
        if not os.path.exists(image_path):
            # Return placeholder image instead of 404 (don't log error)
            return send_file('static/placeholder.png', mimetype='image/png') if os.path.exists('static/placeholder.png') else ('', 404)
        
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving image for product {product_id}: {e}")
        return create_error_response(
            'IMAGE_ERROR',
            'Failed to load image',
            status_code=500
        )

@app.route('/api/products/metadata', methods=['POST'])
def create_metadata_product():
    
    try:
        data = request.get_json()
        
        if not data:
            return create_error_response(
                'MISSING_DATA',
                'No JSON data provided',
                'Please provide product metadata in JSON format',
                status_code=400
            )
        
        # Validate required fields
        sku = data.get('sku')
        product_name = data.get('product_name')
        
        if not sku or not product_name:
            return create_error_response(
                'MISSING_REQUIRED_FIELDS',
                'SKU and product_name are required',
                'Please provide both SKU and product_name',
                status_code=400
            )
        
        # Get optional fields
        category = data.get('category', None)
        price = data.get('price', None)
        performance_history = data.get('performance_history', None)
        is_historical = data.get('is_historical', False)
        
        # Normalize empty strings to None
        if category and str(category).strip() == '':
            category = None
        
        # Create product in database without image
        # Use a placeholder for image_path since it's NOT NULL in schema
        product_id = insert_product(
            image_path='[METADATA_ONLY]',  # Placeholder for Mode 2 (no actual image)
            category=category,
            product_name=product_name,
            sku=sku,
            is_historical=is_historical
        )
        
        # Add performance history if provided
        if performance_history and isinstance(performance_history, list):
            try:
                from database import bulk_insert_performance_history
                from datetime import timedelta
                
                # Convert simple numbers to complex format with auto-generated dates
                performance_records = []
                today = datetime.now()
                
                for i, perf_value in enumerate(performance_history):
                    if isinstance(perf_value, (int, float)) and perf_value >= 0:
                        # Generate monthly dates going backwards
                        date_obj = today - timedelta(days=30 * (len(performance_history) - 1 - i))
                        date_str = date_obj.strftime('%Y-%m-%d')
                        
                        # Simple format: just sales numbers, rest are 0
                        performance_records.append({
                            'date': date_str,
                            'sales': int(perf_value),
                            'views': 0,
                            'conversion_rate': 0.0,
                            'revenue': 0.0
                        })
                
                if performance_records:
                    bulk_insert_performance_history(product_id, performance_records)
                    logger.info(f"Added {len(performance_records)} performance history records for product {product_id}")
            except Exception as e:
                logger.warning(f"Failed to add performance history for product {product_id}: {e}")
        
        logger.info(f"Created metadata-only product: {product_id} (SKU: {sku}, Name: {product_name})")

        # Invalidate CSV cache since product was added
        invalidate_csv_cache()

        return jsonify({
            'success': True,
            'product_id': product_id,
            'message': 'Product created successfully (metadata only)',
            'mode': 'metadata_only'
        }), 200
        
    except Exception as e:
        logger.error(f"Error creating metadata product: {e}")
        return create_error_response(
            'PRODUCT_CREATION_ERROR',
            str(e),
            'Failed to create product with metadata',
            status_code=500
        )


@app.route('/api/products/metadata/batch', methods=['POST'])
def create_metadata_products_batch():
    try:
        data = request.get_json()
        
        if not data:
            return create_error_response(
                'MISSING_DATA',
                'No JSON data provided',
                'Please provide products array in JSON format',
                status_code=400
            )
        
        products = data.get('products', [])
        
        if not products or not isinstance(products, list):
            return create_error_response(
                'INVALID_PRODUCTS',
                'products must be a non-empty array',
                'Example: {"products": [{"sku": "SKU001", "product_name": "Product 1"}, ...]}',
                status_code=400
            )
        
        logger.info(f"[BATCH-METADATA] Starting batch creation for {len(products)} products")
        
        # Step 1: Validate all products (parallel validation)
        logger.info("[BATCH-METADATA] Step 1: Validating products (parallel)")
        from concurrent.futures import ThreadPoolExecutor
        
        def validate_product(item):
            i, product = item
            sku = product.get('sku')
            product_name = product.get('product_name')
            
            if not sku or not product_name:
                return None, f'Product {i+1}: SKU and product_name are required'
            
            category = product.get('category', None)
            is_historical = product.get('is_historical', False)
            
            # Normalize empty strings to None
            if category and str(category).strip() == '':
                category = None
            
            # Extract metadata (all other fields)
            known_fields = {'sku', 'product_name', 'category', 'is_historical', 'performance_history', 'price', 'price_history'}
            metadata = {k: v for k, v in product.items() if k not in known_fields}
            
            return {
                'sku': sku,
                'product_name': product_name,
                'category': category,
                'is_historical': is_historical,
                'performance_history': product.get('performance_history', None),
                'price': product.get('price', None),
                'price_history': product.get('price_history', None),
                'metadata': metadata
            }, None
        
        validated_products = []
        validation_errors = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(validate_product, enumerate(products))
            for validated, error in results:
                if error:
                    validation_errors.append(error)
                else:
                    validated_products.append(validated)
        
        if validation_errors:
            return create_error_response(
                'VALIDATION_ERROR',
                validation_errors[0],
                'Check all products have SKU and product_name',
                status_code=400
            )
        
        logger.info(f"[BATCH-METADATA] ✓ Validated {len(validated_products)} products")
        
        # Step 2: Batch insert products in chunks (incremental)
        logger.info("[BATCH-METADATA] Step 2: Batch inserting products (chunked)")
        from database import bulk_insert_products
        import json
        
        CHUNK_SIZE = 100
        product_ids = []
        
        for chunk_idx in range(0, len(validated_products), CHUNK_SIZE):
            chunk = validated_products[chunk_idx:chunk_idx + CHUNK_SIZE]
            products_to_insert = [
                (
                    '[METADATA_ONLY]', 
                    p['category'], 
                    p['product_name'], 
                    p['sku'], 
                    p['is_historical'],
                    json.dumps(p['metadata']) if p['metadata'] else None
                )
                for p in chunk
            ]
            
            chunk_ids = bulk_insert_products(products_to_insert)
            product_ids.extend(chunk_ids)
            logger.debug(f"[BATCH-METADATA] Chunk {chunk_idx // CHUNK_SIZE + 1}: Inserted {len(chunk_ids)} products")
        
        logger.info(f"[BATCH-METADATA] ✓ Inserted {len(product_ids)} products in {(len(validated_products) + CHUNK_SIZE - 1) // CHUNK_SIZE} chunks")
        
        # Step 3: Batch insert performance histories (chunked)
        logger.info("[BATCH-METADATA] Step 3: Batch inserting performance histories (chunked)")
        from datetime import timedelta
        from database import bulk_insert_performance_history_batch
        
        all_perf_records = []
        PERF_CHUNK_SIZE = 500  # Insert every 500 records
        total_perf_inserted = 0
        
        for product_id, product in zip(product_ids, validated_products):
            if product.get('performance_history') and isinstance(product['performance_history'], list):
                try:
                    today = datetime.now()
                    perf_history = product['performance_history']
                    
                    for j, perf_value in enumerate(perf_history):
                        if isinstance(perf_value, (int, float)) and perf_value >= 0:
                            # Generate monthly dates going backwards
                            date_obj = today - timedelta(days=30 * (len(perf_history) - 1 - j))
                            date_str = date_obj.strftime('%Y-%m-%d')
                            
                            # Simple format: just sales numbers, rest are 0
                            all_perf_records.append((
                                product_id,
                                date_str,
                                int(perf_value),
                                0,  # views
                                0.0,  # conversion_rate
                                0.0  # revenue
                            ))
                            
                            # OPTIMIZATION: Insert incrementally to avoid memory bloat
                            if len(all_perf_records) >= PERF_CHUNK_SIZE:
                                try:
                                    inserted = bulk_insert_performance_history_batch(all_perf_records)
                                    total_perf_inserted += inserted
                                    logger.debug(f"[BATCH-METADATA] Incremental perf insert: {inserted} records (total: {total_perf_inserted})")
                                    all_perf_records = []
                                except Exception as e:
                                    logger.warning(f"[BATCH-METADATA] Incremental perf insert failed: {e}, will retry at end")
                except Exception as e:
                    logger.warning(f"Failed to process performance history for product {product_id}: {e}")
        
        # Insert remaining performance records
        if all_perf_records:
            try:
                inserted = bulk_insert_performance_history_batch(all_perf_records)
                total_perf_inserted += inserted
                logger.info(f"[BATCH-METADATA] ✓ Final perf batch inserted {inserted} records (total: {total_perf_inserted})")
            except Exception as e:
                logger.warning(f"Failed to insert remaining performance histories: {e}")
        
        # Step 4: Batch insert price histories (chunked)
        logger.info("[BATCH-METADATA] Step 4: Batch inserting price histories (chunked)")
        from database import bulk_insert_price_history
        
        all_price_records = []
        PRICE_CHUNK_SIZE = 500  # Insert every 500 records
        total_price_inserted = 0
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        for product_id, product in zip(product_ids, validated_products):
            # Process current price (single value)
            if 'price' in product and product['price'] is not None:
                try:
                    price_value = float(product['price'])
                    if price_value >= 0:
                        all_price_records.append({
                            'product_id': product_id,
                            'date': today_str,
                            'price': price_value,
                            'currency': 'USD'
                        })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid price for product {product_id}: {product['price']}")
            
            # Process price history (array of date:price entries)
            if 'price_history' in product and product['price_history']:
                try:
                    price_history = product['price_history']
                    if isinstance(price_history, str):
                        # Parse semicolon-separated entries: "2025-01-15:29.99;2025-02-15:27.99"
                        entries = price_history.split(';')
                        for entry in entries:
                            if ':' in entry:
                                date_str, price_str = entry.split(':', 1)
                                try:
                                    price_value = float(price_str.strip())
                                    if price_value >= 0:
                                        all_price_records.append({
                                            'product_id': product_id,
                                            'date': date_str.strip(),
                                            'price': price_value,
                                            'currency': 'USD'
                                        })
                                except ValueError:
                                    logger.warning(f"Invalid price in history for product {product_id}: {price_str}")
                    elif isinstance(price_history, list):
                        # Handle array format if provided
                        for item in price_history:
                            if isinstance(item, dict) and 'date' in item and 'price' in item:
                                try:
                                    price_value = float(item['price'])
                                    if price_value >= 0:
                                        all_price_records.append({
                                            'product_id': product_id,
                                            'date': item['date'],
                                            'price': price_value,
                                            'currency': item.get('currency', 'USD')
                                        })
                                except (ValueError, TypeError):
                                    logger.warning(f"Invalid price in history array for product {product_id}")
                except Exception as e:
                    logger.warning(f"Failed to process price history for product {product_id}: {e}")
            
            # OPTIMIZATION: Insert incrementally to avoid memory bloat
            if len(all_price_records) >= PRICE_CHUNK_SIZE:
                try:
                    inserted = bulk_insert_price_history(product_id, all_price_records)
                    total_price_inserted += inserted
                    logger.debug(f"[BATCH-METADATA] Incremental price insert: {inserted} records (total: {total_price_inserted})")
                    all_price_records = []
                except Exception as e:
                    logger.warning(f"[BATCH-METADATA] Incremental price insert failed: {e}, will retry at end")
        
        # Insert remaining price records
        if all_price_records:
            try:
                # Get the last product_id for bulk insert
                if product_ids:
                    inserted = bulk_insert_price_history(product_ids[-1], all_price_records)
                    total_price_inserted += inserted
                    logger.info(f"[BATCH-METADATA] ✓ Final price batch inserted {inserted} records (total: {total_price_inserted})")
            except Exception as e:
                logger.warning(f"Failed to insert remaining price histories: {e}")
        
        logger.info(f"[BATCH-METADATA] ✓ Complete! {len(product_ids)} products created successfully")
        logger.info(f"[BATCH-METADATA] Summary: {total_perf_inserted} performance records, {total_price_inserted} price records inserted")

        # Invalidate CSV cache since products were added
        invalidate_csv_cache()

        return jsonify({
            'success': True,
            'product_ids': product_ids,
            'count': len(product_ids),
            'message': f'Successfully created {len(product_ids)} products',
            'mode': 'metadata_batch'
        }), 200
        
    except Exception as e:
        logger.error(f"Error creating batch metadata products: {e}", exc_info=True)
        return create_error_response(
            'BATCH_CREATION_ERROR',
            str(e),
            'Failed to create batch of metadata products',
            status_code=500
        )


@app.route('/api/metadata-schema', methods=['GET'])
def get_metadata_schema_endpoint():
    """
    Get the current metadata schema (detected columns from CSV upload).

    Returns:
    - 200: Schema with columns and their data types
    - 500: Server error
    """
    try:
        schema = get_metadata_schema()

        return jsonify({
            'success': True,
            'schema': schema,
            'column_count': len(schema)
        }), 200

    except Exception as e:
        logger.error(f"Error getting metadata schema: {e}")
        return create_error_response(
            'SCHEMA_ERROR',
            str(e),
            'Failed to retrieve metadata schema',
            status_code=500
        )


@app.route('/api/metadata-schema', methods=['POST'])
def save_metadata_schema_endpoint():
    try:
        data = request.get_json()

        if not data:
            return create_error_response(
                'MISSING_DATA',
                'No JSON data provided',
                'Please provide schema columns in JSON format',
                status_code=400
            )

        columns = data.get('columns', [])
        clear_existing = data.get('clear_existing', True)

        if not columns:
            return create_error_response(
                'MISSING_COLUMNS',
                'No columns provided',
                'Please provide at least one column definition',
                status_code=400
            )

        # Validate column structure
        for col in columns:
            if 'column_name' not in col:
                return create_error_response(
                    'INVALID_COLUMN',
                    'Each column must have a column_name',
                    'Provide column_name for all columns',
                    status_code=400
                )

        # Clear existing schema if requested
        if clear_existing:
            clear_metadata_schema()

        # Save new schema
        success = save_metadata_schema(columns)

        if success:
            logger.debug(f"Saved metadata schema with {len(columns)} columns")
            return jsonify({
                'success': True,
                'message': f'Saved {len(columns)} column definitions',
                'columns': columns
            }), 200
        else:
            return create_error_response(
                'SCHEMA_SAVE_ERROR',
                'Failed to save schema',
                'Database operation failed',
                status_code=500
            )

    except Exception as e:
        logger.error(f"Error saving metadata schema: {e}")
        return create_error_response(
            'SCHEMA_ERROR',
            str(e),
            'Failed to save metadata schema',
            status_code=500
        )


@app.route('/api/metadata-schema', methods=['DELETE'])
def clear_metadata_schema_endpoint():
    """
    Clear the metadata schema.

    Returns:
    - 200: Schema cleared
    - 500: Server error
    """
    try:
        success = clear_metadata_schema()

        if success:
            logger.debug("Cleared metadata schema")
            return jsonify({
                'success': True,
                'message': 'Metadata schema cleared'
            }), 200
        else:
            return create_error_response(
                'SCHEMA_CLEAR_ERROR',
                'Failed to clear schema',
                'Database operation failed',
                status_code=500
            )

    except Exception as e:
        logger.error(f"Error clearing metadata schema: {e}")
        return create_error_response(
            'SCHEMA_ERROR',
            str(e),
            'Failed to clear metadata schema',
            status_code=500
        )


@app.route('/api/products/upload', methods=['POST'])
def upload_product():

    try:
        # Validate image file is present
        if 'image' not in request.files:
            return create_error_response(
                'MISSING_IMAGE',
                'No image file provided',
                'Please upload an image file (JPEG, PNG, or WebP)',
                status_code=400
            )
        
        file = request.files['image']
        
        # Check if file was actually selected
        if file.filename == '':
            return create_error_response(
                'EMPTY_FILENAME',
                'No file selected',
                'Please select an image file to upload',
                status_code=400
            )
        
        # Validate file extension
        if not allowed_file(file.filename):
            return create_error_response(
                'INVALID_FORMAT',
                f'Unsupported file format',
                'Supported formats: JPEG, PNG, WebP',
                {'filename': file.filename},
                status_code=400
            )
        
        # Get optional fields
        category = request.form.get('category', None)
        product_name = request.form.get('product_name', None)
        sku = request.form.get('sku', None)
        is_historical = request.form.get('is_historical', 'false').lower() == 'true'
        
        # Get performance history (simple format: JSON array of numbers)
        performance_history_str = request.form.get('performance_history', None)
        performance_history = None
        if performance_history_str:
            try:
                import json
                performance_history = json.loads(performance_history_str)
                if not isinstance(performance_history, list):
                    performance_history = None
            except (json.JSONDecodeError, ValueError, TypeError):
                performance_history = None
        
        # Normalize empty strings to None
        if category and category.strip() == '':
            category = None
        if product_name and product_name.strip() == '':
            product_name = None
        if sku and sku.strip() == '':
            sku = None
        
        # Validate and normalize SKU if provided
        if sku:
            is_valid, error_msg = validate_sku_format(sku)
            if not is_valid:
                return create_error_response(
                    'INVALID_SKU',
                    error_msg,
                    'SKU must be alphanumeric with hyphens/underscores, max 50 characters',
                    {'sku': sku},
                    status_code=400
                )
            
            # Normalize SKU
            sku = normalize_sku(sku)
            
            # Check for duplicate SKU (warn but allow)
            if check_sku_exists(sku):
                logger.warning(f"Duplicate SKU detected: {sku}")
        
        # Handle missing category (default to None/NULL)
        # Apply fuzzy matching for category misspellings
        category_warning = None
        if category is not None:
            from product_matching import normalize_category, fuzzy_match_category
            from database import get_all_categories
            
            normalized_cat = normalize_category(category)
            
            if normalized_cat is not None:
                # Get existing categories
                available_categories = get_all_categories()
                
                if available_categories:
                    # Check if category exists exactly
                    category_exists = any(cat.lower() == normalized_cat.lower() for cat in available_categories)
                    
                    if not category_exists:
                        # Try fuzzy matching
                        fuzzy_match = fuzzy_match_category(normalized_cat, available_categories, threshold=2)
                        
                        if fuzzy_match:
                            original_category = category
                            category = fuzzy_match
                            category_warning = f"Category '{original_category}' corrected to '{fuzzy_match}' (similar existing category)"
                            logger.info(f"Fuzzy matched upload category '{original_category}' to '{fuzzy_match}'")
                        else:
                            # New category - that's okay
                            category = normalized_cat
                            logger.info(f"New category '{category}' will be added to catalog")
                    else:
                        # Normalize to match existing case
                        for cat in available_categories:
                            if cat.lower() == normalized_cat.lower():
                                category = cat
                                break
            else:
                category = None
        
        if category is None:
            logger.info("Product uploaded without category, will be stored as NULL")
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            logger.info(f"File saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            return create_error_response(
                'FILE_SAVE_ERROR',
                'Failed to save uploaded file',
                'Please try uploading again',
                {'error': str(e)},
                status_code=500
            )
        
        # Validate image file
        is_valid, error_msg, error_code = validate_image_file(filepath)
        if not is_valid:
            # Clean up invalid file
            try:
                os.remove(filepath)
            except OSError:
                pass

            return create_error_response(
                error_code,
                error_msg,
                'Please upload a valid image file',
                status_code=400
            )
        
        # Insert product into database
        try:
            product_id = insert_product(
                image_path=filepath,
                category=category,
                product_name=product_name,
                sku=sku,
                is_historical=is_historical
            )
            logger.info(f"Product inserted with ID: {product_id}")
        except Exception as e:
            # Clean up file on database error
            try:
                os.remove(filepath)
            except OSError:
                pass
            
            logger.error(f"Database error inserting product: {e}")
            return create_error_response(
                'DATABASE_ERROR',
                'Failed to save product to database',
                'Please try again',
                {'error': str(e)},
                status_code=500
            )
        
        # Extract features from image (CLIP or legacy)
        feature_extraction_status = 'success'
        feature_error = None
        
        try:
            logger.info(f"[UPLOAD-SINGLE] [EXTRACT] ▶ Starting feature extraction for product {product_id}")
            logger.info(f"[UPLOAD-SINGLE] [EXTRACT] Using batch extraction internally (batch_size=1)")
            
            features, embedding_type, embedding_version = extract_features_unified(filepath)
            
            logger.info(f"[UPLOAD-SINGLE] [EXTRACT] ✓ Extraction complete (type: {embedding_type}, version: {embedding_version})")
            
            # Store features in database with embedding type and version
            insert_features(
                product_id=product_id,
                color_features=features['color_features'],
                shape_features=features['shape_features'],
                texture_features=features['texture_features'],
                embedding_type=embedding_type,
                embedding_version=embedding_version
            )
            logger.info(f"[UPLOAD-SINGLE] [EXTRACT] ✓ Features stored in database for product {product_id}")
            
            # Rebuild FAISS index for this category (new product added)
            if embedding_type == 'clip' and is_historical:
                try:
                    from database import rebuild_faiss_index_for_category
                    rebuild_faiss_index_for_category(category)
                    logger.debug(f"Rebuilt FAISS index for category '{category}' after adding product {product_id}")
                except Exception as e:
                    logger.warning(f"Failed to rebuild FAISS index: {e}")
            
        except (InvalidImageFormatError, CorruptedImageError, ImageTooSmallError, ImageProcessingFailedError) as e:
            logger.error(f"Feature extraction failed for product {product_id}: {e.message}")
            feature_extraction_status = 'failed'
            feature_error = {
                'error': e.message,
                'error_code': e.error_code,
                'suggestion': e.suggestion
            }
        except Exception as e:
            logger.error(f"Unexpected error during feature extraction: {e}")
            feature_extraction_status = 'failed'
            feature_error = {
                'error': str(e),
                'error_code': 'UNKNOWN_ERROR',
                'suggestion': 'Please try re-uploading the image'
            }
        
        # Add performance history if provided (simple format: array of numbers)
        if performance_history and isinstance(performance_history, list):
            try:
                from database import bulk_insert_performance_history
                from datetime import timedelta
                
                # Convert simple numbers to complex format with auto-generated dates
                performance_records = []
                today = datetime.now()
                
                for i, perf_value in enumerate(performance_history):
                    if isinstance(perf_value, (int, float)) and perf_value >= 0:
                        # Generate monthly dates going backwards
                        date_obj = today - timedelta(days=30 * (len(performance_history) - 1 - i))
                        date_str = date_obj.strftime('%Y-%m-%d')
                        
                        # Simple format: just sales numbers, rest are 0
                        performance_records.append({
                            'date': date_str,
                            'sales': int(perf_value),
                            'views': 0,
                            'conversion_rate': 0.0,
                            'revenue': 0.0
                        })
                
                if performance_records:
                    bulk_insert_performance_history(product_id, performance_records)
                    logger.info(f"Added {len(performance_records)} performance history records for product {product_id}")
            except Exception as e:
                logger.warning(f"Failed to add performance history for product {product_id}: {e}")
        
        # Prepare response
        response = {
            'status': 'success',
            'product_id': product_id,
            'feature_extraction_status': feature_extraction_status
        }
        
        if feature_error:
            response['feature_extraction_error'] = feature_error
            response['warning'] = 'Product saved but feature extraction failed. You can retry feature extraction later.'
        
        if sku and check_sku_exists(sku, exclude_product_id=product_id):
            response['warning_sku'] = f'SKU "{sku}" already exists in database'
        
        if category_warning:
            response['warning_category'] = category_warning

        # Invalidate CSV cache since a product was added
        invalidate_csv_cache()

        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in upload_product: {e}", exc_info=True)
        return create_error_response(
            'UNKNOWN_ERROR',
            'An unexpected error occurred',
            'Please try again or contact support',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/products/batch-upload', methods=['POST'])
def batch_upload_products():
    try:
        logger.info("[BATCH-UPLOAD] Starting batch upload")

        # MEMORY OPTIMIZATION: Check for file_paths first (new approach - no file uploads)
        import json
        file_paths_json = request.form.get('file_paths', None)

        if file_paths_json:
            # New approach: file paths provided, images already on disk
            try:
                file_paths = json.loads(file_paths_json)
                logger.debug(f"[BATCH-UPLOAD] METHOD: NEW (File Paths) - Processing {len(file_paths)} images")
                saved_files = file_paths  # Use paths directly, no saving needed
                files = None  # No uploaded files
            except json.JSONDecodeError as e:
                return create_error_response(
                    'INVALID_JSON',
                    f'Failed to parse file_paths JSON: {str(e)}',
                    'Ensure file_paths is a valid JSON array',
                    status_code=400
                )
        else:
            # Legacy approach: files are uploaded, need to save them
            files = request.files.getlist('images')
            logger.debug(f"[BATCH-UPLOAD] METHOD: LEGACY (Direct Upload) - Processing {len(files) if files else 0} files")
            saved_files = None  # Will be populated below

            if not files or len(files) == 0:
                return create_error_response(
                    'MISSING_IMAGES',
                    'No image files or file_paths provided',
                    'Provide either file_paths array or upload image files',
                    status_code=400
                )
        
        # Determine total files count based on approach
        file_count = len(file_paths) if file_paths_json else len(files)
        logger.debug(f"[BATCH-UPLOAD] Processing {file_count} images ({('file paths' if file_paths_json else 'uploaded files')})")

        # Optional operation context for auto performance profile
        operation_total_files_raw = request.form.get('operation_total_files')
        operation_total_files = None
        if operation_total_files_raw is not None and str(operation_total_files_raw).strip() != '':
            try:
                operation_total_files = int(operation_total_files_raw)
            except (TypeError, ValueError):
                return create_error_response(
                    'INVALID_OPERATION_TOTAL',
                    f'operation_total_files must be an integer, got: {operation_total_files_raw}',
                    'Send operation_total_files as an integer string',
                    status_code=400
                )

        processing_profile = (request.form.get('processing_profile', 'auto') or 'auto').strip().lower()
        if processing_profile not in SUPPORTED_PROCESSING_PROFILES:
            return create_error_response(
                'INVALID_PROCESSING_PROFILE',
                f"Unsupported processing_profile '{processing_profile}'",
                f"Use one of: {', '.join(sorted(SUPPORTED_PROCESSING_PROFILES))}",
                status_code=400
            )

        rebuild_faiss_raw = request.form.get('rebuild_faiss', 'true')
        rebuild_faiss = str(rebuild_faiss_raw).strip().lower() in ('1', 'true', 'yes', 'on')

        effective_total_files = operation_total_files if operation_total_files is not None else file_count
        if effective_total_files > MAX_UPLOAD_FILES_PER_OPERATION:
            return create_error_response(
                'TOO_MANY_FILES',
                f'Upload contains {effective_total_files:,} files, which exceeds the maximum of {MAX_UPLOAD_FILES_PER_OPERATION:,}',
                f'Split uploads into multiple operations of at most {MAX_UPLOAD_FILES_PER_OPERATION:,} files',
                status_code=400
            )

        # Get optional metadata arrays
        categories = request.form.get('categories', None)
        product_names = request.form.get('product_names', None)
        skus = request.form.get('skus', None)
        metadata_list = request.form.get('metadata', None)
        is_historical = request.form.get('is_historical', 'false').lower() == 'true'

        # Parse JSON arrays
        try:
            categories = json.loads(categories) if categories else [None] * file_count
            product_names = json.loads(product_names) if product_names else [None] * file_count
            skus = json.loads(skus) if skus else [None] * file_count
            metadata_list = json.loads(metadata_list) if metadata_list else [None] * file_count

            # Log categories for debugging
            unique_categories = set(c for c in categories if c is not None)
            logger.debug(f"[BATCH-UPLOAD] Categories received: {len(unique_categories)} unique categories from {len(categories)} products")
            if unique_categories:
                logger.debug(f"[BATCH-UPLOAD] Unique categories: {sorted(unique_categories)}")
        except json.JSONDecodeError as e:
            return create_error_response(
                'INVALID_JSON',
                f'Failed to parse metadata JSON: {str(e)}',
                'Ensure categories, product_names, and skus are valid JSON arrays',
                status_code=400
            )

        # Validate array lengths
        if len(categories) == 1 and file_count > 1:
            # Single category for all products
            categories = categories * file_count

        if len(categories) != file_count:
            return create_error_response(
                'ARRAY_LENGTH_MISMATCH',
                f'categories array length ({len(categories)}) does not match number of images ({file_count})',
                'Provide one category per image or a single category for all',
                status_code=400
            )

        if len(product_names) != file_count:
            return create_error_response(
                'ARRAY_LENGTH_MISMATCH',
                f'product_names array length ({len(product_names)}) does not match number of images ({file_count})',
                'Provide one product name per image',
                status_code=400
            )

        if len(skus) != file_count:
            return create_error_response(
                'ARRAY_LENGTH_MISMATCH',
                f'skus array length ({len(skus)}) does not match number of images ({file_count})',
                'Provide one SKU per image',
                status_code=400
            )

        if len(metadata_list) != file_count:
            return create_error_response(
                'ARRAY_LENGTH_MISMATCH',
                f'metadata array length ({len(metadata_list)}) does not match number of images ({file_count})',
                'Provide one metadata object per image',
                status_code=400
            )

        # Step 1: If using file paths (new approach), validate them directly. Otherwise, save uploaded files.
        if file_paths_json:
            # MEMORY OPTIMIZED: File paths provided, validate directly without saving
            logger.debug("[BATCH-UPLOAD] Step 1: Validating file paths (MEMORY OPTIMIZED - no uploads)")

            valid_files = []
            file_indices = []
            skipped_files = []

            for i, filepath in enumerate(file_paths):
                try:
                    # Check file exists
                    if not os.path.exists(filepath):
                        logger.warning(f"[BATCH-UPLOAD] File path {i+1}: Does not exist: {filepath}")
                        skipped_files.append({'index': i, 'filename': os.path.basename(filepath), 'reason': 'File not found'})
                        continue

                    # Validate image
                    is_valid, error_msg, error_code = validate_image_file(filepath)
                    if not is_valid:
                        logger.warning(f"[BATCH-UPLOAD] File path {i+1}: {error_msg} ({filepath})")
                        skipped_files.append({'index': i, 'filename': os.path.basename(filepath), 'reason': error_msg})
                        continue

                    valid_files.append(filepath)
                    file_indices.append(i)

                except Exception as e:
                    logger.error(f"[BATCH-UPLOAD] File path {i+1}: Error: {e}")
                    skipped_files.append({'index': i, 'filename': os.path.basename(filepath), 'reason': str(e)})
                    continue

            saved_files = valid_files
            logger.debug(f"[BATCH-UPLOAD] Step 1: Validated {len(saved_files)} files (skipped {len(skipped_files)})")
        else:
            # LEGACY: Files uploaded, need to save them and validate
            logger.debug("[BATCH-UPLOAD] Step 1: Saving and validating uploaded files")

            def process_files_batch(files_to_process, file_indices_to_process, attempt=1):
                """Process a batch of files, return (saved_files, file_indices, skipped_files)"""
                saved = []
                indices = []
                skipped = []

                for idx, (i, file) in enumerate(zip(file_indices_to_process, files_to_process)):
                    try:
                        if file.filename == '':
                            logger.warning(f"[BATCH-UPLOAD] Attempt {attempt}: Skipping file {i+1}: Empty filename")
                            skipped.append({'index': i, 'filename': 'unknown', 'reason': 'Empty filename'})
                            continue

                        if not allowed_file(file.filename):
                            logger.warning(f"[BATCH-UPLOAD] Attempt {attempt}: Skipping file {i+1}: Unsupported format ({file.filename})")
                            skipped.append({'index': i, 'filename': file.filename, 'reason': 'Unsupported format'})
                            continue

                        # Save file
                        filename = secure_filename(file.filename)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        unique_filename = f"{timestamp}_{i}_{filename}"
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

                        file.save(filepath)
                        logger.debug(f"[BATCH-UPLOAD] Attempt {attempt}: Saved file {i+1}/{len(files_to_process)}: {filepath}")

                        # Validate image
                        is_valid, error_msg, error_code = validate_image_file(filepath)
                        if not is_valid:
                            logger.warning(f"[BATCH-UPLOAD] Attempt {attempt}: Skipping file {i+1}: {error_msg}")
                            skipped.append({'index': i, 'filename': file.filename, 'reason': error_msg})
                            try:
                                os.remove(filepath)
                            except OSError:
                                pass
                            continue

                        saved.append(filepath)
                        indices.append(i)

                    except Exception as e:
                        logger.error(f"[BATCH-UPLOAD] Attempt {attempt}: Error processing file {i+1}: {e}")
                        skipped.append({'index': i, 'filename': file.filename, 'reason': str(e)})
                        continue

                return saved, indices, skipped

            # First attempt: process all files
            saved_files, file_indices, skipped_files = process_files_batch(files, list(range(len(files))), attempt=1)

            if len(skipped_files) > 0:
                logger.debug(f"[BATCH-UPLOAD] Attempt 1: Skipped {len(skipped_files)} files, processing {len(saved_files)} valid files")

                # Retry skipped files once
                logger.debug(f"[BATCH-UPLOAD] Retrying {len(skipped_files)} skipped files (Attempt 2)")
                retry_files = [files[s['index']] for s in skipped_files]
                retry_indices = [s['index'] for s in skipped_files]

                retry_saved, retry_indices_result, retry_skipped = process_files_batch(retry_files, retry_indices, attempt=2)

                # Merge retry results
                saved_files.extend(retry_saved)
                file_indices.extend(retry_indices_result)

                # Update skipped list with files that failed retry
                skipped_files = retry_skipped

                if len(retry_saved) > 0:
                    logger.debug(f"[BATCH-UPLOAD] Retry successful: {len(retry_saved)} files recovered, {len(retry_skipped)} still skipped")
                else:
                    logger.debug(f"[BATCH-UPLOAD] Retry failed: All {len(retry_skipped)} files still invalid")

        if len(saved_files) == 0:
            return create_error_response(
                'NO_VALID_FILES',
                'No valid image files found in batch',
                'All files were invalid or skipped',
                status_code=400
            )

        logger.debug(f"[BATCH-UPLOAD] {len(saved_files)} files saved and validated")

        # CRASH RESUME: Skip files already in the DB with extracted features.
        # When the app crashes mid-upload, re-uploading the same folder would create
        # duplicate products. This deduplication check compares file paths to skip
        # already-processed images, making re-upload safe after crashes.
        skip_existing_raw = request.form.get('skip_existing', 'false')
        skip_existing = str(skip_existing_raw).strip().lower() in ('1', 'true', 'yes', 'on')

        skipped_existing_count = 0
        if skip_existing and file_paths_json:
            try:
                from database import get_db_connection as _get_conn
                with _get_conn() as conn:
                    cursor = conn.cursor()
                    # Find which file paths already exist in products table with features
                    existing_paths = set()
                    for chunk_start in range(0, len(saved_files), 900):
                        chunk = saved_files[chunk_start:chunk_start + 900]
                        placeholders = ','.join('?' * len(chunk))
                        cursor.execute(f'''
                            SELECT p.image_path FROM products p
                            INNER JOIN features f ON p.id = f.product_id
                            WHERE p.image_path IN ({placeholders})
                        ''', chunk)
                        existing_paths.update(row[0] for row in cursor.fetchall())

                if existing_paths:
                    new_saved = []
                    new_indices = []
                    for i, fpath in enumerate(saved_files):
                        if fpath in existing_paths:
                            skipped_existing_count += 1
                        else:
                            new_saved.append(fpath)
                            new_indices.append(file_indices[i])
                    saved_files = new_saved
                    file_indices = new_indices
                    logger.info(f"[BATCH-UPLOAD] Crash resume: skipped {skipped_existing_count} already-processed images")

                    if len(saved_files) == 0:
                        return jsonify({
                            'status': 'success',
                            'message': 'All images were already processed',
                            'total': skipped_existing_count,
                            'skipped_existing': skipped_existing_count,
                            'successful': 0,
                            'failed': 0,
                            'results': []
                        }), 200
            except Exception as e:
                logger.warning(f"[BATCH-UPLOAD] Skip-existing check failed (continuing without): {e}")

        # Step 2: Insert products into database (THREAD-SAFE: Bulk insert in single transaction)
        logger.debug("[BATCH-UPLOAD] Step 2: Inserting products into database (bulk insert)")

        # THREAD SAFETY & PERFORMANCE:
        # Use bulk_insert_products() instead of parallel insert_product() calls
        # - SQLite can only handle ONE write at a time (even with WAL mode)
        # - Parallel writes cause lock contention and SQLITE_BUSY errors
        # - Bulk insert is faster: single transaction instead of N transactions
        # - This is the CORRECT approach for SQLite batch operations

        from database import bulk_insert_products
        from product_matching import normalize_category
        import json

        # Prepare all product data for bulk insert
        products_to_insert = []
        metadata_category_backfilled = 0
        for i, filepath in enumerate(saved_files):
            original_idx = file_indices[i]
            category = categories[original_idx]
            product_name = product_names[original_idx]
            sku = skus[original_idx]
            metadata = metadata_list[original_idx]

            # Normalize empty strings to None
            if category and str(category).strip() == '':
                category = None
            if product_name and str(product_name).strip() == '':
                product_name = None
            if sku and str(sku).strip() == '':
                sku = None

            # Normalize category (lowercase, trim whitespace, handle "unknown" variations)
            if category is not None:
                category = normalize_category(category)

            # Fallback: if category is still missing after normalization,
            # derive it from linked metadata when possible.
            # This improves Mode 1/Mode 3 candidate filtering and FAISS category routing.
            if category is None:
                inferred_category = _extract_category_from_metadata_payload(metadata)
                if inferred_category is not None:
                    category = inferred_category
                    metadata_category_backfilled += 1

            # Validate SKU if provided
            if sku:
                is_valid, error_msg = validate_sku_format(sku)
                if not is_valid:
                    logger.warning(f"[BATCH-UPLOAD] Invalid SKU for file {i+1}: {error_msg}")
                    sku = None
                else:
                    sku = normalize_sku(sku)

            # Serialize metadata if dict
            if metadata and isinstance(metadata, dict):
                metadata = json.dumps(metadata)

            # Add to bulk insert list: (image_path, category, product_name, sku, is_historical, metadata)
            products_to_insert.append((filepath, category, product_name, sku, is_historical, metadata))

        if metadata_category_backfilled > 0:
            logger.info(
                f"[BATCH-UPLOAD] Backfilled category from metadata for "
                f"{metadata_category_backfilled}/{len(saved_files)} products"
            )

        # Bulk insert all products in single transaction (THREAD-SAFE, FAST)
        try:
            product_ids = bulk_insert_products(products_to_insert)
            inserted_count = len(product_ids)
            logger.debug(f"[BATCH-UPLOAD] ✓ Bulk inserted {inserted_count}/{len(saved_files)} products (single transaction)")
        except Exception as e:
            logger.error(f"[BATCH-UPLOAD] Bulk insert failed: {e}")
            # Fallback: insert one by one (slower but more resilient)
            logger.debug("[BATCH-UPLOAD] Falling back to sequential insert...")
            product_ids = []
            for i, (filepath, category, product_name, sku, is_hist, meta) in enumerate(products_to_insert):
                try:
                    product_id = insert_product(
                        image_path=filepath,
                        category=category,
                        product_name=product_name,
                        sku=sku,
                        is_historical=is_hist,
                        metadata=meta
                    )
                    product_ids.append(product_id)
                    logger.debug(f"[BATCH-UPLOAD] Fallback inserted product {i+1}/{len(products_to_insert)}")
                except Exception as e2:
                    logger.error(f"[BATCH-UPLOAD] Failed to insert product {i+1}: {e2}")
                    product_ids.append(None)
            inserted_count = sum(1 for pid in product_ids if pid is not None)
            logger.debug(f"[BATCH-UPLOAD] Fallback complete: {inserted_count}/{len(saved_files)} products inserted")
        
        # Step 3: Extract features in batch (GPU-optimized parallel processing)
        logger.debug("[BATCH-UPLOAD] Step 3: Extracting features in batch (GPU-optimized)")
        
        from feature_extraction_service import batch_extract_features_unified
        
        # Only extract features for successfully inserted products
        valid_indices = [i for i, pid in enumerate(product_ids) if pid is not None]
        valid_filepaths = [saved_files[i] for i in valid_indices]
        
        total_inserted = 0
        features_failed_count = 0
        feature_extraction_profile = {
            'requested_profile': processing_profile,
            'profile_used': processing_profile
        }

        if valid_filepaths:
            feature_results, feature_extraction_profile = batch_extract_features_unified(
                valid_filepaths,
                processing_profile=processing_profile,
                operation_total_files=effective_total_files
            )
            
            # Step 4: Store features in database - INCREMENTAL BATCH INSERT
            logger.debug("[BATCH-UPLOAD] Step 4: Storing features in database (incremental batch insert)")
            
            from database import serialize_numpy_array, bulk_insert_features
            
            # Collect features for batch insert (incremental to avoid memory bloat)
            features_to_insert = []
            INCREMENTAL_BATCH_SIZE = 32  # Insert every 32 features (matches GPU batch size)
            
            for idx, (filepath, features_dict, embedding_type, embedding_version, error_msg) in enumerate(feature_results):
                original_idx = valid_indices[idx]
                product_id = product_ids[original_idx]
                
                if features_dict is not None:
                    try:
                        # Serialize numpy arrays to bytes
                        color_blob = serialize_numpy_array(features_dict['color_features'])
                        shape_blob = serialize_numpy_array(features_dict['shape_features'])
                        texture_blob = serialize_numpy_array(features_dict['texture_features'])
                        
                        # Add to batch
                        features_to_insert.append((
                            product_id,
                            color_blob,
                            shape_blob,
                            texture_blob,
                            embedding_type,
                            embedding_version
                        ))
                        logger.debug(f"[BATCH-UPLOAD] Collected features for product {product_id}")
                        
                        # OPTIMIZATION: Insert incrementally to match GPU batch size
                        # This starts inserting while GPU is still processing remaining images
                        if len(features_to_insert) >= INCREMENTAL_BATCH_SIZE:
                            try:
                                inserted_count = bulk_insert_features(features_to_insert)
                                total_inserted += inserted_count
                                logger.debug(f"[BATCH-UPLOAD] Incremental insert: {inserted_count} features (total: {total_inserted})")
                                features_to_insert = []  # Clear for next batch
                            except Exception as e:
                                logger.warning(f"[BATCH-UPLOAD] Incremental insert failed: {e}, will retry at end")
                    except Exception as e:
                        logger.error(f"[BATCH-UPLOAD] Failed to serialize features for product {product_id}: {e}")
                else:
                    features_failed_count += 1
                    logger.warning(f"[BATCH-UPLOAD] Feature extraction failed for product {product_id}: {error_msg}")
            
            # Batch insert remaining features
            if features_to_insert:
                try:
                    inserted_count = bulk_insert_features(features_to_insert)
                    total_inserted += inserted_count
                    logger.debug(f"[BATCH-UPLOAD] ✓ Final batch inserted {inserted_count} remaining feature records (total: {total_inserted})")
                except Exception as e:
                    logger.error(f"[BATCH-UPLOAD] Failed to batch insert remaining features: {e}")
        
        # Step 5: FAISS index handling
        # PERFORMANCE: Only rebuild FAISS for historical uploads. FAISS indexes
        # only contain historical products (is_historical=True), so rebuilding
        # after a new-product upload is wasted work (50k products = seconds wasted).
        if inserted_count > 0:
            if rebuild_faiss and is_historical:
                logger.debug("[BATCH-UPLOAD] Step 5: Scheduling FAISS index rebuild (background, historical upload)")

                def rebuild_indexes_background():
                    """Rebuild FAISS indexes in background thread"""
                    try:
                        logger.debug("[BATCH-UPLOAD-BG] Starting background FAISS index rebuild...")
                        from database import rebuild_all_faiss_indexes
                        rebuild_all_faiss_indexes()
                        logger.debug("[BATCH-UPLOAD-BG] ✓ FAISS indexes rebuilt successfully")
                    except Exception as e:
                        logger.warning(f"[BATCH-UPLOAD-BG] Failed to rebuild FAISS indexes: {e}")

                # Start background thread (don't wait for it)
                import threading
                bg_thread = threading.Thread(target=rebuild_indexes_background, daemon=True)
                bg_thread.start()
                logger.debug("[BATCH-UPLOAD] FAISS index rebuild scheduled in background")
            elif rebuild_faiss and not is_historical:
                logger.debug("[BATCH-UPLOAD] Step 5: Skipping FAISS rebuild (new products don't affect FAISS index)")
            else:
                # Safety: invalidate stale in-memory indexes so matching remains correct.
                # Matching will fall back to brute force until a rebuild is requested.
                try:
                    from faiss_index import faiss_manager
                    faiss_manager.invalidate()
                    logger.debug("[BATCH-UPLOAD] Deferred FAISS rebuild for this batch (indexes invalidated)")
                except Exception as e:
                    logger.warning(f"[BATCH-UPLOAD] Failed to invalidate FAISS indexes during deferred rebuild: {e}")
        
        # Prepare response
        results = []

        # Add successful products
        for i, product_id in enumerate(product_ids):
            original_idx = file_indices[i]
            # Get filename - either from files object (legacy upload) or from saved_files (file paths)
            if files is not None:
                filename = files[original_idx].filename
            else:
                # File paths approach - extract basename from filepath
                filename = os.path.basename(saved_files[i])

            if product_id is not None:
                results.append({
                    'index': original_idx,
                    'status': 'success',
                    'product_id': product_id,
                    'filename': filename
                })
            else:
                results.append({
                    'index': original_idx,
                    'status': 'failed',
                    'error': 'Database insertion failed',
                    'filename': filename
                })

        # Add skipped files
        for skipped in skipped_files:
            results.append({
                'index': skipped['index'],
                'status': 'skipped',
                'reason': skipped['reason'],
                'filename': skipped['filename']
            })

        success_count = sum(1 for r in results if r['status'] == 'success')
        skipped_count = len(skipped_files)
        failed_count = sum(1 for r in results if r['status'] == 'failed')

        logger.info(f"[BATCH-UPLOAD] ✓ Complete! {success_count} successful, {failed_count} failed, {skipped_count} skipped")

        # Total count based on approach used
        total_count = len(file_paths) if file_paths_json else len(files)
        
        response_data = {
            'status': 'success',
            'total': total_count,
            'successful': success_count,
            'failed': failed_count,
            'skipped': skipped_count,
            'results': results,
            'features_extracted': total_inserted,
            'features_failed': features_failed_count,
            'categories_backfilled_from_metadata': metadata_category_backfilled,
            'processing_profile_requested': processing_profile,
            'processing_profile_used': feature_extraction_profile.get('profile_used', processing_profile),
            'processing_profile_reason': feature_extraction_profile.get('reason'),
            'processing_clip_model': feature_extraction_profile.get('clip_model_name')
        }
        
        logger.debug(f"[BATCH-UPLOAD] Returning JSON response: status={response_data['status']}, total={response_data['total']}, successful={response_data['successful']}, failed={response_data['failed']}, skipped={response_data['skipped']}, results_count={len(response_data['results'])}")

        # Invalidate CSV cache since products were added
        invalidate_csv_cache()

        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"[BATCH-UPLOAD] Unexpected error: {e}", exc_info=True)
        return create_error_response(
            'BATCH_UPLOAD_ERROR',
            f'Batch upload failed: {str(e)}',
            'Please try again',
            status_code=500
        )


@app.route('/api/products/match', methods=['POST'])
def match_products():
    try:
        # Parse JSON body
        data = request.get_json()
        
        if not data:
            return create_error_response(
                'MISSING_BODY',
                'Request body is required',
                'Send JSON body with product_id',
                status_code=400
            )
        
        # Validate product_id
        if 'product_id' not in data:
            return create_error_response(
                'MISSING_PRODUCT_ID',
                'product_id is required',
                'Include product_id in request body',
                status_code=400
            )
        
        try:
            product_id = int(data['product_id'])
        except (ValueError, TypeError):
            return create_error_response(
                'INVALID_PRODUCT_ID',
                'product_id must be an integer',
                f'Received: {data["product_id"]}',
                status_code=400
            )
        
        # Get optional parameters with validation
        threshold = data.get('threshold', 0.0)
        try:
            threshold = float(threshold)
            if not 0 <= threshold <= 100:
                return create_error_response(
                    'INVALID_THRESHOLD',
                    'threshold must be between 0 and 100',
                    f'Received: {threshold}',
                    status_code=400
                )
        except (ValueError, TypeError):
            return create_error_response(
                'INVALID_THRESHOLD',
                'threshold must be a number',
                f'Received: {threshold}',
                status_code=400
            )
        
        limit = data.get('limit', 10)
        try:
            limit = int(limit)
            if limit < 0:
                return create_error_response(
                    'INVALID_LIMIT',
                    'limit must be non-negative',
                    f'Received: {limit}',
                    status_code=400
                )
            if limit > 100:
                logger.warning(f"Limit {limit} exceeds maximum, capping at 100")
                limit = 100
        except (ValueError, TypeError):
            return create_error_response(
                'INVALID_LIMIT',
                'limit must be an integer',
                f'Received: {limit}',
                status_code=400
            )
        
        match_against_all = data.get('match_against_all', False)
        if not isinstance(match_against_all, bool):
            match_against_all = str(match_against_all).lower() in ['true', '1', 'yes']
        
        # Get optional similarity weights
        color_weight = data.get('color_weight', 0.5)
        shape_weight = data.get('shape_weight', 0.3)
        texture_weight = data.get('texture_weight', 0.2)
        
        # Validate weights
        try:
            color_weight = float(color_weight)
            shape_weight = float(shape_weight)
            texture_weight = float(texture_weight)
            
            # Check if weights sum to 1.0 (with tolerance)
            total_weight = color_weight + shape_weight + texture_weight
            if not (0.99 <= total_weight <= 1.01):
                return create_error_response(
                    'INVALID_WEIGHTS',
                    f'Similarity weights must sum to 1.0, got {total_weight:.3f}',
                    'Adjust weights so they sum to 100%',
                    status_code=400
                )
            
            if color_weight < 0 or shape_weight < 0 or texture_weight < 0:
                return create_error_response(
                    'INVALID_WEIGHTS',
                    'Similarity weights must be non-negative',
                    'All weights must be >= 0',
                    status_code=400
                )
        except (ValueError, TypeError):
            return create_error_response(
                'INVALID_WEIGHTS',
                'Similarity weights must be numbers',
                f'Received: color={color_weight}, shape={shape_weight}, texture={texture_weight}',
                status_code=400
            )
        
        # Get metadata weights - support both dynamic dict and legacy individual weights
        metadata_weights = data.get('metadata_weights', None)

        if metadata_weights is not None:
            # Dynamic weights mode - normalize and validate
            if not isinstance(metadata_weights, dict):
                return create_error_response(
                    'INVALID_METADATA_WEIGHTS',
                    'metadata_weights must be a dictionary',
                    'Provide weights as {"column_name": weight_value}',
                    status_code=400
                )

            # Validate all weights are non-negative numbers
            try:
                for col, weight in metadata_weights.items():
                    if float(weight) < 0:
                        return create_error_response(
                            'INVALID_METADATA_WEIGHTS',
                            f'Weight for {col} must be non-negative',
                            'All weights must be >= 0',
                            status_code=400
                        )
                # Normalize to floats
                metadata_weights = {k: float(v) for k, v in metadata_weights.items()}
            except (ValueError, TypeError) as e:
                return create_error_response(
                    'INVALID_METADATA_WEIGHTS',
                    'All weights must be numbers',
                    str(e),
                    status_code=400
                )

            # Normalize weights to sum to 1.0
            total_weight = sum(metadata_weights.values())
            if total_weight > 0:
                metadata_weights = {k: v / total_weight for k, v in metadata_weights.items()}

            logger.info(f"[MATCH] Using dynamic metadata weights: {metadata_weights}")
        else:
            logger.warning("[MATCH] No metadata_weights provided in request")
        
        # Get hybrid weight
        visual_weight = data.get('visual_weight', 0.50)
        metadata_weight = data.get('metadata_weight', 0.50)
        
        # Validate hybrid weights
        try:
            visual_weight = float(visual_weight)
            metadata_weight = float(metadata_weight)
            
            # Check if weights sum to 1.0 (with tolerance)
            total_hybrid_weight = visual_weight + metadata_weight
            if not (0.99 <= total_hybrid_weight <= 1.01):
                return create_error_response(
                    'INVALID_HYBRID_WEIGHTS',
                    f'Hybrid weights must sum to 1.0, got {total_hybrid_weight:.3f}',
                    'Adjust visual_weight and metadata_weight so they sum to 100%',
                    status_code=400
                )
            
            if visual_weight < 0 or metadata_weight < 0:
                return create_error_response(
                    'INVALID_HYBRID_WEIGHTS',
                    'Hybrid weights must be non-negative',
                    'All weights must be >= 0',
                    status_code=400
                )
        except (ValueError, TypeError):
            return create_error_response(
                'INVALID_HYBRID_WEIGHTS',
                'Hybrid weights must be numbers',
                f'Received: visual={visual_weight}, metadata={metadata_weight}',
                status_code=400
            )
        
        # Detect matching mode based on:
        # 1. Visual features presence
        # 2. Slider weights (if metadata_weight > 0, user wants hybrid)
        try:
            features = get_features_by_product_id(product_id)
            has_features = features is not None
        except Exception:
            has_features = False
        
        # Determine mode based on features and weights
        # Mode 1: Has features + metadata_weight = 0 → Pure visual
        # Mode 2: No features → Pure metadata
        # Mode 3: Has features + metadata_weight > 0 → Hybrid
        
        use_hybrid = has_features and metadata_weight > 0
        
        # Find matches with comprehensive error handling
        try:
            if use_hybrid:
                # Mode 3: Hybrid matching (visual + metadata)
                logger.info(f"Product {product_id} using hybrid matching (visual: {visual_weight*100}%, metadata: {metadata_weight*100}%)")

                result = find_hybrid_matches(
                    product_id=product_id,
                    threshold=threshold,
                    limit=limit,
                    visual_weight=visual_weight,
                    metadata_weight=metadata_weight,
                    metadata_weights=metadata_weights,
                    store_matches=True,
                    skip_invalid_products=True,
                    match_against_all=match_against_all
                )
            elif has_features:
                # Mode 1: Pure visual matching
                logger.info(f"Product {product_id} using visual matching")
                result = find_matches(
                    product_id=product_id,
                    threshold=threshold,
                    limit=limit,
                    match_against_all=match_against_all,
                    include_uncategorized=True,
                    store_matches=True,
                    skip_invalid_products=True,
                    color_weight=color_weight,
                    shape_weight=shape_weight,
                    texture_weight=texture_weight
                )
            else:
                # Mode 2: Metadata matching only (no visual features)
                logger.info(f"Product {product_id} has no visual features, using metadata matching")

                result = find_metadata_matches(
                    product_id=product_id,
                    threshold=threshold,
                    limit=limit,
                    weights=metadata_weights,
                    store_matches=True,
                    skip_invalid_products=True,
                    match_against_all=match_against_all
                )
            
            # Prepare response
            response = {
                'status': 'success',
                'product_id': product_id,
                'matches': result['matches'],
                'total_candidates': result['total_candidates'],
                'successful_matches': result['successful_matches'],
                'failed_matches': result['failed_matches'],
                'filtered_by_threshold': result['filtered_by_threshold'],
                'threshold': threshold,
                'limit': limit
            }
            
            # Include warnings if any
            if result.get('warnings'):
                response['warnings'] = result['warnings']
            
            # Include error details for failed matches if any
            if result.get('errors'):
                response['partial_failures'] = result['errors']
                response['note'] = 'Some matches failed due to data quality issues. See partial_failures for details.'
            
            # Include data quality summary
            if result.get('data_quality_summary'):
                response['data_quality'] = result['data_quality_summary']
            
            return jsonify(response), 200
            
        except ProductNotFoundError as e:
            return create_error_response(
                e.error_code,
                e.message,
                e.suggestion,
                status_code=404
            )
        
        except MissingFeaturesError as e:
            return create_error_response(
                e.error_code,
                e.message,
                e.suggestion,
                {'product_id': product_id},
                status_code=400
            )
        
        except EmptyCatalogError as e:
            # Return empty results with message (not an error)
            return jsonify({
                'status': 'success',
                'product_id': product_id,
                'matches': [],
                'total_candidates': 0,
                'message': e.message,
                'suggestion': e.suggestion
            }), 200
        
        except AllMatchesFailedError as e:
            return create_error_response(
                e.error_code,
                e.message,
                e.suggestion,
                status_code=500
            )
        
        except MatchingError as e:
            return create_error_response(
                e.error_code,
                e.message,
                e.suggestion,
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in match_products: {e}", exc_info=True)
        return create_error_response(
            'UNKNOWN_ERROR',
            'An unexpected error occurred during matching',
            'Please try again or contact support',
            {'error': str(e)},
            status_code=500
        )

@app.route('/api/products/batch-match', methods=['POST'])
def batch_match_products():

    try:
        data = request.get_json()
        
        # Validate request
        if not data:
            return create_error_response(
                'INVALID_REQUEST',
                'Request body is required',
                'Send JSON with product_ids array',
                status_code=400
            )
        
        # MEMORY OPTIMIZATION: Support match_all_new to avoid frontend loading products
        match_all_new = data.get('match_all_new', False)

        if match_all_new:
            # Query new product IDs from database instead of requiring frontend to send them
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM products WHERE is_historical = 0')
                product_ids = [row['id'] for row in cursor.fetchall()]
            logger.info(f"[BATCH] Queried {len(product_ids)} new products from database")
        else:
            # Use provided product IDs
            product_ids = data.get('product_ids', [])

        if not product_ids or not isinstance(product_ids, list):
            return create_error_response(
                'INVALID_PRODUCT_IDS',
                'product_ids must be a non-empty array or match_all_new must be true',
                'Example: {"product_ids": [1, 2, 3]} or {"match_all_new": true}',
                status_code=400
            )
        
        # Get parameters
        threshold = int(data.get('threshold', 0))
        limit = int(data.get('limit', 10))
        match_against_all = data.get('match_against_all', False)
        
        # Get weights
        visual_weight = float(data.get('visual_weight', 0.5))
        metadata_weight = float(data.get('metadata_weight', 0.5))
        
        # Validate weights
        total_weight = visual_weight + metadata_weight
        if not (0.99 <= total_weight <= 1.01):
            return create_error_response(
                'INVALID_WEIGHTS',
                f'Weights must sum to 1.0, got {total_weight:.3f}',
                'Adjust visual_weight and metadata_weight so they sum to 1.0',
                status_code=400
            )
        
        # Get metadata weights
        metadata_weights = data.get('metadata_weights', None)

        if metadata_weights is not None:
            # Dynamic weights mode - normalize
            if not isinstance(metadata_weights, dict):
                return create_error_response(
                    'INVALID_METADATA_WEIGHTS',
                    'metadata_weights must be a dictionary',
                    'Provide weights as {"column_name": weight_value}',
                    status_code=400
                )
            try:
                metadata_weights = {k: float(v) for k, v in metadata_weights.items()}
                total_meta_weight = sum(metadata_weights.values())
                if total_meta_weight > 0:
                    metadata_weights = {k: v / total_meta_weight for k, v in metadata_weights.items()}
            except (ValueError, TypeError) as e:
                return create_error_response(
                    'INVALID_METADATA_WEIGHTS',
                    'All metadata weights must be numbers',
                    str(e),
                    status_code=400
                )
            logger.info(f"[BATCH] Using dynamic metadata weights: {metadata_weights}")
        else:
            # No weights provided - this will likely fail in matching functions if mode requires them
            logger.warning("[BATCH] No metadata_weights provided in request")

        logger.info(f"[BATCH] Starting batch matching for {len(product_ids)} products")
        logger.info(f"[BATCH] Weights - Visual: {visual_weight*100}%, Metadata: {metadata_weight*100}%")
        logger.info(f"[BATCH] Parameters - Threshold: {threshold}, Limit: {limit}, Match all: {match_against_all}")
        
        # Determine matching mode based on weights
        # Mode 1: visual_weight > 0 and metadata_weight == 0
        # Mode 2: visual_weight == 0 and metadata_weight > 0
        # Mode 3: both visual_weight > 0 and metadata_weight > 0
        is_pure_visual = visual_weight > 0 and metadata_weight == 0
        is_pure_metadata = visual_weight == 0 and metadata_weight > 0
        is_hybrid = visual_weight > 0 and metadata_weight > 0
        
        # Call appropriate batch function
        try:
            if is_hybrid:
                # Mode 3: Hybrid batch matching
                logger.info(f"[BATCH] Mode 3 (Hybrid) - Processing {len(product_ids)} products in parallel")
                logger.info(f"[BATCH] Mode 3 will run Mode 1 (Visual) and Mode 2 (Metadata) simultaneously")

                result = batch_find_hybrid_matches(
                    product_ids=product_ids,
                    threshold=threshold,
                    limit=limit,
                    visual_weight=visual_weight,
                    metadata_weight=metadata_weight,
                    metadata_weights=metadata_weights,
                    store_matches=True,
                    skip_invalid_products=True,
                    match_against_all=match_against_all
                )
            elif is_pure_visual:
                # Mode 1: Visual batch matching
                logger.info(f"[BATCH] Mode 1 (Visual) - Processing {len(product_ids)} products in parallel")
                result = batch_find_matches(
                    product_ids=product_ids,
                    threshold=threshold,
                    limit=limit,
                    match_against_all=match_against_all,
                    include_uncategorized=True,
                    store_matches=True,
                    skip_invalid_products=True,
                    preload_catalog=False
                )
            elif is_pure_metadata:
                # Mode 2: Metadata batch matching
                logger.info(f"[BATCH] Mode 2 (Metadata) - Processing {len(product_ids)} products in parallel")
                logger.info(f"[BATCH] Mode 2 will use ThreadPoolExecutor for parallel metadata comparison (no GPU needed)")
                logger.info(f"[BATCH] Mode 2 using dynamic weights: {metadata_weights}")

                result = batch_find_metadata_matches(
                    product_ids=product_ids,
                    threshold=threshold,
                    limit=limit,
                    weights=metadata_weights,
                    store_matches=True,
                    skip_invalid_products=True,
                    match_against_all=match_against_all
                )
            else:
                # Fallback: shouldn't happen, but default to metadata
                logger.warning(f"[BATCH] Unexpected weight combination: visual={visual_weight}, metadata={metadata_weight}. Defaulting to Mode 2 (Metadata)")

                result = batch_find_metadata_matches(
                    product_ids=product_ids,
                    threshold=threshold,
                    limit=limit,
                    weights=metadata_weights,
                    store_matches=True,
                    skip_invalid_products=True,
                    match_against_all=match_against_all
                )
            
            # Persist full per-product result payload in DB and return a lightweight response.
            # Prune stale sessions first to keep DB lean.
            cleanup_stale_match_result_sessions(max_age_hours=24)

            result_rows = result.get('results', []) or []
            match_mode = 'hybrid' if is_hybrid else ('visual' if is_pure_visual else 'metadata')
            session_id = create_match_result_session(
                mode=match_mode,
                threshold=threshold,
                limit_value=limit,
                visual_weight=visual_weight,
                metadata_weight=metadata_weight,
                batch_size=len(product_ids),
                summary=result.get('summary', {}),
                errors=result.get('errors', [])
            )
            stored_results = store_match_result_session_items(session_id, result_rows, replace_existing=True)
            logger.info(f"[BATCH] Stored {stored_results} result rows in DB session {session_id}")

            include_results = bool(data.get('include_results', False))
            response = {
                'status': 'success',
                'batch_size': len(product_ids),
                'session_id': session_id,
                'results_count': len(result_rows),
                'summary': result.get('summary', {}),
                'errors': result.get('errors', [])
            }
            if include_results:
                response['results'] = result_rows
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Batch matching failed: {e}", exc_info=True)
            return create_error_response(
                'BATCH_MATCHING_ERROR',
                f'Batch matching failed: {str(e)}',
                'Check product IDs and try again',
                status_code=500
            )
    
    except Exception as e:
        logger.error(f"Unexpected error in batch_match_products: {e}", exc_info=True)
        return create_error_response(
            'UNKNOWN_ERROR',
            'An unexpected error occurred during batch matching',
            'Please try again or contact support',
            {'error': str(e)},
            status_code=500
        )

@app.route('/api/session/cleanup', methods=['POST'])
def cleanup_session():

    try:
        from database import get_db_connection, invalidate_faiss_index
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM matches')
            deleted_count = cursor.rowcount

        deleted_result_sessions = clear_all_match_result_sessions()
        _invalidate_session_cache()  # Clear in-memory parsed results

        # Invalidate all FAISS indexes to free memory
        invalidate_faiss_index(category=None)  # None = invalidate all categories
        
        logger.info(f"[SESSION-CLEANUP] Deleted {deleted_count} matches on app close")
        logger.info(f"[SESSION-CLEANUP] Deleted {deleted_result_sessions} stored match result sessions")
        logger.info(f"[SESSION-CLEANUP] Invalidated all FAISS indexes")
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up {deleted_count} matches, {deleted_result_sessions} stored result sessions, and cleared indexes',
            'matches_deleted': deleted_count,
            'match_result_sessions_deleted': deleted_result_sessions
        }), 200
        
    except Exception as e:
        logger.error(f"Session cleanup error: {e}", exc_info=True)
        return create_error_response(
            'CLEANUP_ERROR',
            'Failed to clean up session',
            str(e),
            status_code=500
        )

@app.route('/api/products/match-results', methods=['GET'])
def get_match_results():
    """Get stored match results for NEW section products (uploaded via mobile or desktop)

    Returns match results in the same format as batch-match for automatic desktop display.
    Used for polling - returns all stored results from NEW section.

    Returns:
    - 200: List of match results with products and their matches
    """
    try:
        from database import get_db_connection, get_products_by_ids

        logger.debug("[MATCH-RESULTS] Fetching stored match results for NEW section")

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get all products in NEW section that have stored matches,
            # along with their top-5 matches in a single query (avoids N+1)
            cursor.execute('''
                SELECT p.id, p.image_path, p.category, p.product_name, p.sku, p.metadata,
                       m.matched_product_id, m.similarity_score
                FROM products p
                INNER JOIN matches m ON m.new_product_id = p.id
                WHERE p.is_historical = 0
                ORDER BY p.id DESC, m.similarity_score DESC
            ''')

            # Group rows by product and collect all matched IDs for batch lookup
            from collections import OrderedDict
            product_matches = OrderedDict()  # product_id -> {product_row, match_rows}
            all_matched_ids = set()

            for row in cursor.fetchall():
                pid = row['id']
                if pid not in product_matches:
                    product_matches[pid] = {
                        'product_row': row,
                        'match_rows': []
                    }
                # Limit to top 5 matches per product
                if len(product_matches[pid]['match_rows']) < 5:
                    product_matches[pid]['match_rows'].append(row)
                    all_matched_ids.add(row['matched_product_id'])

            # Single batch lookup for ALL matched products (replaces N individual queries)
            matched_products_map = get_products_by_ids(list(all_matched_ids)) if all_matched_ids else {}

            results = []
            for product_id, data in product_matches.items():
                product_row = data['product_row']
                product_data = {
                    'id': product_id,
                    'image_path': product_row['image_path'],
                    'category': product_row['category'],
                    'product_name': product_row['product_name'],
                    'sku': product_row['sku'],
                    'metadata': json.loads(product_row['metadata']) if product_row['metadata'] else {},
                    'has_features': True
                }

                match_list = []
                for match_row in data['match_rows']:
                    matched_product = matched_products_map.get(match_row['matched_product_id'])
                    if matched_product:
                        match_data = {
                            'product_id': match_row['matched_product_id'],
                            'product_name': matched_product.get('product_name') or 'Unknown',
                            'name': matched_product.get('product_name') or 'Unknown',
                            'category': matched_product.get('category'),
                            'sku': matched_product.get('sku'),
                            'similarity_score': match_row['similarity_score'],
                            'image_path': matched_product.get('image_path')
                        }
                        match_list.append(match_data)

                if match_list:
                    results.append({
                        'product_id': product_id,
                        'product_data': product_data,
                        'matches': match_list,
                        'status': 'success'
                    })

            logger.debug(f"[MATCH-RESULTS] Found {len(results)} products with stored matches")

            return jsonify({
                'success': True,
                'results': results,
                'count': len(results)
            }), 200

    except Exception as e:
        logger.error(f"Error fetching match results: {e}", exc_info=True)
        return create_error_response(
            'MATCH_RESULTS_ERROR',
            'Failed to fetch match results',
            str(e),
            status_code=500
        )


@app.route('/api/products/match-results/session', methods=['GET'])
def get_match_results_by_session():
    """Get paginated stored match results for a batch-match session."""
    try:
        session_id = (request.args.get('session_id') or '').strip()
        if not session_id:
            return create_error_response(
                'MISSING_SESSION_ID',
                'session_id is required',
                'Provide ?session_id=<id> in query params',
                status_code=400
            )

        page = request.args.get('page', 1, type=int) or 1
        limit = request.args.get('limit', 250, type=int) or 250
        page = max(1, page)
        limit = max(1, min(limit, 1000))

        session_page = get_match_result_session_page(session_id, page=page, limit=limit)
        if not session_page.get('exists'):
            return create_error_response(
                'SESSION_NOT_FOUND',
                f'No stored match results found for session {session_id}',
                'Run batch match again to generate a new session',
                status_code=404
            )

        return jsonify({
            'success': True,
            'session_id': session_id,
            'page': session_page['page'],
            'limit': session_page['limit'],
            'total_results': session_page['total_results'],
            'total_pages': session_page['total_pages'],
            'summary': session_page.get('summary', {}),
            'errors': session_page.get('errors', []),
            'created_at': session_page.get('created_at'),
            'results': session_page.get('results', [])
        }), 200

    except Exception as e:
        logger.error(f"Error fetching session match results: {e}", exc_info=True)
        return create_error_response(
            'SESSION_RESULTS_ERROR',
            'Failed to fetch stored match results',
            str(e),
            status_code=500
        )


# ---------------------------------------------------------------------------
# In-memory cache for parsed session results.
# Avoids re-reading + re-parsing every JSON blob on each filter/sort request.
# Key = session_id, value = list of parsed dicts.  Max 3 sessions cached.
# ---------------------------------------------------------------------------
_session_result_cache: OrderedDict = OrderedDict()  # LRU cache: session_id -> parsed results
_SESSION_CACHE_MAX = 2  # Keep current + one previous; typically only one active session
_SESSION_CACHE_MAX_BYTES = 150 * 1024 * 1024  # 150MB max total cache size
_session_cache_sizes: Dict[str, int] = {}  # Track approx byte size per session
_session_cache_lock = threading.Lock()

# Cache filtered/sorted query projections (same filters, different page requests).
# Key excludes page/limit so pagination doesn't rescan all results each click.
_session_query_cache: OrderedDict = OrderedDict()  # cache_key -> {'session_id': str, 'results': list, ...}
_SESSION_QUERY_CACHE_MAX = 3
_SESSION_QUERY_CACHE_MAX_BYTES = 120 * 1024 * 1024
_session_query_cache_sizes: Dict[str, int] = {}
_session_query_cache_lock = threading.Lock()


def _estimate_session_size(results: List[Dict[str, Any]]) -> int:
    """Estimate memory size of cached session results (rough approximation)."""
    import sys
    if not results:
        return 0
    # Sample first 10 items and extrapolate (avoids iterating 10k+ items)
    sample = results[:min(10, len(results))]
    avg_size = sum(sys.getsizeof(json.dumps(item)) for item in sample) / len(sample)
    return int(avg_size * len(results))


def _get_cached_session_results(session_id: str) -> Optional[List[Dict[str, Any]]]:
    """Return cached parsed results for a session, or None if not cached."""
    with _session_cache_lock:
        if session_id in _session_result_cache:
            _session_result_cache.move_to_end(session_id)  # O(1) LRU touch
            return _session_result_cache[session_id]
    return None


def _set_cached_session_results(session_id: str, results: List[Dict[str, Any]]) -> None:
    """Cache parsed results for a session, evicting oldest if over limit.

    MEMORY SAFETY: Enforces both count-based and byte-size-based eviction
    to prevent 50MB+ sessions from accumulating.
    """
    estimated_size = _estimate_session_size(results)

    # If a single session exceeds cap, skip caching entirely to avoid OOM risk.
    if estimated_size > _SESSION_CACHE_MAX_BYTES:
        with _session_cache_lock:
            _session_result_cache.pop(session_id, None)
            _session_cache_sizes.pop(session_id, None)
        logger.debug(
            f"Skipping session cache for '{session_id}' (~{estimated_size // 1024}KB > cap)"
        )
        return

    with _session_cache_lock:
        if session_id in _session_result_cache:
            _session_result_cache[session_id] = results
            _session_cache_sizes[session_id] = estimated_size
            _session_result_cache.move_to_end(session_id)
            return

        # Evict until we're under both count and byte limits
        total_cached_bytes = sum(_session_cache_sizes.values())
        while (len(_session_result_cache) >= _SESSION_CACHE_MAX or
               (total_cached_bytes + estimated_size > _SESSION_CACHE_MAX_BYTES and _session_result_cache)):
            evicted_id, _ = _session_result_cache.popitem(last=False)  # O(1) evict oldest
            evicted_size = _session_cache_sizes.pop(evicted_id, 0)
            total_cached_bytes -= evicted_size
            logger.debug(f"Evicted session cache '{evicted_id}' (~{evicted_size // 1024}KB)")

        _session_result_cache[session_id] = results
        _session_cache_sizes[session_id] = estimated_size


def _build_session_query_cache_key(
    session_id: str,
    search_query: str,
    category_filter: str,
    duplicates_only: bool,
    threshold: float,
    dynamic_limit: int,
    dynamic_search: str,
    metadata_filters: Dict[str, Any],
    sort_by: str,
    sort_order: str
) -> str:
    metadata_filters_serialized = json.dumps(
        metadata_filters or {},
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    )
    return (
        f"{session_id}|sq={search_query}|cf={category_filter}|dup={int(bool(duplicates_only))}"
        f"|th={threshold:.4f}|dl={dynamic_limit}|ds={dynamic_search}|mf={metadata_filters_serialized}"
        f"|sb={sort_by}|so={sort_order}"
    )


def _get_cached_session_query_results(cache_key: str) -> Optional[Dict[str, Any]]:
    with _session_query_cache_lock:
        if cache_key in _session_query_cache:
            _session_query_cache.move_to_end(cache_key)
            return _session_query_cache[cache_key]
    return None


def _set_cached_session_query_results(
    cache_key: str,
    session_id: str,
    prepared_results: List[Dict[str, Any]],
    filtered_total_matches: int,
    scanned_results: int
) -> None:
    estimated_size = _estimate_session_size(prepared_results)

    # If one projection is bigger than cache budget, skip caching this query safely.
    if estimated_size > _SESSION_QUERY_CACHE_MAX_BYTES:
        with _session_query_cache_lock:
            _session_query_cache.pop(cache_key, None)
            _session_query_cache_sizes.pop(cache_key, None)
        logger.debug(
            f"Skipping query cache for key '{cache_key}' (~{estimated_size // 1024}KB > cap)"
        )
        return

    with _session_query_cache_lock:
        if cache_key in _session_query_cache:
            _session_query_cache[cache_key] = {
                'session_id': session_id,
                'results': prepared_results,
                'filtered_total_matches': filtered_total_matches,
                'scanned_results': scanned_results
            }
            _session_query_cache_sizes[cache_key] = estimated_size
            _session_query_cache.move_to_end(cache_key)
            return

        total_cached_bytes = sum(_session_query_cache_sizes.values())
        while (
            len(_session_query_cache) >= _SESSION_QUERY_CACHE_MAX or
            (total_cached_bytes + estimated_size > _SESSION_QUERY_CACHE_MAX_BYTES and _session_query_cache)
        ):
            evicted_key, _ = _session_query_cache.popitem(last=False)
            evicted_size = _session_query_cache_sizes.pop(evicted_key, 0)
            total_cached_bytes -= evicted_size
            logger.debug(f"Evicted session query cache '{evicted_key}' (~{evicted_size // 1024}KB)")

        _session_query_cache[cache_key] = {
            'session_id': session_id,
            'results': prepared_results,
            'filtered_total_matches': filtered_total_matches,
            'scanned_results': scanned_results
        }
        _session_query_cache_sizes[cache_key] = estimated_size


def _compute_prepared_session_results(
    session_id: str,
    search_query: str,
    category_filter: str,
    duplicates_only: bool,
    threshold: float,
    dynamic_limit: int,
    dynamic_search: str,
    metadata_filters: Dict[str, Any],
    sort_by: str,
    sort_order: str
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Build filtered/sorted session results and aggregate counters."""
    all_raw_results = _load_session_results(session_id)
    scanned_count = len(all_raw_results)

    prepared_results: List[Dict[str, Any]] = []
    filtered_total_matches = 0

    for raw_result in all_raw_results:
        prepared = _prepare_filtered_session_result(
            raw_result=raw_result,
            search_query=search_query,
            category_filter=category_filter,
            duplicates_only=duplicates_only,
            threshold=threshold,
            dynamic_limit=dynamic_limit,
            dynamic_search=dynamic_search,
            metadata_filters=metadata_filters
        )
        if not prepared:
            continue

        filtered_total_matches += len(prepared.get('matches') or [])
        prepared_results.append(prepared)

    _sort_prepared_session_results(prepared_results, sort_by=sort_by, sort_order=sort_order)
    return prepared_results, filtered_total_matches, scanned_count


def _strip_internal_sort_keys(result_item: Dict[str, Any]) -> Dict[str, Any]:
    """Return a response-safe copy without internal sort helper keys."""
    if not isinstance(result_item, dict):
        return {}
    return {
        key: value
        for key, value in result_item.items()
        if not key.startswith('_sort_')
    }


def _invalidate_session_query_cache(session_id: Optional[str] = None) -> None:
    with _session_query_cache_lock:
        if session_id:
            keys_to_delete = [
                key for key, entry in _session_query_cache.items()
                if isinstance(entry, dict) and entry.get('session_id') == session_id
            ]
            for key in keys_to_delete:
                _session_query_cache.pop(key, None)
                _session_query_cache_sizes.pop(key, None)
        else:
            _session_query_cache.clear()
            _session_query_cache_sizes.clear()


def _invalidate_session_cache(session_id: Optional[str] = None) -> None:
    """Remove a session (or all sessions) from the cache."""
    with _session_cache_lock:
        if session_id:
            _session_result_cache.pop(session_id, None)
            _session_cache_sizes.pop(session_id, None)
        else:
            _session_result_cache.clear()
            _session_cache_sizes.clear()
    _invalidate_session_query_cache(session_id)


def _load_session_results(session_id: str) -> List[Dict[str, Any]]:
    """Load parsed results for a session, using cache if available."""
    cached = _get_cached_session_results(session_id)
    if cached is not None:
        return cached

    parsed: List[Dict[str, Any]] = []
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT result_json FROM match_result_items WHERE session_id = ? ORDER BY id ASC',
            (session_id,)
        )
        while True:
            rows = cursor.fetchmany(300)
            if not rows:
                break
            for row in rows:
                obj = _safe_json_loads(row['result_json'], {})
                if isinstance(obj, dict):
                    parsed.append(obj)

    _set_cached_session_results(session_id, parsed)
    return parsed


def _parse_bool_arg(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _safe_json_loads(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if parsed is not None else default
        except (json.JSONDecodeError, TypeError, ValueError):
            return default
    return default


def _normalize_text(value: Any) -> str:
    if value is None:
        return ''
    return str(value).strip().lower()


def _get_match_similarity(match: Dict[str, Any]) -> float:
    try:
        return float(match.get('similarity_score', 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _match_dynamic_search(match: Dict[str, Any], dynamic_search: str) -> bool:
    if not dynamic_search:
        return True

    metadata_values = match.get('metadata_values') or {}
    haystack = [
        match.get('product_name') or match.get('name') or '',
        match.get('sku') or '',
        match.get('category') or ''
    ]

    if isinstance(metadata_values, dict):
        for val in metadata_values.values():
            if val is not None:
                haystack.append(str(val))

    combined = ' '.join(str(x) for x in haystack if x is not None).lower()
    return dynamic_search in combined


def _match_metadata_filters(match: Dict[str, Any], metadata_filters: Dict[str, Any]) -> bool:
    if not metadata_filters:
        return True

    values = match.get('metadata_values') or {}
    if not isinstance(values, dict):
        return False

    for field, criteria in metadata_filters.items():
        if not isinstance(criteria, dict):
            continue

        raw_val = values.get(field)

        selected_values = criteria.get('values')
        if isinstance(selected_values, (list, tuple, set)) and len(selected_values) > 0:
            selected_set = {str(v) for v in selected_values}
            if raw_val is None or str(raw_val) not in selected_set:
                return False

        if 'equals' in criteria and criteria.get('equals') not in (None, ''):
            if _normalize_text(raw_val) != _normalize_text(criteria.get('equals')):
                return False

        if 'min' in criteria and criteria.get('min') not in (None, ''):
            try:
                if raw_val is None or float(raw_val) < float(criteria.get('min')):
                    return False
            except (TypeError, ValueError):
                return False

        if 'max' in criteria and criteria.get('max') not in (None, ''):
            try:
                if raw_val is None or float(raw_val) > float(criteria.get('max')):
                    return False
            except (TypeError, ValueError):
                return False

    return True


def _parse_product_metadata(product_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(product_data, dict):
        return {}
    metadata = product_data.get('metadata')
    return _safe_json_loads(metadata, {}) if metadata is not None else {}


def _prepare_filtered_session_result(
    raw_result: Dict[str, Any],
    search_query: str,
    category_filter: str,
    duplicates_only: bool,
    threshold: float,
    dynamic_limit: int,
    dynamic_search: str,
    metadata_filters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_result, dict):
        return None

    product_data = raw_result.get('product_data') or {}
    if not isinstance(product_data, dict):
        product_data = {}

    product_name = product_data.get('product_name') or product_data.get('name') or ''
    product_sku = product_data.get('sku') or ''
    product_category = product_data.get('category') or ''
    normalized_product_category = _normalize_text(product_category) or 'uncategorized'

    if search_query:
        searchable = f"{product_name} {product_sku} {product_category}".lower()
        if search_query not in searchable:
            return None

    if category_filter and category_filter not in {'all', ''}:
        if normalized_product_category != _normalize_text(category_filter):
            return None

    matches = raw_result.get('matches') or []
    if not isinstance(matches, list):
        matches = []

    filtered_matches = []
    for match in matches:
        if not isinstance(match, dict):
            continue

        similarity = _get_match_similarity(match)
        if threshold > 30 and similarity < threshold:
            continue

        if not _match_metadata_filters(match, metadata_filters):
            continue

        if not _match_dynamic_search(match, dynamic_search):
            continue

        filtered_matches.append(match)

    if dynamic_limit > 0 and len(filtered_matches) > dynamic_limit:
        filtered_matches = filtered_matches[:dynamic_limit]

    if len(filtered_matches) == 0:
        return None

    if duplicates_only and all(_get_match_similarity(m) <= 90 for m in filtered_matches):
        return None

    prepared = dict(raw_result)
    prepared['matches'] = filtered_matches
    prepared['match_count'] = len(filtered_matches)

    top_similarity = _get_match_similarity(filtered_matches[0]) if filtered_matches else 0.0
    avg_similarity = (
        sum(_get_match_similarity(m) for m in filtered_matches) / len(filtered_matches)
        if filtered_matches else 0.0
    )

    prepared['_sort_top_similarity'] = top_similarity
    prepared['_sort_avg_similarity'] = avg_similarity
    prepared['_sort_match_count'] = len(filtered_matches)
    prepared['_sort_name'] = _normalize_text(product_name)
    prepared['_sort_category'] = normalized_product_category

    metadata_dict = _parse_product_metadata(product_data)
    prepared['_sort_dynamic_meta'] = metadata_dict
    prepared['_sort_product_data'] = product_data

    return prepared


def _sort_prepared_session_results(
    prepared_results: List[Dict[str, Any]],
    sort_by: str,
    sort_order: str
) -> None:
    reverse = _normalize_text(sort_order) != 'asc'
    sort_by_norm = _normalize_text(sort_by) or 'similarity'

    if sort_by_norm == 'similarity':
        prepared_results.sort(key=lambda r: r.get('_sort_top_similarity', 0.0), reverse=True)
        return
    if sort_by_norm == 'avg_similarity':
        prepared_results.sort(key=lambda r: r.get('_sort_avg_similarity', 0.0), reverse=True)
        return
    if sort_by_norm == 'match_count':
        prepared_results.sort(key=lambda r: r.get('_sort_match_count', 0), reverse=True)
        return
    if sort_by_norm == 'name':
        prepared_results.sort(key=lambda r: r.get('_sort_name', ''))
        if reverse:
            prepared_results.reverse()
        return
    if sort_by_norm == 'category':
        prepared_results.sort(key=lambda r: r.get('_sort_category', ''))
        if reverse:
            prepared_results.reverse()
        return

    def dynamic_key(result_item: Dict[str, Any]) -> Tuple[int, Any]:
        product_data = result_item.get('_sort_product_data') or {}
        metadata_dict = result_item.get('_sort_dynamic_meta') or {}

        val = product_data.get(sort_by)
        if val in (None, ''):
            val = metadata_dict.get(sort_by)

        if val in (None, ''):
            return (1, '')

        try:
            numeric_val = float(val)
            return (0, numeric_val)
        except (TypeError, ValueError):
            return (0, str(val).lower())

    prepared_results.sort(key=dynamic_key)
    if reverse:
        prepared_results.reverse()


@app.route('/api/products/match-results/session/query', methods=['GET'])
def query_match_results_by_session():
    """Server-side filtering/sorting/pagination for stored batch-match session results."""
    try:
        session_id = (request.args.get('session_id') or '').strip()
        if not session_id:
            return create_error_response(
                'MISSING_SESSION_ID',
                'session_id is required',
                'Provide ?session_id=<id> in query params',
                status_code=400
            )

        page = max(1, request.args.get('page', 1, type=int) or 1)
        limit = max(1, min(request.args.get('limit', 250, type=int) or 250, 1000))
        threshold = float(request.args.get('threshold', 30.0) or 30.0)
        dynamic_limit = int(request.args.get('dynamic_limit', 0) or 0)

        search_query = _normalize_text(request.args.get('search_query', ''))
        dynamic_search = _normalize_text(request.args.get('dynamic_search', ''))
        category_filter = _normalize_text(request.args.get('filter_category', 'all') or 'all') or 'all'
        duplicates_only = _parse_bool_arg(request.args.get('duplicates_only', False), default=False)
        sort_by = _normalize_text(request.args.get('sort_by', 'similarity') or 'similarity') or 'similarity'
        sort_order = _normalize_text(request.args.get('sort_order', 'desc') or 'desc') or 'desc'

        metadata_filters_raw = request.args.get('metadata_filters')
        metadata_filters = _safe_json_loads(metadata_filters_raw, {}) if metadata_filters_raw else {}
        if not isinstance(metadata_filters, dict):
            metadata_filters = {}

        session_info = get_match_result_session_page(session_id, page=1, limit=1)
        if not session_info.get('exists'):
            return create_error_response(
                'SESSION_NOT_FOUND',
                f'No stored match results found for session {session_id}',
                'Run batch match again to generate a new session',
                status_code=404
            )

        cache_key = _build_session_query_cache_key(
            session_id=session_id,
            search_query=search_query,
            category_filter=category_filter,
            duplicates_only=duplicates_only,
            threshold=threshold,
            dynamic_limit=dynamic_limit,
            dynamic_search=dynamic_search,
            metadata_filters=metadata_filters,
            sort_by=sort_by,
            sort_order=sort_order
        )

        cached_entry = _get_cached_session_query_results(cache_key)
        if cached_entry is not None:
            prepared_results = cached_entry.get('results') or []
            filtered_total_matches = int(cached_entry.get('filtered_total_matches') or 0)
            scanned_count = int(cached_entry.get('scanned_results') or 0)
        else:
            prepared_results, filtered_total_matches, scanned_count = _compute_prepared_session_results(
                session_id=session_id,
                search_query=search_query,
                category_filter=category_filter,
                duplicates_only=duplicates_only,
                threshold=threshold,
                dynamic_limit=dynamic_limit,
                dynamic_search=dynamic_search,
                metadata_filters=metadata_filters,
                sort_by=sort_by,
                sort_order=sort_order
            )
            _set_cached_session_query_results(
                cache_key=cache_key,
                session_id=session_id,
                prepared_results=prepared_results,
                filtered_total_matches=filtered_total_matches,
                scanned_results=scanned_count
            )

        total_results = len(prepared_results)
        total_pages = (total_results + limit - 1) // limit if total_results > 0 else 0
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paged_results = [
            _strip_internal_sort_keys(item)
            for item in prepared_results[start_idx:end_idx]
        ]

        return jsonify({
            'success': True,
            'session_id': session_id,
            'page': page,
            'limit': limit,
            'total_results': total_results,
            'total_pages': total_pages,
            'filtered_total_matches': filtered_total_matches,
            'products_with_matches': total_results,
            'scanned_results': scanned_count,
            'summary': session_info.get('summary', {}),
            'errors': session_info.get('errors', []),
            'results': paged_results
        }), 200

    except Exception as e:
        logger.error(f"Error querying session match results: {e}", exc_info=True)
        return create_error_response(
            'SESSION_QUERY_ERROR',
            'Failed to query stored match results',
            str(e),
            status_code=500
        )


@app.route('/api/products/match-results/session/facets', methods=['GET'])
def get_match_results_session_facets():
    """Build dynamic filter facets for a stored match-results session."""
    try:
        session_id = (request.args.get('session_id') or '').strip()
        if not session_id:
            return create_error_response(
                'MISSING_SESSION_ID',
                'session_id is required',
                'Provide ?session_id=<id> in query params',
                status_code=400
            )

        session_info = get_match_result_session_page(session_id, page=1, limit=1)
        if not session_info.get('exists'):
            return create_error_response(
                'SESSION_NOT_FOUND',
                f'No stored match results found for session {session_id}',
                'Run batch match again to generate a new session',
                status_code=404
            )

        max_sampled_matches = request.args.get('max_sampled_matches', 50000, type=int) or 50000
        max_sampled_matches = max(1000, min(max_sampled_matches, 200000))
        max_unique_values = request.args.get('max_unique_values', 2000, type=int) or 2000
        max_unique_values = max(100, min(max_unique_values, 5000))

        schema = get_metadata_schema() or []
        core_fields = {'id', 'image_path', 'sku', 'name', 'category'}
        schema_types = {
            col.get('column_name'): col.get('data_type', 'string')
            for col in schema
            if col.get('column_name') and col.get('column_name') not in core_fields
        }

        categories = set()
        categorical_values: Dict[str, Dict[str, int]] = {}
        numeric_ranges: Dict[str, Dict[str, Optional[float]]] = {}
        truncated_fields = set()

        for key, dtype in schema_types.items():
            if dtype == 'numeric':
                numeric_ranges[key] = {'min': None, 'max': None}
            else:
                categorical_values[key] = {}

        # Use in-memory cache to avoid re-parsing JSON on every facets request.
        all_raw_results = _load_session_results(session_id)
        scanned_results = len(all_raw_results)
        scanned_matches = 0

        for raw_result in all_raw_results:
            product_data = raw_result.get('product_data') or {}
            if isinstance(product_data, dict):
                category_val = product_data.get('category')
                normalized_category = _normalize_text(category_val)
                categories.add(normalized_category if normalized_category else 'uncategorized')

            matches = raw_result.get('matches') or []
            if not isinstance(matches, list):
                continue

            for match in matches:
                if scanned_matches >= max_sampled_matches:
                    break

                if not isinstance(match, dict):
                    continue

                values = match.get('metadata_values') or {}
                if not isinstance(values, dict):
                    scanned_matches += 1
                    continue

                for key, dtype in schema_types.items():
                    raw_val = values.get(key)
                    if raw_val in (None, ''):
                        continue

                    if dtype == 'numeric':
                        try:
                            numeric_val = float(raw_val)
                        except (TypeError, ValueError):
                            continue

                        range_info = numeric_ranges.setdefault(key, {'min': None, 'max': None})
                        range_info['min'] = numeric_val if range_info['min'] is None else min(range_info['min'], numeric_val)
                        range_info['max'] = numeric_val if range_info['max'] is None else max(range_info['max'], numeric_val)
                    else:
                        value_str = str(raw_val)
                        value_map = categorical_values.setdefault(key, {})
                        if value_str in value_map:
                            value_map[value_str] += 1
                        elif len(value_map) < max_unique_values:
                            value_map[value_str] = 1
                        else:
                            truncated_fields.add(key)

                scanned_matches += 1

            if scanned_matches >= max_sampled_matches:
                break

        metadata_facets: Dict[str, Any] = {}
        for key, dtype in schema_types.items():
            if dtype == 'numeric':
                range_info = numeric_ranges.get(key) or {'min': None, 'max': None}
                metadata_facets[key] = {
                    'type': 'numeric',
                    'min': range_info.get('min'),
                    'max': range_info.get('max')
                }
            else:
                values_map = categorical_values.get(key) or {}
                metadata_facets[key] = {
                    'type': 'categorical',
                    'values': sorted(values_map.keys()),
                    'value_count': len(values_map),
                    'truncated': key in truncated_fields
                }

        return jsonify({
            'success': True,
            'session_id': session_id,
            'categories': sorted(categories),
            'metadata_facets': metadata_facets,
            'scanned_results': scanned_results,
            'scanned_matches': scanned_matches,
            'sample_limited': scanned_matches >= max_sampled_matches,
            'summary': session_info.get('summary', {})
        }), 200

    except Exception as e:
        logger.error(f"Error building session facets: {e}", exc_info=True)
        return create_error_response(
            'SESSION_FACETS_ERROR',
            'Failed to build session filter facets',
            str(e),
            status_code=500
        )

@app.route('/api/products/match-results/session/export-csv', methods=['GET'])
def export_session_results_csv():
    """Stream a CSV export of filtered session results directly from the backend.

    Accepts the same query parameters as /session/query (search, category,
    threshold, metadata_filters, sort, etc.) and returns a downloadable CSV
    file.  Much faster than having the frontend fetch every page.
    """
    try:
        session_id = (request.args.get('session_id') or '').strip()
        if not session_id:
            return create_error_response(
                'MISSING_SESSION_ID', 'session_id is required',
                'Provide ?session_id=<id>', status_code=400)

        threshold = float(request.args.get('threshold', 30.0) or 30.0)
        dynamic_limit = int(request.args.get('dynamic_limit', 0) or 0)
        search_query = _normalize_text(request.args.get('search_query', ''))
        dynamic_search = _normalize_text(request.args.get('dynamic_search', ''))
        category_filter = _normalize_text(request.args.get('filter_category', 'all') or 'all') or 'all'
        duplicates_only = _parse_bool_arg(request.args.get('duplicates_only', False), default=False)
        sort_by = _normalize_text(request.args.get('sort_by', 'similarity') or 'similarity') or 'similarity'
        sort_order = _normalize_text(request.args.get('sort_order', 'desc') or 'desc') or 'desc'
        metadata_filters_raw = request.args.get('metadata_filters')
        metadata_filters = _safe_json_loads(metadata_filters_raw, {}) if metadata_filters_raw else {}
        if not isinstance(metadata_filters, dict):
            metadata_filters = {}

        session_info = get_match_result_session_page(session_id, page=1, limit=1)
        if not session_info.get('exists'):
            return create_error_response(
                'SESSION_NOT_FOUND',
                f'No stored match results found for session {session_id}',
                'Run batch match again', status_code=404)

        cache_key = _build_session_query_cache_key(
            session_id=session_id,
            search_query=search_query,
            category_filter=category_filter,
            duplicates_only=duplicates_only,
            threshold=threshold,
            dynamic_limit=dynamic_limit,
            dynamic_search=dynamic_search,
            metadata_filters=metadata_filters,
            sort_by=sort_by,
            sort_order=sort_order
        )

        cached_entry = _get_cached_session_query_results(cache_key)
        if cached_entry is not None:
            prepared_results = cached_entry.get('results') or []
        else:
            prepared_results, filtered_total_matches, scanned_count = _compute_prepared_session_results(
                session_id=session_id,
                search_query=search_query,
                category_filter=category_filter,
                duplicates_only=duplicates_only,
                threshold=threshold,
                dynamic_limit=dynamic_limit,
                dynamic_search=dynamic_search,
                metadata_filters=metadata_filters,
                sort_by=sort_by,
                sort_order=sort_order
            )
            _set_cached_session_query_results(
                cache_key=cache_key,
                session_id=session_id,
                prepared_results=prepared_results,
                filtered_total_matches=filtered_total_matches,
                scanned_results=scanned_count
            )

        # --- discover metadata score keys across all results ---
        metadata_score_keys: List[str] = []
        seen_keys: set = set()
        for item in prepared_results:
            for match in (item.get('matches') or []):
                for k in (match.get('metadata_scores') or {}).keys():
                    if k not in seen_keys:
                        seen_keys.add(k)
                        metadata_score_keys.append(k)
        metadata_score_keys.sort()

        def _csv_escape(val: Any) -> str:
            if val is None:
                return ''
            s = str(val)
            if ',' in s or '"' in s or '\n' in s:
                return '"' + s.replace('"', '""') + '"'
            return s

        def _generate_csv():
            # Header
            header = ['New Product', 'Category', 'SKU', 'Total Matches',
                      'Avg Similarity', 'Best Score', 'Top Match Score']
            for k in metadata_score_keys:
                header.append(f'Avg {k}')
            header.extend(['Top Match Name', 'Top Match Overall Score'])
            for k in metadata_score_keys:
                header.append(f'Top Match {k}')
            yield ','.join(f'"{h}"' for h in header) + '\n'

            for item in prepared_results:
                product_data = item.get('product_data') or item.get('_sort_product_data') or {}
                matches = item.get('matches') or []
                top_match = matches[0] if matches else None

                # Compute stats
                scores = [_get_match_similarity(m) for m in matches]
                avg_sim = sum(scores) / len(scores) if scores else 0.0
                best_score = max(scores) if scores else 0.0
                top_score = _get_match_similarity(top_match) if top_match else 0.0

                row = [
                    product_data.get('product_name') or product_data.get('name') or '',
                    product_data.get('category') or 'Uncategorized',
                    product_data.get('sku') or '',
                    str(len(matches)),
                    f'{avg_sim:.1f}',
                    f'{best_score:.1f}',
                    f'{top_score:.1f}',
                ]

                # Avg metadata scores
                for k in metadata_score_keys:
                    vals = [m.get('metadata_scores', {}).get(k) for m in matches
                            if m.get('metadata_scores', {}).get(k) is not None]
                    avg_val = sum(float(v) for v in vals) / len(vals) if vals else ''
                    row.append(f'{avg_val:.1f}' if isinstance(avg_val, float) else '')

                # Top match info
                if top_match:
                    row.append(top_match.get('product_name') or top_match.get('name') or 'Unknown')
                    row.append(f'{top_score:.1f}')
                    top_scores = top_match.get('metadata_scores') or {}
                    for k in metadata_score_keys:
                        v = top_scores.get(k)
                        row.append(f'{float(v):.1f}' if v is not None else '')
                else:
                    row.append('No matches')
                    row.append('0')
                    for _ in metadata_score_keys:
                        row.append('')

                yield ','.join(_csv_escape(c) for c in row) + '\n'

        filename = f'match_results_{datetime.now().strftime("%Y%m%d")}.csv'
        return Response(
            stream_with_context(_generate_csv()),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'})

    except Exception as e:
        logger.error(f"Error exporting session CSV: {e}", exc_info=True)
        return create_error_response(
            'SESSION_EXPORT_ERROR', 'Failed to export session results',
            str(e), status_code=500)


@app.route('/api/products/search', methods=['GET'])
def search_products():

    try:
        query = request.args.get('q', '').strip()
        limit = request.args.get('limit', 100, type=int)
        
        if not query:
            return create_error_response(
                'MISSING_QUERY',
                'Search query required',
                'Provide ?q=search_term',
                status_code=400
            )
        
        # Limit max results to prevent abuse while still supporting large-catalog filtering.
        limit = min(limit, 5000)
        
        from database import search_matched_products
        results = search_matched_products(query, limit)
        
        logger.info(f"[SEARCH] Query: '{query}' - Found {len(results)} results")
        
        return jsonify({
            'success': True,
            'query': query,
            'count': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return create_error_response(
            'SEARCH_ERROR',
            'Search failed',
            str(e),
            status_code=500
        )

@app.route('/api/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    try:
        # Get product from database
        try:
            product = get_product_by_id(product_id)
        except Exception as e:
            logger.error(f"Database error retrieving product {product_id}: {e}")
            return create_error_response(
                'DATABASE_ERROR',
                'Failed to retrieve product',
                'Please try again',
                {'error': str(e)},
                status_code=500
            )
        
        # Handle non-existent product
        if not product:
            return create_error_response(
                'PRODUCT_NOT_FOUND',
                f'Product with ID {product_id} not found',
                'Ensure the product ID is correct',
                {'product_id': product_id},
                status_code=404
            )
        
        # Convert to dictionary with NULL fields as null
        product_dict = {
            'id': product['id'],
            'image_path': product['image_path'],
            'category': product['category'],  # Can be NULL
            'product_name': product['product_name'],  # Can be NULL
            'sku': product['sku'],  # Can be NULL
            'is_historical': bool(product['is_historical']),
            'created_at': product['created_at'],
            'metadata': product['metadata']  # Can be NULL
        }
        
        # Debug logging for metadata
        logger.info(f"[GET-PRODUCT] Product {product_id} metadata type: {type(product['metadata'])}")
        if product['metadata']:
            logger.info(f"[GET-PRODUCT] Product {product_id} metadata value: {product['metadata'][:200] if isinstance(product['metadata'], str) else product['metadata']}")
        
        # Check feature extraction status
        feature_status = 'pending'
        feature_error = None
        
        try:
            features = get_features_by_product_id(product_id)
            if features:
                feature_status = 'success'
                # Include feature dimensions for verification
                product_dict['features'] = {
                    'color_features_dim': len(features['color_features']),
                    'shape_features_dim': len(features['shape_features']),
                    'texture_features_dim': len(features['texture_features'])
                }
            else:
                feature_status = 'pending'
        except Exception as e:
            logger.error(f"Error checking features for product {product_id}: {e}")
            feature_status = 'failed'
            feature_error = str(e)
        
        product_dict['feature_extraction_status'] = feature_status
        if feature_error:
            product_dict['feature_extraction_error'] = feature_error
        
        response = {
            'status': 'success',
            'product': product_dict
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in get_product: {e}", exc_info=True)
        return create_error_response(
            'UNKNOWN_ERROR',
            'An unexpected error occurred',
            'Please try again or contact support',
            {'error': str(e)},
            status_code=500
        )


# Error handlers for common HTTP errors
@app.errorhandler(RequestEntityTooLarge)
def request_entity_too_large(error):
    max_bytes = int(app.config.get('MAX_CONTENT_LENGTH') or 0)
    max_mb = max(1, round(max_bytes / (1024 * 1024))) if max_bytes > 0 else 500
    logger.warning(f"413 Payload Too Large: {request.method} {request.path} ({max_mb}MB limit)")
    return create_error_response(
        'PAYLOAD_TOO_LARGE',
        f'Upload payload exceeds the {max_mb}MB request limit',
        'Use smaller upload batches or enable desktop file-path transport',
        status_code=413
    )


@app.errorhandler(404)
def not_found(error):
    # Silently ignore browser icon requests (common browser behavior, not actual errors)
    ignored_paths = [
        '/favicon.ico',
        '/apple-touch-icon.png',
        '/apple-touch-icon-precomposed.png',
        '/apple-touch-icon-120x120.png',
        '/apple-touch-icon-120x120-precomposed.png'
    ]

    if request.path in ignored_paths:
        # Return 204 No Content silently without logging
        return '', 204

    logger.error(f"404 Not Found: {request.method} {request.path}")
    logger.error(f"Full URL: {request.url}")
    logger.error(f"Error details: {error}")
    return create_error_response(
        'NOT_FOUND',
        f'Endpoint not found: {request.method} {request.path}',
        'Check the API documentation for valid endpoints',
        status_code=404
    )

@app.errorhandler(405)
def method_not_allowed(error):
    logger.error(f"405 Method Not Allowed: {request.method} {request.path}")
    return create_error_response(
        'METHOD_NOT_ALLOWED',
        f'HTTP method {request.method} not allowed for {request.path}',
        'Check the API documentation for allowed methods',
        status_code=405
    )

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return create_error_response(
        'INTERNAL_ERROR',
        'Internal server error',
        'Please try again or contact support',
        status_code=500
    )

@app.route('/api/products/<int:product_id>/price-history', methods=['GET'])
def get_product_price_history(product_id):
    """
    Get price history for a product.
    
    Query parameters:
    - limit: Maximum number of records (optional, default: 12)
    
    Returns:
    - 200: Success with price history and statistics
    - 404: Product not found
    - 500: Server error
    """
    try:
        # Check if product exists
        product = get_product_by_id(product_id)
        if not product:
            return create_error_response(
                'PRODUCT_NOT_FOUND',
                f'Product with ID {product_id} not found',
                status_code=404
            )
        
        # Get limit parameter
        limit = request.args.get('limit', 12, type=int)
        if limit < 1:
            limit = 12
        if limit > 100:
            limit = 100
        
        # Get price history
        price_records = get_price_history(product_id, limit=limit)
        
        # Convert to list of dicts
        price_list = []
        for record in price_records:
            price_list.append({
                'date': record['date'],
                'price': record['price'],
                'currency': record['currency']
            })
        
        # Get statistics
        stats = get_price_statistics(product_id)
        
        response = {
            'status': 'success',
            'product_id': product_id,
            'price_history': price_list,
            'statistics': stats,
            'has_price_data': len(price_list) > 0
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error retrieving price history for product {product_id}: {e}", exc_info=True)
        return create_error_response(
            'PRICE_HISTORY_ERROR',
            'Failed to retrieve price history',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )

@app.route('/api/products/<int:product_id>/price-history', methods=['POST'])
def add_product_price_history(product_id):
    """
    Add price history records for a product.
    
    JSON body:
    - prices: Array of price records with 'date', 'price', and optional 'currency'
    
    Returns:
    - 200: Success with number of records added (0 if no valid records)
    - 400: Validation error
    - 404: Product not found
    - 500: Server error
    """
    try:
        # Check if product exists
        product = get_product_by_id(product_id)
        if not product:
            return create_error_response(
                'PRODUCT_NOT_FOUND',
                f'Product with ID {product_id} not found',
                status_code=404
            )
        
        # Parse JSON body
        data = request.get_json()
        if not data or 'prices' not in data:
            # Return 200 OK with 0 records inserted (graceful handling for empty requests)
            return jsonify({
                'status': 'success',
                'product_id': product_id,
                'records_inserted': 0,
                'records_validated': 0,
                'note': 'No price data provided'
            }), 200
        
        prices = data['prices']
        if not isinstance(prices, list):
            return create_error_response(
                'INVALID_PRICES',
                'prices must be an array',
                status_code=400
            )
        
        # If prices array is empty, return 200 OK (graceful handling)
        if len(prices) == 0:
            return jsonify({
                'status': 'success',
                'product_id': product_id,
                'records_inserted': 0,
                'records_validated': 0,
                'note': 'Empty price array provided'
            }), 200
        
        # Validate price records
        valid_records = []
        errors = []
        
        for i, record in enumerate(prices):
            if not isinstance(record, dict):
                errors.append(f"Record {i}: must be an object")
                continue
            
            date = record.get('date')
            price = record.get('price')
            currency = record.get('currency', 'USD')
            
            # Validate date format (YYYY-MM-DD)
            if not date:
                errors.append(f"Record {i}: date is required")
                continue
            
            try:
                datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                errors.append(f"Record {i}: invalid date format (use YYYY-MM-DD)")
                continue
            
            # Validate price
            if price is None:
                errors.append(f"Record {i}: price is required")
                continue
            
            try:
                price = float(price)
                if price < 0:
                    errors.append(f"Record {i}: price must be non-negative")
                    continue
            except (ValueError, TypeError):
                errors.append(f"Record {i}: price must be a number")
                continue
            
            valid_records.append({
                'date': date,
                'price': price,
                'currency': currency
            })
        
        # If no valid records after validation, return 200 OK (graceful handling)
        if not valid_records:
            return jsonify({
                'status': 'success',
                'product_id': product_id,
                'records_inserted': 0,
                'records_validated': 0,
                'validation_errors': errors,
                'note': f'All {len(prices)} record(s) failed validation'
            }), 200
        
        # Insert price history
        inserted = bulk_insert_price_history(product_id, valid_records)
        
        response = {
            'status': 'success',
            'product_id': product_id,
            'records_inserted': inserted,
            'records_validated': len(valid_records)
        }
        
        if errors:
            response['validation_errors'] = errors
            response['note'] = f'{len(errors)} record(s) skipped due to validation errors'
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error adding price history for product {product_id}: {e}", exc_info=True)
        return create_error_response(
            'PRICE_HISTORY_ERROR',
            'Failed to add price history',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )

@app.route('/api/products/<int:product_id>/performance-history', methods=['GET'])
def get_product_performance_history(product_id):
    try:
        # Check if product exists
        product = get_product_by_id(product_id)
        if not product:
            return create_error_response(
                'PRODUCT_NOT_FOUND',
                f'Product with ID {product_id} not found',
                status_code=404
            )
        
        # Get limit parameter
        limit = request.args.get('limit', 12, type=int)
        if limit < 1:
            limit = 12
        if limit > 100:
            limit = 100
        
        # Get performance history
        performance_records = get_performance_history(product_id, limit=limit)
        
        # Convert to list of dicts
        performance_list = []
        for record in performance_records:
            performance_list.append({
                'date': record['date'],
                'sales': record['sales'],
                'views': record['views'],
                'conversion_rate': record['conversion_rate'],
                'revenue': record['revenue']
            })
        
        # Get statistics
        stats = get_performance_statistics(product_id)
        
        response = {
            'status': 'success',
            'product_id': product_id,
            'performance_history': performance_list,
            'statistics': stats,
            'has_performance_data': len(performance_list) > 0
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error retrieving performance history for product {product_id}: {e}", exc_info=True)
        return create_error_response(
            'PERFORMANCE_HISTORY_ERROR',
            'Failed to retrieve performance history',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )

@app.route('/api/products/<int:product_id>/performance-history', methods=['POST'])
def add_product_performance_history(product_id):

    try:
        # Check if product exists
        product = get_product_by_id(product_id)
        if not product:
            return create_error_response(
                'PRODUCT_NOT_FOUND',
                f'Product with ID {product_id} not found',
                status_code=404
            )
        
        # Parse JSON body
        data = request.get_json()
        if not data or 'performance' not in data:
            return create_error_response(
                'MISSING_PERFORMANCE',
                'performance array is required',
                'Send JSON body with performance array',
                status_code=400
            )
        
        performance = data['performance']
        if not isinstance(performance, list):
            return create_error_response(
                'INVALID_PERFORMANCE',
                'performance must be an array',
                status_code=400
            )
        
        # Validate performance records
        valid_records = []
        errors = []
        
        for i, record in enumerate(performance):
            if not isinstance(record, dict):
                errors.append(f"Record {i}: must be an object")
                continue
            
            date = record.get('date')
            sales = record.get('sales', 0)
            views = record.get('views', 0)
            conversion_rate = record.get('conversion_rate', 0.0)
            revenue = record.get('revenue', 0.0)
            
            # Validate date format (YYYY-MM-DD)
            if not date:
                errors.append(f"Record {i}: date is required")
                continue
            
            try:
                datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                errors.append(f"Record {i}: invalid date format (use YYYY-MM-DD)")
                continue
            
            # Validate numeric fields
            try:
                sales = int(sales)
                if sales < 0:
                    errors.append(f"Record {i}: sales must be non-negative")
                    continue
            except (ValueError, TypeError):
                errors.append(f"Record {i}: sales must be a number")
                continue
            
            try:
                views = int(views)
                if views < 0:
                    errors.append(f"Record {i}: views must be non-negative")
                    continue
            except (ValueError, TypeError):
                errors.append(f"Record {i}: views must be a number")
                continue
            
            try:
                conversion_rate = float(conversion_rate)
                if conversion_rate < 0 or conversion_rate > 100:
                    errors.append(f"Record {i}: conversion_rate must be between 0 and 100")
                    continue
            except (ValueError, TypeError):
                errors.append(f"Record {i}: conversion_rate must be a number")
                continue
            
            try:
                revenue = float(revenue)
                if revenue < 0:
                    errors.append(f"Record {i}: revenue must be non-negative")
                    continue
            except (ValueError, TypeError):
                errors.append(f"Record {i}: revenue must be a number")
                continue
            
            valid_records.append({
                'date': date,
                'sales': sales,
                'views': views,
                'conversion_rate': conversion_rate,
                'revenue': revenue
            })
        
        if not valid_records:
            return create_error_response(
                'NO_VALID_RECORDS',
                'No valid performance records provided',
                'Check the validation errors',
                {'errors': errors},
                status_code=400
            )
        
        # Insert performance history
        inserted = bulk_insert_performance_history(product_id, valid_records)
        
        response = {
            'status': 'success',
            'product_id': product_id,
            'records_inserted': inserted,
            'records_validated': len(valid_records)
        }
        
        if errors:
            response['validation_errors'] = errors
            response['note'] = f'{len(errors)} record(s) skipped due to validation errors'
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error adding performance history for product {product_id}: {e}", exc_info=True)
        return create_error_response(
            'PERFORMANCE_HISTORY_ERROR',
            'Failed to add performance history',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )

@app.route('/api/features/info', methods=['GET'])
def get_feature_extraction_info_endpoint():
    """
    Get feature extraction configuration and status.
    
    Returns information about whether CLIP or legacy features are being used.
    
    Returns:
    - 200: Success with feature extraction info
    - 500: Server error
    """
    try:
        info = get_feature_extraction_info()
        
        return jsonify({
            'status': 'success',
            'feature_extraction': info
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting feature extraction info: {e}", exc_info=True)
        return create_error_response(
            'FEATURE_INFO_ERROR',
            'Failed to get feature extraction information',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/clip/info', methods=['GET'])
def get_clip_info():
    """
    Get CLIP model information and status.
    
    Returns:
    - 200: Success with CLIP model info
    - 500: Server error
    """
    try:
        from image_processing_clip import get_model_info, is_clip_available
        
        if not is_clip_available():
            return jsonify({
                'status': 'unavailable',
                'message': 'CLIP is not available. PyTorch or sentence-transformers not installed.',
                'suggestion': 'Install dependencies: pip install torch sentence-transformers'
            }), 200
        
        info = get_model_info()
        
        return jsonify({
            'status': 'success',
            'clip_info': info
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting CLIP info: {e}", exc_info=True)
        return create_error_response(
            'CLIP_INFO_ERROR',
            'Failed to get CLIP information',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/clip/cache/clear', methods=['POST'])
def clear_clip_cache():
    """
    Clear CLIP model cache.
    
    JSON body (optional):
    - keep_config: Keep configuration file (default: true)
    
    Returns:
    - 200: Success with cache clear info
    - 500: Server error
    """
    try:
        from image_processing_clip import clear_model_cache
        
        data = request.get_json() or {}
        keep_config = data.get('keep_config', True)
        
        result = clear_model_cache(keep_config=keep_config)
        
        return jsonify({
            'status': 'success',
            'result': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing CLIP cache: {e}", exc_info=True)
        return create_error_response(
            'CACHE_CLEAR_ERROR',
            'Failed to clear CLIP cache',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/clip/cache/size', methods=['GET'])
def get_clip_cache_size():
    """
    Get CLIP cache size information.
    
    Returns:
    - 200: Success with cache size info
    - 500: Server error
    """
    try:
        from image_processing_clip import get_cache_size
        
        size_info = get_cache_size()
        
        return jsonify({
            'status': 'success',
            'cache_size': size_info
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting cache size: {e}", exc_info=True)
        return create_error_response(
            'CACHE_SIZE_ERROR',
            'Failed to get cache size',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/clip/model/set', methods=['POST'])
def set_clip_model():
    """
    Set preferred CLIP model.
    
    JSON body:
    - model_name: CLIP model name (required)
    
    Returns:
    - 200: Success
    - 400: Invalid model name
    - 500: Server error
    """
    try:
        from image_processing_clip import set_model_preference, AVAILABLE_MODELS
        
        data = request.get_json()
        
        if not data or 'model_name' not in data:
            return create_error_response(
                'MISSING_MODEL_NAME',
                'model_name is required',
                'Include model_name in request body',
                status_code=400
            )
        
        model_name = data['model_name']
        
        if model_name not in AVAILABLE_MODELS:
            return create_error_response(
                'INVALID_MODEL_NAME',
                f'Invalid model name: {model_name}',
                f'Available models: {", ".join(AVAILABLE_MODELS.keys())}',
                status_code=400
            )
        
        result = set_model_preference(model_name)
        
        return jsonify({
            'status': 'success',
            'result': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting CLIP model: {e}", exc_info=True)
        return create_error_response(
            'MODEL_SET_ERROR',
            'Failed to set CLIP model',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/clip/config', methods=['GET'])
def get_clip_config():
    """
    Get CLIP configuration.
    
    Returns:
    - 200: Success with config
    - 500: Server error
    """
    try:
        from image_processing_clip import load_clip_config
        
        config = load_clip_config()
        
        return jsonify({
            'status': 'success',
            'config': config
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting CLIP config: {e}", exc_info=True)
        return create_error_response(
            'CONFIG_ERROR',
            'Failed to get CLIP configuration',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/clip/config', methods=['POST'])
def update_clip_config():
    """
    Update CLIP configuration.
    
    JSON body:
    - use_clip: Enable/disable CLIP (optional)
    - fallback_to_legacy: Enable/disable fallback to legacy features (optional)
    
    Returns:
    - 200: Success
    - 500: Server error
    """
    try:
        from image_processing_clip import enable_clip, set_fallback_to_legacy, load_clip_config
        
        data = request.get_json() or {}
        
        results = []
        
        if 'use_clip' in data:
            result = enable_clip(data['use_clip'])
            results.append(result)
        
        if 'fallback_to_legacy' in data:
            result = set_fallback_to_legacy(data['fallback_to_legacy'])
            results.append(result)
        
        # Get updated config
        config = load_clip_config()
        
        return jsonify({
            'status': 'success',
            'config': config,
            'updates': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating CLIP config: {e}", exc_info=True)
        return create_error_response(
            'CONFIG_UPDATE_ERROR',
            'Failed to update CLIP configuration',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/clip/download-instructions', methods=['GET'])
def get_clip_download_instructions():
    """
    Get manual download instructions for CLIP model.
    
    Query parameters:
    - model_name: CLIP model name (optional, default: clip-ViT-B-32)
    
    Returns:
    - 200: Success with instructions
    - 500: Server error
    """
    try:
        from image_processing_clip import get_manual_download_instructions
        
        model_name = request.args.get('model_name', 'clip-ViT-B-32')
        
        instructions = get_manual_download_instructions(model_name)
        
        return jsonify({
            'status': 'success',
            'instructions': instructions
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting download instructions: {e}", exc_info=True)
        return create_error_response(
            'INSTRUCTIONS_ERROR',
            'Failed to get download instructions',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


# ============ Catalog Management API Endpoints ============

@app.route('/api/catalog/stats', methods=['GET'])
def get_catalog_statistics():
    """
    Get comprehensive catalog statistics.
    
    Returns:
    - 200: Success with catalog stats
    - 500: Server error
    """
    try:
        stats = get_catalog_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting catalog stats: {e}", exc_info=True)
        return create_error_response(
            'STATS_ERROR',
            'Failed to get catalog statistics',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/categories', methods=['GET'])
def get_categories():
    """
    Get list of all categories.
    
    Returns:
    - 200: Success with categories list
    - 500: Server error
    """
    try:
        categories = get_all_categories()
        return jsonify({'categories': categories}), 200
    except Exception as e:
        logger.error(f"Error getting categories: {e}", exc_info=True)
        return create_error_response(
            'CATEGORIES_ERROR',
            'Failed to get categories',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/products', methods=['GET'])
def get_catalog_products():
    try:
        # Check if specific IDs are requested (batch fetch mode)
        ids_param = request.args.get('ids', '')
        if ids_param:
            # Batch fetch mode: get specific products by ID
            try:
                requested_ids = [int(id.strip()) for id in ids_param.split(',') if id.strip()]
                if not requested_ids:
                    return jsonify({
                        'products': [],
                        'total': 0,
                        'page': 1,
                        'limit': 0,
                        'pages': 0,
                        'total_pages': 0
                    }), 200

                # Keep order while removing duplicates.
                product_ids = list(dict.fromkeys(requested_ids))
                max_ids = 2000
                if len(product_ids) > max_ids:
                    logger.info(f"[GET-PRODUCTS] Truncating batch ID request from {len(product_ids)} to {max_ids}")
                    product_ids = product_ids[:max_ids]

                logger.info(f"[GET-PRODUCTS] Batch fetch mode: {len(product_ids)} IDs")

                products_by_id = get_products_by_ids(product_ids)
                products_with_features = set()

                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    chunk_size = 900  # Stay below SQLite variable limit (typically 999)
                    for idx in range(0, len(product_ids), chunk_size):
                        chunk = product_ids[idx:idx + chunk_size]
                        if not chunk:
                            continue
                        placeholders = ','.join('?' * len(chunk))
                        cursor.execute(
                            f'''
                            SELECT DISTINCT product_id
                            FROM features
                            WHERE product_id IN ({placeholders})
                            ''',
                            chunk
                        )
                        products_with_features.update(row['product_id'] for row in cursor.fetchall())

                products = []
                for product_id in product_ids:
                    product = products_by_id.get(product_id)
                    if not product:
                        continue

                    product_dict = dict(product)
                    product_dict['has_features'] = product_id in products_with_features
                    product_dict['filename'] = os.path.basename(product_dict.get('image_path') or '')
                    product_dict['is_historical'] = bool(product_dict.get('is_historical'))

                    # Parse and include metadata JSON for frontend sorting/filtering
                    if product_dict.get('metadata'):
                        try:
                            product_dict['metadata'] = json.loads(product_dict['metadata'])
                        except (json.JSONDecodeError, TypeError):
                            product_dict['metadata'] = {}
                    else:
                        product_dict['metadata'] = {}

                    products.append(product_dict)

                logger.info(f"[GET-PRODUCTS] Batch result: {len(products)} products fetched")

                return jsonify({
                    'products': products,
                    'total': len(products),
                    'page': 1,
                    'limit': len(products),
                    'pages': 1,
                    'total_pages': 1
                }), 200

            except ValueError as e:
                return create_error_response(
                    'INVALID_IDS',
                    'Invalid product IDs format',
                    'IDs must be comma-separated integers',
                    status_code=400
                )

        # Normal pagination mode
        raw_page = request.args.get('page', 1)
        raw_limit = request.args.get('limit', 50)
        (page, limit), _ = validate_page_params(raw_page, raw_limit, max_limit=500)
        search = sanitize_search_query(request.args.get('search', ''))
        category = request.args.get('category', '')
        product_type = request.args.get('type', '')
        features = request.args.get('features', '')
        sort_by = request.args.get('sort', 'date_desc')

        # Convert type filter
        is_historical = None
        if product_type == 'historical':
            is_historical = True
        elif product_type == 'new':
            is_historical = False

        # Convert features filter
        has_features = None
        if features == 'has_features':
            has_features = True
        elif features == 'no_features':
            has_features = False

        logger.info(f"[GET-PRODUCTS] Query: type={product_type}, is_historical={is_historical}, limit={limit}")

        result = get_products_paginated(
            page=page,
            limit=limit,
            search=search if search else None,
            category=category if category else None,
            is_historical=is_historical,
            has_features=has_features,
            sort_by=sort_by
        )

        logger.info(f"[GET-PRODUCTS] Result: {result['total']} total products, {len(result['products'])} returned")

        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting products: {e}", exc_info=True)
        return create_error_response(
            'PRODUCTS_ERROR',
            'Failed to get products',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/products/<int:product_id>', methods=['PUT'])
def update_catalog_product(product_id):
    try:
        product = get_product_by_id(product_id)
        if not product:
            return create_error_response(
                'PRODUCT_NOT_FOUND',
                f'Product with ID {product_id} not found',
                status_code=404
            )
        
        data = request.get_json() or {}
        
        # Validate category
        category = data.get('category')
        if category is not None:
            category, error = validate_category(category)
            if error:
                return create_error_response('INVALID_CATEGORY', error, status_code=400)
        
        # Validate product_name
        product_name = data.get('product_name')
        if product_name is not None:
            product_name, error = validate_product_name(product_name)
            if error:
                return create_error_response('INVALID_NAME', error, status_code=400)
        
        # Validate SKU
        sku = data.get('sku')
        if sku is not None:
            sku, error = validate_sku(sku)
            if error:
                return create_error_response('INVALID_SKU', error, status_code=400)
        
        success = update_product(
            product_id,
            category=category,
            product_name=product_name,
            sku=sku
        )

        # Invalidate CSV cache since product was updated
        invalidate_csv_cache()

        return jsonify({
            'status': 'success',
            'message': 'Product updated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating product {product_id}: {e}", exc_info=True)
        return create_error_response(
            'UPDATE_ERROR',
            'Failed to update product',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/products/<int:product_id>', methods=['DELETE'])
def delete_catalog_product(product_id):
    """
    Delete a product and its associated data.
    
    Returns:
    - 200: Success
    - 404: Product not found
    - 500: Server error
    """
    try:
        product = get_product_by_id(product_id)
        if not product:
            return create_error_response(
                'PRODUCT_NOT_FOUND',
                f'Product with ID {product_id} not found',
                status_code=404
            )
        
        # Get image path and category before deletion
        image_path = product['image_path']
        category = product['category']
        
        # Delete product
        success = delete_product(product_id)
        
        # Invalidate FAISS index for this category
        try:
            from database import invalidate_faiss_index
            invalidate_faiss_index(category)
            logger.debug(f"Invalidated FAISS index for category '{category}'")
        except Exception as e:
            logger.warning(f"Failed to invalidate FAISS index: {e}")
        
        # Delete image file - ONLY if it's in the uploads folder (not user source folder)
        if image_path and os.path.exists(image_path):
            # Safety check: only delete files in uploads folder, never delete user source files
            uploads_folder = app.config['UPLOAD_FOLDER']
            is_managed_file = os.path.realpath(image_path).startswith(os.path.realpath(uploads_folder))

            if is_managed_file:
                try:
                    os.remove(image_path)
                    logger.debug(f"Deleted managed image file: {image_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete image file: {e}")
            else:
                logger.debug(f"Skipped deletion of external image file (not in uploads folder): {image_path}")

        # Invalidate CSV cache since product was deleted
        invalidate_csv_cache()

        return jsonify({
            'status': 'success',
            'message': 'Product deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting product {product_id}: {e}", exc_info=True)
        return create_error_response(
            'DELETE_ERROR',
            'Failed to delete product',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/products/<int:product_id>/reextract', methods=['POST'])
def reextract_product_features(product_id):
    """
    Re-extract features for a product.
    
    Returns:
    - 200: Success
    - 404: Product not found
    - 500: Server error
    """
    try:
        product = get_product_by_id(product_id)
        if not product:
            return create_error_response(
                'PRODUCT_NOT_FOUND',
                f'Product with ID {product_id} not found',
                status_code=404
            )
        
        image_path = product['image_path']
        if not image_path or not os.path.exists(image_path):
            return create_error_response(
                'IMAGE_NOT_FOUND',
                'Product image file not found',
                'The image file may have been deleted',
                status_code=400
            )
        
        # Delete existing features
        delete_features(product_id)
        
        # Re-extract features
        features, embedding_type, embedding_version = extract_features_unified(image_path)
        
        # Store new features
        insert_features(
            product_id=product_id,
            color_features=features['color_features'],
            shape_features=features['shape_features'],
            texture_features=features['texture_features'],
            embedding_type=embedding_type,
            embedding_version=embedding_version
        )
        
        # Invalidate FAISS index for this category (features changed)
        if embedding_type == 'clip' and product['is_historical']:
            try:
                from database import invalidate_faiss_index
                category = product['category']
                invalidate_faiss_index(category)
                logger.debug(f"Invalidated FAISS index for category '{category}' after re-extracting features for product {product_id}")
            except Exception as e:
                logger.warning(f"Failed to invalidate FAISS index: {e}")
        
        return jsonify({
            'status': 'success',
            'message': f'Features re-extracted successfully (type: {embedding_type})'
        }), 200
        
    except Exception as e:
        logger.error(f"Error re-extracting features for product {product_id}: {e}", exc_info=True)
        return create_error_response(
            'REEXTRACT_ERROR',
            'Failed to re-extract features',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/products/bulk-delete', methods=['POST'])
def bulk_delete_catalog_products():
    """
    Delete multiple products at once with validation.
    
    JSON body:
    - product_ids: List of product IDs to delete (max 100)
    
    Returns:
    - 200: Success with count
    - 400: Invalid request or validation error
    - 500: Server error
    """
    try:
        data = request.get_json()
        if not data or 'product_ids' not in data:
            return create_error_response(
                'MISSING_IDS',
                'product_ids array is required',
                status_code=400
            )
        
        # Validate product IDs
        product_ids, error = validate_product_ids(data['product_ids'], max_count=100)
        if error:
            return create_error_response('INVALID_IDS', error, status_code=400)
        
        deleted_count = bulk_delete_products(product_ids)
        
        # Invalidate all FAISS indexes (bulk delete may affect multiple categories)
        try:
            from database import invalidate_faiss_index
            invalidate_faiss_index()  # Invalidate all categories
            logger.debug(f"Invalidated all FAISS indexes after bulk delete")
        except Exception as e:
            logger.warning(f"Failed to invalidate FAISS indexes: {e}")

        # Invalidate CSV cache since products were deleted
        invalidate_csv_cache()

        return jsonify({
            'status': 'success',
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} product(s)'
        }), 200
        
    except Exception as e:
        logger.error(f"Error bulk deleting products: {e}", exc_info=True)
        return create_error_response(
            'BULK_DELETE_ERROR',
            'Failed to delete products',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/products/bulk-update', methods=['POST'])
def bulk_update_catalog_products():
    """
    Update multiple products at once with strict validation.
    
    JSON body:
    - product_ids: List of product IDs to update (max 100)
    - category: New category (optional, max 100 chars)
    - product_name: New name (optional, max 200 chars)
    - sku: New SKU (optional, max 50 chars)
    
    Returns:
    - 200: Success with count
    - 400: Invalid request or validation error
    - 500: Server error
    """
    try:
        data = request.get_json()
        if not data or 'product_ids' not in data:
            return create_error_response(
                'MISSING_IDS',
                'product_ids array is required',
                status_code=400
            )
        
        # Validate product IDs
        product_ids, error = validate_product_ids(data['product_ids'], max_count=100)
        if error:
            return create_error_response('INVALID_IDS', error, status_code=400)
        
        # Validate category
        category = data.get('category')
        if category is not None:
            category, error = validate_category(category)
            if error:
                return create_error_response('INVALID_CATEGORY', error, status_code=400)
        
        # Validate product_name
        product_name = data.get('product_name')
        if product_name is not None:
            product_name, error = validate_product_name(product_name)
            if error:
                return create_error_response('INVALID_NAME', error, status_code=400)
        
        # Validate SKU
        sku = data.get('sku')
        if sku is not None:
            sku, error = validate_sku(sku)
            if error:
                return create_error_response('INVALID_SKU', error, status_code=400)
        
        updated_count = bulk_update_products(
            product_ids,
            category=category,
            product_name=product_name,
            sku=sku
        )

        # Invalidate CSV cache since products were updated
        invalidate_csv_cache()

        return jsonify({
            'status': 'success',
            'updated_count': updated_count,
            'message': f'Updated {updated_count} product(s)'
        }), 200
        
    except Exception as e:
        logger.error(f"Error bulk updating products: {e}", exc_info=True)
        return create_error_response(
            'BULK_UPDATE_ERROR',
            'Failed to update products',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/products/bulk-reextract', methods=['POST'])
def bulk_reextract_features():
    """
    Re-extract features for multiple products.
    
    JSON body:
    - product_ids: List of product IDs
    
    Returns:
    - 200: Success with counts
    - 400: Invalid request
    - 500: Server error
    """
    try:
        data = request.get_json()
        if not data or 'product_ids' not in data:
            return create_error_response(
                'MISSING_IDS',
                'product_ids array is required',
                status_code=400
            )
        
        product_ids = data['product_ids']
        if not isinstance(product_ids, list):
            return create_error_response(
                'INVALID_IDS',
                'product_ids must be an array',
                status_code=400
            )
        
        success_count = 0
        fail_count = 0
        
        for product_id in product_ids:
            try:
                product = get_product_by_id(product_id)
                if not product:
                    fail_count += 1
                    continue
                
                image_path = product['image_path']
                if not image_path or not os.path.exists(image_path):
                    fail_count += 1
                    continue
                
                # Delete existing features
                delete_features(product_id)
                
                # Re-extract features
                features, embedding_type, embedding_version = extract_features_unified(image_path)
                
                # Store new features
                insert_features(
                    product_id=product_id,
                    color_features=features['color_features'],
                    shape_features=features['shape_features'],
                    texture_features=features['texture_features'],
                    embedding_type=embedding_type,
                    embedding_version=embedding_version
                )
                
                success_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to re-extract features for product {product_id}: {e}")
                fail_count += 1
        
        return jsonify({
            'status': 'success',
            'success_count': success_count,
            'fail_count': fail_count,
            'message': f'Re-extracted features for {success_count} product(s), {fail_count} failed'
        }), 200
        
    except Exception as e:
        logger.error(f"Error bulk re-extracting features: {e}", exc_info=True)
        return create_error_response(
            'BULK_REEXTRACT_ERROR',
            'Failed to re-extract features',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/cleanup', methods=['POST'])
def cleanup_catalog():
    """
    Clean up products by type.
    
    JSON body:
    - type: 'all', 'historical', 'new', or 'matches'
    
    Returns:
    - 200: Success with counts
    - 400: Invalid type
    - 500: Server error
    """
    try:
        data = request.get_json()
        if not data or 'type' not in data:
            return create_error_response(
                'MISSING_TYPE',
                'type is required',
                status_code=400
            )
        
        cleanup_type = data['type']
        
        if cleanup_type == 'matches':
            deleted = clear_all_matches()
            return jsonify({
                'status': 'success',
                'matches_deleted': deleted,
                'message': f'Deleted {deleted} match(es)'
            }), 200
        
        if cleanup_type not in ['all', 'historical', 'new']:
            return create_error_response(
                'INVALID_TYPE',
                'type must be all, historical, new, or matches',
                status_code=400
            )
        
        result = clear_products_by_type(cleanup_type)
        
        return jsonify({
            'status': 'success',
            **result,
            'message': f'Deleted {result["products_deleted"]} product(s)'
        }), 200
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)
        return create_error_response(
            'CLEANUP_ERROR',
            'Cleanup failed',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/cleanup/categories', methods=['POST'])
def cleanup_by_categories():
    """
    Clean up products by categories.
    
    JSON body:
    - categories: List of category names to delete
    
    Returns:
    - 200: Success with count
    - 400: Invalid request
    - 500: Server error
    """
    try:
        data = request.get_json()
        if not data or 'categories' not in data:
            return create_error_response(
                'MISSING_CATEGORIES',
                'categories array is required',
                status_code=400
            )
        
        categories = data['categories']
        if not isinstance(categories, list):
            return create_error_response(
                'INVALID_CATEGORIES',
                'categories must be an array',
                status_code=400
            )
        
        result = clear_products_by_categories(categories)
        
        return jsonify({
            'status': 'success',
            **result,
            'message': f'Deleted {result["products_deleted"]} product(s) from selected categories'
        }), 200
        
    except Exception as e:
        logger.error(f"Error cleaning up categories: {e}", exc_info=True)
        return create_error_response(
            'CATEGORY_CLEANUP_ERROR',
            'Category cleanup failed',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/cleanup/by-date', methods=['POST'])
def cleanup_by_date():
    """
    Clean up products older than specified days.
    
    JSON body:
    - older_than_days: Number of days
    
    Returns:
    - 200: Success with count
    - 400: Invalid request
    - 500: Server error
    """
    try:
        data = request.get_json()
        if not data or 'older_than_days' not in data:
            return create_error_response(
                'MISSING_DAYS',
                'older_than_days is required',
                status_code=400
            )
        
        try:
            days = int(data['older_than_days'])
            if days < 1:
                raise ValueError()
        except (ValueError, TypeError):
            return create_error_response(
                'INVALID_DAYS',
                'older_than_days must be a positive integer',
                status_code=400
            )
        
        result = clear_products_by_date(days)
        
        return jsonify({
            'status': 'success',
            **result,
            'message': f'Deleted {result["products_deleted"]} product(s) older than {days} days'
        }), 200
        
    except Exception as e:
        logger.error(f"Error cleaning up by date: {e}", exc_info=True)
        return create_error_response(
            'DATE_CLEANUP_ERROR',
            'Date cleanup failed',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/vacuum', methods=['POST'])
def vacuum_catalog_database():
    """
    Vacuum the database to reclaim disk space.
    
    Returns:
    - 200: Success with size info
    - 500: Server error
    """
    try:
        result = vacuum_database()
        
        return jsonify({
            'status': 'success',
            **result,
            'message': f'Database vacuumed. Reclaimed {result["space_reclaimed_mb"]} MB'
        }), 200
        
    except Exception as e:
        logger.error(f"Error vacuuming database: {e}", exc_info=True)
        return create_error_response(
            'VACUUM_ERROR',
            'Database vacuum failed',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/clear-images', methods=['POST'])
def clear_catalog_images():
    """
    Clear all uploaded image files but keep metadata.
    
    Returns:
    - 200: Success with count
    - 500: Server error
    """
    try:
        result = clear_uploaded_images()
        
        return jsonify({
            'status': 'success',
            **result,
            'message': f'Deleted {result["files_deleted"]} image file(s), reclaimed {result["space_reclaimed_mb"]} MB'
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing images: {e}", exc_info=True)
        return create_error_response(
            'CLEAR_IMAGES_ERROR',
            'Failed to clear images',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/working-catalog/stats', methods=['GET'])
def get_working_catalog_stats():
    """
    Get live statistics for the working catalog (main database).

    Returns:
    - 200: Success with current stats
    - 500: Server error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Count products in main database
            cursor.execute('SELECT COUNT(*) FROM products')
            product_count = cursor.fetchone()[0]

            # Count products with features
            cursor.execute('SELECT COUNT(DISTINCT product_id) FROM features')
            features_count = cursor.fetchone()[0]

            # Count by type
            cursor.execute('SELECT COUNT(*) FROM products WHERE is_historical = 1')
            historical_count = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM products WHERE is_historical = 0')
            new_count = cursor.fetchone()[0]

        return jsonify({
            'success': True,
            'product_count': product_count,
            'features_count': features_count,
            'historical_count': historical_count,
            'new_count': new_count,
            'is_active': True
        }), 200

    except Exception as e:
        logger.error(f"Error getting working catalog stats: {e}", exc_info=True)
        return create_error_response(
            'WORKING_CATALOG_STATS_ERROR',
            'Failed to get working catalog stats',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/working-catalog/clear', methods=['POST'])
def clear_working_catalog():
    """
    Clear all data from the working catalog (main database).
    WARNING: This deletes all products and features.

    Returns:
    - 200: Success with deletion counts
    - 500: Server error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Count before deletion
            cursor.execute('SELECT COUNT(*) FROM features')
            features_before = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM products')
            products_before = cursor.fetchone()[0]

            # Delete all features first (due to foreign key)
            cursor.execute('DELETE FROM features')
            features_deleted = cursor.rowcount

            # Delete all products
            cursor.execute('DELETE FROM products')
            products_deleted = cursor.rowcount

            conn.commit()

        # Invalidate CSV cache since products were deleted
        invalidate_csv_cache()

        return jsonify({
            'success': True,
            'products_deleted': products_deleted,
            'features_deleted': features_deleted,
            'message': f'Cleared {products_deleted} products and {features_deleted} feature sets from working catalog'
        }), 200

    except Exception as e:
        logger.error(f"Error clearing working catalog: {e}", exc_info=True)
        return create_error_response(
            'WORKING_CATALOG_CLEAR_ERROR',
            'Failed to clear working catalog',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalog/export', methods=['GET'])
def export_catalog():
    """
    Export catalog to CSV.
    
    Returns:
    - 200: CSV file download
    - 500: Server error
    """
    try:
        return Response(
            stream_with_context(stream_catalog_csv()),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=catalog-backup-{datetime.now().strftime("%Y%m%d")}.csv'
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting catalog: {e}", exc_info=True)
        return create_error_response(
            'EXPORT_ERROR',
            'Failed to export catalog',
            'Please try again',
            {'error': str(e)},
            status_code=500
        )


# ============ Catalog Snapshot API Endpoints ============

# Snapshot/catalog endpoints moved to dedicated module to keep this file focused.
try:
    from .routes.snapshot_routes import snapshot_bp, configure_snapshot_routes  # type: ignore
except ImportError:
    from routes.snapshot_routes import snapshot_bp, configure_snapshot_routes  # type: ignore

configure_snapshot_routes(
    logger=logger,
    create_error_response=create_error_response,
    invalidate_csv_cache=invalidate_csv_cache,
    invalidate_catalog_categories_cache=invalidate_catalog_categories_cache,
    get_cached_csv=get_cached_csv,
    cache_csv_data=cache_csv_data,
    crash_detected=crash_detected,
)
if 'snapshot_routes' not in app.blueprints:
    app.register_blueprint(snapshot_bp)

@app.route('/api/csv/extract', methods=['GET'])
def extract_csv_from_current_db():
    """Extract CSV from current main database for a specific section

    Query Parameters:
    - type: 'historical' or 'new'

    Returns CSV content as downloadable file or error.
    Used by "Add to Existing" feature to let users review current catalog before adding.
    """
    try:
        from snapshot_manager import extract_csv_from_db, DEFAULT_DB_PATH

        # Get type parameter (historical or new)
        csv_type = request.args.get('type', 'historical').lower()
        if csv_type not in ['historical', 'new']:
            return create_error_response(
                'INVALID_TYPE',
                f"Invalid type '{csv_type}'. Must be 'historical' or 'new'",
                status_code=400
            )

        is_historical = (csv_type == 'historical')

        # Check if main database exists
        if not os.path.exists(DEFAULT_DB_PATH):
            return create_error_response(
                'NO_DATABASE',
                'Main database not found',
                status_code=404
            )

        # Extract CSV from current main database
        result = extract_csv_from_db(DEFAULT_DB_PATH, is_historical=is_historical)

        if not result:
            return create_error_response(
                'EXTRACTION_FAILED',
                f'No {csv_type} products found or failed to extract CSV',
                status_code=404
            )

        csv_content, row_count = result

        # Generate filename with current date
        filename = f"{csv_type}-products-{datetime.now().strftime('%Y%m%d')}.csv"

        # Return CSV as downloadable file
        return send_file(
            io.BytesIO(csv_content.encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"Error extracting CSV from database: {e}", exc_info=True)
        return create_error_response(
            'EXTRACTION_ERROR',
            'Failed to extract CSV from database',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/memory/cleanup', methods=['POST'])
def trigger_memory_cleanup():
    """
    Manually trigger memory cleanup to free resources.
    
    Useful for:
    - Long-running sessions
    - After processing large catalogs
    - Testing memory leak fixes
    
    Returns:
    - 200: Success with cleanup stats
    - 500: Server error
    """
    try:
        import gc
        
        cleanup_stats = {
            'clip_model_cleared': False,
            'feature_cache_cleared': False,
            'csv_cache_cleared': False,
            'category_cache_cleared': False,
            'garbage_collected': 0,
            'cuda_cache_cleared': False
        }
        
        # Clear CLIP model cache
        try:
            from image_processing_clip import clear_clip_model_cache
            clear_clip_model_cache()
            cleanup_stats['clip_model_cleared'] = True
            logger.info("CLIP model cache cleared via API")
        except Exception as e:
            logger.warning(f"Failed to clear CLIP model cache: {e}")

        # Clear feature cache singleton
        try:
            from feature_cache import clear_all_caches
            clear_all_caches()
            cleanup_stats['feature_cache_cleared'] = True
            logger.info("Feature cache cleared via API")
        except Exception as e:
            logger.warning(f"Failed to clear feature cache: {e}")

        # Clear in-process CSV/category caches
        try:
            invalidate_csv_cache()
            cleanup_stats['csv_cache_cleared'] = True
        except Exception as e:
            logger.warning(f"Failed to clear CSV cache: {e}")

        try:
            invalidate_catalog_categories_cache()
            cleanup_stats['category_cache_cleared'] = True
        except Exception as e:
            logger.warning(f"Failed to clear catalog category cache: {e}")
        
        # Force garbage collection
        try:
            collected = gc.collect()
            cleanup_stats['garbage_collected'] = collected
            logger.info(f"Garbage collection freed {collected} objects")
        except Exception as e:
            logger.warning(f"Failed to run garbage collection: {e}")
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cleanup_stats['cuda_cache_cleared'] = True
                logger.info("CUDA cache cleared via API")
        except Exception:
            pass

        return jsonify({
            'status': 'success',
            'message': 'Memory cleanup completed',
            'stats': cleanup_stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}", exc_info=True)
        return create_error_response(
            'CLEANUP_ERROR',
            'Failed to perform memory cleanup',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/memory/stats', methods=['GET'])
def get_memory_stats():
    """
    Get current memory usage statistics.
    
    Returns:
    - 200: Success with memory stats
    - 500: Server error
    """
    try:
        import gc
        import psutil
        import os
        
        # Get process memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        stats = {
            'process_memory_mb': round(memory_info.rss / 1024 / 1024, 2),
            'process_memory_percent': round(process.memory_percent(), 2),
            'garbage_objects': len(gc.get_objects()),
            'garbage_stats': gc.get_stats()
        }
        
        # Add GPU memory if available
        try:
            from image_processing_clip import get_gpu_memory_info
            gpu_info = get_gpu_memory_info()
            if gpu_info.get('available'):
                stats['gpu_memory'] = gpu_info
        except Exception:
            pass

        # Add CLIP model info
        try:
            from image_processing_clip import get_model_info
            model_info = get_model_info()
            stats['clip_model'] = {
                'loaded': model_info.get('loaded', False),
                'model_name': model_info.get('model_name'),
                'device': model_info.get('device'),
                'cache_size_mb': model_info.get('cache_size_mb', 0)
            }
        except Exception:
            pass
        
        return jsonify({
            'status': 'success',
            'stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}", exc_info=True)
        return create_error_response(
            'STATS_ERROR',
            'Failed to get memory stats',
            {'error': str(e)},
            status_code=500
        )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
