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
from typing import Optional, List
from werkzeug.utils import secure_filename
from datetime import datetime

# Set PyTorch environment variables BEFORE importing torch/CLIP
# This fixes GPU memory fragmentation issues with AMD ROCm
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

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
    vacuum_database, clear_uploaded_images, export_catalog_csv, delete_features,
    get_db_connection,
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
CORS(app)

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
# No max content length - handle large files gracefully with proper error handling

# Ensure upload directory exists (get_uploads_dir() already does this, but keeping for clarity)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
_csv_cache_max_size = 10  # Max 10 cached CSVs (~10-100MB depending on size)

def cache_csv_data(cache_key, csv_data):
    """Add CSV data to cache with LRU eviction"""
    global _csv_cache
    # Remove oldest entry if cache is full
    if len(_csv_cache) >= _csv_cache_max_size:
        oldest_key, oldest_data = _csv_cache.popitem(last=False)
        logger.debug(f"CSV cache evicted oldest entry: {oldest_key}")
    _csv_cache[cache_key] = csv_data
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
        except:
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
    except:
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
        except:
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

@app.route('/api/network/local-ip', methods=['GET'])
def get_local_ip_endpoint():
    """Get local IP address for mobile connection"""
    import socket

    def get_local_ip():
        try:
            # Connect to a public DNS server to find local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            logger.debug(f"Failed to get local IP: {e}")
            return "127.0.0.1"

    primary_ip = get_local_ip()
    port = request.environ.get('SERVER_PORT', 5000)

    return jsonify({
        'primary_ip': primary_ip,
        'port': port,
        'mobile_url': f'http://{primary_ip}:{port}/mobile'
    })

@app.route('/api/mobile/auth', methods=['POST'])
def mobile_auth():
    try:
        data = request.get_json() or {}
        password = data.get('password', '').strip()

        if not password:
            logger.warning(f"Mobile auth failed: missing password from {request.remote_addr}")
            return create_error_response('MISSING_PASSWORD', 'Password required', status_code=400)

        # Validate password (constant-time comparison)
        from config import validate_mobile_password
        if not validate_mobile_password(password):
            logger.warning(f"Mobile auth failed: invalid password from {request.remote_addr}")
            return create_error_response('INVALID_PASSWORD', 'Incorrect password', status_code=401)

        # Get available catalogs efficiently
        from snapshot_manager import list_snapshots, get_loaded_snapshot_info

        try:
            catalogs_dict = list_snapshots()
            loaded_info = get_loaded_snapshot_info()
            loaded_file = loaded_info.get('snapshot_file', '')

            # Get historical catalogs (most common use case for mobile)
            # Combine both historical and new catalogs for selection
            historical_catalogs = catalogs_dict.get('historical', [])
            new_catalogs = catalogs_dict.get('new', [])
            all_catalogs = historical_catalogs + new_catalogs

            # Limit response to first 50 catalogs to prevent memory issues
            catalogs_response = []
            loaded_catalogs = []
            from snapshot_manager import get_snapshot_connection

            for catalog in all_catalogs[:50]:
                catalog_file = catalog.get('snapshot_file', '')
                is_loaded = catalog_file == loaded_file

                # Check catalog compatibility (has metadata_schema table)
                is_compatible = True
                try:
                    catalog_path = os.path.join(BACKEND_DIR, 'catalogs', catalog_file)
                    if os.path.exists(catalog_path):
                        with get_snapshot_connection(catalog_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute('''
                                SELECT name FROM sqlite_master
                                WHERE type='table' AND name='metadata_schema'
                            ''')
                            is_compatible = cursor.fetchone() is not None
                except Exception as e:
                    logger.debug(f"Could not check compatibility for {catalog_file}: {e}")
                    is_compatible = False

                catalog_info = {
                    'id': catalog_file,
                    'name': catalog.get('name', 'Unnamed'),
                    'product_count': catalog.get('product_count', 0),
                    'is_loaded': is_loaded,
                    'is_compatible': is_compatible
                }

                # Add warning suffix for incompatible catalogs
                if not is_compatible:
                    catalog_info['name'] += ' ⚠️ (Incompatible - Old Version)'

                catalogs_response.append(catalog_info)

                if is_loaded:
                    loaded_catalogs.append(catalog.get('name', 'Unnamed'))

            logger.info(f"📱 Mobile auth successful from {request.remote_addr}")
            logger.info(f"📱 Currently loaded file: '{loaded_file}'")
            logger.info(f"📱 Returned {len(catalogs_response)} catalogs, {len(loaded_catalogs)} marked as loaded")
            if loaded_catalogs:
                logger.info(f"📱 Loaded catalogs: {loaded_catalogs}")
                # Debug: Show catalog IDs
                logger.info(f"📱 Catalog IDs: {[c['id'] for c in catalogs_response]}")

            return jsonify({
                'valid': True,
                'catalogs': catalogs_response,
                'modes': ['mode1', 'mode3']
            }), 200

        except Exception as e:
            logger.error(f"Error getting catalogs for mobile: {e}", exc_info=True)
            return create_error_response('CATALOG_ERROR', 'Failed to load catalogs', status_code=500)

    except Exception as e:
        logger.error(f"Mobile auth error: {e}", exc_info=True)
        return create_error_response('AUTH_ERROR', 'Authentication failed', status_code=500)

@app.route('/api/mobile/config', methods=['GET'])
def mobile_config():
    try:
        password = request.headers.get('X-Mobile-Password', '').strip()

        if not password:
            logger.warning(f"Mobile config failed: missing password from {request.remote_addr}")
            return create_error_response('MISSING_AUTH', 'Password required', status_code=401)

        from config import validate_mobile_password
        if not validate_mobile_password(password):
            logger.warning(f"Mobile config failed: invalid password from {request.remote_addr}")
            return create_error_response('UNAUTHORIZED', 'Invalid password', status_code=401)

        from snapshot_manager import get_loaded_snapshot_info

        try:
            loaded_info = get_loaded_snapshot_info()

            return jsonify({
                'authorized': True,
                'loaded_catalog': {
                    'name': loaded_info.get('name', 'Unknown'),
                    'file': loaded_info.get('snapshot_file', ''),
                    'product_count': loaded_info.get('product_count', 0)
                }
            }), 200

        except Exception as e:
            logger.error(f"Error getting loaded catalog for mobile: {e}", exc_info=True)
            return create_error_response('CONFIG_ERROR', 'Failed to load config', status_code=500)

    except Exception as e:
        logger.error(f"Mobile config error: {e}", exc_info=True)
        return create_error_response('CONFIG_ERROR', 'Config request failed', status_code=500)

@app.route('/api/mobile/catalog-schema', methods=['GET'])
def get_catalog_schema():
    
    try:
        password = request.headers.get('X-Mobile-Password', '').strip()

        if not password:
            logger.warning(f"Mobile catalog-schema failed: missing password from {request.remote_addr}")
            return create_error_response('MISSING_AUTH', 'Password required', status_code=401)

        from config import validate_mobile_password
        if not validate_mobile_password(password):
            logger.warning(f"Mobile catalog-schema failed: invalid password from {request.remote_addr}")
            return create_error_response('UNAUTHORIZED', 'Invalid password', status_code=401)

        catalog_id = request.args.get('catalog_id', '').strip()
        if not catalog_id:
            logger.warning(f"Mobile catalog-schema: missing catalog_id from {request.remote_addr}")
            return create_error_response('MISSING_CATALOG', 'catalog_id required', status_code=400)

        # Validate catalog_id is safe (prevent path traversal)
        if '..' in catalog_id or '/' in catalog_id or '\\' in catalog_id:
            logger.warning(f"Mobile catalog-schema: suspicious catalog_id from {request.remote_addr}")
            return create_error_response('INVALID_CATALOG', 'Invalid catalog_id', status_code=400)

        from snapshot_manager import get_snapshot_connection
        catalog_path = os.path.join(BACKEND_DIR, 'catalogs', catalog_id)

        if not os.path.exists(catalog_path):
            logger.warning(f"Mobile catalog-schema: catalog not found: {catalog_id}")
            return create_error_response('CATALOG_NOT_FOUND', 'Catalog not found', status_code=404)

        try:
            with get_snapshot_connection(catalog_path) as conn:
                cursor = conn.cursor()

                # Check if metadata_schema table exists (old catalogs might not have it)
                cursor.execute('''
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='metadata_schema'
                ''')
                has_metadata_schema = cursor.fetchone() is not None

                metadata_fields = []

                if has_metadata_schema:
                    # Get dynamic metadata fields from metadata_schema
                    # These apply to HISTORICAL products (is_historical=1)
                    cursor.execute('''
                        SELECT column_name, data_type, display_name
                        FROM metadata_schema
                        WHERE is_active = 1
                        ORDER BY rowid
                    ''')

                    rows = cursor.fetchall()
                    for row in rows:
                        metadata_fields.append({
                            'column_name': row[0],
                            'data_type': row[1],
                            'display_name': row[2] or row[0]
                        })
                else:
                    logger.warning(f"Mobile catalog-schema: ⚠️ Catalog {catalog_id} is missing metadata_schema table (old/incompatible version)")

            if len(metadata_fields) == 0:
                logger.info(f"Mobile catalog-schema: ⚠️ No metadata fields found for {catalog_id}. Only base fields (category, product_name, sku) will be available.")
            else:
                logger.info(f"Mobile catalog-schema: ✓ Returned {len(metadata_fields)} metadata fields for {catalog_id}: {[f['column_name'] for f in metadata_fields]}")

            return jsonify({
                'base_fields': [
                    {
                        'column_name': 'category',
                        'data_type': 'string',
                        'display_name': 'Category'
                    }
                ],
                'metadata_fields': metadata_fields,
                'catalog_id': catalog_id,
                'note': 'All fields are optional - leave blank if not applicable'
            }), 200

        except Exception as e:
            logger.error(f"Error getting catalog schema for {catalog_id}: {e}", exc_info=True)
            return create_error_response('SCHEMA_ERROR', 'Failed to load catalog schema', status_code=500)

    except Exception as e:
        logger.error(f"Mobile catalog-schema error: {e}", exc_info=True)
        return create_error_response('SCHEMA_ERROR', 'Schema request failed', status_code=500)

@app.route('/api/mobile/catalog-categories/clear-cache', methods=['POST'])
def clear_catalog_categories_cache():
    
    try:
        password = request.headers.get('X-Mobile-Password', '').strip()

        if not password:
            return create_error_response('MISSING_AUTH', 'Password required', status_code=401)

        from config import validate_mobile_password
        if not validate_mobile_password(password):
            return create_error_response('UNAUTHORIZED', 'Invalid password', status_code=401)

        catalog_id = request.args.get('catalog_id', None)
        invalidate_catalog_categories_cache(catalog_id)

        message = f'Cleared category cache for {catalog_id}' if catalog_id else 'Cleared all category caches'
        logger.info(f"[CACHE] {message}")

        return jsonify({
            'status': 'success',
            'message': message
        }), 200

    except Exception as e:
        logger.error(f"Error clearing category cache: {e}", exc_info=True)
        return create_error_response('CACHE_ERROR', 'Failed to clear cache', status_code=500)


@app.route('/api/mobile/catalog-categories', methods=['GET'])
def get_catalog_categories():
    
    try:
        password = request.headers.get('X-Mobile-Password', '').strip()

        if not password:
            logger.warning(f"Mobile catalog-categories failed: missing password from {request.remote_addr}")
            return create_error_response('MISSING_AUTH', 'Password required', status_code=401)

        from config import validate_mobile_password
        if not validate_mobile_password(password):
            logger.warning(f"Mobile catalog-categories failed: invalid password from {request.remote_addr}")
            return create_error_response('UNAUTHORIZED', 'Invalid password', status_code=401)

        catalog_id = request.args.get('catalog_id', '').strip()
        if not catalog_id:
            logger.warning(f"Mobile catalog-categories: missing catalog_id from {request.remote_addr}")
            return create_error_response('MISSING_CATALOG', 'catalog_id required', status_code=400)

        # Validate catalog_id is safe (prevent path traversal)
        if '..' in catalog_id or '/' in catalog_id or '\\' in catalog_id:
            logger.warning(f"Mobile catalog-categories: suspicious catalog_id from {request.remote_addr}")
            return create_error_response('INVALID_CATALOG', 'Invalid catalog_id', status_code=400)

        from snapshot_manager import get_snapshot_connection
        catalog_path = os.path.join(BACKEND_DIR, 'catalogs', catalog_id)

        if not os.path.exists(catalog_path):
            logger.warning(f"Mobile catalog-categories: catalog not found: {catalog_id}")
            return create_error_response('CATALOG_NOT_FOUND', 'Catalog not found', status_code=404)

        # PERFORMANCE: Check cache first (avoids DISTINCT query on huge catalogs)
        cached_categories = get_cached_catalog_categories(catalog_id)
        if cached_categories is not None:
            logger.debug(f"Mobile catalog-categories: cache hit for {catalog_id} ({len(cached_categories)} categories)")
            return jsonify({
                'categories': cached_categories,
                'catalog_id': catalog_id,
                'cached': True
            }), 200

        try:
            with get_snapshot_connection(catalog_path) as conn:
                cursor = conn.cursor()

                # Ensure index exists for performance (idempotent - safe to run multiple times)
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_products_category
                    ON products(category)
                ''')

                # Get unique categories from products
                # OPTIMIZED: Index on category column makes DISTINCT fast even for huge catalogs
                cursor.execute('''
                    SELECT DISTINCT category
                    FROM products
                    WHERE category IS NOT NULL AND category != ''
                    ORDER BY category
                ''')

                categories = [row[0] for row in cursor.fetchall()]

            # Cache the result for future requests
            cache_catalog_categories(catalog_id, categories)

            logger.debug(f"Mobile catalog-categories: returned {len(categories)} categories for {catalog_id}")

            return jsonify({
                'categories': categories,
                'catalog_id': catalog_id,
                'cached': False
            }), 200

        except Exception as e:
            logger.error(f"Error getting categories for {catalog_id}: {e}", exc_info=True)
            return create_error_response('CATEGORIES_ERROR', 'Failed to load categories', status_code=500)

    except Exception as e:
        logger.error(f"Mobile catalog-categories error: {e}", exc_info=True)
        return create_error_response('CATEGORIES_ERROR', 'Categories request failed', status_code=500)

@app.route('/api/mobile/log', methods=['POST'])
def mobile_log():
    """Accept log messages from mobile frontend for server-side logging

    Request JSON:
        {
            "level": "info" | "warning" | "error",
            "message": "Log message",
            "data": {} (optional)
        }
    """
    try:
        password = request.headers.get('X-Mobile-Password', '').strip()

        # Allow logging without auth for debugging
        data = request.get_json()
        level = data.get('level', 'info').lower()
        message = data.get('message', '')
        extra_data = data.get('data', {})

        log_message = f"📱 [MOBILE] {message}"
        if extra_data:
            log_message += f" | Data: {extra_data}"

        if level == 'error':
            logger.error(log_message)
        elif level == 'warning':
            logger.warning(log_message)
        else:
            logger.info(log_message)

        return jsonify({'logged': True}), 200
    except Exception as e:
        logger.error(f"Mobile log error: {e}")
        return jsonify({'logged': False}), 500

@app.route('/api/mobile/password', methods=['GET', 'POST'])
def mobile_password_management():
    
    try:
        from config import get_mobile_password, save_mobile_password

        if request.method == 'GET':
            try:
                password = get_mobile_password()
                return jsonify({'password': password}), 200
            except Exception as e:
                logger.error(f"Error getting mobile password: {e}")
                return create_error_response('PASSWORD_ERROR', 'Failed to get password', status_code=500)

        # POST: Update password
        try:
            data = request.get_json() or {}
            new_password = data.get('new_password', '').strip()

            # Validate password format
            if not new_password:
                logger.warning(f"Mobile password update: empty password from {request.remote_addr}")
                return create_error_response('INVALID_PASSWORD', 'Password required', status_code=400)

            if len(new_password) != 6:
                logger.warning(f"Mobile password update: wrong length ({len(new_password)}) from {request.remote_addr}")
                return create_error_response('INVALID_PASSWORD', 'Must be exactly 6 characters', status_code=400)

            if not new_password.isdigit():
                logger.warning(f"Mobile password update: non-digit password from {request.remote_addr}")
                return create_error_response('INVALID_PASSWORD', 'Must contain only digits (0-9)', status_code=400)

            save_mobile_password(new_password)
            logger.info(f"Mobile password updated from {request.remote_addr}")

            return jsonify({
                'success': True,
                'password': new_password,
                'message': 'Mobile password updated'
            }), 200

        except Exception as e:
            logger.error(f"Error saving mobile password: {e}", exc_info=True)
            return create_error_response('PASSWORD_ERROR', 'Failed to save password', status_code=500)

    except Exception as e:
        logger.error(f"Mobile password management error: {e}", exc_info=True)
        return create_error_response('PASSWORD_ERROR', 'Password request failed', status_code=500)

@app.route('/api/mobile/upload-and-match', methods=['POST'])
def mobile_upload_and_match():
    
    try:
        # Validate mobile password
        password = request.headers.get('X-Mobile-Password', '').strip()

        if not password:
            logger.warning(f"[MOBILE] Auth failed: missing password from {request.remote_addr}")
            return create_error_response('MISSING_AUTH', 'Password required', status_code=401)

        from config import validate_mobile_password
        if not validate_mobile_password(password):
            logger.warning(f"[MOBILE] Auth failed: invalid password from {request.remote_addr}")
            return create_error_response('UNAUTHORIZED', 'Invalid password', status_code=401)

        # Validate required form data
        if 'image' not in request.files:
            return create_error_response('MISSING_IMAGE', 'Image file required', status_code=400)

        catalog_id = request.form.get('catalog_id', '').strip()
        if not catalog_id:
            return create_error_response('MISSING_CATALOG', 'catalog_id required', status_code=400)

        mode = request.form.get('mode', '').strip().lower()
        if mode not in ['mode1', 'mode3']:
            return create_error_response('INVALID_MODE', 'mode must be "mode1" or "mode3"', status_code=400)

        # Validate catalog_id is safe
        if '..' in catalog_id or '/' in catalog_id or '\\' in catalog_id:
            logger.warning(f"[MOBILE] Suspicious catalog_id from {request.remote_addr}")
            return create_error_response('INVALID_CATALOG', 'Invalid catalog_id', status_code=400)

        from snapshot_manager import get_snapshot_connection
        catalog_path = os.path.join(BACKEND_DIR, 'catalogs', catalog_id)

        if not os.path.exists(catalog_path):
            logger.warning(f"[MOBILE] Catalog not found: {catalog_id}")
            return create_error_response('CATALOG_NOT_FOUND', 'Catalog not found', status_code=404)

        try:
            import json
            import uuid

            # STEP 1: Load catalog (use_existing mode - same as desktop)
            logger.info(f"[MOBILE] Step 1: Loading catalog {catalog_id}")
            from snapshot_manager import load_snapshot_to_main_db

            load_result = load_snapshot_to_main_db(catalog_id)
            if load_result.get('error'):
                logger.error(f"[MOBILE] Failed to load catalog: {load_result['error']}")
                return create_error_response('LOAD_ERROR', 'Failed to load catalog', status_code=500)

            logger.info(f"[MOBILE] Catalog loaded: {load_result.get('product_count')} products")

            # STEP 2: Load historical products (skip batch-upload, use_existing mode)
            logger.info(f"[MOBILE] Step 2: Loading historical products (use_existing)")
            # In use_existing mode, backend just loads existing products from DB, no upload
            # Frontend state will be updated when batch-match response comes back

            # STEP 3: Clear new section (replace mode) - silent operation
            logger.info(f"[MOBILE] Step 3: Clearing new section (replace mode)")
            try:
                from database import get_db_connection

                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    # Delete all new products (is_historical = 0)
                    cursor.execute('DELETE FROM products WHERE is_historical = 0')
                    deleted_count = cursor.rowcount
                    conn.commit()
                    logger.info(f"[MOBILE] Cleared {deleted_count} new products")
            except Exception as e:
                logger.warning(f"[MOBILE] Cleanup error (non-fatal): {e}")
                # Continue anyway - products may be empty already

            # STEP 4: Batch upload new product
            logger.info(f"[MOBILE] Step 4: Uploading new product via batch-upload")

            file = request.files['image']
            if file.filename == '':
                return create_error_response('EMPTY_FILENAME', 'No file selected', status_code=400)

            if not allowed_file(file.filename):
                return create_error_response('INVALID_FORMAT', 'Unsupported file format', status_code=400)

            # Get product fields
            category = request.form.get('category', None)
            product_name = request.form.get('product_name', None)
            sku = request.form.get('sku', None)

            # Normalize empty strings
            if category and category.strip() == '':
                category = None
            if product_name and product_name.strip() == '':
                product_name = None
            if sku and sku.strip() == '':
                sku = None

            # Normalize category (lowercase, trim whitespace, handle "unknown" variations)
            if category is not None:
                from product_matching import normalize_category
                category = normalize_category(category)

            # Save image file
            file_ext = secure_filename(file.filename).rsplit('.', 1)[1].lower() if '.' in secure_filename(file.filename) else 'jpg'
            unique_filename = f"{uuid.uuid4()}.{file_ext}"
            uploads_dir = os.path.join(BACKEND_DIR, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            filepath = os.path.join(uploads_dir, unique_filename)

            try:
                file.save(filepath)
                logger.info(f"[MOBILE] Image saved: {filepath}")
            except Exception as e:
                logger.error(f"[MOBILE] Failed to save image: {e}")
                return create_error_response('SAVE_ERROR', 'Failed to save image', status_code=500)

            # Call batch-upload backend functions directly (same as desktop batch-upload)
            try:
                from database import insert_product, insert_features
                from feature_extraction_service import extract_features_unified

                # Handle mode 3 metadata if provided
                metadata = None
                if mode == 'mode3':
                    # Collect all metadata_<column_name> fields from request form
                    metadata_dict = {}
                    for key in request.form.keys():
                        if key.startswith('metadata_'):
                            column_name = key.replace('metadata_', '')
                            value = request.form.get(key)
                            if value and str(value).strip() != '':
                                metadata_dict[column_name] = value

                    if metadata_dict:
                        metadata = json.dumps(metadata_dict)
                        logger.info(f"[MOBILE] Mode 3 metadata collected: {list(metadata_dict.keys())}")

                product_id = insert_product(
                    image_path=filepath,
                    category=category,
                    product_name=product_name,
                    sku=sku,
                    is_historical=False,
                    metadata=metadata
                )
                logger.info(f"[MOBILE] Product inserted: ID {product_id}")

                # Extract features
                try:
                    features, embedding_type, embedding_version = extract_features_unified(filepath)
                    insert_features(
                        product_id=product_id,
                        color_features=features['color_features'],
                        shape_features=features['shape_features'],
                        texture_features=features['texture_features'],
                        embedding_type=embedding_type,
                        embedding_version=embedding_version
                    )
                    logger.info(f"[MOBILE] Features extracted for product {product_id}")
                except Exception as e:
                    logger.warning(f"[MOBILE] Feature extraction failed (non-fatal): {e}")

            except Exception as e:
                try:
                    os.remove(filepath)
                except:
                    pass
                logger.error(f"[MOBILE] Upload failed: {e}")
                return create_error_response('UPLOAD_ERROR', 'Failed to upload product', status_code=500)

            # STEP 5: Batch match (same as desktop REPLACE & PROCESS)
            logger.info(f"[MOBILE] Step 5: Matching product {product_id} (mode {mode})")

            try:
                from product_matching import batch_find_matches
                from hybrid_matching import batch_find_hybrid_matches
                from database import get_metadata_schema

                matches = []

                if mode == 'mode1':
                    # Mode 1: Visual only - use batch version (single product wrapped in list)
                    batch_result = batch_find_matches(
                        product_ids=[product_id],
                        threshold=0,
                        limit=5,
                        match_against_all=False,
                        include_uncategorized=True,
                        store_matches=True,
                        skip_invalid_products=True,
                        preload_catalog=False
                    )
                    # Extract matches from batch result format
                    if batch_result['results'] and batch_result['results'][0]['status'] == 'success':
                        matches = batch_result['results'][0]['matches']
                    logger.info(f"[MOBILE] Mode 1 visual matching: {len(matches)} results")
                elif mode == 'mode3':
                    # Mode 3: Hybrid matching (visual + metadata combined)
                    # Uses metadata schema created during CSV upload (part 1)
                    schema = get_metadata_schema()

                    if schema:
                        # Build weights from schema (equal weight for all columns)
                        metadata_weights = {col['column_name']: 1.0 for col in schema}
                        total = sum(metadata_weights.values())
                        metadata_weights = {k: v / total for k, v in metadata_weights.items()}

                        # Call batch hybrid matching (50% visual, 50% metadata)
                        batch_result = batch_find_hybrid_matches(
                            product_ids=[product_id],
                            threshold=0,
                            limit=5,
                            visual_weight=0.5,
                            metadata_weight=0.5,
                            metadata_weights=metadata_weights,
                            store_matches=True,
                            skip_invalid_products=True,
                            match_against_all=False
                        )
                        # Extract matches from batch result format
                        if batch_result['results'] and batch_result['results'][0]['status'] == 'success':
                            matches = batch_result['results'][0]['matches']
                        logger.info(f"[MOBILE] Mode 3 hybrid matching: {len(matches)} results")
                    else:
                        # No metadata schema - fall back to visual (batch version)
                        logger.info("[MOBILE] No metadata schema found - falling back to visual matching")
                        batch_result = batch_find_matches(
                            product_ids=[product_id],
                            threshold=0,
                            limit=5,
                            match_against_all=False,
                            include_uncategorized=True,
                            store_matches=True,
                            skip_invalid_products=True,
                            preload_catalog=False
                        )
                        # Extract matches from batch result format
                        if batch_result['results'] and batch_result['results'][0]['status'] == 'success':
                            matches = batch_result['results'][0]['matches']

            except Exception as e:
                logger.warning(f"[MOBILE] Matching error: {e}")
                matches = []

            # Format matches
            matches_response = []
            for match in matches[:5]:
                matches_response.append({
                    'id': match.get('product_id') or match.get('id'),
                    'name': match.get('product_name') or match.get('name') or 'Unknown',
                    'category': match.get('category') or 'N/A',
                    'score': match.get('score', 0),
                    'sku': match.get('sku')
                })

            logger.info(f"[MOBILE] Complete: Product {product_id}, {len(matches_response)} matches")

            # Invalidate CSV cache since catalog was loaded and products modified
            invalidate_csv_cache()

            # Return full response with all data for frontend to update state
            return jsonify({
                'success': True,
                'product_id': product_id,
                'mode': mode,
                'catalog_id': catalog_id,
                'upload_status': 'success',
                'matches': matches_response,
                'match_count': len(matches_response)
            }), 200

        except Exception as e:
            logger.error(f"[MOBILE] Orchestration error: {e}", exc_info=True)
            return create_error_response('ORCHESTRATION_ERROR', 'Processing failed', status_code=500)

    except Exception as e:
        logger.error(f"[MOBILE] Request error: {e}", exc_info=True)
        return create_error_response('MOBILE_ERROR', 'Request failed', status_code=500)

# Simple flag to notify main app that mobile results are ready
_mobile_results_flag = {'ready': False, 'timestamp': None}

@app.route('/api/mobile/results-ready', methods=['POST'])
def mobile_results_ready():
    """Mobile notifies main app that results are ready

    Called by mobile-upload after successful match completion.
    Sets a flag that main app polls to know when to fetch results.

    No auth required (runs on same backend).
    """
    global _mobile_results_flag

    try:
        import time
        _mobile_results_flag['ready'] = True
        _mobile_results_flag['timestamp'] = time.time()

        logger.info("[MOBILE] Results ready flag set - notifying main app")

        return jsonify({
            'success': True,
            'message': 'Main app notified'
        }), 200
    except Exception as e:
        logger.error(f"[MOBILE] Failed to set results flag: {e}")
        return create_error_response('FLAG_ERROR', 'Failed to set results flag', status_code=500)

@app.route('/api/mobile/check-flag', methods=['GET'])
def check_mobile_results_flag():
    """Main app checks if mobile has results ready

    Returns the current flag state.
    """
    global _mobile_results_flag

    try:
        flag_ready = _mobile_results_flag['ready']
        if flag_ready:
            logger.info(f"[MOBILE] Check flag: ready={flag_ready} (will trigger results polling)")
        return jsonify({
            'ready': flag_ready,
            'timestamp': _mobile_results_flag['timestamp']
        }), 200
    except Exception as e:
        logger.error(f"[MOBILE] Failed to check flag: {e}")
        return create_error_response('CHECK_ERROR', 'Failed to check flag', status_code=500)

@app.route('/api/mobile/clear-flag', methods=['POST'])
def clear_mobile_results_flag():
    """Main app clears the flag after displaying results

    Resets flag so mobile can set it again for next upload.
    """
    global _mobile_results_flag

    try:
        _mobile_results_flag['ready'] = False
        _mobile_results_flag['timestamp'] = None

        logger.info("[MOBILE] Results flag cleared")

        return jsonify({
            'success': True,
            'message': 'Flag cleared'
        }), 200
    except Exception as e:
        logger.error(f"[MOBILE] Failed to clear flag: {e}")
        return create_error_response('CLEAR_ERROR', 'Failed to clear flag', status_code=500)

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
            logger.info(f"Saved metadata schema with {len(columns)} columns")
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
            logger.info("Cleared metadata schema")
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
            except:
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
            except:
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
            except:
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
                logger.info(f"[BATCH-UPLOAD] METHOD: NEW (File Paths) - Processing {len(file_paths)} images")
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
            logger.info(f"[BATCH-UPLOAD] METHOD: LEGACY (Direct Upload) - Processing {len(files) if files else 0} files")
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
        logger.info(f"[BATCH-UPLOAD] Processing {file_count} images ({('file paths' if file_paths_json else 'uploaded files')})")

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
            logger.info(f"[BATCH-UPLOAD] Categories received: {len(unique_categories)} unique categories from {len(categories)} products")
            if unique_categories:
                logger.info(f"[BATCH-UPLOAD] Unique categories: {sorted(unique_categories)}")
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
            logger.info("[BATCH-UPLOAD] Step 1: Validating file paths (MEMORY OPTIMIZED - no uploads)")

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
            logger.info(f"[BATCH-UPLOAD] Step 1: Validated {len(saved_files)} files (skipped {len(skipped_files)})")
        else:
            # LEGACY: Files uploaded, need to save them and validate
            logger.info("[BATCH-UPLOAD] Step 1: Saving and validating uploaded files")

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
                            except:
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
                logger.info(f"[BATCH-UPLOAD] Attempt 1: Skipped {len(skipped_files)} files, processing {len(saved_files)} valid files")

                # Retry skipped files once
                logger.info(f"[BATCH-UPLOAD] Retrying {len(skipped_files)} skipped files (Attempt 2)")
                retry_files = [files[s['index']] for s in skipped_files]
                retry_indices = [s['index'] for s in skipped_files]

                retry_saved, retry_indices_result, retry_skipped = process_files_batch(retry_files, retry_indices, attempt=2)

                # Merge retry results
                saved_files.extend(retry_saved)
                file_indices.extend(retry_indices_result)

                # Update skipped list with files that failed retry
                skipped_files = retry_skipped

                if len(retry_saved) > 0:
                    logger.info(f"[BATCH-UPLOAD] Retry successful: {len(retry_saved)} files recovered, {len(retry_skipped)} still skipped")
                else:
                    logger.info(f"[BATCH-UPLOAD] Retry failed: All {len(retry_skipped)} files still invalid")

        if len(saved_files) == 0:
            return create_error_response(
                'NO_VALID_FILES',
                'No valid image files found in batch',
                'All files were invalid or skipped',
                status_code=400
            )

        logger.info(f"[BATCH-UPLOAD] {len(saved_files)} files saved and validated")

        # Step 2: Insert products into database (THREAD-SAFE: Bulk insert in single transaction)
        logger.info("[BATCH-UPLOAD] Step 2: Inserting products into database (bulk insert)")

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

        # Bulk insert all products in single transaction (THREAD-SAFE, FAST)
        try:
            product_ids = bulk_insert_products(products_to_insert)
            inserted_count = len(product_ids)
            logger.info(f"[BATCH-UPLOAD] ✓ Bulk inserted {inserted_count}/{len(saved_files)} products (single transaction)")
        except Exception as e:
            logger.error(f"[BATCH-UPLOAD] Bulk insert failed: {e}")
            # Fallback: insert one by one (slower but more resilient)
            logger.info("[BATCH-UPLOAD] Falling back to sequential insert...")
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
            logger.info(f"[BATCH-UPLOAD] Fallback complete: {inserted_count}/{len(saved_files)} products inserted")
        
        # Step 3: Extract features in batch (GPU-optimized parallel processing)
        logger.info("[BATCH-UPLOAD] Step 3: Extracting features in batch (GPU-optimized)")
        
        from feature_extraction_service import batch_extract_features_unified
        
        # Only extract features for successfully inserted products
        valid_indices = [i for i, pid in enumerate(product_ids) if pid is not None]
        valid_filepaths = [saved_files[i] for i in valid_indices]
        
        if valid_filepaths:
            feature_results = batch_extract_features_unified(valid_filepaths)
            
            # Step 4: Store features in database - INCREMENTAL BATCH INSERT
            logger.info("[BATCH-UPLOAD] Step 4: Storing features in database (incremental batch insert)")
            
            from database import serialize_numpy_array, bulk_insert_features
            
            # Collect features for batch insert (incremental to avoid memory bloat)
            features_to_insert = []
            INCREMENTAL_BATCH_SIZE = 32  # Insert every 32 features (matches GPU batch size)
            total_inserted = 0
            
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
                    logger.warning(f"[BATCH-UPLOAD] Feature extraction failed for product {product_id}: {error_msg}")
            
            # Batch insert remaining features
            if features_to_insert:
                try:
                    inserted_count = bulk_insert_features(features_to_insert)
                    total_inserted += inserted_count
                    logger.info(f"[BATCH-UPLOAD] ✓ Final batch inserted {inserted_count} remaining feature records (total: {total_inserted})")
                except Exception as e:
                    logger.error(f"[BATCH-UPLOAD] Failed to batch insert remaining features: {e}")
        
        # Step 5: Rebuild FAISS indexes in background (don't block response)
        # Always rebuild FAISS indexes when new products are added (both historical and new)
        logger.info("[BATCH-UPLOAD] Step 5: Scheduling FAISS index rebuild (background)")
        
        def rebuild_indexes_background():
            """Rebuild FAISS indexes in background thread"""
            try:
                logger.info("[BATCH-UPLOAD-BG] Starting background FAISS index rebuild...")
                from database import rebuild_all_faiss_indexes
                rebuild_all_faiss_indexes()
                logger.info("[BATCH-UPLOAD-BG] ✓ FAISS indexes rebuilt successfully")
            except Exception as e:
                logger.warning(f"[BATCH-UPLOAD-BG] Failed to rebuild FAISS indexes: {e}")
        
        # Start background thread (don't wait for it)
        import threading
        bg_thread = threading.Thread(target=rebuild_indexes_background, daemon=True)
        bg_thread.start()
        logger.info("[BATCH-UPLOAD] FAISS index rebuild scheduled in background")
        
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
            'results': results
        }
        
        logger.info(f"[BATCH-UPLOAD] Returning JSON response: status={response_data['status']}, total={response_data['total']}, successful={response_data['successful']}, failed={response_data['failed']}, skipped={response_data['skipped']}, results_count={len(response_data['results'])}")

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
        except:
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
            
            # Prepare response
            response = {
                'status': 'success',
                'batch_size': len(product_ids),
                'results': result['results'],
                'summary': result['summary'],
                'errors': result.get('errors', [])
            }
            
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
        
        # Invalidate all FAISS indexes to free memory
        invalidate_faiss_index(category=None)  # None = invalidate all categories
        
        logger.info(f"[SESSION-CLEANUP] Deleted {deleted_count} matches on app close")
        logger.info(f"[SESSION-CLEANUP] Invalidated all FAISS indexes")
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up {deleted_count} matches and cleared indexes',
            'matches_deleted': deleted_count
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
        from database import get_db_connection, get_product_by_id

        logger.info("[MATCH-RESULTS] Fetching stored match results for NEW section")

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get all products in NEW section that have stored matches
            cursor.execute('''
                SELECT DISTINCT p.id, p.image_path, p.category, p.product_name, p.sku, p.metadata
                FROM products p
                WHERE p.is_historical = 0
                AND EXISTS (
                    SELECT 1 FROM matches
                    WHERE new_product_id = p.id
                )
                ORDER BY p.id DESC
            ''')

            products = cursor.fetchall()
            results = []

            for product_row in products:
                product_id = product_row['id']

                # Get stored matches for this product
                # Only select base columns that exist in all snapshot versions
                cursor.execute('''
                    SELECT matched_product_id, similarity_score
                    FROM matches
                    WHERE new_product_id = ?
                    ORDER BY similarity_score DESC
                    LIMIT 5
                ''', (product_id,))

                matches = cursor.fetchall()

                if matches:
                    # Build result in same format as batch-match
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
                    for match in matches:
                        matched_product = get_product_by_id(match['matched_product_id'])
                        if matched_product:
                            # Build match data matching batch-match format
                            # Note: matched_product is a sqlite3.Row, use bracket notation
                            match_data = {
                                'product_id': match['matched_product_id'],
                                'product_name': matched_product['product_name'] or 'Unknown',
                                'name': matched_product['product_name'] or 'Unknown',
                                'category': matched_product['category'],
                                'sku': matched_product['sku'],
                                'similarity_score': match['similarity_score'],  # Frontend expects this field
                                'image_path': matched_product['image_path']
                            }

                            match_list.append(match_data)

                    if match_list:
                        results.append({
                            'product_id': product_id,
                            'product_data': product_data,
                            'matches': match_list,
                            'status': 'success'
                        })

            logger.info(f"[MATCH-RESULTS] Found {len(results)} products with stored matches")

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
        
        # Limit max results to prevent abuse
        limit = min(limit, 1000)
        
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
                product_ids = [int(id.strip()) for id in ids_param.split(',') if id.strip()]
                logger.info(f"[GET-PRODUCTS] Batch fetch mode: {len(product_ids)} IDs")

                products = []
                for product_id in product_ids:
                    product = get_product_by_id(product_id)
                    if product:
                        # Convert sqlite3.Row to dict so we can add has_features flag
                        product_dict = dict(product)
                        # Add has_features flag
                        features = get_features_by_product_id(product_id)
                        product_dict['has_features'] = features is not None
                        
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
                    'pages': 1
                }), 200

            except ValueError as e:
                return create_error_response(
                    'INVALID_IDS',
                    'Invalid product IDs format',
                    'IDs must be comma-separated integers',
                    status_code=400
                )

        # Normal pagination mode
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 50, type=int)
        search = request.args.get('search', '')
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
            is_managed_file = os.path.abspath(image_path).startswith(os.path.abspath(uploads_folder))

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
        from flask import Response
        
        csv_content = export_catalog_csv()
        
        return Response(
            csv_content,
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

@app.route('/api/catalogs/list', methods=['GET'])
def list_catalog_snapshots():
    """
    List all available catalog snapshots.
    
    Returns:
    - 200: Success with historical and new snapshot lists
    - 500: Server error
    """
    try:
        from snapshot_manager import list_snapshots, migrate_legacy_database
        
        # Check for migration on first access
        migrate_legacy_database()
        
        result = list_snapshots()
        
        if result.get('error'):
            return create_error_response(
                'LIST_ERROR',
                result['error'],
                status_code=500
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing snapshots: {e}", exc_info=True)
        return create_error_response(
            'LIST_ERROR',
            'Failed to list snapshots',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/storage', methods=['GET'])
def get_snapshot_storage_info():
    """
    Get total disk space used by all snapshots.

    Returns:
    - 200: Success with storage breakdown
    - 500: Server error
    """
    try:
        from snapshot_manager import get_total_snapshot_storage

        result = get_total_snapshot_storage()

        return jsonify({
            'status': 'success',
            **result
        }), 200

    except Exception as e:
        logger.error(f"Error getting snapshot storage info: {e}", exc_info=True)
        return create_error_response(
            'STORAGE_ERROR',
            'Failed to get storage information',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/check-duplicate/<section>', methods=['GET'])
def check_catalog_duplicate(section):
    """
    Check if a duplicate snapshot already exists for a catalog section.

    Args:
        section: 'historical' or 'new'

    Returns:
    - 200: Success with duplicate check results
    - 400: Invalid section
    - 500: Server error
    """
    try:
        if section not in ['historical', 'new']:
            return create_error_response(
                'INVALID_SECTION',
                f"Invalid section: {section}. Must be 'historical' or 'new'",
                status_code=400
            )

        from snapshot_manager import check_snapshot_duplicate

        result = check_snapshot_duplicate(section)

        if result.get('is_duplicate'):
            logger.info(f"Duplicate detected for {section} catalog: {result['existing_snapshot']}")

        return jsonify({
            'status': 'success',
            **result
        }), 200

    except Exception as e:
        logger.error(f"Error checking for duplicate snapshot: {e}", exc_info=True)
        return create_error_response(
            'DUPLICATE_CHECK_ERROR',
            'Failed to check for duplicate',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/save-with-dialog', methods=['POST'])
def save_catalog_with_dialog_choice():
    """
    Save catalog snapshot based on user's dialog choice.

    JSON body:
    {
      "section": "historical" or "new",
      "snapshot_name": "User provided name (for persistent saves)",
      "choice": "skip" | "session" | "persistent",
      "operation": "what triggered this (e.g., 'catalog_replace', 'bulk_import')"
    }

    Returns:
    - 200: Success with snapshot info
    - 400: Invalid parameters
    - 500: Server error
    """
    try:
        data = request.get_json() or {}
        section = data.get('section', '').lower()
        snapshot_name = data.get('snapshot_name', '')
        choice = data.get('choice', '').lower()
        operation = data.get('operation')

        # Validate
        if section not in ['historical', 'new']:
            return create_error_response(
                'INVALID_SECTION',
                f"Invalid section: {section}. Must be 'historical' or 'new'",
                status_code=400
            )

        if choice not in ['skip', 'session', 'persistent']:
            return create_error_response(
                'INVALID_CHOICE',
                f"Invalid choice: {choice}. Must be 'skip', 'session', or 'persistent'",
                status_code=400
            )

        if choice == 'persistent' and not snapshot_name:
            return create_error_response(
                'MISSING_NAME',
                "Snapshot name required for persistent saves",
                status_code=400
            )

        from snapshot_manager import save_snapshot_with_dialog_choice

        result = save_snapshot_with_dialog_choice(section, snapshot_name, choice, operation)

        if 'error' in result:
            return create_error_response(
                'SAVE_ERROR',
                result['error'],
                status_code=500
            )

        return jsonify({
            'status': 'success',
            **result
        }), 200

    except Exception as e:
        logger.error(f"Error saving catalog with dialog: {e}", exc_info=True)
        return create_error_response(
            'SAVE_ERROR',
            'Failed to save catalog',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/check-crash-recovery', methods=['GET'])
def check_crash_recovery():
    """
    Check if app crashed and offer recovery option.

    Returns:
    - 200: With crash detection result and available recovery snapshots
    """
    try:
        crash_detected = globals().get('crash_detected', False)

        recovery_info = {
            'crash_detected': crash_detected,
            'available_recovery': None
        }

        if crash_detected:
            # Find most recent session autosave
            from snapshot_manager import list_snapshots

            all_snapshots = list_snapshots()

            # Check both historical and new for session autosaves
            latest_session = None
            latest_time = None

            for section_key in ['historical', 'new']:
                for snapshot in all_snapshots.get(section_key, []):
                    if snapshot.get('session_only') and 'error' not in snapshot:
                        created_at = snapshot.get('created_at')
                        if created_at and (latest_time is None or created_at > latest_time):
                            latest_session = snapshot
                            latest_time = created_at

            if latest_session:
                recovery_info['recovery_snapshot'] = {
                    'id': latest_session.get('snapshot_file'),  # Use snapshot_file as ID
                    'name': latest_session.get('name'),
                    'section': 'historical' if latest_session.get('is_historical') else 'new',
                    'created_at': latest_session.get('created_at'),
                    'product_count': latest_session.get('product_count', 0),
                    'created_by_operation': latest_session.get('created_by_operation')
                }

        return jsonify({
            'status': 'success',
            **recovery_info
        }), 200

    except Exception as e:
        logger.error(f"Error checking crash recovery: {e}", exc_info=True)
        return create_error_response(
            'RECOVERY_CHECK_ERROR',
            'Failed to check crash recovery',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/create', methods=['POST'])
def create_catalog_snapshot():
    """
    Create a new catalog snapshot.
    
    JSON body:
    - name: Snapshot display name (required)
    - is_historical: Whether this is a historical catalog (default: true)
    - description: Optional description
    - tags: Optional list of tags
    
    Returns:
    - 200: Success with snapshot info
    - 400: Validation error
    - 500: Server error
    """
    try:
        from snapshot_manager import create_snapshot
        
        data = request.get_json()
        
        if not data or 'name' not in data:
            return create_error_response(
                'MISSING_NAME',
                'Snapshot name is required',
                status_code=400
            )
        
        name = data['name'].strip()
        if not name:
            return create_error_response(
                'INVALID_NAME',
                'Snapshot name cannot be empty',
                status_code=400
            )
        
        if len(name) > 100:
            return create_error_response(
                'NAME_TOO_LONG',
                'Snapshot name must be 100 characters or less',
                status_code=400
            )
        
        is_historical = data.get('is_historical', True)
        description = data.get('description', '')
        tags = data.get('tags', [])
        
        result = create_snapshot(
            name=name,
            is_historical=is_historical,
            description=description,
            tags=tags
        )
        
        if result.get('error'):
            return create_error_response(
                'CREATE_ERROR',
                result['error'],
                status_code=400
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error creating snapshot: {e}", exc_info=True)
        return create_error_response(
            'CREATE_ERROR',
            'Failed to create snapshot',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/<path:snapshot_name>', methods=['DELETE'])
def delete_catalog_snapshot(snapshot_name):
    """
    Delete a catalog snapshot.
    
    Returns:
    - 200: Success
    - 400: Cannot delete active snapshot
    - 404: Snapshot not found
    - 500: Server error
    """
    try:
        from snapshot_manager import delete_snapshot
        
        result = delete_snapshot(snapshot_name)

        if result.get('error'):
            if 'not found' in result['error'].lower():
                return create_error_response(
                    'NOT_FOUND',
                    result['error'],
                    status_code=404
                )
            return create_error_response(
                'DELETE_ERROR',
                result['error'],
                status_code=400
            )

        # Clear category cache for the deleted catalog
        invalidate_catalog_categories_cache(snapshot_name)
        logger.info(f"[CACHE] Cleared category cache for deleted catalog: {snapshot_name}")

        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting snapshot: {e}", exc_info=True)
        return create_error_response(
            'DELETE_ERROR',
            'Failed to delete snapshot',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/<path:snapshot_name>/rename', methods=['PUT'])
def rename_catalog_snapshot(snapshot_name):
    """
    Rename a catalog snapshot.
    
    JSON body:
    - new_name: New display name (required)
    
    Returns:
    - 200: Success
    - 400: Validation error
    - 404: Snapshot not found
    - 500: Server error
    """
    try:
        from snapshot_manager import rename_snapshot
        
        data = request.get_json()
        
        if not data or 'new_name' not in data:
            return create_error_response(
                'MISSING_NAME',
                'New name is required',
                status_code=400
            )
        
        new_name = data['new_name'].strip()
        if not new_name:
            return create_error_response(
                'INVALID_NAME',
                'New name cannot be empty',
                status_code=400
            )
        
        result = rename_snapshot(snapshot_name, new_name)
        
        if result.get('error'):
            if 'not found' in result['error'].lower():
                return create_error_response(
                    'NOT_FOUND',
                    result['error'],
                    status_code=404
                )
            return create_error_response(
                'RENAME_ERROR',
                result['error'],
                status_code=400
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error renaming snapshot: {e}", exc_info=True)
        return create_error_response(
            'RENAME_ERROR',
            'Failed to rename snapshot',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/merge', methods=['POST'])
def merge_catalog_snapshots():
    """
    Merge multiple snapshots into a new one.
    
    JSON body:
    - snapshots: List of snapshot filenames to merge (required, min 2)
    - new_name: Name for merged snapshot (required)
    - is_historical: Whether merged snapshot is historical (default: true)
    
    Returns:
    - 200: Success with merged snapshot info
    - 400: Validation error
    - 500: Server error
    """
    try:
        from snapshot_manager import merge_snapshots
        
        data = request.get_json()
        
        if not data:
            return create_error_response(
                'MISSING_DATA',
                'Request body is required',
                status_code=400
            )
        
        snapshots = data.get('snapshots', [])
        if len(snapshots) < 2:
            return create_error_response(
                'INVALID_SNAPSHOTS',
                'At least 2 snapshots are required for merge',
                status_code=400
            )
        
        new_name = data.get('new_name', '').strip()
        if not new_name:
            return create_error_response(
                'MISSING_NAME',
                'New name is required',
                status_code=400
            )
        
        is_historical = data.get('is_historical', True)
        
        result = merge_snapshots(snapshots, new_name, is_historical)
        
        if result.get('error'):
            return create_error_response(
                'MERGE_ERROR',
                result['error'],
                status_code=400
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error merging snapshots: {e}", exc_info=True)
        return create_error_response(
            'MERGE_ERROR',
            'Failed to merge snapshots',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/active', methods=['GET'])
def get_active_catalog_snapshots():
    """
    Get currently active catalog snapshots.
    
    Returns:
    - 200: Success with active snapshot lists
    - 500: Server error
    """
    try:
        from snapshot_manager import get_active_catalogs, get_combined_products_count
        
        active = get_active_catalogs()
        counts = get_combined_products_count()
        
        return jsonify({
            'status': 'success',
            **active,
            **counts
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting active catalogs: {e}", exc_info=True)
        return create_error_response(
            'ACTIVE_ERROR',
            'Failed to get active catalogs',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/active', methods=['POST'])
def set_active_catalog_snapshots():
    """
    Set active catalog snapshots.
    
    JSON body:
    - historical: List of historical snapshot filenames
    - new: List of new product snapshot filenames
    
    Returns:
    - 200: Success
    - 400: Validation error
    - 500: Server error
    """
    try:
        from snapshot_manager import set_active_catalogs
        
        data = request.get_json()
        
        if not data:
            return create_error_response(
                'MISSING_DATA',
                'Request body is required',
                status_code=400
            )
        
        historical = data.get('historical', [])
        new = data.get('new', [])
        
        if not isinstance(historical, list) or not isinstance(new, list):
            return create_error_response(
                'INVALID_DATA',
                'historical and new must be arrays',
                status_code=400
            )
        
        result = set_active_catalogs(historical, new)
        
        if result.get('error'):
            return create_error_response(
                'SET_ACTIVE_ERROR',
                result['error'],
                status_code=400
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting active catalogs: {e}", exc_info=True)
        return create_error_response(
            'SET_ACTIVE_ERROR',
            'Failed to set active catalogs',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/<path:snapshot_name>/info', methods=['GET'])
def get_catalog_snapshot_info(snapshot_name):
    """
    Get detailed info for a specific snapshot.
    
    Returns:
    - 200: Success with snapshot info
    - 404: Snapshot not found
    - 500: Server error
    """
    try:
        from snapshot_manager import get_snapshot_info
        
        result = get_snapshot_info(snapshot_name)
        
        if result.get('error'):
            if 'not found' in result['error'].lower():
                return create_error_response(
                    'NOT_FOUND',
                    result['error'],
                    status_code=404
                )
            return create_error_response(
                'INFO_ERROR',
                result['error'],
                status_code=500
            )
        
        return jsonify({
            'status': 'success',
            'snapshot': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting snapshot info: {e}", exc_info=True)
        return create_error_response(
            'INFO_ERROR',
            'Failed to get snapshot info',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/export', methods=['POST'])
def export_catalog_snapshot():
    """
    Export a snapshot as a downloadable .zip file.
    
    JSON body:
    - snapshot: Snapshot filename to export (required)
    
    Returns:
    - 200: Zip file download
    - 400: Validation error
    - 404: Snapshot not found
    - 500: Server error
    """
    try:
        from snapshot_manager import export_snapshot, CATALOGS_DIR
        
        data = request.get_json()
        
        if not data or 'snapshot' not in data:
            return create_error_response(
                'MISSING_SNAPSHOT',
                'Snapshot name is required',
                status_code=400
            )
        
        snapshot_name = data['snapshot']
        result = export_snapshot(snapshot_name)
        
        if result.get('error'):
            if 'not found' in result['error'].lower():
                return create_error_response(
                    'NOT_FOUND',
                    result['error'],
                    status_code=404
                )
            return create_error_response(
                'EXPORT_ERROR',
                result['error'],
                status_code=500
            )
        
        # Return the zip file
        zip_path = result['zip_path']
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=os.path.basename(zip_path)
        )
        
    except Exception as e:
        logger.error(f"Error exporting snapshot: {e}", exc_info=True)
        return create_error_response(
            'EXPORT_ERROR',
            'Failed to export snapshot',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/import', methods=['POST'])
def import_catalog_snapshot():
    """
    Import a snapshot from an uploaded .zip file.
    
    Form data:
    - file: Zip file to import (required)
    
    Returns:
    - 200: Success with imported snapshot info
    - 400: Validation error
    - 500: Server error
    """
    try:
        from snapshot_manager import import_snapshot, CATALOGS_DIR
        
        if 'file' not in request.files:
            return create_error_response(
                'MISSING_FILE',
                'Zip file is required',
                status_code=400
            )
        
        file = request.files['file']
        
        if file.filename == '':
            return create_error_response(
                'EMPTY_FILENAME',
                'No file selected',
                status_code=400
            )
        
        if not file.filename.endswith('.zip'):
            return create_error_response(
                'INVALID_FORMAT',
                'File must be a .zip archive',
                status_code=400
            )
        
        # Save uploaded file temporarily
        temp_path = os.path.join(CATALOGS_DIR, f"temp-import-{datetime.now().strftime('%Y%m%d%H%M%S')}.zip")
        file.save(temp_path)
        
        try:
            result = import_snapshot(temp_path)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if result.get('error'):
            return create_error_response(
                'IMPORT_ERROR',
                result['error'],
                status_code=400
            )

        # Invalidate CSV cache since catalog was imported
        invalidate_csv_cache()

        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error importing snapshot: {e}", exc_info=True)
        return create_error_response(
            'IMPORT_ERROR',
            'Failed to import snapshot',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/save-current', methods=['POST'])
def save_current_as_snapshot():
    """
    Save the current main database as a new snapshot.
    
    JSON body:
    - name: Snapshot name (required)
    - description: Optional description
    - tags: Optional list of tags
    
    Returns:
    - 200: Success with snapshot info
    - 400: Validation error
    - 500: Server error
    """
    try:
        from snapshot_manager import save_main_db_as_snapshot
        
        data = request.get_json()
        
        if not data or 'name' not in data:
            return create_error_response(
                'MISSING_NAME',
                'Snapshot name is required',
                status_code=400
            )
        
        name = data['name'].strip()
        if not name:
            return create_error_response(
                'INVALID_NAME',
                'Snapshot name cannot be empty',
                status_code=400
            )
        
        description = data.get('description', '')
        tags = data.get('tags', [])

        # Check if this is a session save (temporary, expires in 1 hour)
        session_only = 'session' in [tag.lower() for tag in tags] if tags else False

        logger.info(f"[SNAPSHOT-API] Saving snapshot '{name}' - Type: {'Session' if session_only else 'Persistent'}, Tags: {tags}")
        result = save_main_db_as_snapshot(name, description, tags, session_only=session_only)
        
        if result.get('error'):
            return create_error_response(
                'SAVE_ERROR',
                result['error'],
                status_code=400
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error saving current as snapshot: {e}", exc_info=True)
        return create_error_response(
            'SAVE_ERROR',
            'Failed to save snapshot',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/load/<path:snapshot_name>', methods=['POST'])
def load_snapshot_to_main(snapshot_name):
    """
    Load a snapshot into the main database.
    
    This replaces the current main database with the snapshot contents.
    
    Returns:
    - 200: Success
    - 404: Snapshot not found
    - 500: Server error
    """
    try:
        from snapshot_manager import load_snapshot_to_main_db
        
        result = load_snapshot_to_main_db(snapshot_name)
        
        if result.get('error'):
            if 'not found' in result['error'].lower():
                return create_error_response(
                    'NOT_FOUND',
                    result['error'],
                    status_code=404
                )
            return create_error_response(
                'LOAD_ERROR',
                result['error'],
                status_code=500
            )
        
        # Invalidate CSV cache since snapshot was loaded
        invalidate_csv_cache()

        # Rebuild FAISS indexes after loading snapshot
        logger.info("Rebuilding FAISS indexes after snapshot load...")
        try:
            from database import rebuild_all_faiss_indexes
            faiss_stats = rebuild_all_faiss_indexes()
            if 'error' not in faiss_stats:
                logger.info(f"FAISS indexes rebuilt: {faiss_stats['categories_processed']} categories")
                result['faiss_indexes_rebuilt'] = faiss_stats['categories_processed']
        except Exception as e:
            logger.warning(f"Failed to rebuild FAISS indexes: {e}")
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error loading snapshot: {e}", exc_info=True)
        return create_error_response(
            'LOAD_ERROR',
            'Failed to load snapshot',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/main-db-stats', methods=['GET'])
def get_main_database_stats():
    """
    Get statistics about the main database and loaded snapshot info.
    
    Returns:
    - 200: Success with stats
    - 500: Server error
    """
    try:
        from snapshot_manager import get_main_db_stats
        
        stats = get_main_db_stats()
        
        return jsonify({
            'status': 'success',
            **stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting main db stats: {e}", exc_info=True)
        return create_error_response(
            'STATS_ERROR',
            'Failed to get database stats',
            {'error': str(e)},
            status_code=500
        )


@app.route('/api/catalogs/csv-content', methods=['GET'])
def get_catalog_csv_content():
    """Get CSV content for a specific section from the loaded catalog

    Query Parameters:
    - section: 'historical' or 'new' (default: 'historical')

    Returns:
    - 200: Success with CSV content and metadata
    - 404: No catalog loaded or no CSV found
    - 400: Invalid section parameter
    - 500: Server error
    """
    try:
        from snapshot_manager import get_loaded_snapshot_info, get_csv_from_snapshot, CATALOGS_DIR

        # Get section parameter (default to 'historical')
        section = request.args.get('section', 'historical').lower()

        if section not in ['historical', 'new']:
            logger.warning(f"[CSV-AUTO-LOAD] Invalid section: {section}")
            return create_error_response(
                'INVALID_SECTION',
                f"Invalid section '{section}'. Must be 'historical' or 'new'",
                status_code=400
            )

        is_historical = (section == 'historical')

        # Get currently loaded snapshot
        loaded_info = get_loaded_snapshot_info()
        snapshot_file = loaded_info.get('snapshot_file') if loaded_info.get('loaded') else None

        # Check cache first (cache key: snapshot_file + section)
        cache_key = f"{snapshot_file}:{section}"
        csv_data = get_cached_csv(cache_key)
        if csv_data:
            logger.debug(f"[CSV-AUTO-LOAD] ✓ Serving {section} CSV from cache")
            logger.debug(f"[CSV-AUTO-LOAD] ✓ LOADED (cached): {section} CSV - {csv_data['filename']} ({csv_data['row_count']} rows)")
            return jsonify({
                'has_csv': True,
                'section': section,
                'csv_content': csv_data['csv_content'],
                'filename': csv_data['filename'],
                'row_count': csv_data['row_count'],
                'uploaded_at': csv_data.get('uploaded_at'),
                'cached': True
            }), 200

        logger.debug(f"[CSV-AUTO-LOAD] ▶ Request for {section} CSV auto-load (not cached)")

        if loaded_info.get('loaded'):
            # Snapshot is explicitly loaded
            logger.debug(f"[CSV-AUTO-LOAD]   Loaded snapshot: {snapshot_file}")
        else:
            # No explicit snapshot loaded - try to use main database directly
            # This handles the case where user selected "use_existing" without explicitly loading a snapshot
            logger.debug(f"[CSV-AUTO-LOAD] No explicit snapshot loaded - checking main database for CSV metadata...")

        # Load CSV - either from snapshot or main database
        csv_data = None
        from snapshot_manager import extract_csv_from_db, DEFAULT_DB_PATH

        if snapshot_file:
            # Get CSV from snapshot for specific section
            snapshot_path = os.path.join(CATALOGS_DIR, snapshot_file)

            if not os.path.exists(snapshot_path):
                logger.error(f"[CSV-AUTO-LOAD] ✗ Snapshot file not found at: {snapshot_path}")
                return create_error_response(
                    'SNAPSHOT_NOT_FOUND',
                    f'Snapshot file not found: {snapshot_file}',
                    status_code=404
                )

            logger.debug(f"[CSV-AUTO-LOAD]   Retrieving {section} CSV from snapshot...")
            csv_data = get_csv_from_snapshot(snapshot_path, is_historical=is_historical)

            # Fallback: if snapshot doesn't have CSV (e.g., created before CSV column fix), extract from main DB
            if not csv_data:
                logger.debug(f"[CSV-AUTO-LOAD]   Snapshot has no CSV data (old snapshot?), falling back to main database...")
                try:
                    csv_result = extract_csv_from_db(DEFAULT_DB_PATH, is_historical=is_historical)
                    if csv_result:
                        csv_content, row_count = csv_result
                        csv_data = {
                            'csv_content': csv_content,
                            'filename': f"{section}-{datetime.now().strftime('%Y%m%d')}.csv",
                            'row_count': row_count,
                            'uploaded_at': datetime.now().isoformat(),
                            'section': section
                        }
                        logger.debug(f"[CSV-AUTO-LOAD] ✓ Fallback: Extracted from main DB: {csv_data['filename']} ({row_count} rows)")
                except Exception as e:
                    logger.warning(f"[CSV-AUTO-LOAD] Could not extract CSV from main database: {e}")
                    csv_data = None
        else:
            # No snapshot loaded - try to extract CSV from main database
            # This is used when user selected "use_existing" without formally loading a snapshot
            logger.debug(f"[CSV-AUTO-LOAD]   Extracting {section} CSV from main database...")

            try:
                csv_result = extract_csv_from_db(DEFAULT_DB_PATH, is_historical=is_historical)
                if csv_result:
                    csv_content, row_count = csv_result
                    csv_data = {
                        'csv_content': csv_content,
                        'filename': f"{section}-{datetime.now().strftime('%Y%m%d')}.csv",
                        'row_count': row_count,
                        'uploaded_at': datetime.now().isoformat(),
                        'section': section
                    }
                    logger.debug(f"[CSV-AUTO-LOAD] ✓ Extracted from main DB: {csv_data['filename']} ({row_count} rows)")
            except Exception as e:
                logger.warning(f"[CSV-AUTO-LOAD] Could not extract CSV from main database: {e}")
                csv_data = None

        if not csv_data:
            logger.debug(f"[CSV-AUTO-LOAD] ✗ No {section} CSV found in snapshot (normal if not used)")
            return jsonify({
                'has_csv': False,
                'section': section,
                'message': f'No CSV data for {section} section'
            }), 200

        # Cache the result for future requests (with LRU eviction)
        cache_csv_data(cache_key, csv_data)
        logger.debug(f"[CSV-AUTO-LOAD] ✓ LOADED: {section} CSV - {csv_data['filename']} ({csv_data['row_count']} rows)")
        return jsonify({
            'has_csv': True,
            'section': section,
            'csv_content': csv_data['csv_content'],
            'filename': csv_data['filename'],
            'row_count': csv_data['row_count'],
            'uploaded_at': csv_data['uploaded_at']
        }), 200

    except Exception as e:
        logger.error(f"[CSV-AUTO-LOAD] ✗ ERROR: {str(e)}", exc_info=True)
        return create_error_response(
            'CSV_RETRIEVAL_ERROR',
            'Failed to retrieve CSV content',
            {'error': str(e)},
            status_code=500
        )


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
        except:
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
        except:
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
        except:
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
    app.run(host='127.0.0.1', port=5000, debug=True)
