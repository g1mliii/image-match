from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import re
import logging
from werkzeug.utils import secure_filename
from datetime import datetime

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
    vacuum_database, clear_uploaded_images, export_catalog_csv, delete_features
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
# No max content length - handle large files gracefully with proper error handling

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
init_db()

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

# Memory leak fix: Add cleanup handlers for app shutdown
# DISABLED: This was running after EVERY request, killing performance
# The CLIP model should stay cached in memory for fast processing
# @app.teardown_appcontext
# def cleanup_resources(exception=None):
#     """
#     Clean up resources on app context teardown to prevent memory leaks.
#     
#     This handler is called when the Flask app context is torn down,
#     which happens at the end of each request or when the app shuts down.
#     """
#     # Only do full cleanup on app shutdown (not on every request)
#     # We detect shutdown by checking if we're in the main thread
#     import threading
#     if threading.current_thread() is threading.main_thread():
#         return  # Skip cleanup during normal request handling
#     
#     try:
#         # Clear CLIP model cache (350MB+ memory)
#         from image_processing_clip import clear_clip_model_cache
#         clear_clip_model_cache()
#         logger.info("CLIP model cache cleared")
#     except Exception as e:
#         logger.warning(f"Failed to clear CLIP model cache: {e}")
#     
#     # Force garbage collection
#     import gc
#     gc.collect()
#     
#     # Clear CUDA/GPU cache if available
#     try:
#         import torch
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             logger.info("CUDA cache cleared")
#     except:
#         pass


def cleanup_on_shutdown():
    """
    Explicit cleanup function to call on application shutdown.
    This should be called from main.py when the desktop app closes.
    """
    logger.info("Starting application shutdown cleanup...")
    
    try:
        # Clear CLIP model cache (350MB+ memory)
        from image_processing_clip import clear_clip_model_cache
        clear_clip_model_cache()
        logger.info("✓ CLIP model cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear CLIP model cache: {e}")
    
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

@app.route('/catalog-manager')
def catalog_manager():
    """Serve the Catalog Manager tool"""
    return send_from_directory(app.static_folder, 'catalog-manager.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'})

@app.route('/api/gpu/status', methods=['GET'])
def get_gpu_status():
    """
    Get GPU acceleration status and performance information.
    
    Returns:
        JSON with GPU status:
        - available: bool - Whether GPU is available
        - device: str - Device type (cuda, rocm, mps, cpu)
        - gpu_name: str - GPU model name (if available)
        - throughput: str - Estimated throughput (images/sec)
        - first_run: bool - Whether CLIP model needs to be downloaded
        - error: str - Error message if GPU initialization failed
    """
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
    """
    Create a product with metadata only (no image) - Mode 2 support.
    
    JSON body:
    - sku: Product SKU (required)
    - product_name: Product name (required)
    - category: Product category (optional)
    - price: Product price (optional)
    - performance_history: Performance history array (optional)
    - is_historical: Boolean (default: false)
    
    Returns:
    - 200: Success with product_id
    - 400: Validation error
    - 500: Server error
    """
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
    """
    Create multiple metadata-only products in one batch (Mode 2 CSV import).
    
    JSON body:
    {
        "products": [
            {
                "sku": "SKU001",
                "product_name": "Product 1",
                "category": "Electronics",
                "is_historical": true,
                "performance_history": [100, 150, 200]
            },
            ...
        ]
    }
    
    Returns:
    - 200: Success with product_ids
    - 400: Validation error
    - 500: Server error
    """
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
            
            return {
                'sku': sku,
                'product_name': product_name,
                'category': category,
                'is_historical': is_historical,
                'performance_history': product.get('performance_history', None)
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
        
        CHUNK_SIZE = 100
        product_ids = []
        
        for chunk_idx in range(0, len(validated_products), CHUNK_SIZE):
            chunk = validated_products[chunk_idx:chunk_idx + CHUNK_SIZE]
            products_to_insert = [
                ('[METADATA_ONLY]', p['category'], p['product_name'], p['sku'], p['is_historical'])
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
        
        logger.info(f"[BATCH-METADATA] ✓ Complete! {len(product_ids)} products created successfully")
        
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


@app.route('/api/products/upload', methods=['POST'])
def upload_product():
    """
    Upload a new product with image and optional metadata.
    
    Form data:
    - image: Image file (required) - JPEG, PNG, or WebP
    - category: Product category (optional, can be NULL)
    - product_name: Product name (optional)
    - sku: Product SKU (optional, alphanumeric with hyphens/underscores)
    
    Returns:
    - 200: Success with product_id and feature extraction status
    - 400: Validation error
    - 500: Server error
    """
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
    """
    Batch upload multiple products with images and optional metadata.
    Uses parallel GPU batch processing for CLIP feature extraction.
    
    Form data (multipart/form-data):
    - images: Multiple image files (required) - JPEG, PNG, or WebP
    - categories: JSON array of categories (optional, same length as images or single value)
    - product_names: JSON array of product names (optional, same length as images)
    - skus: JSON array of SKUs (optional, same length as images)
    - is_historical: Boolean (default: false)
    
    Returns:
    - 200: Success with results for each product
    - 400: Validation error
    - 500: Server error
    """
    try:
        logger.info("[BATCH-UPLOAD] Starting batch upload")
        
        # Get uploaded files
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return create_error_response(
                'MISSING_IMAGES',
                'No image files provided',
                'Please upload at least one image file',
                status_code=400
            )
        
        logger.info(f"[BATCH-UPLOAD] Received {len(files)} images")
        
        # Get optional metadata arrays
        import json
        
        categories = request.form.get('categories', None)
        product_names = request.form.get('product_names', None)
        skus = request.form.get('skus', None)
        is_historical = request.form.get('is_historical', 'false').lower() == 'true'
        
        # Parse JSON arrays
        try:
            categories = json.loads(categories) if categories else [None] * len(files)
            product_names = json.loads(product_names) if product_names else [None] * len(files)
            skus = json.loads(skus) if skus else [None] * len(files)
        except json.JSONDecodeError as e:
            return create_error_response(
                'INVALID_JSON',
                f'Failed to parse metadata JSON: {str(e)}',
                'Ensure categories, product_names, and skus are valid JSON arrays',
                status_code=400
            )
        
        # Validate array lengths
        if len(categories) == 1 and len(files) > 1:
            # Single category for all products
            categories = categories * len(files)
        
        if len(categories) != len(files):
            return create_error_response(
                'ARRAY_LENGTH_MISMATCH',
                f'categories array length ({len(categories)}) does not match number of images ({len(files)})',
                'Provide one category per image or a single category for all',
                status_code=400
            )
        
        if len(product_names) != len(files):
            return create_error_response(
                'ARRAY_LENGTH_MISMATCH',
                f'product_names array length ({len(product_names)}) does not match number of images ({len(files)})',
                'Provide one product name per image',
                status_code=400
            )
        
        if len(skus) != len(files):
            return create_error_response(
                'ARRAY_LENGTH_MISMATCH',
                f'skus array length ({len(skus)}) does not match number of images ({len(files)})',
                'Provide one SKU per image',
                status_code=400
            )
        
        # Step 1: Save all files and validate (skip invalid ones, retry once)
        logger.info("[BATCH-UPLOAD] Step 1: Saving and validating files")
        
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
        
        logger.info(f"[BATCH-UPLOAD] {len(saved_files)} files saved and validated (after retry)")
        
        # Step 2: Insert products into database (incremental to overlap with feature extraction)
        logger.info("[BATCH-UPLOAD] Step 2: Inserting products into database (incremental)")
        product_ids = []
        PRODUCT_BATCH_SIZE = 32  # Insert every 32 products (matches GPU batch size)
        
        for i, filepath in enumerate(saved_files):
            original_idx = file_indices[i]  # Map back to original file index
            category = categories[original_idx]
            product_name = product_names[original_idx]
            sku = skus[original_idx]
            
            # Normalize empty strings to None
            if category and str(category).strip() == '':
                category = None
            if product_name and str(product_name).strip() == '':
                product_name = None
            if sku and str(sku).strip() == '':
                sku = None
            
            # Validate SKU if provided
            if sku:
                is_valid, error_msg = validate_sku_format(sku)
                if not is_valid:
                    logger.warning(f"[BATCH-UPLOAD] Invalid SKU for file {i+1}: {error_msg}")
                    sku = None
                else:
                    sku = normalize_sku(sku)
            
            # Insert product
            try:
                product_id = insert_product(
                    image_path=filepath,
                    category=category,
                    product_name=product_name,
                    sku=sku,
                    is_historical=is_historical
                )
                product_ids.append(product_id)
                logger.debug(f"[BATCH-UPLOAD] Inserted product {i+1}/{len(saved_files)}: ID={product_id}")
            except Exception as e:
                logger.error(f"[BATCH-UPLOAD] Database error for file {i+1}: {e}")
                # Continue with other products
                product_ids.append(None)
        
        inserted_count = sum(1 for pid in product_ids if pid is not None)
        logger.info(f"[BATCH-UPLOAD] Inserted {inserted_count}/{len(saved_files)} products")
        
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
        
        # Step 5: Rebuild FAISS indexes for affected categories
        # Always rebuild FAISS indexes when new products are added (both historical and new)
        logger.info("[BATCH-UPLOAD] Step 5: Rebuilding FAISS indexes")
        try:
            from database import rebuild_all_faiss_indexes
            rebuild_all_faiss_indexes()
            logger.info("[BATCH-UPLOAD] FAISS indexes rebuilt")
        except Exception as e:
            logger.warning(f"[BATCH-UPLOAD] Failed to rebuild FAISS indexes: {e}")
        
        # Prepare response
        results = []
        
        # Add successful products
        for i, product_id in enumerate(product_ids):
            original_idx = file_indices[i]
            if product_id is not None:
                results.append({
                    'index': original_idx,
                    'status': 'success',
                    'product_id': product_id,
                    'filename': files[original_idx].filename
                })
            else:
                results.append({
                    'index': original_idx,
                    'status': 'failed',
                    'error': 'Database insertion failed',
                    'filename': files[original_idx].filename
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
        
        return jsonify({
            'status': 'success',
            'total': len(files),
            'successful': success_count,
            'failed': failed_count,
            'skipped': skipped_count,
            'results': results
        }), 200
        
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
    """
    Find similar products for a given product.
    
    JSON body:
    - product_id: ID of product to match (required)
    - threshold: Minimum similarity score 0-100 (optional, default: 0)
    - limit: Maximum number of matches (optional, default: 10)
    - match_against_all: Match against all categories (optional, default: false)
    
    Returns:
    - 200: Success with match results
    - 400: Validation error
    - 404: Product not found
    - 500: Server error
    """
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
        
        # Get metadata weights if provided
        sku_weight = data.get('sku_weight', 0.30)
        name_weight = data.get('name_weight', 0.25)
        category_weight_meta = data.get('category_weight', 0.20)
        price_weight = data.get('price_weight', 0.15)
        performance_weight = data.get('performance_weight', 0.10)
        
        # Get hybrid weights if provided
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
                    sku_weight=sku_weight,
                    name_weight=name_weight,
                    category_weight=category_weight_meta,
                    price_weight=price_weight,
                    performance_weight=performance_weight,
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
                    sku_weight=sku_weight,
                    name_weight=name_weight,
                    category_weight=category_weight_meta,
                    price_weight=price_weight,
                    performance_weight=performance_weight,
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
    """
    Batch match multiple products with full parallelization.
    
    This endpoint processes multiple products in parallel using:
    - Mode 1 (Visual): FAISS + GPU + ThreadPoolExecutor
    - Mode 2 (Metadata): Batch fetching + ThreadPoolExecutor
    - Mode 3 (Hybrid): Both modes in parallel, then merge
    
    Request body:
    {
        "product_ids": [1, 2, 3, ...],
        "threshold": 50,
        "limit": 10,
        "visual_weight": 0.5,
        "metadata_weight": 0.5,
        "match_against_all": false
    }
    
    Returns:
    - 200: Success with batch results
    - 400: Invalid request
    - 500: Server error
    """
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
        
        product_ids = data.get('product_ids', [])
        if not product_ids or not isinstance(product_ids, list):
            return create_error_response(
                'INVALID_PRODUCT_IDS',
                'product_ids must be a non-empty array',
                'Example: {"product_ids": [1, 2, 3]}',
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
        sku_weight = float(data.get('sku_weight', 0.30))
        name_weight = float(data.get('name_weight', 0.25))
        category_weight = float(data.get('category_weight', 0.20))
        price_weight = float(data.get('price_weight', 0.15))
        performance_weight = float(data.get('performance_weight', 0.10))
        
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
                    sku_weight=sku_weight,
                    name_weight=name_weight,
                    category_weight=category_weight,
                    price_weight=price_weight,
                    performance_weight=performance_weight,
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
                    skip_invalid_products=True
                )
            elif is_pure_metadata:
                # Mode 2: Metadata batch matching
                logger.info(f"[BATCH] Mode 2 (Metadata) - Processing {len(product_ids)} products in parallel")
                logger.info(f"[BATCH] Mode 2 metadata weights: SKU={sku_weight}, Name={name_weight}, Category={category_weight}, Price={price_weight}, Performance={performance_weight}")
                logger.info(f"[BATCH] Mode 2 will use ThreadPoolExecutor for parallel metadata comparison (no GPU needed)")
                result = batch_find_metadata_matches(
                    product_ids=product_ids,
                    threshold=threshold,
                    limit=limit,
                    sku_weight=sku_weight,
                    name_weight=name_weight,
                    category_weight=category_weight,
                    price_weight=price_weight,
                    performance_weight=performance_weight,
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
                    sku_weight=sku_weight,
                    name_weight=name_weight,
                    category_weight=category_weight,
                    price_weight=price_weight,
                    performance_weight=performance_weight,
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
            
            logger.info(f"[BATCH] ✓ Batch matching complete!")
            logger.info(f"[BATCH] Results - Successful: {result['summary']['successful']}, Failed: {result['summary']['failed']}")
            logger.info(f"[BATCH] Success rate: {result['summary']['success_rate']}%")
            logger.info(f"[BATCH] Total matches found: {result['summary']['total_matches']}")
            logger.info(f"[BATCH] Batch insert used: {result['summary']['batch_insert_used']}")
            
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
    """
    Clean up session data (matches) on app close.
    Keeps catalogs but deletes all match results and clears FAISS indexes.
    
    Returns:
    - 200: Cleanup successful
    - 500: Server error
    """
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

@app.route('/api/products/search', methods=['GET'])
def search_products():
    """
    Fast search for products by name, SKU, or category.
    Uses database indexes for optimal performance.
    
    Query parameters:
    - q: Search query (required)
    - limit: Maximum results (default: 100, max: 1000)
    
    Returns:
    - 200: List of matching products
    - 400: Missing search query
    - 500: Server error
    """
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
    """
    Get product details by ID.
    
    Returns:
    - 200: Success with product details
    - 404: Product not found
    - 500: Server error
    """
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
    return create_error_response(
        'NOT_FOUND',
        'Endpoint not found',
        'Check the API documentation for valid endpoints',
        status_code=404
    )

@app.errorhandler(405)
def method_not_allowed(error):
    return create_error_response(
        'METHOD_NOT_ALLOWED',
        'HTTP method not allowed for this endpoint',
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
    - 200: Success with number of records added
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
            return create_error_response(
                'MISSING_PRICES',
                'prices array is required',
                'Send JSON body with prices array',
                status_code=400
            )
        
        prices = data['prices']
        if not isinstance(prices, list):
            return create_error_response(
                'INVALID_PRICES',
                'prices must be an array',
                status_code=400
            )
        
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
        
        if not valid_records:
            return create_error_response(
                'NO_VALID_RECORDS',
                'No valid price records provided',
                'Check the validation errors',
                {'errors': errors},
                status_code=400
            )
        
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
    """
    Get performance history for a product.
    
    Query parameters:
    - limit: Maximum number of records (optional, default: 12)
    
    Returns:
    - 200: Success with performance history and statistics
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
    """
    Add performance history records for a product.
    
    JSON body:
    - performance: Array of performance records with 'date', 'sales', 'views', 'conversion_rate', 'revenue'
    
    Returns:
    - 200: Success with number of records added
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
    """
    Get paginated products with filtering.
    
    Query parameters:
    - page: Page number (default: 1)
    - limit: Products per page (default: 50)
    - search: Search query
    - category: Category filter
    - type: 'historical' or 'new'
    - features: 'has_features' or 'no_features'
    - sort: Sort order
    
    Returns:
    - 200: Success with products list
    - 500: Server error
    """
    try:
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
    """
    Update a product's metadata with strict validation.
    
    JSON body:
    - category: New category (optional, max 100 chars)
    - product_name: New name (optional, max 200 chars)
    - sku: New SKU (optional, max 50 chars, alphanumeric with hyphens/underscores)
    
    Returns:
    - 200: Success
    - 400: Validation error
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
        
        # Delete image file
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                logger.warning(f"Failed to delete image file: {e}")
        
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
        
        result = save_main_db_as_snapshot(name, description, tags)
        
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
