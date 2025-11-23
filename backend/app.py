from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import re
import logging
from werkzeug.utils import secure_filename
from datetime import datetime

from database import (
    init_db, insert_product, get_product_by_id, get_features_by_product_id,
    insert_features, validate_sku_format, normalize_sku, check_sku_exists,
    insert_price_history, bulk_insert_price_history, get_price_history,
    get_price_statistics, link_price_history, get_products_with_price_history,
    insert_performance_history, bulk_insert_performance_history, get_performance_history,
    get_performance_statistics, link_performance_history, get_products_with_performance_history
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
    find_hybrid_matches,
    MatchingError, ProductNotFoundError, MissingFeaturesError,
    EmptyCatalogError, AllMatchesFailedError
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
            return create_error_response(
                'IMAGE_NOT_FOUND',
                'Image file not found',
                status_code=404
            )
        
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error serving image for product {product_id}: {e}")
        return create_error_response(
            'IMAGE_ERROR',
            'Failed to load image',
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
                is_historical=False
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
            features, embedding_type, embedding_version = extract_features_unified(filepath)
            
            # Store features in database with embedding type and version
            insert_features(
                product_id=product_id,
                color_features=features['color_features'],
                shape_features=features['shape_features'],
                texture_features=features['texture_features'],
                embedding_type=embedding_type,
                embedding_version=embedding_version
            )
            logger.info(f"Features extracted and stored for product {product_id} (type: {embedding_type}, version: {embedding_version})")
            
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
                    color_weight=color_weight,
                    shape_weight=shape_weight,
                    texture_weight=texture_weight,
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
                from datetime import datetime
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
                from datetime import datetime
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


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
