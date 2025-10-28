from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import logging
from werkzeug.utils import secure_filename
from datetime import datetime

from database import (
    init_db, insert_product, get_product_by_id, get_features_by_product_id,
    insert_features, get_historical_products, count_products,
    validate_sku_format, normalize_sku, check_sku_exists
)
from image_processing import (
    extract_all_features, validate_image_file,
    ImageProcessingError, InvalidImageFormatError, CorruptedImageError,
    ImageTooSmallError, ImageProcessingFailedError
)
from product_matching import (
    find_matches, batch_find_matches,
    MatchingError, ProductNotFoundError, MissingFeaturesError,
    EmptyCatalogError, AllMatchesFailedError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
# No max content length - handle large files gracefully with proper error handling

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
init_db()

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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'})

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
        
        # Extract features from image
        feature_extraction_status = 'success'
        feature_error = None
        
        try:
            features = extract_all_features(filepath)
            
            # Store features in database
            insert_features(
                product_id=product_id,
                color_features=features['color_features'],
                shape_features=features['shape_features'],
                texture_features=features['texture_features']
            )
            logger.info(f"Features extracted and stored for product {product_id}")
            
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
        
        # Find matches with comprehensive error handling
        try:
            result = find_matches(
                product_id=product_id,
                threshold=threshold,
                limit=limit,
                match_against_all=match_against_all,
                include_uncategorized=True,
                store_matches=True,
                skip_invalid_products=True
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

@app.route('/api/products/historical', methods=['GET'])
def get_historical_products():
    """
    Get historical products with pagination and filtering.
    
    Query parameters:
    - category: Filter by category (optional, handles NULL)
    - include_uncategorized: Include products with NULL category (optional, default: false)
    - page: Page number (optional, default: 1)
    - limit: Items per page (optional, default: 25, max: 100)
    - search: Search query (optional)
    
    Returns:
    - 200: Success with product list
    - 400: Validation error
    """
    try:
        # Get query parameters
        category = request.args.get('category', None)
        include_uncategorized = request.args.get('include_uncategorized', 'false').lower() in ['true', '1', 'yes']
        search = request.args.get('search', None)
        
        # Normalize empty category to None
        if category and category.strip() == '':
            category = None
        
        # Validate pagination parameters
        try:
            page = int(request.args.get('page', 1))
            if page < 1:
                return create_error_response(
                    'INVALID_PAGE',
                    'page must be >= 1',
                    f'Received: {page}',
                    status_code=400
                )
        except (ValueError, TypeError):
            return create_error_response(
                'INVALID_PAGE',
                'page must be an integer',
                f'Received: {request.args.get("page")}',
                status_code=400
            )
        
        try:
            limit = int(request.args.get('limit', 25))
            if limit < 1:
                return create_error_response(
                    'INVALID_LIMIT',
                    'limit must be >= 1',
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
                f'Received: {request.args.get("limit")}',
                status_code=400
            )
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Get products with error handling
        try:
            if search:
                # Use search function
                from database import search_products
                products = search_products(
                    query=search,
                    category=category,
                    is_historical=True,
                    limit=limit
                )
                total_count = len(products)  # Approximate for search
            else:
                # Use pagination function
                products = get_historical_products(
                    category=category,
                    limit=limit,
                    offset=offset,
                    include_uncategorized=include_uncategorized
                )
                
                # Get total count for pagination
                total_count = count_products(category=category, is_historical=True)
            
            # Convert sqlite3.Row objects to dictionaries
            products_list = []
            for product in products:
                product_dict = {
                    'id': product['id'],
                    'image_path': product['image_path'],
                    'category': product['category'],
                    'product_name': product['product_name'],
                    'sku': product['sku'],
                    'is_historical': bool(product['is_historical']),
                    'created_at': product['created_at']
                }
                
                # Check if product has features
                try:
                    features = get_features_by_product_id(product['id'])
                    product_dict['has_features'] = features is not None
                except:
                    product_dict['has_features'] = False
                
                products_list.append(product_dict)
            
            # Calculate pagination info
            total_pages = (total_count + limit - 1) // limit if limit > 0 else 0
            
            response = {
                'status': 'success',
                'products': products_list,
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total_count': total_count,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                },
                'filters': {
                    'category': category,
                    'include_uncategorized': include_uncategorized,
                    'search': search
                }
            }
            
            # Return empty results if no matches (not an error)
            if not products_list:
                response['message'] = 'No historical products found'
                if category:
                    response['suggestion'] = f'Try a different category or add products to "{category}"'
                elif search:
                    response['suggestion'] = 'Try a different search query'
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Database error retrieving historical products: {e}")
            return create_error_response(
                'DATABASE_ERROR',
                'Failed to retrieve historical products',
                'Please try again',
                {'error': str(e)},
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in get_historical_products: {e}", exc_info=True)
        return create_error_response(
            'UNKNOWN_ERROR',
            'An unexpected error occurred',
            'Please try again or contact support',
            {'error': str(e)},
            status_code=500
        )

@app.route('/api/products/historical', methods=['POST'])
def add_historical_product():
    """
    Add a new historical product to the catalog.
    
    Form data:
    - image: Image file (required) - JPEG, PNG, or WebP
    - category: Product category (optional, can be NULL)
    - product_name: Product name (optional)
    - sku: Product SKU (optional, alphanumeric with hyphens/underscores)
    
    Returns:
    - 200: Success with product_id
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
        sku_warning = None
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
            
            # Check for duplicate SKU (warn but allow based on config)
            if check_sku_exists(sku):
                logger.warning(f"Duplicate SKU detected for historical product: {sku}")
                sku_warning = f'SKU "{sku}" already exists in database'
                # Allow duplicate - just warn
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"historical_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            logger.info(f"Historical product file saved: {filepath}")
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
        
        # Insert product into database (marked as historical)
        try:
            product_id = insert_product(
                image_path=filepath,
                category=category,
                product_name=product_name,
                sku=sku,
                is_historical=True  # Mark as historical
            )
            logger.info(f"Historical product inserted with ID: {product_id}")
        except Exception as e:
            # Clean up file on database error
            try:
                os.remove(filepath)
            except:
                pass
            
            logger.error(f"Database error inserting historical product: {e}")
            return create_error_response(
                'DATABASE_ERROR',
                'Failed to save product to database',
                'Please try again',
                {'error': str(e)},
                status_code=500
            )
        
        # Extract features from image
        feature_extraction_status = 'success'
        feature_error = None
        
        try:
            features = extract_all_features(filepath)
            
            # Store features in database
            insert_features(
                product_id=product_id,
                color_features=features['color_features'],
                shape_features=features['shape_features'],
                texture_features=features['texture_features']
            )
            logger.info(f"Features extracted and stored for historical product {product_id}")
            
        except (InvalidImageFormatError, CorruptedImageError, ImageTooSmallError, ImageProcessingFailedError) as e:
            logger.error(f"Feature extraction failed for historical product {product_id}: {e.message}")
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
            'is_historical': True,
            'feature_extraction_status': feature_extraction_status
        }
        
        if feature_error:
            response['feature_extraction_error'] = feature_error
            response['warning'] = 'Product saved but feature extraction failed. You can retry feature extraction later.'
        
        if sku_warning:
            response['warning_sku'] = sku_warning
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in add_historical_product: {e}", exc_info=True)
        return create_error_response(
            'UNKNOWN_ERROR',
            'An unexpected error occurred',
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

@app.route('/api/batch/match', methods=['POST'])
def batch_match():
    """
    Batch match multiple products.
    
    JSON body:
    - product_ids: List of product IDs to match (required)
    - threshold: Minimum similarity score 0-100 (optional, default: 0)
    - limit: Maximum matches per product (optional, default: 10)
    - match_against_all: Match against all categories (optional, default: false)
    - continue_on_error: Continue processing on errors (optional, default: true)
    
    Returns:
    - 200: Success with batch results and summary
    - 400: Validation error
    - 500: Server error
    """
    try:
        # Parse JSON body
        data = request.get_json()
        
        if not data:
            return create_error_response(
                'MISSING_BODY',
                'Request body is required',
                'Send JSON body with product_ids array',
                status_code=400
            )
        
        # Validate product_ids
        if 'product_ids' not in data:
            return create_error_response(
                'MISSING_PRODUCT_IDS',
                'product_ids is required',
                'Include product_ids array in request body',
                status_code=400
            )
        
        product_ids = data['product_ids']
        
        if not isinstance(product_ids, list):
            return create_error_response(
                'INVALID_PRODUCT_IDS',
                'product_ids must be an array',
                f'Received: {type(product_ids).__name__}',
                status_code=400
            )
        
        if len(product_ids) == 0:
            return create_error_response(
                'EMPTY_PRODUCT_IDS',
                'product_ids array is empty',
                'Provide at least one product ID',
                status_code=400
            )
        
        if len(product_ids) > 100:
            return create_error_response(
                'TOO_MANY_PRODUCTS',
                f'Too many products in batch ({len(product_ids)})',
                'Maximum 100 products per batch',
                status_code=400
            )
        
        # Validate all product IDs before processing
        invalid_ids = []
        for pid in product_ids:
            try:
                int(pid)
            except (ValueError, TypeError):
                invalid_ids.append(pid)
        
        if invalid_ids:
            return create_error_response(
                'INVALID_PRODUCT_IDS',
                'Some product IDs are not valid integers',
                'All product IDs must be integers',
                {'invalid_ids': invalid_ids},
                status_code=400
            )
        
        # Convert to integers
        product_ids = [int(pid) for pid in product_ids]
        
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
        
        continue_on_error = data.get('continue_on_error', True)
        if not isinstance(continue_on_error, bool):
            continue_on_error = str(continue_on_error).lower() in ['true', '1', 'yes']
        
        # Process batch with error isolation
        logger.info(f"Starting batch match for {len(product_ids)} products")
        
        try:
            result = batch_find_matches(
                product_ids=product_ids,
                threshold=threshold,
                limit=limit,
                match_against_all=match_against_all,
                include_uncategorized=True,
                store_matches=True,
                continue_on_error=continue_on_error,
                skip_invalid_products=True
            )
            
            # Prepare response with detailed summary
            response = {
                'status': 'success',
                'summary': result['summary'],
                'results': result['results'],
                'threshold': threshold,
                'limit': limit
            }
            
            # Include error details if any
            if result.get('errors'):
                response['errors'] = result['errors']
                response['note'] = 'Some products failed to match. See errors for details.'
            
            # Add data quality information
            total_data_issues = 0
            for res in result['results']:
                if res.get('data_quality_issues'):
                    total_data_issues += sum(res['data_quality_issues'].values())
            
            if total_data_issues > 0:
                response['data_quality_note'] = f'Encountered {total_data_issues} data quality issues across all products'
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Error during batch matching: {e}", exc_info=True)
            return create_error_response(
                'BATCH_MATCH_ERROR',
                'Batch matching failed',
                'Please try again with fewer products or check product data',
                {'error': str(e)},
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in batch_match: {e}", exc_info=True)
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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
