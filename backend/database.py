import sqlite3
import os
import numpy as np
import io
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple

# Configure logging
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'product_matching.db')

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def migrate_features_table():
    """Migrate existing features table to support CLIP embeddings
    
    Adds embedding_type and embedding_version columns if they don't exist.
    Safe to run multiple times - checks if columns exist first.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check if features table exists first
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='features'")
        if not cursor.fetchone():
            # Table doesn't exist yet, skip migration
            return
        
        # Check if embedding_type column exists
        cursor.execute("PRAGMA table_info(features)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'embedding_type' not in columns:
            logger.info("Adding embedding_type column to features table")
            cursor.execute('''
                ALTER TABLE features 
                ADD COLUMN embedding_type TEXT DEFAULT 'legacy'
            ''')
            # Update existing rows to have 'legacy' type
            cursor.execute('''
                UPDATE features 
                SET embedding_type = 'legacy' 
                WHERE embedding_type IS NULL
            ''')
        
        if 'embedding_version' not in columns:
            logger.info("Adding embedding_version column to features table")
            cursor.execute('''
                ALTER TABLE features 
                ADD COLUMN embedding_version TEXT DEFAULT NULL
            ''')
        
        conn.commit()
        logger.info("Features table migration complete")

def init_db():
    """Initialize database with schema
    
    Note: All fields except image_path are optional to handle messy real-world data.
    Category can be NULL and will be inferred or set to 'unknown' during processing.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create products table - only image_path is required
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                category TEXT,
                product_name TEXT,
                sku TEXT,
                is_historical BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Create features table
        # Supports both legacy (color/shape/texture) and CLIP embeddings
        # embedding_type: 'legacy' or 'clip'
        # embedding_version: for future model updates (e.g., 'clip-ViT-B-32')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                color_features BLOB NOT NULL,
                shape_features BLOB NOT NULL,
                texture_features BLOB NOT NULL,
                embedding_type TEXT DEFAULT 'legacy',
                embedding_version TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        ''')
        
        # Create matches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                new_product_id INTEGER NOT NULL,
                matched_product_id INTEGER NOT NULL,
                similarity_score REAL NOT NULL,
                color_score REAL NOT NULL,
                shape_score REAL NOT NULL,
                texture_score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (new_product_id) REFERENCES products(id),
                FOREIGN KEY (matched_product_id) REFERENCES products(id)
            )
        ''')
        
        # Create indexes for performance optimization (Task 14)
        # These indexes significantly improve query performance for large catalogs (1000+ products)
        # Note: Indexes on nullable columns still work, NULL values are indexed
        
        # Index for category filtering - speeds up category-based matching
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_products_category 
            ON products(category)
        ''')
        
        # Index for historical/new product filtering
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_products_is_historical 
            ON products(is_historical)
        ''')
        
        # Composite index for efficient category + historical filtering
        # This is the most important index for matching performance
        # Allows fast retrieval of historical products in a specific category
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_products_category_historical 
            ON products(category, is_historical)
        ''')
        
        # Index for match result retrieval sorted by score
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_matches_new_product 
            ON matches(new_product_id, similarity_score DESC)
        ''')
        
        # Index for feature lookups - speeds up feature retrieval during matching
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_features_product_id
            ON features(product_id)
        ''')
        
        # Create price_history table for tracking historical prices
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                price REAL NOT NULL,
                currency TEXT DEFAULT 'USD',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for price_history table for efficient querying
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_price_history_product_id
            ON price_history(product_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_price_history_date
            ON price_history(product_id, date)
        ''')
        
        # Create performance_history table for tracking sales/performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                sales INTEGER DEFAULT 0,
                views INTEGER DEFAULT 0,
                conversion_rate REAL DEFAULT 0.0,
                revenue REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for performance_history table
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_performance_history_product_id
            ON performance_history(product_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_performance_history_date
            ON performance_history(product_id, date)
        ''')
        
        conn.commit()
        print("Database initialized successfully with performance indexes")
    
    # Run migration to add CLIP support columns if needed
    migrate_features_table()

def get_product_by_id(product_id):
    """Get product by ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
        return cursor.fetchone()

def insert_product(image_path, category=None, product_name=None, sku=None, is_historical=False, metadata=None):
    """Insert a new product
    
    Args:
        image_path: Required - path to product image
        category: Optional - product category (can be NULL if unknown)
        product_name: Optional - product name
        sku: Optional - product SKU
        is_historical: Whether this is a historical catalog product
        metadata: Optional - JSON string with additional metadata
    
    Returns:
        int: ID of inserted product
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO products (image_path, category, product_name, sku, is_historical, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_path, category, product_name, sku, is_historical, metadata))
        return cursor.lastrowid

def get_historical_products(category=None, limit=100, offset=0, include_uncategorized=False):
    """Get historical products with optional category filter
    
    Args:
        category: Optional category filter (None returns all)
        limit: Maximum number of results
        offset: Pagination offset
        include_uncategorized: If True and category is specified, also include products with NULL category
    
    Returns:
        List of product rows
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if category:
            if include_uncategorized:
                cursor.execute('''
                    SELECT * FROM products 
                    WHERE is_historical = 1 AND (category = ? OR category IS NULL)
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                ''', (category, limit, offset))
            else:
                cursor.execute('''
                    SELECT * FROM products 
                    WHERE is_historical = 1 AND category = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                ''', (category, limit, offset))
        else:
            cursor.execute('''
                SELECT * FROM products 
                WHERE is_historical = 1
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
        return cursor.fetchall()

def update_product(product_id: int, image_path: Optional[str] = None, 
                  category: Optional[str] = None, product_name: Optional[str] = None,
                  sku: Optional[str] = None, metadata: Optional[str] = None) -> bool:
    """Update product fields"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build dynamic update query
        updates = []
        params = []
        
        if image_path is not None:
            updates.append("image_path = ?")
            params.append(image_path)
        if category is not None:
            updates.append("category = ?")
            params.append(category)
        if product_name is not None:
            updates.append("product_name = ?")
            params.append(product_name)
        if sku is not None:
            updates.append("sku = ?")
            params.append(sku)
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(metadata)
        
        if not updates:
            return False
        
        params.append(product_id)
        query = f"UPDATE products SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        return cursor.rowcount > 0

def delete_product(product_id: int) -> bool:
    """Delete a product and its associated features"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Delete associated features first
        cursor.execute('DELETE FROM features WHERE product_id = ?', (product_id,))
        
        # Delete associated matches
        cursor.execute('''
            DELETE FROM matches 
            WHERE new_product_id = ? OR matched_product_id = ?
        ''', (product_id, product_id))
        
        # Delete product
        cursor.execute('DELETE FROM products WHERE id = ?', (product_id,))
        return cursor.rowcount > 0

def get_products_by_category(category: Optional[str], is_historical: Optional[bool] = None, 
                            include_uncategorized: bool = False) -> List[sqlite3.Row]:
    """Get all products in a specific category
    
    Args:
        category: Category to filter by (None returns uncategorized products)
        is_historical: Optional filter for historical vs new products
        include_uncategorized: If True, include products with NULL category
    
    Returns:
        List of product rows
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if category is None:
            # Get uncategorized products
            if is_historical is not None:
                cursor.execute('''
                    SELECT * FROM products 
                    WHERE category IS NULL AND is_historical = ?
                    ORDER BY created_at DESC
                ''', (is_historical,))
            else:
                cursor.execute('''
                    SELECT * FROM products 
                    WHERE category IS NULL
                    ORDER BY created_at DESC
                ''')
        else:
            # Get products in specific category
            if include_uncategorized:
                if is_historical is not None:
                    cursor.execute('''
                        SELECT * FROM products 
                        WHERE (category = ? OR category IS NULL) AND is_historical = ?
                        ORDER BY created_at DESC
                    ''', (category, is_historical))
                else:
                    cursor.execute('''
                        SELECT * FROM products 
                        WHERE category = ? OR category IS NULL
                        ORDER BY created_at DESC
                    ''', (category,))
            else:
                if is_historical is not None:
                    cursor.execute('''
                        SELECT * FROM products 
                        WHERE category = ? AND is_historical = ?
                        ORDER BY created_at DESC
                    ''', (category, is_historical))
                else:
                    cursor.execute('''
                        SELECT * FROM products 
                        WHERE category = ?
                        ORDER BY created_at DESC
                    ''', (category,))
        
        return cursor.fetchall()

def count_products(category: Optional[str] = None, is_historical: Optional[bool] = None) -> int:
    """Count products with optional filters"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if category is not None:
            conditions.append("category = ?")
            params.append(category)
        if is_historical is not None:
            conditions.append("is_historical = ?")
            params.append(is_historical)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT COUNT(*) FROM products WHERE {where_clause}"
        
        cursor.execute(query, params)
        return cursor.fetchone()[0]

# Feature storage and retrieval functions with numpy array serialization

def serialize_numpy_array(array: np.ndarray) -> bytes:
    """Serialize numpy array to bytes for storage in BLOB"""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()

def deserialize_numpy_array(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to numpy array"""
    buffer = io.BytesIO(blob)
    return np.load(buffer, allow_pickle=False)

def insert_features(product_id: int, color_features: np.ndarray, 
                    shape_features: np.ndarray, texture_features: np.ndarray,
                    embedding_type: str = 'legacy', embedding_version: Optional[str] = None) -> int:
    """Insert feature vectors for a product
    
    Args:
        product_id: Product ID
        color_features: Color feature vector (or CLIP embedding for 'clip' type)
        shape_features: Shape feature vector (or empty for 'clip' type)
        texture_features: Texture feature vector (or empty for 'clip' type)
        embedding_type: 'legacy' or 'clip'
        embedding_version: Model version (e.g., 'clip-ViT-B-32')
    
    Returns:
        Feature record ID
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Serialize numpy arrays to bytes
        color_blob = serialize_numpy_array(color_features)
        shape_blob = serialize_numpy_array(shape_features)
        texture_blob = serialize_numpy_array(texture_features)
        
        cursor.execute('''
            INSERT INTO features (product_id, color_features, shape_features, texture_features, 
                                 embedding_type, embedding_version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (product_id, color_blob, shape_blob, texture_blob, embedding_type, embedding_version))
        
        return cursor.lastrowid

def get_features_by_product_id(product_id: int) -> Optional[Dict[str, Any]]:
    """Get feature vectors for a product
    
    Returns:
        Dictionary with feature arrays and metadata:
        - color_features: Color features or CLIP embedding
        - shape_features: Shape features (empty for CLIP)
        - texture_features: Texture features (empty for CLIP)
        - embedding_type: 'legacy' or 'clip'
        - embedding_version: Model version string
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT color_features, shape_features, texture_features, 
                   embedding_type, embedding_version
            FROM features 
            WHERE product_id = ?
        ''', (product_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'color_features': deserialize_numpy_array(row['color_features']),
            'shape_features': deserialize_numpy_array(row['shape_features']),
            'texture_features': deserialize_numpy_array(row['texture_features']),
            'embedding_type': row['embedding_type'] or 'legacy',
            'embedding_version': row['embedding_version']
        }

def update_features(product_id: int, color_features: Optional[np.ndarray] = None,
                   shape_features: Optional[np.ndarray] = None, 
                   texture_features: Optional[np.ndarray] = None) -> bool:
    """Update feature vectors for a product"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build dynamic update query
        updates = []
        params = []
        
        if color_features is not None:
            updates.append("color_features = ?")
            params.append(serialize_numpy_array(color_features))
        if shape_features is not None:
            updates.append("shape_features = ?")
            params.append(serialize_numpy_array(shape_features))
        if texture_features is not None:
            updates.append("texture_features = ?")
            params.append(serialize_numpy_array(texture_features))
        
        if not updates:
            return False
        
        params.append(product_id)
        query = f"UPDATE features SET {', '.join(updates)} WHERE product_id = ?"
        cursor.execute(query, params)
        return cursor.rowcount > 0

def delete_features(product_id: int) -> bool:
    """Delete feature vectors for a product"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM features WHERE product_id = ?', (product_id,))
        return cursor.rowcount > 0

def get_all_features_by_category(category: Optional[str] = None, is_historical: bool = True,
                                include_uncategorized: bool = False, 
                                embedding_type: Optional[str] = None) -> List[Tuple[int, Dict[str, Any]]]:
    """Get all feature vectors for products in a category
    
    PERFORMANCE OPTIMIZED (Task 14):
    - Uses composite index (category, is_historical) for fast filtering
    - Filters at database level before loading features into memory
    - Single JOIN query instead of multiple queries
    - Critical for performance with large catalogs (1000+ products)
    
    Args:
        category: Category to filter by (None returns all products)
        is_historical: Filter for historical vs new products
        include_uncategorized: If True and category specified, also include NULL category products
        embedding_type: Filter by embedding type ('legacy', 'clip', or None for all)
    
    Returns:
        List of tuples (product_id, features_dict)
        features_dict includes: color_features, shape_features, texture_features, 
                                embedding_type, embedding_version, category
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build base query with embedding type filter
        embedding_filter = ""
        params = []
        
        if category is None:
            # Get all products regardless of category
            query = '''
                SELECT p.id, p.category, f.color_features, f.shape_features, f.texture_features,
                       f.embedding_type, f.embedding_version
                FROM products p
                JOIN features f ON p.id = f.product_id
                WHERE p.is_historical = ?
            '''
            params.append(is_historical)
        elif include_uncategorized:
            # Get products in category OR with NULL category
            query = '''
                SELECT p.id, p.category, f.color_features, f.shape_features, f.texture_features,
                       f.embedding_type, f.embedding_version
                FROM products p
                JOIN features f ON p.id = f.product_id
                WHERE (p.category = ? OR p.category IS NULL) AND p.is_historical = ?
            '''
            params.extend([category, is_historical])
        else:
            # Get products in specific category only
            query = '''
                SELECT p.id, p.category, f.color_features, f.shape_features, f.texture_features,
                       f.embedding_type, f.embedding_version
                FROM products p
                JOIN features f ON p.id = f.product_id
                WHERE p.category = ? AND p.is_historical = ?
            '''
            params.extend([category, is_historical])
        
        # Add embedding type filter if specified
        if embedding_type is not None:
            query += ' AND f.embedding_type = ?'
            params.append(embedding_type)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            product_id = row['id']
            features = {
                'color_features': deserialize_numpy_array(row['color_features']),
                'shape_features': deserialize_numpy_array(row['shape_features']),
                'texture_features': deserialize_numpy_array(row['texture_features']),
                'embedding_type': row['embedding_type'] or 'legacy',
                'embedding_version': row['embedding_version'],
                'category': row['category']  # Include category in results for reference
            }
            results.append((product_id, features))
        
        return results

# Match storage functions

def insert_match(new_product_id: int, matched_product_id: int, 
                similarity_score: float, color_score: float,
                shape_score: float, texture_score: float) -> int:
    """Insert a match result"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO matches 
            (new_product_id, matched_product_id, similarity_score, 
             color_score, shape_score, texture_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (new_product_id, matched_product_id, similarity_score,
              color_score, shape_score, texture_score))
        return cursor.lastrowid

def get_matches_for_product(new_product_id: int, limit: int = 10) -> List[sqlite3.Row]:
    """Get match results for a new product"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT m.*, p.image_path, p.category, p.product_name, p.sku
            FROM matches m
            JOIN products p ON m.matched_product_id = p.id
            WHERE m.new_product_id = ?
            ORDER BY m.similarity_score DESC
            LIMIT ?
        ''', (new_product_id, limit))
        return cursor.fetchall()

def delete_matches_for_product(product_id: int) -> bool:
    """Delete all matches for a product"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM matches 
            WHERE new_product_id = ? OR matched_product_id = ?
        ''', (product_id, product_id))
        return cursor.rowcount > 0

# Utility functions for handling missing/incomplete data

def get_products_without_category(is_historical: Optional[bool] = None) -> List[sqlite3.Row]:
    """Get all products that are missing category information
    
    Useful for identifying products that need manual categorization or 
    category inference from image analysis.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if is_historical is not None:
            cursor.execute('''
                SELECT * FROM products 
                WHERE category IS NULL AND is_historical = ?
                ORDER BY created_at DESC
            ''', (is_historical,))
        else:
            cursor.execute('''
                SELECT * FROM products 
                WHERE category IS NULL
                ORDER BY created_at DESC
            ''')
        return cursor.fetchall()

def get_products_without_features() -> List[sqlite3.Row]:
    """Get all products that don't have feature vectors extracted yet
    
    Useful for batch processing to extract features from images.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.* FROM products p
            LEFT JOIN features f ON p.id = f.product_id
            WHERE f.id IS NULL
            ORDER BY p.created_at DESC
        ''')
        return cursor.fetchall()

def get_incomplete_products() -> List[Dict[str, Any]]:
    """Get products with missing critical information
    
    Returns a list of products with flags indicating what data is missing.
    Useful for data quality monitoring and cleanup.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                p.id,
                p.image_path,
                p.category,
                p.product_name,
                p.sku,
                p.is_historical,
                CASE WHEN f.id IS NULL THEN 1 ELSE 0 END as missing_features,
                CASE WHEN p.category IS NULL THEN 1 ELSE 0 END as missing_category,
                CASE WHEN p.product_name IS NULL THEN 1 ELSE 0 END as missing_name,
                CASE WHEN p.sku IS NULL THEN 1 ELSE 0 END as missing_sku
            FROM products p
            LEFT JOIN features f ON p.id = f.product_id
            WHERE p.category IS NULL 
               OR p.product_name IS NULL 
               OR p.sku IS NULL 
               OR f.id IS NULL
            ORDER BY p.created_at DESC
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'image_path': row['image_path'],
                'category': row['category'],
                'product_name': row['product_name'],
                'sku': row['sku'],
                'is_historical': row['is_historical'],
                'missing_features': bool(row['missing_features']),
                'missing_category': bool(row['missing_category']),
                'missing_name': bool(row['missing_name']),
                'missing_sku': bool(row['missing_sku'])
            })
        
        return results

def get_all_categories() -> List[str]:
    """Get list of all unique categories in the database
    
    Excludes NULL categories. Useful for category selection and validation.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT category 
            FROM products 
            WHERE category IS NOT NULL
            ORDER BY category
        ''')
        return [row['category'] for row in cursor.fetchall()]

def bulk_update_category(product_ids: List[int], category: str) -> int:
    """Update category for multiple products at once
    
    Useful for batch categorization of products.
    
    Returns:
        Number of products updated
    """
    if not product_ids:
        return 0
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(product_ids))
        query = f"UPDATE products SET category = ? WHERE id IN ({placeholders})"
        cursor.execute(query, [category] + product_ids)
        return cursor.rowcount

def re_extract_with_clip(product_id: int) -> bool:
    """Re-extract features for a product using CLIP embeddings
    
    This function is used to upgrade legacy features to CLIP embeddings.
    It deletes the old features and returns True to indicate re-extraction is needed.
    The actual CLIP extraction should be done by the caller.
    
    Args:
        product_id: Product ID to re-extract
    
    Returns:
        True if old features were deleted and re-extraction is needed
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check if product has features
        cursor.execute('SELECT id FROM features WHERE product_id = ?', (product_id,))
        if not cursor.fetchone():
            return False
        
        # Delete old features
        cursor.execute('DELETE FROM features WHERE product_id = ?', (product_id,))
        return cursor.rowcount > 0

def batch_re_extract_with_clip(category: Optional[str] = None, 
                               is_historical: Optional[bool] = None) -> List[int]:
    """Batch re-extraction tool for upgrading entire catalog to CLIP
    
    Returns list of product IDs that need re-extraction.
    
    Args:
        category: Optional category filter
        is_historical: Optional filter for historical vs new products
    
    Returns:
        List of product IDs with legacy features that need re-extraction
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build query to find products with legacy features
        conditions = ["f.embedding_type = 'legacy'"]
        params = []
        
        if category is not None:
            conditions.append("p.category = ?")
            params.append(category)
        
        if is_historical is not None:
            conditions.append("p.is_historical = ?")
            params.append(is_historical)
        
        where_clause = " AND ".join(conditions)
        
        query = f'''
            SELECT p.id
            FROM products p
            JOIN features f ON p.id = f.product_id
            WHERE {where_clause}
        '''
        
        cursor.execute(query, params)
        return [row['id'] for row in cursor.fetchall()]


# SKU-specific utility functions for real-world data handling

def get_products_by_sku(sku: str, case_sensitive: bool = False) -> List[sqlite3.Row]:
    """Get all products with a specific SKU
    
    Args:
        sku: SKU to search for
        case_sensitive: If True, perform case-sensitive search
    
    Returns:
        List of products with matching SKU (may be multiple if duplicates exist)
    
    Note: In real-world data, SKUs may be duplicated due to data entry errors
    or intentional reuse. This function returns all matches.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if case_sensitive:
            cursor.execute('''
                SELECT * FROM products 
                WHERE sku = ?
                ORDER BY created_at DESC
            ''', (sku,))
        else:
            cursor.execute('''
                SELECT * FROM products 
                WHERE LOWER(sku) = LOWER(?)
                ORDER BY created_at DESC
            ''', (sku,))
        return cursor.fetchall()

def check_sku_exists(sku: str, exclude_product_id: Optional[int] = None, 
                     case_sensitive: bool = False) -> bool:
    """Check if SKU already exists in database
    
    Args:
        sku: SKU to check
        exclude_product_id: Optional product ID to exclude from check (for updates)
        case_sensitive: If True, perform case-sensitive check
    
    Returns:
        True if SKU exists, False otherwise
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if exclude_product_id:
            if case_sensitive:
                cursor.execute('''
                    SELECT COUNT(*) FROM products 
                    WHERE sku = ? AND id != ?
                ''', (sku, exclude_product_id))
            else:
                cursor.execute('''
                    SELECT COUNT(*) FROM products 
                    WHERE LOWER(sku) = LOWER(?) AND id != ?
                ''', (sku, exclude_product_id))
        else:
            if case_sensitive:
                cursor.execute('''
                    SELECT COUNT(*) FROM products 
                    WHERE sku = ?
                ''', (sku,))
            else:
                cursor.execute('''
                    SELECT COUNT(*) FROM products 
                    WHERE LOWER(sku) = LOWER(?)
                ''', (sku,))
        
        return cursor.fetchone()[0] > 0

def get_duplicate_skus() -> List[Dict[str, Any]]:
    """Get all SKUs that appear multiple times in the database
    
    Returns:
        List of dicts with 'sku' and 'count' keys, ordered by count descending
    
    Useful for data quality monitoring and identifying potential data issues.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT sku, COUNT(*) as count
            FROM products
            WHERE sku IS NOT NULL
            GROUP BY LOWER(sku)
            HAVING COUNT(*) > 1
            ORDER BY count DESC, sku
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'sku': row['sku'],
                'count': row['count']
            })
        
        return results

def get_products_without_sku(is_historical: Optional[bool] = None) -> List[sqlite3.Row]:
    """Get all products that are missing SKU information
    
    Args:
        is_historical: Optional filter for historical vs new products
    
    Returns:
        List of products without SKU
    
    Useful for identifying products that need SKU assignment.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if is_historical is not None:
            cursor.execute('''
                SELECT * FROM products 
                WHERE sku IS NULL AND is_historical = ?
                ORDER BY created_at DESC
            ''', (is_historical,))
        else:
            cursor.execute('''
                SELECT * FROM products 
                WHERE sku IS NULL
                ORDER BY created_at DESC
            ''')
        return cursor.fetchall()

def search_products(query: str, search_fields: List[str] = None, 
                   category: Optional[str] = None,
                   is_historical: Optional[bool] = None,
                   limit: int = 100) -> List[sqlite3.Row]:
    """Search products across multiple fields
    
    Args:
        query: Search query string
        search_fields: Fields to search in (default: ['product_name', 'sku', 'category'])
        category: Optional category filter
        is_historical: Optional filter for historical vs new products
        limit: Maximum number of results
    
    Returns:
        List of matching products
    
    Note: Handles NULL values gracefully - NULL fields won't match but won't cause errors.
    """
    if search_fields is None:
        search_fields = ['product_name', 'sku', 'category']
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build search conditions
        search_conditions = []
        for field in search_fields:
            search_conditions.append(f"LOWER({field}) LIKE LOWER(?)")
        
        search_clause = " OR ".join(search_conditions)
        
        # Build WHERE clause
        where_parts = [f"({search_clause})"]
        params = [f"%{query}%"] * len(search_fields)
        
        if category is not None:
            where_parts.append("category = ?")
            params.append(category)
        
        if is_historical is not None:
            where_parts.append("is_historical = ?")
            params.append(is_historical)
        
        where_clause = " AND ".join(where_parts)
        params.append(limit)
        
        query_sql = f'''
            SELECT * FROM products 
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        '''
        
        cursor.execute(query_sql, params)
        return cursor.fetchall()

def get_data_quality_stats() -> Dict[str, Any]:
    """Get comprehensive data quality statistics
    
    Returns:
        Dictionary with various data quality metrics
    
    Useful for monitoring data completeness and identifying issues.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Total products
        cursor.execute('SELECT COUNT(*) FROM products')
        total_products = cursor.fetchone()[0]
        
        # Products with missing fields
        cursor.execute('SELECT COUNT(*) FROM products WHERE product_name IS NULL')
        missing_name = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM products WHERE sku IS NULL')
        missing_sku = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM products WHERE category IS NULL')
        missing_category = cursor.fetchone()[0]
        
        # Products without features
        cursor.execute('''
            SELECT COUNT(*) FROM products p
            LEFT JOIN features f ON p.id = f.product_id
            WHERE f.id IS NULL
        ''')
        missing_features = cursor.fetchone()[0]
        
        # Duplicate SKUs
        cursor.execute('''
            SELECT COUNT(DISTINCT LOWER(sku)) FROM products
            WHERE sku IS NOT NULL
            GROUP BY LOWER(sku)
            HAVING COUNT(*) > 1
        ''')
        duplicate_sku_count = len(cursor.fetchall())
        
        # Historical vs new products
        cursor.execute('SELECT COUNT(*) FROM products WHERE is_historical = 1')
        historical_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM products WHERE is_historical = 0')
        new_count = cursor.fetchone()[0]
        
        # Categories
        cursor.execute('SELECT COUNT(DISTINCT category) FROM products WHERE category IS NOT NULL')
        category_count = cursor.fetchone()[0]
        
        return {
            'total_products': total_products,
            'historical_products': historical_count,
            'new_products': new_count,
            'missing_name': missing_name,
            'missing_sku': missing_sku,
            'missing_category': missing_category,
            'missing_features': missing_features,
            'duplicate_skus': duplicate_sku_count,
            'unique_categories': category_count,
            'completeness': {
                'name': round((total_products - missing_name) / total_products * 100, 1) if total_products > 0 else 0,
                'sku': round((total_products - missing_sku) / total_products * 100, 1) if total_products > 0 else 0,
                'category': round((total_products - missing_category) / total_products * 100, 1) if total_products > 0 else 0,
                'features': round((total_products - missing_features) / total_products * 100, 1) if total_products > 0 else 0
            }
        }

def validate_sku_format(sku: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Validate SKU format according to real-world data handling rules
    
    Args:
        sku: SKU string to validate (can be None)
    
    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid or None
        - (False, error_message) if invalid
    
    Rules:
        - NULL/None is valid (optional field)
        - Empty string is treated as NULL
        - Max length: 50 characters
        - Allowed characters: alphanumeric, hyphens, underscores
        - Whitespace is trimmed
    """
    import re
    
    # NULL/None is valid
    if sku is None:
        return True, None
    
    # Trim whitespace
    sku = sku.strip()
    
    # Empty string is treated as NULL (valid)
    if sku == '':
        return True, None
    
    # Check length
    if len(sku) > 50:
        return False, "SKU too long (max 50 characters)"
    
    # Check format: alphanumeric, hyphens, underscores only
    if not re.match(r'^[A-Za-z0-9\-_]+$', sku):
        return False, "Invalid SKU format. Use only letters, numbers, hyphens, and underscores."
    
    return True, None

def normalize_sku(sku: Optional[str]) -> Optional[str]:
    """Normalize SKU for consistent storage
    
    Args:
        sku: SKU string to normalize
    
    Returns:
        Normalized SKU or None
    
    Normalization:
        - Trim whitespace
        - Convert to uppercase for consistency
        - Empty string becomes None
    """
    if sku is None:
        return None
    
    sku = sku.strip().upper()
    
    if sku == '':
        return None
    
    return sku


# Price History Functions

def insert_price_history(product_id: int, date: str, price: float, currency: str = 'USD') -> int:
    """Insert a price history record for a product
    
    Args:
        product_id: ID of the product
        date: Date in YYYY-MM-DD format
        price: Price value
        currency: Currency code (default: USD)
    
    Returns:
        ID of inserted price history record
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO price_history (product_id, date, price, currency)
            VALUES (?, ?, ?, ?)
        ''', (product_id, date, price, currency))
        return cursor.lastrowid

def bulk_insert_price_history(product_id: int, price_records: List[Dict[str, Any]]) -> int:
    """Bulk insert price history records for a product
    
    Args:
        product_id: ID of the product
        price_records: List of dicts with 'date', 'price', and optional 'currency' keys
    
    Returns:
        Number of records inserted
    """
    if not price_records:
        return 0
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        records = []
        for record in price_records:
            date = record.get('date')
            price = record.get('price')
            currency = record.get('currency', 'USD')
            
            if date and price is not None:
                records.append((product_id, date, price, currency))
        
        if records:
            cursor.executemany('''
                INSERT INTO price_history (product_id, date, price, currency)
                VALUES (?, ?, ?, ?)
            ''', records)
            return len(records)
        
        return 0

def get_price_history(product_id: int, limit: int = 12) -> List[sqlite3.Row]:
    """Get price history for a product
    
    Args:
        product_id: ID of the product
        limit: Maximum number of records to return (default: 12 months)
    
    Returns:
        List of price history records ordered by date descending
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM price_history
            WHERE product_id = ?
            ORDER BY date DESC
            LIMIT ?
        ''', (product_id, limit))
        return cursor.fetchall()

def get_price_statistics(product_id: int) -> Optional[Dict[str, Any]]:
    """Calculate price statistics for a product
    
    Args:
        product_id: ID of the product
    
    Returns:
        Dictionary with min, max, average, current price, and trend
        Returns None if no price history exists
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get all prices ordered by date
        cursor.execute('''
            SELECT price, date FROM price_history
            WHERE product_id = ?
            ORDER BY date ASC
        ''', (product_id,))
        
        records = cursor.fetchall()
        
        if not records:
            return None
        
        prices = [r['price'] for r in records]
        
        # Calculate statistics
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        current_price = records[-1]['price']  # Most recent
        
        # Determine trend (compare last 2 prices if available)
        trend = 'stable'
        if len(records) >= 2:
            prev_price = records[-2]['price']
            if current_price > prev_price * 1.05:  # 5% threshold
                trend = 'up'
            elif current_price < prev_price * 0.95:
                trend = 'down'
        
        return {
            'min': round(min_price, 2),
            'max': round(max_price, 2),
            'average': round(avg_price, 2),
            'current': round(current_price, 2),
            'trend': trend,
            'data_points': len(records)
        }

def link_price_history(source_product_id: int, target_product_id: int) -> int:
    """Link price history from one product to another
    
    Used when matching products to automatically link historical price data
    
    Args:
        source_product_id: Product to copy price history from
        target_product_id: Product to copy price history to
    
    Returns:
        Number of price records linked
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get price history from source
        cursor.execute('''
            SELECT date, price, currency FROM price_history
            WHERE product_id = ?
            ORDER BY date DESC
            LIMIT 12
        ''', (source_product_id,))
        
        source_records = cursor.fetchall()
        
        if not source_records:
            return 0
        
        # Insert into target (avoid duplicates by checking date)
        inserted = 0
        for record in source_records:
            try:
                cursor.execute('''
                    INSERT INTO price_history (product_id, date, price, currency)
                    SELECT ?, ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM price_history
                        WHERE product_id = ? AND date = ?
                    )
                ''', (target_product_id, record['date'], record['price'], record['currency'],
                      target_product_id, record['date']))
                
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                logger.error(f"Error linking price record: {e}")
                continue
        
        return inserted

def delete_price_history(product_id: int) -> bool:
    """Delete all price history for a product
    
    Args:
        product_id: ID of the product
    
    Returns:
        True if any records were deleted
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM price_history WHERE product_id = ?', (product_id,))
        return cursor.rowcount > 0

def get_products_with_price_history(category: Optional[str] = None, 
                                   is_historical: Optional[bool] = None) -> List[sqlite3.Row]:
    """Get all products with their latest price from price history
    
    Args:
        category: Optional category filter
        is_historical: Optional filter for historical vs new products
    
    Returns:
        List of rows with product_id and price (latest price for each product)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if category is not None:
            conditions.append('p.category = ?')
            params.append(category)
        
        if is_historical is not None:
            conditions.append('p.is_historical = ?')
            params.append(is_historical)
        
        where_clause = 'WHERE ' + ' AND '.join(conditions) if conditions else ''
        
        # Get latest price for each product using subquery
        cursor.execute(f'''
            SELECT 
                p.id as product_id,
                ph.price
            FROM products p
            INNER JOIN price_history ph ON p.id = ph.product_id
            INNER JOIN (
                SELECT product_id, MAX(date) as max_date
                FROM price_history
                GROUP BY product_id
            ) latest ON ph.product_id = latest.product_id AND ph.date = latest.max_date
            {where_clause}
        ''', params)
        
        return cursor.fetchall()


# Performance History Functions

def insert_performance_history(product_id: int, date: str, sales: int = 0, 
                               views: int = 0, conversion_rate: float = 0.0, 
                               revenue: float = 0.0) -> int:
    """Insert a performance history record for a product
    
    Args:
        product_id: ID of the product
        date: Date in YYYY-MM-DD format
        sales: Number of sales
        views: Number of views
        conversion_rate: Conversion rate percentage (0-100)
        revenue: Total revenue
    
    Returns:
        ID of inserted performance history record
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO performance_history (product_id, date, sales, views, conversion_rate, revenue)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (product_id, date, sales, views, conversion_rate, revenue))
        return cursor.lastrowid

def bulk_insert_performance_history(product_id: int, performance_records: List[Dict[str, Any]]) -> int:
    """Bulk insert performance history records for a product
    
    Args:
        product_id: ID of the product
        performance_records: List of dicts with 'date', 'sales', 'views', 'conversion_rate', 'revenue' keys
    
    Returns:
        Number of records inserted
    """
    if not performance_records:
        return 0
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        records = []
        for record in performance_records:
            date = record.get('date')
            sales = record.get('sales', 0)
            views = record.get('views', 0)
            conversion_rate = record.get('conversion_rate', 0.0)
            revenue = record.get('revenue', 0.0)
            
            if date:
                records.append((product_id, date, sales, views, conversion_rate, revenue))
        
        if records:
            cursor.executemany('''
                INSERT INTO performance_history (product_id, date, sales, views, conversion_rate, revenue)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', records)
            return len(records)
        
        return 0

def get_performance_history(product_id: int, limit: int = 12) -> List[sqlite3.Row]:
    """Get performance history for a product
    
    Args:
        product_id: ID of the product
        limit: Maximum number of records to return (default: 12 months)
    
    Returns:
        List of performance history records ordered by date descending
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM performance_history
            WHERE product_id = ?
            ORDER BY date DESC
            LIMIT ?
        ''', (product_id, limit))
        return cursor.fetchall()

def get_performance_statistics(product_id: int) -> Optional[Dict[str, Any]]:
    """Calculate performance statistics for a product
    
    Args:
        product_id: ID of the product
    
    Returns:
        Dictionary with total sales, avg conversion, total revenue, trends
        Returns None if no performance history exists
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get all performance records ordered by date
        cursor.execute('''
            SELECT sales, views, conversion_rate, revenue, date 
            FROM performance_history
            WHERE product_id = ?
            ORDER BY date ASC
        ''', (product_id,))
        
        records = cursor.fetchall()
        
        if not records:
            return None
        
        # Calculate statistics
        total_sales = sum(r['sales'] for r in records)
        total_views = sum(r['views'] for r in records)
        total_revenue = sum(r['revenue'] for r in records)
        avg_conversion = sum(r['conversion_rate'] for r in records) / len(records)
        
        # Calculate trends (compare last 2 records if available)
        sales_trend = 'stable'
        conversion_trend = 'stable'
        
        if len(records) >= 2:
            prev_sales = records[-2]['sales']
            current_sales = records[-1]['sales']
            if current_sales > prev_sales * 1.1:  # 10% threshold
                sales_trend = 'up'
            elif current_sales < prev_sales * 0.9:
                sales_trend = 'down'
            
            prev_conversion = records[-2]['conversion_rate']
            current_conversion = records[-1]['conversion_rate']
            if current_conversion > prev_conversion * 1.05:  # 5% threshold
                conversion_trend = 'up'
            elif current_conversion < prev_conversion * 0.95:
                conversion_trend = 'down'
        
        return {
            'total_sales': total_sales,
            'total_views': total_views,
            'total_revenue': round(total_revenue, 2),
            'avg_conversion': round(avg_conversion, 2),
            'avg_sales_per_period': round(total_sales / len(records), 1),
            'sales_trend': sales_trend,
            'conversion_trend': conversion_trend,
            'data_points': len(records)
        }

def link_performance_history(source_product_id: int, target_product_id: int) -> int:
    """Link performance history from one product to another
    
    Used when matching products to automatically link historical performance data
    
    Args:
        source_product_id: Product to copy performance history from
        target_product_id: Product to copy performance history to
    
    Returns:
        Number of performance records linked
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get performance history from source
        cursor.execute('''
            SELECT date, sales, views, conversion_rate, revenue 
            FROM performance_history
            WHERE product_id = ?
            ORDER BY date DESC
            LIMIT 12
        ''', (source_product_id,))
        
        source_records = cursor.fetchall()
        
        if not source_records:
            return 0
        
        # Insert into target (avoid duplicates by checking date)
        inserted = 0
        for record in source_records:
            try:
                cursor.execute('''
                    INSERT INTO performance_history (product_id, date, sales, views, conversion_rate, revenue)
                    SELECT ?, ?, ?, ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM performance_history
                        WHERE product_id = ? AND date = ?
                    )
                ''', (target_product_id, record['date'], record['sales'], record['views'], 
                      record['conversion_rate'], record['revenue'],
                      target_product_id, record['date']))
                
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                logger.error(f"Error linking performance record: {e}")
                continue
        
        return inserted

def delete_performance_history(product_id: int) -> bool:
    """Delete all performance history for a product
    
    Args:
        product_id: ID of the product
    
    Returns:
        True if any records were deleted
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM performance_history WHERE product_id = ?', (product_id,))
        return cursor.rowcount > 0

def get_products_with_performance_history(category: Optional[str] = None, 
                                         is_historical: Optional[bool] = None) -> List[sqlite3.Row]:
    """Get all products with their latest performance metrics from performance history
    
    Args:
        category: Optional category filter
        is_historical: Optional filter for historical vs new products
    
    Returns:
        List of rows with product_id, sales, views, conversion_rate, revenue (latest for each product)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        if category is not None:
            conditions.append('p.category = ?')
            params.append(category)
        
        if is_historical is not None:
            conditions.append('p.is_historical = ?')
            params.append(is_historical)
        
        where_clause = 'WHERE ' + ' AND '.join(conditions) if conditions else ''
        
        # Get latest performance metrics for each product using subquery
        cursor.execute(f'''
            SELECT 
                p.id as product_id,
                perf.sales,
                perf.views,
                perf.conversion_rate,
                perf.revenue
            FROM products p
            INNER JOIN performance_history perf ON p.id = perf.product_id
            INNER JOIN (
                SELECT product_id, MAX(date) as max_date
                FROM performance_history
                GROUP BY product_id
            ) latest ON perf.product_id = latest.product_id AND perf.date = latest.max_date
            {where_clause}
        ''', params)
        
        return cursor.fetchall()



def get_all_products(is_historical=None):
    """
    Get all products, optionally filtered by is_historical flag.
    
    Args:
        is_historical: Filter by historical flag (None = all products)
    
    Returns:
        List of product dictionaries
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            if is_historical is None:
                cursor.execute('''
                    SELECT id, image_path, category, product_name, sku, is_historical, created_at, metadata
                    FROM products
                    ORDER BY created_at DESC
                ''')
            else:
                cursor.execute('''
                    SELECT id, image_path, category, product_name, sku, is_historical, created_at, metadata
                    FROM products
                    WHERE is_historical = ?
                    ORDER BY created_at DESC
                ''', (1 if is_historical else 0,))
            
            products = cursor.fetchall()
            return products
            
        except Exception as e:
            logger.error(f"Error getting all products: {e}")
            raise
