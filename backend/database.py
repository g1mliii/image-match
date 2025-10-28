import sqlite3
import os
import numpy as np
import io
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple

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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                color_features BLOB NOT NULL,
                shape_features BLOB NOT NULL,
                texture_features BLOB NOT NULL,
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
        
        # Create indexes for performance
        # Note: Indexes on nullable columns still work, NULL values are indexed
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_products_category 
            ON products(category)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_products_is_historical 
            ON products(is_historical)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_products_category_historical 
            ON products(category, is_historical)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_matches_new_product 
            ON matches(new_product_id, similarity_score DESC)
        ''')
        
        conn.commit()
        print("Database initialized successfully")

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
                    shape_features: np.ndarray, texture_features: np.ndarray) -> int:
    """Insert feature vectors for a product"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Serialize numpy arrays to bytes
        color_blob = serialize_numpy_array(color_features)
        shape_blob = serialize_numpy_array(shape_features)
        texture_blob = serialize_numpy_array(texture_features)
        
        cursor.execute('''
            INSERT INTO features (product_id, color_features, shape_features, texture_features)
            VALUES (?, ?, ?, ?)
        ''', (product_id, color_blob, shape_blob, texture_blob))
        
        return cursor.lastrowid

def get_features_by_product_id(product_id: int) -> Optional[Dict[str, np.ndarray]]:
    """Get feature vectors for a product"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT color_features, shape_features, texture_features 
            FROM features 
            WHERE product_id = ?
        ''', (product_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'color_features': deserialize_numpy_array(row['color_features']),
            'shape_features': deserialize_numpy_array(row['shape_features']),
            'texture_features': deserialize_numpy_array(row['texture_features'])
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
                                include_uncategorized: bool = False) -> List[Tuple[int, Dict[str, np.ndarray]]]:
    """Get all feature vectors for products in a category
    
    Args:
        category: Category to filter by (None returns all products)
        is_historical: Filter for historical vs new products
        include_uncategorized: If True and category specified, also include NULL category products
    
    Returns:
        List of tuples (product_id, features_dict)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if category is None:
            # Get all products regardless of category
            cursor.execute('''
                SELECT p.id, p.category, f.color_features, f.shape_features, f.texture_features
                FROM products p
                JOIN features f ON p.id = f.product_id
                WHERE p.is_historical = ?
            ''', (is_historical,))
        elif include_uncategorized:
            # Get products in category OR with NULL category
            cursor.execute('''
                SELECT p.id, p.category, f.color_features, f.shape_features, f.texture_features
                FROM products p
                JOIN features f ON p.id = f.product_id
                WHERE (p.category = ? OR p.category IS NULL) AND p.is_historical = ?
            ''', (category, is_historical))
        else:
            # Get products in specific category only
            cursor.execute('''
                SELECT p.id, p.category, f.color_features, f.shape_features, f.texture_features
                FROM products p
                JOIN features f ON p.id = f.product_id
                WHERE p.category = ? AND p.is_historical = ?
            ''', (category, is_historical))
        
        results = []
        for row in cursor.fetchall():
            product_id = row['id']
            features = {
                'color_features': deserialize_numpy_array(row['color_features']),
                'shape_features': deserialize_numpy_array(row['shape_features']),
                'texture_features': deserialize_numpy_array(row['texture_features']),
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
