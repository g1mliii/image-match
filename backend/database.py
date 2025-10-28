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
