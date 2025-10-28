import sqlite3
import os
from contextlib import contextmanager

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
    """Initialize database with schema"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                category TEXT NOT NULL,
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
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_products_category 
            ON products(category)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_products_is_historical 
            ON products(is_historical)
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

def insert_product(image_path, category, product_name=None, sku=None, is_historical=False, metadata=None):
    """Insert a new product"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO products (image_path, category, product_name, sku, is_historical, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_path, category, product_name, sku, is_historical, metadata))
        return cursor.lastrowid

def get_historical_products(category=None, limit=100, offset=0):
    """Get historical products with optional category filter"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if category:
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
