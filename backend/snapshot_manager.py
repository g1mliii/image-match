"""
Catalog Snapshot Management System

This module provides functionality for managing multiple catalog snapshots,
allowing users to create, manage, and combine multiple catalog databases.

Each snapshot is an independent SQLite database with the same schema as the
main product_matching.db, plus a snapshot_metadata table for tracking.
"""

import sqlite3
import os
import json
import shutil
import zipfile
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Directory paths
BACKEND_DIR = os.path.dirname(__file__)
CATALOGS_DIR = os.path.join(BACKEND_DIR, 'catalogs')
CONFIG_DIR = os.path.join(BACKEND_DIR, 'config')
ACTIVE_CATALOGS_FILE = os.path.join(CONFIG_DIR, 'active_catalogs.json')
DEFAULT_DB_PATH = os.path.join(BACKEND_DIR, 'product_matching.db')

# Ensure directories exist
os.makedirs(CATALOGS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


def sanitize_snapshot_name(name: str) -> str:
    """Sanitize snapshot name for use as filename
    
    Args:
        name: User-provided snapshot name
        
    Returns:
        Sanitized name safe for filesystem
    """
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
    sanitized = sanitized.strip()
    sanitized = re.sub(r'\s+', '-', sanitized)  # Replace spaces with hyphens
    sanitized = sanitized[:100]  # Max length
    
    if not sanitized:
        sanitized = f"snapshot-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    return sanitized


def get_snapshot_db_path(snapshot_name: str) -> str:
    """Get full path to snapshot database file"""
    if not snapshot_name.endswith('.db'):
        snapshot_name = f"{snapshot_name}.db"
    return os.path.join(CATALOGS_DIR, snapshot_name)


def get_snapshot_uploads_dir(snapshot_name: str) -> str:
    """Get path to snapshot's uploads directory"""
    base_name = snapshot_name.replace('.db', '')
    return os.path.join(CATALOGS_DIR, base_name, 'uploads')


@contextmanager
def get_snapshot_connection(snapshot_path: str):
    """Context manager for snapshot database connections"""
    conn = sqlite3.connect(snapshot_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_snapshot_db(snapshot_path: str, name: str, is_historical: bool = True,
                     description: str = None, tags: List[str] = None) -> bool:
    """Initialize a new snapshot database with full schema
    
    Args:
        snapshot_path: Path to the new database file
        name: Display name for the snapshot
        is_historical: Whether this is a historical catalog
        description: Optional description
        tags: Optional list of tags
        
    Returns:
        True if successful
    """
    try:
        with get_snapshot_connection(snapshot_path) as conn:
            cursor = conn.cursor()
            
            # Create snapshot_metadata table (unique to snapshots)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS snapshot_metadata (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version TEXT DEFAULT '1.0',
                    description TEXT,
                    product_count INTEGER DEFAULT 0,
                    is_historical BOOLEAN DEFAULT 1,
                    tags TEXT,
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert metadata
            cursor.execute('''
                INSERT INTO snapshot_metadata (name, is_historical, description, tags)
                VALUES (?, ?, ?, ?)
            ''', (name, is_historical, description, json.dumps(tags or [])))
            
            # Create products table
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
            
            # Create price_history table
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
            
            # Create performance_history table
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
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_products_is_historical ON products(is_historical)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_products_category_historical ON products(category, is_historical)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_product_id ON features(product_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_new_product ON matches(new_product_id, similarity_score DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_history_product_id ON price_history(product_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_history_product_id ON performance_history(product_id)')
            
            conn.commit()
            
        # Create uploads directory for this snapshot
        uploads_dir = get_snapshot_uploads_dir(snapshot_path)
        os.makedirs(uploads_dir, exist_ok=True)
        
        logger.info(f"Initialized snapshot database: {snapshot_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize snapshot database: {e}")
        return False


def create_snapshot(name: str, is_historical: bool = True, 
                   description: str = None, tags: List[str] = None) -> Dict[str, Any]:
    """Create a new empty snapshot
    
    Args:
        name: Display name for the snapshot
        is_historical: Whether this is a historical catalog
        description: Optional description
        tags: Optional list of tags
        
    Returns:
        Dictionary with snapshot info or error
    """
    try:
        sanitized_name = sanitize_snapshot_name(name)
        db_filename = f"{sanitized_name}.db"
        db_path = get_snapshot_db_path(sanitized_name)
        
        # Check if already exists
        if os.path.exists(db_path):
            return {'error': f'Snapshot "{sanitized_name}" already exists'}
        
        # Initialize the database
        if not init_snapshot_db(db_path, name, is_historical, description, tags):
            return {'error': 'Failed to initialize snapshot database'}
        
        return {
            'success': True,
            'snapshot_file': db_filename,
            'name': name,
            'path': db_path,
            'is_historical': is_historical
        }
        
    except Exception as e:
        logger.error(f"Error creating snapshot: {e}")
        return {'error': str(e)}


def list_snapshots() -> Dict[str, Any]:
    """List all available snapshots with metadata
    
    Returns:
        Dictionary with historical and new snapshot lists
    """
    try:
        historical = []
        new_products = []
        
        if not os.path.exists(CATALOGS_DIR):
            return {'historical': [], 'new': []}
        
        for filename in os.listdir(CATALOGS_DIR):
            if filename.endswith('.db'):
                db_path = os.path.join(CATALOGS_DIR, filename)
                info = get_snapshot_info(filename)
                
                if info and not info.get('error'):
                    if info.get('is_historical', True):
                        historical.append(info)
                    else:
                        new_products.append(info)
        
        # Sort by created_at descending
        historical.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        new_products.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return {
            'historical': historical,
            'new': new_products
        }
        
    except Exception as e:
        logger.error(f"Error listing snapshots: {e}")
        return {'error': str(e), 'historical': [], 'new': []}


def get_snapshot_info(snapshot_file: str) -> Optional[Dict[str, Any]]:
    """Get metadata and stats for a specific snapshot
    
    Args:
        snapshot_file: Snapshot filename (with or without .db extension)
        
    Returns:
        Dictionary with snapshot info or None
    """
    try:
        if not snapshot_file.endswith('.db'):
            snapshot_file = f"{snapshot_file}.db"
            
        db_path = os.path.join(CATALOGS_DIR, snapshot_file)
        
        if not os.path.exists(db_path):
            return {'error': f'Snapshot not found: {snapshot_file}'}
        
        with get_snapshot_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Get metadata
            cursor.execute('SELECT * FROM snapshot_metadata LIMIT 1')
            metadata_row = cursor.fetchone()
            
            if not metadata_row:
                # Legacy database without metadata - create default
                name = snapshot_file.replace('.db', '')
                metadata = {
                    'name': name,
                    'created_at': None,
                    'version': '1.0',
                    'description': None,
                    'is_historical': True,
                    'tags': []
                }
            else:
                metadata = {
                    'name': metadata_row['name'],
                    'created_at': metadata_row['created_at'],
                    'version': metadata_row['version'],
                    'description': metadata_row['description'],
                    'is_historical': bool(metadata_row['is_historical']),
                    'tags': json.loads(metadata_row['tags'] or '[]'),
                    'last_modified': metadata_row['last_modified']
                }
            
            # Get product count
            cursor.execute('SELECT COUNT(*) FROM products')
            product_count = cursor.fetchone()[0]
            
            # Get category breakdown
            cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM products 
                GROUP BY category 
                ORDER BY count DESC
            ''')
            categories = [{'category': row['category'], 'count': row['count']} 
                         for row in cursor.fetchall()]
        
        # Get file size
        size_bytes = os.path.getsize(db_path)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        
        # Get uploads folder size
        uploads_dir = get_snapshot_uploads_dir(db_path)
        uploads_size_mb = 0
        if os.path.exists(uploads_dir):
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, filenames in os.walk(uploads_dir)
                for f in filenames
            )
            uploads_size_mb = round(total_size / (1024 * 1024), 2)
        
        return {
            'snapshot_file': snapshot_file,
            'path': db_path,
            **metadata,
            'product_count': product_count,
            'size_mb': size_mb,
            'uploads_size_mb': uploads_size_mb,
            'total_size_mb': round(size_mb + uploads_size_mb, 2),
            'categories': categories
        }
        
    except Exception as e:
        logger.error(f"Error getting snapshot info for {snapshot_file}: {e}")
        return {'error': str(e), 'snapshot_file': snapshot_file}


def delete_snapshot(snapshot_file: str) -> Dict[str, Any]:
    """Delete a snapshot and its associated files
    
    Args:
        snapshot_file: Snapshot filename
        
    Returns:
        Dictionary with result
    """
    try:
        if not snapshot_file.endswith('.db'):
            snapshot_file = f"{snapshot_file}.db"
            
        db_path = os.path.join(CATALOGS_DIR, snapshot_file)
        
        if not os.path.exists(db_path):
            return {'error': f'Snapshot not found: {snapshot_file}'}
        
        # Check if snapshot is currently active
        active = get_active_catalogs()
        all_active = active.get('active_historical', []) + active.get('active_new', [])
        if snapshot_file in all_active:
            return {'error': 'Cannot delete active snapshot. Deselect it first.'}
        
        # Delete database file
        os.remove(db_path)
        
        # Delete uploads directory if exists
        uploads_dir = get_snapshot_uploads_dir(db_path)
        if os.path.exists(uploads_dir):
            shutil.rmtree(uploads_dir)
        
        # Also try to delete parent directory if empty
        parent_dir = os.path.dirname(uploads_dir)
        if os.path.exists(parent_dir) and not os.listdir(parent_dir):
            os.rmdir(parent_dir)
        
        logger.info(f"Deleted snapshot: {snapshot_file}")
        return {'success': True, 'deleted': snapshot_file}
        
    except Exception as e:
        logger.error(f"Error deleting snapshot {snapshot_file}: {e}")
        return {'error': str(e)}


def rename_snapshot(old_name: str, new_name: str) -> Dict[str, Any]:
    """Rename a snapshot
    
    Args:
        old_name: Current snapshot filename
        new_name: New display name
        
    Returns:
        Dictionary with result
    """
    try:
        if not old_name.endswith('.db'):
            old_name = f"{old_name}.db"
            
        old_path = os.path.join(CATALOGS_DIR, old_name)
        
        if not os.path.exists(old_path):
            return {'error': f'Snapshot not found: {old_name}'}
        
        sanitized_new = sanitize_snapshot_name(new_name)
        new_filename = f"{sanitized_new}.db"
        new_path = os.path.join(CATALOGS_DIR, new_filename)
        
        if os.path.exists(new_path) and old_path != new_path:
            return {'error': f'Snapshot "{sanitized_new}" already exists'}
        
        # Update metadata in database
        with get_snapshot_connection(old_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE snapshot_metadata 
                SET name = ?, last_modified = CURRENT_TIMESTAMP
            ''', (new_name,))
        
        # Rename file if needed
        if old_path != new_path:
            os.rename(old_path, new_path)
            
            # Rename uploads directory if exists
            old_uploads = get_snapshot_uploads_dir(old_path)
            new_uploads = get_snapshot_uploads_dir(new_path)
            if os.path.exists(old_uploads):
                os.makedirs(os.path.dirname(new_uploads), exist_ok=True)
                shutil.move(old_uploads, new_uploads)
            
            # Update active catalogs config if needed
            active = get_active_catalogs()
            updated = False
            
            if old_name in active.get('active_historical', []):
                active['active_historical'] = [
                    new_filename if f == old_name else f 
                    for f in active['active_historical']
                ]
                updated = True
                
            if old_name in active.get('active_new', []):
                active['active_new'] = [
                    new_filename if f == old_name else f 
                    for f in active['active_new']
                ]
                updated = True
            
            if updated:
                set_active_catalogs(active['active_historical'], active['active_new'])
        
        logger.info(f"Renamed snapshot {old_name} to {new_filename}")
        return {
            'success': True,
            'old_name': old_name,
            'new_name': new_filename,
            'display_name': new_name
        }
        
    except Exception as e:
        logger.error(f"Error renaming snapshot: {e}")
        return {'error': str(e)}


def merge_snapshots(snapshot_files: List[str], new_name: str, 
                   is_historical: bool = True) -> Dict[str, Any]:
    """Merge multiple snapshots into a new one
    
    Args:
        snapshot_files: List of snapshot filenames to merge
        new_name: Name for the merged snapshot
        is_historical: Whether the merged snapshot is historical
        
    Returns:
        Dictionary with result
    """
    try:
        if len(snapshot_files) < 2:
            return {'error': 'Need at least 2 snapshots to merge'}
        
        # Create new snapshot
        result = create_snapshot(
            name=new_name,
            is_historical=is_historical,
            description=f"Merged from: {', '.join(snapshot_files)}",
            tags=['merged']
        )
        
        if result.get('error'):
            return result
        
        new_db_path = result['path']
        new_uploads_dir = get_snapshot_uploads_dir(new_db_path)
        os.makedirs(new_uploads_dir, exist_ok=True)
        
        total_products = 0
        
        with get_snapshot_connection(new_db_path) as new_conn:
            new_cursor = new_conn.cursor()
            
            for snapshot_file in snapshot_files:
                if not snapshot_file.endswith('.db'):
                    snapshot_file = f"{snapshot_file}.db"
                    
                source_path = os.path.join(CATALOGS_DIR, snapshot_file)
                
                if not os.path.exists(source_path):
                    logger.warning(f"Snapshot not found during merge: {snapshot_file}")
                    continue
                
                with get_snapshot_connection(source_path) as source_conn:
                    source_cursor = source_conn.cursor()
                    
                    # Copy products
                    source_cursor.execute('SELECT * FROM products')
                    products = source_cursor.fetchall()
                    
                    for product in products:
                        # Copy image file if exists
                        old_image_path = product['image_path']
                        new_image_path = old_image_path
                        
                        if old_image_path and os.path.exists(old_image_path):
                            filename = os.path.basename(old_image_path)
                            new_image_path = os.path.join(new_uploads_dir, filename)
                            
                            # Handle duplicate filenames
                            counter = 1
                            while os.path.exists(new_image_path):
                                name, ext = os.path.splitext(filename)
                                new_image_path = os.path.join(
                                    new_uploads_dir, f"{name}_{counter}{ext}"
                                )
                                counter += 1
                            
                            shutil.copy2(old_image_path, new_image_path)
                        
                        # Insert product
                        new_cursor.execute('''
                            INSERT INTO products 
                            (image_path, category, product_name, sku, is_historical, metadata)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            new_image_path,
                            product['category'],
                            product['product_name'],
                            product['sku'],
                            is_historical,
                            product['metadata']
                        ))
                        
                        new_product_id = new_cursor.lastrowid
                        old_product_id = product['id']
                        
                        # Copy features
                        source_cursor.execute(
                            'SELECT * FROM features WHERE product_id = ?',
                            (old_product_id,)
                        )
                        features = source_cursor.fetchone()
                        
                        if features:
                            new_cursor.execute('''
                                INSERT INTO features 
                                (product_id, color_features, shape_features, texture_features,
                                 embedding_type, embedding_version)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (
                                new_product_id,
                                features['color_features'],
                                features['shape_features'],
                                features['texture_features'],
                                features['embedding_type'],
                                features['embedding_version']
                            ))
                        
                        # Copy price history
                        source_cursor.execute(
                            'SELECT * FROM price_history WHERE product_id = ?',
                            (old_product_id,)
                        )
                        for price in source_cursor.fetchall():
                            new_cursor.execute('''
                                INSERT INTO price_history 
                                (product_id, date, price, currency)
                                VALUES (?, ?, ?, ?)
                            ''', (new_product_id, price['date'], price['price'], price['currency']))
                        
                        # Copy performance history
                        source_cursor.execute(
                            'SELECT * FROM performance_history WHERE product_id = ?',
                            (old_product_id,)
                        )
                        for perf in source_cursor.fetchall():
                            new_cursor.execute('''
                                INSERT INTO performance_history 
                                (product_id, date, sales, views, conversion_rate, revenue)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (
                                new_product_id, perf['date'], perf['sales'],
                                perf['views'], perf['conversion_rate'], perf['revenue']
                            ))
                        
                        total_products += 1
            
            # Update product count in metadata
            new_cursor.execute('''
                UPDATE snapshot_metadata SET product_count = ?
            ''', (total_products,))
        
        logger.info(f"Merged {len(snapshot_files)} snapshots into {new_name} ({total_products} products)")
        
        return {
            'success': True,
            'snapshot_file': result['snapshot_file'],
            'products_merged': total_products,
            'source_snapshots': snapshot_files
        }
        
    except Exception as e:
        logger.error(f"Error merging snapshots: {e}")
        return {'error': str(e)}


def export_snapshot(snapshot_file: str, output_path: str = None) -> Dict[str, Any]:
    """Export a snapshot as a .zip file
    
    Args:
        snapshot_file: Snapshot filename
        output_path: Optional output path (defaults to catalogs directory)
        
    Returns:
        Dictionary with result including zip path
    """
    try:
        if not snapshot_file.endswith('.db'):
            snapshot_file = f"{snapshot_file}.db"
            
        db_path = os.path.join(CATALOGS_DIR, snapshot_file)
        
        if not os.path.exists(db_path):
            return {'error': f'Snapshot not found: {snapshot_file}'}
        
        base_name = snapshot_file.replace('.db', '')
        
        if output_path is None:
            output_path = os.path.join(CATALOGS_DIR, f"{base_name}-export.zip")
        
        uploads_dir = get_snapshot_uploads_dir(db_path)
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add database file
            zipf.write(db_path, snapshot_file)
            
            # Add uploads directory if exists
            if os.path.exists(uploads_dir):
                for root, dirs, files in os.walk(uploads_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join('uploads', file)
                        zipf.write(file_path, arcname)
        
        zip_size = os.path.getsize(output_path)
        
        logger.info(f"Exported snapshot {snapshot_file} to {output_path}")
        
        return {
            'success': True,
            'zip_path': output_path,
            'size_mb': round(zip_size / (1024 * 1024), 2)
        }
        
    except Exception as e:
        logger.error(f"Error exporting snapshot: {e}")
        return {'error': str(e)}


def import_snapshot(zip_path: str) -> Dict[str, Any]:
    """Import a snapshot from a .zip file
    
    Args:
        zip_path: Path to the zip file
        
    Returns:
        Dictionary with result
    """
    try:
        if not os.path.exists(zip_path):
            return {'error': f'Zip file not found: {zip_path}'}
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            # Find the .db file
            db_files = [f for f in zipf.namelist() if f.endswith('.db')]
            
            if not db_files:
                return {'error': 'No database file found in zip'}
            
            db_filename = db_files[0]
            base_name = db_filename.replace('.db', '')
            
            # Check if already exists
            target_db_path = os.path.join(CATALOGS_DIR, db_filename)
            if os.path.exists(target_db_path):
                # Generate unique name
                counter = 1
                while os.path.exists(target_db_path):
                    new_name = f"{base_name}-imported-{counter}.db"
                    target_db_path = os.path.join(CATALOGS_DIR, new_name)
                    counter += 1
                db_filename = os.path.basename(target_db_path)
                base_name = db_filename.replace('.db', '')
            
            # Extract database
            with zipf.open(db_files[0]) as source:
                with open(target_db_path, 'wb') as target:
                    target.write(source.read())
            
            # Extract uploads
            uploads_dir = get_snapshot_uploads_dir(target_db_path)
            os.makedirs(uploads_dir, exist_ok=True)
            
            for file_info in zipf.namelist():
                if file_info.startswith('uploads/') and not file_info.endswith('/'):
                    filename = os.path.basename(file_info)
                    target_path = os.path.join(uploads_dir, filename)
                    
                    with zipf.open(file_info) as source:
                        with open(target_path, 'wb') as target:
                            target.write(source.read())
        
        # Get info about imported snapshot
        info = get_snapshot_info(db_filename)
        
        logger.info(f"Imported snapshot from {zip_path} as {db_filename}")
        
        return {
            'success': True,
            'snapshot_file': db_filename,
            'info': info
        }
        
    except Exception as e:
        logger.error(f"Error importing snapshot: {e}")
        return {'error': str(e)}


def get_active_catalogs() -> Dict[str, List[str]]:
    """Get currently active catalog snapshots
    
    Returns:
        Dictionary with active_historical and active_new lists
    """
    try:
        if os.path.exists(ACTIVE_CATALOGS_FILE):
            with open(ACTIVE_CATALOGS_FILE, 'r') as f:
                return json.load(f)
        return {'active_historical': [], 'active_new': []}
    except Exception as e:
        logger.error(f"Error reading active catalogs: {e}")
        return {'active_historical': [], 'active_new': []}


def set_active_catalogs(historical_list: List[str], new_list: List[str]) -> Dict[str, Any]:
    """Set active catalog snapshots
    
    Args:
        historical_list: List of historical snapshot filenames
        new_list: List of new product snapshot filenames
        
    Returns:
        Dictionary with result
    """
    try:
        # Validate that all snapshots exist
        for snapshot in historical_list + new_list:
            if not snapshot.endswith('.db'):
                snapshot = f"{snapshot}.db"
            path = os.path.join(CATALOGS_DIR, snapshot)
            if not os.path.exists(path):
                return {'error': f'Snapshot not found: {snapshot}'}
        
        config = {
            'active_historical': historical_list,
            'active_new': new_list
        }
        
        with open(ACTIVE_CATALOGS_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Set active catalogs: {len(historical_list)} historical, {len(new_list)} new")
        
        return {'success': True, **config}
        
    except Exception as e:
        logger.error(f"Error setting active catalogs: {e}")
        return {'error': str(e)}


def update_snapshot_version(snapshot_file: str) -> str:
    """Increment snapshot version after modification
    
    Args:
        snapshot_file: Snapshot filename
        
    Returns:
        New version string
    """
    try:
        if not snapshot_file.endswith('.db'):
            snapshot_file = f"{snapshot_file}.db"
            
        db_path = os.path.join(CATALOGS_DIR, snapshot_file)
        
        with get_snapshot_connection(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT version FROM snapshot_metadata LIMIT 1')
            row = cursor.fetchone()
            
            if row:
                current = row['version'] or '1.0'
                parts = current.split('.')
                if len(parts) >= 2:
                    major, minor = int(parts[0]), int(parts[1])
                    new_version = f"{major}.{minor + 1}"
                else:
                    new_version = '1.1'
                
                cursor.execute('''
                    UPDATE snapshot_metadata 
                    SET version = ?, last_modified = CURRENT_TIMESTAMP
                ''', (new_version,))
                
                return new_version
        
        return '1.0'
        
    except Exception as e:
        logger.error(f"Error updating snapshot version: {e}")
        return '1.0'


def update_snapshot_product_count(snapshot_file: str) -> int:
    """Update product count in snapshot metadata
    
    Args:
        snapshot_file: Snapshot filename
        
    Returns:
        New product count
    """
    try:
        if not snapshot_file.endswith('.db'):
            snapshot_file = f"{snapshot_file}.db"
            
        db_path = os.path.join(CATALOGS_DIR, snapshot_file)
        
        with get_snapshot_connection(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM products')
            count = cursor.fetchone()[0]
            
            cursor.execute('''
                UPDATE snapshot_metadata 
                SET product_count = ?, last_modified = CURRENT_TIMESTAMP
            ''', (count,))
            
            return count
        
    except Exception as e:
        logger.error(f"Error updating product count: {e}")
        return 0


def migrate_legacy_database() -> Dict[str, Any]:
    """Migrate existing product_matching.db to snapshot system
    
    Returns:
        Dictionary with migration result
    """
    try:
        if not os.path.exists(DEFAULT_DB_PATH):
            return {'migrated': False, 'reason': 'No legacy database found'}
        
        # Check if already migrated
        default_snapshot = os.path.join(CATALOGS_DIR, 'default-catalog.db')
        if os.path.exists(default_snapshot):
            return {'migrated': False, 'reason': 'Already migrated'}
        
        # Copy database to catalogs directory
        shutil.copy2(DEFAULT_DB_PATH, default_snapshot)
        
        # Add metadata table
        with get_snapshot_connection(default_snapshot) as conn:
            cursor = conn.cursor()
            
            # Check if metadata table exists
            cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='snapshot_metadata'
            ''')
            
            if not cursor.fetchone():
                cursor.execute('''
                    CREATE TABLE snapshot_metadata (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        version TEXT DEFAULT '1.0',
                        description TEXT,
                        product_count INTEGER DEFAULT 0,
                        is_historical BOOLEAN DEFAULT 1,
                        tags TEXT,
                        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Get product count
                cursor.execute('SELECT COUNT(*) FROM products')
                count = cursor.fetchone()[0]
                
                cursor.execute('''
                    INSERT INTO snapshot_metadata 
                    (name, description, product_count, is_historical, tags)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    'Default Catalog',
                    'Migrated from legacy database',
                    count,
                    True,
                    json.dumps(['migrated', 'legacy'])
                ))
        
        # Set as active
        set_active_catalogs(['default-catalog.db'], [])
        
        logger.info("Migrated legacy database to snapshot system")
        
        return {
            'migrated': True,
            'snapshot_file': 'default-catalog.db',
            'message': 'Your catalog has been migrated to the new snapshot system'
        }
        
    except Exception as e:
        logger.error(f"Error migrating legacy database: {e}")
        return {'error': str(e)}


def get_combined_products_count() -> Dict[str, int]:
    """Get total product counts from all active snapshots
    
    Returns:
        Dictionary with historical and new product counts
    """
    active = get_active_catalogs()
    historical_count = 0
    new_count = 0
    
    for snapshot_file in active.get('active_historical', []):
        info = get_snapshot_info(snapshot_file)
        if info and not info.get('error'):
            historical_count += info.get('product_count', 0)
    
    for snapshot_file in active.get('active_new', []):
        info = get_snapshot_info(snapshot_file)
        if info and not info.get('error'):
            new_count += info.get('product_count', 0)
    
    return {
        'historical_count': historical_count,
        'new_count': new_count,
        'total': historical_count + new_count
    }


# ============ Main Database Integration (Option C) ============

def save_main_db_as_snapshot(name: str, description: str = None, 
                             tags: List[str] = None) -> Dict[str, Any]:
    """Save the current main database as a new snapshot
    
    This copies product_matching.db to a new snapshot file in catalogs/
    
    Args:
        name: Display name for the snapshot
        description: Optional description
        tags: Optional list of tags
        
    Returns:
        Dictionary with result
    """
    try:
        if not os.path.exists(DEFAULT_DB_PATH):
            return {'error': 'No main database found to save'}
        
        sanitized_name = sanitize_snapshot_name(name)
        db_filename = f"{sanitized_name}.db"
        db_path = os.path.join(CATALOGS_DIR, db_filename)
        
        # Check if already exists
        if os.path.exists(db_path):
            return {'error': f'Snapshot "{sanitized_name}" already exists'}
        
        # Copy the main database
        shutil.copy2(DEFAULT_DB_PATH, db_path)
        
        # Add/update metadata table
        with get_snapshot_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if metadata table exists
            cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='snapshot_metadata'
            ''')
            
            if not cursor.fetchone():
                cursor.execute('''
                    CREATE TABLE snapshot_metadata (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        version TEXT DEFAULT '1.0',
                        description TEXT,
                        product_count INTEGER DEFAULT 0,
                        is_historical BOOLEAN DEFAULT 1,
                        tags TEXT,
                        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            
            # Get product count
            cursor.execute('SELECT COUNT(*) FROM products')
            count = cursor.fetchone()[0]
            
            # Clear existing metadata and insert new
            cursor.execute('DELETE FROM snapshot_metadata')
            cursor.execute('''
                INSERT INTO snapshot_metadata 
                (name, description, product_count, is_historical, tags)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                name,
                description or f'Saved from main database on {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                count,
                True,
                json.dumps(tags or ['saved'])
            ))
        
        # Create uploads directory for this snapshot
        uploads_dir = get_snapshot_uploads_dir(db_path)
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Copy uploaded images if they exist
        main_uploads = os.path.join(BACKEND_DIR, 'uploads')
        if os.path.exists(main_uploads):
            for filename in os.listdir(main_uploads):
                src = os.path.join(main_uploads, filename)
                dst = os.path.join(uploads_dir, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        
        logger.info(f"Saved main database as snapshot: {db_filename}")
        
        return {
            'success': True,
            'snapshot_file': db_filename,
            'name': name,
            'product_count': count
        }
        
    except Exception as e:
        logger.error(f"Error saving main db as snapshot: {e}")
        return {'error': str(e)}


def load_snapshot_to_main_db(snapshot_file: str) -> Dict[str, Any]:
    """Load a snapshot into the main database
    
    This replaces product_matching.db with the snapshot contents
    
    Args:
        snapshot_file: Snapshot filename to load
        
    Returns:
        Dictionary with result
    """
    try:
        if not snapshot_file.endswith('.db'):
            snapshot_file = f"{snapshot_file}.db"
            
        snapshot_path = os.path.join(CATALOGS_DIR, snapshot_file)
        
        if not os.path.exists(snapshot_path):
            return {'error': f'Snapshot not found: {snapshot_file}'}
        
        # Get snapshot info before loading
        info = get_snapshot_info(snapshot_file)
        
        # Backup current main db (optional safety measure)
        if os.path.exists(DEFAULT_DB_PATH):
            backup_path = DEFAULT_DB_PATH + '.backup'
            shutil.copy2(DEFAULT_DB_PATH, backup_path)
        
        # Copy snapshot to main database
        shutil.copy2(snapshot_path, DEFAULT_DB_PATH)
        
        # Copy uploaded images
        snapshot_uploads = get_snapshot_uploads_dir(snapshot_path)
        main_uploads = os.path.join(BACKEND_DIR, 'uploads')
        
        # Clear main uploads first
        if os.path.exists(main_uploads):
            for filename in os.listdir(main_uploads):
                filepath = os.path.join(main_uploads, filename)
                if os.path.isfile(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
        
        # Copy snapshot uploads to main
        if os.path.exists(snapshot_uploads):
            os.makedirs(main_uploads, exist_ok=True)
            for filename in os.listdir(snapshot_uploads):
                src = os.path.join(snapshot_uploads, filename)
                dst = os.path.join(main_uploads, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        
        # Store which snapshot is loaded (for UI display)
        config = get_active_catalogs()
        config['loaded_snapshot'] = snapshot_file
        config['loaded_at'] = datetime.now().isoformat()
        
        with open(ACTIVE_CATALOGS_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Loaded snapshot {snapshot_file} to main database")
        
        return {
            'success': True,
            'snapshot_file': snapshot_file,
            'name': info.get('name', snapshot_file),
            'product_count': info.get('product_count', 0)
        }
        
    except Exception as e:
        logger.error(f"Error loading snapshot to main db: {e}")
        return {'error': str(e)}


def get_loaded_snapshot_info() -> Dict[str, Any]:
    """Get info about which snapshot is currently loaded in main db
    
    Returns:
        Dictionary with loaded snapshot info or None if none loaded
    """
    try:
        config = get_active_catalogs()
        loaded = config.get('loaded_snapshot')
        
        if not loaded:
            return {'loaded': False, 'message': 'No snapshot loaded - using default database'}
        
        # Check if snapshot still exists
        snapshot_path = os.path.join(CATALOGS_DIR, loaded)
        if not os.path.exists(snapshot_path):
            return {
                'loaded': False, 
                'message': 'Previously loaded snapshot no longer exists'
            }
        
        info = get_snapshot_info(loaded)
        
        return {
            'loaded': True,
            'snapshot_file': loaded,
            'name': info.get('name', loaded),
            'loaded_at': config.get('loaded_at'),
            'product_count': info.get('product_count', 0)
        }
        
    except Exception as e:
        logger.error(f"Error getting loaded snapshot info: {e}")
        return {'loaded': False, 'error': str(e)}


def get_main_db_stats() -> Dict[str, Any]:
    """Get statistics about the main database
    
    Returns:
        Dictionary with main database stats
    """
    try:
        if not os.path.exists(DEFAULT_DB_PATH):
            return {
                'exists': False,
                'total_products': 0,
                'historical_products': 0,
                'new_products': 0
            }
        
        import sqlite3
        conn = sqlite3.connect(DEFAULT_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Total products
        cursor.execute('SELECT COUNT(*) FROM products')
        total = cursor.fetchone()[0]
        
        # Historical
        cursor.execute('SELECT COUNT(*) FROM products WHERE is_historical = 1')
        historical = cursor.fetchone()[0]
        
        # New
        cursor.execute('SELECT COUNT(*) FROM products WHERE is_historical = 0')
        new = cursor.fetchone()[0]
        
        conn.close()
        
        # Get loaded snapshot info
        loaded_info = get_loaded_snapshot_info()
        
        # Get file size
        size_mb = round(os.path.getsize(DEFAULT_DB_PATH) / (1024 * 1024), 2)
        
        return {
            'exists': True,
            'total_products': total,
            'historical_products': historical,
            'new_products': new,
            'size_mb': size_mb,
            'loaded_snapshot': loaded_info
        }
        
    except Exception as e:
        logger.error(f"Error getting main db stats: {e}")
        return {'exists': False, 'error': str(e)}
