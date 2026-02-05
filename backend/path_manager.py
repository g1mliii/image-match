"""
Centralized Path Management
Handles all file paths for both development and bundled app deployment.
"""

import os
import sys
import platform
from pathlib import Path

def is_bundled():
    """Check if running as PyInstaller frozen executable"""
    return getattr(sys, 'frozen', False)

def get_app_data_dir():
    r"""Get the application data directory

    Returns:
        Windows: %APPDATA%\ProductMatcher\
        macOS: ~/Library/Application Support/ProductMatcher/
        Linux: ~/.local/share/ProductMatcher/
    """
    home = os.path.expanduser("~")
    system = platform.system()

    if system == 'Windows':
        app_data = os.environ.get('APPDATA', os.path.join(home, 'AppData', 'Roaming'))
        data_dir = os.path.join(app_data, 'ProductMatcher')
    elif system == 'Darwin':  # macOS
        data_dir = os.path.join(home, 'Library', 'Application Support', 'ProductMatcher')
    else:  # Linux
        data_dir = os.path.join(home, '.local', 'share', 'ProductMatcher')

    # Create if doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_backend_dir():
    """Get backend directory (where backend modules are located)"""
    if is_bundled():
        # In bundled app, backend is in the same directory as executable
        return os.path.join(os.path.dirname(sys.executable), 'backend')
    else:
        # In development, use __file__ location
        return os.path.dirname(os.path.abspath(__file__))

def get_database_path():
    r"""Get database file path

    Bundled: %APPDATA%\ProductMatcher\product_matching.db
    Development: backend/product_matching.db
    """
    if is_bundled():
        app_data = get_app_data_dir()
        return os.path.join(app_data, 'product_matching.db')
    else:
        backend_dir = get_backend_dir()
        return os.path.join(backend_dir, 'product_matching.db')

def get_uploads_dir():
    r"""Get uploads directory for temporary image storage

    Bundled: %APPDATA%\ProductMatcher\uploads\
    Development: backend/uploads/
    """
    if is_bundled():
        app_data = get_app_data_dir()
        uploads_dir = os.path.join(app_data, 'uploads')
    else:
        backend_dir = get_backend_dir()
        uploads_dir = os.path.join(backend_dir, 'uploads')

    os.makedirs(uploads_dir, exist_ok=True)
    return uploads_dir

def get_catalogs_dir():
    r"""Get catalogs directory for snapshot storage

    Bundled: %APPDATA%\ProductMatcher\catalogs\
    Development: backend/catalogs/
    """
    if is_bundled():
        app_data = get_app_data_dir()
        catalogs_dir = os.path.join(app_data, 'catalogs')
    else:
        backend_dir = get_backend_dir()
        catalogs_dir = os.path.join(backend_dir, 'catalogs')

    os.makedirs(catalogs_dir, exist_ok=True)
    return catalogs_dir

def get_config_dir():
    r"""Get config directory for storing settings

    Bundled: %APPDATA%\ProductMatcher\config\
    Development: backend/config/
    """
    if is_bundled():
        app_data = get_app_data_dir()
        config_dir = os.path.join(app_data, 'config')
    else:
        backend_dir = get_backend_dir()
        config_dir = os.path.join(backend_dir, 'config')

    os.makedirs(config_dir, exist_ok=True)
    return config_dir

def get_clip_cache_dir():
    """Get CLIP model cache directory

    All platforms: ~/.cache/clip-models/
    (CLIP is bundled in PyInstaller, but also uses home directory cache)
    """
    cache_dir = os.path.join(os.path.expanduser("~"), '.cache', 'clip-models')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_staging_dir():
    """Get staging directory for inter-window communication

    All platforms: ./staging/ (relative to project root)
    Note: Only used in development/bundled app main window
    """
    if is_bundled():
        # In bundled app, use AppData
        app_data = get_app_data_dir()
        staging_dir = os.path.join(app_data, 'staging')
    else:
        # In development, use project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        staging_dir = os.path.join(project_root, 'staging')

    os.makedirs(staging_dir, exist_ok=True)
    return staging_dir

def get_downloads_dir():
    """Get user's Downloads directory

    All platforms: ~/Downloads/
    """
    downloads = os.path.join(os.path.expanduser("~"), 'Downloads')
    os.makedirs(downloads, exist_ok=True)
    return downloads

# Convenience exports
__all__ = [
    'is_bundled',
    'get_app_data_dir',
    'get_backend_dir',
    'get_database_path',
    'get_uploads_dir',
    'get_catalogs_dir',
    'get_config_dir',
    'get_clip_cache_dir',
    'get_staging_dir',
    'get_downloads_dir',
]
