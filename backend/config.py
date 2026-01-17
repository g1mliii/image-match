"""
Configuration Module
Centralized configuration to avoid circular import issues
"""
import os
import logging

logger = logging.getLogger(__name__)

# Backend directory
BACKEND_DIR = os.path.dirname(__file__)

# ============ Debug Mode Configuration ============
# Debug mode enables verbose logging for troubleshooting
# Set via environment variable: DEBUG_MODE=true
# Or via config file: backend/debug.conf (create from debug.conf.example)
DEBUG_MODE = os.environ.get('DEBUG_MODE', '').lower() in ('true', '1', 'yes')

# Fallback to config file if exists
DEBUG_CONFIG_PATH = os.path.join(BACKEND_DIR, 'debug.conf')
if not DEBUG_MODE and os.path.exists(DEBUG_CONFIG_PATH):
    try:
        with open(DEBUG_CONFIG_PATH, 'r') as f:
            content = f.read().strip().lower()
            DEBUG_MODE = content in ('true', '1', 'yes', 'on')
    except:
        pass

def is_debug_mode():
    """Check if debug mode is enabled (for use by other modules)"""
    return DEBUG_MODE

# Log debug status on module load
logger.info(f"Debug mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")
# ==================================================

# ============ Mobile Upload Configuration ============
import secrets
import json
from datetime import datetime
from pathlib import Path

CONFIG_DIR = os.path.join(BACKEND_DIR, 'config')
MOBILE_CONFIG_FILE = os.path.join(CONFIG_DIR, 'mobile_config.json')

# Cache for performance (avoid repeated file I/O)
_mobile_password_cache = None
_mobile_password_mtime = None

def get_mobile_password():
    """Get or generate mobile upload password (6-digit PIN)

    Performance optimized:
    - Caches password in memory
    - Only reads file if it was modified externally
    - Generates new password on first run

    Returns:
        str: 6-digit PIN (e.g., "123456")
    """
    global _mobile_password_cache, _mobile_password_mtime

    try:
        # Check if config file exists and hasn't changed
        if os.path.exists(MOBILE_CONFIG_FILE):
            current_mtime = os.path.getmtime(MOBILE_CONFIG_FILE)

            # If cached and file unchanged, use cache
            if _mobile_password_cache is not None and _mobile_password_mtime == current_mtime:
                return _mobile_password_cache

            # File modified or first check - read it
            with open(MOBILE_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                _mobile_password_cache = config.get('password')
                _mobile_password_mtime = current_mtime
                return _mobile_password_cache
        else:
            # No config file - generate new password
            password = ''.join(str(secrets.randbelow(10)) for _ in range(6))
            save_mobile_password(password)
            return password
    except Exception as e:
        logger.error(f"Error reading mobile password config: {e}")
        # Fallback: generate temporary password (not persistent)
        return ''.join(str(secrets.randbelow(10)) for _ in range(6))

def save_mobile_password(password):
    """Save mobile upload password

    Args:
        password: 6-digit PIN string
    """
    global _mobile_password_cache, _mobile_password_mtime

    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        config = {
            'password': str(password),
            'updated_at': datetime.now().isoformat()
        }
        with open(MOBILE_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        # Update cache
        _mobile_password_cache = str(password)
        _mobile_password_mtime = os.path.getmtime(MOBILE_CONFIG_FILE)

        logger.info(f"Mobile password updated")
    except Exception as e:
        logger.error(f"Error saving mobile password config: {e}")

def validate_mobile_password(provided_password):
    """Validate mobile upload password (constant-time comparison)

    Uses constant-time comparison to prevent timing attacks.

    Args:
        provided_password: Password string to validate

    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        stored = get_mobile_password()
        # Use secrets.compare_digest for constant-time comparison
        # Prevents timing attacks where attacker measures response time
        return secrets.compare_digest(str(provided_password), stored)
    except Exception as e:
        logger.error(f"Error validating password: {e}")
        return False

# ==================================================
