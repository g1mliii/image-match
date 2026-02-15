"""
Configuration Module
Centralized configuration to avoid circular import issues
"""
import os
import logging
from path_manager import get_backend_dir, get_config_dir

logger = logging.getLogger(__name__)

# Backend directory
BACKEND_DIR = get_backend_dir()

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
from urllib.parse import urlparse

CONFIG_DIR = get_config_dir()
MOBILE_CONFIG_FILE = os.path.join(CONFIG_DIR, 'mobile_config.json')
MOBILE_NETWORK_CONFIG_FILE = os.path.join(CONFIG_DIR, 'mobile_network_config.json')
MOBILE_NGROK_CONFIG_FILE = os.path.join(CONFIG_DIR, 'ngrok_config.json')

# Cache for performance (avoid repeated file I/O)
_mobile_password_cache = None
_mobile_password_mtime = None
_mobile_remote_url_cache = None
_mobile_remote_url_mtime = None
_mobile_remote_url_loaded = False
_mobile_ngrok_token_cache = None
_mobile_ngrok_token_mtime = None
_mobile_ngrok_token_loaded = False

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

# ============ Mobile Remote URL Configuration ============
def _normalize_mobile_remote_url(remote_url):
    """Normalize and validate remote mobile URL (HTTPS only)."""
    if remote_url is None:
        return None

    value = str(remote_url).strip()
    if not value:
        return None

    parsed = urlparse(value)
    if parsed.scheme.lower() != 'https' or not parsed.netloc:
        raise ValueError("Remote URL must be a valid HTTPS URL")

    # Default to /mobile when no path is provided
    path = parsed.path if parsed.path else '/mobile'
    cleaned = parsed._replace(path=path, params='', query='', fragment='')
    normalized = cleaned.geturl()

    # Preserve path but remove trailing slash for consistency
    if normalized.endswith('/') and path != '/':
        normalized = normalized[:-1]

    return normalized


def get_mobile_remote_url():
    """Get optional remote mobile URL from env var or config file."""
    global _mobile_remote_url_cache, _mobile_remote_url_mtime, _mobile_remote_url_loaded

    # Environment variable takes precedence (ephemeral/ops-friendly)
    env_url = os.environ.get('MOBILE_REMOTE_URL', '').strip()
    if env_url:
        try:
            return _normalize_mobile_remote_url(env_url)
        except ValueError as e:
            logger.warning(f"Ignoring invalid MOBILE_REMOTE_URL: {e}")
            return None

    try:
        if os.path.exists(MOBILE_NETWORK_CONFIG_FILE):
            current_mtime = os.path.getmtime(MOBILE_NETWORK_CONFIG_FILE)

            if _mobile_remote_url_loaded and _mobile_remote_url_mtime == current_mtime:
                return _mobile_remote_url_cache

            with open(MOBILE_NETWORK_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                raw_url = config.get('remote_url')
                normalized = _normalize_mobile_remote_url(raw_url)
                _mobile_remote_url_cache = normalized
                _mobile_remote_url_mtime = current_mtime
                _mobile_remote_url_loaded = True
                return normalized

        _mobile_remote_url_loaded = False
        return None
    except ValueError as e:
        logger.warning(f"Invalid remote mobile URL in config: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading mobile network config: {e}")
        return None


def save_mobile_remote_url(remote_url):
    """Persist optional remote mobile URL. Pass empty value to clear it."""
    global _mobile_remote_url_cache, _mobile_remote_url_mtime, _mobile_remote_url_loaded

    normalized = _normalize_mobile_remote_url(remote_url)

    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        config = {
            'remote_url': normalized,
            'updated_at': datetime.now().isoformat()
        }
        with open(MOBILE_NETWORK_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        _mobile_remote_url_cache = normalized
        _mobile_remote_url_mtime = os.path.getmtime(MOBILE_NETWORK_CONFIG_FILE)
        _mobile_remote_url_loaded = True
        logger.info("Mobile remote URL updated")
    except Exception as e:
        logger.error(f"Error saving mobile network config: {e}")
        raise

# ============ ngrok Token Configuration ============
def _normalize_ngrok_authtoken(token):
    """Normalize/validate ngrok auth token."""
    if token is None:
        return None

    value = str(token).strip()
    if not value:
        return None

    if any(ch.isspace() for ch in value):
        raise ValueError("ngrok token must not contain spaces")
    if len(value) < 20:
        raise ValueError("ngrok token appears too short")
    if len(value) > 300:
        raise ValueError("ngrok token appears too long")

    return value


def get_ngrok_authtoken():
    """Get ngrok auth token from env var or config file."""
    global _mobile_ngrok_token_cache, _mobile_ngrok_token_mtime, _mobile_ngrok_token_loaded

    env_token = os.environ.get('NGROK_AUTHTOKEN', '').strip()
    if env_token:
        try:
            return _normalize_ngrok_authtoken(env_token)
        except ValueError as e:
            logger.warning(f"Ignoring invalid NGROK_AUTHTOKEN: {e}")
            return None

    try:
        if os.path.exists(MOBILE_NGROK_CONFIG_FILE):
            current_mtime = os.path.getmtime(MOBILE_NGROK_CONFIG_FILE)

            if _mobile_ngrok_token_loaded and _mobile_ngrok_token_mtime == current_mtime:
                return _mobile_ngrok_token_cache

            with open(MOBILE_NGROK_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                raw_token = config.get('authtoken')
                normalized = _normalize_ngrok_authtoken(raw_token)
                _mobile_ngrok_token_cache = normalized
                _mobile_ngrok_token_mtime = current_mtime
                _mobile_ngrok_token_loaded = True
                return normalized

        _mobile_ngrok_token_loaded = False
        return None
    except ValueError as e:
        logger.warning(f"Invalid ngrok token in config: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading ngrok config: {e}")
        return None


def save_ngrok_authtoken(token):
    """Persist ngrok auth token."""
    global _mobile_ngrok_token_cache, _mobile_ngrok_token_mtime, _mobile_ngrok_token_loaded

    normalized = _normalize_ngrok_authtoken(token)
    if not normalized:
        raise ValueError("ngrok token is required")

    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        payload = {
            'authtoken': normalized,
            'updated_at': datetime.now().isoformat()
        }
        with open(MOBILE_NGROK_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        _mobile_ngrok_token_cache = normalized
        _mobile_ngrok_token_mtime = os.path.getmtime(MOBILE_NGROK_CONFIG_FILE)
        _mobile_ngrok_token_loaded = True
        logger.info("ngrok token updated")
    except Exception as e:
        logger.error(f"Error saving ngrok config: {e}")
        raise


def clear_ngrok_authtoken():
    """Clear persisted ngrok auth token."""
    global _mobile_ngrok_token_cache, _mobile_ngrok_token_mtime, _mobile_ngrok_token_loaded
    try:
        if os.path.exists(MOBILE_NGROK_CONFIG_FILE):
            os.remove(MOBILE_NGROK_CONFIG_FILE)
        _mobile_ngrok_token_cache = None
        _mobile_ngrok_token_mtime = None
        _mobile_ngrok_token_loaded = True
        logger.info("ngrok token cleared")
    except Exception as e:
        logger.error(f"Error clearing ngrok config: {e}")
        raise

# ==================================================
