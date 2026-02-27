"""Mobile endpoints and helpers extracted from app_core.py."""

import ipaddress
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

mobile_bp = Blueprint('mobile_routes', __name__)

# Dependencies injected from app_core to avoid circular imports.
logger = None
create_error_response = None
invalidate_csv_cache = None
invalidate_catalog_categories_cache = None
BACKEND_DIR = None
get_product_by_id = None
get_product_metadata = None
insert_product = None
insert_features = None
extract_features_unified = None
validate_category = None
validate_product_name = None
validate_sku = None
allowed_file = None

_CATEGORY_METADATA_KEYS = {'category', 'product_category', 'productcategory'}
_CATEGORY_METADATA_HINTS = ('category', 'product_category', 'productcategory')


def _normalize_metadata_key(key):
    return str(key).strip().lower().replace('-', '_').replace(' ', '_')


def configure_mobile_routes(**deps):
    """Inject app_core dependencies used by mobile endpoints."""
    global logger
    global create_error_response
    global invalidate_csv_cache
    global invalidate_catalog_categories_cache
    global BACKEND_DIR
    global get_product_by_id
    global get_product_metadata
    global insert_product
    global insert_features
    global extract_features_unified
    global validate_category
    global validate_product_name
    global validate_sku
    global allowed_file

    logger = deps['logger']
    create_error_response = deps['create_error_response']
    invalidate_csv_cache = deps['invalidate_csv_cache']
    invalidate_catalog_categories_cache = deps['invalidate_catalog_categories_cache']
    BACKEND_DIR = deps['BACKEND_DIR']
    get_product_by_id = deps['get_product_by_id']
    get_product_metadata = deps['get_product_metadata']
    insert_product = deps['insert_product']
    insert_features = deps['insert_features']
    extract_features_unified = deps['extract_features_unified']
    validate_category = deps['validate_category']
    validate_product_name = deps['validate_product_name']
    validate_sku = deps['validate_sku']
    allowed_file = deps['allowed_file']


def _extract_category_from_metadata_payload(metadata_payload):
    """Extract normalized category from metadata payload when top-level category is missing."""
    metadata_obj = None

    if isinstance(metadata_payload, dict):
        metadata_obj = metadata_payload
    elif isinstance(metadata_payload, str):
        raw = metadata_payload.strip()
        if not raw:
            return None

        raw_lower = raw.lower()
        if not any(hint in raw_lower for hint in _CATEGORY_METADATA_HINTS):
            return None

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                metadata_obj = parsed
            else:
                return None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    else:
        return None

    if not metadata_obj:
        return None

    from product_matching import normalize_category

    # Rule 1: explicit "category" present but empty/unknown => keep uncategorized.
    for key, value in metadata_obj.items():
        if _normalize_metadata_key(key) != 'category':
            continue
        category_raw = '' if value is None else str(value).strip()
        if not category_raw:
            return None
        return normalize_category(category_raw)

    # Rule 2: backfill from aliases only when explicit "category" is absent.
    for key, value in metadata_obj.items():
        key_norm = _normalize_metadata_key(key)
        if key_norm not in _CATEGORY_METADATA_KEYS or key_norm == 'category':
            continue

        category_raw = '' if value is None else str(value).strip()
        if not category_raw:
            continue

        normalized = normalize_category(category_raw)
        if normalized is not None:
            return normalized

    return None


# Mobile security settings
MOBILE_SESSION_TTL_SECONDS = 15 * 60
MOBILE_AUTH_WINDOW_SECONDS = 10 * 60
MOBILE_AUTH_MAX_FAILED_ATTEMPTS = 5
MOBILE_AUTH_LOCKOUT_SECONDS = 15 * 60
MOBILE_SESSION_SWEEP_INTERVAL_SECONDS = 60
MOBILE_AUTH_SWEEP_INTERVAL_SECONDS = 120
MOBILE_MAX_ACTIVE_SESSIONS = 5000
MOBILE_MAX_AUTH_TRACKED_IPS = 5000
MOBILE_MAX_FAILED_ATTEMPTS_TRACKED_PER_IP = 12

_mobile_sessions = {}
_mobile_sessions_lock = threading.Lock()
_mobile_auth_failures = {}
_mobile_auth_failures_lock = threading.Lock()
_mobile_last_session_sweep_ts = 0.0
_mobile_last_auth_sweep_ts = 0.0
_ngrok_process = None
_ngrok_started_by_app = False
_ngrok_process_lock = threading.Lock()


def _is_loopback_ip(value):
    """Check if an IP address is loopback."""
    try:
        return ipaddress.ip_address(value).is_loopback
    except Exception:
        return False


def _get_client_ip():
    """Resolve request client IP with safe proxy handling for local tunnel proxy."""
    remote_addr = (request.remote_addr or '').strip()
    if _is_loopback_ip(remote_addr):
        # Some local reverse tunnels send the real client IP in this header.
        cf_ip = (request.headers.get('CF-Connecting-IP') or '').strip()
        if cf_ip:
            return cf_ip

        # ngrok and other trusted local reverse proxies typically use XFF.
        xff = (request.headers.get('X-Forwarded-For') or '').strip()
        if xff:
            return xff.split(',')[0].strip()

    return remote_addr or 'unknown'


def _cleanup_expired_mobile_sessions_locked(now_ts, force=False):
    global _mobile_last_session_sweep_ts

    should_sweep = (
        force
        or (now_ts - _mobile_last_session_sweep_ts) >= MOBILE_SESSION_SWEEP_INTERVAL_SECONDS
        or len(_mobile_sessions) > MOBILE_MAX_ACTIVE_SESSIONS
    )
    if not should_sweep:
        return

    _mobile_last_session_sweep_ts = now_ts
    expired = [token for token, session in _mobile_sessions.items() if session.get('expires_at', 0) <= now_ts]
    for token in expired:
        _mobile_sessions.pop(token, None)

    # Hard cap to avoid unbounded growth during prolonged high churn.
    if len(_mobile_sessions) > MOBILE_MAX_ACTIVE_SESSIONS:
        overflow = len(_mobile_sessions) - MOBILE_MAX_ACTIVE_SESSIONS
        oldest_tokens = sorted(
            _mobile_sessions.items(),
            key=lambda item: item[1].get('last_seen_at', item[1].get('issued_at', 0))
        )[:overflow]
        for token, _ in oldest_tokens:
            _mobile_sessions.pop(token, None)


def _issue_mobile_session_token(client_ip):
    token = secrets.token_urlsafe(48)
    now_ts = time.time()
    with _mobile_sessions_lock:
        _cleanup_expired_mobile_sessions_locked(now_ts)
        _mobile_sessions[token] = {
            'client_ip': client_ip,
            'issued_at': now_ts,
            'last_seen_at': now_ts,
            'expires_at': now_ts + MOBILE_SESSION_TTL_SECONDS
        }
    return token


def _validate_mobile_session_token(session_token, client_ip):
    now_ts = time.time()
    with _mobile_sessions_lock:
        _cleanup_expired_mobile_sessions_locked(now_ts)
        session = _mobile_sessions.get(session_token)
        if session is None:
            return False, 'Invalid or expired session token'

        # Bind session to client IP to reduce token replay risk.
        session_ip = session.get('client_ip')
        if session_ip and client_ip and session_ip != client_ip:
            _mobile_sessions.pop(session_token, None)
            return False, 'Session token is not valid for this client'

        # Sliding expiration on active use.
        session['last_seen_at'] = now_ts
        session['expires_at'] = now_ts + MOBILE_SESSION_TTL_SECONDS
        return True, None


def _cleanup_mobile_auth_failures_locked(now_ts, force=False):
    global _mobile_last_auth_sweep_ts

    should_sweep = (
        force
        or (now_ts - _mobile_last_auth_sweep_ts) >= MOBILE_AUTH_SWEEP_INTERVAL_SECONDS
        or len(_mobile_auth_failures) > MOBILE_MAX_AUTH_TRACKED_IPS
    )
    if not should_sweep:
        return

    _mobile_last_auth_sweep_ts = now_ts

    stale_clients = []
    for client_ip, state in _mobile_auth_failures.items():
        attempts = [
            ts for ts in state.get('failed_attempts', [])
            if now_ts - ts <= MOBILE_AUTH_WINDOW_SECONDS
        ]
        state['failed_attempts'] = attempts[-MOBILE_MAX_FAILED_ATTEMPTS_TRACKED_PER_IP:]

        lockout_until = state.get('lockout_until', 0)
        if lockout_until <= now_ts and not state['failed_attempts']:
            stale_clients.append(client_ip)

    for client_ip in stale_clients:
        _mobile_auth_failures.pop(client_ip, None)

    # Hard cap to avoid unbounded growth with many unique IPs.
    if len(_mobile_auth_failures) > MOBILE_MAX_AUTH_TRACKED_IPS:
        overflow = len(_mobile_auth_failures) - MOBILE_MAX_AUTH_TRACKED_IPS
        oldest_clients = sorted(
            _mobile_auth_failures.items(),
            key=lambda item: item[1].get('last_seen_at', 0)
        )[:overflow]
        for client_ip, _ in oldest_clients:
            _mobile_auth_failures.pop(client_ip, None)


def _check_mobile_auth_lockout_seconds(client_ip):
    now_ts = time.time()
    with _mobile_auth_failures_lock:
        _cleanup_mobile_auth_failures_locked(now_ts)
        state = _mobile_auth_failures.get(client_ip)
        if state is None:
            return 0
        state['last_seen_at'] = now_ts

        lockout_until = state.get('lockout_until', 0)
        if lockout_until > now_ts:
            return max(1, int(lockout_until - now_ts))

        failed_attempts = [
            ts for ts in state.get('failed_attempts', [])
            if now_ts - ts <= MOBILE_AUTH_WINDOW_SECONDS
        ]
        if failed_attempts:
            state['failed_attempts'] = failed_attempts[-MOBILE_MAX_FAILED_ATTEMPTS_TRACKED_PER_IP:]
        else:
            _mobile_auth_failures.pop(client_ip, None)

        return 0


def _record_mobile_auth_failure(client_ip):
    now_ts = time.time()
    with _mobile_auth_failures_lock:
        _cleanup_mobile_auth_failures_locked(now_ts)
        state = _mobile_auth_failures.setdefault(client_ip, {'failed_attempts': [], 'lockout_until': 0, 'last_seen_at': now_ts})
        state['last_seen_at'] = now_ts
        attempts = [ts for ts in state.get('failed_attempts', []) if now_ts - ts <= MOBILE_AUTH_WINDOW_SECONDS]
        attempts.append(now_ts)
        state['failed_attempts'] = attempts[-MOBILE_MAX_FAILED_ATTEMPTS_TRACKED_PER_IP:]

        if len(attempts) >= MOBILE_AUTH_MAX_FAILED_ATTEMPTS:
            state['lockout_until'] = now_ts + MOBILE_AUTH_LOCKOUT_SECONDS
            return MOBILE_AUTH_LOCKOUT_SECONDS

    return 0


def _clear_mobile_auth_failures(client_ip):
    with _mobile_auth_failures_lock:
        _mobile_auth_failures.pop(client_ip, None)


def _revoke_all_mobile_sessions():
    with _mobile_sessions_lock:
        _mobile_sessions.clear()


def _require_localhost_request():
    """Restrict sensitive endpoints to same-machine desktop app access."""
    remote_addr = (request.remote_addr or '').strip()
    if _is_loopback_ip(remote_addr):
        return None

    logger.warning(f"Blocked non-local request for local-only endpoint from {remote_addr}")
    return create_error_response(
        'FORBIDDEN',
        'This endpoint is only available from the desktop app on this machine',
        status_code=403
    )


def _require_mobile_session():
    """Validate mobile session token from request headers."""
    client_ip = _get_client_ip()
    session_token = (request.headers.get('X-Mobile-Session') or '').strip()

    if not session_token:
        logger.warning(f"Mobile request missing session token from {client_ip}")
        return None, create_error_response('MISSING_AUTH', 'Session token required', status_code=401)

    is_valid, error_message = _validate_mobile_session_token(session_token, client_ip)
    if not is_valid:
        logger.warning(f"Mobile request with invalid session token from {client_ip}")
        return None, create_error_response('UNAUTHORIZED', error_message, status_code=401)

    return client_ip, None


def _get_local_ip():
    import socket
    try:
        # Connect to a public DNS server to find local IP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception as e:
        logger.debug(f"Failed to get local IP: {e}")
        return "127.0.0.1"


def _build_mobile_connection_info(include_password=False):
    from config import get_mobile_password, get_mobile_remote_url

    primary_ip = _get_local_ip()
    port = request.environ.get('SERVER_PORT', 5000)
    localhost_url = f'http://127.0.0.1:{port}/mobile'
    lan_url = f'http://{primary_ip}:{port}/mobile'
    remote_url = get_mobile_remote_url()

    payload = {
        'primary_ip': primary_ip,
        'port': port,
        'localhost_url': localhost_url,
        'mobile_url': lan_url,
        'lan_url': lan_url,
        'remote_url': remote_url,
        'remote_enabled': bool(remote_url)
    }

    if include_password:
        payload['password'] = get_mobile_password()

    return payload


def _discover_ngrok_public_url(timeout_seconds=0.8):
    """Discover active ngrok HTTPS public URL from the local ngrok agent API."""
    api_url = (os.environ.get('NGROK_API_URL') or 'http://127.0.0.1:4040/api/tunnels').strip()
    timeout_seconds = max(0.2, min(float(timeout_seconds), 5.0))

    try:
        request_obj = urllib.request.Request(
            api_url,
            headers={'Accept': 'application/json'}
        )
        with urllib.request.urlopen(request_obj, timeout=timeout_seconds) as response:
            if response.status != 200:
                return None, f"ngrok API returned HTTP {response.status}"
            payload = json.loads(response.read().decode('utf-8', errors='replace'))
    except urllib.error.URLError:
        return None, (
            f"Could not reach ngrok API at {api_url}. "
            "Start ngrok first (example: ngrok http http://127.0.0.1:8000)."
        )
    except Exception as e:
        return None, f"Failed reading ngrok API: {e}"

    tunnels = payload.get('tunnels', [])
    if not isinstance(tunnels, list) or not tunnels:
        return None, "No active ngrok tunnels found."

    https_url = None
    for tunnel in tunnels:
        public_url = str((tunnel or {}).get('public_url', '')).strip()
        if public_url.startswith('https://'):
            https_url = public_url
            break

    if not https_url:
        return None, "No HTTPS ngrok tunnel found. Start ngrok with HTTPS enabled."

    return https_url.rstrip('/'), None


def _resolve_ngrok_binary_path():
    """Resolve ngrok binary path using env override, bundled path, then PATH."""
    env_path = (os.environ.get('NGROK_PATH') or '').strip().strip('"').strip("'")
    if env_path and os.path.isfile(env_path):
        return env_path, 'env'

    project_root = os.path.dirname(BACKEND_DIR)
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else None
    meipass_dir = getattr(sys, '_MEIPASS', None)

    if sys.platform == 'win32':
        bundled_rel = os.path.join('third_party', 'ngrok', 'windows', 'ngrok.exe')
    elif sys.platform == 'darwin':
        bundled_rel = os.path.join('third_party', 'ngrok', 'macos', 'ngrok')
    else:
        bundled_rel = os.path.join('third_party', 'ngrok', 'linux', 'ngrok')

    candidates = [os.path.join(project_root, bundled_rel)]
    if meipass_dir:
        candidates.append(os.path.join(meipass_dir, bundled_rel))
    if exe_dir:
        candidates.append(os.path.join(exe_dir, bundled_rel))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate, 'bundled'

    path_binary = shutil.which('ngrok')
    if path_binary:
        return path_binary, 'path'

    return None, None


def _apply_ngrok_authtoken_to_agent(token):
    """Persist token into ngrok config for better first-run reliability."""
    ngrok_binary, _ = _resolve_ngrok_binary_path()
    if not ngrok_binary:
        return False, 'ngrok binary not found'

    try:
        command = [ngrok_binary, 'config', 'add-authtoken', token]
        run_kwargs = {
            'check': False,
            'capture_output': True,
            'text': True
        }
        if sys.platform == 'win32':
            run_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(command, **run_kwargs)
        if result.returncode != 0:
            stderr_text = (result.stderr or '').strip()
            return False, stderr_text or 'ngrok returned a non-zero exit code'
        return True, None
    except Exception as e:
        return False, str(e)


def _start_ngrok_tunnel_for_backend(port):
    """Start ngrok tunnel for this local backend and return public URL."""
    global _ngrok_process, _ngrok_started_by_app

    public_url, _ = _discover_ngrok_public_url(timeout_seconds=0.4)
    if public_url:
        return public_url, None

    # Give an existing external/startup ngrok a brief grace window to expose API URL.
    for _ in range(6):
        public_url, _ = _discover_ngrok_public_url(timeout_seconds=0.3)
        if public_url:
            return public_url, None
        time.sleep(0.15)

    ngrok_binary, binary_source = _resolve_ngrok_binary_path()
    if not ngrok_binary:
        return None, 'ngrok binary not found. Install ngrok or bundle it with the app.'

    from config import get_ngrok_authtoken
    token = get_ngrok_authtoken()
    if not token:
        return None, 'ngrok token is not configured. Use SETUP TOKEN first.'

    with _ngrok_process_lock:
        process_running = (_ngrok_process is not None and _ngrok_process.poll() is None)
    if process_running:
        for _ in range(24):
            public_url, _ = _discover_ngrok_public_url(timeout_seconds=0.3)
            if public_url:
                return public_url, None
            time.sleep(0.25)
        return None, 'ngrok process is running but public URL is not ready yet'

    with _ngrok_process_lock:
        # Another request may have started ngrok while we were waiting above.
        if _ngrok_process is not None and _ngrok_process.poll() is None:
            process_running = True
        else:
            process_running = False

        if process_running:
            # Fast path: let the running process expose URL without launching a duplicate.
            pass
        else:
            command = [ngrok_binary, 'http', f'http://127.0.0.1:{port}']
            child_env = os.environ.copy()
            child_env['NGROK_AUTHTOKEN'] = token

            popen_kwargs = {
                'stdout': subprocess.DEVNULL,
                'stderr': subprocess.STDOUT,
                'env': child_env
            }
            if sys.platform == 'win32':
                popen_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

            try:
                _ngrok_process = subprocess.Popen(command, **popen_kwargs)
                _ngrok_started_by_app = True
                logger.info(f"Started ngrok tunnel ({binary_source}) for local port {port}")
            except Exception as e:
                _ngrok_process = None
                _ngrok_started_by_app = False
                return None, f'Failed to start ngrok: {e}'

    for _ in range(30):
        if _ngrok_process is not None and _ngrok_process.poll() is not None:
            return None, 'ngrok exited immediately. Check token validity and ngrok version.'
        public_url, _ = _discover_ngrok_public_url(timeout_seconds=0.3)
        if public_url:
            return public_url, None
        time.sleep(0.25)

    return None, 'ngrok started but URL is not ready yet'


def _stop_ngrok_tunnel_for_backend():
    """Stop ngrok process started by this backend instance."""
    global _ngrok_process, _ngrok_started_by_app

    with _ngrok_process_lock:
        if not _ngrok_started_by_app or _ngrok_process is None:
            return False, 'No app-managed ngrok process to stop'

        process = _ngrok_process
        _ngrok_process = None
        _ngrok_started_by_app = False

    if process.poll() is not None:
        return True, None

    try:
        process.terminate()
        process.wait(timeout=3)
        return True, None
    except Exception:
        try:
            process.kill()
            process.wait(timeout=2)
            return True, None
        except Exception as e:
            return False, str(e)


def stop_ngrok_tunnel_for_backend():
    """Public helper for app-level shutdown cleanup."""
    return _stop_ngrok_tunnel_for_backend()


@mobile_bp.route('/api/network/local-ip', methods=['GET'])
def get_local_ip_endpoint():
    """Get local/LAN and optional secure remote mobile URLs."""
    return jsonify(_build_mobile_connection_info(include_password=False))


@mobile_bp.route('/api/mobile/connection-info', methods=['GET'])
def get_mobile_connection_info():
    """Local-only endpoint for desktop app modal (includes current PIN)."""
    local_only_error = _require_localhost_request()
    if local_only_error:
        return local_only_error

    try:
        return jsonify(_build_mobile_connection_info(include_password=True)), 200
    except Exception as e:
        logger.error(f"Error loading mobile connection info: {e}", exc_info=True)
        return create_error_response('CONNECTION_INFO_ERROR', 'Failed to load connection info', status_code=500)


@mobile_bp.route('/api/mobile/remote-url', methods=['GET', 'POST'])
def mobile_remote_url_management():
    """Local-only management for secure remote mobile URL."""
    local_only_error = _require_localhost_request()
    if local_only_error:
        return local_only_error

    from config import get_mobile_remote_url, save_mobile_remote_url

    env_remote_url = (os.environ.get('MOBILE_REMOTE_URL') or '').strip()
    source = 'env' if env_remote_url else 'config'

    if request.method == 'GET':
        return jsonify({
            'remote_url': get_mobile_remote_url(),
            'source': source,
            'editable': not bool(env_remote_url)
        }), 200

    if env_remote_url:
        return create_error_response(
            'REMOTE_URL_LOCKED',
            'Remote URL is managed by MOBILE_REMOTE_URL environment variable',
            status_code=409
        )

    try:
        data = request.get_json() or {}
        remote_url = str(data.get('remote_url', '')).strip()
        save_mobile_remote_url(remote_url)
        return jsonify({
            'success': True,
            'remote_url': get_mobile_remote_url()
        }), 200
    except ValueError as e:
        return create_error_response('INVALID_REMOTE_URL', str(e), status_code=400)
    except Exception as e:
        logger.error(f"Error saving mobile remote URL: {e}", exc_info=True)
        return create_error_response('REMOTE_URL_ERROR', 'Failed to save remote URL', status_code=500)


@mobile_bp.route('/api/mobile/ngrok/status', methods=['GET'])
def mobile_ngrok_status():
    """Local-only ngrok status for desktop modal setup UX."""
    local_only_error = _require_localhost_request()
    if local_only_error:
        return local_only_error

    from config import get_ngrok_authtoken

    ngrok_binary, binary_source = _resolve_ngrok_binary_path()
    token = get_ngrok_authtoken()
    token_source = 'env' if (os.environ.get('NGROK_AUTHTOKEN') or '').strip() else ('config' if token else None)
    public_url, _ = _discover_ngrok_public_url()

    process_running = False
    with _ngrok_process_lock:
        process_running = (_ngrok_process is not None and _ngrok_process.poll() is None)

    return jsonify({
        'ngrok_installed': bool(ngrok_binary),
        'ngrok_binary_source': binary_source,
        'ngrok_binary_path': ngrok_binary,
        'has_token': bool(token),
        'token_source': token_source,
        'public_url': public_url,
        'tunnel_running': bool(public_url or process_running),
        'managed_by_app': bool(_ngrok_started_by_app),
        'auto_start_enabled': str(os.environ.get('AUTO_START_NGROK', 'true')).strip().lower() not in ('0', 'false', 'no', 'off')
    }), 200


@mobile_bp.route('/api/mobile/ngrok/token', methods=['POST'])
def mobile_ngrok_token_management():
    """Local-only: save ngrok token and optionally start tunnel immediately."""
    local_only_error = _require_localhost_request()
    if local_only_error:
        return local_only_error

    from config import get_ngrok_authtoken, save_ngrok_authtoken, clear_ngrok_authtoken, save_mobile_remote_url, get_mobile_remote_url

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        action = str(data.get('action', 'save')).strip().lower()

        if action == 'clear':
            try:
                clear_ngrok_authtoken()
                stopped, stop_error = _stop_ngrok_tunnel_for_backend()
                return jsonify({
                    'success': True,
                    'has_token': False,
                    'ngrok_stopped': bool(stopped),
                    'warning': stop_error
                }), 200
            except Exception as e:
                logger.error(f"Failed clearing ngrok token: {e}", exc_info=True)
                return create_error_response('NGROK_TOKEN_CLEAR_ERROR', 'Failed to clear ngrok token', status_code=500)

        token = str(data.get('authtoken', '')).strip()
        if not token:
            return create_error_response('MISSING_NGROK_TOKEN', 'ngrok token is required', status_code=400)

        try:
            save_ngrok_authtoken(token)
        except ValueError as e:
            return create_error_response('INVALID_NGROK_TOKEN', str(e), status_code=400)
        except Exception:
            return create_error_response('NGROK_TOKEN_SAVE_ERROR', 'Failed to save ngrok token', status_code=500)

        configured, configure_error = _apply_ngrok_authtoken_to_agent(token)
        if not configured:
            logger.warning(f"ngrok token saved but ngrok config command failed: {configure_error}")

        start_now = bool(data.get('start_now', True))
        public_url = None
        start_error = None
        if start_now:
            port = int(request.environ.get('SERVER_PORT', 8000) or 8000)
            public_url, start_error = _start_ngrok_tunnel_for_backend(port)
            if public_url and not (os.environ.get('MOBILE_REMOTE_URL') or '').strip():
                try:
                    save_mobile_remote_url(public_url)
                except Exception as e:
                    logger.warning(f"Could not save auto-detected ngrok URL: {e}")

        return jsonify({
            'success': True,
            'has_token': bool(get_ngrok_authtoken()),
            'ngrok_configured': bool(configured),
            'warning': configure_error,
            'public_url': public_url,
            'start_error': start_error,
            'remote_url': get_mobile_remote_url()
        }), 200


@mobile_bp.route('/api/mobile/remote-url/auto-ngrok', methods=['POST'])
def mobile_remote_url_auto_ngrok():
    """Local-only: detect (or start) ngrok HTTPS URL and save as mobile remote URL."""
    local_only_error = _require_localhost_request()
    if local_only_error:
        return local_only_error

    from config import get_mobile_remote_url, save_mobile_remote_url

    env_remote_url = (os.environ.get('MOBILE_REMOTE_URL') or '').strip()
    if env_remote_url:
        return create_error_response(
            'REMOTE_URL_LOCKED',
            'Remote URL is managed by MOBILE_REMOTE_URL environment variable',
            status_code=409
        )

    ngrok_public_url, discover_error = _discover_ngrok_public_url()
    if not ngrok_public_url:
        port = int(request.environ.get('SERVER_PORT', 8000) or 8000)
        ngrok_public_url, start_error = _start_ngrok_tunnel_for_backend(port)

        if not ngrok_public_url and start_error:
            discover_error = start_error

    if not ngrok_public_url:
        return create_error_response(
            'NGROK_DISCOVERY_FAILED',
            discover_error or 'Unable to detect ngrok URL',
            suggestion='Configure ngrok token, then click AUTO NGROK again',
            status_code=503
        )

    try:
        save_mobile_remote_url(ngrok_public_url)
        return jsonify({
            'success': True,
            'remote_url': get_mobile_remote_url(),
            'detected_public_url': ngrok_public_url
        }), 200
    except ValueError as e:
        return create_error_response('INVALID_REMOTE_URL', str(e), status_code=400)
    except Exception as e:
        logger.error(f"Error auto-saving ngrok remote URL: {e}", exc_info=True)
        return create_error_response('REMOTE_URL_ERROR', 'Failed to save ngrok remote URL', status_code=500)

@mobile_bp.route('/api/mobile/auth', methods=['POST'])
def mobile_auth():
    try:
        client_ip = _get_client_ip()
        lockout_seconds = _check_mobile_auth_lockout_seconds(client_ip)
        if lockout_seconds > 0:
            logger.warning(f"Mobile auth blocked by rate limit from {client_ip}")
            return create_error_response(
                'RATE_LIMITED',
                'Too many failed attempts. Try again later.',
                details={'retry_after_seconds': lockout_seconds},
                status_code=429
            )

        data = request.get_json() or {}
        password = data.get('password', '').strip()

        if not password:
            logger.warning(f"Mobile auth failed: missing password from {client_ip}")
            return create_error_response('MISSING_PASSWORD', 'Password required', status_code=400)

        # Validate password (constant-time comparison)
        from config import validate_mobile_password
        if not validate_mobile_password(password):
            new_lockout = _record_mobile_auth_failure(client_ip)
            logger.warning(f"Mobile auth failed: invalid password from {client_ip}")
            if new_lockout > 0:
                return create_error_response(
                    'RATE_LIMITED',
                    'Too many failed attempts. Try again later.',
                    details={'retry_after_seconds': new_lockout},
                    status_code=429
                )
            return create_error_response('INVALID_PASSWORD', 'Incorrect password', status_code=401)

        _clear_mobile_auth_failures(client_ip)
        session_token = _issue_mobile_session_token(client_ip)

        # Get available catalogs efficiently
        from snapshot_manager import list_snapshots, get_loaded_snapshot_info

        try:
            catalogs_dict = list_snapshots()
            loaded_info = get_loaded_snapshot_info()
            loaded_file = loaded_info.get('snapshot_file', '')

            # Get historical catalogs (most common use case for mobile)
            # Combine both historical and new catalogs for selection
            historical_catalogs = catalogs_dict.get('historical', [])
            new_catalogs = catalogs_dict.get('new', [])
            all_catalogs = historical_catalogs + new_catalogs

            # Limit response to first 50 catalogs to prevent memory issues
            catalogs_response = []
            loaded_catalogs = []
            from snapshot_manager import get_snapshot_connection

            for catalog in all_catalogs[:50]:
                catalog_file = catalog.get('snapshot_file', '')
                is_loaded = catalog_file == loaded_file

                # Check catalog compatibility (has metadata_schema table)
                is_compatible = True
                try:
                    catalog_path = os.path.join(BACKEND_DIR, 'catalogs', catalog_file)
                    if os.path.exists(catalog_path):
                        with get_snapshot_connection(catalog_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute('''
                                SELECT name FROM sqlite_master
                                WHERE type='table' AND name='metadata_schema'
                            ''')
                            is_compatible = cursor.fetchone() is not None
                except Exception as e:
                    logger.debug(f"Could not check compatibility for {catalog_file}: {e}")
                    is_compatible = False

                catalog_info = {
                    'id': catalog_file,
                    'name': catalog.get('name', 'Unnamed'),
                    'product_count': catalog.get('product_count', 0),
                    'is_loaded': is_loaded,
                    'is_compatible': is_compatible
                }

                # Add warning suffix for incompatible catalogs
                if not is_compatible:
                    catalog_info['name'] += ' ⚠️ (Incompatible - Old Version)'

                catalogs_response.append(catalog_info)

                if is_loaded:
                    loaded_catalogs.append(catalog.get('name', 'Unnamed'))

            logger.debug(f"Mobile auth successful from {client_ip}")
            logger.debug(f"📱 Currently loaded file: '{loaded_file}'")
            logger.debug(f"📱 Returned {len(catalogs_response)} catalogs, {len(loaded_catalogs)} marked as loaded")
            if loaded_catalogs:
                logger.debug(f"📱 Loaded catalogs: {loaded_catalogs}")
                logger.debug(f"📱 Catalog IDs: {[c['id'] for c in catalogs_response]}")

            return jsonify({
                'valid': True,
                'session_token': session_token,
                'expires_in_seconds': MOBILE_SESSION_TTL_SECONDS,
                'catalogs': catalogs_response,
                'modes': ['mode1', 'mode3']
            }), 200

        except Exception as e:
            logger.error(f"Error getting catalogs for mobile: {e}", exc_info=True)
            return create_error_response('CATALOG_ERROR', 'Failed to load catalogs', status_code=500)

    except Exception as e:
        logger.error(f"Mobile auth error: {e}", exc_info=True)
        return create_error_response('AUTH_ERROR', 'Authentication failed', status_code=500)

@mobile_bp.route('/api/mobile/config', methods=['GET'])
def mobile_config():
    try:
        _, auth_error = _require_mobile_session()
        if auth_error:
            return auth_error

        from snapshot_manager import get_loaded_snapshot_info

        try:
            loaded_info = get_loaded_snapshot_info()

            return jsonify({
                'authorized': True,
                'loaded_catalog': {
                    'name': loaded_info.get('name', 'Unknown'),
                    'file': loaded_info.get('snapshot_file', ''),
                    'product_count': loaded_info.get('product_count', 0)
                }
            }), 200

        except Exception as e:
            logger.error(f"Error getting loaded catalog for mobile: {e}", exc_info=True)
            return create_error_response('CONFIG_ERROR', 'Failed to load config', status_code=500)

    except Exception as e:
        logger.error(f"Mobile config error: {e}", exc_info=True)
        return create_error_response('CONFIG_ERROR', 'Config request failed', status_code=500)

@mobile_bp.route('/api/mobile/catalog-schema', methods=['GET'])
def get_catalog_schema():
    
    try:
        client_ip, auth_error = _require_mobile_session()
        if auth_error:
            return auth_error

        catalog_id = request.args.get('catalog_id', '').strip()
        if not catalog_id:
            logger.warning(f"Mobile catalog-schema: missing catalog_id from {client_ip}")
            return create_error_response('MISSING_CATALOG', 'catalog_id required', status_code=400)

        # Validate catalog_id is safe (prevent path traversal)
        if '..' in catalog_id or '/' in catalog_id or '\\' in catalog_id:
            logger.warning(f"Mobile catalog-schema: suspicious catalog_id from {client_ip}")
            return create_error_response('INVALID_CATALOG', 'Invalid catalog_id', status_code=400)

        from snapshot_manager import get_snapshot_connection
        catalog_path = os.path.join(BACKEND_DIR, 'catalogs', catalog_id)

        if not os.path.exists(catalog_path):
            logger.warning(f"Mobile catalog-schema: catalog not found: {catalog_id}")
            return create_error_response('CATALOG_NOT_FOUND', 'Catalog not found', status_code=404)

        try:
            with get_snapshot_connection(catalog_path) as conn:
                cursor = conn.cursor()

                # Check if metadata_schema table exists (old catalogs might not have it)
                cursor.execute('''
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='metadata_schema'
                ''')
                has_metadata_schema = cursor.fetchone() is not None

                metadata_fields = []

                if has_metadata_schema:
                    # Get dynamic metadata fields from metadata_schema
                    # These apply to HISTORICAL products (is_historical=1)
                    cursor.execute('''
                        SELECT column_name, data_type, display_name
                        FROM metadata_schema
                        WHERE is_active = 1
                        ORDER BY rowid
                    ''')

                    rows = cursor.fetchall()
                    for row in rows:
                        metadata_fields.append({
                            'column_name': row[0],
                            'data_type': row[1],
                            'display_name': row[2] or row[0]
                        })
                else:
                    logger.debug(f"Mobile catalog-schema: Catalog {catalog_id} is missing metadata_schema table (old/incompatible version)")

            if len(metadata_fields) == 0:
                logger.debug(f"Mobile catalog-schema: no metadata fields found for {catalog_id}. Only base fields available.")
            else:
                logger.debug(f"Mobile catalog-schema: returned {len(metadata_fields)} metadata fields for {catalog_id}: {[f['column_name'] for f in metadata_fields]}")

            return jsonify({
                'base_fields': [
                    {
                        'column_name': 'category',
                        'data_type': 'string',
                        'display_name': 'Category'
                    }
                ],
                'metadata_fields': metadata_fields,
                'catalog_id': catalog_id,
                'note': 'All fields are optional - leave blank if not applicable'
            }), 200

        except Exception as e:
            logger.error(f"Error getting catalog schema for {catalog_id}: {e}", exc_info=True)
            return create_error_response('SCHEMA_ERROR', 'Failed to load catalog schema', status_code=500)

    except Exception as e:
        logger.error(f"Mobile catalog-schema error: {e}", exc_info=True)
        return create_error_response('SCHEMA_ERROR', 'Schema request failed', status_code=500)

@mobile_bp.route('/api/mobile/catalog-categories/clear-cache', methods=['POST'])
def clear_catalog_categories_cache():
    
    try:
        _, auth_error = _require_mobile_session()
        if auth_error:
            return auth_error

        catalog_id = request.args.get('catalog_id', None)
        invalidate_catalog_categories_cache(catalog_id)

        message = f'Cleared category cache for {catalog_id}' if catalog_id else 'Cleared all category caches'
        logger.debug(f"[CACHE] {message}")

        return jsonify({
            'status': 'success',
            'message': message
        }), 200

    except Exception as e:
        logger.error(f"Error clearing category cache: {e}", exc_info=True)
        return create_error_response('CACHE_ERROR', 'Failed to clear cache', status_code=500)


@mobile_bp.route('/api/mobile/catalog-categories', methods=['GET'])
def get_catalog_categories():
    
    try:
        client_ip, auth_error = _require_mobile_session()
        if auth_error:
            return auth_error

        catalog_id = request.args.get('catalog_id', '').strip()
        if not catalog_id:
            logger.warning(f"Mobile catalog-categories: missing catalog_id from {client_ip}")
            return create_error_response('MISSING_CATALOG', 'catalog_id required', status_code=400)

        # Validate catalog_id is safe (prevent path traversal)
        if '..' in catalog_id or '/' in catalog_id or '\\' in catalog_id:
            logger.warning(f"Mobile catalog-categories: suspicious catalog_id from {client_ip}")
            return create_error_response('INVALID_CATALOG', 'Invalid catalog_id', status_code=400)

        from snapshot_manager import get_snapshot_connection
        catalog_path = os.path.join(BACKEND_DIR, 'catalogs', catalog_id)

        if not os.path.exists(catalog_path):
            logger.warning(f"Mobile catalog-categories: catalog not found: {catalog_id}")
            return create_error_response('CATALOG_NOT_FOUND', 'Catalog not found', status_code=404)

        # PERFORMANCE: Check cache first (avoids DISTINCT query on huge catalogs)
        cached_categories = get_cached_catalog_categories(catalog_id)
        if cached_categories is not None:
            logger.debug(f"Mobile catalog-categories: cache hit for {catalog_id} ({len(cached_categories)} categories)")
            return jsonify({
                'categories': cached_categories,
                'catalog_id': catalog_id,
                'cached': True
            }), 200

        try:
            with get_snapshot_connection(catalog_path) as conn:
                cursor = conn.cursor()

                # Ensure index exists for performance (idempotent - safe to run multiple times)
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_products_category
                    ON products(category)
                ''')

                # Get unique categories from products
                # OPTIMIZED: Index on category column makes DISTINCT fast even for huge catalogs
                cursor.execute('''
                    SELECT DISTINCT category
                    FROM products
                    WHERE category IS NOT NULL AND category != ''
                    ORDER BY category
                ''')

                categories = [row[0] for row in cursor.fetchall()]

            # Cache the result for future requests
            cache_catalog_categories(catalog_id, categories)

            logger.debug(f"Mobile catalog-categories: returned {len(categories)} categories for {catalog_id}")

            return jsonify({
                'categories': categories,
                'catalog_id': catalog_id,
                'cached': False
            }), 200

        except Exception as e:
            logger.error(f"Error getting categories for {catalog_id}: {e}", exc_info=True)
            return create_error_response('CATEGORIES_ERROR', 'Failed to load categories', status_code=500)

    except Exception as e:
        logger.error(f"Mobile catalog-categories error: {e}", exc_info=True)
        return create_error_response('CATEGORIES_ERROR', 'Categories request failed', status_code=500)

@mobile_bp.route('/api/mobile/log', methods=['POST'])
def mobile_log():
    """Accept log messages from mobile frontend for server-side logging

    Request JSON:
        {
            "level": "info" | "warning" | "error",
            "message": "Log message",
            "data": {} (optional)
        }
    """
    try:
        _, auth_error = _require_mobile_session()
        if auth_error:
            return auth_error

        data = request.get_json(silent=True) or {}
        level = str(data.get('level', 'info')).lower()
        message = str(data.get('message', ''))
        extra_data = data.get('data', {})

        # Guardrails: prevent oversized client log payloads
        if len(message) > 500:
            message = message[:500] + '...'

        log_message = f"📱 [MOBILE] {message}"
        if extra_data:
            extra_preview = str(extra_data)
            if len(extra_preview) > 300:
                extra_preview = extra_preview[:300] + '...'
            log_message += f" | Data: {extra_preview}"

        if level == 'error':
            logger.error(log_message)
        elif level == 'warning':
            logger.warning(log_message)
        elif level == 'debug':
            logger.debug(log_message)
        else:
            # Keep client info logs out of normal server logs
            logger.debug(log_message)

        return jsonify({'logged': True}), 200
    except Exception as e:
        logger.error(f"Mobile log error: {e}")
        return jsonify({'logged': False}), 500

@mobile_bp.route('/api/mobile/password', methods=['GET', 'POST'])
def mobile_password_management():
    
    try:
        local_only_error = _require_localhost_request()
        if local_only_error:
            return local_only_error

        from config import get_mobile_password, save_mobile_password

        if request.method == 'GET':
            try:
                password = get_mobile_password()
                return jsonify({'password': password}), 200
            except Exception as e:
                logger.error(f"Error getting mobile password: {e}")
                return create_error_response('PASSWORD_ERROR', 'Failed to get password', status_code=500)

        # POST: Update password
        try:
            data = request.get_json() or {}

            # Backward-compatible generate mode used by desktop UI
            action = str(data.get('action', '')).strip().lower()
            if action == 'generate':
                new_password = f"{secrets.randbelow(1_000_000):06d}"
                save_mobile_password(new_password)
                _revoke_all_mobile_sessions()
                logger.debug(f"Mobile password generated from {request.remote_addr}")
                return jsonify({
                    'success': True,
                    'password': new_password,
                    'message': 'Mobile password generated'
                }), 200

            # Manual update mode (expects 6-digit password)
            new_password = str(data.get('new_password', data.get('password', ''))).strip()

            # Validate password format
            if not new_password:
                logger.warning(f"Mobile password update: empty password from {request.remote_addr}")
                return create_error_response('INVALID_PASSWORD', 'Password required', status_code=400)

            if len(new_password) != 6:
                logger.warning(f"Mobile password update: wrong length ({len(new_password)}) from {request.remote_addr}")
                return create_error_response('INVALID_PASSWORD', 'Must be exactly 6 characters', status_code=400)

            if not new_password.isdigit():
                logger.warning(f"Mobile password update: non-digit password from {request.remote_addr}")
                return create_error_response('INVALID_PASSWORD', 'Must contain only digits (0-9)', status_code=400)

            save_mobile_password(new_password)
            _revoke_all_mobile_sessions()
            logger.debug(f"Mobile password updated from {request.remote_addr}")

            return jsonify({
                'success': True,
                'password': new_password,
                'message': 'Mobile password updated'
            }), 200

        except Exception as e:
            logger.error(f"Error saving mobile password: {e}", exc_info=True)
            return create_error_response('PASSWORD_ERROR', 'Failed to save password', status_code=500)

    except Exception as e:
        logger.error(f"Mobile password management error: {e}", exc_info=True)
        return create_error_response('PASSWORD_ERROR', 'Password request failed', status_code=500)

@mobile_bp.route('/api/mobile/upload-and-match', methods=['POST'])
def mobile_upload_and_match():
    
    try:
        client_ip, auth_error = _require_mobile_session()
        if auth_error:
            return auth_error

        # Validate required form data
        if 'image' not in request.files:
            return create_error_response('MISSING_IMAGE', 'Image file required', status_code=400)

        catalog_id = request.form.get('catalog_id', '').strip()
        if not catalog_id:
            return create_error_response('MISSING_CATALOG', 'catalog_id required', status_code=400)

        mode = request.form.get('mode', '').strip().lower()
        if mode not in ['mode1', 'mode3']:
            return create_error_response('INVALID_MODE', 'mode must be "mode1" or "mode3"', status_code=400)

        # Validate catalog_id is safe
        if '..' in catalog_id or '/' in catalog_id or '\\' in catalog_id:
            logger.warning(f"[MOBILE] Suspicious catalog_id from {client_ip}")
            return create_error_response('INVALID_CATALOG', 'Invalid catalog_id', status_code=400)

        from snapshot_manager import get_snapshot_connection
        catalog_path = os.path.join(BACKEND_DIR, 'catalogs', catalog_id)

        if not os.path.exists(catalog_path):
            logger.warning(f"[MOBILE] Catalog not found: {catalog_id}")
            return create_error_response('CATALOG_NOT_FOUND', 'Catalog not found', status_code=404)

        try:
            import json
            import uuid

            # STEP 1: Load catalog (use_existing mode - same as desktop)
            logger.debug(f"[MOBILE] Step 1: Loading catalog {catalog_id}")
            from snapshot_manager import load_snapshot_to_main_db, get_loaded_snapshot_info

            loaded_info = get_loaded_snapshot_info()
            currently_loaded = loaded_info.get('snapshot_file') if loaded_info.get('loaded') else None
            if currently_loaded == catalog_id:
                logger.debug(f"[MOBILE] Catalog {catalog_id} already loaded, skipping reload")
            else:
                load_result = load_snapshot_to_main_db(catalog_id)
                if load_result.get('error'):
                    logger.error(f"[MOBILE] Failed to load catalog: {load_result['error']}")
                    return create_error_response('LOAD_ERROR', 'Failed to load catalog', status_code=500)
                logger.debug(f"[MOBILE] Catalog loaded: {load_result.get('product_count')} products")

            # STEP 2: Load historical products (skip batch-upload, use_existing mode)
            logger.debug(f"[MOBILE] Step 2: Loading historical products (use_existing)")
            # In use_existing mode, backend just loads existing products from DB, no upload
            # Frontend state will be updated when batch-match response comes back

            # STEP 3: Clear new section (replace mode) - silent operation
            logger.debug(f"[MOBILE] Step 3: Clearing new section (replace mode)")
            try:
                from database import clear_products_by_type

                cleanup_result = clear_products_by_type('new')
                deleted_count = cleanup_result.get('products_deleted', 0)
                logger.debug(f"[MOBILE] Cleared {deleted_count} new products")
            except Exception as e:
                logger.warning(f"[MOBILE] Cleanup error (non-fatal): {e}")
                # Continue anyway - products may be empty already

            # STEP 4: Batch upload new product
            logger.debug(f"[MOBILE] Step 4: Uploading new product via batch-upload")

            file = request.files['image']
            if file.filename == '':
                return create_error_response('EMPTY_FILENAME', 'No file selected', status_code=400)

            if not allowed_file(file.filename):
                return create_error_response('INVALID_FORMAT', 'Unsupported file format', status_code=400)

            # Get product fields
            category = request.form.get('category', None)
            product_name = request.form.get('product_name', None)
            sku = request.form.get('sku', None)

            # Normalize empty strings
            if category and category.strip() == '':
                category = None
            if product_name and product_name.strip() == '':
                product_name = None
            if sku and sku.strip() == '':
                sku = None

            # Normalize category (lowercase, trim whitespace, handle "unknown" variations)
            if category is not None:
                from product_matching import normalize_category
                category = normalize_category(category)

            # Save image file
            file_ext = secure_filename(file.filename).rsplit('.', 1)[1].lower() if '.' in secure_filename(file.filename) else 'jpg'
            unique_filename = f"{uuid.uuid4()}.{file_ext}"
            uploads_dir = os.path.join(BACKEND_DIR, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            filepath = os.path.join(uploads_dir, unique_filename)

            try:
                file.save(filepath)
                logger.debug(f"[MOBILE] Image saved: {filepath}")
            except Exception as e:
                logger.error(f"[MOBILE] Failed to save image: {e}")
                return create_error_response('SAVE_ERROR', 'Failed to save image', status_code=500)

            # Call batch-upload backend functions directly (same as desktop batch-upload)
            try:
                from database import insert_product, insert_features
                from feature_extraction_service import extract_features_unified

                # Handle mode 3 metadata if provided
                metadata = None
                if mode == 'mode3':
                    # Collect all metadata_<column_name> fields from request form
                    metadata_dict = {}
                    for key in request.form.keys():
                        if key.startswith('metadata_'):
                            column_name = key.replace('metadata_', '')
                            value = request.form.get(key)
                            if value and str(value).strip() != '':
                                metadata_dict[column_name] = value

                    if metadata_dict:
                        metadata = json.dumps(metadata_dict)
                        logger.debug(f"[MOBILE] Mode 3 metadata collected: {list(metadata_dict.keys())}")

                if category is None and metadata:
                    inferred_category = _extract_category_from_metadata_payload(metadata)
                    if inferred_category is not None:
                        category = inferred_category
                        logger.debug("[MOBILE] Backfilled missing category from metadata payload")

                product_id = insert_product(
                    image_path=filepath,
                    category=category,
                    product_name=product_name,
                    sku=sku,
                    is_historical=False,
                    metadata=metadata
                )
                logger.debug(f"[MOBILE] Product inserted: ID {product_id}")

                # Extract features
                try:
                    features, embedding_type, embedding_version = extract_features_unified(filepath)
                    insert_features(
                        product_id=product_id,
                        color_features=features['color_features'],
                        shape_features=features['shape_features'],
                        texture_features=features['texture_features'],
                        embedding_type=embedding_type,
                        embedding_version=embedding_version
                    )
                    logger.debug(f"[MOBILE] Features extracted for product {product_id}")
                except Exception as e:
                    logger.warning(f"[MOBILE] Feature extraction failed (non-fatal): {e}")

            except Exception as e:
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                logger.error(f"[MOBILE] Upload failed: {e}")
                return create_error_response('UPLOAD_ERROR', 'Failed to upload product', status_code=500)

            # STEP 5: Batch match (same as desktop REPLACE & PROCESS)
            logger.debug(f"[MOBILE] Step 5: Matching product {product_id} (mode {mode})")

            try:
                from product_matching import batch_find_matches
                from hybrid_matching import batch_find_hybrid_matches
                from database import get_metadata_schema

                matches = []

                if mode == 'mode1':
                    # Mode 1: Visual only - use batch version (single product wrapped in list)
                    batch_result = batch_find_matches(
                        product_ids=[product_id],
                        threshold=0,
                        limit=5,
                        match_against_all=False,
                        include_uncategorized=True,
                        store_matches=True,
                        skip_invalid_products=True,
                        preload_catalog=False
                    )
                    # Extract matches from batch result format
                    if batch_result['results'] and batch_result['results'][0]['status'] == 'success':
                        matches = batch_result['results'][0]['matches']
                    logger.debug(f"[MOBILE] Mode 1 visual matching: {len(matches)} results")
                elif mode == 'mode3':
                    # Mode 3: Hybrid matching (visual + metadata combined)
                    # Uses metadata schema created during CSV upload (part 1)
                    schema = get_metadata_schema()

                    if schema:
                        # Build weights from schema (equal weight for all columns)
                        metadata_weights = {col['column_name']: 1.0 for col in schema}
                        total = sum(metadata_weights.values())
                        metadata_weights = {k: v / total for k, v in metadata_weights.items()}

                        # Call batch hybrid matching (50% visual, 50% metadata)
                        batch_result = batch_find_hybrid_matches(
                            product_ids=[product_id],
                            threshold=0,
                            limit=5,
                            visual_weight=0.5,
                            metadata_weight=0.5,
                            metadata_weights=metadata_weights,
                            store_matches=True,
                            skip_invalid_products=True,
                            match_against_all=False
                        )
                        # Extract matches from batch result format
                        if batch_result['results'] and batch_result['results'][0]['status'] == 'success':
                            matches = batch_result['results'][0]['matches']
                        logger.debug(f"[MOBILE] Mode 3 hybrid matching: {len(matches)} results")
                    else:
                        # No metadata schema - fall back to visual (batch version)
                        logger.debug("[MOBILE] No metadata schema found - falling back to visual matching")
                        batch_result = batch_find_matches(
                            product_ids=[product_id],
                            threshold=0,
                            limit=5,
                            match_against_all=False,
                            include_uncategorized=True,
                            store_matches=True,
                            skip_invalid_products=True,
                            preload_catalog=False
                        )
                        # Extract matches from batch result format
                        if batch_result['results'] and batch_result['results'][0]['status'] == 'success':
                            matches = batch_result['results'][0]['matches']

            except Exception as e:
                logger.warning(f"[MOBILE] Matching error: {e}")
                matches = []

            def _coerce_json_dict(raw_value):
                if isinstance(raw_value, dict):
                    return raw_value
                if isinstance(raw_value, str):
                    try:
                        parsed = json.loads(raw_value)
                        return parsed if isinstance(parsed, dict) else {}
                    except Exception:
                        return {}
                return {}

            def _sanitize_scalar(value, max_len=120):
                if value is None:
                    return None
                if isinstance(value, (int, float, bool)):
                    return value
                if isinstance(value, str):
                    return value[:max_len]
                return str(value)[:max_len]

            def _sanitize_mobile_metadata_values(raw_values, metadata_scores_dict, max_keys=12):
                values = _coerce_json_dict(raw_values)
                if not values:
                    return {}

                sensitive_keys = {'image_path', 'created_at', 'updated_at', 'id', 'product_id', 'is_historical'}
                prioritized_keys = []
                if isinstance(metadata_scores_dict, dict):
                    prioritized_keys.extend([k for k in metadata_scores_dict.keys() if isinstance(k, str)])

                # Keep only keys relevant to score chips (plus cap for payload safety).
                sanitized = {}
                for key in prioritized_keys:
                    if key in sensitive_keys or key not in values:
                        continue
                    clean_value = _sanitize_scalar(values.get(key))
                    if clean_value is None:
                        continue
                    sanitized[key] = clean_value
                    if len(sanitized) >= max_keys:
                        break

                return sanitized

            def _sanitize_mobile_metadata_object(raw_meta, max_keys=20):
                values = _coerce_json_dict(raw_meta)
                if not values:
                    return {}

                sensitive_keys = {'image_path', 'created_at', 'updated_at', 'id', 'product_id', 'is_historical'}
                sanitized = {}
                for key, value in values.items():
                    if key in sensitive_keys:
                        continue
                    clean_value = _sanitize_scalar(value)
                    if clean_value is None:
                        continue
                    sanitized[str(key)] = clean_value
                    if len(sanitized) >= max_keys:
                        break
                return sanitized

            # Format matches
            detailed_matches_response = []
            legacy_matches_response = []
            for match in matches[:5]:
                match_id = match.get('product_id') or match.get('id')
                similarity_score = match.get('score', match.get('similarity_score', 0))

                metadata_scores = _coerce_json_dict(match.get('metadata_scores'))
                metadata_values = _sanitize_mobile_metadata_values(
                    match.get('metadata_values'),
                    metadata_scores
                )

                image_path = match.get('image_path') or ''
                filename = os.path.basename(image_path) if image_path else None

                image_url = None
                try:
                    if match_id is not None:
                        image_url = f"/api/products/{int(match_id)}/image"
                except Exception:
                    image_url = None

                detailed_matches_response.append({
                    'id': match_id,
                    'name': match.get('product_name') or match.get('name') or 'Unknown',
                    'category': match.get('category') or 'N/A',
                    'score': similarity_score,
                    'similarity_score': similarity_score,
                    'sku': match.get('sku'),
                    'filename': filename,
                    'image_url': image_url,
                    'color_score': match.get('color_score'),
                    'shape_score': match.get('shape_score'),
                    'texture_score': match.get('texture_score'),
                    'visual_score': match.get('visual_score'),
                    'metadata_score': match.get('metadata_score'),
                    'sku_score': match.get('sku_score'),
                    'name_score': match.get('name_score'),
                    'category_score': match.get('category_score'),
                    'price_score': match.get('price_score'),
                    'performance_score': match.get('performance_score'),
                    'metadata_scores': metadata_scores,
                    'metadata_values': metadata_values,
                    'is_potential_duplicate': bool(match.get('is_potential_duplicate'))
                })
                legacy_matches_response.append({
                    'id': match_id,
                    'name': match.get('product_name') or match.get('name') or 'Unknown',
                    'category': match.get('category') or 'N/A',
                    'score': similarity_score,
                    'sku': match.get('sku')
                })

            # Build single result group for mobile UI (desktop-style rendering)
            uploaded_product = None
            try:
                uploaded_product = get_product_by_id(product_id)
            except Exception as e:
                logger.debug(f"[MOBILE] Could not load uploaded product details: {e}")

            def _field(row, key, default=None):
                if row is None:
                    return default
                try:
                    value = row[key]
                except Exception:
                    if isinstance(row, dict):
                        value = row.get(key, default)
                    else:
                        value = default
                return default if value is None else value

            uploaded_name = _field(uploaded_product, 'product_name') or product_name or secure_filename(file.filename).rsplit('.', 1)[0]
            uploaded_category = _field(uploaded_product, 'category') or category or 'Uncategorized'
            uploaded_sku = _field(uploaded_product, 'sku') or sku

            uploaded_metadata = _sanitize_mobile_metadata_object(_field(uploaded_product, 'metadata'))

            similarity_values = []
            for item in detailed_matches_response:
                try:
                    similarity_values.append(float(item.get('similarity_score') or 0))
                except Exception:
                    similarity_values.append(0.0)

            top_score = max(similarity_values) if similarity_values else 0.0
            avg_score = (sum(similarity_values) / len(similarity_values)) if similarity_values else 0.0

            mobile_result_group = {
                'mode': mode,
                'query_product': {
                    'id': product_id,
                    'name': uploaded_name or f'Uploaded Product {product_id}',
                    'category': uploaded_category,
                    'sku': uploaded_sku,
                    'image_url': f"/api/products/{product_id}/image",
                    'metadata': uploaded_metadata
                },
                'summary': {
                    'total_matches': len(detailed_matches_response),
                    'top_score': round(top_score, 1),
                    'average_score': round(avg_score, 1)
                },
                'matches': detailed_matches_response
            }

            logger.debug(f"[MOBILE] Complete: Product {product_id}, {len(detailed_matches_response)} matches")

            # Invalidate CSV cache since catalog was loaded and products modified
            invalidate_csv_cache()

            # Return full response with all data for frontend to update state
            return jsonify({
                'success': True,
                'product_id': product_id,
                'mode': mode,
                'catalog_id': catalog_id,
                'upload_status': 'success',
                'mobile_result': mobile_result_group,
                'result_groups': [mobile_result_group],
                'result_group_count': 1,
                'matches': legacy_matches_response,
                'match_count': len(legacy_matches_response)
            }), 200

        except Exception as e:
            logger.error(f"[MOBILE] Orchestration error: {e}", exc_info=True)
            return create_error_response('ORCHESTRATION_ERROR', 'Processing failed', status_code=500)

    except Exception as e:
        logger.error(f"[MOBILE] Request error: {e}", exc_info=True)
        return create_error_response('MOBILE_ERROR', 'Request failed', status_code=500)

# Simple flag to notify main app that mobile results are ready
_mobile_results_flag = {'ready': False, 'timestamp': None}

@mobile_bp.route('/api/mobile/results-ready', methods=['POST'])
def mobile_results_ready():
    """Mobile notifies main app that results are ready

    Called by mobile-upload after successful match completion.
    Sets a flag that main app polls to know when to fetch results.

    Requires mobile session token from authenticated mobile page.
    """
    global _mobile_results_flag

    try:
        _, auth_error = _require_mobile_session()
        if auth_error:
            return auth_error

        import time
        _mobile_results_flag['ready'] = True
        _mobile_results_flag['timestamp'] = time.time()

        logger.debug("[MOBILE] Results ready flag set - notifying main app")

        return jsonify({
            'success': True,
            'message': 'Main app notified'
        }), 200
    except Exception as e:
        logger.error(f"[MOBILE] Failed to set results flag: {e}")
        return create_error_response('FLAG_ERROR', 'Failed to set results flag', status_code=500)

@mobile_bp.route('/api/mobile/check-flag', methods=['GET'])
def check_mobile_results_flag():
    """Main app checks if mobile has results ready

    Returns the current flag state.
    """
    global _mobile_results_flag

    try:
        local_only_error = _require_localhost_request()
        if local_only_error:
            return local_only_error

        flag_ready = _mobile_results_flag['ready']
        if flag_ready:
            logger.debug(f"[MOBILE] Check flag: ready={flag_ready} (will trigger results polling)")
        return jsonify({
            'ready': flag_ready,
            'timestamp': _mobile_results_flag['timestamp']
        }), 200
    except Exception as e:
        logger.error(f"[MOBILE] Failed to check flag: {e}")
        return create_error_response('CHECK_ERROR', 'Failed to check flag', status_code=500)

@mobile_bp.route('/api/mobile/clear-flag', methods=['POST'])
def clear_mobile_results_flag():
    """Main app clears the flag after displaying results

    Resets flag so mobile can set it again for next upload.
    """
    global _mobile_results_flag

    try:
        local_only_error = _require_localhost_request()
        if local_only_error:
            return local_only_error

        _mobile_results_flag['ready'] = False
        _mobile_results_flag['timestamp'] = None

        logger.debug("[MOBILE] Results flag cleared")

        return jsonify({
            'success': True,
            'message': 'Flag cleared'
        }), 200
    except Exception as e:
        logger.error(f"[MOBILE] Failed to clear flag: {e}")
        return create_error_response('CLEAR_ERROR', 'Failed to clear flag', status_code=500)

