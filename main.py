"""
Product Matching System - Desktop Application
Multi-window architecture using Flask backend with webview frontend.

Architecture:
- Main Window: Source of truth, stays in place, never navigates away
- Child Windows: CSV Builder and Catalog Manager open as separate windows
- File Staging: Child windows save output to staging/ directory and signal main app
"""
import sys
import os

# Enable UTF-8 mode for Windows console to support Unicode characters (▶, ✓, etc.)
# This must be set BEFORE any other imports
if sys.platform == 'win32':
    os.environ['PYTHONUTF8'] = '1'
    # Reconfigure stdout/stderr to use UTF-8 with error replacement
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import webview
import threading
import time
import platform
import base64
import json
import uuid
import atexit
import signal
import subprocess
import shutil
import urllib.request
import urllib.error
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app import app
from path_manager import get_staging_dir, get_downloads_dir
from config import get_ngrok_authtoken

# Global reference to main window
main_window = None
child_windows = {}

# Staging directory for inter-window communication
STAGING_DIR = get_staging_dir()
APP_PORT = 8000  # Use 8000 (ports 5000-5003 sometimes blocked by Windows)

# ngrok process state (optional best-effort auto-start)
ngrok_process = None
ngrok_started_by_app = False
cleanup_has_run = False


def ensure_staging_dir():
    """Create staging directory if it doesn't exist"""
    if not os.path.exists(STAGING_DIR):
        os.makedirs(STAGING_DIR)
    return STAGING_DIR


def clean_staging_dir():
    """Clean up old staging files (older than 1 hour)"""
    try:
        if not os.path.exists(STAGING_DIR):
            print("Staging directory does not exist, skipping cleanup")
            return

        cutoff = time.time() - 3600  # 1 hour ago
        files_deleted = 0
        space_freed = 0

        for filename in os.listdir(STAGING_DIR):
            filepath = os.path.join(STAGING_DIR, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                try:
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    files_deleted += 1
                    space_freed += file_size
                except Exception as e:
                    print(f"Warning: Failed to delete staging file {filename}: {e}")

        if files_deleted > 0:
            space_mb = round(space_freed / (1024 * 1024), 2)
            print(f"✓ Staging cleanup: Deleted {files_deleted} files (older than 1 hour), {space_mb}MB freed")
        else:
            print("Staging cleanup: No old files to remove")
    except Exception as e:
        print(f"Warning: Failed to clean staging directory: {e}")


class WebViewAPI:
    """API bridge for JavaScript to access native webview features"""

    def _get_downloads_folder(self):
        """Get the user's Downloads folder path (cross-platform)"""
        return get_downloads_dir()

    def save_file_auto(self, content, filename):
        """
        Auto-save file to Downloads folder (like browser downloads)
        """
        try:
            if isinstance(content, str) and content.startswith('data:'):
                content = content.split(',', 1)[1]
                content = base64.b64decode(content)
            elif isinstance(content, str):
                content = content.encode('utf-8')

            downloads_folder = self._get_downloads_folder()
            filepath = os.path.join(downloads_folder, filename)

            base, ext = os.path.splitext(filepath)
            counter = 1
            while os.path.exists(filepath):
                filepath = f"{base} ({counter}){ext}"
                counter += 1

            with open(filepath, 'wb') as f:
                f.write(content)

            return filepath
        except Exception as e:
            print(f"Error auto-saving file: {e}")
            return None

    def save_file(self, content, filename, file_types=('CSV Files (*.csv)', 'All Files (*.*)')):
        """
        Save file using native file dialog
        """
        try:
            if isinstance(content, str) and content.startswith('data:'):
                content = content.split(',', 1)[1]
                content = base64.b64decode(content)
            elif isinstance(content, str):
                content = content.encode('utf-8')

            file_dialog_enum = getattr(webview, 'FileDialog', None)
            save_dialog_type = file_dialog_enum.SAVE if file_dialog_enum and hasattr(file_dialog_enum, 'SAVE') else webview.SAVE_DIALOG

            result = webview.windows[0].create_file_dialog(
                save_dialog_type,
                save_filename=filename,
                file_types=file_types
            )

            if result:
                filepath = result if isinstance(result, str) else result[0]
                with open(filepath, 'wb') as f:
                    f.write(content)
                return filepath
            return None
        except Exception as e:
            print(f"Error saving file: {e}")
            return None

    def select_folder(self):
        """
        Open native folder selection dialog.
        Returns list of image file path info objects from selected folder, or None if cancelled.
        MEMORY OPTIMIZATION: Returns only file paths, not base64 data. Images are processed directly from disk.
        """
        try:
            file_dialog_enum = getattr(webview, 'FileDialog', None)
            if file_dialog_enum and hasattr(file_dialog_enum, 'FOLDER'):
                folder_dialog_type = file_dialog_enum.FOLDER
            else:
                folder_dialog_type = webview.FOLDER_DIALOG

            result = webview.windows[0].create_file_dialog(
                folder_dialog_type
            )

            if not result:
                return None

            folder_path = result[0] if isinstance(result, tuple) else result
            print(f"[FOLDER] Selected: {folder_path}")

            # Get the selected folder name to prefix paths (mimics browser webkitRelativePath behavior)
            # Browser gives: "SelectedFolder/Subfolder/image.jpg"
            # We need to match this format for category detection
            folder_name = os.path.basename(folder_path)

            # Collect all image files from folder and subfolders
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
            files_info = []

            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in image_extensions:
                        full_path = os.path.join(root, filename)
                        # Get relative path from selected folder
                        rel_path = os.path.relpath(full_path, folder_path)
                        # Prefix with folder name to match browser webkitRelativePath format
                        # "Subfolder/image.jpg" -> "SelectedFolder/Subfolder/image.jpg"
                        webkit_style_path = os.path.join(folder_name, rel_path).replace(os.sep, '/')

                        try:
                            files_info.append({
                                'name': filename,
                                'path': full_path,  # Absolute file path for backend to read directly
                                'relativePath': webkit_style_path,
                                'size': os.path.getsize(full_path)
                            })
                        except Exception as e:
                            print(f"[FOLDER] Error getting file info for {filename}: {e}")

            print(f"[FOLDER] Found {len(files_info)} images - returning file paths only")
            return files_info

        except Exception as e:
            print(f"Error selecting folder: {e}")
            return None

    # ========== Multi-Window API ==========

    def open_csv_builder(self, current_mode='visual', target_section='historical', staging_window_id=None):
        """
        Open CSV Builder in a new child window.
        Args:
            current_mode: The matching mode ('visual', 'metadata', 'hybrid')
            target_section: Which section to load CSV into ('historical' or 'new')
            staging_window_id: Optional window ID for fetching staged file data
        Returns:
            Window ID for tracking
        """
        global child_windows

        try:
            port = APP_PORT
            window_id = staging_window_id or f"csv_builder_{uuid.uuid4().hex[:8]}"

            # Pass context via query params
            url = f'http://127.0.0.1:{port}/static/csv-builder.html?mode={current_mode}&section={target_section}&window_id={window_id}'

            child_window = webview.create_window(
                'CSV Builder',
                url,
                width=1000,
                height=700,
                resizable=True,
                min_size=(800, 600),
                text_select=True,
                js_api=ChildWindowAPI(window_id, 'csv_builder'),
            )

            child_windows[window_id] = child_window
            print(f"[WINDOW] Opened CSV Builder: {window_id}")
            return window_id

        except Exception as e:
            print(f"Error opening CSV Builder: {e}")
            return None

    def open_catalog_manager(self):
        """
        Open Catalog Manager in a new child window.
        Returns:
            Window ID for tracking
        """
        global child_windows

        try:
            port = APP_PORT
            window_id = f"catalog_manager_{uuid.uuid4().hex[:8]}"

            url = f'http://127.0.0.1:{port}/catalog-manager?window_id={window_id}'

            child_window = webview.create_window(
                'Catalog Manager',
                url,
                width=1100,
                height=800,
                resizable=True,
                min_size=(900, 600),
                text_select=True,
                js_api=ChildWindowAPI(window_id, 'catalog_manager'),
            )

            child_windows[window_id] = child_window
            print(f"[WINDOW] Opened Catalog Manager: {window_id}")
            return window_id

        except Exception as e:
            print(f"Error opening Catalog Manager: {e}")
            return None

    def check_staged_file(self, section='historical'):
        """
        Check for a staged CSV file from CSV Builder.
        Returns:
            Dict with file path and metadata if found, None otherwise
        """
        try:
            staging_dir = ensure_staging_dir()
            manifest_path = os.path.join(staging_dir, f'{section}_manifest.json')

            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)

                # Verify the CSV file exists
                csv_path = manifest.get('csv_path')
                if csv_path and os.path.exists(csv_path):
                    return manifest

            return None
        except Exception as e:
            print(f"Error checking staged file: {e}")
            return None

    def consume_staged_file(self, section='historical'):
        """
        Read and consume a staged CSV file (deletes after reading).
        Returns:
            Dict with csv_content and metadata, or None
        """
        try:
            staging_dir = ensure_staging_dir()
            manifest_path = os.path.join(staging_dir, f'{section}_manifest.json')

            if not os.path.exists(manifest_path):
                return None

            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            csv_path = manifest.get('csv_path')
            if not csv_path or not os.path.exists(csv_path):
                return None

            # Read CSV content
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_content = f.read()

            # Clean up staging files
            os.remove(csv_path)
            os.remove(manifest_path)

            return {
                'csv_content': csv_content,
                'section': section,
                'timestamp': manifest.get('timestamp'),
                'product_count': manifest.get('product_count', 0)
            }

        except Exception as e:
            print(f"Error consuming staged file: {e}")
            return None

    def notify_main_window(self, event_type, data=None):
        """
        Send a notification to the main window (used by child windows).
        """
        global main_window
        try:
            if main_window:
                # Evaluate JavaScript in main window to handle the event
                js_code = f"window.handleChildWindowEvent && window.handleChildWindowEvent('{event_type}', {json.dumps(data or {})})"
                main_window.evaluate_js(js_code)
                return True
        except Exception as e:
            print(f"Error notifying main window: {e}")
        return False


class ChildWindowAPI(WebViewAPI):
    """Extended API for child windows with staging file support"""

    def __init__(self, window_id, window_type):
        super().__init__()
        self.window_id = window_id
        self.window_type = window_type

    def save_to_staging(self, csv_content, section='historical', product_count=0):
        """
        Save CSV content to staging directory and create manifest.
        This is called by CSV Builder when user clicks "Load" in Step 5.
        Returns:
            Path to the manifest file, or None on error
        """
        try:
            staging_dir = ensure_staging_dir()
            timestamp = datetime.now().isoformat()

            # Save CSV file
            csv_filename = f'{section}_{self.window_id}.csv'
            csv_path = os.path.join(staging_dir, csv_filename)

            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)

            # Create manifest
            manifest = {
                'csv_path': csv_path,
                'section': section,
                'timestamp': timestamp,
                'product_count': product_count,
                'window_id': self.window_id
            }

            manifest_path = os.path.join(staging_dir, f'{section}_manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)

            print(f"[STAGING] Saved CSV for {section}: {product_count} products")
            return manifest_path

        except Exception as e:
            print(f"Error saving to staging: {e}")
            return None

    def close_and_notify(self, event_type, data=None):
        """
        Close this child window and notify the main window.
        """
        global child_windows, main_window

        try:
            # Notify main window first
            if main_window:
                js_code = f"window.handleChildWindowEvent && window.handleChildWindowEvent('{event_type}', {json.dumps(data or {})})"
                main_window.evaluate_js(js_code)

            # Close this window
            if self.window_id in child_windows:
                window = child_windows[self.window_id]
                del child_windows[self.window_id]
                window.destroy()

            return True
        except Exception as e:
            print(f"Error in close_and_notify: {e}")
            return False

    def signal_catalog_change(self, action, details=None):
        """
        Signal that catalog has changed (used by Catalog Manager).
        Main window can refresh its catalog display.
        """
        return self.notify_main_window('catalog_changed', {
            'action': action,
            'details': details or {}
        })

    def close_window(self):
        """
        Close this child window without notifying main app.
        Used for simple close/cancel actions.
        """
        global child_windows

        try:
            if self.window_id in child_windows:
                window = child_windows[self.window_id]
                del child_windows[self.window_id]
                window.destroy()
            return True
        except Exception as e:
            print(f"Error closing window: {e}")
            return False

def _is_truthy_env(value):
    return str(value or '').strip().lower() in ('1', 'true', 'yes', 'on')


def _should_auto_start_ngrok():
    # Enabled by default; set AUTO_START_NGROK=false to disable.
    raw = os.environ.get('AUTO_START_NGROK')
    if raw is None:
        return True
    return _is_truthy_env(raw)


def _discover_ngrok_https_url(timeout_seconds=0.6):
    api_url = (os.environ.get('NGROK_API_URL') or 'http://127.0.0.1:4040/api/tunnels').strip()
    timeout_seconds = max(0.2, min(float(timeout_seconds), 5.0))
    try:
        req = urllib.request.Request(api_url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode('utf-8', errors='replace'))
    except Exception:
        return None

    tunnels = payload.get('tunnels', [])
    if not isinstance(tunnels, list):
        return None

    for tunnel in tunnels:
        public_url = str((tunnel or {}).get('public_url', '')).strip()
        if public_url.startswith('https://'):
            return public_url.rstrip('/')
    return None


def _resolve_ngrok_binary_path():
    """Resolve ngrok from env override, bundled app paths, then PATH."""
    env_path = (os.environ.get('NGROK_PATH') or '').strip().strip('"').strip("'")
    if env_path and os.path.isfile(env_path):
        return env_path, 'env'

    app_dir = os.path.dirname(os.path.abspath(__file__))
    exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else None
    meipass_dir = getattr(sys, '_MEIPASS', None)

    if sys.platform == 'win32':
        bundled_rel = os.path.join('third_party', 'ngrok', 'windows', 'ngrok.exe')
    elif sys.platform == 'darwin':
        bundled_rel = os.path.join('third_party', 'ngrok', 'macos', 'ngrok')
    else:
        bundled_rel = os.path.join('third_party', 'ngrok', 'linux', 'ngrok')

    bundled_candidates = [os.path.join(app_dir, bundled_rel)]
    if meipass_dir:
        bundled_candidates.append(os.path.join(meipass_dir, bundled_rel))
    if exe_dir:
        bundled_candidates.append(os.path.join(exe_dir, bundled_rel))

    for candidate in bundled_candidates:
        if os.path.isfile(candidate):
            return candidate, 'bundled'

    path_binary = shutil.which('ngrok')
    if path_binary:
        return path_binary, 'path'

    return None, None


def _auto_configure_remote_url_from_ngrok(port):
    # Respect explicitly managed remote URL.
    if (os.environ.get('MOBILE_REMOTE_URL') or '').strip():
        print("[NGROK] Skipping remote URL auto-save (MOBILE_REMOTE_URL is set)")
        return

    endpoint = f'http://127.0.0.1:{port}/api/mobile/remote-url/auto-ngrok'
    req = urllib.request.Request(endpoint, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode('utf-8', errors='replace')
            payload = json.loads(body) if body else {}
            remote_url = payload.get('remote_url')
            if remote_url:
                print(f"[NGROK] Remote mobile URL auto-configured: {remote_url}")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8', errors='replace')
            payload = json.loads(body) if body else {}
            message = payload.get('error') or payload.get('suggestion') or str(e)
        except Exception:
            message = str(e)
        print(f"[NGROK] Remote URL auto-config skipped: {message}")
    except Exception as e:
        print(f"[NGROK] Remote URL auto-config skipped: {e}")


def start_ngrok_tunnel(port):
    """Start ngrok tunnel in background (best-effort, non-fatal)."""
    global ngrok_process, ngrok_started_by_app

    if not _should_auto_start_ngrok():
        print("[NGROK] Auto-start disabled (AUTO_START_NGROK=false)")
        return False

    if _discover_ngrok_https_url(timeout_seconds=0.35):
        print("[NGROK] Existing tunnel detected; using running ngrok instance")
        _auto_configure_remote_url_from_ngrok(port)
        return True

    ngrok_path, ngrok_source = _resolve_ngrok_binary_path()
    if not ngrok_path:
        print("[NGROK] ngrok binary not found (PATH/bundled); skipping auto-start")
        return False

    token = get_ngrok_authtoken()
    if not token:
        print("[NGROK] Token not configured yet. Use CONNECT PHONE -> SETUP TOKEN once.")
        return False

    cmd = [ngrok_path, 'http', f'http://127.0.0.1:{port}']
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
        ngrok_process = subprocess.Popen(cmd, **popen_kwargs)
        ngrok_started_by_app = True
        print(f"[NGROK] Starting tunnel in background ({ngrok_source})...")
    except Exception as e:
        print(f"[NGROK] Failed to start: {e}")
        ngrok_process = None
        ngrok_started_by_app = False
        return False

    # Wait briefly for ngrok local API and auto-configure remote URL if ready.
    ready_url = None
    for _ in range(20):
        if ngrok_process and ngrok_process.poll() is not None:
            break
        ready_url = _discover_ngrok_https_url(timeout_seconds=0.3)
        if ready_url:
            break
        time.sleep(0.25)

    if ready_url:
        print(f"[NGROK] Tunnel ready: {ready_url}")
        _auto_configure_remote_url_from_ngrok(port)
        return True

    print("[NGROK] Started, but URL not ready yet (it may appear in a few seconds)")
    return True


def stop_ngrok_tunnel():
    """Stop ngrok only if this app started it."""
    global ngrok_process, ngrok_started_by_app

    if not ngrok_started_by_app:
        return

    process = ngrok_process
    ngrok_process = None
    ngrok_started_by_app = False

    if process is None:
        return
    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=3)
        print("[NGROK] Tunnel stopped")
    except Exception:
        try:
            process.kill()
            process.wait(timeout=2)
            print("[NGROK] Tunnel killed on shutdown")
        except Exception as e:
            print(f"[NGROK] Failed to stop tunnel cleanly: {e}")


def start_flask():
    """Start Flask server in a separate thread"""
    # Bind to 0.0.0.0 to allow network access from mobile devices
    app.run(host='0.0.0.0', port=APP_PORT, debug=False, use_reloader=False)


def cleanup_on_exit():
    """Clean up resources when application exits"""
    global child_windows, cleanup_has_run

    if cleanup_has_run:
        return
    cleanup_has_run = True

    # Close all child windows
    for window_id, window in list(child_windows.items()):
        try:
            window.destroy()
        except Exception:
            pass
    child_windows.clear()

    # Stop background ngrok tunnel if it was started by this app.
    stop_ngrok_tunnel()

    # Clean staging directory
    clean_staging_dir()

    # Clean up backend resources
    try:
        from backend.app import cleanup_on_shutdown
        cleanup_on_shutdown()
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")


# Register cleanup for graceful shutdown on exit
atexit.register(cleanup_on_exit)

def signal_handler(signum, frame):
    """Handle termination signals (SIGINT, SIGTERM)"""
    print(f"\n[SIGNAL] Received signal {signum}, initiating graceful shutdown...")
    cleanup_on_exit()
    sys.exit(0)

# Register signal handlers for Ctrl+C and kill commands
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
print("[STARTUP] Signal handlers registered (SIGINT, SIGTERM)")


def main():
    """Main application entry point"""
    global main_window

    # Start Flask in background thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    # Wait for Flask to start
    time.sleep(2)

    # Clean up old staging files on startup
    clean_staging_dir()

    # Platform detection
    system = platform.system()
    port = APP_PORT

    # Optional: start ngrok automatically for remote mobile access.
    start_ngrok_tunnel(port)

    # Create API instance for main window
    api = WebViewAPI()

    # Create main window (source of truth - never navigates away)
    main_window = webview.create_window(
        'Product Matching System',
        f'http://127.0.0.1:{port}',
        width=1200,
        height=800,
        resizable=True,
        min_size=(800, 600),
        text_select=True,
        js_api=api,
    )

    try:
        webview.start()
    finally:
        cleanup_on_exit()


if __name__ == '__main__':
    main()
