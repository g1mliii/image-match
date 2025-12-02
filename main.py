"""
Product Matching System - Desktop Application
Simple launcher using Flask backend with webview frontend
"""
import webview
import threading
import sys
import os
import time
import platform

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app import app

def start_flask():
    """Start Flask server in a separate thread"""
    # macOS uses port 5001 to avoid AirPlay Receiver conflict on port 5000
    port = 5001 if platform.system() == 'Darwin' else 5000
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)

def main():
    """Main application entry point"""
    # Start Flask in background thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    # Wait for Flask to start
    time.sleep(2)
    
    # Detect platform and set appropriate icon
    # Windows: .ico file
    # macOS: .icns file (will be created during packaging)
    # Linux: .png file
    icon_path = None
    system = platform.system()
    
    if system == 'Windows':
        if os.path.exists('app_icon.ico'):
            icon_path = 'app_icon.ico'
    elif system == 'Darwin':  # macOS
        if os.path.exists('app_icon.icns'):
            icon_path = 'app_icon.icns'
        elif os.path.exists('app_icon.png'):
            icon_path = 'app_icon.png'
    else:  # Linux
        if os.path.exists('app_icon.png'):
            icon_path = 'app_icon.png'
    
    # Create webview window with native OS styling
    # Pywebview automatically uses native window decorations:
    # - Windows: Standard Windows title bar with minimize/maximize/close
    # - macOS: Standard macOS title bar with traffic lights (red/yellow/green)
    # - Linux: Standard Linux window manager decorations
    
    # Use platform-specific port (macOS uses 5001 to avoid AirPlay conflict)
    port = 5001 if system == 'Darwin' else 5000
    
    window = webview.create_window(
        'Product Matching System',
        f'http://127.0.0.1:{port}',
        width=1200,
        height=800,
        resizable=True,
        min_size=(800, 600),
        text_select=True,
        # Custom icon (automatically uses correct format per OS)
        # icon=icon_path,  # Uncomment when you add platform-specific icons
        
        # Optional: Frameless mode for custom title bar (cross-platform)
        # frameless=True,
        # easy_drag=True,  # Allows dragging frameless window
    )
    
    webview.start()

if __name__ == '__main__':
    main()
