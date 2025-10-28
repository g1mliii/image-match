"""
Product Matching System - Desktop Application
Simple launcher using Flask backend with webview frontend
"""
import webview
import threading
import sys
import os
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app import app

def start_flask():
    """Start Flask server in a separate thread"""
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

def main():
    """Main application entry point"""
    # Start Flask in background thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    # Wait for Flask to start
    time.sleep(2)
    
    # Create and start webview window
    window = webview.create_window(
        'Product Matching System',
        'http://127.0.0.1:5000',
        width=1200,
        height=800,
        resizable=True,
        min_size=(800, 600)
    )
    
    webview.start()

if __name__ == '__main__':
    main()
