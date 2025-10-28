from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from database import init_db

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
init_db()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Backend is running'})

@app.route('/api/products/upload', methods=['POST'])
def upload_product():
    # Placeholder for product upload endpoint
    return jsonify({'status': 'success', 'message': 'Upload endpoint ready'})

@app.route('/api/products/match', methods=['POST'])
def match_products():
    # Placeholder for matching endpoint
    return jsonify({'status': 'success', 'message': 'Match endpoint ready'})

@app.route('/api/products/historical', methods=['GET', 'POST'])
def historical_products():
    # Placeholder for historical products endpoint
    return jsonify({'status': 'success', 'message': 'Historical endpoint ready'})

@app.route('/api/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    # Placeholder for get product endpoint
    return jsonify({'status': 'success', 'message': f'Get product {product_id} endpoint ready'})

@app.route('/api/batch/match', methods=['POST'])
def batch_match():
    # Placeholder for batch match endpoint
    return jsonify({'status': 'success', 'message': 'Batch match endpoint ready'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
