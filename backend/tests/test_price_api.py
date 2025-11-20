"""
Test price history API endpoints
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app
from database import init_db, insert_product
import tempfile
import shutil
import json

def test_price_api():
    """Test price history API endpoints"""
    print("Testing price history API...")
    
    # Create a temporary database for testing
    import database
    original_db_path = database.DB_PATH
    temp_dir = tempfile.mkdtemp()
    database.DB_PATH = os.path.join(temp_dir, 'test.db')
    
    try:
        # Initialize database
        init_db()
        print("✓ Database initialized")
        
        # Create test client
        app.config['TESTING'] = True
        client = app.test_client()
        
        # Create a test product
        product_id = insert_product(
            image_path='/test/image.jpg',
            category='test',
            product_name='Test Product',
            sku='TEST-001',
            is_historical=True
        )
        print(f"✓ Created test product with ID: {product_id}")
        
        # Test POST price history
        price_data = {
            'prices': [
                {'date': '2024-01-15', 'price': 29.99},
                {'date': '2024-02-15', 'price': 31.50},
                {'date': '2024-03-15', 'price': 28.75}
            ]
        }
        
        response = client.post(
            f'/api/products/{product_id}/price-history',
            data=json.dumps(price_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = json.loads(response.data)
        assert data['status'] == 'success', f"Expected success, got {data}"
        assert data['records_inserted'] == 3, f"Expected 3 records, got {data['records_inserted']}"
        print(f"✓ POST price history: {data['records_inserted']} records inserted")
        
        # Test GET price history
        response = client.get(f'/api/products/{product_id}/price-history')
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = json.loads(response.data)
        assert data['status'] == 'success', f"Expected success, got {data}"
        assert len(data['price_history']) == 3, f"Expected 3 records, got {len(data['price_history'])}"
        assert data['statistics'] is not None, "Expected statistics"
        print(f"✓ GET price history: {len(data['price_history'])} records retrieved")
        print(f"  Statistics: min=${data['statistics']['min']}, max=${data['statistics']['max']}, avg=${data['statistics']['average']}")
        
        # Test GET for non-existent product
        response = client.get('/api/products/99999/price-history')
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        print("✓ GET non-existent product returns 404")
        
        # Test POST with invalid data
        invalid_data = {
            'prices': [
                {'date': 'invalid-date', 'price': 29.99}
            ]
        }
        
        response = client.post(
            f'/api/products/{product_id}/price-history',
            data=json.dumps(invalid_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("✓ POST invalid data returns 400")
        
        # Test POST with negative price
        negative_data = {
            'prices': [
                {'date': '2024-01-15', 'price': -10.00}
            ]
        }
        
        response = client.post(
            f'/api/products/{product_id}/price-history',
            data=json.dumps(negative_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("✓ POST negative price returns 400")
        
        print("\n✅ All API tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original database path and cleanup
        database.DB_PATH = original_db_path
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    success = test_price_api()
    sys.exit(0 if success else 1)
