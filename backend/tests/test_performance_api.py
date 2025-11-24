"""
Test performance history API endpoints
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app
from database import init_db, insert_product
import tempfile
import shutil
import json

def test_performance_api():
    """Test performance history API endpoints"""
    print("Testing performance history API...")
    
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
        
        # Test POST performance history
        performance_data = {
            'performance': [
                {'date': '2024-01-15', 'sales': 150, 'views': 1200, 'conversion_rate': 12.5, 'revenue': 1800.0},
                {'date': '2024-02-15', 'sales': 180, 'views': 1500, 'conversion_rate': 12.0, 'revenue': 2160.0},
                {'date': '2024-03-15', 'sales': 200, 'views': 1800, 'conversion_rate': 11.1, 'revenue': 2400.0}
            ]
        }
        
        response = client.post(
            f'/api/products/{product_id}/performance-history',
            data=json.dumps(performance_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = json.loads(response.data)
        assert data['status'] == 'success', f"Expected success, got {data}"
        assert data['records_inserted'] == 3, f"Expected 3 records, got {data['records_inserted']}"
        print(f"✓ POST performance history: {data['records_inserted']} records inserted")
        
        # Test GET performance history
        response = client.get(f'/api/products/{product_id}/performance-history')
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = json.loads(response.data)
        assert data['status'] == 'success', f"Expected success, got {data}"
        assert len(data['performance_history']) == 3, f"Expected 3 records, got {len(data['performance_history'])}"
        assert data['statistics'] is not None, "Expected statistics"
        print(f"✓ GET performance history: {len(data['performance_history'])} records retrieved")
        print(f"  Statistics: sales={data['statistics']['total_sales']}, revenue=${data['statistics']['total_revenue']}")
        
        # Test GET for non-existent product
        response = client.get('/api/products/99999/performance-history')
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        print("✓ GET non-existent product returns 404")
        
        # Test POST with invalid data (invalid date)
        invalid_data = {
            'performance': [
                {'date': 'invalid-date', 'sales': 100, 'views': 1000, 'conversion_rate': 10.0, 'revenue': 1000.0}
            ]
        }
        
        response = client.post(
            f'/api/products/{product_id}/performance-history',
            data=json.dumps(invalid_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("✓ POST invalid date returns 400")
        
        # Test POST with negative sales
        negative_data = {
            'performance': [
                {'date': '2024-01-15', 'sales': -10, 'views': 1000, 'conversion_rate': 10.0, 'revenue': 1000.0}
            ]
        }
        
        response = client.post(
            f'/api/products/{product_id}/performance-history',
            data=json.dumps(negative_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("✓ POST negative sales returns 400")
        
        # Test POST with invalid conversion rate (> 100)
        invalid_conversion = {
            'performance': [
                {'date': '2024-01-15', 'sales': 100, 'views': 1000, 'conversion_rate': 150.0, 'revenue': 1000.0}
            ]
        }
        
        response = client.post(
            f'/api/products/{product_id}/performance-history',
            data=json.dumps(invalid_conversion),
            content_type='application/json'
        )
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("✓ POST invalid conversion rate returns 400")
        
        # Test POST with missing required field (date)
        missing_date = {
            'performance': [
                {'sales': 100, 'views': 1000, 'conversion_rate': 10.0, 'revenue': 1000.0}
            ]
        }
        
        response = client.post(
            f'/api/products/{product_id}/performance-history',
            data=json.dumps(missing_date),
            content_type='application/json'
        )
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("✓ POST missing date returns 400")
        
        # Test POST with partial valid data (should insert valid ones)
        partial_data = {
            'performance': [
                {'date': '2024-04-15', 'sales': 100, 'views': 1000, 'conversion_rate': 10.0, 'revenue': 1000.0},
                {'date': 'invalid', 'sales': 200, 'views': 2000, 'conversion_rate': 10.0, 'revenue': 2000.0},
                {'date': '2024-05-15', 'sales': 150, 'views': 1500, 'conversion_rate': 10.0, 'revenue': 1500.0}
            ]
        }
        
        response = client.post(
            f'/api/products/{product_id}/performance-history',
            data=json.dumps(partial_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = json.loads(response.data)
        assert data['records_inserted'] == 2, f"Expected 2 valid records, got {data['records_inserted']}"
        assert 'validation_errors' in data, "Expected validation errors"
        print(f"✓ POST partial valid data: {data['records_inserted']} inserted, {len(data['validation_errors'])} skipped")
        
        print("\n✅ All API tests passed!")
        
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
    success = test_performance_api()
    sys.exit(0 if success else 1)
