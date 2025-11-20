"""
Test price history functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import (
    init_db, insert_product, insert_price_history, bulk_insert_price_history,
    get_price_history, get_price_statistics, link_price_history,
    get_products_with_price_history, delete_price_history
)
import tempfile
import shutil

def test_price_history():
    """Test price history CRUD operations"""
    print("Testing price history functionality...")
    
    # Create a temporary database for testing
    import database
    original_db_path = database.DB_PATH
    temp_dir = tempfile.mkdtemp()
    database.DB_PATH = os.path.join(temp_dir, 'test.db')
    
    try:
        # Initialize database
        init_db()
        print("✓ Database initialized")
        
        # Create a test product
        product_id = insert_product(
            image_path='/test/image.jpg',
            category='test',
            product_name='Test Product',
            sku='TEST-001',
            is_historical=True
        )
        print(f"✓ Created test product with ID: {product_id}")
        
        # Test single price insert
        price_id = insert_price_history(product_id, '2024-01-15', 29.99, 'USD')
        assert price_id > 0, "Failed to insert price history"
        print(f"✓ Inserted single price record with ID: {price_id}")
        
        # Test bulk price insert
        price_records = [
            {'date': '2024-02-15', 'price': 31.50},
            {'date': '2024-03-15', 'price': 28.75},
            {'date': '2024-04-15', 'price': 32.00},
            {'date': '2024-05-15', 'price': 30.25}
        ]
        inserted_count = bulk_insert_price_history(product_id, price_records)
        assert inserted_count == 4, f"Expected 4 records, got {inserted_count}"
        print(f"✓ Bulk inserted {inserted_count} price records")
        
        # Test get price history
        history = get_price_history(product_id, limit=10)
        assert len(history) == 5, f"Expected 5 records, got {len(history)}"
        print(f"✓ Retrieved {len(history)} price history records")
        
        # Test price statistics
        stats = get_price_statistics(product_id)
        assert stats is not None, "Failed to get price statistics"
        assert stats['min'] == 28.75, f"Expected min 28.75, got {stats['min']}"
        assert stats['max'] == 32.00, f"Expected max 32.00, got {stats['max']}"
        assert stats['current'] == 30.25, f"Expected current 30.25, got {stats['current']}"
        assert stats['trend'] in ['up', 'down', 'stable'], f"Invalid trend: {stats['trend']}"
        print(f"✓ Price statistics: min=${stats['min']}, max=${stats['max']}, avg=${stats['average']}, current=${stats['current']}, trend={stats['trend']}")
        
        # Test price history linking
        product_id_2 = insert_product(
            image_path='/test/image2.jpg',
            category='test',
            product_name='Test Product 2',
            sku='TEST-002',
            is_historical=False
        )
        linked_count = link_price_history(product_id, product_id_2)
        assert linked_count == 5, f"Expected 5 linked records, got {linked_count}"
        print(f"✓ Linked {linked_count} price records to new product")
        
        # Verify linked prices
        history_2 = get_price_history(product_id_2, limit=10)
        assert len(history_2) == 5, f"Expected 5 linked records, got {len(history_2)}"
        print(f"✓ Verified {len(history_2)} linked price records")
        
        # Test get products with price history
        products_with_prices = get_products_with_price_history()
        assert len(products_with_prices) == 2, f"Expected 2 products with prices, got {len(products_with_prices)}"
        print(f"✓ Found {len(products_with_prices)} products with price history")
        
        # Test delete price history
        deleted = delete_price_history(product_id)
        assert deleted, "Failed to delete price history"
        history_after_delete = get_price_history(product_id, limit=10)
        assert len(history_after_delete) == 0, f"Expected 0 records after delete, got {len(history_after_delete)}"
        print("✓ Successfully deleted price history")
        
        print("\n✅ All price history tests passed!")
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
    success = test_price_history()
    sys.exit(0 if success else 1)
