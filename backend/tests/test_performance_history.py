"""
Test performance history functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from database import (
    init_db, insert_product, insert_performance_history, bulk_insert_performance_history,
    get_performance_history, get_performance_statistics, link_performance_history,
    get_products_with_performance_history, delete_performance_history
)
import tempfile
import shutil

def test_performance_history():
    """Test performance history CRUD operations"""
    print("Testing performance history functionality...")
    
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
        
        # Test single performance insert
        perf_id = insert_performance_history(product_id, '2024-01-15', 150, 1200, 12.5, 1800.0)
        assert perf_id > 0, "Failed to insert performance history"
        print(f"✓ Inserted single performance record with ID: {perf_id}")
        
        # Test bulk performance insert
        performance_records = [
            {'date': '2024-02-15', 'sales': 180, 'views': 1500, 'conversion_rate': 12.0, 'revenue': 2160.0},
            {'date': '2024-03-15', 'sales': 200, 'views': 1800, 'conversion_rate': 11.1, 'revenue': 2400.0},
            {'date': '2024-04-15', 'sales': 220, 'views': 2000, 'conversion_rate': 11.0, 'revenue': 2640.0},
            {'date': '2024-05-15', 'sales': 190, 'views': 1700, 'conversion_rate': 11.2, 'revenue': 2280.0}
        ]
        inserted_count = bulk_insert_performance_history(product_id, performance_records)
        assert inserted_count == 4, f"Expected 4 records, got {inserted_count}"
        print(f"✓ Bulk inserted {inserted_count} performance records")
        
        # Test get performance history
        history = get_performance_history(product_id, limit=10)
        assert len(history) == 5, f"Expected 5 records, got {len(history)}"
        print(f"✓ Retrieved {len(history)} performance history records")
        
        # Test performance statistics
        stats = get_performance_statistics(product_id)
        assert stats is not None, "Failed to get performance statistics"
        assert stats['total_sales'] == 940, f"Expected total_sales 940, got {stats['total_sales']}"
        assert stats['total_views'] == 8200, f"Expected total_views 8200, got {stats['total_views']}"
        assert stats['total_revenue'] == 11280.0, f"Expected total_revenue 11280.0, got {stats['total_revenue']}"
        assert stats['sales_trend'] in ['up', 'down', 'stable'], f"Invalid sales_trend: {stats['sales_trend']}"
        print(f"✓ Performance statistics: sales={stats['total_sales']}, views={stats['total_views']}, revenue=${stats['total_revenue']}, trend={stats['sales_trend']}")
        
        # Test performance history linking
        product_id_2 = insert_product(
            image_path='/test/image2.jpg',
            category='test',
            product_name='Test Product 2',
            sku='TEST-002',
            is_historical=False
        )
        linked_count = link_performance_history(product_id, product_id_2)
        assert linked_count == 5, f"Expected 5 linked records, got {linked_count}"
        print(f"✓ Linked {linked_count} performance records to new product")
        
        # Verify linked performance
        history_2 = get_performance_history(product_id_2, limit=10)
        assert len(history_2) == 5, f"Expected 5 linked records, got {len(history_2)}"
        print(f"✓ Verified {len(history_2)} linked performance records")
        
        # Test get products with performance history
        products_with_perf = get_products_with_performance_history()
        assert len(products_with_perf) == 2, f"Expected 2 products with performance, got {len(products_with_perf)}"
        print(f"✓ Found {len(products_with_perf)} products with performance history")
        
        # Test delete performance history
        deleted = delete_performance_history(product_id)
        assert deleted, "Failed to delete performance history"
        history_after_delete = get_performance_history(product_id, limit=10)
        assert len(history_after_delete) == 0, f"Expected 0 records after delete, got {len(history_after_delete)}"
        print("✓ Successfully deleted performance history")
        
        # Test edge cases
        print("\n Testing edge cases...")
        
        # Test with zero values
        perf_id = insert_performance_history(product_id, '2024-06-15', 0, 0, 0.0, 0.0)
        assert perf_id > 0, "Failed to insert performance with zero values"
        print("✓ Handled zero values correctly")
        
        # Test with high values
        perf_id = insert_performance_history(product_id, '2024-07-15', 10000, 100000, 10.0, 100000.0)
        assert perf_id > 0, "Failed to insert performance with high values"
        print("✓ Handled high values correctly")
        
        # Test with decimal conversion rates
        perf_id = insert_performance_history(product_id, '2024-08-15', 100, 1000, 10.5, 1050.0)
        assert perf_id > 0, "Failed to insert performance with decimal conversion"
        print("✓ Handled decimal conversion rates correctly")
        
        # Test bulk insert with missing fields (should use defaults)
        partial_records = [
            {'date': '2024-09-15', 'sales': 100},  # Missing views, conversion, revenue
            {'date': '2024-10-15', 'views': 1000},  # Missing sales, conversion, revenue
        ]
        inserted = bulk_insert_performance_history(product_id, partial_records)
        assert inserted == 2, f"Expected 2 records with defaults, got {inserted}"
        print("✓ Handled missing fields with defaults")
        
        # Test empty bulk insert
        inserted = bulk_insert_performance_history(product_id, [])
        assert inserted == 0, "Expected 0 for empty bulk insert"
        print("✓ Handled empty bulk insert")
        
        # Test invalid date (should be skipped)
        invalid_records = [
            {'date': None, 'sales': 100, 'views': 1000, 'conversion_rate': 10.0, 'revenue': 1000.0}
        ]
        inserted = bulk_insert_performance_history(product_id, invalid_records)
        assert inserted == 0, "Expected 0 for invalid date"
        print("✓ Handled invalid date correctly")
        
        print("\n✅ All performance history tests passed!")
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
    success = test_performance_history()
    sys.exit(0 if success else 1)
