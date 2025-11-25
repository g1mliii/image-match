"""
End-to-End Test for Catalog Snapshot System

Tests the complete workflow:
1. User uploads products to main database
2. User saves current catalog as snapshot
3. User loads a different snapshot
4. Verify matching still works correctly
"""

import pytest
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from snapshot_manager import (
    save_main_db_as_snapshot, load_snapshot_to_main_db,
    get_main_db_stats, delete_snapshot, list_snapshots
)
from database import get_product_by_id, count_products


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestSnapshotE2E:
    """End-to-end tests for snapshot system"""
    
    def test_full_workflow(self, client):
        """Test complete workflow: upload → save → load → match"""
        
        # Step 1: Get initial main db stats
        response = client.get('/api/catalogs/main-db-stats')
        assert response.status_code == 200
        initial_data = json.loads(response.data)
        initial_count = initial_data.get('total_products', 0)
        
        print(f"\n1. Initial main DB: {initial_count} products")
        
        # Step 2: Save current catalog as snapshot
        response = client.post('/api/catalogs/save-current',
            data=json.dumps({'name': 'E2E Test Snapshot 1'}),
            content_type='application/json'
        )
        assert response.status_code == 200
        save_data = json.loads(response.data)
        snapshot1_file = save_data['snapshot_file']
        
        print(f"2. Saved as snapshot: {snapshot1_file}")
        
        # Step 3: Verify snapshot was created
        response = client.get('/api/catalogs/list')
        assert response.status_code == 200
        list_data = json.loads(response.data)
        snapshot_names = [s['snapshot_file'] for s in list_data.get('historical', [])]
        assert snapshot1_file in snapshot_names
        
        print(f"3. Verified snapshot exists in list")
        
        # Step 4: Get snapshot info
        response = client.get(f'/api/catalogs/{snapshot1_file}/info')
        assert response.status_code == 200
        info_data = json.loads(response.data)
        assert info_data['snapshot']['name'] == 'E2E Test Snapshot 1'
        assert info_data['snapshot']['product_count'] == initial_count
        
        print(f"4. Snapshot info verified: {initial_count} products")
        
        # Step 5: Load snapshot back to main db
        response = client.post(f'/api/catalogs/load/{snapshot1_file}')
        assert response.status_code == 200
        load_data = json.loads(response.data)
        assert load_data.get('success') == True
        
        print(f"5. Loaded snapshot back to main DB")
        
        # Step 6: Verify main db stats show loaded snapshot
        response = client.get('/api/catalogs/main-db-stats')
        assert response.status_code == 200
        stats_data = json.loads(response.data)
        assert stats_data['loaded_snapshot']['loaded'] == True
        assert stats_data['loaded_snapshot']['snapshot_file'] == snapshot1_file
        
        print(f"6. Main DB shows loaded snapshot: {snapshot1_file}")
        
        # Step 7: Verify product count matches
        assert stats_data['total_products'] == initial_count
        
        print(f"7. Product count verified: {initial_count}")
        
        # Cleanup
        delete_snapshot(snapshot1_file)
        print(f"8. Cleaned up test snapshot")
        
        print("\n✓ Full workflow test passed!")
    
    def test_save_load_preserves_data(self):
        """Test that save/load preserves all product data"""
        import time
        
        # Initialize database first
        from database import init_db
        init_db()
        
        # Get current product count
        initial_count = count_products()
        
        if initial_count == 0:
            pytest.skip("No products in database to test")
        
        # Get first product details
        from database import get_all_products
        products = get_all_products()
        if not products:
            pytest.skip("No products to test")
        
        first_product = products[0]
        original_id = first_product['id']
        original_category = first_product['category']
        original_sku = first_product['sku']
        
        print(f"\nOriginal product: ID={original_id}, category={original_category}, sku={original_sku}")
        
        # Save as snapshot with unique name
        unique_name = f'Data Preservation Test {int(time.time())}'
        result = save_main_db_as_snapshot(unique_name)
        assert result.get('success') == True
        snapshot_file = result['snapshot_file']
        
        print(f"Saved as: {snapshot_file}")
        
        # Load it back
        result = load_snapshot_to_main_db(snapshot_file)
        assert result.get('success') == True
        
        print(f"Loaded back from snapshot")
        
        # Verify product data is preserved
        products_after = get_all_products()
        assert len(products_after) == initial_count
        
        # Find the same product
        restored_product = next((p for p in products_after if p['id'] == original_id), None)
        assert restored_product is not None
        assert restored_product['category'] == original_category
        assert restored_product['sku'] == original_sku
        
        print(f"✓ Data preserved: ID={restored_product['id']}, category={restored_product['category']}, sku={restored_product['sku']}")
        
        # Cleanup
        delete_snapshot(snapshot_file)
        print(f"Cleaned up test snapshot")


class TestSnapshotWorkflows:
    """Test comprehensive snapshot workflows"""
    
    def test_workflow_save_switch_restore(self, client):
        """
        Workflow: Save catalog A → Load catalog B → Restore catalog A
        """
        from database import init_db, insert_product, insert_features, clear_products_by_type
        import numpy as np
        
        # Initialize
        init_db()
        clear_products_by_type('all')
        
        # Step 1: Create Catalog A with some products
        print("\n1. Creating Catalog A...")
        catalog_a_ids = []
        for i in range(3):
            product_id = insert_product(
                image_path=f'/test/catalog_a/product_{i}.jpg',
                category='catalog_a_category',
                product_name=f'Catalog A Product {i}',
                sku=f'CAT-A-{i:03d}',
                is_historical=True
            )
            catalog_a_ids.append(product_id)
            insert_features(
                product_id=product_id,
                color_features=np.random.rand(512).astype(np.float32),
                shape_features=np.zeros(7, dtype=np.float32),
                texture_features=np.zeros(256, dtype=np.float32),
                embedding_type='clip',
                embedding_version='test'
            )
        
        # Verify Catalog A
        response = client.get('/api/catalogs/main-db-stats')
        stats = json.loads(response.data)
        assert stats['total_products'] == 3
        print(f"   Catalog A has {stats['total_products']} products")
        
        # Step 2: Save Catalog A as snapshot
        print("2. Saving Catalog A as snapshot...")
        response = client.post('/api/catalogs/save-current',
            data=json.dumps({'name': 'Catalog A Snapshot'}),
            content_type='application/json'
        )
        assert response.status_code == 200
        save_data = json.loads(response.data)
        snapshot_a = save_data['snapshot_file']
        print(f"   Saved as: {snapshot_a}")
        
        # Step 3: Clear and create Catalog B
        print("3. Creating Catalog B...")
        clear_products_by_type('all')
        
        catalog_b_ids = []
        for i in range(5):
            product_id = insert_product(
                image_path=f'/test/catalog_b/product_{i}.jpg',
                category='catalog_b_category',
                product_name=f'Catalog B Product {i}',
                sku=f'CAT-B-{i:03d}',
                is_historical=True
            )
            catalog_b_ids.append(product_id)
            insert_features(
                product_id=product_id,
                color_features=np.random.rand(512).astype(np.float32),
                shape_features=np.zeros(7, dtype=np.float32),
                texture_features=np.zeros(256, dtype=np.float32),
                embedding_type='clip',
                embedding_version='test'
            )
        
        # Verify Catalog B
        response = client.get('/api/catalogs/main-db-stats')
        stats = json.loads(response.data)
        assert stats['total_products'] == 5
        print(f"   Catalog B has {stats['total_products']} products")
        
        # Step 4: Save Catalog B as snapshot
        print("4. Saving Catalog B as snapshot...")
        response = client.post('/api/catalogs/save-current',
            data=json.dumps({'name': 'Catalog B Snapshot'}),
            content_type='application/json'
        )
        assert response.status_code == 200
        save_data = json.loads(response.data)
        snapshot_b = save_data['snapshot_file']
        print(f"   Saved as: {snapshot_b}")
        
        # Step 5: Load Catalog A back
        print("5. Loading Catalog A back...")
        response = client.post(f'/api/catalogs/load/{snapshot_a}')
        assert response.status_code == 200
        
        # Verify Catalog A is restored
        response = client.get('/api/catalogs/main-db-stats')
        stats = json.loads(response.data)
        assert stats['total_products'] == 3
        assert stats['loaded_snapshot']['snapshot_file'] == snapshot_a
        print(f"   Catalog A restored: {stats['total_products']} products")
        
        # Step 6: Verify products are from Catalog A
        response = client.get('/api/catalog/products?limit=100')
        products_data = json.loads(response.data)
        categories = [p['category'] for p in products_data['products']]
        assert all(cat == 'catalog_a_category' for cat in categories)
        print("   ✓ All products are from Catalog A")
        
        # Step 7: Switch to Catalog B
        print("6. Switching to Catalog B...")
        response = client.post(f'/api/catalogs/load/{snapshot_b}')
        assert response.status_code == 200
        
        # Verify Catalog B is loaded
        response = client.get('/api/catalogs/main-db-stats')
        stats = json.loads(response.data)
        assert stats['total_products'] == 5
        assert stats['loaded_snapshot']['snapshot_file'] == snapshot_b
        print(f"   Catalog B loaded: {stats['total_products']} products")
        
        # Verify products are from Catalog B
        response = client.get('/api/catalog/products?limit=100')
        products_data = json.loads(response.data)
        categories = [p['category'] for p in products_data['products']]
        assert all(cat == 'catalog_b_category' for cat in categories)
        print("   ✓ All products are from Catalog B")
        
        # Cleanup
        delete_snapshot(snapshot_a)
        delete_snapshot(snapshot_b)
        print("\n✓ Workflow test passed!")
    
    def test_workflow_multiple_snapshots_switching(self, client):
        """
        Workflow: Create multiple snapshots and switch between them rapidly
        """
        from database import init_db, insert_product, insert_features, clear_products_by_type
        import numpy as np
        
        init_db()
        clear_products_by_type('all')
        
        snapshots = []
        
        # Create 3 different catalogs and save as snapshots
        for catalog_num in range(3):
            print(f"\nCreating Catalog {catalog_num + 1}...")
            clear_products_by_type('all')
            
            # Create products for this catalog
            for i in range(2 + catalog_num):  # 2, 3, 4 products
                product_id = insert_product(
                    image_path=f'/test/catalog_{catalog_num}/product_{i}.jpg',
                    category=f'catalog_{catalog_num}_category',
                    product_name=f'Catalog {catalog_num} Product {i}',
                    sku=f'CAT-{catalog_num}-{i:03d}',
                    is_historical=True
                )
                insert_features(
                    product_id=product_id,
                    color_features=np.random.rand(512).astype(np.float32),
                    shape_features=np.zeros(7, dtype=np.float32),
                    texture_features=np.zeros(256, dtype=np.float32),
                    embedding_type='clip',
                    embedding_version='test'
                )
            
            # Save as snapshot
            response = client.post('/api/catalogs/save-current',
                data=json.dumps({'name': f'Multi Catalog {catalog_num}'}),
                content_type='application/json'
            )
            assert response.status_code == 200
            save_data = json.loads(response.data)
            snapshots.append(save_data['snapshot_file'])
            print(f"   Saved as: {save_data['snapshot_file']}")
        
        # Now switch between them multiple times
        print("\nSwitching between snapshots...")
        for i, snapshot in enumerate([snapshots[0], snapshots[2], snapshots[1], snapshots[0]]):
            response = client.post(f'/api/catalogs/load/{snapshot}')
            assert response.status_code == 200
            
            # Verify correct catalog is loaded
            response = client.get('/api/catalogs/main-db-stats')
            stats = json.loads(response.data)
            assert stats['loaded_snapshot']['snapshot_file'] == snapshot
            print(f"   Switch {i + 1}: Loaded {snapshot} ({stats['total_products']} products)")
        
        # Cleanup
        for snapshot in snapshots:
            delete_snapshot(snapshot)
        
        print("\n✓ Multiple snapshot switching test passed!")
    
    def test_workflow_snapshot_with_matches(self, client):
        """
        Workflow: Create catalog with matches → Save → Clear → Load → Verify matches preserved
        """
        from database import (
            init_db, insert_product, insert_features, insert_match,
            clear_products_by_type, get_matches_for_product
        )
        import numpy as np
        import time
        
        init_db()
        clear_products_by_type('all')
        
        unique_suffix = int(time.time())
        
        print("\n1. Creating catalog with historical and new products...")
        
        # Create historical products
        hist_ids = []
        for i in range(3):
            product_id = insert_product(
                image_path=f'/test/historical_{i}.jpg',
                category='test_category',
                product_name=f'Historical {i}',
                sku=f'HIST-{i:03d}',
                is_historical=True
            )
            hist_ids.append(product_id)
            insert_features(
                product_id=product_id,
                color_features=np.random.rand(512).astype(np.float32),
                shape_features=np.zeros(7, dtype=np.float32),
                texture_features=np.zeros(256, dtype=np.float32),
                embedding_type='clip',
                embedding_version='test'
            )
        
        # Create new products
        new_ids = []
        for i in range(2):
            product_id = insert_product(
                image_path=f'/test/new_{i}.jpg',
                category='test_category',
                product_name=f'New {i}',
                sku=f'NEW-{i:03d}',
                is_historical=False
            )
            new_ids.append(product_id)
            insert_features(
                product_id=product_id,
                color_features=np.random.rand(512).astype(np.float32),
                shape_features=np.zeros(7, dtype=np.float32),
                texture_features=np.zeros(256, dtype=np.float32),
                embedding_type='clip',
                embedding_version='test'
            )
        
        # Create matches
        print("2. Creating matches...")
        match_count = 0
        for new_id in new_ids:
            for hist_id in hist_ids:
                sim_score = 0.75 + (match_count * 0.05)
                insert_match(
                    new_product_id=new_id,
                    matched_product_id=hist_id,
                    similarity_score=sim_score,
                    color_score=sim_score,
                    shape_score=sim_score,
                    texture_score=sim_score
                )
                match_count += 1
        
        print(f"   Created {match_count} matches")
        
        # Verify matches exist
        response = client.get('/api/catalog/stats')
        stats = json.loads(response.data)
        assert stats['total_matches'] == match_count
        
        # Save as snapshot
        print("3. Saving as snapshot...")
        snapshot_name = f'Catalog With Matches {unique_suffix}'
        response = client.post('/api/catalogs/save-current',
            data=json.dumps({'name': snapshot_name}),
            content_type='application/json'
        )
        assert response.status_code == 200
        save_data = json.loads(response.data)
        snapshot_file = save_data['snapshot_file']
        print(f"   Saved as: {snapshot_file}")
        
        # Clear everything
        print("4. Clearing main database...")
        clear_products_by_type('all')
        
        response = client.get('/api/catalogs/main-db-stats')
        stats = json.loads(response.data)
        assert stats['total_products'] == 0
        
        response = client.get('/api/catalog/stats')
        catalog_stats = json.loads(response.data)
        assert catalog_stats['total_matches'] == 0
        print("   Main database cleared")
        
        # Load snapshot back
        print("5. Loading snapshot back...")
        response = client.post(f'/api/catalogs/load/{snapshot_file}')
        assert response.status_code == 200
        
        # Verify products restored
        response = client.get('/api/catalogs/main-db-stats')
        stats = json.loads(response.data)
        assert stats['total_products'] == 5
        assert stats['historical_products'] == 3
        assert stats['new_products'] == 2
        print(f"   Products restored: {stats['total_products']}")
        
        # Verify matches restored
        response = client.get('/api/catalog/stats')
        catalog_stats = json.loads(response.data)
        assert catalog_stats['total_matches'] == match_count
        print(f"   Matches restored: {catalog_stats['total_matches']}")
        
        # Verify match details are correct
        from database import get_all_products
        products = get_all_products()
        new_products = [p for p in products if not p['is_historical']]
        
        for new_product in new_products:
            matches = get_matches_for_product(new_product['id'])
            assert len(matches) == 3  # Each new product matched with 3 historical
            print(f"   Product {new_product['sku']} has {len(matches)} matches")
        
        # Cleanup
        delete_snapshot(snapshot_file)
        print("\n✓ Snapshot with matches test passed!")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
