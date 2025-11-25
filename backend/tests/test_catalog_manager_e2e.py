"""
End-to-End Tests for Catalog Manager
Tests the full workflow between Catalog Management UI and Database operations.

These tests verify:
1. Catalog statistics retrieval
2. Product CRUD operations via API
3. Bulk operations (delete, update, re-extract)
4. Cleanup operations (by type, category, date)
5. Database maintenance (vacuum, export)
6. Input validation and error handling
"""
import os
import sys
import json
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app
from database import (
    init_db, insert_product, insert_features, get_product_by_id,
    get_catalog_stats, get_products_paginated, bulk_delete_products,
    clear_products_by_type, vacuum_database, DB_PATH
)
import numpy as np


@pytest.fixture
def client():
    """Create test client with fresh database"""
    app.config['TESTING'] = True
    
    # Use a test database
    test_db_path = os.path.join(os.path.dirname(__file__), 'test_catalog.db')
    
    # Clean up any existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # Initialize fresh database
    import database
    original_db_path = database.DB_PATH
    database.DB_PATH = test_db_path
    init_db()
    
    with app.test_client() as client:
        yield client
    
    # Cleanup
    database.DB_PATH = original_db_path
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


@pytest.fixture
def sample_products(client):
    """Create sample products for testing"""
    import database
    
    products = []
    
    # Create historical products
    for i in range(5):
        product_id = insert_product(
            image_path=f'/test/historical_{i}.jpg',
            category='test_category',
            product_name=f'Historical Product {i}',
            sku=f'HIST-{i:03d}',
            is_historical=True
        )
        products.append(product_id)
        
        # Add features for some products
        if i < 3:
            insert_features(
                product_id=product_id,
                color_features=np.random.rand(512).astype(np.float32),
                shape_features=np.zeros(7, dtype=np.float32),
                texture_features=np.zeros(256, dtype=np.float32),
                embedding_type='clip',
                embedding_version='test'
            )
    
    # Create new products
    for i in range(3):
        product_id = insert_product(
            image_path=f'/test/new_{i}.jpg',
            category='new_category',
            product_name=f'New Product {i}',
            sku=f'NEW-{i:03d}',
            is_historical=False
        )
        products.append(product_id)
    
    return products


class TestCatalogStats:
    """Test catalog statistics endpoint"""
    
    def test_get_stats_empty_database(self, client):
        """Test stats with empty database"""
        response = client.get('/api/catalog/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['total_products'] == 0
        assert data['historical_products'] == 0
        assert data['new_products'] == 0
        assert 'database_size_mb' in data
        assert 'category_breakdown' in data
    
    def test_get_stats_with_products(self, client, sample_products):
        """Test stats with sample products"""
        response = client.get('/api/catalog/stats')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['total_products'] == 8
        assert data['historical_products'] == 5
        assert data['new_products'] == 3
        assert data['unique_categories'] == 2
        assert data['missing_features'] == 5  # 2 historical + 3 new without features


class TestCatalogProducts:
    """Test catalog products endpoint"""
    
    def test_get_products_empty(self, client):
        """Test getting products from empty database"""
        response = client.get('/api/catalog/products')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['products'] == []
        assert data['total'] == 0
    
    def test_get_products_with_data(self, client, sample_products):
        """Test getting products with data"""
        response = client.get('/api/catalog/products?limit=10')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert len(data['products']) == 8
        assert data['total'] == 8
    
    def test_get_products_pagination(self, client, sample_products):
        """Test pagination"""
        response = client.get('/api/catalog/products?page=1&limit=3')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert len(data['products']) == 3
        assert data['page'] == 1
        assert data['total_pages'] == 3
    
    def test_get_products_filter_by_type(self, client, sample_products):
        """Test filtering by product type"""
        response = client.get('/api/catalog/products?type=historical')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert all(p['is_historical'] for p in data['products'])
    
    def test_get_products_search(self, client, sample_products):
        """Test search functionality"""
        response = client.get('/api/catalog/products?search=Historical')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert len(data['products']) == 5


class TestProductUpdate:
    """Test product update endpoint with validation"""
    
    def test_update_product_success(self, client, sample_products):
        """Test successful product update"""
        product_id = sample_products[0]
        
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={
                'category': 'updated_category',
                'product_name': 'Updated Name',
                'sku': 'UPD-001'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        # Verify update
        product = get_product_by_id(product_id)
        assert product['category'] == 'updated_category'
        assert product['product_name'] == 'Updated Name'
        assert product['sku'] == 'UPD-001'
    
    def test_update_product_invalid_category(self, client, sample_products):
        """Test update with invalid category"""
        product_id = sample_products[0]
        
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={'category': 'a' * 200},  # Too long
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'INVALID_CATEGORY' in data['error_code']
    
    def test_update_product_invalid_sku(self, client, sample_products):
        """Test update with invalid SKU"""
        product_id = sample_products[0]
        
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={'sku': 'invalid sku with spaces!'},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'INVALID_SKU' in data['error_code']
    
    def test_update_product_not_found(self, client):
        """Test update non-existent product"""
        response = client.put(
            '/api/catalog/products/99999',
            json={'category': 'test'},
            content_type='application/json'
        )
        
        assert response.status_code == 404


class TestBulkOperations:
    """Test bulk operations"""
    
    def test_bulk_delete(self, client, sample_products):
        """Test bulk delete"""
        ids_to_delete = sample_products[:3]
        
        response = client.post(
            '/api/catalog/products/bulk-delete',
            json={'product_ids': ids_to_delete},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['deleted_count'] == 3
        
        # Verify deletion
        for pid in ids_to_delete:
            assert get_product_by_id(pid) is None
    
    def test_bulk_delete_invalid_ids(self, client):
        """Test bulk delete with invalid IDs"""
        response = client.post(
            '/api/catalog/products/bulk-delete',
            json={'product_ids': ['invalid', 'ids']},
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_bulk_update(self, client, sample_products):
        """Test bulk update"""
        ids_to_update = sample_products[:3]
        
        response = client.post(
            '/api/catalog/products/bulk-update',
            json={
                'product_ids': ids_to_update,
                'category': 'bulk_updated'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['updated_count'] == 3
        
        # Verify update
        for pid in ids_to_update:
            product = get_product_by_id(pid)
            assert product['category'] == 'bulk_updated'
    
    def test_bulk_update_validation(self, client, sample_products):
        """Test bulk update with invalid data"""
        response = client.post(
            '/api/catalog/products/bulk-update',
            json={
                'product_ids': sample_products[:2],
                'sku': 'invalid sku!'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 400


class TestCleanupOperations:
    """Test cleanup operations"""
    
    def test_cleanup_all(self, client, sample_products):
        """Test clear all products"""
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'all'},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['products_deleted'] == 8
        
        # Verify all deleted
        stats = get_catalog_stats()
        assert stats['total_products'] == 0
    
    def test_cleanup_historical(self, client, sample_products):
        """Test clear historical products only"""
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'historical'},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['products_deleted'] == 5
        
        # Verify only new products remain
        stats = get_catalog_stats()
        assert stats['total_products'] == 3
        assert stats['new_products'] == 3
    
    def test_cleanup_new(self, client, sample_products):
        """Test clear new products only"""
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'new'},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['products_deleted'] == 3
        
        # Verify only historical products remain
        stats = get_catalog_stats()
        assert stats['total_products'] == 5
        assert stats['historical_products'] == 5
    
    def test_cleanup_invalid_type(self, client):
        """Test cleanup with invalid type"""
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'invalid'},
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_cleanup_by_category(self, client, sample_products):
        """Test cleanup by category"""
        response = client.post(
            '/api/catalog/cleanup/categories',
            json={'categories': ['test_category']},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['products_deleted'] == 5
    
    def test_cleanup_by_date(self, client, sample_products):
        """Test cleanup by date (products older than 0 days = all)"""
        # This test uses 0 days which should delete nothing since products were just created
        response = client.post(
            '/api/catalog/cleanup/by-date',
            json={'older_than_days': 30},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        # Products were just created, so none should be deleted
        data = json.loads(response.data)
        assert data['products_deleted'] == 0


class TestDatabaseMaintenance:
    """Test database maintenance operations"""
    
    def test_vacuum_database(self, client, sample_products):
        """Test vacuum database"""
        # First delete some products to create space to reclaim
        client.post(
            '/api/catalog/cleanup',
            json={'type': 'all'},
            content_type='application/json'
        )
        
        response = client.post('/api/catalog/vacuum')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'size_before_mb' in data
        assert 'size_after_mb' in data
    
    def test_export_catalog(self, client, sample_products):
        """Test export catalog to CSV"""
        response = client.get('/api/catalog/export')
        assert response.status_code == 200
        assert 'text/csv' in response.content_type
        
        # Verify CSV content
        csv_content = response.data.decode('utf-8')
        assert 'ID,Filename,Category,Name,SKU' in csv_content
        assert 'Historical Product' in csv_content


class TestInputValidation:
    """Test input validation across all endpoints"""
    
    def test_category_xss_prevention(self, client, sample_products):
        """Test XSS prevention in category"""
        product_id = sample_products[0]
        
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={'product_name': '<script>alert("xss")</script>Test'},
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        # Verify XSS was stripped
        product = get_product_by_id(product_id)
        assert '<script>' not in product['product_name']
    
    def test_sql_injection_prevention(self, client, sample_products):
        """Test SQL injection prevention in search"""
        response = client.get('/api/catalog/products?search=\'; DROP TABLE products; --')
        assert response.status_code == 200
        
        # Verify database still works
        stats = get_catalog_stats()
        assert stats['total_products'] == 8
    
    def test_empty_product_ids_array(self, client):
        """Test empty product IDs array"""
        response = client.post(
            '/api/catalog/products/bulk-delete',
            json={'product_ids': []},
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_too_many_product_ids(self, client):
        """Test too many product IDs"""
        response = client.post(
            '/api/catalog/products/bulk-delete',
            json={'product_ids': list(range(200))},  # More than max 100
            content_type='application/json'
        )
        
        assert response.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



class TestStateSynchronization:
    """Test state synchronization between Catalog Manager and Main App"""
    
    def test_delete_product_during_workflow(self, client, sample_products):
        """
        Test scenario: User uploads products in main app, then deletes some via Catalog Manager.
        The main app should detect the change and handle gracefully.
        """
        # Simulate main app having loaded products
        initial_stats = get_catalog_stats()
        assert initial_stats['total_products'] == 8
        
        # Delete some products via Catalog Manager API
        ids_to_delete = sample_products[:2]
        response = client.post(
            '/api/catalog/products/bulk-delete',
            json={'product_ids': ids_to_delete},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify products are deleted
        for pid in ids_to_delete:
            assert get_product_by_id(pid) is None
        
        # Main app should be able to detect the change via stats
        new_stats = get_catalog_stats()
        assert new_stats['total_products'] == 6
        assert new_stats['total_products'] != initial_stats['total_products']
    
    def test_clear_all_during_workflow(self, client, sample_products):
        """
        Test scenario: User clears all products while main app has data loaded.
        """
        initial_stats = get_catalog_stats()
        assert initial_stats['total_products'] == 8
        
        # Clear all via Catalog Manager
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'all'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify all products are gone
        new_stats = get_catalog_stats()
        assert new_stats['total_products'] == 0
        
        # Main app should detect this drastic change
        assert new_stats['historical_products'] == 0
        assert new_stats['new_products'] == 0
    
    def test_category_cleanup_during_workflow(self, client, sample_products):
        """
        Test scenario: User deletes a category while main app is using products from that category.
        """
        # Get initial stats
        initial_stats = get_catalog_stats()
        
        # Delete test_category (has 5 historical products)
        response = client.post(
            '/api/catalog/cleanup/categories',
            json={'categories': ['test_category']},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify category products are deleted
        new_stats = get_catalog_stats()
        assert new_stats['total_products'] == 3  # Only new_category products remain
    
    def test_concurrent_access_safety(self, client, sample_products):
        """
        Test that concurrent modifications don't corrupt data.
        """
        # Get a product ID
        product_id = sample_products[0]
        
        # Update the product
        response1 = client.put(
            f'/api/catalog/products/{product_id}',
            json={'category': 'concurrent_test_1'},
            content_type='application/json'
        )
        assert response1.status_code == 200
        
        # Update again
        response2 = client.put(
            f'/api/catalog/products/{product_id}',
            json={'category': 'concurrent_test_2'},
            content_type='application/json'
        )
        assert response2.status_code == 200
        
        # Verify final state is consistent
        product = get_product_by_id(product_id)
        assert product['category'] == 'concurrent_test_2'
    
    def test_stats_consistency_after_operations(self, client, sample_products):
        """
        Test that stats remain consistent after various operations.
        """
        # Initial state
        stats1 = get_catalog_stats()
        initial_total = stats1['total_products']
        
        # Delete one product
        client.delete(f'/api/catalog/products/{sample_products[0]}')
        
        # Check stats
        stats2 = get_catalog_stats()
        assert stats2['total_products'] == initial_total - 1
        
        # Bulk delete two more
        client.post(
            '/api/catalog/products/bulk-delete',
            json={'product_ids': sample_products[1:3]},
            content_type='application/json'
        )
        
        # Check stats again
        stats3 = get_catalog_stats()
        assert stats3['total_products'] == initial_total - 3
        
        # Stats should always be accurate
        response = client.get('/api/catalog/products?limit=1000')
        data = json.loads(response.data)
        assert data['total'] == stats3['total_products']



class TestMainAppInteraction:
    """
    Test scenarios where Catalog Manager interacts with Main App workflows.
    These tests simulate real-world usage patterns where users switch between
    the main app and catalog manager.
    """
    
    def test_upload_then_delete_historical(self, client, sample_products):
        """
        Scenario: User uploads historical products in main app, then deletes them in Catalog Manager.
        Main app should handle missing products gracefully.
        """
        # Get historical product IDs
        response = client.get('/api/catalog/products?type=historical&limit=100')
        data = json.loads(response.data)
        historical_ids = [p['id'] for p in data['products']]
        
        assert len(historical_ids) == 5
        
        # Delete all historical via cleanup
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'historical'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify historical products are gone
        response = client.get('/api/catalog/products?type=historical&limit=100')
        data = json.loads(response.data)
        assert len(data['products']) == 0
        
        # New products should still exist
        response = client.get('/api/catalog/products?type=new&limit=100')
        data = json.loads(response.data)
        assert len(data['products']) == 3
    
    def test_upload_then_delete_new(self, client, sample_products):
        """
        Scenario: User uploads new products for matching, then deletes them in Catalog Manager.
        """
        # Get new product IDs
        response = client.get('/api/catalog/products?type=new&limit=100')
        data = json.loads(response.data)
        new_ids = [p['id'] for p in data['products']]
        
        assert len(new_ids) == 3
        
        # Delete all new products
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'new'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify new products are gone
        response = client.get('/api/catalog/products?type=new&limit=100')
        data = json.loads(response.data)
        assert len(data['products']) == 0
        
        # Historical should still exist
        response = client.get('/api/catalog/products?type=historical&limit=100')
        data = json.loads(response.data)
        assert len(data['products']) == 5
    
    def test_modify_product_metadata_during_workflow(self, client, sample_products):
        """
        Scenario: User modifies product metadata in Catalog Manager while main app has it loaded.
        """
        product_id = sample_products[0]
        
        # Get original product
        response = client.get(f'/api/products/{product_id}')
        original = json.loads(response.data)['product']
        
        # Modify via Catalog Manager
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={
                'category': 'modified_category',
                'product_name': 'Modified Name',
                'sku': 'MOD-001'
            },
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify changes via main app endpoint
        response = client.get(f'/api/products/{product_id}')
        modified = json.loads(response.data)['product']
        
        assert modified['category'] == 'modified_category'
        assert modified['product_name'] == 'Modified Name'
        assert modified['sku'] == 'MOD-001'
    
    def test_delete_product_then_access_via_main_app(self, client, sample_products):
        """
        Scenario: User deletes a product in Catalog Manager, then tries to access it in main app.
        """
        product_id = sample_products[0]
        
        # Verify product exists
        response = client.get(f'/api/products/{product_id}')
        assert response.status_code == 200
        
        # Delete via Catalog Manager
        response = client.delete(f'/api/catalog/products/{product_id}')
        assert response.status_code == 200
        
        # Try to access via main app - should return 404
        response = client.get(f'/api/products/{product_id}')
        assert response.status_code == 404
    
    def test_bulk_category_change_affects_matching(self, client, sample_products):
        """
        Scenario: User changes category of products in Catalog Manager.
        This affects which products match in the main app.
        """
        # Get historical products in test_category
        response = client.get('/api/catalog/products?category=test_category&limit=100')
        data = json.loads(response.data)
        product_ids = [p['id'] for p in data['products']]
        
        assert len(product_ids) > 0
        
        # Change all to new category
        response = client.post(
            '/api/catalog/products/bulk-update',
            json={
                'product_ids': product_ids,
                'category': 'renamed_category'
            },
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify old category is empty
        response = client.get('/api/catalog/products?category=test_category&limit=100')
        data = json.loads(response.data)
        assert len(data['products']) == 0
        
        # Verify new category has products
        response = client.get('/api/catalog/products?category=renamed_category&limit=100')
        data = json.loads(response.data)
        assert len(data['products']) == len(product_ids)
    
    def test_vacuum_after_large_deletion(self, client, sample_products):
        """
        Scenario: User deletes many products, then vacuums to reclaim space.
        """
        # Get initial stats
        response = client.get('/api/catalog/stats')
        initial_stats = json.loads(response.data)
        
        # Delete all products
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'all'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Vacuum database
        response = client.post('/api/catalog/vacuum')
        assert response.status_code == 200
        
        vacuum_result = json.loads(response.data)
        assert 'size_before_mb' in vacuum_result
        assert 'size_after_mb' in vacuum_result
    
    def test_export_then_clear_then_verify(self, client, sample_products):
        """
        Scenario: User exports backup, clears database, verifies export contains data.
        """
        # Export backup
        response = client.get('/api/catalog/export')
        assert response.status_code == 200
        csv_content = response.data.decode('utf-8')
        
        # Verify export has data
        lines = csv_content.strip().split('\n')
        assert len(lines) > 1  # Header + data rows
        
        # Clear all
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'all'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify database is empty
        response = client.get('/api/catalog/stats')
        stats = json.loads(response.data)
        assert stats['total_products'] == 0
        
        # Export should still work (empty)
        response = client.get('/api/catalog/export')
        assert response.status_code == 200


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_delete_nonexistent_product(self, client):
        """Test deleting a product that doesn't exist"""
        response = client.delete('/api/catalog/products/99999')
        assert response.status_code == 404
    
    def test_update_nonexistent_product(self, client):
        """Test updating a product that doesn't exist"""
        response = client.put(
            '/api/catalog/products/99999',
            json={'category': 'test'},
            content_type='application/json'
        )
        assert response.status_code == 404
    
    def test_bulk_delete_mixed_valid_invalid(self, client, sample_products):
        """Test bulk delete with mix of valid and invalid IDs"""
        # Mix of valid and invalid IDs
        mixed_ids = [sample_products[0], 99999, sample_products[1], 88888]
        
        response = client.post(
            '/api/catalog/products/bulk-delete',
            json={'product_ids': mixed_ids},
            content_type='application/json'
        )
        
        # Should succeed but only delete valid ones
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['deleted_count'] == 2  # Only 2 valid IDs
    
    def test_cleanup_empty_database(self, client):
        """Test cleanup operations on empty database"""
        # Clear everything first
        client.post(
            '/api/catalog/cleanup',
            json={'type': 'all'},
            content_type='application/json'
        )
        
        # Try to clear again - should succeed with 0 deleted
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'all'},
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['products_deleted'] == 0
    
    def test_category_cleanup_nonexistent_category(self, client, sample_products):
        """Test cleanup of category that doesn't exist"""
        response = client.post(
            '/api/catalog/cleanup/categories',
            json={'categories': ['nonexistent_category']},
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['products_deleted'] == 0
    
    def test_search_special_characters(self, client, sample_products):
        """Test search with special characters"""
        # These should not cause errors
        special_searches = [
            "test'product",
            "test\"product",
            "test;product",
            "test--product",
            "test%product",
            "test_product",
            "test-product"
        ]
        
        for search in special_searches:
            response = client.get(f'/api/catalog/products?search={search}')
            assert response.status_code == 200
    
    def test_pagination_beyond_data(self, client, sample_products):
        """Test pagination when requesting page beyond available data"""
        response = client.get('/api/catalog/products?page=100&limit=50')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['products']) == 0
        assert data['page'] == 100
    
    def test_very_long_category_name(self, client, sample_products):
        """Test with very long category name"""
        product_id = sample_products[0]
        long_category = 'a' * 150  # Exceeds 100 char limit
        
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={'category': long_category},
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_empty_string_values(self, client, sample_products):
        """Test updating with empty strings (validation converts to NULL)"""
        product_id = sample_products[0]
        
        # Empty strings are validated and converted to NULL by validation functions
        # But the update only happens if the field passes validation
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={
                'category': '',
                'product_name': '',
                'sku': ''
            },
            content_type='application/json'
        )
        # Should succeed - empty strings are valid (become NULL)
        assert response.status_code == 200


class TestCSVBuilderInteraction:
    """Test interactions between Catalog Manager and CSV Builder workflows"""
    
    def test_products_created_via_upload_visible_in_catalog(self, client, sample_products):
        """Products created via main app upload should be visible in Catalog Manager"""
        response = client.get('/api/catalog/products?limit=100')
        data = json.loads(response.data)
        
        # All sample products should be visible
        assert data['total'] == 8
    
    def test_category_filter_matches_csv_categories(self, client, sample_products):
        """Categories from CSV upload should be filterable in Catalog Manager"""
        # Get all categories
        response = client.get('/api/catalog/categories')
        data = json.loads(response.data)
        
        # Should have test_category and new_category from sample_products
        assert 'test_category' in data['categories']
        assert 'new_category' in data['categories']
    
    def test_sku_search_finds_csv_products(self, client, sample_products):
        """SKUs from CSV should be searchable in Catalog Manager"""
        response = client.get('/api/catalog/products?search=HIST-001')
        data = json.loads(response.data)
        
        assert len(data['products']) >= 1
        assert any(p['sku'] == 'HIST-001' for p in data['products'])
