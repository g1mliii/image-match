"""
Full End-to-End Workflow Tests
Tests complete user journeys across Main App, Catalog Manager, and CSV Builder.

These tests verify:
1. CSV Builder state synchronization with catalog changes
2. Full user workflows from upload to matching to cleanup
3. Cross-tool interactions and state consistency
4. Edge cases when switching between tools
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
    clear_products_by_type, vacuum_database, DB_PATH, insert_match
)
import numpy as np


@pytest.fixture
def client():
    """Create test client with fresh database"""
    app.config['TESTING'] = True
    
    test_db_path = os.path.join(os.path.dirname(__file__), 'test_workflow.db')
    
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    import database
    original_db_path = database.DB_PATH
    database.DB_PATH = test_db_path
    init_db()
    
    with app.test_client() as client:
        yield client
    
    database.DB_PATH = original_db_path
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


@pytest.fixture
def full_catalog(client):
    """Create a full catalog with historical and new products, features, and matches"""
    import database
    
    products = {'historical': [], 'new': []}
    
    # Create historical products with various categories
    categories = ['plates', 'bowls', 'cups', 'placemats', 'napkins']
    for cat_idx, category in enumerate(categories):
        for i in range(3):
            product_id = insert_product(
                image_path=f'/test/{category}/product_{i}.jpg',
                category=category,
                product_name=f'{category.title()} Item {i}',
                sku=f'{category.upper()[:3]}-{i:03d}',
                is_historical=True
            )
            products['historical'].append(product_id)
            
            # Add features
            insert_features(
                product_id=product_id,
                color_features=np.random.rand(512).astype(np.float32),
                shape_features=np.zeros(7, dtype=np.float32),
                texture_features=np.zeros(256, dtype=np.float32),
                embedding_type='clip',
                embedding_version='test'
            )
    
    # Create new products
    for i in range(5):
        category = categories[i % len(categories)]
        product_id = insert_product(
            image_path=f'/test/new/new_product_{i}.jpg',
            category=category,
            product_name=f'New {category.title()} {i}',
            sku=f'NEW-{i:03d}',
            is_historical=False
        )
        products['new'].append(product_id)
        
        # Add features
        insert_features(
            product_id=product_id,
            color_features=np.random.rand(512).astype(np.float32),
            shape_features=np.zeros(7, dtype=np.float32),
            texture_features=np.zeros(256, dtype=np.float32),
            embedding_type='clip',
            embedding_version='test'
        )
    
    # Create some matches between new and historical products
    # Note: With CLIP, color/shape/texture scores are set to same value as similarity_score
    # for database compatibility (schema requires NOT NULL)
    for new_id in products['new'][:3]:
        for hist_id in products['historical'][:2]:
            sim_score = np.random.uniform(0.5, 0.95)
            insert_match(
                new_product_id=new_id,
                matched_product_id=hist_id,
                similarity_score=sim_score,
                color_score=sim_score,  # CLIP sets all to same value
                shape_score=sim_score,  # CLIP sets all to same value
                texture_score=sim_score  # CLIP sets all to same value
            )
    
    return products


class TestCSVBuilderStateSynchronization:
    """Test CSV Builder state synchronization with catalog changes"""
    
    def test_csv_builder_data_transfer_from_main_app(self, client, full_catalog):
        """
        Scenario: User uploads folder in main app, opens CSV Builder.
        CSV Builder should receive the file list.
        """
        # Simulate main app storing file data for CSV Builder
        # This is done via sessionStorage in browser, we test the API side
        
        # Get products that would be sent to CSV Builder
        response = client.get('/api/catalog/products?type=historical&limit=100')
        data = json.loads(response.data)
        
        # Verify we have products to work with
        assert len(data['products']) == 15  # 5 categories * 3 products
        
        # Each product should have filename info
        for product in data['products']:
            assert 'image_path' in product or 'filename' in product
    
    def test_catalog_cleanup_affects_csv_builder_data(self, client, full_catalog):
        """
        Scenario: User has CSV Builder open, then clears catalog via Catalog Manager.
        CSV Builder should detect the change.
        """
        # Get initial stats
        response = client.get('/api/catalog/stats')
        initial_stats = json.loads(response.data)
        assert initial_stats['total_products'] == 20
        
        # Clear all via Catalog Manager
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'all'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # CSV Builder would detect this via stats check
        response = client.get('/api/catalog/stats')
        new_stats = json.loads(response.data)
        assert new_stats['total_products'] == 0
        
        # The change is detectable
        assert new_stats['total_products'] != initial_stats['total_products']
    
    def test_category_deletion_affects_csv_builder(self, client, full_catalog):
        """
        Scenario: User is building CSV for 'plates' category, then deletes it.
        """
        # Get plates products
        response = client.get('/api/catalog/products?category=plates&limit=100')
        data = json.loads(response.data)
        plates_count = len(data['products'])
        assert plates_count > 0  # Should have some plates
        
        # Delete plates category
        response = client.post(
            '/api/catalog/cleanup/categories',
            json={'categories': ['plates']},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify plates are gone
        response = client.get('/api/catalog/products?category=plates&limit=100')
        data = json.loads(response.data)
        assert len(data['products']) == 0
    
    def test_product_modification_during_csv_build(self, client, full_catalog):
        """
        Scenario: User is building CSV, modifies a product in Catalog Manager.
        """
        product_id = full_catalog['historical'][0]
        
        # Get original product
        response = client.get(f'/api/products/{product_id}')
        original = json.loads(response.data)['product']
        
        # Modify via Catalog Manager
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={
                'category': 'modified_category',
                'sku': 'MOD-999'
            },
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # CSV Builder would see updated data
        response = client.get(f'/api/products/{product_id}')
        modified = json.loads(response.data)['product']
        
        assert modified['category'] == 'modified_category'
        assert modified['sku'] == 'MOD-999'


class TestFullUserWorkflow:
    """Test complete user workflows from start to finish"""
    
    def test_complete_workflow_upload_match_cleanup(self, client):
        """
        Full workflow: Upload historical -> Upload new -> Match -> View results -> Cleanup
        """
        # Step 1: Upload historical products
        historical_ids = []
        for i in range(5):
            product_id = insert_product(
                image_path=f'/test/historical_{i}.jpg',
                category='test_category',
                product_name=f'Historical {i}',
                sku=f'HIST-{i:03d}',
                is_historical=True
            )
            historical_ids.append(product_id)
            insert_features(
                product_id=product_id,
                color_features=np.random.rand(512).astype(np.float32),
                shape_features=np.zeros(7, dtype=np.float32),
                texture_features=np.zeros(256, dtype=np.float32),
                embedding_type='clip',
                embedding_version='test'
            )
        
        # Verify historical products
        response = client.get('/api/catalog/stats')
        stats = json.loads(response.data)
        assert stats['historical_products'] == 5
        
        # Step 2: Upload new products
        new_ids = []
        for i in range(3):
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
        
        # Verify new products
        response = client.get('/api/catalog/stats')
        stats = json.loads(response.data)
        assert stats['new_products'] == 3
        
        # Step 3: Create matches
        # With CLIP, we only have one similarity score (cosine similarity of embeddings)
        # color/shape/texture scores are set to 0 as they're not used
        for new_id in new_ids:
            for hist_id in historical_ids[:2]:
                sim_score = np.random.uniform(0.6, 0.9)
                insert_match(
                    new_product_id=new_id,
                    matched_product_id=hist_id,
                    similarity_score=sim_score,
                    color_score=0.0,  # Not used with CLIP
                    shape_score=0.0,  # Not used with CLIP
                    texture_score=0.0  # Not used with CLIP
                )
        
        # Verify matches were created
        response = client.get('/api/catalog/stats')
        stats = json.loads(response.data)
        assert stats['total_matches'] == 6  # 3 new * 2 historical
        
        # Step 4: View results via Catalog Manager
        response = client.get('/api/catalog/products?limit=100')
        data = json.loads(response.data)
        assert data['total'] == 8
        
        # Step 5: Cleanup - clear matches
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'matches'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify matches cleared but products remain
        response = client.get('/api/catalog/stats')
        stats = json.loads(response.data)
        assert stats['total_matches'] == 0
        assert stats['total_products'] == 8
        
        # Step 6: Cleanup - clear new products
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'new'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify only historical remain
        response = client.get('/api/catalog/stats')
        stats = json.loads(response.data)
        assert stats['historical_products'] == 5
        assert stats['new_products'] == 0
    
    def test_workflow_with_csv_builder_integration(self, client, full_catalog):
        """
        Workflow: Use CSV Builder to prepare data -> Upload -> Match -> Export
        """
        # Step 1: CSV Builder would get file list from main app
        response = client.get('/api/catalog/products?type=historical&limit=100')
        historical_data = json.loads(response.data)
        
        # Step 2: CSV Builder exports template (simulated)
        # In real app, this generates CSV with filenames
        filenames = [p.get('filename', p.get('image_path', '').split('/')[-1]) 
                    for p in historical_data['products']]
        assert len(filenames) == 15
        
        # Step 3: User fills in CSV and uploads (already done via full_catalog fixture)
        
        # Step 4: Export results
        response = client.get('/api/catalog/export')
        assert response.status_code == 200
        assert 'text/csv' in response.content_type
        
        csv_content = response.data.decode('utf-8')
        assert 'plates' in csv_content.lower() or 'PLA-' in csv_content
    
    def test_workflow_interrupted_by_catalog_cleanup(self, client, full_catalog):
        """
        Workflow: User is mid-workflow, then clears catalog from another tab.
        """
        # Step 1: User has loaded products
        response = client.get('/api/catalog/stats')
        initial_stats = json.loads(response.data)
        assert initial_stats['total_products'] == 20
        
        # Step 2: User switches to Catalog Manager and clears historical
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'historical'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Step 3: Main app detects change via stats
        response = client.get('/api/catalog/stats')
        new_stats = json.loads(response.data)
        
        # Historical products are gone
        assert new_stats['historical_products'] == 0
        assert new_stats['new_products'] == 5
        
        # Step 4: Main app should reset state (tested via stats change detection)
        assert new_stats['total_products'] != initial_stats['total_products']


class TestCrossToolInteractions:
    """Test interactions between Main App, Catalog Manager, and CSV Builder"""
    
    def test_main_app_to_catalog_manager_to_csv_builder(self, client, full_catalog):
        """
        User flow: Main App -> Catalog Manager -> CSV Builder -> Back to Main App
        """
        # Main App: Check initial state
        response = client.get('/api/catalog/stats')
        stats = json.loads(response.data)
        initial_total = stats['total_products']
        
        # Catalog Manager: Delete some products
        response = client.post(
            '/api/catalog/products/bulk-delete',
            json={'product_ids': full_catalog['historical'][:3]},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # CSV Builder: Would see updated product list
        response = client.get('/api/catalog/products?type=historical&limit=100')
        data = json.loads(response.data)
        assert len(data['products']) == 12  # 15 - 3 deleted
        
        # Back to Main App: Stats reflect changes
        response = client.get('/api/catalog/stats')
        stats = json.loads(response.data)
        assert stats['total_products'] == initial_total - 3
    
    def test_concurrent_modifications_from_multiple_tools(self, client, full_catalog):
        """
        Simulate concurrent modifications from different tools.
        """
        product_id = full_catalog['historical'][0]
        
        # Tool 1 (Main App): Get product
        response = client.get(f'/api/products/{product_id}')
        assert response.status_code == 200
        
        # Tool 2 (Catalog Manager): Modify product
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={'category': 'modified_by_catalog_manager'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Tool 1 (Main App): Get product again - should see changes
        response = client.get(f'/api/products/{product_id}')
        data = json.loads(response.data)
        assert data['product']['category'] == 'modified_by_catalog_manager'
    
    def test_delete_product_used_in_match(self, client, full_catalog):
        """
        Delete a historical product that has matches.
        """
        # Get a historical product with matches
        hist_id = full_catalog['historical'][0]
        
        # Verify it exists
        response = client.get(f'/api/products/{hist_id}')
        assert response.status_code == 200
        
        # Delete it
        response = client.delete(f'/api/catalog/products/{hist_id}')
        assert response.status_code == 200
        
        # Verify it's gone
        response = client.get(f'/api/products/{hist_id}')
        assert response.status_code == 404
        
        # Stats should reflect the deletion
        response = client.get('/api/catalog/stats')
        stats = json.loads(response.data)
        assert stats['historical_products'] == 14  # 15 - 1


class TestCatalogManagerFeatures:
    """Verify all Catalog Manager features from task requirements"""
    
    def test_overview_tab_statistics(self, client, full_catalog):
        """Test Overview tab shows correct statistics"""
        response = client.get('/api/catalog/stats')
        data = json.loads(response.data)
        
        # Required stats
        assert 'total_products' in data
        assert 'historical_products' in data
        assert 'new_products' in data
        assert 'database_size_mb' in data
        assert 'total_matches' in data
        assert 'unique_categories' in data
        assert 'category_breakdown' in data
        
        # Data quality stats
        assert 'missing_features' in data
        assert 'missing_category' in data
        assert 'missing_sku' in data
        assert 'duplicate_skus' in data
        
        # Verify values
        assert data['total_products'] == 20
        assert data['historical_products'] == 15
        assert data['new_products'] == 5
        assert data['unique_categories'] == 5
    
    def test_browse_tab_pagination(self, client, full_catalog):
        """Test Browse tab pagination"""
        # Page 1
        response = client.get('/api/catalog/products?page=1&limit=5')
        data = json.loads(response.data)
        assert len(data['products']) == 5
        assert data['page'] == 1
        assert data['total'] == 20
        assert data['total_pages'] == 4
        
        # Page 2
        response = client.get('/api/catalog/products?page=2&limit=5')
        data = json.loads(response.data)
        assert len(data['products']) == 5
        assert data['page'] == 2
    
    def test_browse_tab_filtering(self, client, full_catalog):
        """Test Browse tab filtering options"""
        # Filter by category
        response = client.get('/api/catalog/products?category=plates')
        data = json.loads(response.data)
        assert all(p['category'] == 'plates' for p in data['products'])
        
        # Filter by type
        response = client.get('/api/catalog/products?type=historical')
        data = json.loads(response.data)
        assert all(p['is_historical'] for p in data['products'])
        
        # Filter by features
        response = client.get('/api/catalog/products?features=has_features')
        data = json.loads(response.data)
        assert all(p['has_features'] for p in data['products'])
        
        # Search
        response = client.get('/api/catalog/products?search=plates')
        data = json.loads(response.data)
        assert len(data['products']) > 0
    
    def test_browse_tab_sorting(self, client, full_catalog):
        """Test Browse tab sorting options"""
        # Sort by name ascending
        response = client.get('/api/catalog/products?sort=name_asc')
        data = json.loads(response.data)
        names = [p['product_name'] for p in data['products'] if p['product_name']]
        assert names == sorted(names)
        
        # Sort by name descending
        response = client.get('/api/catalog/products?sort=name_desc')
        data = json.loads(response.data)
        names = [p['product_name'] for p in data['products'] if p['product_name']]
        assert names == sorted(names, reverse=True)
    
    def test_cleanup_tab_all_options(self, client, full_catalog):
        """Test all Cleanup tab options"""
        # Test clear matches
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'matches'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Test clear new
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'new'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        stats = get_catalog_stats()
        assert stats['new_products'] == 0
        assert stats['historical_products'] == 15
        
        # Test clear historical
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'historical'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        stats = get_catalog_stats()
        assert stats['total_products'] == 0
    
    def test_cleanup_by_category(self, client, full_catalog):
        """Test cleanup by category"""
        response = client.post(
            '/api/catalog/cleanup/categories',
            json={'categories': ['plates', 'bowls']},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify those categories are gone
        response = client.get('/api/catalog/categories')
        data = json.loads(response.data)
        assert 'plates' not in data['categories']
        assert 'bowls' not in data['categories']
        assert 'cups' in data['categories']
    
    def test_cleanup_by_date(self, client, full_catalog):
        """Test cleanup by date"""
        response = client.post(
            '/api/catalog/cleanup/by-date',
            json={'older_than_days': 30},
            content_type='application/json'
        )
        assert response.status_code == 200
        # Products just created, so none should be deleted
    
    def test_vacuum_database(self, client, full_catalog):
        """Test vacuum database"""
        # Delete some products first
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
    
    def test_export_backup(self, client, full_catalog):
        """Test export backup"""
        response = client.get('/api/catalog/export')
        assert response.status_code == 200
        assert 'text/csv' in response.content_type
        
        csv_content = response.data.decode('utf-8')
        # Should have header and data
        lines = csv_content.strip().split('\n')
        assert len(lines) > 1
        assert 'ID' in lines[0] or 'id' in lines[0].lower()
    
    def test_product_detail_edit(self, client, full_catalog):
        """Test editing product details"""
        product_id = full_catalog['historical'][0]
        
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={
                'category': 'edited_category',
                'product_name': 'Edited Name',
                'sku': 'EDIT-001'
            },
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify changes
        product = get_product_by_id(product_id)
        assert product['category'] == 'edited_category'
        assert product['product_name'] == 'Edited Name'
        assert product['sku'] == 'EDIT-001'
    
    def test_bulk_operations(self, client, full_catalog):
        """Test bulk operations"""
        ids = full_catalog['historical'][:5]
        
        # Bulk update
        response = client.post(
            '/api/catalog/products/bulk-update',
            json={
                'product_ids': ids,
                'category': 'bulk_updated'
            },
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify
        for pid in ids:
            product = get_product_by_id(pid)
            assert product['category'] == 'bulk_updated'
        
        # Bulk delete
        response = client.post(
            '/api/catalog/products/bulk-delete',
            json={'product_ids': ids},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Verify
        for pid in ids:
            assert get_product_by_id(pid) is None


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    def test_empty_database_operations(self, client):
        """Test operations on empty database"""
        # Stats should work
        response = client.get('/api/catalog/stats')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['total_products'] == 0
        
        # Products should return empty
        response = client.get('/api/catalog/products')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['products'] == []
        
        # Cleanup should work
        response = client.post(
            '/api/catalog/cleanup',
            json={'type': 'all'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Export should work
        response = client.get('/api/catalog/export')
        assert response.status_code == 200
    
    def test_invalid_product_id(self, client):
        """Test operations with invalid product ID"""
        response = client.get('/api/products/99999')
        assert response.status_code == 404
        
        response = client.delete('/api/catalog/products/99999')
        assert response.status_code == 404
        
        response = client.put(
            '/api/catalog/products/99999',
            json={'category': 'test'},
            content_type='application/json'
        )
        assert response.status_code == 404
    
    def test_invalid_input_validation(self, client, full_catalog):
        """Test input validation"""
        product_id = full_catalog['historical'][0]
        
        # Invalid category (too long)
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={'category': 'a' * 200},
            content_type='application/json'
        )
        assert response.status_code == 400
        
        # Invalid SKU (special characters)
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={'sku': 'invalid sku!@#'},
            content_type='application/json'
        )
        assert response.status_code == 400
        
        # Empty product IDs for bulk delete
        response = client.post(
            '/api/catalog/products/bulk-delete',
            json={'product_ids': []},
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_xss_prevention(self, client, full_catalog):
        """Test XSS prevention"""
        product_id = full_catalog['historical'][0]
        
        response = client.put(
            f'/api/catalog/products/{product_id}',
            json={'product_name': '<script>alert("xss")</script>Test'},
            content_type='application/json'
        )
        assert response.status_code == 200
        
        product = get_product_by_id(product_id)
        assert '<script>' not in product['product_name']
    
    def test_sql_injection_prevention(self, client, full_catalog):
        """Test SQL injection prevention"""
        response = client.get('/api/catalog/products?search=\'; DROP TABLE products; --')
        assert response.status_code == 200
        
        # Database should still work
        stats = get_catalog_stats()
        assert stats['total_products'] == 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
