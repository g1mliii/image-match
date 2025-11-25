"""
Tests for the Catalog Snapshot API Endpoints
"""

import pytest
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from snapshot_manager import delete_snapshot, set_active_catalogs, CATALOGS_DIR


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup test snapshots after each test"""
    yield
    # Cleanup any test snapshots
    for filename in os.listdir(CATALOGS_DIR):
        if filename.startswith('api-test-') and filename.endswith('.db'):
            try:
                set_active_catalogs([], [])  # Deactivate first
                delete_snapshot(filename)
            except:
                pass


class TestSnapshotAPI:
    """Test suite for snapshot API endpoints"""
    
    def test_list_snapshots(self, client):
        """Test GET /api/catalogs/list"""
        response = client.get('/api/catalogs/list')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'historical' in data
        assert 'new' in data
        assert isinstance(data['historical'], list)
        assert isinstance(data['new'], list)
    
    def test_create_snapshot(self, client):
        """Test POST /api/catalogs/create"""
        response = client.post('/api/catalogs/create',
            data=json.dumps({
                'name': 'api-test-create',
                'is_historical': True,
                'description': 'Test snapshot',
                'tags': ['test']
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data.get('success') == True
        assert 'snapshot_file' in data
    
    def test_create_snapshot_missing_name(self, client):
        """Test POST /api/catalogs/create with missing name"""
        response = client.post('/api/catalogs/create',
            data=json.dumps({'is_historical': True}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_get_active_catalogs(self, client):
        """Test GET /api/catalogs/active"""
        response = client.get('/api/catalogs/active')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'active_historical' in data
        assert 'active_new' in data
    
    def test_set_active_catalogs(self, client):
        """Test POST /api/catalogs/active"""
        # First create a snapshot
        create_response = client.post('/api/catalogs/create',
            data=json.dumps({
                'name': 'api-test-active',
                'is_historical': True
            }),
            content_type='application/json'
        )
        create_data = json.loads(create_response.data)
        snapshot_file = create_data['snapshot_file']
        
        # Set it as active
        response = client.post('/api/catalogs/active',
            data=json.dumps({
                'historical': [snapshot_file],
                'new': []
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert snapshot_file in data.get('active_historical', [])
        
        # Cleanup: deactivate
        client.post('/api/catalogs/active',
            data=json.dumps({'historical': [], 'new': []}),
            content_type='application/json'
        )
    
    def test_get_snapshot_info(self, client):
        """Test GET /api/catalogs/{name}/info"""
        # First create a snapshot
        create_response = client.post('/api/catalogs/create',
            data=json.dumps({
                'name': 'api-test-info',
                'is_historical': True,
                'description': 'Info test'
            }),
            content_type='application/json'
        )
        create_data = json.loads(create_response.data)
        snapshot_file = create_data['snapshot_file']
        
        # Get info
        response = client.get(f'/api/catalogs/{snapshot_file}/info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'snapshot' in data
        assert data['snapshot']['name'] == 'api-test-info'
    
    def test_delete_snapshot(self, client):
        """Test DELETE /api/catalogs/{name}"""
        # First create a snapshot
        create_response = client.post('/api/catalogs/create',
            data=json.dumps({
                'name': 'api-test-delete',
                'is_historical': True
            }),
            content_type='application/json'
        )
        create_data = json.loads(create_response.data)
        snapshot_file = create_data['snapshot_file']
        
        # Delete it
        response = client.delete(f'/api/catalogs/{snapshot_file}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data.get('success') == True
    
    def test_rename_snapshot(self, client):
        """Test PUT /api/catalogs/{name}/rename"""
        # First create a snapshot
        create_response = client.post('/api/catalogs/create',
            data=json.dumps({
                'name': 'api-test-rename-original',
                'is_historical': True
            }),
            content_type='application/json'
        )
        create_data = json.loads(create_response.data)
        snapshot_file = create_data['snapshot_file']
        
        # Rename it
        response = client.put(f'/api/catalogs/{snapshot_file}/rename',
            data=json.dumps({'new_name': 'api-test-rename-new'}),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data.get('success') == True

