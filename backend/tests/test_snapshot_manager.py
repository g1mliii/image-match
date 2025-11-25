"""
Tests for the Catalog Snapshot Management System
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapshot_manager import (
    create_snapshot, list_snapshots, get_snapshot_info,
    delete_snapshot, rename_snapshot, merge_snapshots,
    get_active_catalogs, set_active_catalogs,
    sanitize_snapshot_name, get_snapshot_db_path,
    CATALOGS_DIR, CONFIG_DIR
)


class TestSnapshotManager:
    """Test suite for snapshot management functions"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test"""
        # Store original state
        self.test_snapshots = []
        yield
        # Cleanup test snapshots
        for snapshot in self.test_snapshots:
            try:
                delete_snapshot(snapshot)
            except:
                pass
    
    def test_sanitize_snapshot_name(self):
        """Test snapshot name sanitization"""
        assert sanitize_snapshot_name("Test Snapshot") == "Test-Snapshot"
        assert sanitize_snapshot_name("test/invalid:name") == "testinvalidname"
        assert sanitize_snapshot_name("  spaces  ") == "spaces"
        assert sanitize_snapshot_name("") != ""  # Should generate default name
    
    def test_create_snapshot(self):
        """Test creating a new snapshot"""
        result = create_snapshot(
            name="Test Create Snapshot",
            is_historical=True,
            description="Test description",
            tags=["test", "unit"]
        )
        
        assert result.get('success') == True
        assert 'snapshot_file' in result
        self.test_snapshots.append(result['snapshot_file'])
        
        # Verify file was created
        db_path = get_snapshot_db_path(result['snapshot_file'])
        assert os.path.exists(db_path)
    
    def test_create_duplicate_snapshot(self):
        """Test that creating duplicate snapshot fails"""
        import time
        unique_name = f"Duplicate Test {int(time.time())}"
        
        result1 = create_snapshot(name=unique_name, is_historical=True)
        assert result1.get('success') == True
        self.test_snapshots.append(result1['snapshot_file'])
        
        result2 = create_snapshot(name=unique_name, is_historical=True)
        assert 'error' in result2
        assert 'already exists' in result2['error'].lower()
    
    def test_list_snapshots(self):
        """Test listing snapshots"""
        # Create test snapshots
        hist_result = create_snapshot(name="List Test Historical", is_historical=True)
        new_result = create_snapshot(name="List Test New", is_historical=False)
        
        self.test_snapshots.append(hist_result['snapshot_file'])
        self.test_snapshots.append(new_result['snapshot_file'])
        
        # List snapshots
        snapshots = list_snapshots()
        
        assert 'historical' in snapshots
        assert 'new' in snapshots
        
        # Find our test snapshots
        hist_names = [s['name'] for s in snapshots['historical']]
        new_names = [s['name'] for s in snapshots['new']]
        
        assert "List Test Historical" in hist_names
        assert "List Test New" in new_names
    
    def test_get_snapshot_info(self):
        """Test getting snapshot info"""
        result = create_snapshot(
            name="Info Test",
            is_historical=True,
            description="Test info",
            tags=["info", "test"]
        )
        self.test_snapshots.append(result['snapshot_file'])
        
        info = get_snapshot_info(result['snapshot_file'])
        
        assert info['name'] == "Info Test"
        assert info['is_historical'] == True
        assert info['description'] == "Test info"
        assert info['tags'] == ["info", "test"]
        assert info['product_count'] == 0
        assert 'size_mb' in info
    
    def test_delete_snapshot(self):
        """Test deleting a snapshot"""
        result = create_snapshot(name="Delete Test", is_historical=True)
        snapshot_file = result['snapshot_file']
        
        # Verify it exists
        db_path = get_snapshot_db_path(snapshot_file)
        assert os.path.exists(db_path)
        
        # Delete it
        delete_result = delete_snapshot(snapshot_file)
        assert delete_result.get('success') == True
        
        # Verify it's gone
        assert not os.path.exists(db_path)
    
    def test_delete_active_snapshot_fails(self):
        """Test that deleting an active snapshot fails"""
        result = create_snapshot(name="Active Delete Test", is_historical=True)
        self.test_snapshots.append(result['snapshot_file'])
        
        # Set as active
        set_active_catalogs([result['snapshot_file']], [])
        
        # Try to delete
        delete_result = delete_snapshot(result['snapshot_file'])
        assert 'error' in delete_result
        assert 'active' in delete_result['error'].lower()
        
        # Cleanup: deactivate first
        set_active_catalogs([], [])
    
    def test_rename_snapshot(self):
        """Test renaming a snapshot"""
        result = create_snapshot(name="Rename Test Original", is_historical=True)
        self.test_snapshots.append(result['snapshot_file'])
        
        rename_result = rename_snapshot(result['snapshot_file'], "Rename Test New Name")
        
        assert rename_result.get('success') == True
        assert 'new_name' in rename_result
        
        # Update tracked snapshot for cleanup
        self.test_snapshots.remove(result['snapshot_file'])
        self.test_snapshots.append(rename_result['new_name'])
        
        # Verify new name in info
        info = get_snapshot_info(rename_result['new_name'])
        assert info['name'] == "Rename Test New Name"
    
    def test_active_catalogs(self):
        """Test getting and setting active catalogs"""
        import time
        unique_suffix = int(time.time())
        
        result1 = create_snapshot(name=f"Active Test 1 {unique_suffix}", is_historical=True)
        assert result1.get('success') == True
        result2 = create_snapshot(name=f"Active Test 2 {unique_suffix}", is_historical=False)
        assert result2.get('success') == True
        
        self.test_snapshots.append(result1['snapshot_file'])
        self.test_snapshots.append(result2['snapshot_file'])
        
        # Set active
        set_result = set_active_catalogs(
            [result1['snapshot_file']],
            [result2['snapshot_file']]
        )
        
        assert set_result.get('success') == True
        
        # Get active
        active = get_active_catalogs()
        assert result1['snapshot_file'] in active['active_historical']
        assert result2['snapshot_file'] in active['active_new']
        
        # Cleanup
        set_active_catalogs([], [])
        
        assert set_result.get('success') == True
        
        # Get active
        active = get_active_catalogs()
