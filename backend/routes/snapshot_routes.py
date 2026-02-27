"""Snapshot and catalog-management endpoints extracted from app_core.py."""

import os
from datetime import datetime

from flask import Blueprint, jsonify, request, send_file

snapshot_bp = Blueprint('snapshot_routes', __name__)

# Dependencies injected from app_core to avoid circular imports.
logger = None
create_error_response = None
invalidate_csv_cache = None
invalidate_catalog_categories_cache = None
get_cached_csv = None
cache_csv_data = None
crash_detected = False


def configure_snapshot_routes(**deps):
    """Inject app_core dependencies used by snapshot endpoints."""
    global logger
    global create_error_response
    global invalidate_csv_cache
    global invalidate_catalog_categories_cache
    global get_cached_csv
    global cache_csv_data
    global crash_detected

    logger = deps['logger']
    create_error_response = deps['create_error_response']
    invalidate_csv_cache = deps['invalidate_csv_cache']
    invalidate_catalog_categories_cache = deps['invalidate_catalog_categories_cache']
    get_cached_csv = deps['get_cached_csv']
    cache_csv_data = deps['cache_csv_data']
    crash_detected = bool(deps.get('crash_detected', False))


@snapshot_bp.route('/api/catalogs/list', methods=['GET'])
def list_catalog_snapshots():
    """
    List all available catalog snapshots.
    
    Returns:
    - 200: Success with historical and new snapshot lists
    - 500: Server error
    """
    try:
        from snapshot_manager import list_snapshots, migrate_legacy_database
        
        # Check for migration on first access
        migrate_legacy_database()
        
        result = list_snapshots()
        
        if result.get('error'):
            return create_error_response(
                'LIST_ERROR',
                result['error'],
                status_code=500
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing snapshots: {e}", exc_info=True)
        return create_error_response(
            'LIST_ERROR',
            'Failed to list snapshots',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/storage', methods=['GET'])
def get_snapshot_storage_info():
    """
    Get total disk space used by all snapshots.

    Returns:
    - 200: Success with storage breakdown
    - 500: Server error
    """
    try:
        from snapshot_manager import get_total_snapshot_storage

        result = get_total_snapshot_storage()

        return jsonify({
            'status': 'success',
            **result
        }), 200

    except Exception as e:
        logger.error(f"Error getting snapshot storage info: {e}", exc_info=True)
        return create_error_response(
            'STORAGE_ERROR',
            'Failed to get storage information',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/check-duplicate/<section>', methods=['GET'])
def check_catalog_duplicate(section):
    """
    Check if a duplicate snapshot already exists for a catalog section.

    Args:
        section: 'historical' or 'new'

    Returns:
    - 200: Success with duplicate check results
    - 400: Invalid section
    - 500: Server error
    """
    try:
        if section not in ['historical', 'new']:
            return create_error_response(
                'INVALID_SECTION',
                f"Invalid section: {section}. Must be 'historical' or 'new'",
                status_code=400
            )

        from snapshot_manager import check_snapshot_duplicate

        result = check_snapshot_duplicate(section)

        if result.get('is_duplicate'):
            logger.info(f"Duplicate detected for {section} catalog: {result['existing_snapshot']}")

        return jsonify({
            'status': 'success',
            **result
        }), 200

    except Exception as e:
        logger.error(f"Error checking for duplicate snapshot: {e}", exc_info=True)
        return create_error_response(
            'DUPLICATE_CHECK_ERROR',
            'Failed to check for duplicate',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/save-with-dialog', methods=['POST'])
def save_catalog_with_dialog_choice():
    """
    Save catalog snapshot based on user's dialog choice.

    JSON body:
    {
      "section": "historical" or "new",
      "snapshot_name": "User provided name (for persistent saves)",
      "choice": "skip" | "session" | "persistent",
      "operation": "what triggered this (e.g., 'catalog_replace', 'bulk_import')"
    }

    Returns:
    - 200: Success with snapshot info
    - 400: Invalid parameters
    - 500: Server error
    """
    try:
        data = request.get_json() or {}
        section = data.get('section', '').lower()
        snapshot_name = data.get('snapshot_name', '')
        choice = data.get('choice', '').lower()
        operation = data.get('operation')

        # Validate
        if section not in ['historical', 'new']:
            return create_error_response(
                'INVALID_SECTION',
                f"Invalid section: {section}. Must be 'historical' or 'new'",
                status_code=400
            )

        if choice not in ['skip', 'session', 'persistent']:
            return create_error_response(
                'INVALID_CHOICE',
                f"Invalid choice: {choice}. Must be 'skip', 'session', or 'persistent'",
                status_code=400
            )

        if choice == 'persistent' and not snapshot_name:
            return create_error_response(
                'MISSING_NAME',
                "Snapshot name required for persistent saves",
                status_code=400
            )

        from snapshot_manager import save_snapshot_with_dialog_choice

        result = save_snapshot_with_dialog_choice(section, snapshot_name, choice, operation)

        if 'error' in result:
            return create_error_response(
                'SAVE_ERROR',
                result['error'],
                status_code=500
            )

        return jsonify({
            'status': 'success',
            **result
        }), 200

    except Exception as e:
        logger.error(f"Error saving catalog with dialog: {e}", exc_info=True)
        return create_error_response(
            'SAVE_ERROR',
            'Failed to save catalog',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/check-crash-recovery', methods=['GET'])
def check_crash_recovery():
    """
    Check if app crashed and offer recovery option.

    Returns:
    - 200: With crash detection result and available recovery snapshots
    """
    try:
        crash_detected = globals().get('crash_detected', False)

        recovery_info = {
            'crash_detected': crash_detected,
            'available_recovery': None
        }

        if crash_detected:
            # Find most recent session autosave
            from snapshot_manager import list_snapshots

            all_snapshots = list_snapshots()

            # Check both historical and new for session autosaves
            latest_session = None
            latest_time = None

            for section_key in ['historical', 'new']:
                for snapshot in all_snapshots.get(section_key, []):
                    if snapshot.get('session_only') and 'error' not in snapshot:
                        created_at = snapshot.get('created_at')
                        if created_at and (latest_time is None or created_at > latest_time):
                            latest_session = snapshot
                            latest_time = created_at

            if latest_session:
                recovery_info['recovery_snapshot'] = {
                    'id': latest_session.get('snapshot_file'),  # Use snapshot_file as ID
                    'name': latest_session.get('name'),
                    'section': 'historical' if latest_session.get('is_historical') else 'new',
                    'created_at': latest_session.get('created_at'),
                    'product_count': latest_session.get('product_count', 0),
                    'created_by_operation': latest_session.get('created_by_operation')
                }

        return jsonify({
            'status': 'success',
            **recovery_info
        }), 200

    except Exception as e:
        logger.error(f"Error checking crash recovery: {e}", exc_info=True)
        return create_error_response(
            'RECOVERY_CHECK_ERROR',
            'Failed to check crash recovery',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/create', methods=['POST'])
def create_catalog_snapshot():
    """
    Create a new catalog snapshot.
    
    JSON body:
    - name: Snapshot display name (required)
    - is_historical: Whether this is a historical catalog (default: true)
    - description: Optional description
    - tags: Optional list of tags
    
    Returns:
    - 200: Success with snapshot info
    - 400: Validation error
    - 500: Server error
    """
    try:
        from snapshot_manager import create_snapshot
        
        data = request.get_json()
        
        if not data or 'name' not in data:
            return create_error_response(
                'MISSING_NAME',
                'Snapshot name is required',
                status_code=400
            )
        
        name = data['name'].strip()
        if not name:
            return create_error_response(
                'INVALID_NAME',
                'Snapshot name cannot be empty',
                status_code=400
            )
        
        if len(name) > 100:
            return create_error_response(
                'NAME_TOO_LONG',
                'Snapshot name must be 100 characters or less',
                status_code=400
            )
        
        is_historical = data.get('is_historical', True)
        description = data.get('description', '')
        tags = data.get('tags', [])
        
        result = create_snapshot(
            name=name,
            is_historical=is_historical,
            description=description,
            tags=tags
        )
        
        if result.get('error'):
            return create_error_response(
                'CREATE_ERROR',
                result['error'],
                status_code=400
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error creating snapshot: {e}", exc_info=True)
        return create_error_response(
            'CREATE_ERROR',
            'Failed to create snapshot',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/<path:snapshot_name>', methods=['DELETE'])
def delete_catalog_snapshot(snapshot_name):
    """
    Delete a catalog snapshot.
    
    Returns:
    - 200: Success
    - 400: Cannot delete active snapshot
    - 404: Snapshot not found
    - 500: Server error
    """
    try:
        from snapshot_manager import delete_snapshot
        
        result = delete_snapshot(snapshot_name)

        if result.get('error'):
            if 'not found' in result['error'].lower():
                return create_error_response(
                    'NOT_FOUND',
                    result['error'],
                    status_code=404
                )
            return create_error_response(
                'DELETE_ERROR',
                result['error'],
                status_code=400
            )

        # Clear category cache for the deleted catalog
        invalidate_catalog_categories_cache(snapshot_name)
        logger.info(f"[CACHE] Cleared category cache for deleted catalog: {snapshot_name}")

        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting snapshot: {e}", exc_info=True)
        return create_error_response(
            'DELETE_ERROR',
            'Failed to delete snapshot',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/<path:snapshot_name>/rename', methods=['PUT'])
def rename_catalog_snapshot(snapshot_name):
    """
    Rename a catalog snapshot.
    
    JSON body:
    - new_name: New display name (required)
    
    Returns:
    - 200: Success
    - 400: Validation error
    - 404: Snapshot not found
    - 500: Server error
    """
    try:
        from snapshot_manager import rename_snapshot
        
        data = request.get_json()
        
        if not data or 'new_name' not in data:
            return create_error_response(
                'MISSING_NAME',
                'New name is required',
                status_code=400
            )
        
        new_name = data['new_name'].strip()
        if not new_name:
            return create_error_response(
                'INVALID_NAME',
                'New name cannot be empty',
                status_code=400
            )
        
        result = rename_snapshot(snapshot_name, new_name)
        
        if result.get('error'):
            if 'not found' in result['error'].lower():
                return create_error_response(
                    'NOT_FOUND',
                    result['error'],
                    status_code=404
                )
            return create_error_response(
                'RENAME_ERROR',
                result['error'],
                status_code=400
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error renaming snapshot: {e}", exc_info=True)
        return create_error_response(
            'RENAME_ERROR',
            'Failed to rename snapshot',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/merge', methods=['POST'])
def merge_catalog_snapshots():
    """
    Merge multiple snapshots into a new one.
    
    JSON body:
    - snapshots: List of snapshot filenames to merge (required, min 2)
    - new_name: Name for merged snapshot (required)
    - is_historical: Whether merged snapshot is historical (default: true)
    
    Returns:
    - 200: Success with merged snapshot info
    - 400: Validation error
    - 500: Server error
    """
    try:
        from snapshot_manager import merge_snapshots
        
        data = request.get_json()
        
        if not data:
            return create_error_response(
                'MISSING_DATA',
                'Request body is required',
                status_code=400
            )
        
        snapshots = data.get('snapshots', [])
        if len(snapshots) < 2:
            return create_error_response(
                'INVALID_SNAPSHOTS',
                'At least 2 snapshots are required for merge',
                status_code=400
            )
        
        new_name = data.get('new_name', '').strip()
        if not new_name:
            return create_error_response(
                'MISSING_NAME',
                'New name is required',
                status_code=400
            )
        
        is_historical = data.get('is_historical', True)
        
        result = merge_snapshots(snapshots, new_name, is_historical)
        
        if result.get('error'):
            return create_error_response(
                'MERGE_ERROR',
                result['error'],
                status_code=400
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error merging snapshots: {e}", exc_info=True)
        return create_error_response(
            'MERGE_ERROR',
            'Failed to merge snapshots',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/active', methods=['GET'])
def get_active_catalog_snapshots():
    """
    Get currently active catalog snapshots.
    
    Returns:
    - 200: Success with active snapshot lists
    - 500: Server error
    """
    try:
        from snapshot_manager import get_active_catalogs, get_combined_products_count
        
        active = get_active_catalogs()
        counts = get_combined_products_count()
        
        return jsonify({
            'status': 'success',
            **active,
            **counts
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting active catalogs: {e}", exc_info=True)
        return create_error_response(
            'ACTIVE_ERROR',
            'Failed to get active catalogs',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/active', methods=['POST'])
def set_active_catalog_snapshots():
    """
    Set active catalog snapshots.
    
    JSON body:
    - historical: List of historical snapshot filenames
    - new: List of new product snapshot filenames
    
    Returns:
    - 200: Success
    - 400: Validation error
    - 500: Server error
    """
    try:
        from snapshot_manager import set_active_catalogs
        
        data = request.get_json()
        
        if not data:
            return create_error_response(
                'MISSING_DATA',
                'Request body is required',
                status_code=400
            )
        
        historical = data.get('historical', [])
        new = data.get('new', [])
        
        if not isinstance(historical, list) or not isinstance(new, list):
            return create_error_response(
                'INVALID_DATA',
                'historical and new must be arrays',
                status_code=400
            )
        
        result = set_active_catalogs(historical, new)
        
        if result.get('error'):
            return create_error_response(
                'SET_ACTIVE_ERROR',
                result['error'],
                status_code=400
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting active catalogs: {e}", exc_info=True)
        return create_error_response(
            'SET_ACTIVE_ERROR',
            'Failed to set active catalogs',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/<path:snapshot_name>/info', methods=['GET'])
def get_catalog_snapshot_info(snapshot_name):
    """
    Get detailed info for a specific snapshot.
    
    Returns:
    - 200: Success with snapshot info
    - 404: Snapshot not found
    - 500: Server error
    """
    try:
        from snapshot_manager import get_snapshot_info
        
        result = get_snapshot_info(snapshot_name)
        
        if result.get('error'):
            if 'not found' in result['error'].lower():
                return create_error_response(
                    'NOT_FOUND',
                    result['error'],
                    status_code=404
                )
            return create_error_response(
                'INFO_ERROR',
                result['error'],
                status_code=500
            )
        
        return jsonify({
            'status': 'success',
            'snapshot': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting snapshot info: {e}", exc_info=True)
        return create_error_response(
            'INFO_ERROR',
            'Failed to get snapshot info',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/export', methods=['POST'])
def export_catalog_snapshot():
    """
    Export a snapshot as a downloadable .zip file.
    
    JSON body:
    - snapshot: Snapshot filename to export (required)
    
    Returns:
    - 200: Zip file download
    - 400: Validation error
    - 404: Snapshot not found
    - 500: Server error
    """
    try:
        from snapshot_manager import export_snapshot, CATALOGS_DIR
        
        data = request.get_json()
        
        if not data or 'snapshot' not in data:
            return create_error_response(
                'MISSING_SNAPSHOT',
                'Snapshot name is required',
                status_code=400
            )
        
        snapshot_name = data['snapshot']
        result = export_snapshot(snapshot_name)
        
        if result.get('error'):
            if 'not found' in result['error'].lower():
                return create_error_response(
                    'NOT_FOUND',
                    result['error'],
                    status_code=404
                )
            return create_error_response(
                'EXPORT_ERROR',
                result['error'],
                status_code=500
            )
        
        # Return the zip file
        zip_path = result['zip_path']
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=os.path.basename(zip_path)
        )
        
    except Exception as e:
        logger.error(f"Error exporting snapshot: {e}", exc_info=True)
        return create_error_response(
            'EXPORT_ERROR',
            'Failed to export snapshot',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/import', methods=['POST'])
def import_catalog_snapshot():
    """
    Import a snapshot from an uploaded .zip file.
    
    Form data:
    - file: Zip file to import (required)
    
    Returns:
    - 200: Success with imported snapshot info
    - 400: Validation error
    - 500: Server error
    """
    try:
        from snapshot_manager import import_snapshot, CATALOGS_DIR
        
        if 'file' not in request.files:
            return create_error_response(
                'MISSING_FILE',
                'Zip file is required',
                status_code=400
            )
        
        file = request.files['file']
        
        if file.filename == '':
            return create_error_response(
                'EMPTY_FILENAME',
                'No file selected',
                status_code=400
            )
        
        if not file.filename.endswith('.zip'):
            return create_error_response(
                'INVALID_FORMAT',
                'File must be a .zip archive',
                status_code=400
            )
        
        # Save uploaded file temporarily
        temp_path = os.path.join(CATALOGS_DIR, f"temp-import-{datetime.now().strftime('%Y%m%d%H%M%S')}.zip")
        file.save(temp_path)
        
        try:
            result = import_snapshot(temp_path)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if result.get('error'):
            return create_error_response(
                'IMPORT_ERROR',
                result['error'],
                status_code=400
            )

        # Invalidate CSV cache since catalog was imported
        invalidate_csv_cache()

        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error importing snapshot: {e}", exc_info=True)
        return create_error_response(
            'IMPORT_ERROR',
            'Failed to import snapshot',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/save-current', methods=['POST'])
def save_current_as_snapshot():
    """
    Save the current main database as a new snapshot.
    
    JSON body:
    - name: Snapshot name (required)
    - description: Optional description
    - tags: Optional list of tags
    
    Returns:
    - 200: Success with snapshot info
    - 400: Validation error
    - 500: Server error
    """
    try:
        from snapshot_manager import save_main_db_as_snapshot
        
        data = request.get_json()
        
        if not data or 'name' not in data:
            return create_error_response(
                'MISSING_NAME',
                'Snapshot name is required',
                status_code=400
            )
        
        name = data['name'].strip()
        if not name:
            return create_error_response(
                'INVALID_NAME',
                'Snapshot name cannot be empty',
                status_code=400
            )
        
        description = data.get('description', '')
        tags = data.get('tags', [])
        skip_if_empty_raw = data.get('skip_if_empty', False)
        skip_if_empty = (
            skip_if_empty_raw
            if isinstance(skip_if_empty_raw, bool)
            else str(skip_if_empty_raw).strip().lower() in ('1', 'true', 'yes', 'on')
        )

        # Check if this is a session save (temporary, expires in 1 hour)
        session_only = 'session' in [tag.lower() for tag in tags] if tags else False

        logger.info(
            f"[SNAPSHOT-API] Saving snapshot '{name}' - Type: {'Session' if session_only else 'Persistent'}, "
            f"Tags: {tags}, SkipIfEmpty: {skip_if_empty}"
        )
        result = save_main_db_as_snapshot(
            name,
            description,
            tags,
            session_only=session_only,
            skip_if_empty=skip_if_empty
        )
        
        if result.get('error'):
            return create_error_response(
                'SAVE_ERROR',
                result['error'],
                status_code=400
            )
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error saving current as snapshot: {e}", exc_info=True)
        return create_error_response(
            'SAVE_ERROR',
            'Failed to save snapshot',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/load/<path:snapshot_name>', methods=['POST'])
def load_snapshot_to_main(snapshot_name):
    """
    Load a snapshot into the main database.
    
    This replaces the current main database with the snapshot contents.
    
    Returns:
    - 200: Success
    - 404: Snapshot not found
    - 500: Server error
    """
    try:
        from snapshot_manager import load_snapshot_to_main_db
        
        result = load_snapshot_to_main_db(snapshot_name)
        
        if result.get('error'):
            if result.get('error_code') == 'INTERNAL_SNAPSHOT_BLOCKED':
                return create_error_response(
                    'FORBIDDEN',
                    result['error'],
                    status_code=403
                )
            if 'not found' in result['error'].lower():
                return create_error_response(
                    'NOT_FOUND',
                    result['error'],
                    status_code=404
                )
            return create_error_response(
                'LOAD_ERROR',
                result['error'],
                status_code=500
            )
        
        # Invalidate CSV cache since snapshot was loaded
        invalidate_csv_cache()

        # Rebuild FAISS indexes after loading snapshot
        logger.info("Rebuilding FAISS indexes after snapshot load...")
        try:
            from database import rebuild_all_faiss_indexes
            faiss_stats = rebuild_all_faiss_indexes()
            if 'error' not in faiss_stats:
                logger.info(f"FAISS indexes rebuilt: {faiss_stats['categories_processed']} categories")
                result['faiss_indexes_rebuilt'] = faiss_stats['categories_processed']
        except Exception as e:
            logger.warning(f"Failed to rebuild FAISS indexes: {e}")
        
        return jsonify({
            'status': 'success',
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error loading snapshot: {e}", exc_info=True)
        return create_error_response(
            'LOAD_ERROR',
            'Failed to load snapshot',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/main-db-stats', methods=['GET'])
def get_main_database_stats():
    """
    Get statistics about the main database and loaded snapshot info.
    
    Returns:
    - 200: Success with stats
    - 500: Server error
    """
    try:
        from snapshot_manager import get_main_db_stats
        
        stats = get_main_db_stats()
        
        return jsonify({
            'status': 'success',
            **stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting main db stats: {e}", exc_info=True)
        return create_error_response(
            'STATS_ERROR',
            'Failed to get database stats',
            {'error': str(e)},
            status_code=500
        )


@snapshot_bp.route('/api/catalogs/csv-content', methods=['GET'])
def get_catalog_csv_content():
    """Get CSV content for a specific section from the loaded catalog

    Query Parameters:
    - section: 'historical' or 'new' (default: 'historical')

    Returns:
    - 200: Success with CSV content and metadata
    - 404: No catalog loaded or no CSV found
    - 400: Invalid section parameter
    - 500: Server error
    """
    try:
        from snapshot_manager import get_loaded_snapshot_info, get_csv_from_snapshot, CATALOGS_DIR

        # Get section parameter (default to 'historical')
        section = request.args.get('section', 'historical').lower()

        if section not in ['historical', 'new']:
            logger.warning(f"[CSV-AUTO-LOAD] Invalid section: {section}")
            return create_error_response(
                'INVALID_SECTION',
                f"Invalid section '{section}'. Must be 'historical' or 'new'",
                status_code=400
            )

        is_historical = (section == 'historical')

        # Get currently loaded snapshot
        loaded_info = get_loaded_snapshot_info()
        snapshot_file = loaded_info.get('snapshot_file') if loaded_info.get('loaded') else None

        # Check cache first (cache key: snapshot_file + section)
        cache_key = f"{snapshot_file}:{section}"
        csv_data = get_cached_csv(cache_key)
        if csv_data:
            logger.debug(f"[CSV-AUTO-LOAD] ✓ Serving {section} CSV from cache")
            logger.debug(f"[CSV-AUTO-LOAD] ✓ LOADED (cached): {section} CSV - {csv_data['filename']} ({csv_data['row_count']} rows)")
            return jsonify({
                'has_csv': True,
                'section': section,
                'csv_content': csv_data['csv_content'],
                'filename': csv_data['filename'],
                'row_count': csv_data['row_count'],
                'uploaded_at': csv_data.get('uploaded_at'),
                'cached': True
            }), 200

        logger.debug(f"[CSV-AUTO-LOAD] ▶ Request for {section} CSV auto-load (not cached)")

        if loaded_info.get('loaded'):
            # Snapshot is explicitly loaded
            logger.debug(f"[CSV-AUTO-LOAD]   Loaded snapshot: {snapshot_file}")
        else:
            # No explicit snapshot loaded - try to use main database directly
            # This handles the case where user selected "use_existing" without explicitly loading a snapshot
            logger.debug(f"[CSV-AUTO-LOAD] No explicit snapshot loaded - checking main database for CSV metadata...")

        # Load CSV - either from snapshot or main database
        csv_data = None
        from snapshot_manager import extract_csv_from_db, DEFAULT_DB_PATH

        if snapshot_file:
            # Get CSV from snapshot for specific section
            snapshot_path = os.path.join(CATALOGS_DIR, snapshot_file)

            if not os.path.exists(snapshot_path):
                logger.error(f"[CSV-AUTO-LOAD] ✗ Snapshot file not found at: {snapshot_path}")
                return create_error_response(
                    'SNAPSHOT_NOT_FOUND',
                    f'Snapshot file not found: {snapshot_file}',
                    status_code=404
                )

            logger.debug(f"[CSV-AUTO-LOAD]   Retrieving {section} CSV from snapshot...")
            csv_data = get_csv_from_snapshot(snapshot_path, is_historical=is_historical)

            # Fallback: if snapshot doesn't have CSV (e.g., created before CSV column fix), extract from main DB
            if not csv_data:
                logger.debug(f"[CSV-AUTO-LOAD]   Snapshot has no CSV data (old snapshot?), falling back to main database...")
                try:
                    csv_result = extract_csv_from_db(DEFAULT_DB_PATH, is_historical=is_historical)
                    if csv_result:
                        csv_content, row_count = csv_result
                        csv_data = {
                            'csv_content': csv_content,
                            'filename': f"{section}-{datetime.now().strftime('%Y%m%d')}.csv",
                            'row_count': row_count,
                            'uploaded_at': datetime.now().isoformat(),
                            'section': section
                        }
                        logger.debug(f"[CSV-AUTO-LOAD] ✓ Fallback: Extracted from main DB: {csv_data['filename']} ({row_count} rows)")
                except Exception as e:
                    logger.warning(f"[CSV-AUTO-LOAD] Could not extract CSV from main database: {e}")
                    csv_data = None
        else:
            # No snapshot loaded - try to extract CSV from main database
            # This is used when user selected "use_existing" without formally loading a snapshot
            logger.debug(f"[CSV-AUTO-LOAD]   Extracting {section} CSV from main database...")

            try:
                csv_result = extract_csv_from_db(DEFAULT_DB_PATH, is_historical=is_historical)
                if csv_result:
                    csv_content, row_count = csv_result
                    csv_data = {
                        'csv_content': csv_content,
                        'filename': f"{section}-{datetime.now().strftime('%Y%m%d')}.csv",
                        'row_count': row_count,
                        'uploaded_at': datetime.now().isoformat(),
                        'section': section
                    }
                    logger.debug(f"[CSV-AUTO-LOAD] ✓ Extracted from main DB: {csv_data['filename']} ({row_count} rows)")
            except Exception as e:
                logger.warning(f"[CSV-AUTO-LOAD] Could not extract CSV from main database: {e}")
                csv_data = None

        if not csv_data:
            logger.debug(f"[CSV-AUTO-LOAD] ✗ No {section} CSV found in snapshot (normal if not used)")
            return jsonify({
                'has_csv': False,
                'section': section,
                'message': f'No CSV data for {section} section'
            }), 200

        # Cache the result for future requests (with LRU eviction)
        cache_csv_data(cache_key, csv_data)
        logger.debug(f"[CSV-AUTO-LOAD] ✓ LOADED: {section} CSV - {csv_data['filename']} ({csv_data['row_count']} rows)")
        return jsonify({
            'has_csv': True,
            'section': section,
            'csv_content': csv_data['csv_content'],
            'filename': csv_data['filename'],
            'row_count': csv_data['row_count'],
            'uploaded_at': csv_data['uploaded_at']
        }), 200

    except Exception as e:
        logger.error(f"[CSV-AUTO-LOAD] ✗ ERROR: {str(e)}", exc_info=True)
        return create_error_response(
            'CSV_RETRIEVAL_ERROR',
            'Failed to retrieve CSV content',
            {'error': str(e)},
            status_code=500
        )


