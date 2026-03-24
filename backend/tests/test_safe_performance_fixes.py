import io
import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import database
import product_matching
import snapshot_manager


def _make_feature_vectors(scale: float = 1.0):
    color = np.arange(1, 257, dtype=np.float32) * scale
    color /= color.sum()

    shape = np.linspace(0.1, 0.7, 7, dtype=np.float32) * scale

    texture = np.arange(256, 0, -1, dtype=np.float32) * scale
    texture /= texture.sum()

    return color, shape, texture


def _serialize_array(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    original_db_path = database.DB_PATH
    database.close_all_db_connections()
    monkeypatch.setattr(database, 'DB_PATH', str(tmp_path / 'test.db'))
    database.init_db()
    yield
    database.close_all_db_connections()
    monkeypatch.setattr(database, 'DB_PATH', original_db_path)


@pytest.fixture
def isolated_snapshot_dirs(tmp_path, monkeypatch):
    catalogs_dir = tmp_path / 'catalogs'
    config_dir = tmp_path / 'config'
    catalogs_dir.mkdir()
    config_dir.mkdir()

    monkeypatch.setattr(snapshot_manager, 'CATALOGS_DIR', str(catalogs_dir))
    monkeypatch.setattr(snapshot_manager, 'CONFIG_DIR', str(config_dir))
    monkeypatch.setattr(
        snapshot_manager,
        'ACTIVE_CATALOGS_FILE',
        str(config_dir / 'active_catalogs.json')
    )
    snapshot_manager.invalidate_snapshot_list_cache()
    yield
    snapshot_manager.invalidate_snapshot_list_cache()


def test_latest_feature_query_returns_newest_row_and_index_exists(isolated_database):
    product_id = database.insert_product(
        image_path='/tmp/product-a.jpg',
        category='chairs',
        product_name='Chair A',
        sku='CHAIR-A',
        is_historical=True
    )
    other_product_id = database.insert_product(
        image_path='/tmp/product-b.jpg',
        category='chairs',
        product_name='Chair B',
        sku='CHAIR-B',
        is_historical=True
    )

    color_v1, shape_v1, texture_v1 = _make_feature_vectors(scale=1.0)
    color_v2, shape_v2, texture_v2 = _make_feature_vectors(scale=2.0)
    other_color, other_shape, other_texture = _make_feature_vectors(scale=3.0)

    database.insert_features(product_id, color_v1, shape_v1, texture_v1)
    database.insert_features(product_id, color_v2, shape_v2, texture_v2)
    database.insert_features(other_product_id, other_color, other_shape, other_texture)

    batches = list(database.iter_all_features_by_category(
        category='chairs',
        is_historical=True,
        batch_size=1
    ))
    flattened = [item for batch in batches for item in batch]

    assert len(batches) == 2
    assert len(flattened) == 2

    by_product = {pid: features for pid, features in flattened}
    assert np.allclose(by_product[product_id]['color_features'], color_v2)
    assert np.allclose(by_product[product_id]['shape_features'], shape_v2)
    assert np.allclose(by_product[product_id]['texture_features'], texture_v2)
    assert np.allclose(by_product[other_product_id]['color_features'], other_color)

    with sqlite3.connect(database.DB_PATH) as conn:
        index_rows = conn.execute('PRAGMA index_list(features)').fetchall()
        index_names = {row[1] for row in index_rows}
        assert 'idx_features_product_id_id' in index_names

        index_info = conn.execute('PRAGMA index_info(idx_features_product_id_id)').fetchall()
        indexed_columns = [row[2] for row in index_info]
        assert indexed_columns == ['product_id', 'id']


def test_find_matches_uses_streamed_bruteforce_batches(isolated_database, monkeypatch):
    monkeypatch.setattr(product_matching, 'CLIP_AVAILABLE', False)
    monkeypatch.setattr(product_matching, '_BRUTE_FORCE_FEATURE_BATCH_SIZE', 1)

    color, shape, texture = _make_feature_vectors()

    for suffix in ('A', 'B', 'C'):
        historical_id = database.insert_product(
            image_path=f'/tmp/historical-{suffix}.jpg',
            category='placemats',
            product_name=f'Historical {suffix}',
            sku=f'H-{suffix}',
            is_historical=True
        )
        database.insert_features(historical_id, color, shape, texture)

    new_product_id = database.insert_product(
        image_path='/tmp/new.jpg',
        category='placemats',
        product_name='New Product',
        sku='NEW-1',
        is_historical=False
    )
    database.insert_features(new_product_id, color, shape, texture)

    result = product_matching.find_matches(
        product_id=new_product_id,
        threshold=30.0,
        limit=10,
        store_matches=False
    )

    assert result['total_candidates'] == 3
    assert result['successful_matches'] == 3
    assert result['failed_matches'] == 0
    assert len(result['matches']) == 3
    assert all(match['similarity_score'] >= 99.0 for match in result['matches'])


def test_merge_snapshots_preserves_large_history_batches(isolated_snapshot_dirs):
    created = snapshot_manager.create_snapshot(name='source-catalog', is_historical=True)
    assert created.get('success') is True
    empty_created = snapshot_manager.create_snapshot(name='empty-source', is_historical=True)
    assert empty_created.get('success') is True

    source_db_path = snapshot_manager.get_snapshot_db_path(created['snapshot_file'])
    color, shape, texture = _make_feature_vectors()

    with sqlite3.connect(source_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''
            INSERT INTO products (image_path, category, product_name, sku, is_historical, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            ('/tmp/source-product.jpg', 'textiles', 'Source Product', 'SRC-1', 1, '{}')
        )
        product_id = cursor.lastrowid
        cursor.execute(
            '''
            INSERT INTO features
            (product_id, color_features, shape_features, texture_features, embedding_type, embedding_version)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (
                product_id,
                _serialize_array(color),
                _serialize_array(shape),
                _serialize_array(texture),
                'legacy',
                None
            )
        )

        price_rows = [
            (product_id, f'2024-01-{(i % 28) + 1:02d}', float(i), 'USD')
            for i in range(1005)
        ]
        perf_rows = [
            (product_id, f'2024-02-{(i % 28) + 1:02d}', i, i * 2, 1.5, float(i * 3))
            for i in range(1005)
        ]

        cursor.executemany(
            'INSERT INTO price_history (product_id, date, price, currency) VALUES (?, ?, ?, ?)',
            price_rows
        )
        cursor.executemany(
            '''
            INSERT INTO performance_history
            (product_id, date, sales, views, conversion_rate, revenue)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            perf_rows
        )
        conn.commit()

    merged = snapshot_manager.merge_snapshots(
        [created['snapshot_file'], empty_created['snapshot_file']],
        new_name='merged-catalog',
        is_historical=True
    )
    assert merged.get('success') is True

    merged_db_path = snapshot_manager.get_snapshot_db_path(merged['snapshot_file'])
    with sqlite3.connect(merged_db_path) as conn:
        product_count = conn.execute('SELECT COUNT(*) FROM products').fetchone()[0]
        feature_count = conn.execute('SELECT COUNT(*) FROM features').fetchone()[0]
        price_count = conn.execute('SELECT COUNT(*) FROM price_history').fetchone()[0]
        perf_count = conn.execute('SELECT COUNT(*) FROM performance_history').fetchone()[0]

    assert product_count == 1
    assert feature_count == 1
    assert price_count == 1005
    assert perf_count == 1005
