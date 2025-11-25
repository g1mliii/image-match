"""
Test CSV Builder Linking System

This test validates the flexible metadata linking system in the CSV Builder,
including various linking strategies, real-world data handling, and edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import re
from datetime import date, timedelta


# ===== Linking Strategy Functions (Python equivalents of JS) =====

def sanitize_field(field):
    """Sanitize a field value - handles null, empty, and common null representations"""
    if field is None:
        return ''
    
    sanitized = str(field).strip()
    
    # Remove surrounding quotes
    if (sanitized.startswith('"') and sanitized.endswith('"')) or \
       (sanitized.startswith("'") and sanitized.endswith("'")):
        sanitized = sanitized[1:-1]
    
    # Handle common null representations
    null_values = ['null', 'NULL', 'undefined', 'UNDEFINED', 'N/A', 'n/a', 'NA', 'na', '-', '']
    if sanitized in null_values:
        return ''
    
    return sanitized


def normalize_header_name(header):
    """Normalize header names to standard format"""
    header_map = {
        'product_name': 'name',
        'productname': 'name',
        'product name': 'name',
        'item_name': 'name',
        'itemname': 'name',
        'product_sku': 'sku',
        'productsku': 'sku',
        'item_sku': 'sku',
        'itemsku': 'sku',
        'product_id': 'sku',
        'productid': 'sku',
        'item_id': 'sku',
        'itemid': 'sku',
        'file_name': 'filename',
        'file': 'filename',
        'image': 'filename',
        'image_name': 'filename',
        'imagename': 'filename',
        'cat': 'category',
        'product_category': 'category',
        'productcategory': 'category',
        'unit_price': 'price',
        'unitprice': 'price',
        'cost': 'price',
        'amount': 'price'
    }
    
    normalized = re.sub(r'[^a-z0-9]', '', header.lower())
    return header_map.get(normalized, re.sub(r'[^a-z0-9_]', '', header.lower()))


def parse_price(price_str):
    """Parse and validate price string"""
    if not price_str or not isinstance(price_str, str):
        return None
    
    # Remove currency symbols and whitespace
    cleaned = re.sub(r'[$€£¥₹,\s]', '', price_str).strip()
    
    # Handle negative prices (invalid)
    if cleaned.startswith('-'):
        return None
    
    try:
        price = float(cleaned)
        if price < 0 or not (price == price):  # Check for NaN
            return None
        return round(price * 100) / 100
    except (ValueError, TypeError):
        return None


def normalize_for_fuzzy_match(s):
    """Normalize string for fuzzy matching"""
    if not s or not isinstance(s, str):
        return ''
    
    return re.sub(r'\s+', ' ', 
                  re.sub(r'[^a-z0-9\s]', '', 
                         re.sub(r'[_\-\.]', ' ', s.lower()))).strip()


def calculate_fuzzy_score(str1, str2):
    """Calculate fuzzy match score between two strings"""
    if not str1 or not str2:
        return 0
    
    # Exact match
    if str1 == str2:
        return 1.0
    
    # Contains match
    if str1 in str2 or str2 in str1:
        shorter = str1 if len(str1) < len(str2) else str2
        longer = str2 if len(str1) < len(str2) else str1
        return len(shorter) / len(longer)
    
    # Word overlap
    words1 = [w for w in str1.split() if len(w) > 1]
    words2 = [w for w in str2.split() if len(w) > 1]
    
    if not words1 or not words2:
        return 0
    
    matching_words = 0
    for w1 in words1:
        for w2 in words2:
            if w1 == w2 or w1 in w2 or w2 in w1:
                matching_words += 1
                break
    
    return matching_words / max(len(words1), len(words2))


def link_by_filename_equals_sku(product, imported_data):
    """Link by filename = SKU strategy"""
    if not product or not product.get('filename'):
        return None
    
    # Remove extension
    filename_sku = re.sub(r'\.[^.]+$', '', product['filename']).strip()
    if not filename_sku:
        return None
    
    for data in imported_data:
        if not data or not data.get('sku'):
            continue
        if str(data['sku']).strip().lower() == filename_sku.lower():
            return data
    
    return None


def link_by_filename_contains_sku(product, imported_data, pattern=r'[A-Z]+-\d+'):
    """Link by filename contains SKU strategy"""
    if not product or not product.get('filename'):
        return None
    
    try:
        match = re.search(pattern, product['filename'], re.IGNORECASE)
        if match:
            extracted_sku = match.group(0)
            for data in imported_data:
                if not data or not data.get('sku'):
                    continue
                if str(data['sku']).strip().lower() == extracted_sku.lower():
                    return data
    except re.error:
        pass
    
    return None


def link_by_folder_equals_sku(product, imported_data):
    """Link by folder name = SKU strategy"""
    if not product or not product.get('category'):
        return None
    
    folder_sku = str(product['category']).strip()
    if not folder_sku:
        return None
    
    for data in imported_data:
        if not data or not data.get('sku'):
            continue
        if str(data['sku']).strip().lower() == folder_sku.lower():
            return data
    
    return None


def link_by_fuzzy_name(product, imported_data):
    """Link by fuzzy name matching strategy"""
    if not product or not product.get('filename'):
        return None
    
    # Clean filename
    clean_filename = normalize_for_fuzzy_match(
        re.sub(r'\.[^.]+$', '', product['filename'])
    )
    
    if not clean_filename or len(clean_filename) < 2:
        return None
    
    best_match = None
    best_score = 0
    
    for data in imported_data:
        if not data or not data.get('name'):
            continue
        
        clean_name = normalize_for_fuzzy_match(data['name'])
        if not clean_name:
            continue
        
        score = calculate_fuzzy_score(clean_filename, clean_name)
        
        if score > best_score and score >= 0.5:
            best_score = score
            best_match = data
    
    return best_match


# ===== Test Cases =====

def test_sanitize_field():
    """Test field sanitization"""
    # Normal values
    assert sanitize_field('hello') == 'hello'
    assert sanitize_field('  hello  ') == 'hello'
    
    # Quoted values
    assert sanitize_field('"hello"') == 'hello'
    assert sanitize_field("'hello'") == 'hello'
    
    # Null representations
    assert sanitize_field('null') == ''
    assert sanitize_field('NULL') == ''
    assert sanitize_field('N/A') == ''
    assert sanitize_field('undefined') == ''
    assert sanitize_field('-') == ''
    assert sanitize_field('') == ''
    assert sanitize_field(None) == ''
    
    print("✓ sanitize_field test passed")


def test_normalize_header_name():
    """Test header name normalization"""
    assert normalize_header_name('product_name') == 'name'
    assert normalize_header_name('productname') == 'name'
    assert normalize_header_name('product_sku') == 'sku'
    assert normalize_header_name('filename') == 'filename'  # Direct match
    assert normalize_header_name('unit_price') == 'price'
    assert normalize_header_name('unitprice') == 'price'
    assert normalize_header_name('category') == 'category'
    # Custom fields keep their normalized form (underscores preserved in fallback)
    result = normalize_header_name('custom_field')
    assert 'custom' in result
    
    print("✓ normalize_header_name test passed")


def test_parse_price():
    """Test price parsing"""
    # Valid prices
    assert parse_price('29.99') == 29.99
    assert parse_price('$29.99') == 29.99
    assert parse_price('€29.99') == 29.99
    assert parse_price('1,234.56') == 1234.56
    assert parse_price('  45.00  ') == 45.00
    assert parse_price('0') == 0.0
    
    # Invalid prices
    assert parse_price('-29.99') is None
    assert parse_price('abc') is None
    assert parse_price('') is None
    assert parse_price(None) is None
    
    print("✓ parse_price test passed")


def test_link_by_filename_equals_sku():
    """Test filename = SKU linking strategy"""
    imported_data = [
        {'sku': 'PM-001', 'name': 'Blue Placemat', 'price': '29.99'},
        {'sku': 'DW-002', 'name': 'White Plate', 'price': '45.00'},
        {'sku': 'TX-003', 'name': 'Cotton Napkins', 'price': '15.99'}
    ]
    
    # Exact match
    product = {'filename': 'PM-001.jpg', 'category': 'placemats'}
    result = link_by_filename_equals_sku(product, imported_data)
    assert result is not None
    assert result['sku'] == 'PM-001'
    
    # Case insensitive
    product = {'filename': 'pm-001.jpg', 'category': 'placemats'}
    result = link_by_filename_equals_sku(product, imported_data)
    assert result is not None
    assert result['sku'] == 'PM-001'
    
    # No match
    product = {'filename': 'unknown.jpg', 'category': 'placemats'}
    result = link_by_filename_equals_sku(product, imported_data)
    assert result is None
    
    # Empty filename
    product = {'filename': '', 'category': 'placemats'}
    result = link_by_filename_equals_sku(product, imported_data)
    assert result is None
    
    # Null product
    result = link_by_filename_equals_sku(None, imported_data)
    assert result is None
    
    print("✓ link_by_filename_equals_sku test passed")


def test_link_by_filename_contains_sku():
    """Test filename contains SKU linking strategy"""
    imported_data = [
        {'sku': 'PM-001', 'name': 'Blue Placemat', 'price': '29.99'},
        {'sku': 'DW-002', 'name': 'White Plate', 'price': '45.00'},
        {'sku': '12345', 'name': 'Numeric SKU Product', 'price': '19.99'}
    ]
    
    # SKU in middle of filename
    product = {'filename': 'photo_PM-001_front.jpg', 'category': 'placemats'}
    result = link_by_filename_contains_sku(product, imported_data)
    assert result is not None
    assert result['sku'] == 'PM-001'
    
    # SKU at start
    product = {'filename': 'DW-002_main.jpg', 'category': 'dinnerware'}
    result = link_by_filename_contains_sku(product, imported_data)
    assert result is not None
    assert result['sku'] == 'DW-002'
    
    # Numeric pattern
    product = {'filename': 'product_12345_v2.jpg', 'category': 'misc'}
    result = link_by_filename_contains_sku(product, imported_data, r'\d{4,}')
    assert result is not None
    assert result['sku'] == '12345'
    
    # No match
    product = {'filename': 'random_image.jpg', 'category': 'misc'}
    result = link_by_filename_contains_sku(product, imported_data)
    assert result is None
    
    print("✓ link_by_filename_contains_sku test passed")


def test_link_by_folder_equals_sku():
    """Test folder name = SKU linking strategy"""
    imported_data = [
        {'sku': 'PM-001', 'name': 'Blue Placemat', 'price': '29.99'},
        {'sku': 'DW-002', 'name': 'White Plate', 'price': '45.00'}
    ]
    
    # Folder matches SKU
    product = {'filename': 'image.jpg', 'category': 'PM-001'}
    result = link_by_folder_equals_sku(product, imported_data)
    assert result is not None
    assert result['sku'] == 'PM-001'
    
    # Case insensitive
    product = {'filename': 'image.jpg', 'category': 'dw-002'}
    result = link_by_folder_equals_sku(product, imported_data)
    assert result is not None
    assert result['sku'] == 'DW-002'
    
    # No category
    product = {'filename': 'image.jpg', 'category': ''}
    result = link_by_folder_equals_sku(product, imported_data)
    assert result is None
    
    print("✓ link_by_folder_equals_sku test passed")


def test_link_by_fuzzy_name():
    """Test fuzzy name matching strategy"""
    imported_data = [
        {'sku': 'PM-001', 'name': 'Blue Ceramic Placemat', 'price': '29.99'},
        {'sku': 'DW-002', 'name': 'White Dinner Plate Set', 'price': '45.00'},
        {'sku': 'TX-003', 'name': 'Cotton Table Napkins', 'price': '15.99'}
    ]
    
    # Partial match
    product = {'filename': 'blue-placemat.jpg', 'category': 'placemats'}
    result = link_by_fuzzy_name(product, imported_data)
    assert result is not None
    assert result['sku'] == 'PM-001'
    
    # Word overlap
    product = {'filename': 'dinner_plate.jpg', 'category': 'dinnerware'}
    result = link_by_fuzzy_name(product, imported_data)
    assert result is not None
    assert result['sku'] == 'DW-002'
    
    # Underscore to space conversion
    product = {'filename': 'cotton_napkins.jpg', 'category': 'textiles'}
    result = link_by_fuzzy_name(product, imported_data)
    assert result is not None
    assert result['sku'] == 'TX-003'
    
    # No match (too different)
    product = {'filename': 'xyz123.jpg', 'category': 'misc'}
    result = link_by_fuzzy_name(product, imported_data)
    assert result is None
    
    print("✓ link_by_fuzzy_name test passed")


def test_fuzzy_score_calculation():
    """Test fuzzy score calculation"""
    # Exact match
    assert calculate_fuzzy_score('hello', 'hello') == 1.0
    
    # Contains match
    score = calculate_fuzzy_score('blue', 'blue placemat')
    assert score > 0 and score < 1
    
    # Word overlap
    score = calculate_fuzzy_score('blue ceramic plate', 'ceramic dinner plate')
    assert score > 0.3  # Should have some overlap
    
    # No match
    assert calculate_fuzzy_score('abc', 'xyz') == 0
    
    # Empty strings
    assert calculate_fuzzy_score('', 'hello') == 0
    assert calculate_fuzzy_score('hello', '') == 0
    
    print("✓ fuzzy_score_calculation test passed")


def test_real_world_data_handling():
    """Test handling of messy real-world data"""
    imported_data = [
        {'sku': 'PM-001', 'name': 'Blue Placemat', 'price': '29.99'},
        {'sku': '', 'name': 'No SKU Product', 'price': '19.99'},  # Missing SKU
        {'sku': None, 'name': 'Null SKU Product', 'price': '9.99'},  # Null SKU
        {'sku': 'DW-002', 'name': '', 'price': ''},  # Missing name and price
        {'sku': 'TX-003', 'name': 'Valid Product', 'price': 'invalid'},  # Invalid price
    ]
    
    # Should find valid SKU
    product = {'filename': 'PM-001.jpg', 'category': 'placemats'}
    result = link_by_filename_equals_sku(product, imported_data)
    assert result is not None
    assert result['sku'] == 'PM-001'
    
    # Should handle missing SKUs gracefully
    product = {'filename': 'no-sku.jpg', 'category': 'misc'}
    result = link_by_filename_equals_sku(product, imported_data)
    assert result is None  # Should not crash
    
    # Should handle empty imported data
    result = link_by_filename_equals_sku(product, [])
    assert result is None
    
    # Should handle None in imported data
    result = link_by_filename_equals_sku(product, [None, {'sku': 'PM-001'}])
    assert result is None  # No match for 'no-sku'
    
    print("✓ real_world_data_handling test passed")


def test_large_dataset_performance():
    """Test performance with large datasets"""
    import time
    
    # Generate large dataset
    imported_data = [
        {'sku': f'SKU-{i:05d}', 'name': f'Product {i}', 'price': f'{i * 0.99:.2f}'}
        for i in range(10000)
    ]
    
    products = [
        {'filename': f'SKU-{i:05d}.jpg', 'category': 'test'}
        for i in range(1000)
    ]
    
    # Test filename = SKU strategy
    start = time.time()
    matched = 0
    for product in products:
        result = link_by_filename_equals_sku(product, imported_data)
        if result:
            matched += 1
    elapsed = time.time() - start
    
    assert matched == 1000, f"Expected 1000 matches, got {matched}"
    assert elapsed < 5.0, f"Performance too slow: {elapsed:.2f}s for 1000 products"
    
    print(f"✓ large_dataset_performance test passed ({elapsed:.2f}s for 1000 products)")


def test_special_characters_handling():
    """Test handling of special characters in data"""
    imported_data = [
        {'sku': 'PM-001', 'name': 'Blue "Premium" Placemat', 'price': '29.99'},
        {'sku': 'DW-002', 'name': "White Plate's Set", 'price': '45.00'},
        {'sku': 'TX-003', 'name': 'Cotton, Linen & Silk', 'price': '15.99'},
        {'sku': 'SP-004', 'name': 'Product (Special)', 'price': '19.99'},
    ]
    
    # Should match despite special chars in name
    product = {'filename': 'PM-001.jpg', 'category': 'placemats'}
    result = link_by_filename_equals_sku(product, imported_data)
    assert result is not None
    assert 'Premium' in result['name']
    
    # Fuzzy match with special chars
    product = {'filename': 'cotton-linen-silk.jpg', 'category': 'textiles'}
    result = link_by_fuzzy_name(product, imported_data)
    assert result is not None
    assert result['sku'] == 'TX-003'
    
    print("✓ special_characters_handling test passed")


def test_unicode_handling():
    """Test handling of unicode characters"""
    imported_data = [
        {'sku': 'PM-001', 'name': 'Café Placemat', 'price': '29.99'},
        {'sku': 'DW-002', 'name': '日本語製品', 'price': '45.00'},
        {'sku': 'TX-003', 'name': 'Naïve Design', 'price': '15.99'},
    ]
    
    # Should handle unicode in SKU matching
    product = {'filename': 'PM-001.jpg', 'category': 'placemats'}
    result = link_by_filename_equals_sku(product, imported_data)
    assert result is not None
    assert 'Café' in result['name']
    
    print("✓ unicode_handling test passed")


def test_duplicate_sku_handling():
    """Test handling of duplicate SKUs in imported data"""
    imported_data = [
        {'sku': 'PM-001', 'name': 'First Product', 'price': '29.99'},
        {'sku': 'PM-001', 'name': 'Duplicate SKU', 'price': '19.99'},  # Duplicate
        {'sku': 'DW-002', 'name': 'Unique Product', 'price': '45.00'},
    ]
    
    # Should return first match
    product = {'filename': 'PM-001.jpg', 'category': 'placemats'}
    result = link_by_filename_equals_sku(product, imported_data)
    assert result is not None
    assert result['name'] == 'First Product'  # First match wins
    
    print("✓ duplicate_sku_handling test passed")


def run_all_tests():
    """Run all linking tests"""
    print("\n" + "="*60)
    print("CSV Builder Linking System Tests")
    print("="*60 + "\n")
    
    tests = [
        test_sanitize_field,
        test_normalize_header_name,
        test_parse_price,
        test_link_by_filename_equals_sku,
        test_link_by_filename_contains_sku,
        test_link_by_folder_equals_sku,
        test_link_by_fuzzy_name,
        test_fuzzy_score_calculation,
        test_real_world_data_handling,
        test_large_dataset_performance,
        test_special_characters_handling,
        test_unicode_handling,
        test_duplicate_sku_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("🎉 All linking tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
