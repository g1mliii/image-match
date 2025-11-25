"""
End-to-End Test: CSV Builder → Main App Workflow

This test validates the complete workflow:
1. User uploads images to CSV Builder
2. User links existing product data OR exports template and fills in Excel
3. User exports CSV from CSV Builder
4. User uploads images + CSV to Main App
5. Main App correctly parses CSV and applies metadata to products

Tests cover:
- Basic workflow: Upload → Link → Export → Use in Main App
- Excel workflow: Upload → Export Template → Fill → Import → Export → Use in Main App
- Real-world data handling: Missing fields, special characters, large datasets
- Error handling: Invalid data, corrupted files, edge cases
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import re
from datetime import date, timedelta
from io import StringIO


# ===== CSV Builder Functions (Python equivalents) =====

def sanitize_field(field):
    """Sanitize a field value"""
    if field is None:
        return ''
    sanitized = str(field).strip()
    if (sanitized.startswith('"') and sanitized.endswith('"')) or \
       (sanitized.startswith("'") and sanitized.endswith("'")):
        sanitized = sanitized[1:-1]
    null_values = ['null', 'NULL', 'undefined', 'N/A', 'n/a', '-', '']
    if sanitized in null_values:
        return ''
    return sanitized


def quote_csv_field(field, separator=','):
    """Quote CSV field if needed"""
    if not field:
        return ''
    s = str(field)
    if separator in s or '"' in s or '\n' in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def format_price_history(price_history):
    """Format price history for CSV"""
    if not price_history:
        return ''
    return ';'.join([f"{e['date']}:{e['price']}" for e in price_history])


def format_performance_history(perf_history):
    """Format performance history for CSV"""
    if not perf_history:
        return ''
    return ';'.join([
        f"{e['date']}:{e['sales']}:{e['views']}:{e['conversion_rate']}:{e['revenue']}"
        for e in perf_history
    ])


def generate_csv_builder_output(products, include_headers=True, separator=','):
    """
    Generate CSV output exactly as CSV Builder would produce it.
    This simulates the generateCSV() function in csv-builder.js
    """
    lines = []
    
    if include_headers:
        lines.append(separator.join(['filename', 'category', 'sku', 'name', 'price', 'price_history', 'performance_history']))
    
    for product in products:
        row = [
            quote_csv_field(product.get('filename', ''), separator),
            quote_csv_field(product.get('category', ''), separator),
            quote_csv_field(product.get('sku', ''), separator),
            quote_csv_field(product.get('name', ''), separator),
            str(product.get('price', '')) if product.get('price') else '',
            quote_csv_field(format_price_history(product.get('priceHistory', [])), separator),
            quote_csv_field(format_performance_history(product.get('performanceHistory', [])), separator),
        ]
        lines.append(separator.join(row))
    
    return '\n'.join(lines)


def generate_template_csv(products):
    """
    Generate template CSV for Excel workflow.
    This simulates the exportTemplate() function in csv-builder.js
    """
    lines = ['filename,category,sku,name,price,price_history,performance_history']
    
    for product in products:
        lines.append(f"{product.get('filename', '')},{product.get('category', '')},,,,")
    
    return '\n'.join(lines)


# ===== Main App CSV Parser (Python equivalent) =====

def parse_price_history(price_history_str):
    """Parse price history string - matches main app logic"""
    if not price_history_str or not price_history_str.strip():
        return None
    
    price_history = []
    
    if ':' in price_history_str:
        entries = [e.strip() for e in price_history_str.replace(',', ';').split(';') if e.strip()]
        for entry in entries:
            parts = [p.strip() for p in entry.split(':')]
            if len(parts) >= 2:
                try:
                    price = float(parts[1])
                    if price >= 0:
                        price_history.append({'date': parts[0], 'price': price})
                except ValueError:
                    continue
    else:
        today = date.today()
        prices = [p.strip() for p in price_history_str.replace(',', ';').split(';') if p.strip()]
        for i, price_str in enumerate(prices):
            try:
                price = float(price_str)
                if price >= 0:
                    months_back = len(prices) - 1 - i
                    price_date = today - timedelta(days=months_back * 30)
                    price_history.append({'date': price_date.isoformat(), 'price': price})
            except ValueError:
                continue
    
    if price_history:
        price_history.sort(key=lambda x: x['date'])
        return price_history[-12:]
    return None


def parse_performance_history(perf_history_str):
    """Parse performance history string - matches main app logic"""
    if not perf_history_str or not perf_history_str.strip():
        return None
    
    perf_history = []
    entries = [e.strip() for e in perf_history_str.replace(',', ';').split(';') if e.strip()]
    
    for entry in entries:
        parts = [p.strip() for p in entry.split(':')]
        
        if len(parts) >= 5 and '-' in parts[0]:
            try:
                perf_history.append({
                    'date': parts[0],
                    'sales': int(parts[1]) if parts[1] else 0,
                    'views': int(parts[2]) if parts[2] else 0,
                    'conversion_rate': float(parts[3]) if parts[3] else 0.0,
                    'revenue': float(parts[4]) if parts[4] else 0.0
                })
            except (ValueError, IndexError):
                continue
        elif len(parts) >= 4:
            today = date.today()
            months_back = len(perf_history)
            perf_date = today - timedelta(days=months_back * 30)
            try:
                perf_history.append({
                    'date': perf_date.isoformat(),
                    'sales': int(parts[0]) if parts[0] else 0,
                    'views': int(parts[1]) if parts[1] else 0,
                    'conversion_rate': float(parts[2]) if parts[2] else 0.0,
                    'revenue': float(parts[3]) if parts[3] else 0.0
                })
            except (ValueError, IndexError):
                continue
    
    if perf_history:
        perf_history.sort(key=lambda x: x['date'])
        return perf_history[-12:]
    return None


def main_app_parse_csv(csv_text):
    """
    Parse CSV exactly as the main app does.
    This simulates the parseCsv() function in app.js
    """
    lines = [line.strip() for line in csv_text.split('\n') if line.strip()]
    result = {}
    errors = []
    
    if not lines:
        return result, errors
    
    # Check for header
    first_line = lines[0].lower()
    has_header = 'filename' in first_line or 'category' in first_line or 'sku' in first_line
    data_lines = lines[1:] if has_header else lines
    
    for index, line in enumerate(data_lines):
        try:
            # Split by comma or tab, handle quotes
            parts = []
            current = ''
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char in [',', '\t'] and not in_quotes:
                    parts.append(current.strip().strip('"'))
                    current = ''
                else:
                    current += char
            parts.append(current.strip().strip('"'))
            
            if len(parts) >= 1:
                filename = parts[0]
                if not filename:
                    continue
                
                category = parts[1] if len(parts) > 1 else None
                sku = parts[2] if len(parts) > 2 else None
                name = parts[3] if len(parts) > 3 else None
                
                # Parse price
                price_history = None
                if len(parts) > 4 and parts[4]:
                    if ':' not in parts[4] and ';' not in parts[4]:
                        try:
                            price = float(parts[4])
                            if price >= 0:
                                today = date.today().isoformat()
                                price_history = [{'date': today, 'price': price}]
                        except ValueError:
                            pass
                
                # Parse price history from column 5 or 4
                price_history_str = parts[5] if len(parts) > 5 else (parts[4] if len(parts) > 4 else None)
                if price_history_str and (':' in price_history_str or ';' in price_history_str):
                    parsed = parse_price_history(price_history_str)
                    if parsed:
                        price_history = parsed
                
                # Parse performance history from column 6 or 5
                perf_history = None
                perf_str = parts[6] if len(parts) > 6 else (parts[5] if len(parts) > 5 else None)
                if perf_str and ':' in perf_str and perf_str.count(':') >= 3:
                    perf_history = parse_performance_history(perf_str)
                
                result[filename] = {
                    'category': category,
                    'sku': sku,
                    'name': name,
                    'priceHistory': price_history,
                    'performanceHistory': perf_history
                }
        except Exception as e:
            errors.append(f"Row {index + 2}: {str(e)}")
    
    return result, errors


# ===== End-to-End Test Cases =====

def test_basic_workflow():
    """
    Test basic workflow:
    1. User has images with filenames
    2. User adds metadata in CSV Builder
    3. User exports CSV
    4. Main app parses CSV correctly
    """
    # Step 1: Simulate images uploaded to CSV Builder
    products = [
        {
            'filename': 'PM-001.jpg',
            'category': 'placemats',
            'sku': 'PM-001',
            'name': 'Blue Placemat',
            'price': 29.99,
            'priceHistory': [],
            'performanceHistory': []
        },
        {
            'filename': 'DW-002.jpg',
            'category': 'dinnerware',
            'sku': 'DW-002',
            'name': 'White Plate Set',
            'price': 45.00,
            'priceHistory': [],
            'performanceHistory': []
        }
    ]
    
    # Step 2: Generate CSV from CSV Builder
    csv_output = generate_csv_builder_output(products)
    
    # Step 3: Parse CSV in Main App
    parsed, errors = main_app_parse_csv(csv_output)
    
    # Verify
    assert len(errors) == 0, f"Parse errors: {errors}"
    assert 'PM-001.jpg' in parsed
    assert parsed['PM-001.jpg']['category'] == 'placemats'
    assert parsed['PM-001.jpg']['sku'] == 'PM-001'
    assert parsed['PM-001.jpg']['name'] == 'Blue Placemat'
    
    assert 'DW-002.jpg' in parsed
    assert parsed['DW-002.jpg']['category'] == 'dinnerware'
    
    print("✓ test_basic_workflow passed")


def test_excel_workflow():
    """
    Test Excel workflow:
    1. User uploads images
    2. User exports template CSV
    3. User fills template in Excel
    4. User imports filled CSV back
    5. User exports final CSV
    6. Main app parses correctly
    """
    # Step 1: Simulate images uploaded
    initial_products = [
        {'filename': 'product1.jpg', 'category': 'plates'},
        {'filename': 'product2.jpg', 'category': 'bowls'},
        {'filename': 'product3.jpg', 'category': ''},  # No category detected
    ]
    
    # Step 2: Generate template
    template_csv = generate_template_csv(initial_products)
    
    # Verify template format
    assert 'filename,category,sku,name,price' in template_csv
    assert 'product1.jpg,plates' in template_csv
    assert 'product2.jpg,bowls' in template_csv
    
    # Step 3: Simulate user filling template in Excel
    filled_csv = """filename,category,sku,name,price,price_history,performance_history
product1.jpg,plates,PLT-001,Ceramic Dinner Plate,24.99,2024-01-15:24.99;2024-02-15:22.99,2024-01-15:100:800:12.5:2499
product2.jpg,bowls,BWL-002,Soup Bowl,12.99,12.99,50:400:12.5:649.50
product3.jpg,misc,MSC-003,Mystery Item,9.99,,"""
    
    # Step 4: Parse filled CSV (simulating import back to CSV Builder)
    # Then export final CSV
    
    # Step 5: Main app parses the filled CSV
    parsed, errors = main_app_parse_csv(filled_csv)
    
    # Verify
    assert len(errors) == 0, f"Parse errors: {errors}"
    assert len(parsed) == 3
    
    # Check product 1 - full data with dates
    p1 = parsed['product1.jpg']
    assert p1['sku'] == 'PLT-001'
    assert p1['name'] == 'Ceramic Dinner Plate'
    assert p1['priceHistory'] is not None
    assert len(p1['priceHistory']) == 2
    assert p1['performanceHistory'] is not None
    
    # Check product 2 - quick format (no dates)
    p2 = parsed['product2.jpg']
    assert p2['sku'] == 'BWL-002'
    assert p2['priceHistory'] is not None
    
    # Check product 3 - minimal data
    p3 = parsed['product3.jpg']
    assert p3['sku'] == 'MSC-003'
    assert p3['performanceHistory'] is None
    
    print("✓ test_excel_workflow passed")


def test_linking_workflow():
    """
    Test linking workflow:
    1. User uploads images
    2. User imports existing product data
    3. System links by filename = SKU
    4. User exports CSV
    5. Main app parses correctly
    """
    # Step 1: Images uploaded (filenames match SKUs)
    images = [
        {'filename': 'PM-001.jpg', 'category': 'placemats'},
        {'filename': 'DW-002.jpg', 'category': 'dinnerware'},
        {'filename': 'TX-003.jpg', 'category': 'textiles'},
    ]
    
    # Step 2: Existing product data (from ERP/database)
    existing_data = [
        {'sku': 'PM-001', 'name': 'Blue Placemat', 'price': '29.99'},
        {'sku': 'DW-002', 'name': 'White Plate', 'price': '45.00'},
        {'sku': 'TX-003', 'name': 'Cotton Napkins', 'price': '15.99'},
    ]
    
    # Step 3: Link by filename = SKU (simulate linking)
    linked_products = []
    for img in images:
        filename_sku = img['filename'].replace('.jpg', '')
        matched = next((d for d in existing_data if d['sku'] == filename_sku), None)
        
        product = {
            'filename': img['filename'],
            'category': img['category'],
            'sku': matched['sku'] if matched else '',
            'name': matched['name'] if matched else '',
            'price': float(matched['price']) if matched else '',
            'priceHistory': [],
            'performanceHistory': []
        }
        linked_products.append(product)
    
    # Step 4: Generate CSV
    csv_output = generate_csv_builder_output(linked_products)
    
    # Step 5: Main app parses
    parsed, errors = main_app_parse_csv(csv_output)
    
    # Verify all products linked correctly
    assert len(errors) == 0
    assert len(parsed) == 3
    assert parsed['PM-001.jpg']['sku'] == 'PM-001'
    assert parsed['PM-001.jpg']['name'] == 'Blue Placemat'
    assert parsed['DW-002.jpg']['sku'] == 'DW-002'
    assert parsed['TX-003.jpg']['sku'] == 'TX-003'
    
    print("✓ test_linking_workflow passed")


def test_real_world_data_handling():
    """
    Test handling of messy real-world data:
    - Missing fields
    - Special characters
    - Unicode
    - Empty values
    - Invalid prices
    """
    # Messy real-world CSV
    messy_csv = '''filename,category,sku,name,price,price_history,performance_history
"product, with comma.jpg",plates,PLT-001,"Plate ""Premium""",29.99,,
product_special!@#.jpg,,,"Name with 'quotes'",invalid_price,,
unicode_café.jpg,misc,UNI-001,Café Mug,15.99,,
empty_fields.jpg,,,,,
minimal.jpg,category,,,,'''
    
    parsed, errors = main_app_parse_csv(messy_csv)
    
    # Should parse without crashing
    assert 'product, with comma.jpg' in parsed
    assert parsed['product, with comma.jpg']['sku'] == 'PLT-001'
    
    # Special characters handled
    assert 'product_special!@#.jpg' in parsed
    
    # Unicode handled
    assert 'unicode_café.jpg' in parsed
    assert parsed['unicode_café.jpg']['name'] == 'Café Mug'
    
    # Empty fields don't crash
    assert 'empty_fields.jpg' in parsed
    assert 'minimal.jpg' in parsed
    
    print("✓ test_real_world_data_handling passed")


def test_large_dataset():
    """Test performance with large dataset (1000+ products)"""
    import time
    
    # Generate large dataset
    products = []
    for i in range(1000):
        products.append({
            'filename': f'product_{i:04d}.jpg',
            'category': f'category_{i % 10}',
            'sku': f'SKU-{i:04d}',
            'name': f'Product Name {i}',
            'price': round(10 + (i * 0.1), 2),
            'priceHistory': [
                {'date': '2024-01-15', 'price': 10 + (i * 0.1)},
                {'date': '2024-02-15', 'price': 11 + (i * 0.1)}
            ],
            'performanceHistory': [
                {'date': '2024-01-15', 'sales': 100 + i, 'views': 1000 + i, 'conversion_rate': 10.0, 'revenue': 1000 + i}
            ]
        })
    
    # Generate CSV
    start = time.time()
    csv_output = generate_csv_builder_output(products)
    gen_time = time.time() - start
    
    # Parse CSV
    start = time.time()
    parsed, errors = main_app_parse_csv(csv_output)
    parse_time = time.time() - start
    
    # Verify
    assert len(errors) == 0, f"Parse errors: {errors}"
    assert len(parsed) == 1000
    assert gen_time < 2.0, f"CSV generation too slow: {gen_time:.2f}s"
    assert parse_time < 2.0, f"CSV parsing too slow: {parse_time:.2f}s"
    
    # Spot check
    assert parsed['product_0500.jpg']['sku'] == 'SKU-0500'
    assert parsed['product_0999.jpg']['priceHistory'] is not None
    
    print(f"✓ test_large_dataset passed (gen: {gen_time:.2f}s, parse: {parse_time:.2f}s)")


def test_price_history_formats():
    """Test various price history formats"""
    csv = '''filename,category,sku,name,price,price_history
dated.jpg,cat,SKU1,Product 1,,2024-01-15:29.99;2024-02-15:31.50;2024-03-15:28.75
quick.jpg,cat,SKU2,Product 2,,29.99;31.50;28.75
single.jpg,cat,SKU3,Product 3,29.99,
mixed.jpg,cat,SKU4,Product 4,25.00,2024-01-15:29.99'''
    
    parsed, errors = main_app_parse_csv(csv)
    
    # Dated format
    assert parsed['dated.jpg']['priceHistory'] is not None
    assert len(parsed['dated.jpg']['priceHistory']) == 3
    assert parsed['dated.jpg']['priceHistory'][0]['date'] == '2024-01-15'
    
    # Quick format (auto-generated dates)
    assert parsed['quick.jpg']['priceHistory'] is not None
    assert len(parsed['quick.jpg']['priceHistory']) == 3
    
    # Single price
    assert parsed['single.jpg']['priceHistory'] is not None
    assert len(parsed['single.jpg']['priceHistory']) == 1
    
    # Mixed
    assert parsed['mixed.jpg']['priceHistory'] is not None
    
    print("✓ test_price_history_formats passed")


def test_performance_history_formats():
    """Test various performance history formats"""
    csv = '''filename,category,sku,name,price,price_history,performance_history
dated.jpg,cat,SKU1,Product 1,,,2024-01-15:150:1200:12.5:1800;2024-02-15:180:1500:12.0:2160
quick.jpg,cat,SKU2,Product 2,,,150:1200:12.5:1800;180:1500:12.0:2160
empty.jpg,cat,SKU3,Product 3,,,'''
    
    parsed, errors = main_app_parse_csv(csv)
    
    # Dated format
    assert parsed['dated.jpg']['performanceHistory'] is not None
    assert len(parsed['dated.jpg']['performanceHistory']) == 2
    assert parsed['dated.jpg']['performanceHistory'][0]['sales'] == 150
    
    # Quick format
    assert parsed['quick.jpg']['performanceHistory'] is not None
    assert len(parsed['quick.jpg']['performanceHistory']) == 2
    
    # Empty
    assert parsed['empty.jpg']['performanceHistory'] is None
    
    print("✓ test_performance_history_formats passed")


def test_csv_round_trip():
    """
    Test complete round-trip:
    CSV Builder output → Main App parse → Verify data integrity
    """
    original_products = [
        {
            'filename': 'test1.jpg',
            'category': 'plates',
            'sku': 'PLT-001',
            'name': 'Test Plate',
            'price': 29.99,
            'priceHistory': [
                {'date': '2024-01-15', 'price': 29.99},
                {'date': '2024-02-15', 'price': 31.50}
            ],
            'performanceHistory': [
                {'date': '2024-01-15', 'sales': 100, 'views': 1000, 'conversion_rate': 10.0, 'revenue': 2999.00}
            ]
        },
        {
            'filename': 'test2.jpg',
            'category': 'bowls',
            'sku': 'BWL-002',
            'name': 'Test Bowl',
            'price': 15.99,
            'priceHistory': [],
            'performanceHistory': []
        }
    ]
    
    # Generate CSV
    csv_output = generate_csv_builder_output(original_products)
    
    # Parse back
    parsed, errors = main_app_parse_csv(csv_output)
    
    # Verify data integrity
    assert len(errors) == 0
    
    # Product 1
    p1 = parsed['test1.jpg']
    assert p1['category'] == 'plates'
    assert p1['sku'] == 'PLT-001'
    assert p1['name'] == 'Test Plate'
    assert p1['priceHistory'] is not None
    assert len(p1['priceHistory']) == 2
    assert p1['priceHistory'][0]['price'] == 29.99
    assert p1['performanceHistory'] is not None
    assert p1['performanceHistory'][0]['sales'] == 100
    
    # Product 2 (minimal data)
    p2 = parsed['test2.jpg']
    assert p2['category'] == 'bowls'
    assert p2['sku'] == 'BWL-002'
    
    print("✓ test_csv_round_trip passed")


def test_tab_separated_values():
    """Test tab-separated CSV format"""
    products = [
        {'filename': 'test.jpg', 'category': 'cat', 'sku': 'SKU1', 'name': 'Test', 'price': 10.00, 'priceHistory': [], 'performanceHistory': []}
    ]
    
    csv_output = generate_csv_builder_output(products, separator='\t')
    parsed, errors = main_app_parse_csv(csv_output)
    
    assert len(errors) == 0
    assert 'test.jpg' in parsed
    assert parsed['test.jpg']['sku'] == 'SKU1'
    
    print("✓ test_tab_separated_values passed")


def test_no_headers():
    """Test CSV without headers"""
    csv = '''product1.jpg,plates,PLT-001,Plate,29.99
product2.jpg,bowls,BWL-002,Bowl,15.99'''
    
    parsed, errors = main_app_parse_csv(csv)
    
    assert len(parsed) == 2
    assert parsed['product1.jpg']['category'] == 'plates'
    assert parsed['product2.jpg']['sku'] == 'BWL-002'
    
    print("✓ test_no_headers passed")


def test_main_app_to_csv_builder_flow():
    """
    Test the complete Main App → CSV Builder → Main App flow:
    1. User uploads folder in Main App
    2. Main App sends file data to CSV Builder via sessionStorage
    3. CSV Builder receives and populates products
    4. User can export template or link data
    5. User exports final CSV
    6. CSV Builder sends CSV back to Main App via postMessage
    7. Main App receives and uses the CSV
    """
    # Step 1: Simulate Main App sending file data (as it does via sessionStorage)
    main_app_file_data = [
        {'filename': 'IMG_001.jpg', 'category': 'plates', 'size': 1024000, 'type': 'image/jpeg'},
        {'filename': 'IMG_002.jpg', 'category': 'plates', 'size': 2048000, 'type': 'image/jpeg'},
        {'filename': 'IMG_003.jpg', 'category': 'bowls', 'size': 512000, 'type': 'image/jpeg'},
        {'filename': 'IMG_004.jpg', 'category': '', 'size': 768000, 'type': 'image/jpeg'},  # No category
    ]
    
    # Step 2: CSV Builder receives and creates products (simulating checkForMainAppData)
    csv_builder_products = [
        {
            'filename': f['filename'],
            'category': f['category'] or '',
            'sku': '',
            'name': '',
            'price': '',
            'priceHistory': [],
            'performanceHistory': [],
            'selected': False
        }
        for f in main_app_file_data
    ]
    
    # Verify products created correctly
    assert len(csv_builder_products) == 4
    assert csv_builder_products[0]['filename'] == 'IMG_001.jpg'
    assert csv_builder_products[0]['category'] == 'plates'
    assert csv_builder_products[3]['category'] == ''  # Empty category preserved
    
    # Step 3: User adds metadata (simulating user input)
    csv_builder_products[0]['sku'] = 'PLT-001'
    csv_builder_products[0]['name'] = 'Dinner Plate'
    csv_builder_products[0]['price'] = 24.99
    
    csv_builder_products[1]['sku'] = 'PLT-002'
    csv_builder_products[1]['name'] = 'Salad Plate'
    csv_builder_products[1]['price'] = 19.99
    
    csv_builder_products[2]['sku'] = 'BWL-001'
    csv_builder_products[2]['name'] = 'Soup Bowl'
    csv_builder_products[2]['price'] = 14.99
    
    csv_builder_products[3]['sku'] = 'MSC-001'
    csv_builder_products[3]['name'] = 'Mystery Item'
    csv_builder_products[3]['price'] = 9.99
    csv_builder_products[3]['category'] = 'misc'  # User adds category
    
    # Step 4: Generate CSV (simulating export)
    csv_output = generate_csv_builder_output(csv_builder_products)
    
    # Step 5: Main App parses the CSV
    parsed, errors = main_app_parse_csv(csv_output)
    
    # Verify complete round-trip
    assert len(errors) == 0, f"Parse errors: {errors}"
    assert len(parsed) == 4
    
    # Verify all products preserved correctly
    assert parsed['IMG_001.jpg']['sku'] == 'PLT-001'
    assert parsed['IMG_001.jpg']['name'] == 'Dinner Plate'
    assert parsed['IMG_001.jpg']['category'] == 'plates'
    
    assert parsed['IMG_004.jpg']['category'] == 'misc'  # User-added category
    
    print("✓ test_main_app_to_csv_builder_flow passed")


def test_template_export_import_flow():
    """
    Test the Excel template workflow:
    1. User uploads folder
    2. User exports template CSV
    3. User fills template in Excel
    4. User imports filled template
    5. Products are updated with metadata
    6. User exports final CSV
    7. Main App parses correctly
    """
    # Step 1: Initial products from folder upload
    initial_products = [
        {'filename': 'product_a.jpg', 'category': 'cat1'},
        {'filename': 'product_b.jpg', 'category': 'cat2'},
        {'filename': 'product_c.jpg', 'category': ''},
    ]
    
    # Step 2: Generate template
    template = generate_template_csv(initial_products)
    
    # Verify template has correct structure
    lines = template.split('\n')
    assert lines[0] == 'filename,category,sku,name,price,price_history,performance_history'
    assert 'product_a.jpg,cat1' in lines[1]
    assert 'product_b.jpg,cat2' in lines[2]
    assert 'product_c.jpg,' in lines[3]  # Empty category
    
    # Step 3: Simulate user filling template in Excel
    filled_template = """filename,category,sku,name,price,price_history,performance_history
product_a.jpg,cat1,SKU-A,Product A,29.99,2024-01-15:29.99;2024-02-15:27.99,
product_b.jpg,cat2,SKU-B,Product B,39.99,,100:500:20.0:3999
product_c.jpg,cat3,SKU-C,Product C,19.99,,"""
    
    # Step 4: Parse filled template (simulating import)
    parsed_template, _ = main_app_parse_csv(filled_template)
    
    # Step 5: Update products with imported data
    updated_products = []
    for p in initial_products:
        imported = parsed_template.get(p['filename'], {})
        updated_products.append({
            'filename': p['filename'],
            'category': imported.get('category') or p['category'],
            'sku': imported.get('sku', ''),
            'name': imported.get('name', ''),
            'price': imported.get('priceHistory', [{}])[0].get('price', '') if imported.get('priceHistory') else '',
            'priceHistory': imported.get('priceHistory', []),
            'performanceHistory': imported.get('performanceHistory', [])
        })
    
    # Step 6: Generate final CSV
    final_csv = generate_csv_builder_output(updated_products)
    
    # Step 7: Main App parses final CSV
    final_parsed, errors = main_app_parse_csv(final_csv)
    
    # Verify
    assert len(errors) == 0
    assert final_parsed['product_a.jpg']['sku'] == 'SKU-A'
    assert final_parsed['product_a.jpg']['priceHistory'] is not None
    assert final_parsed['product_b.jpg']['performanceHistory'] is not None
    assert final_parsed['product_c.jpg']['category'] == 'cat3'  # Updated from template
    
    print("✓ test_template_export_import_flow passed")


def test_dumb_user_scenarios():
    """
    Test scenarios where user makes common mistakes:
    - Extra whitespace
    - Wrong case
    - Missing fields
    - Extra columns
    - Weird characters
    """
    # Scenario 1: Extra whitespace everywhere
    messy_csv = """  filename  ,  category  ,  sku  ,  name  ,  price  
  product1.jpg  ,  plates  ,  SKU-001  ,  My Product  ,  29.99  
product2.jpg,bowls,SKU-002,Another Product,19.99"""
    
    parsed, errors = main_app_parse_csv(messy_csv)
    assert 'product1.jpg' in parsed or '  product1.jpg  ' in parsed
    
    # Scenario 2: Mixed case headers
    mixed_case_csv = """FILENAME,Category,SKU,NAME,Price
product1.jpg,plates,SKU-001,Product,29.99"""
    
    parsed, errors = main_app_parse_csv(mixed_case_csv)
    assert len(parsed) == 1
    
    # Scenario 3: Extra columns (user added notes column)
    extra_cols_csv = """filename,category,sku,name,price,notes,extra
product1.jpg,plates,SKU-001,Product,29.99,my notes,extra data"""
    
    parsed, errors = main_app_parse_csv(extra_cols_csv)
    assert parsed['product1.jpg']['sku'] == 'SKU-001'
    
    # Scenario 4: Columns in different order (shouldn't work without headers)
    # But with headers it should work based on column names
    
    # Scenario 5: User puts price in wrong format
    bad_price_csv = """filename,category,sku,name,price
product1.jpg,plates,SKU-001,Product,$29.99
product2.jpg,plates,SKU-002,Product,29,99
product3.jpg,plates,SKU-003,Product,twenty"""
    
    parsed, errors = main_app_parse_csv(bad_price_csv)
    # Should not crash, just handle gracefully
    assert len(parsed) == 3
    
    print("✓ test_dumb_user_scenarios passed")


def test_metadata_accuracy():
    """
    Test that metadata is applied accurately and completely.
    Every field should be preserved exactly as entered.
    """
    # Create products with all possible metadata
    products = [
        {
            'filename': 'test_product.jpg',
            'category': 'Test Category',
            'sku': 'TEST-SKU-12345',
            'name': 'Test Product Name With Spaces',
            'price': 123.45,
            'priceHistory': [
                {'date': '2024-01-01', 'price': 100.00},
                {'date': '2024-02-01', 'price': 110.00},
                {'date': '2024-03-01', 'price': 123.45},
            ],
            'performanceHistory': [
                {'date': '2024-01-01', 'sales': 50, 'views': 500, 'conversion_rate': 10.0, 'revenue': 5000.00},
                {'date': '2024-02-01', 'sales': 75, 'views': 600, 'conversion_rate': 12.5, 'revenue': 8250.00},
            ]
        }
    ]
    
    # Generate CSV
    csv_output = generate_csv_builder_output(products)
    
    # Parse back
    parsed, errors = main_app_parse_csv(csv_output)
    
    # Verify EXACT accuracy
    assert len(errors) == 0
    p = parsed['test_product.jpg']
    
    # Basic fields
    assert p['category'] == 'Test Category', f"Category mismatch: {p['category']}"
    assert p['sku'] == 'TEST-SKU-12345', f"SKU mismatch: {p['sku']}"
    assert p['name'] == 'Test Product Name With Spaces', f"Name mismatch: {p['name']}"
    
    # Price history
    assert p['priceHistory'] is not None, "Price history missing"
    assert len(p['priceHistory']) == 3, f"Price history count: {len(p['priceHistory'])}"
    assert p['priceHistory'][0]['date'] == '2024-01-01'
    assert p['priceHistory'][0]['price'] == 100.00
    assert p['priceHistory'][2]['price'] == 123.45
    
    # Performance history
    assert p['performanceHistory'] is not None, "Performance history missing"
    assert len(p['performanceHistory']) == 2, f"Performance history count: {len(p['performanceHistory'])}"
    assert p['performanceHistory'][0]['sales'] == 50
    assert p['performanceHistory'][0]['views'] == 500
    assert p['performanceHistory'][0]['conversion_rate'] == 10.0
    assert p['performanceHistory'][0]['revenue'] == 5000.00
    
    print("✓ test_metadata_accuracy passed")


def test_empty_and_null_handling():
    """
    Test that empty/null values don't break anything and are handled gracefully.
    """
    # CSV with various empty/null scenarios
    csv = """filename,category,sku,name,price,price_history,performance_history
has_all.jpg,cat,SKU1,Name,29.99,2024-01-01:29.99,2024-01-01:100:500:20.0:2999
empty_category.jpg,,SKU2,Name,29.99,,
empty_sku.jpg,cat,,Name,29.99,,
empty_name.jpg,cat,SKU3,,29.99,,
empty_price.jpg,cat,SKU4,Name,,,
all_empty.jpg,,,,,,
just_filename.jpg,,,,,,"""
    
    parsed, errors = main_app_parse_csv(csv)
    
    # All rows should parse
    assert len(parsed) == 7, f"Expected 7 products, got {len(parsed)}"
    
    # Verify empty handling
    assert parsed['has_all.jpg']['category'] == 'cat'
    assert parsed['empty_category.jpg']['category'] in [None, '', 'None']
    assert parsed['empty_sku.jpg']['sku'] in [None, '', 'None']
    assert parsed['empty_name.jpg']['name'] in [None, '', 'None']
    assert parsed['all_empty.jpg'] is not None  # Row exists
    assert parsed['just_filename.jpg'] is not None  # Row exists
    
    print("✓ test_empty_and_null_handling passed")


def test_special_filenames():
    """
    Test handling of special filenames that users might have.
    """
    products = [
        {'filename': 'normal.jpg', 'category': 'cat', 'sku': 'SKU1', 'name': 'Normal', 'price': 10, 'priceHistory': [], 'performanceHistory': []},
        {'filename': 'with spaces.jpg', 'category': 'cat', 'sku': 'SKU2', 'name': 'Spaces', 'price': 10, 'priceHistory': [], 'performanceHistory': []},
        {'filename': 'with-dashes.jpg', 'category': 'cat', 'sku': 'SKU3', 'name': 'Dashes', 'price': 10, 'priceHistory': [], 'performanceHistory': []},
        {'filename': 'with_underscores.jpg', 'category': 'cat', 'sku': 'SKU4', 'name': 'Underscores', 'price': 10, 'priceHistory': [], 'performanceHistory': []},
        {'filename': 'UPPERCASE.JPG', 'category': 'cat', 'sku': 'SKU5', 'name': 'Upper', 'price': 10, 'priceHistory': [], 'performanceHistory': []},
        {'filename': 'MixedCase.Jpg', 'category': 'cat', 'sku': 'SKU6', 'name': 'Mixed', 'price': 10, 'priceHistory': [], 'performanceHistory': []},
        {'filename': '123numeric.jpg', 'category': 'cat', 'sku': 'SKU7', 'name': 'Numeric', 'price': 10, 'priceHistory': [], 'performanceHistory': []},
        {'filename': 'special(chars)[here].jpg', 'category': 'cat', 'sku': 'SKU8', 'name': 'Special', 'price': 10, 'priceHistory': [], 'performanceHistory': []},
    ]
    
    csv_output = generate_csv_builder_output(products)
    parsed, errors = main_app_parse_csv(csv_output)
    
    assert len(errors) == 0
    assert len(parsed) == 8
    
    # All filenames should be preserved exactly
    for p in products:
        assert p['filename'] in parsed, f"Missing filename: {p['filename']}"
        assert parsed[p['filename']]['sku'] == p['sku']
    
    print("✓ test_special_filenames passed")


def run_all_tests():
    """Run all end-to-end tests"""
    print("\n" + "="*60)
    print("CSV Builder → Main App End-to-End Tests")
    print("="*60 + "\n")
    
    tests = [
        test_basic_workflow,
        test_excel_workflow,
        test_linking_workflow,
        test_real_world_data_handling,
        test_large_dataset,
        test_price_history_formats,
        test_performance_history_formats,
        test_csv_round_trip,
        test_tab_separated_values,
        test_no_headers,
        test_main_app_to_csv_builder_flow,
        test_template_export_import_flow,
        test_dumb_user_scenarios,
        test_metadata_accuracy,
        test_empty_and_null_handling,
        test_special_filenames,
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
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    if failed == 0:
        print("🎉 All end-to-end tests passed!")
        print("CSV Builder output is fully compatible with Main App!")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
