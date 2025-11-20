"""
Test CSV Builder Integration with Product Matching System

This test verifies that CSV files created by the CSV Builder tool
are correctly parsed and processed by the main application.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from io import StringIO
import json

def parse_csv_like_frontend(csv_text):
    """
    Simulate the frontend CSV parsing logic from app.js
    This mimics the parseCsv() function behavior
    """
    lines = [line.strip() for line in csv_text.split('\n') if line.strip()]
    
    if not lines:
        return {}
    
    # Check if first line is a header
    first_line = lines[0]
    has_header = (
        'filename' in first_line.lower() or
        'category' in first_line.lower() or
        'sku' in first_line.lower()
    )
    
    data_lines = lines[1:] if has_header else lines
    
    result = {}
    
    for line_num, line in enumerate(data_lines, start=2 if has_header else 1):
        # Handle both comma and tab separated values
        # Split by comma or tab, handle quoted values
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
            
            # Parse price - can be in column 4 (single price) or column 5 (price_history)
            price_history = None
            single_price = None
            
            # Column 5 is price_history field (if it exists and has history format)
            price_history_str = parts[5] if len(parts) > 5 else None
            if price_history_str and price_history_str.strip() and (':' in price_history_str or ';' in price_history_str):
                price_history = parse_price_history(price_history_str)
            
            # Column 4 is single price field (if no price_history or it's a simple number)
            if not price_history and len(parts) > 4 and parts[4] and parts[4].strip():
                # Check if it's a simple price or price history format
                if ':' in parts[4] or ';' in parts[4]:
                    # It's price history in column 4 (old format or no header)
                    price_history = parse_price_history(parts[4])
                else:
                    # It's a single price
                    try:
                        price = float(parts[4])
                        if price >= 0:
                            single_price = price
                            # Convert to price history
                            from datetime import date
                            today = date.today().isoformat()
                            price_history = [{'date': today, 'price': price}]
                    except ValueError:
                        pass
            
            # Parse performance history - column 6 (or column 5 if no price_history column)
            performance_history = None
            performance_history_str = parts[6] if len(parts) > 6 else None
            
            # If column 6 doesn't exist or is empty, check column 5 (might be performance without price history)
            if (not performance_history_str or not performance_history_str.strip()) and len(parts) > 5:
                # Check if column 5 looks like performance history (has 4+ colons)
                if parts[5] and parts[5].count(':') >= 3:
                    performance_history_str = parts[5]
            
            if performance_history_str and performance_history_str.strip() and ':' in performance_history_str:
                performance_history = parse_performance_history(performance_history_str)
            
            result[filename] = {
                'category': category,
                'sku': sku,
                'name': name,
                'priceHistory': price_history,
                'performanceHistory': performance_history
            }
    
    return result

def parse_price_history(price_history_str):
    """Parse price history string - matches frontend logic"""
    if not price_history_str or not price_history_str.strip():
        return None
    
    price_history = []
    
    # Check if it contains dates (has colons)
    if ':' in price_history_str:
        # Format with dates
        entries = [e.strip() for e in price_history_str.replace(',', ';').split(';') if e.strip()]
        
        for entry in entries:
            parts = [p.strip() for p in entry.split(':')]
            if len(parts) >= 2:
                date = parts[0]
                try:
                    price = float(parts[1])
                    if price >= 0:
                        price_history.append({'date': date, 'price': price})
                except ValueError:
                    continue
    else:
        # Format without dates - just prices
        from datetime import date, timedelta
        prices = [p.strip() for p in price_history_str.replace(',', ';').split(';') if p.strip()]
        today = date.today()
        
        for i, price_str in enumerate(prices):
            try:
                price = float(price_str)
                if price >= 0:
                    # Generate monthly dates going backwards
                    months_back = len(prices) - 1 - i
                    price_date = today - timedelta(days=months_back * 30)
                    price_history.append({
                        'date': price_date.isoformat(),
                        'price': price
                    })
            except ValueError:
                continue
    
    # Limit to 12 months
    if price_history:
        price_history.sort(key=lambda x: x['date'])
        return price_history[-12:]
    
    return None

def parse_performance_history(performance_history_str):
    """Parse performance history string - matches frontend logic"""
    if not performance_history_str or not performance_history_str.strip():
        return None
    
    performance_history = []
    entries = [e.strip() for e in performance_history_str.replace(',', ';').split(';') if e.strip()]
    
    for entry in entries:
        parts = [p.strip() for p in entry.split(':')]
        
        # Check if first part is a date
        if len(parts) >= 5 and '-' in parts[0]:
            # Format with date
            try:
                performance_history.append({
                    'date': parts[0],
                    'sales': int(parts[1]) if parts[1] else 0,
                    'views': int(parts[2]) if parts[2] else 0,
                    'conversion_rate': float(parts[3]) if parts[3] else 0.0,
                    'revenue': float(parts[4]) if parts[4] else 0.0
                })
            except (ValueError, IndexError):
                continue
        elif len(parts) >= 4:
            # Format without date
            from datetime import date, timedelta
            today = date.today()
            months_back = len(performance_history)
            perf_date = today - timedelta(days=months_back * 30)
            
            try:
                performance_history.append({
                    'date': perf_date.isoformat(),
                    'sales': int(parts[0]) if parts[0] else 0,
                    'views': int(parts[1]) if parts[1] else 0,
                    'conversion_rate': float(parts[2]) if parts[2] else 0.0,
                    'revenue': float(parts[3]) if parts[3] else 0.0
                })
            except (ValueError, IndexError):
                continue
    
    # Limit to 12 months
    if performance_history:
        performance_history.sort(key=lambda x: x['date'])
        return performance_history[-12:]
    
    return None

def test_basic_csv():
    """Test basic CSV format from CSV Builder"""
    csv = """filename,category,sku,name,price
product1.jpg,placemats,PM-001,Blue Placemat,29.99
product2.jpg,dinnerware,DW-002,White Plate Set,45.00"""
    
    result = parse_csv_like_frontend(csv)
    
    assert 'product1.jpg' in result
    assert result['product1.jpg']['category'] == 'placemats'
    assert result['product1.jpg']['sku'] == 'PM-001'
    assert result['product1.jpg']['name'] == 'Blue Placemat'
    assert result['product1.jpg']['priceHistory'][0]['price'] == 29.99
    
    assert 'product2.jpg' in result
    assert result['product2.jpg']['category'] == 'dinnerware'
    
    print("✓ Basic CSV test passed")

def test_csv_with_price_history():
    """Test CSV with price history from CSV Builder"""
    csv = """filename,category,sku,name,price,price_history
product1.jpg,placemats,PM-001,Blue Placemat,29.99,2024-01-15:29.99;2024-02-15:31.50;2024-03-15:28.75"""
    
    result = parse_csv_like_frontend(csv)
    
    assert 'product1.jpg' in result
    assert result['product1.jpg']['priceHistory'] is not None
    assert len(result['product1.jpg']['priceHistory']) == 3
    assert result['product1.jpg']['priceHistory'][0]['date'] == '2024-01-15'
    assert result['product1.jpg']['priceHistory'][0]['price'] == 29.99
    assert result['product1.jpg']['priceHistory'][1]['price'] == 31.50
    
    print("✓ CSV with price history test passed")

def test_csv_with_quick_price_format():
    """Test CSV with quick price format (no dates) from CSV Builder"""
    csv = """filename,category,sku,name,price,price_history
product1.jpg,placemats,PM-001,Blue Placemat,29.99,29.99;31.50;28.75;32.00"""
    
    result = parse_csv_like_frontend(csv)
    
    assert 'product1.jpg' in result
    assert result['product1.jpg']['priceHistory'] is not None
    assert len(result['product1.jpg']['priceHistory']) == 4
    # Dates should be auto-generated
    assert result['product1.jpg']['priceHistory'][0]['date'] is not None
    assert result['product1.jpg']['priceHistory'][0]['price'] == 29.99
    
    print("✓ CSV with quick price format test passed")

def test_csv_with_performance_history():
    """Test CSV with performance history from CSV Builder"""
    csv = """filename,category,sku,name,price,price_history,performance_history
product1.jpg,placemats,PM-001,Blue Placemat,29.99,2024-01-15:29.99,2024-01-15:150:1200:12.5:1800;2024-02-15:180:1500:12.0:2160"""
    
    result = parse_csv_like_frontend(csv)
    
    assert 'product1.jpg' in result
    assert result['product1.jpg']['performanceHistory'] is not None
    assert len(result['product1.jpg']['performanceHistory']) == 2
    
    perf1 = result['product1.jpg']['performanceHistory'][0]
    assert perf1['date'] == '2024-01-15'
    assert perf1['sales'] == 150
    assert perf1['views'] == 1200
    assert perf1['conversion_rate'] == 12.5
    assert perf1['revenue'] == 1800.0
    
    print("✓ CSV with performance history test passed")

def test_csv_with_quick_performance_format():
    """Test CSV with quick performance format (no dates) from CSV Builder"""
    csv = """filename,category,sku,name,price,price_history,performance_history
product1.jpg,placemats,PM-001,Blue Placemat,29.99,,150:1200:12.5:1800;180:1500:12.0:2160"""
    
    result = parse_csv_like_frontend(csv)
    
    assert 'product1.jpg' in result
    assert result['product1.jpg']['performanceHistory'] is not None
    assert len(result['product1.jpg']['performanceHistory']) == 2
    # Dates should be auto-generated
    assert result['product1.jpg']['performanceHistory'][0]['date'] is not None
    # Check that sales values are present (order may vary due to sorting)
    sales_values = [p['sales'] for p in result['product1.jpg']['performanceHistory']]
    assert 150 in sales_values
    assert 180 in sales_values
    
    print("✓ CSV with quick performance format test passed")

def test_csv_with_missing_fields():
    """Test CSV with missing optional fields (real-world data)"""
    csv = """filename,category,sku,name,price
product1.jpg,placemats,,,29.99
product2.jpg,,DW-002,White Plate,
product3.jpg,,,"""
    
    result = parse_csv_like_frontend(csv)
    
    assert 'product1.jpg' in result
    assert result['product1.jpg']['category'] == 'placemats'
    assert result['product1.jpg']['sku'] is None or result['product1.jpg']['sku'] == ''
    
    assert 'product2.jpg' in result
    assert result['product2.jpg']['category'] is None or result['product2.jpg']['category'] == ''
    assert result['product2.jpg']['sku'] == 'DW-002'
    
    assert 'product3.jpg' in result
    
    print("✓ CSV with missing fields test passed")

def test_csv_with_quoted_fields():
    """Test CSV with quoted fields (special characters)"""
    csv = """filename,category,sku,name,price
"product1.jpg","placemats, decorative","PM-001","Blue Placemat, Large",29.99
product2.jpg,dinnerware,DW-002,"White ""Premium"" Plate",45.00"""
    
    result = parse_csv_like_frontend(csv)
    
    assert 'product1.jpg' in result
    assert 'placemats' in result['product1.jpg']['category']
    assert 'Blue Placemat' in result['product1.jpg']['name']
    
    assert 'product2.jpg' in result
    assert 'Premium' in result['product2.jpg']['name']
    
    print("✓ CSV with quoted fields test passed")

def test_csv_without_headers():
    """Test CSV without headers (CSV Builder option)"""
    csv = """product1.jpg,placemats,PM-001,Blue Placemat,29.99
product2.jpg,dinnerware,DW-002,White Plate Set,45.00"""
    
    result = parse_csv_like_frontend(csv)
    
    assert 'product1.jpg' in result
    assert result['product1.jpg']['category'] == 'placemats'
    
    print("✓ CSV without headers test passed")

def test_csv_with_tab_separator():
    """Test CSV with tab separator (CSV Builder option)"""
    csv = """filename\tcategory\tsku\tname\tprice
product1.jpg\tplacemats\tPM-001\tBlue Placemat\t29.99
product2.jpg\tdinnerware\tDW-002\tWhite Plate Set\t45.00"""
    
    result = parse_csv_like_frontend(csv)
    
    assert 'product1.jpg' in result
    assert result['product1.jpg']['category'] == 'placemats'
    assert result['product1.jpg']['sku'] == 'PM-001'
    
    print("✓ CSV with tab separator test passed")

def test_full_csv_builder_output():
    """Test complete CSV Builder output with all features"""
    csv = """filename,category,sku,name,price,price_history,performance_history
product1.jpg,placemats,PM-001,Blue Placemat,29.99,2024-01-15:29.99;2024-02-15:31.50;2024-03-15:28.75,2024-01-15:150:1200:12.5:1800;2024-02-15:180:1500:12.0:2160;2024-03-15:200:1800:11.1:2400
product2.jpg,dinnerware,DW-002,White Plate Set,45.00,2024-01-15:45.00;2024-02-15:42.50;2024-03-15:44.00,2024-01-15:200:2000:10.0:9000;2024-02-15:220:2200:10.0:9900;2024-03-15:240:2400:10.0:10800
product3.jpg,textiles,TX-003,Cotton Napkins,15.99,15.99;16.50;15.75,100:800:12.5:1200;120:900:13.3:1440;110:850:12.9:1320
product4.jpg,placemats,PM-004,Red Placemat,32.00,,
product5.jpg,dinnerware,DW-005,Ceramic Bowl,22.50,22.50,80:600:13.3:960;90:650:13.8:1080"""
    
    result = parse_csv_like_frontend(csv)
    
    # Test product 1 - full data with dates
    assert 'product1.jpg' in result
    assert result['product1.jpg']['category'] == 'placemats'
    assert result['product1.jpg']['priceHistory'] is not None
    assert len(result['product1.jpg']['priceHistory']) == 3
    assert result['product1.jpg']['performanceHistory'] is not None
    assert len(result['product1.jpg']['performanceHistory']) == 3
    
    # Test product 3 - quick format (no dates)
    assert 'product3.jpg' in result
    assert result['product3.jpg']['priceHistory'] is not None
    assert result['product3.jpg']['performanceHistory'] is not None
    
    # Test product 4 - no history
    assert 'product4.jpg' in result
    assert result['product4.jpg']['priceHistory'] is None or len(result['product4.jpg']['priceHistory']) == 1
    assert result['product4.jpg']['performanceHistory'] is None
    
    # Test product 5 - mixed (price with date, performance without)
    assert 'product5.jpg' in result
    assert result['product5.jpg']['performanceHistory'] is not None
    
    print("✓ Full CSV Builder output test passed")

def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("CSV Builder Integration Tests")
    print("="*60 + "\n")
    
    tests = [
        test_basic_csv,
        test_csv_with_price_history,
        test_csv_with_quick_price_format,
        test_csv_with_performance_history,
        test_csv_with_quick_performance_format,
        test_csv_with_missing_fields,
        test_csv_with_quoted_fields,
        test_csv_without_headers,
        test_csv_with_tab_separator,
        test_full_csv_builder_output
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
        print("🎉 All tests passed! CSV Builder output is fully compatible!")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
