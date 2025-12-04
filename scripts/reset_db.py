import sqlite3

conn = sqlite3.connect('backend/product_matching.db')
cursor = conn.cursor()

print('Current database state:')
cursor.execute('SELECT COUNT(*) FROM products WHERE is_historical = 1')
print(f'Historical products: {cursor.fetchone()[0]}')

cursor.execute('SELECT COUNT(*) FROM products WHERE is_historical = 0')
print(f'New products: {cursor.fetchone()[0]}')

response = input('\nDo you want to clear ALL products? (yes/no): ')
if response.lower() == 'yes':
    cursor.execute('DELETE FROM matches')
    cursor.execute('DELETE FROM features')
    cursor.execute('DELETE FROM price_history')
    cursor.execute('DELETE FROM performance_history')
    cursor.execute('DELETE FROM products')
    conn.commit()
    print('\n✓ Database cleared! All products deleted.')
    print('\nNext steps:')
    print('1. Go to Historical Catalog section')
    print('2. Select "Replace catalog" option')
    print('3. Upload your historical product images')
    print('4. Then go to New Products section')
    print('5. Upload your new product images')
    print('6. Run matching')
else:
    print('No changes made.')

conn.close()
