"""Test script to verify database initialization"""
from database import init_db, get_db_connection

if __name__ == '__main__':
    print("Initializing database...")
    init_db()
    
    print("\nVerifying tables...")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Tables created: {[table['name'] for table in tables]}")
        
        # Check indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = cursor.fetchall()
        print(f"Indexes created: {[index['name'] for index in indexes]}")
    
    print("\nDatabase setup complete!")
