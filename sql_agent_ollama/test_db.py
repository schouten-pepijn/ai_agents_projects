"""Simple test script to verify the SQL agent database and tools work."""

from sql_agent import lc_db

def test_database():
    print("Testing database connection and tools...")
    
    # Test 1: List tables
    print("\n1. Testing list tables:")
    tables = lc_db.get_usable_table_names()
    print(f"Tables: {', '.join(tables)}")
    
    # Test 2: Describe tables
    print("\n2. Testing describe tables:")
    schema = lc_db.get_table_info(table_names=list(tables))
    print(f"Schema:\n{schema}")
    
    # Test 3: Simple queries
    print("\n3. Testing queries:")
    try:
        result = lc_db.run("SELECT COUNT(*) as customer_count FROM customers")
        print(f"Customer count: {result}")
        
        result2 = lc_db.run("SELECT name, country FROM customers LIMIT 3")
        print(f"Sample customers: {result2}")
        
        result3 = lc_db.run("SELECT c.name, o.amount, o.status FROM customers c JOIN orders o ON c.customer_id = o.customer_id LIMIT 3")
        print(f"Sample orders with customers: {result3}")
        
    except (ValueError, RuntimeError) as e:
        print(f"Error running query: {e}")
    
    print("\nDatabase test completed!")

if __name__ == "__main__":
    test_database()
