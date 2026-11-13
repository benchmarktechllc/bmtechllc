import sqlite3
from pathlib import Path

def seed_database():
    db_path = Path("data/store_sales.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table matching schema_guide.md
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            order_date TEXT NOT NULL
        )
    """)

    # Clear existing data if re-running
    cursor.execute("DELETE FROM orders")

    # Insert sample orders (including edge cases like VOIDED and REFUNDED)
    sample_orders = [
        ("Alice Smith", 120.50, "COMPLETED", "2026-10-01"),
        ("Bob Jones", 45.00, "COMPLETED", "2026-10-02"),
        ("Charlie Brown", 250.00, "VOIDED", "2026-10-03"),
        ("Diana Prince", 89.99, "REFUNDED", "2026-10-04"),
        ("Ethan Hunt", 310.00, "COMPLETED", "2026-10-05"),
        ("Fiona Glenanne", 75.25, "COMPLETED", "2026-10-06")
    ]

    cursor.executemany("""
        INSERT INTO orders (customer_name, amount, status, order_date)
        VALUES (?, ?, ?, ?)
    """, sample_orders)

    conn.commit()
    conn.close()
    print("Database successfully created and seeded at data/store_sales.db")

if __name__ == "__main__":
    seed_database()