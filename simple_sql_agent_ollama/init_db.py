import os
from config import DB_PATH
from sqlalchemy import create_engine, text


# Ensure database directory exists
# os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

ENGINE = create_engine(f"sqlite:///{DB_PATH}", future=True, echo=False)

def init_db():
    with ENGINE.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS customers"))
        conn.execute(text("DROP TABLE IF EXISTS orders"))
        conn.execute(text("""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name        TEXT NOT NULL,
                country     TEXT NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE orders (
                order_id     INTEGER PRIMARY KEY,
                customer_id  INTEGER NOT NULL,
                order_date   TEXT NOT NULL,
                amount       REAL NOT NULL,
                status       TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        """))

        # Seed sample data
        customers = [
            (1, "Alice", "NL"),
            (2, "Bob", "DE"),
            (3, "Carol", "NL"),
            (4, "Dieter", "BE"),
        ]
        for c in customers:
            conn.execute(text("INSERT INTO customers VALUES (:id, :n, :cty)"),
                            {"id": c[0], "n": c[1], "cty": c[2]})

        orders = [
            (101, 1, "2025-07-01", 120.50, "shipped"),
            (102, 2, "2025-07-03",  80.00, "pending"),
            (103, 1, "2025-07-05",  50.00, "canceled"),
            (104, 3, "2025-07-09", 199.99, "shipped"),
            (105, 4, "2025-08-01",  10.00, "pending"),
        ]
        for o in orders:
            conn.execute(
                text("INSERT INTO orders VALUES (:oid, :cid, :dt, :amt, :st)"),
                {"oid": o[0], "cid": o[1], "dt": o[2], "amt": o[3], "st": o[4]},
            )