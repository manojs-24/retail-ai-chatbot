from pathlib import Path

import pandas as pd

from backend.core.database import engine

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

FILES = {
    "users": "users.csv",
    "products": "products.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "product_reviews": "product_reviews.csv",
}

print("=" * 50)
print("Loading CSV files into SQLite...")
print("=" * 50)

for table, filename in FILES.items():

    csv_path = DATA_DIR / filename

    df = pd.read_csv(csv_path)

    # Convert date columns automatically
    for column in df.columns:
        if "date" in column.lower():
            df[column] = pd.to_datetime(df[column]).dt.date

    # Convert verified_purchase into bool
    if "verified_purchase" in df.columns:
        df["verified_purchase"] = (
            df["verified_purchase"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

    df.to_sql(
        table,
        con=engine,
        if_exists="append",
        index=False,
    )

    print(f"✓ {table:<20} {len(df)} rows inserted")

print("\nDatabase seeding completed successfully.")