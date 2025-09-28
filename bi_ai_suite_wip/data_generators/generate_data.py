import polars as pl
import numpy as np
from datetime import date

np.random.seed(87)


def generate_sample_sales_data() -> pl.DataFrame:
    dates = pl.date_range(
        start=date(2024, 1, 1), end=date(2024, 9, 1), interval="1d", eager=True
    )

    data = {
        "date": dates,
        "revenue": np.random.normal(50000, 10000, len(dates))
        + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 5000,
        "orders": np.random.poisson(100, len(dates)),
        "customers": np.random.randint(50, 150, len(dates)),
        "region": np.random.choice(["North", "South", "East", "West"], len(dates)),
        "product_category": np.random.choice(
            ["Electronics", "Clothing", "Books", "Home"], len(dates)
        ),
    }

    return pl.DataFrame(data)


def generate_sample_kpi_data() -> dict:
    return {
        "customer_satisfaction": 0.87,
        "conversion_rate": 0.034,
        "average_order_value": 125.50,
        "churn_rate": 0.05,
        "monthly_active_users": 12500,
        "revenue_growth": 0.15,
    }
