import json
from langchain_core.tools import tool
from data_generators.generate_data import (
    generate_sample_sales_data,
    generate_sample_kpi_data,
)


@tool
def collect_sales_data() -> str:
    """Collect sales data from sources"""

    data = generate_sample_sales_data()
    return data

@tool
def collect_kpi_data() -> str:
    """Collect KPI metrics"""

    kpis = generate_sample_kpi_data()
    return f"Collected KPI metrics: {json.dumps(kpis, indent=2)}"
