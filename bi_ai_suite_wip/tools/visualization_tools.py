import matplotlib.pyplot as plt
import base64
import json
from io import BytesIO
from datetime import datetime
from langchain_core.tools import tool
from data_generators.generate_data import generate_sample_sales_data


@tool
def create_revenue_chart() -> str:
    """Create revenue visualization"""

    data = generate_sample_sales_data()

    plt.figure(figsize=(12, 6), tight_layout=True)
    plt.plot(data["date"], data["revenue"])
    plt.title("Daily Revenue Over Time")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    chart_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()

    return f"Revenue chart created and saved (base64 length: {len(chart_b64)})"


@tool
def generate_executive_report(analysis_results: str, chart_info: str) -> str:
    """Generate executive summary report"""
    report = {
        "title": "Business Intelligence Executive Summary",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "key_metrics": {
            "total_revenue": "$4.2M",
            "growth_rate": "15%",
            "customer_satisfaction": "87%",
        },
        "insights": [
            "Strong revenue growth driven by electronics category",
            "Customer satisfaction remains high",
            "Seasonal patterns suggest Q4 opportunity",
        ],
        "recommendations": [
            "Increase inventory for Q4 peak season",
            "Focus marketing on electronics category",
            "Implement customer retention programs",
        ],
    }
    return json.dumps(report, indent=2)
