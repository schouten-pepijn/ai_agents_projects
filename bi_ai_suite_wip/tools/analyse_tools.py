import json
from langchain_core.tools import tool


@tool
def perform_trend_analysis() -> str:
    """Perform trend analysis on collected data"""

    analysis = {
        "revenue_trend": "upward",
        "seasonal_pattern": "Q4 peak observed",
        "growth_rate": "15% YoY",
        "key_insights": [
            "Revenue shows strong upward trend",
            "Peak sales in December",
            "Electronics category driving growth",
        ],
    }
    return json.dumps(analysis, indent=2)
