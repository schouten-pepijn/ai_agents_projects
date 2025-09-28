from datetime import datetime
from langchain_core.tools import tool
import json
from data_generators.generate_data import generate_sample_kpi_data


@tool
def check_kpi_thresholds(kpi_data: str) -> str:
    """Monitor KPIs and generate alerts if thresholds are breached"""
    kpis = generate_sample_kpi_data()

    alerts = []
    thresholds = {
        "customer_satisfaction": 0.8,
        "conversion_rate": 0.03,
        "churn_rate": 0.08,
    }

    for metric, threshold in thresholds.items():
        if metric in kpis:
            if metric == "churn_rate" and kpis[metric] > threshold:
                alerts.append(
                    f"ALERT: {metric} ({kpis[metric]:.2%}) exceeds threshold ({threshold:.2%})"
                )
            elif metric != "churn_rate" and kpis[metric] < threshold:
                alerts.append(
                    f"ALERT: {metric} ({kpis[metric]:.2%}) below threshold ({threshold:.2%})"
                )

    if not alerts:
        alerts.append("All KPIs within acceptable ranges")

    return json.dumps({"alerts": alerts, "timestamp": datetime.now().isoformat()})
