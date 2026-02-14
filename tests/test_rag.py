from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from supply_chain_ai.rag import SupplyChainCopilot


def test_copilot_retrieves_and_answers() -> None:
    kpi_summary = pd.DataFrame([{"kpi": "service_level_pct", "value": 97.1}])
    stockout_probability = pd.DataFrame([{"region": "West", "stockout_probability": 12.3}])
    supplier_risk_scores = pd.DataFrame(
        [
            {
                "supplier_id": "SUP-09",
                "supplier_risk_score": 78.5,
                "risk_tier": "critical",
                "on_time_rate": 0.81,
                "average_delay_days": 2.5,
            }
        ]
    )
    anomaly_alerts = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "region": "West",
                "product_id": "Widget_A",
                "alert_severity": "high",
                "anomaly_reason": "sudden_sales_drop",
                "delivery_delay_days": 2.1,
                "transport_cost": 58.7,
            }
        ]
    )
    forecast_model_metrics = pd.DataFrame(
        [
            {
                "model": "random_forest",
                "test_rmse": 12.2,
                "cv_rmse_mean": 13.5,
                "cv_rmse_std": 1.1,
                "bias_proxy_mean_error": -0.2,
            }
        ]
    )

    copilot = SupplyChainCopilot.from_tables(
        kpi_summary=kpi_summary,
        stockout_probability=stockout_probability,
        supplier_risk_scores=supplier_risk_scores,
        anomaly_alerts=anomaly_alerts,
        forecast_model_metrics=forecast_model_metrics,
    )
    response = copilot.ask("Which supplier has highest disruption risk?")
    assert "answer" in response
    assert response["retrieved"]
