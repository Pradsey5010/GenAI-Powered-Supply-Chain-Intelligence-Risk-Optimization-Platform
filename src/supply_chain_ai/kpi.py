"""KPI calculations and dashboard-ready table builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _safe_ratio(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    return numerator / np.maximum(denominator, 1e-6)


@dataclass
class DashboardTables:
    demand_forecast_vs_actual: pd.DataFrame
    stockout_probability: pd.DataFrame
    supplier_risk_scores: pd.DataFrame
    anomaly_alerts: pd.DataFrame
    forecast_accuracy_metrics: pd.DataFrame
    kpi_summary: pd.DataFrame


def build_dashboard_tables(
    base_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    anomalies_frame: pd.DataFrame,
    supplier_risk_frame: pd.DataFrame,
    forecast_model_metrics: pd.DataFrame,
) -> DashboardTables:
    """Build all dashboard datasets consumed by Power BI / Tableau."""

    demand_forecast_vs_actual = forecast_frame[
        [
            "date",
            "region",
            "product_id",
            "supplier_id",
            "actual_sales",
            "forecast_sales",
            "forecast_error",
            "forecast_error_pct",
            "best_model",
            "inventory_level",
            "units_ordered",
            "units_fulfilled",
        ]
    ].copy()
    demand_forecast_vs_actual["stockout_flag"] = (
        (demand_forecast_vs_actual["units_fulfilled"] < demand_forecast_vs_actual["units_ordered"])
        | (demand_forecast_vs_actual["inventory_level"] < demand_forecast_vs_actual["actual_sales"])
    ).astype(int)

    stockout_probability = (
        demand_forecast_vs_actual.groupby("region", as_index=False)["stockout_flag"].mean().rename(
            columns={"stockout_flag": "stockout_probability"}
        )
    )
    stockout_probability["stockout_probability"] = (stockout_probability["stockout_probability"] * 100).round(2)

    anomaly_cols = [
        "date",
        "region",
        "product_id",
        "supplier_id",
        "sales_qty",
        "delivery_delay_days",
        "transport_cost",
        "anomaly_score",
        "alert_severity",
        "anomaly_reason",
    ]
    available_anomaly_cols = [col for col in anomaly_cols if col in anomalies_frame.columns]
    anomaly_alerts = anomalies_frame[available_anomaly_cols].copy() if not anomalies_frame.empty else anomalies_frame.copy()

    mae = np.mean(np.abs(demand_forecast_vs_actual["forecast_error"]))
    rmse = float(np.sqrt(np.mean(np.square(demand_forecast_vs_actual["forecast_error"]))))
    mape = np.mean(
        np.abs(_safe_ratio(demand_forecast_vs_actual["forecast_error"], demand_forecast_vs_actual["actual_sales"]))
    ) * 100
    bias = np.mean(demand_forecast_vs_actual["forecast_error"])
    forecast_accuracy_metrics = pd.DataFrame(
        [
            {"metric": "mae", "value": float(mae)},
            {"metric": "rmse", "value": rmse},
            {"metric": "mape_pct", "value": float(mape)},
            {"metric": "bias_mean_error", "value": float(bias)},
        ]
    )

    service_level_pct = float(_safe_ratio(base_frame["units_fulfilled"].sum(), base_frame["units_ordered"].sum()) * 100)
    fill_rate_pct = float(np.mean(_safe_ratio(base_frame["units_fulfilled"], base_frame["units_ordered"])) * 100)
    lead_time_variability_pct = float(
        _safe_ratio(base_frame["lead_time_days"].std(), base_frame["lead_time_days"].mean()) * 100
    )
    forecast_error_pct = float(mape)
    anomaly_rate_pct = float(_safe_ratio(len(anomalies_frame), len(base_frame)) * 100)
    high_risk_suppliers = int((supplier_risk_frame["risk_tier"].isin(["high", "critical"])).sum())

    kpi_summary = pd.DataFrame(
        [
            {"kpi": "service_level_pct", "value": round(service_level_pct, 2)},
            {"kpi": "fill_rate_pct", "value": round(fill_rate_pct, 2)},
            {"kpi": "lead_time_variability_pct", "value": round(lead_time_variability_pct, 2)},
            {"kpi": "forecast_error_pct", "value": round(forecast_error_pct, 2)},
            {"kpi": "anomaly_rate_pct", "value": round(anomaly_rate_pct, 2)},
            {"kpi": "high_risk_supplier_count", "value": float(high_risk_suppliers)},
        ]
    )

    forecast_accuracy_enriched = forecast_model_metrics.copy()
    forecast_accuracy_enriched["selected_best_model"] = forecast_model_metrics["test_rmse"].eq(
        forecast_model_metrics["test_rmse"].min()
    )

    return DashboardTables(
        demand_forecast_vs_actual=demand_forecast_vs_actual,
        stockout_probability=stockout_probability,
        supplier_risk_scores=supplier_risk_frame.copy(),
        anomaly_alerts=anomaly_alerts,
        forecast_accuracy_metrics=pd.concat(
            [forecast_accuracy_metrics, forecast_accuracy_enriched], ignore_index=True, sort=False
        ),
        kpi_summary=kpi_summary,
    )
