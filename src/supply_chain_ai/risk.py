"""Supplier risk scoring model for supply chain resilience."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SupplierRiskWeights:
    late_delivery: float = 0.24
    average_delay: float = 0.16
    defect_rate: float = 0.18
    cost_volatility: float = 0.14
    lead_time_variability: float = 0.1
    disruption_rate: float = 0.1
    fulfillment_risk: float = 0.08


class SupplierRiskScorer:
    """Calculate supplier risk using weighted operational metrics."""

    def __init__(self, weights: SupplierRiskWeights | None = None) -> None:
        self.weights = weights or SupplierRiskWeights()

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        min_val = float(series.min())
        max_val = float(series.max())
        if np.isclose(min_val, max_val):
            return pd.Series(np.full(len(series), 0.5), index=series.index)
        return (series - min_val) / (max_val - min_val)

    @staticmethod
    def _risk_tier(score: float) -> str:
        if score >= 75:
            return "critical"
        if score >= 50:
            return "high"
        if score >= 25:
            return "medium"
        return "low"

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        required = {
            "supplier_id",
            "delivery_delay_days",
            "defect_rate",
            "transport_cost",
            "lead_time_days",
            "units_ordered",
            "units_fulfilled",
            "is_disruption_event",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Supplier risk scoring missing columns: {', '.join(sorted(missing))}")

        by_supplier = frame.groupby("supplier_id", as_index=False).agg(
            total_orders=("units_ordered", "sum"),
            total_fulfilled=("units_fulfilled", "sum"),
            on_time_rate=("delivery_delay_days", lambda x: float((x <= 1).mean())),
            average_delay_days=("delivery_delay_days", lambda x: float(np.maximum(x, 0).mean())),
            defect_rate_mean=("defect_rate", "mean"),
            transport_cost_std=("transport_cost", "std"),
            transport_cost_mean=("transport_cost", "mean"),
            lead_time_std=("lead_time_days", "std"),
            disruption_rate=("is_disruption_event", "mean"),
        )
        by_supplier["transport_cost_std"] = by_supplier["transport_cost_std"].fillna(0.0)
        by_supplier["lead_time_std"] = by_supplier["lead_time_std"].fillna(0.0)
        by_supplier["cost_volatility"] = by_supplier["transport_cost_std"] / np.maximum(
            by_supplier["transport_cost_mean"], 1e-6
        )
        by_supplier["fulfillment_rate"] = by_supplier["total_fulfilled"] / np.maximum(by_supplier["total_orders"], 1e-6)

        by_supplier["late_delivery_risk"] = 1.0 - by_supplier["on_time_rate"]
        by_supplier["fulfillment_risk"] = 1.0 - by_supplier["fulfillment_rate"]

        by_supplier["late_delivery_norm"] = self._normalize(by_supplier["late_delivery_risk"])
        by_supplier["avg_delay_norm"] = self._normalize(by_supplier["average_delay_days"])
        by_supplier["defect_norm"] = self._normalize(by_supplier["defect_rate_mean"])
        by_supplier["cost_volatility_norm"] = self._normalize(by_supplier["cost_volatility"])
        by_supplier["lead_var_norm"] = self._normalize(by_supplier["lead_time_std"])
        by_supplier["disruption_norm"] = self._normalize(by_supplier["disruption_rate"])
        by_supplier["fulfillment_risk_norm"] = self._normalize(by_supplier["fulfillment_risk"])

        weighted = (
            by_supplier["late_delivery_norm"] * self.weights.late_delivery
            + by_supplier["avg_delay_norm"] * self.weights.average_delay
            + by_supplier["defect_norm"] * self.weights.defect_rate
            + by_supplier["cost_volatility_norm"] * self.weights.cost_volatility
            + by_supplier["lead_var_norm"] * self.weights.lead_time_variability
            + by_supplier["disruption_norm"] * self.weights.disruption_rate
            + by_supplier["fulfillment_risk_norm"] * self.weights.fulfillment_risk
        )

        by_supplier["supplier_risk_score"] = (weighted * 100).round(2)
        by_supplier["risk_tier"] = by_supplier["supplier_risk_score"].map(self._risk_tier)
        by_supplier = by_supplier.sort_values("supplier_risk_score", ascending=False).reset_index(drop=True)
        return by_supplier
