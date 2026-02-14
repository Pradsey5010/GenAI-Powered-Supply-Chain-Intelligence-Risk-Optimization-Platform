"""Unsupervised and statistical anomaly detection for supply chain operations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class SupplyChainAnomalyDetector:
    """Detect sales, logistics and cost anomalies using multiple methods."""

    def __init__(
        self,
        contamination: float = 0.03,
        dbscan_eps: float = 1.2,
        dbscan_min_samples: int = 20,
        zscore_threshold: float = 2.5,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.zscore_threshold = zscore_threshold
        self.random_state = random_state

    @staticmethod
    def _build_alert_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["sales_drop_zscore_flag"] == 1:
            reasons.append("sudden_sales_drop")
        if row["delay_zscore_flag"] == 1:
            reasons.append("delivery_delay_spike")
        if row["cost_zscore_flag"] == 1:
            reasons.append("transport_cost_spike")
        if row["isolation_forest_flag"] == 1:
            reasons.append("isolation_forest_outlier")
        if row["dbscan_flag"] == 1:
            reasons.append("dbscan_noise_point")
        return ", ".join(reasons) if reasons else "none"

    def detect(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        required = {"date", "region", "product_id", "sales_qty", "delivery_delay_days", "transport_cost", "lead_time_days"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Anomaly detection missing columns: {', '.join(sorted(missing))}")

        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values(["region", "product_id", "date"]).reset_index(drop=True)

        # Sudden sales drop detection using group-wise percentage change z-scores.
        data["sales_pct_change"] = data.groupby(["region", "product_id"])["sales_qty"].pct_change().fillna(0.0)
        sales_change_z = stats.zscore(data["sales_pct_change"].replace([np.inf, -np.inf], 0.0), nan_policy="omit")
        sales_change_z = np.nan_to_num(sales_change_z, nan=0.0)
        data["sales_change_z"] = sales_change_z
        data["sales_drop_zscore_flag"] = (
            (data["sales_pct_change"] < -0.15) & (data["sales_change_z"] < -self.zscore_threshold)
        ).astype(int)

        # Statistical z-score rules for delay and cost spikes.
        delay_z = stats.zscore(data["delivery_delay_days"].astype(float), nan_policy="omit")
        delay_z = np.nan_to_num(delay_z, nan=0.0)
        cost_z = stats.zscore(data["transport_cost"].astype(float), nan_policy="omit")
        cost_z = np.nan_to_num(cost_z, nan=0.0)
        data["delivery_delay_z"] = delay_z
        data["transport_cost_z"] = cost_z
        data["delay_zscore_flag"] = (data["delivery_delay_z"] > self.zscore_threshold).astype(int)
        data["cost_zscore_flag"] = (data["transport_cost_z"] > self.zscore_threshold).astype(int)

        # Unsupervised methods over operational signals.
        model_features = ["sales_qty", "delivery_delay_days", "transport_cost", "lead_time_days", "inventory_level"]
        for col in model_features:
            if col not in data.columns:
                data[col] = 0.0

        clean_matrix = data[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaled = StandardScaler().fit_transform(clean_matrix)

        isolation_forest = IsolationForest(
            n_estimators=250,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        iso_pred = isolation_forest.fit_predict(scaled)
        data["isolation_forest_flag"] = (iso_pred == -1).astype(int)

        min_samples = min(self.dbscan_min_samples, max(5, int(len(data) * 0.01)))
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=min_samples)
        db_pred = dbscan.fit_predict(scaled)
        data["dbscan_flag"] = (db_pred == -1).astype(int)

        data["anomaly_score"] = data[
            [
                "sales_drop_zscore_flag",
                "delay_zscore_flag",
                "cost_zscore_flag",
                "isolation_forest_flag",
                "dbscan_flag",
            ]
        ].sum(axis=1)
        data["alert_severity"] = pd.cut(
            data["anomaly_score"],
            bins=[-0.1, 0.9, 2.0, 5.1],
            labels=["none", "medium", "high"],
        ).astype(str)
        data["anomaly_reason"] = data.apply(self._build_alert_reason, axis=1)

        anomalies = data[data["anomaly_score"] > 0].copy()
        return anomalies.sort_values(["anomaly_score", "date"], ascending=[False, False]).reset_index(drop=True)
