"""Data utilities and feature engineering for supply chain analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "region",
    "product_id",
    "supplier_id",
    "sales_qty",
    "units_ordered",
    "units_fulfilled",
    "inventory_level",
    "lead_time_days",
    "delivery_delay_days",
    "transport_cost",
    "defect_rate",
    "is_disruption_event",
}


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration used for synthetic dataset generation."""

    start_date: str = "2023-01-01"
    periods: int = 730
    random_state: int = 42
    regions: tuple[str, ...] = ("North", "South", "East", "West")
    products: tuple[str, ...] = ("Widget_A", "Widget_B", "Widget_C", "Widget_D", "Widget_E")
    suppliers: tuple[str, ...] = ("SUP-01", "SUP-02", "SUP-03", "SUP-04", "SUP-05", "SUP-06")


def _seasonality(day_idx: np.ndarray, period: int, strength: float, phase: float = 0.0) -> np.ndarray:
    return 1.0 + strength * np.sin((2.0 * np.pi * day_idx / period) + phase)


def generate_synthetic_supply_chain_data(config: SyntheticDataConfig | None = None) -> pd.DataFrame:
    """
    Generate a realistic historical supply chain dataset.

    The generated frame includes demand, inventory, fulfillment, logistics and quality signals
    needed for forecasting, anomaly detection and supplier risk scoring.
    """

    cfg = config or SyntheticDataConfig()
    rng = np.random.default_rng(cfg.random_state)

    dates = pd.date_range(cfg.start_date, periods=cfg.periods, freq="D")
    day_index = np.arange(cfg.periods)

    region_effect = {region: effect for region, effect in zip(cfg.regions, [1.05, 0.95, 1.0, 1.08])}
    product_base = {
        product: base for product, base in zip(cfg.products, [120.0, 95.0, 80.0, 110.0, 65.0])
    }
    supplier_lead_base = {
        supplier: lead for supplier, lead in zip(cfg.suppliers, [5.0, 6.5, 7.2, 4.6, 8.0, 5.5])
    }
    supplier_defect_base = {
        supplier: defect for supplier, defect in zip(cfg.suppliers, [0.012, 0.018, 0.025, 0.011, 0.03, 0.016])
    }

    rows: list[dict[str, float | str | int | pd.Timestamp]] = []
    for region in cfg.regions:
        for product in cfg.products:
            supplier_rotation = rng.choice(cfg.suppliers, size=cfg.periods, replace=True)
            weekly = _seasonality(day_index, period=7, strength=0.09, phase=rng.uniform(0, 1))
            yearly = _seasonality(day_index, period=365, strength=0.18, phase=rng.uniform(0, 2))
            trend = 1.0 + (day_index * rng.uniform(0.0002, 0.0007))
            promo_uplift = 1.0 + (rng.random(cfg.periods) < 0.05) * rng.uniform(0.08, 0.25, cfg.periods)
            base_series = product_base[product] * region_effect[region] * weekly * yearly * trend * promo_uplift
            noise = rng.normal(loc=0.0, scale=10.0, size=cfg.periods)
            demand = np.maximum(1.0, base_series + noise)

            disruptions = (rng.random(cfg.periods) < 0.025).astype(int)
            disruption_scale = np.where(disruptions == 1, rng.uniform(0.35, 0.75, cfg.periods), 1.0)
            demand = demand * disruption_scale

            ordered = demand * rng.uniform(1.04, 1.18, cfg.periods)
            lead_time = np.array([supplier_lead_base[s] for s in supplier_rotation]) + rng.normal(0, 0.7, cfg.periods)
            delay = rng.normal(0.0, 0.9, cfg.periods) + disruptions * rng.uniform(1.5, 6.0, cfg.periods)
            delay = np.maximum(-1.0, delay)

            inventory = demand * rng.uniform(0.75, 1.35, cfg.periods) + rng.normal(8.0, 4.0, cfg.periods)
            inventory = np.maximum(0.0, inventory)
            fulfilled = np.minimum(ordered, inventory + demand * rng.uniform(0.05, 0.3, cfg.periods))
            fulfilled = np.maximum(0.0, fulfilled)

            transport = (
                28.0
                + demand * rng.uniform(0.02, 0.05, cfg.periods)
                + np.maximum(delay, 0.0) * rng.uniform(1.5, 4.2, cfg.periods)
                + disruptions * rng.uniform(10.0, 28.0, cfg.periods)
            )
            defect = np.array([supplier_defect_base[s] for s in supplier_rotation]) + rng.normal(0.0, 0.004, cfg.periods)
            defect = np.clip(defect, 0.0, 0.15)

            for idx, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "region": region,
                        "product_id": product,
                        "supplier_id": supplier_rotation[idx],
                        "sales_qty": float(demand[idx]),
                        "units_ordered": float(ordered[idx]),
                        "units_fulfilled": float(fulfilled[idx]),
                        "inventory_level": float(inventory[idx]),
                        "lead_time_days": float(max(1.0, lead_time[idx])),
                        "delivery_delay_days": float(delay[idx]),
                        "transport_cost": float(max(1.0, transport[idx])),
                        "defect_rate": float(defect[idx]),
                        "is_disruption_event": int(disruptions[idx]),
                    }
                )

    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return frame


def _ensure_schema(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    checked = frame.copy()
    checked["date"] = pd.to_datetime(checked["date"])

    numeric_cols = [
        "sales_qty",
        "units_ordered",
        "units_fulfilled",
        "inventory_level",
        "lead_time_days",
        "delivery_delay_days",
        "transport_cost",
        "defect_rate",
    ]
    for col in numeric_cols:
        checked[col] = pd.to_numeric(checked[col], errors="coerce")
    checked["is_disruption_event"] = pd.to_numeric(checked["is_disruption_event"], errors="coerce").fillna(0).astype(int)
    checked = checked.dropna(subset=["date", "region", "product_id", "supplier_id", "sales_qty"])
    return checked.sort_values("date").reset_index(drop=True)


def load_supply_chain_data(path: str) -> pd.DataFrame:
    """Load and validate a user-supplied CSV dataset."""

    frame = pd.read_csv(path)
    return _ensure_schema(frame)


def add_forecasting_features(
    frame: pd.DataFrame,
    group_cols: Iterable[str] = ("region", "product_id"),
    target_col: str = "sales_qty",
    lags: Iterable[int] = (1, 7, 14, 28),
    rolling_windows: Iterable[int] = (7, 14, 28),
) -> pd.DataFrame:
    """
    Add lag, rolling and seasonality features required by the forecasting engine.
    """

    if target_col not in frame.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    if "date" not in frame.columns:
        raise ValueError("Input dataframe must include a 'date' column.")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    group_cols = list(group_cols)
    data = data.sort_values(group_cols + ["date"]).reset_index(drop=True)

    grouped = data.groupby(group_cols, group_keys=False)
    for lag in lags:
        data[f"{target_col}_lag_{lag}"] = grouped[target_col].shift(lag)

    for window in rolling_windows:
        data[f"{target_col}_roll_mean_{window}"] = grouped[target_col].shift(1).rolling(window=window).mean()
        data[f"{target_col}_roll_std_{window}"] = grouped[target_col].shift(1).rolling(window=window).std()

    day_of_week = data["date"].dt.dayofweek
    month = data["date"].dt.month
    day_of_year = data["date"].dt.dayofyear
    data["day_of_week"] = day_of_week
    data["month"] = month
    data["quarter"] = data["date"].dt.quarter
    data["is_weekend"] = day_of_week.isin([5, 6]).astype(int)
    data["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    data["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)
    data["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    data["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    return data
