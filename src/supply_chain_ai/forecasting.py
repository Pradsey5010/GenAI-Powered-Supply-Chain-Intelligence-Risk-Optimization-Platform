"""Demand forecasting engine with model comparison and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import add_forecasting_features


@dataclass
class ForecastingResult:
    best_model_name: str
    model_metrics: pd.DataFrame
    forecast_frame: pd.DataFrame
    fitted_models: dict[str, Pipeline]


class DemandForecaster:
    """Train and evaluate multiple forecasting models."""

    def __init__(
        self,
        target_col: str = "sales_qty",
        date_col: str = "date",
        cv_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.target_col = target_col
        self.date_col = date_col
        self.cv_splits = cv_splits
        self.random_state = random_state

    def _build_models(self) -> dict[str, Any]:
        return {
            "random_forest": RandomForestRegressor(
                n_estimators=400,
                max_depth=18,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=350,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
            ),
            "ridge": Ridge(alpha=1.0),
        }

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = add_forecasting_features(frame=frame, target_col=self.target_col)
        modeling_frame = enriched.dropna().reset_index(drop=True)
        if modeling_frame.empty:
            raise ValueError("Feature engineering removed all rows. Provide more historical data.")
        return modeling_frame

    def _time_split(self, frame: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        unique_dates = np.array(sorted(frame[self.date_col].dt.normalize().unique()))
        if len(unique_dates) < 20:
            raise ValueError("Not enough unique dates to run a robust train/test time split.")
        split_idx = int(len(unique_dates) * (1 - test_size))
        split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
        split_date = unique_dates[split_idx]
        train_frame = frame[frame[self.date_col] < split_date].copy()
        test_frame = frame[frame[self.date_col] >= split_date].copy()
        if train_frame.empty or test_frame.empty:
            raise ValueError("Time split produced empty train or test partitions.")
        return train_frame, test_frame

    def train_and_evaluate(
        self,
        frame: pd.DataFrame,
        test_size: float = 0.2,
    ) -> ForecastingResult:
        """
        Train forecasting models and return holdout predictions plus diagnostics.

        Bias-variance analysis is reported using practical proxies:
        - Bias proxy: mean signed prediction error on the test set.
        - Variance proxy: standard deviation of CV validation RMSE.
        """

        modeling_frame = self._prepare_frame(frame)
        train_frame, test_frame = self._time_split(modeling_frame, test_size=test_size)

        protected_cols = {self.target_col, self.date_col}
        features = [col for col in modeling_frame.columns if col not in protected_cols]
        categorical_features = [
            col
            for col in features
            if str(modeling_frame[col].dtype) in {"object", "category"} or col in {"region", "product_id", "supplier_id"}
        ]
        numeric_features = [col for col in features if col not in categorical_features]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("num", StandardScaler(), numeric_features),
            ]
        )

        X_train = train_frame[features]
        y_train = train_frame[self.target_col]
        X_test = test_frame[features]
        y_test = test_frame[self.target_col]

        tscv_splits = min(self.cv_splits, max(2, len(train_frame[self.date_col].dt.normalize().unique()) - 1))
        tscv = TimeSeriesSplit(n_splits=tscv_splits)

        model_metrics: list[dict[str, float | str]] = []
        predictions_by_model: dict[str, np.ndarray] = {}
        fitted_models: dict[str, Pipeline] = {}

        for model_name, estimator in self._build_models().items():
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            cv = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
                return_train_score=True,
                n_jobs=-1,
            )
            train_rmse = -cv["train_score"]
            cv_rmse = -cv["test_score"]

            pipeline.fit(X_train, y_train)
            test_pred = pipeline.predict(X_test)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            bias_proxy = float(np.mean(y_test - test_pred))
            variance_proxy = float(np.std(cv_rmse))
            generalization_gap = float(np.mean(cv_rmse) - np.mean(train_rmse))

            fitted_models[model_name] = pipeline
            predictions_by_model[model_name] = test_pred

            model_metrics.append(
                {
                    "model": model_name,
                    "test_rmse": test_rmse,
                    "cv_rmse_mean": float(np.mean(cv_rmse)),
                    "cv_rmse_std": float(np.std(cv_rmse)),
                    "train_rmse_mean": float(np.mean(train_rmse)),
                    "bias_proxy_mean_error": bias_proxy,
                    "variance_proxy_cv_rmse_std": variance_proxy,
                    "generalization_gap_rmse": generalization_gap,
                }
            )

        metrics_df = pd.DataFrame(model_metrics).sort_values("test_rmse").reset_index(drop=True)
        best_model_name = str(metrics_df.iloc[0]["model"])
        best_pred = predictions_by_model[best_model_name]

        forecast_frame = test_frame[
            [self.date_col, "region", "product_id", "supplier_id", self.target_col, "inventory_level", "units_ordered", "units_fulfilled"]
        ].copy()
        forecast_frame = forecast_frame.rename(columns={self.target_col: "actual_sales"})
        forecast_frame["forecast_sales"] = best_pred
        forecast_frame["forecast_error"] = forecast_frame["actual_sales"] - forecast_frame["forecast_sales"]
        forecast_frame["forecast_error_pct"] = (
            forecast_frame["forecast_error"] / np.maximum(forecast_frame["actual_sales"], 1e-6) * 100.0
        )
        forecast_frame["best_model"] = best_model_name

        return ForecastingResult(
            best_model_name=best_model_name,
            model_metrics=metrics_df,
            forecast_frame=forecast_frame.reset_index(drop=True),
            fitted_models=fitted_models,
        )
