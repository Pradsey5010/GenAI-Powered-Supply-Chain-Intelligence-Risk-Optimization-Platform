"""End-to-end orchestration for the AI supply chain decision support system."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .anomaly import SupplyChainAnomalyDetector
from .data import SyntheticDataConfig, generate_synthetic_supply_chain_data, load_supply_chain_data
from .forecasting import DemandForecaster
from .kpi import DashboardTables, build_dashboard_tables
from .rag import SupplyChainCopilot
from .risk import SupplierRiskScorer


@dataclass
class PipelineOutputs:
    root_dir: Path
    dashboard_dir: Path
    model_dir: Path
    copilot_dir: Path


class SupplyChainDecisionSupportSystem:
    """Build demand, risk and anomaly insights for decision support."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.forecaster = DemandForecaster(random_state=random_state)
        self.anomaly_detector = SupplyChainAnomalyDetector(random_state=random_state)
        self.risk_scorer = SupplierRiskScorer()

    @staticmethod
    def _prepare_output_dirs(output_dir: str | Path) -> PipelineOutputs:
        root = Path(output_dir)
        dashboard = root / "dashboard"
        model_dir = root / "models"
        copilot = root / "copilot"
        for path in [root, dashboard, model_dir, copilot]:
            path.mkdir(parents=True, exist_ok=True)
        return PipelineOutputs(root_dir=root, dashboard_dir=dashboard, model_dir=model_dir, copilot_dir=copilot)

    @staticmethod
    def _persist_dashboard_tables(tables: DashboardTables, dashboard_dir: Path) -> None:
        tables.demand_forecast_vs_actual.to_csv(dashboard_dir / "demand_forecast_vs_actual.csv", index=False)
        tables.stockout_probability.to_csv(dashboard_dir / "stockout_probability.csv", index=False)
        tables.supplier_risk_scores.to_csv(dashboard_dir / "supplier_risk_scores.csv", index=False)
        tables.anomaly_alerts.to_csv(dashboard_dir / "anomaly_alerts.csv", index=False)
        tables.forecast_accuracy_metrics.to_csv(dashboard_dir / "forecast_accuracy_metrics.csv", index=False)
        tables.kpi_summary.to_csv(dashboard_dir / "kpi_summary.csv", index=False)

    def run(
        self,
        input_csv: str | None = None,
        output_dir: str | Path = "outputs",
        synthetic_periods: int = 730,
        build_copilot: bool = True,
    ) -> dict[str, Any]:
        outputs = self._prepare_output_dirs(output_dir=output_dir)

        if input_csv:
            base_frame = load_supply_chain_data(input_csv)
        else:
            base_frame = generate_synthetic_supply_chain_data(
                SyntheticDataConfig(periods=synthetic_periods, random_state=self.random_state)
            )

        base_frame.to_csv(outputs.root_dir / "historical_supply_chain_data.csv", index=False)

        forecasting_result = self.forecaster.train_and_evaluate(base_frame)
        anomalies = self.anomaly_detector.detect(base_frame)
        supplier_scores = self.risk_scorer.score(base_frame)
        tables = build_dashboard_tables(
            base_frame=base_frame,
            forecast_frame=forecasting_result.forecast_frame,
            anomalies_frame=anomalies,
            supplier_risk_frame=supplier_scores,
            forecast_model_metrics=forecasting_result.model_metrics,
        )
        self._persist_dashboard_tables(tables=tables, dashboard_dir=outputs.dashboard_dir)

        # Save best forecasting model for reuse.
        best_model_pipeline = forecasting_result.fitted_models[forecasting_result.best_model_name]
        joblib.dump(best_model_pipeline, outputs.model_dir / "best_demand_model.joblib")
        forecasting_result.model_metrics.to_csv(outputs.model_dir / "forecast_model_benchmark.csv", index=False)

        metadata = {
            "best_forecasting_model": forecasting_result.best_model_name,
            "records_processed": int(len(base_frame)),
            "alerts_detected": int(len(anomalies)),
            "suppliers_scored": int(len(supplier_scores)),
        }
        with (outputs.root_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        sample_responses: list[dict[str, Any]] = []
        if build_copilot:
            copilot = SupplyChainCopilot.from_tables(
                kpi_summary=tables.kpi_summary,
                stockout_probability=tables.stockout_probability,
                supplier_risk_scores=tables.supplier_risk_scores,
                anomaly_alerts=tables.anomaly_alerts,
                forecast_model_metrics=forecasting_result.model_metrics,
            )
            copilot.index.save(outputs.copilot_dir / "embedding_index.pkl")

            qa_set = [
                {"question": "Why did Region West experience stockouts?", "expected_keywords": ["west", "stockout"]},
                {
                    "question": "Which supplier has highest disruption risk?",
                    "expected_keywords": ["supplier", "risk", "critical"],
                },
                {
                    "question": "Explain forecast error trend in Q4.",
                    "expected_keywords": ["forecast", "rmse", "bias"],
                },
            ]
            eval_df = copilot.evaluate(qa_set=qa_set, top_k=4)
            eval_df.to_csv(outputs.copilot_dir / "llm_evaluation.csv", index=False)

            for item in qa_set:
                sample_responses.append(copilot.ask(item["question"]))

            with (outputs.copilot_dir / "sample_responses.json").open("w", encoding="utf-8") as handle:
                json.dump(sample_responses, handle, indent=2)

        return {
            "output_dir": str(outputs.root_dir),
            "dashboard_dir": str(outputs.dashboard_dir),
            "model_dir": str(outputs.model_dir),
            "copilot_dir": str(outputs.copilot_dir),
            "metadata": metadata,
            "copilot_samples": sample_responses,
        }
