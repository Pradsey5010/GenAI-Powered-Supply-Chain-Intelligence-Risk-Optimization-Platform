from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from supply_chain_ai.pipeline import SupplyChainDecisionSupportSystem


def test_pipeline_generates_dashboard_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    system = SupplyChainDecisionSupportSystem(random_state=42)
    results = system.run(output_dir=output_dir, synthetic_periods=120, build_copilot=False)

    dashboard_dir = Path(results["dashboard_dir"])
    expected = [
        "demand_forecast_vs_actual.csv",
        "stockout_probability.csv",
        "supplier_risk_scores.csv",
        "anomaly_alerts.csv",
        "forecast_accuracy_metrics.csv",
        "kpi_summary.csv",
    ]
    for name in expected:
        assert (dashboard_dir / name).exists(), f"Missing dashboard export: {name}"

    kpi = pd.read_csv(dashboard_dir / "kpi_summary.csv")
    assert not kpi.empty
    assert {"kpi", "value"}.issubset(set(kpi.columns))
