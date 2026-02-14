#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

python3 scripts/run_pipeline.py --output-dir outputs --no-copilot

mkdir -p dashboard_data
cp outputs/dashboard/demand_forecast_vs_actual.csv dashboard_data/
cp outputs/dashboard/stockout_probability.csv dashboard_data/
cp outputs/dashboard/supplier_risk_scores.csv dashboard_data/
cp outputs/dashboard/anomaly_alerts.csv dashboard_data/
cp outputs/dashboard/forecast_accuracy_metrics.csv dashboard_data/
cp outputs/dashboard/kpi_summary.csv dashboard_data/

echo "Dashboard CSVs published to: $ROOT_DIR/dashboard_data"
