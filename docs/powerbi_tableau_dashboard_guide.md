# Power BI / Tableau Executive Dashboard Guide

This guide maps generated pipeline outputs to dashboard visuals and KPIs.

## 1. Data sources

Load these files from `outputs/dashboard/`:

1. `demand_forecast_vs_actual.csv`
2. `stockout_probability.csv`
3. `supplier_risk_scores.csv`
4. `anomaly_alerts.csv`
5. `forecast_accuracy_metrics.csv`
6. `kpi_summary.csv`

## 2. Recommended dashboard pages

### Page A: Executive Overview

- KPI cards:
  - Service level %
  - Fill rate %
  - Forecast error %
  - Lead-time variability %
  - High-risk supplier count
- Trend chart:
  - Forecast vs Actual over time
- Alert card:
  - Total anomalies in current period

### Page B: Demand Forecast Performance

- Line chart:
  - Axis: `date`
  - Values: `actual_sales`, `forecast_sales`
- Heatmap:
  - `region` x `product_id` with average absolute forecast error
- Table:
  - Top products by forecast error %

### Page C: Supply Chain Risk & Anomalies

- Supplier risk bar chart:
  - Axis: `supplier_id`
  - Value: `supplier_risk_score`
  - Color by `risk_tier`
- Anomaly table:
  - `date`, `region`, `product_id`, `alert_severity`, `anomaly_reason`
- Stockout map/bar:
  - `region` vs `stockout_probability`

### Page D: Model Governance

- Model comparison table:
  - `model`, `test_rmse`, `cv_rmse_mean`, `cv_rmse_std`, `bias_proxy_mean_error`
- Bias-variance chart:
  - x-axis: model
  - y-axis: bias and variance proxies

## 3. KPI formulas (reference)

- Service level % = sum(units_fulfilled) / sum(units_ordered) * 100
- Fill rate % = mean(units_fulfilled / units_ordered) * 100
- Lead-time variability % = std(lead_time_days) / mean(lead_time_days) * 100
- Forecast error % = mean(abs((actual - forecast)/actual)) * 100
- Stockout probability % = mean(stockout_flag) * 100

## 4. Filters / slicers

Use slicers for:

- Date
- Region
- Product
- Supplier
- Risk tier
- Alert severity

## 5. Refresh workflow

1. Re-run pipeline:
   - `python scripts/run_pipeline.py --output-dir outputs`
2. Refresh data source in Power BI / Tableau.
3. Validate key cards against `kpi_summary.csv`.
