# GenAI-Powered Supply Chain Decision Support System

End-to-end AI-driven supply chain intelligence platform that combines:

1. **Demand forecasting**
2. **Supply chain anomaly detection**
3. **Supplier risk scoring**
4. **GenAI copilot (RAG with embeddings + LLM)**
5. **Dashboard-ready KPI exports for Power BI / Tableau**

---

## What this system does

### 1) Demand Forecasting Engine

Uses historical supply chain sales data and feature engineering:

- Lag features (1, 7, 14, 28)
- Rolling mean/std (7, 14, 28)
- Seasonality encoding (sin/cos for day-of-week, month, day-of-year)

Models included:

- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression

Evaluation:

- Holdout RMSE
- Time-series cross-validation RMSE
- Bias-variance proxies:
  - Bias proxy: mean signed test error
  - Variance proxy: std-dev of CV RMSE

### 2) Anomaly Detection in Operations

Detects:

- Sudden drop in sales
- Delivery delays
- Cost spikes

Methods:

- Isolation Forest
- DBSCAN
- Z-score statistical flags (SciPy)

### 3) Supplier Risk Scoring

Builds supplier-level risk score using weighted operational factors:

- Late delivery behavior
- Average delay
- Defect rate
- Cost volatility
- Lead-time variability
- Disruption rate
- Fulfillment risk

Outputs:

- `supplier_risk_score` (0-100)
- Risk tier (`low`, `medium`, `high`, `critical`)

### 4) GenAI Supply Chain Copilot (RAG)

Builds a retrieval-augmented generation pipeline:

- Embeds KPI/anomaly/risk summaries
- Uses vector similarity search
- Answers natural language questions

Sample questions:

- "Why did Region West experience stockouts?"
- "Which supplier has highest disruption risk?"
- "Explain forecast error trend in Q4."

LLM behavior:

- Uses OpenAI if `OPENAI_API_KEY` is set
- Falls back to local rule-based natural-language summarization otherwise

### 5) Dashboard Outputs (Power BI / Tableau)

The pipeline exports dashboard-ready CSVs:

- `demand_forecast_vs_actual.csv`
- `stockout_probability.csv`
- `supplier_risk_scores.csv`
- `anomaly_alerts.csv`
- `forecast_accuracy_metrics.csv`
- `kpi_summary.csv`

KPIs covered:

- Service level %
- Fill rate
- Lead time variability
- Forecast error %
- Stockout probability
- High-risk supplier count

---

## Project structure

```text
.
├── dashboard_data
│   ├── demand_forecast_vs_actual.csv
│   ├── stockout_probability.csv
│   ├── supplier_risk_scores.csv
│   ├── anomaly_alerts.csv
│   ├── forecast_accuracy_metrics.csv
│   └── kpi_summary.csv
├── src/supply_chain_ai
│   ├── data.py
│   ├── forecasting.py
│   ├── anomaly.py
│   ├── risk.py
│   ├── kpi.py
│   ├── rag.py
│   └── pipeline.py
├── scripts
│   ├── generate_sample_data.py
│   ├── run_pipeline.py
│   └── ask_copilot.py
├── docs
│   └── powerbi_tableau_dashboard_guide.md
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run end-to-end pipeline

### Option A: Run with generated synthetic data

```bash
python scripts/run_pipeline.py --output-dir outputs
```

### Option B: Run with your own historical dataset

Your CSV must contain:

- `date`, `region`, `product_id`, `supplier_id`
- `sales_qty`, `units_ordered`, `units_fulfilled`, `inventory_level`
- `lead_time_days`, `delivery_delay_days`, `transport_cost`
- `defect_rate`, `is_disruption_event`

```bash
python scripts/run_pipeline.py --input-csv path/to/data.csv --output-dir outputs
```

### Option C: Generate + publish Power BI CSVs into tracked folder

```bash
bash scripts/publish_dashboard_data.sh
```

---

## Ask the GenAI Copilot

After pipeline run:

```bash
python scripts/ask_copilot.py \
  --index-path outputs/copilot/embedding_index.pkl \
  --question "Why did Region West experience stockouts?"
```

---

## Outputs

### Core artifacts

- `outputs/historical_supply_chain_data.csv`
- `outputs/models/best_demand_model.joblib`
- `outputs/models/forecast_model_benchmark.csv`
- `outputs/run_metadata.json`

### Dashboard artifacts

- `outputs/dashboard/*.csv`

### GitHub-tracked dashboard dataset folder

- `dashboard_data/*.csv`
- This folder is intended for direct Power BI Desktop import or GitHub raw URL imports.

### Copilot artifacts

- `outputs/copilot/embedding_index.pkl`
- `outputs/copilot/sample_responses.json`
- `outputs/copilot/llm_evaluation.csv`

---

## Dashboarding in Power BI / Tableau

See detailed mapping and chart recommendations in:

- `docs/powerbi_tableau_dashboard_guide.md`

---

## Notes

- Designed to be production-prototype ready and extensible.
- Works with Pandas, NumPy, SciPy, scikit-learn.
- RAG layer can be upgraded to managed vector DBs and enterprise LLM endpoints.