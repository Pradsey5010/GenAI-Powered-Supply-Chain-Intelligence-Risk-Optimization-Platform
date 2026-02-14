# GenAI-Powered Supply Chain Intelligence & Risk Optimization Platform

An end-to-end AI-driven decision support system for supply chain teams to:

- Forecast demand more accurately
- Detect operational anomalies early
- Score supplier risk with explainable factors
- Generate natural language business insights using RAG (LLM + embeddings)
- Feed executive dashboards in Power BI / Tableau

---

## 1) Project Purpose

Modern supply chains generate large volumes of operational data, but decision-makers often struggle to convert that data into timely actions.

This project provides a unified analytics + GenAI system that transforms raw supply chain data into:

1. **Predictive insights** (future demand)
2. **Diagnostic insights** (what went wrong and where)
3. **Risk insights** (which suppliers are vulnerable)
4. **Prescriptive guidance** (what to do next)

---

## 2) How the System Works (Step-by-Step)

### Step 1: Data ingestion and preparation

The system reads historical supply chain data (or generates synthetic data) with fields such as:

- sales quantity
- orders and fulfillment
- inventory
- delivery delays
- lead time
- transport cost
- supplier quality/disruption signals

Then it validates schema and performs preprocessing.

### Step 2: Demand forecasting engine

Feature engineering is applied:

- Lag features: `t-1`, `t-7`, `t-14`, `t-28`
- Rolling statistics: moving mean/std windows
- Seasonality encoding: day/week/month cyclical signals

Models are trained and benchmarked:

- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression

Evaluation includes:

- RMSE on holdout test data
- Time-series cross-validation
- Bias/variance proxy diagnostics

### Step 3: Anomaly detection engine

Operational anomalies are detected using:

- Isolation Forest
- DBSCAN
- Z-score statistical thresholds (SciPy)

Target events include:

- sudden sales drops
- delay spikes
- cost spikes

Each alert includes severity and reason codes.

### Step 4: Supplier risk scoring

Each supplier receives a composite risk score (0-100) and tier (`low/medium/high/critical`) based on:

- on-time delivery performance
- average delay
- defect rate
- transport cost volatility
- lead-time variability
- disruption frequency
- fulfillment risk

### Step 5: KPI generation and dashboard exports

The system computes and exports key business KPIs and tables for BI tools:

- demand forecast vs actual
- stockout probability
- supplier risk ranking
- anomaly alerts
- forecast accuracy metrics
- executive KPI summary

### Step 6: GenAI supply chain copilot (RAG)

A RAG layer is built from KPI/risk/anomaly/model summaries:

1. Text chunks are embedded
2. Similar context is retrieved for a user question
3. LLM (or local fallback) generates concise explanation + recommendations

Example questions:

- "Why did Region West experience stockouts?"
- "Which supplier has highest disruption risk?"
- "Explain forecast error trend in Q4."

---

## 3) Tech Stack

### Core data and ML

- Python 3
- Pandas
- NumPy
- SciPy
- scikit-learn
- joblib

### GenAI and retrieval

- TF-IDF embeddings (vector search prototype)
- cosine similarity retrieval
- OpenAI API integration (optional via `OPENAI_API_KEY`)

### Visualization

- Power BI
- Tableau

### Engineering and testing

- pytest
- Git/GitHub branch workflow

---

## 4) Key Features

- End-to-end pipeline orchestration
- Multi-model demand forecasting with performance comparison
- Statistical + unsupervised anomaly detection
- Explainable supplier risk scoring
- Dashboard-ready data products
- Natural-language copilot with evidence-backed retrieval
- Lightweight LLM evaluation support

---

## 5) High-Level Architecture

```text
                    +----------------------+
                    | Historical SC Data   |
                    | (CSV / Warehouse)    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Data Validation &    |
                    | Feature Engineering  |
                    +----------+-----------+
                               |
        +----------------------+----------------------+
        |                                             |
        v                                             v
+-------------------------+                +-------------------------+
| Demand Forecast Engine  |                | Anomaly Detection       |
| RF / GB / Ridge         |                | IF / DBSCAN / Z-score   |
+------------+------------+                +------------+------------+
             |                                          |
             +------------------+-----------------------+
                                |
                                v
                      +-----------------------+
                      | Supplier Risk Scorer  |
                      +-----------+-----------+
                                  |
                                  v
                      +-----------------------+
                      | KPI & Dashboard Data  |
                      | (Power BI / Tableau)  |
                      +-----------+-----------+
                                  |
                                  v
                      +-----------------------+
                      | RAG Copilot           |
                      | Embeddings + LLM      |
                      +-----------------------+
```

---

## 6) Dashboard KPIs

Primary KPIs produced:

- Service level %
- Fill rate %
- Lead time variability %
- Forecast error %
- Stockout probability %
- High-risk supplier count

---

## 7) Future Scope

- Replace TF-IDF with dense embedding models and vector databases (FAISS/Pinecone/Weaviate)
- Add probabilistic forecasting and confidence intervals
- Integrate near-real-time streaming anomaly detection
- Add optimization layer for inventory and replenishment decisions
- Expand explainability with SHAP and root-cause workflows
- Add MLOps: model registry, drift monitoring, automated retraining
- Add role-based access and enterprise authentication for production deployments

---

## 8) How to Run / Access / Execute the Project

> This section describes the standard execution flow after code is available in the working branch/release.

### Prerequisites

- Python 3.10+
- pip
- Git
- (Optional) OpenAI API key for external LLM responses

### Setup

```bash
git clone <your-repo-url>
cd GenAI-Powered-Supply-Chain-Intelligence-Risk-Optimization-Platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run with generated sample data

```bash
python3 scripts/run_pipeline.py --output-dir outputs
```

### Run with your own historical CSV

```bash
python3 scripts/run_pipeline.py --input-csv path/to/your_data.csv --output-dir outputs
```

### Ask the GenAI copilot

```bash
python3 scripts/ask_copilot.py \
  --index-path outputs/copilot/embedding_index.pkl \
  --question "Which supplier has highest disruption risk?"
```

### Connect to Power BI / Tableau

Import CSVs from `outputs/dashboard/`:

- `demand_forecast_vs_actual.csv`
- `stockout_probability.csv`
- `supplier_risk_scores.csv`
- `anomaly_alerts.csv`
- `forecast_accuracy_metrics.csv`
- `kpi_summary.csv`

Build visuals from these datasets to create an executive dashboard.

---

## 9) Expected Output Artifacts

After successful execution, typical artifacts are:

- `outputs/historical_supply_chain_data.csv`
- `outputs/models/best_demand_model.joblib`
- `outputs/models/forecast_model_benchmark.csv`
- `outputs/dashboard/*.csv`
- `outputs/copilot/embedding_index.pkl`
- `outputs/copilot/sample_responses.json`
- `outputs/copilot/llm_evaluation.csv`
- `outputs/run_metadata.json`

---

## 10) Summary

This platform combines forecasting, anomaly intelligence, supplier risk modeling, and GenAI explainability into one decision support workflow for supply chain operations teams.