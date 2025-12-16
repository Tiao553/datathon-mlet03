# DAGs (Directed Acyclic Graphs)

This directory contains all Airflow DAG definitions for the MLOps platform.

## Current DAGs

### `drift_monitoring.py`
**Schedule:** Weekly (@weekly - every Sunday at midnight)
**Purpose:** Monitor data and model drift

**Tasks:**
1. `check_data_drift`: Calculate PSI (Population Stability Index) for feature distributions
2. `check_model_performance`: Evaluate model metrics against baseline
3. `generate_drift_report`: Aggregate results and send alerts

**Status:** Placeholder implementation. TODO: Add actual drift calculation logic.

## Adding New DAGs

1. Create a new Python file in this directory
2. Define your DAG using Airflow's DAG API
3. The file will be automatically picked up by Airflow (mounted via Docker volume)

## Testing DAGs Locally

```bash
# Start Airflow
cd infrastructure/local
docker-compose up -d airflow_webserver airflow_scheduler

# Access Airflow UI
open http://localhost:8080
# Login: admin / admin
```
