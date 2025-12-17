# Data Drift Detection Tools - Comparison

## Overview
Based on research, here are the top 3 open-source Python tools for data drift detection:

## 1. **Evidently AI** ⭐ (Recommended for this project)

**Strengths:**
- Comprehensive drift detection (data, target, prediction drift)
- 100+ built-in metrics
- Statistical tests: KS, Chi-Squared, Wasserstein, PSI
- **Excellent for text data** (embedding drift)
- Beautiful interactive dashboards (HTML, Jupyter, Streamlit)
- Easy MLOps integration (MLflow, Grafana, Prometheus)

**Best for:** Startups, small teams, diverse data types (tabular + text)

**Installation:**
```bash
pip install evidently
```

**Example Usage:**
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df)
report.save_html("drift_report.html")
```

---

## 2. **Deepchecks**

**Strengths:**
- Holistic validation (data integrity + model validation)
- Supports tabular + computer vision
- Automated test suites
- Domain classifier for multivariate drift

**Best for:** Rigorous testing, CV tasks, data quality focus

**Installation:**
```bash
pip install deepchecks
```

---

## 3. **Whylogs**

**Strengths:**
- Efficient data profiling (lightweight summaries)
- Scalable (batch + streaming)
- Enterprise-grade (WhyLabs platform)
- Privacy-preserving (no raw data storage)

**Best for:** Large-scale enterprise, real-time monitoring

**Installation:**
```bash
pip install whylogs
```

---

## Recommendation for This Project

**Use Evidently AI** because:
1. We work with **text data** (resumes, job descriptions) → Evidently has embedding drift detection
2. Easy integration with **MLflow** (already in our stack)
3. User-friendly dashboards for stakeholders
4. Open-source and lightweight

## Implementation Plan

1. Add `evidently` to `data_pipeline/requirements.txt`
2. Update `dags/drift_monitoring.py` to use Evidently
3. Generate weekly HTML reports and save to MLflow artifacts
4. Set up alerts if drift > threshold (PSI > 0.15)
