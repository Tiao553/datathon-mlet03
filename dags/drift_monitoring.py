"""
Drift Monitoring DAG - Weekly Health Check with Evidently AI

This DAG runs weekly to monitor data and model drift using Evidently AI.
It checks for:
1. Data Distribution Drift (using Evidently's statistical tests)
2. Data Quality Issues
3. Feature-level drift detection

Reports are saved as HTML and key metrics are logged to MLflow.

Author: MLOps Team
Date: 2025-12-15
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.drift_detection import (
    load_training_reference_data,
    load_production_data,
    calculate_drift_report,
    check_drift_threshold
)

# Default arguments for the DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'drift_monitoring_weekly',
    default_args=default_args,
    description='Weekly drift monitoring using Evidently AI',
    schedule_interval='@daily',  # Runs every day at midnight
    start_date=datetime(2025, 12, 15),
    catchup=False,
    tags=['monitoring', 'drift', 'mlops', 'evidently'],
)


def check_data_drift(**context):
    """
    Check for data drift using Evidently AI.
    
    This function:
    1. Loads reference (training) data
    2. Loads recent production data (last 7 days)
    3. Runs Evidently drift detection
    4. Generates HTML report
    5. Returns drift metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Evidently AI data drift check...")
    
    try:
        # Load data
        logger.info("Loading reference data...")
        reference_data = load_training_reference_data()
        
        logger.info("Loading production data (last 7 days)...")
        current_data = load_production_data(days=7)
        
        # Check if we have data
        if reference_data.empty or current_data.empty:
            logger.warning("Insufficient data for drift detection. Skipping...")
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "dataset_drift": False,
                "drift_share": 0.0
            }
        
        # Calculate drift
        logger.info("Calculating drift metrics with Evidently...")
        report, drift_summary = calculate_drift_report(reference_data, current_data)
        
        logger.info(f"Drift detection completed!")
        logger.info(f"Dataset drift detected: {drift_summary['dataset_drift']}")
        logger.info(f"Drift share: {drift_summary['drift_share']:.2%}")
        logger.info(f"Drifted columns: {drift_summary['number_of_drifted_columns']}")
        logger.info(f"Report saved to: {drift_summary['report_path']}")
        
        # Add status
        drift_summary['status'] = 'ok'
        
        return drift_summary
        
    except Exception as e:
        logger.error(f"Error during drift detection: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "dataset_drift": None,
            "drift_share": None
        }


def check_model_performance(**context):
    """
    Check model performance metrics.
    
    In a full implementation, this would:
    1. Load recent predictions + ground truth (if available)
    2. Calculate metrics (AUC, Precision, Recall)
    3. Compare with baseline
    4. Alert if degradation > threshold
    
    For now, this is a placeholder that returns mock metrics.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model performance check...")
    
    # TODO: Implement actual performance monitoring
    # This requires:
    # - Storing predictions with timestamps
    # - Collecting ground truth labels (feedback loop)
    # - Calculating metrics over time windows
    
    logger.info("Model performance check completed (placeholder).")
    return {"status": "ok", "auc": 0.85, "note": "placeholder_implementation"}


def generate_drift_report(**context):
    """
    Generate a summary report and send alerts if needed.
    """
    logger = logging.getLogger(__name__)
    
    # Pull results from previous tasks
    ti = context['ti']
    data_drift_result = ti.xcom_pull(task_ids='check_data_drift')
    model_perf_result = ti.xcom_pull(task_ids='check_model_performance')
    
    # Check if drift exceeds threshold
    alert_needed = False
    if data_drift_result.get('status') == 'ok':
        alert_needed = check_drift_threshold(data_drift_result, threshold=0.15)
    
    # Build report
    report = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║       Weekly Drift Monitoring Report (Evidently AI)     ║
    ╚══════════════════════════════════════════════════════════╝
    
    📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    📊 DATA DRIFT STATUS
    ────────────────────────────────────────────────────────────
    Status: {data_drift_result.get('status', 'unknown').upper()}
    Dataset Drift Detected: {data_drift_result.get('dataset_drift', 'N/A')}
    Drift Share: {data_drift_result.get('drift_share', 0):.2%}
    Drifted Columns: {data_drift_result.get('number_of_drifted_columns', 'N/A')}
    Report: {data_drift_result.get('report_path', 'N/A')}
    
    🎯 MODEL PERFORMANCE
    ────────────────────────────────────────────────────────────
    Status: {model_perf_result.get('status', 'unknown').upper()}
    Current AUC: {model_perf_result.get('auc', 'N/A')}
    
    🚨 ALERT STATUS
    ────────────────────────────────────────────────────────────
    Alert Needed: {'YES - Drift exceeds 15% threshold!' if alert_needed else 'No'}
    
    ════════════════════════════════════════════════════════════
    """
    
    logger.info(report)
    
    if alert_needed:
        logger.warning("⚠️  DRIFT ALERT: Drift threshold exceeded!")
        # TODO: Send alert via Slack/Email
        # send_slack_message(report)
    
    return {
        "report": report,
        "alert_needed": alert_needed,
        "timestamp": datetime.now().isoformat()
    }


# Task definitions
task_data_drift = PythonOperator(
    task_id='check_data_drift',
    python_callable=check_data_drift,
    dag=dag,
)

task_model_performance = PythonOperator(
    task_id='check_model_performance',
    python_callable=check_model_performance,
    dag=dag,
)

task_generate_report = PythonOperator(
    task_id='generate_drift_report',
    python_callable=generate_drift_report,
    dag=dag,
)

# Task dependencies
[task_data_drift, task_model_performance] >> task_generate_report
