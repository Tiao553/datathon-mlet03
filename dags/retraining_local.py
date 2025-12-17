from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
import os
import polars as pl
from pathlib import Path

# Define default args
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_new_jobs(**context):
    """
    Checks if there are new jobs in the curated zone compared to a tracking file.
    Returns True to proceed, False to skip.
    """
    try:
        curated_path = Path('/opt/airflow/data/curated/jobs.parquet')
        tracker_path = Path('/opt/airflow/monitoring/retraining_tracker.json')
        
        if not curated_path.exists():
            print("Curated jobs file not found. Skipping.")
            return False
            
        # Get current job IDs
        df = pl.read_parquet(curated_path)
        current_ids = set(df['codigo_vaga'].to_list())
        
        # Get previous IDs
        import json
        processed_ids = set()
        if tracker_path.exists():
            with open(tracker_path, 'r') as f:
                data = json.load(f)
                processed_ids = set(data.get('processed_job_ids', []))
        
        # Check for difference
        new_ids = current_ids - processed_ids
        
        if new_ids:
            print(f"Found {len(new_ids)} new jobs. Proceeding with retraining.")
            # Update tracker (optimistic update, or do it at end of DAG)
            # ideally we update ONLY after success, but for ShortCircuit simple logic:
            # We can just return True here and have a final task update the tracker.
            return True
        else:
            print("No new jobs found. Skipping retraining.")
            return False
            
    except Exception as e:
        print(f"Error checking for new jobs: {e}")
        return False # Fail safe? Or True to force retry? False to avoid noise.

def update_tracker(**context):
    """Updates the tracker file with the latest job IDs after successful training."""
    curated_path = Path('/opt/airflow/data/curated/jobs.parquet')
    tracker_path = Path('/opt/airflow/monitoring/retraining_tracker.json')
    
    if curated_path.exists():
        df = pl.read_parquet(curated_path)
        current_ids = df['codigo_vaga'].to_list()
        import json
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tracker_path, 'w') as f:
            json.dump({'processed_job_ids': current_ids, 'last_run': datetime.now().isoformat()}, f)

# Define DAG
with DAG(
    'retraining_local',
    default_args=default_args,
    description='Local Retraining Pipeline (Checks for new data)',
    schedule_interval='0 0 * * 0',  # Weekly: Every Sunday at midnight
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['mlops', 'retraining', 'local'],
) as dag:
    
    # Task 0: Check for new data
    check_new_data = ShortCircuitOperator(
        task_id='check_new_data',
        python_callable=check_new_jobs,
    )

    # Task 1: Update Feature Store (Extract & Transform)
    feature_engineering = BashOperator(
        task_id='feature_engineering',
        bash_command='python3 /opt/airflow/data_pipeline/main_feature_engineering.py',
    )

    # Task 2: Retrain Models
    train_models = BashOperator(
        task_id='train_models',
        bash_command='python3 /opt/airflow/data_pipeline/main_training.py',
    )
    
    # Task 3: Update Tracker
    update_tracker_task = ShortCircuitOperator(
        task_id='update_tracker',
        python_callable=update_tracker,
    )

    check_new_data >> feature_engineering >> train_models >> update_tracker_task
