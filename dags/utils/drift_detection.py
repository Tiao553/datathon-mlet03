"""
Drift Detection Utilities using Evidently AI

This module provides helper functions for detecting data drift
in production inference data compared to training data.
"""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, Tuple
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric


def load_training_reference_data(data_path: str = "/opt/airflow/data/curated") -> pd.DataFrame:
    """
    Load reference (training) data for drift comparison.
    
    Args:
        data_path: Path to curated data directory
        
    Returns:
        pandas DataFrame with training data
    """
    # Load from parquet (assuming we have a reference dataset)
    reference_path = Path(data_path) / "training_reference.parquet"
    
    if reference_path.exists():
        df = pl.read_parquet(reference_path).to_pandas()
    else:
        # Fallback: use a sample from the full dataset
        # In production, you should create a proper reference dataset
        print(f"Warning: Reference data not found at {reference_path}")
        print("Using fallback: loading sample from full dataset")
        # This is a placeholder - adjust based on your actual data structure
        df = pd.DataFrame()
    
    return df


def load_production_data(days: int = 7, data_path: str = "/opt/airflow/data") -> pd.DataFrame:
    """
    Load recent production inference data.
    
    Args:
        days: Number of days of recent data to load
        data_path: Path to data directory
        
    Returns:
        pandas DataFrame with production data
    """
    # In a real implementation, you would:
    # 1. Query your inference logs/database
    # 2. Filter by date range
    # 3. Return the data
    
    # Placeholder implementation
    print(f"Loading last {days} days of production data...")
    # This should be replaced with actual data loading logic
    df = pd.DataFrame()
    
    return df


def calculate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str = "/opt/airflow/logs/drift_reports"
) -> Tuple[Report, Dict]:
    """
    Calculate drift metrics using Evidently AI.
    
    Args:
        reference_data: Training/reference dataset
        current_data: Recent production dataset
        output_path: Where to save HTML reports
        
    Returns:
        Tuple of (Evidently Report object, drift summary dict)
    """
    # Create Evidently report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
    ])
    
    # Run the report
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save HTML report
    Path(output_path).mkdir(parents=True, exist_ok=True)
    report_file = Path(output_path) / f"drift_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(str(report_file))
    
    # Extract key metrics
    report_dict = report.as_dict()
    
    # Parse drift summary
    drift_summary = {
        "dataset_drift": report_dict["metrics"][2]["result"]["dataset_drift"],
        "drift_share": report_dict["metrics"][2]["result"]["drift_share"],
        "number_of_drifted_columns": report_dict["metrics"][2]["result"]["number_of_drifted_columns"],
        "report_path": str(report_file)
    }
    
    return report, drift_summary


def check_drift_threshold(drift_summary: Dict, threshold: float = 0.15) -> bool:
    """
    Check if drift exceeds acceptable threshold.
    
    Args:
        drift_summary: Dictionary with drift metrics
        threshold: Maximum acceptable drift share (default 0.15 = 15%)
        
    Returns:
        True if drift is above threshold (alert needed)
    """
    drift_share = drift_summary.get("drift_share", 0)
    return drift_share > threshold
