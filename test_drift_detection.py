"""
Test script for drift detection DAG

This script tests the drift detection logic locally without Airflow.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add dags to path
dags_path = Path(__file__).parent / 'dags'
sys.path.insert(0, str(dags_path))

from utils.drift_detection import (
    calculate_drift_report,
    check_drift_threshold
)


def create_sample_data():
    """Create sample training and production data for testing."""
    np.random.seed(42)
    
    # Training data (reference)
    n_train = 1000
    train_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_train),
        'feature_2': np.random.normal(5, 2, n_train),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_train),
        'feature_4': np.random.uniform(0, 100, n_train),
    })
    
    # Production data - with some drift
    n_prod = 500
    prod_data = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, n_prod),  # Slight drift
        'feature_2': np.random.normal(5, 2, n_prod),      # No drift
        'feature_3': np.random.choice(['A', 'B', 'C', 'D'], n_prod),  # New category
        'feature_4': np.random.uniform(0, 100, n_prod),   # No drift
    })
    
    return train_data, prod_data


def test_drift_detection():
    """Test the drift detection functionality."""
    print("=" * 60)
    print("Testing Evidently AI Drift Detection")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    train_data, prod_data = create_sample_data()
    print(f"   Training data: {train_data.shape}")
    print(f"   Production data: {prod_data.shape}")
    
    # Calculate drift
    print("\n2. Running Evidently drift detection...")
    try:
        report, drift_summary = calculate_drift_report(
            reference_data=train_data,
            current_data=prod_data,
            output_path="/tmp/drift_reports"
        )
        
        print("\n3. Drift Detection Results:")
        print(f"   ✓ Report generated successfully")
        print(f"   Dataset Drift: {drift_summary['dataset_drift']}")
        print(f"   Drift Share: {drift_summary['drift_share']:.2%}")
        print(f"   Drifted Columns: {drift_summary['number_of_drifted_columns']}")
        print(f"   Report saved to: {drift_summary['report_path']}")
        
        # Check threshold
        print("\n4. Checking drift threshold...")
        alert_needed = check_drift_threshold(drift_summary, threshold=0.15)
        if alert_needed:
            print("   ⚠️  ALERT: Drift exceeds 15% threshold!")
        else:
            print("   ✓ Drift is within acceptable limits")
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during drift detection: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_drift_detection()
    sys.exit(0 if success else 1)
