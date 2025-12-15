import mlflow
import os
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentTracker:
    def __init__(self, experiment_name: str, tracking_uri: str = "file:./mlruns"):
        """
        Initialize the MLflow experiment tracker.
        
        Args:
            experiment_name: Name of the experiment in MLflow.
            tracking_uri: URI for MLflow tracking (default: local ./mlruns).
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        self.experiment = mlflow.set_experiment(experiment_name)
        
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(run_name=run_name)
        
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to the current run."""
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to the current run."""
        mlflow.log_metrics(metrics)
        
    def log_model(self, model, artifact_path: str):
        """Log a model to the current run."""
        # Generic logger, can be specialized for sklearn/lightgbm if needed
        if hasattr(model, 'save_model'): # LightGBM
             mlflow.lightgbm.log_model(model, artifact_path)
        else:
             mlflow.sklearn.log_model(model, artifact_path)

    def log_artifact(self, local_path: str):
        """Log a local file as an artifact."""
        mlflow.log_artifact(local_path)
