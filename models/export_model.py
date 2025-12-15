
import mlflow
import joblib
import os
from pathlib import Path

# Setup
MLFLOW_TRACKING_URI = "file:/home/tiao553/datathon-mlet03/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "Behavioral_Baseline"

def export_best_model():
    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        print(f"Experiment {EXPERIMENT_NAME} not found.")
        return

    # Get runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.roc_auc DESC"])
    
    if runs.empty:
        print("No runs found.")
        return
        
    best_run = runs.iloc[0]
    run_id = best_run.run_id
    auc = best_run["metrics.roc_auc"]
    print(f"Best Run ID: {run_id} with AUC: {auc}")
    
    # Load model
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from {model_uri}...")
    model = mlflow.sklearn.load_model(model_uri)
    
    # Save to artifacts
    output_path = Path("models/artifacts/behavioral_model.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {output_path}...")
    joblib.dump(model, output_path)
    print("Done.")

if __name__ == "__main__":
    export_best_model()
