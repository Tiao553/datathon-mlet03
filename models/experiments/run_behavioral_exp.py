import sys
import os
import joblib
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Adjust path to import tracking
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from model.experiments.tracking import ExperimentTracker

def train_behavioral_model():
    # Initialize tracker
    tracker = ExperimentTracker("behavioral_score_experiment")
    
    with tracker.start_run(run_name="lgbm_baseline"):
        # 1. Load Data (Assuming data is available as in notebooks)
        # For now, using the path identified in planning
        data_path = "data/feature_store/resultado_final.parquet"
        if not os.path.exists(data_path):
            print(f"Data not found at {data_path}")
            return

        print("Loading data...")
        df_pl = pl.read_parquet(data_path)
        
        # 2. Preprocessing (Logic from Model_engagement_score_final.py)
        # Re-implementing simplified logic for the experiment
        print("Preprocessing...")
        
        # Target definition
        status_sucesso = [
            "CONTRATADO PELA DECISION", "CONTRATADO COMO HUNTING", "DOCUMENTAÇÃO PJ",
            "ENCAMINHADO AO REQUISITANTE", "APROVADO", "ENTREVISTA TÉCNICA"
        ]
        
        df_pl = df_pl.with_columns(
            engajado=pl.when(pl.col("p_situacao_candidado").is_in(status_sucesso))
            .then(1)
            .otherwise(0)
        )
        
        # Features to likely be available (simplified for initial run)
        # Note: In a real scenario we would call the robust feature engineering pipeline here
        # For this experiment script, we'll try to use existing columns or fail gracefully if feature eng not done
        
        # Checking if some feature columns exist, otherwise creating dummies for test
        available_cols = df_pl.columns
        features_num = ['app_experiencia_anos', 'tamanho_cv'] # Placeholder names, need to match actual data
        features_cat = ['p_recrutador']
        
        # Let's rely on what we saw in the file content previously or just basic ones
        # The previous file had extensive FE. For this phase 1, we assume the parquet MIGHT have raw data 
        # that needs FE or already has it. 
        # Given 'resultado_final.parquet' name, it likely has some features.
        
        # Let's use a try-except block to just genericize categorical/numerical detection for the experiment
        df = df_pl.to_pandas()
        X = df.drop(columns=['engajado', 'p_situacao_candidado'], errors='ignore')
        y = df['engajado']
        
        # Simple auto-detection for the experiment
        # Filter out list/array columns which cause OneHotEncoder to fail
        valid_cols = []
        for col in X.columns:
            # Check first non-null value
            first_val = X[col].dropna().iloc[0] if not X[col].dropna().empty else None
            if isinstance(first_val, (list, np.ndarray)):
                print(f"Skipping list/array column: {col}")
                continue
            valid_cols.append(col)
            
        X = X[valid_cols]

        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Log params
        params = {
            "model_type": "LightGBM",
            "test_size": 0.2,
            "random_state": 42
        }
        tracker.log_params(params)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Simple pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ],
            remainder='drop'
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Train
        print("Training model...")
        clf = lgb.LGBMClassifier(random_state=42)
        clf.fit(X_train_processed, y_train)
        
        # Eval
        y_pred = clf.predict(X_test_processed)
        y_proba = clf.predict_proba(X_test_processed)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"AUC: {auc}")
        print(f"Accuracy: {acc}")
        
        tracker.log_metrics({"auc": auc, "accuracy": acc})
        tracker.log_model(clf, "model")
        
        print("Experiment run complete.")

if __name__ == "__main__":
    train_behavioral_model()
