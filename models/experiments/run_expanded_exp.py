import polars as pl
import lightgbm as lgb
import numpy as np
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import sys

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data_pipeline.pipe.features.cleanning_and_accurate import gerar_features
from data_pipeline.pipe.scoring.behavioral import BehavioralScorer

def run_experiment():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("behavioral_score_expanded")

    with mlflow.start_run():
        # 1. Load Data
        data_path = "data/feature_store/resultado_final.parquet"
        if not os.path.exists(data_path):
            print("Data not found.")
            return

        df = pl.read_parquet(data_path)
        
        # 2. Feature Engineering
        # Reuse the logic from the pipeline to ensure consistency
        df = gerar_features(df)
        
        # Instantiate scorer just to access its feature engineering logic
        scorer = BehavioralScorer()
        df = scorer._feature_engineering(df)
        
        # 3. Prepare Target
        # Define target based on status (as documented in model_decisions.md)
        positive_status = [
            "CONTRATADO PELA DECISION", "CONTRATADO COMO HUNTING", "DOCUMENTAÇÃO PJ",
            "ENCAMINHADO AO REQUISITANTE", "APROVADO", "ENTREVISTA TÉCNICA"
        ]
        
        df = df.with_columns(
            pl.when(pl.col("p_status").is_in(positive_status))
            .then(1)
            .otherwise(0)
            .alias("target")
        )
        
        # 4. Select Features
        features = [
            "sentimento_comentario_score", 
            "contem_palavra_chave_positiva",
            "contem_palavra_chave_negativa",
            "tempo_entre_criacao_e_candidatura",
            "tempo_desde_ultima_atualizacao" # assuming this exists from gerar_features logic or similiar
        ]
        
        # Check which features actually exist
        available_features = [f for f in features if f in df.columns]
        
        # Convert to Pandas for LightGBM
        # Handle Categoricals if we add them (e.g. p_recrutador_tratado)
        if "p_recrutador_tratado" in df.columns:
            available_features.append("p_recrutador_tratado")
            
        X = df.select(available_features).to_pandas()
        y = df.select("target").to_pandas()
        
        # Handle categoricals
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category')

        # 5. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 6. Train Model
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': True,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'max_depth': 15,
            'verbose': -1
        }
        
        mlflow.log_params(params)
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(30)])
        
        # 7. Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"AUC: {auc}")
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)
        
        # 8. Log Model
        mlflow.lightgbm.log_model(model, "model")
        
        # Save locally for API use
        joblib.dump(model, "model/artifacts/behavioral_model_expanded.pkl")
        print("Model saved to model/artifacts/behavioral_model_expanded.pkl")

if __name__ == "__main__":
    run_experiment()
