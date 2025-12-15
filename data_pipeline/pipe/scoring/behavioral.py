import polars as pl
import lightgbm as lgb
import joblib
import os
import sys
import re
from typing import Optional

# Add path to find pipe module if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from data_pipeline.pipe.transform.curated_transform import normalize_dataframe

class BehavioralScorer:
    def __init__(self, model_path: str = "models/artifacts/behavioral_model.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_model()
        
        # Keywords for sentiment/behavioral features
        self.regex_palavras_positivas = r"(?i)(interessado|motivado|bom perfil|gostei|excelente|promissor|encaminhar|aprovado)"
        self.regex_palavras_negativas = r"(?i)(desistiu|não tem perfil|fraco|recusou|sem experiência|confuso|ruim)"
        
        self.common_recruiters = [
            "Michelle", "Daniella", "Stefany", "Katia", "Ana", "Raquel", "Mônica", "Thaise"
        ]

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"Failed to load model from {self.model_path}: {e}")
        else:
            print(f"Model not found at {self.model_path}. Using rules-based fallback (or mock).")

    def _feature_engineering(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies specific feature engineering required for the behavioral model.
        Assumes 'gerar_features' from pipe has already been run.
        """
        # Ensure regex columns exist
        if 'p_comentario' not in df.columns:
            df = df.with_columns(pl.lit("").alias("p_comentario"))
            
        # Apply normalization (clean string columns mainly)
        # Assuming date columns list might be passed or inferred, for now empty or standard set
        # This improves robustness against dirty input in API
        try:
           df = normalize_dataframe(df)
        except Exception as e:
           print(f"Normalization warning: {e}")

        df = df.with_columns(
            contem_palavra_chave_positiva=pl.col('p_comentario').str.contains(self.regex_palavras_positivas).fill_null(False).cast(pl.Int8),
            contem_palavra_chave_negativa=pl.col('p_comentario').str.contains(self.regex_palavras_negativas).fill_null(False).cast(pl.Int8),
            p_recrutador_tratado=pl.when(pl.col("p_recrutador").is_in(self.common_recruiters))
                                .then(pl.col("p_recrutador"))
                                .otherwise(pl.lit("Outros"))
        )
        
        df = df.with_columns(
            sentimento_comentario_score=pl.when(pl.col('contem_palavra_chave_positiva') == 1).then(1)
                                       .when(pl.col('contem_palavra_chave_negativa') == 1).then(-1)
                                       .otherwise(0)
        )
        
        return df

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        df_processed = self._feature_engineering(df)
        
        # If model is loaded, predict proba. If not, maybe use 'sentimento_comentario_score' scaled.
        if self.model:
            # We need to select the exact columns the model expects.
            # Since we don't have the exact feature list validated, this part is risky.
            # For now, we return a heuristic score if model prediction fails or is not robust.
            try:
                # Mocking prediction for robustness in this draft
                # In real prod, we would ensure columns match exactly model.booster_.feature_name()
                pass
            except:
                pass
        
        # Heuristic Fallback (or if model is missing)
        # Normalize score between 0 and 1 roughly
        # 1 (postive) -> 0.8, 0 (neutral) -> 0.5, -1 (negative) -> 0.2
        return df_processed.with_columns(
            score_behavioral = pl.when(pl.col("sentimento_comentario_score") == 1).then(0.9)
                               .when(pl.col("sentimento_comentario_score") == -1).then(0.2)
                               .otherwise(0.5) 
                               # Add slight random noise or other factors
                               # + (pl.col("dias_desde_ultima_atualizacao") < 30).cast(pl.Float32) * 0.1
        ).select(["codigo_candidato", "codigo_vaga", "score_behavioral"])
