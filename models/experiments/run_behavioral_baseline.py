import polars as pl
import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from rapidfuzz import fuzz
from scipy.stats import randint, uniform

# --- Configuration ---
# Adjusted path: script is in models/experiments (depth 2) -> ../../data/curated
DATA_DIR = Path("../../data/curated")
EXPERIMENT_NAME = "Behavioral_Baseline"
MLFLOW_TRACKING_URI = "file:/home/tiao553/datathon-mlet03/mlruns"

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

# --- Feature Engineering Functions (Ported from Analysis) ---

def gerar_indicador(nome: str, col_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(col_expr.is_not_null() & (col_expr != ""))
        .then(True)
        .otherwise(False)
        .alias(nome)
    )

def comparar_locais_com_fuzzy(locais_candidato: list[str], locais_vaga: list[str], threshold: int = 70) -> bool:
    for loc_candidato in locais_candidato:
        for loc_vaga in locais_vaga:
            if loc_candidato and loc_vaga:
                score = fuzz.partial_ratio(str(loc_candidato).upper(), str(loc_vaga).upper())
                if score >= threshold:
                    return True
    return False

def remover_pii_e_engineering(df: pl.DataFrame) -> pl.DataFrame:
    print("Executing PII Removal and Indicator Creation...")
    pii_cols = [
        "p_nome", "app_ib_telefone_recado", "app_ib_telefone", "app_ib_email",
        "app_ip_nome", "app_ip_cpf", "app_ip_email", "app_ip_email_secundario",
        "app_ip_data_nascimento", "app_ip_telefone_celular", "app_ip_telefone_recado",
        "app_ip_endereco", "app_ip_skype", "app_ip_url_linkedin", "app_ip_facebook",
    ]

    # Indicators
    df = df.with_columns([
        gerar_indicador("ind_app_telefone", pl.col("app_ib_telefone")),
        gerar_indicador("ind_app_email", pl.col("app_ib_email")),
        gerar_indicador("ind_app_linkedin", pl.col("app_ip_url_linkedin")),
        gerar_indicador("ind_app_endereco", pl.col("app_ip_endereco")),
        gerar_indicador("ind_app_facebook", pl.col("app_ip_facebook")),
    ])
    
    return df.drop([c for c in pii_cols if c in df.columns])

def preprocess_text_features(df: pl.DataFrame) -> pl.DataFrame:
    print("Engineering Text & behavioral Features...")
    
    # 1. Profile Completeness
    profile_cols = [
        "app_ib_objetivo_profissional", "app_ib_local", "app_ip_sexo", "app_ip_estado_civil",
        "app_ip_pcd", "app_ip_remuneracao", "app_fei_nivel_academico", "app_fei_nivel_ingles"
    ]
    
    # Calculate percentage of non-nulls (Checking columns exist first to avoid error)
    valid_cols = [c for c in profile_cols if c in df.columns]
    if valid_cols:
        completeness_expr = sum([pl.col(c).is_not_null().cast(pl.Float64) for c in valid_cols]) / len(valid_cols)
    else:
        completeness_expr = pl.lit(0.0)
    
    df = df.with_columns([
        completeness_expr.alias("percentual_perfil_completo"),
        pl.col("app_cv_pt").str.len_chars().fill_null(0).alias("tamanho_cv")
    ])

    # 2. Regex Sentiment
    regex_negative = r"(?i)(n(a|ã)o\sresponde|desisti|sem\sinteresse|n(a|ã)o\sretorna|n(a|ã)o\satende)"
    regex_positive = r"(?i)(proativ|interessad|bom\sperfil|responsiv|gostou|avan(ç|c)ou|performou\sbem)"

    df = df.with_columns([
        pl.col("p_comentario").str.contains(regex_negative).fill_null(False).alias("contem_palavra_chave_negativa"),
        pl.col("p_comentario").str.contains(regex_positive).fill_null(False).alias("contem_palavra_chave_positiva")
    ])

    # Score
    df = df.with_columns(
        pl.when(pl.col("contem_palavra_chave_positiva"))
        .then(1)
        .when(pl.col("contem_palavra_chave_negativa"))
        .then(-1)
        .otherwise(0)
        .alias("sentimento_comentario_score")
    )

    return df

def create_target(df: pl.DataFrame) -> pl.DataFrame:
    print("Creating Target Variable 'engajado'...")
    success_statuses = [
        "CONTRATADO PELA DECISION", "CONTRATADO COMO HUNTING", "DOCUMENTAÇÃO PJ",
        "ENCAMINHADO AO REQUISITANTE", "APROVADO", "ENTREVISTA TÉCNICA"
    ]
    
    return df.with_columns(
        pl.col("p_situacao_candidado").is_in(success_statuses).cast(pl.Int8).alias("engajado")
    )

def load_and_merge_data() -> pl.DataFrame:
    print("Loading Data...")
    try:
        jobs = pl.read_parquet(DATA_DIR / "jobs.parquet")
        applicants = pl.read_parquet(DATA_DIR / "applicants.parquet")
        prospects = pl.read_parquet(DATA_DIR / "prospects.parquet")
    except Exception as e:
        print(f"Error loading data from {DATA_DIR}: {e}")
        # Fallback to local execution directory if running from root without proper relative paths
        print("Attempting alternate path...")
        DATA_DIR_ALT = Path("data/curated")
        jobs = pl.read_parquet(DATA_DIR_ALT / "jobs.parquet")
        applicants = pl.read_parquet(DATA_DIR_ALT / "applicants.parquet")
        prospects = pl.read_parquet(DATA_DIR_ALT / "prospects.parquet")

    print(f"Prospects shape: {prospects.shape}")
    # Prefixing columns to match analysis logic
    # Applicants -> app_
    # Jobs -> job_
    
    # Identify non-prefixed columns and rename
    # We prefix ALL columns for applicants and jobs to map schema like 'ib_telefone' -> 'app_ib_telefone'
    
    applicants = applicants.select([
        pl.col(c).alias(f"app_{c}") for c in applicants.columns
    ])
    
    jobs = jobs.select([
        pl.col(c).alias(f"job_{c}") for c in jobs.columns
    ])
    
    # Now Keys are:
    # Applicants: app_codigo_candidato
    # Jobs: job_codigo_vaga
    # Prospects: p_codigo, codigo_vaga
    
    print("Merging Data...")
    df = prospects.join(applicants, left_on="p_codigo", right_on="app_codigo_candidato", how="inner")
    df = df.join(jobs, left_on="codigo_vaga", right_on="job_codigo_vaga", how="inner")
    print(f"Merged shape: {df.shape}")
    
    return df

def clean_and_prepare(df: pl.DataFrame) -> pd.DataFrame:
    df = remover_pii_e_engineering(df)
    df = preprocess_text_features(df)
    df = create_target(df)
    
    # Recruiter Grouping
    recruiter_counts = df.group_by("p_recrutador").count()
    rare_recruiters = recruiter_counts.filter(pl.col("count") < 10)["p_recrutador"]
    
    df = df.with_columns(
        pl.when(pl.col("p_recrutador").is_in(rare_recruiters))
        .then(pl.lit("Outros"))
        .otherwise(pl.col("p_recrutador"))
        .alias("p_recrutador_tratado")
    )
    
    # Handle NaN in numeric features before conversion if any
    # (LightGBM handles NaN, but StandardScaler might complain if not handled or configured)
    
    return df.to_pandas()

def train_model(pdf: pd.DataFrame):
    setup_mlflow()
    
    features_num = [
        "percentual_perfil_completo", "tamanho_cv", 
        "sentimento_comentario_score"
    ]
    features_cat = ["p_recrutador_tratado"]
    target = "engajado"
    
    # Check if features exist
    missing_cols = [c for c in features_num + features_cat if c not in pdf.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    X = pdf[features_num + features_cat]
    y = pdf[target]

    print(f"Target distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
    ])

    print("Preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Starting Training with MLflow...")
    with mlflow.start_run():
        mlflow.log_param("model", "LightGBM")
        
        lgbm = lgb.LGBMClassifier(is_unbalance=True, random_state=42, verbose=-1)
        
        param_dist = {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 0.1),
            'num_leaves': randint(10, 40),
            'max_depth': [-1, 10, 20],
        }
        
        search = RandomizedSearchCV(
            lgbm, 
            param_distributions=param_dist, 
            n_iter=10, 
            cv=3, 
            scoring='roc_auc', 
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train_processed, y_train)
        best_model = search.best_estimator_
        
        mlflow.log_params(search.best_params_)
        
        y_proba = best_model.predict_proba(X_test_processed)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"Test AUC: {auc:.4f}")
        mlflow.log_metric("roc_auc", auc)
        
        mlflow.sklearn.log_model(best_model, "model")
        print("Run Complete.")

if __name__ == "__main__":
    df_pl = load_and_merge_data()
    df_pd = clean_and_prepare(df_pl)
    train_model(df_pd)
