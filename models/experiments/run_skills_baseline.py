
import polars as pl
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DATA_DIR = Path("/home/tiao553/datathon-mlet03/data/curated")
EXPERIMENT_NAME = "Skills_Baseline"
MLFLOW_TRACKING_URI = "file:/home/tiao553/datathon-mlet03/mlruns"

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

# --- Logic from model_technical_score.ipynb ---

LEVEL_MAP = {
    # Idiomas
    'NENHUM': 0, 'BÁSICO': 1, 'TÉCNICO': 1.5, 'INTERMEDIÁRIO': 2, 'AVANÇADO': 3, 'FLUENTE': 4,
    # Níveis Profissionais
    'JÚNIOR': 1, 'JUNIOR': 1, 'PLENO': 2, 'SÊNIOR': 3, 'SENIOR': 3, 'ESPECIALISTA': 4, 'LÍDER': 5,
    # Níveis Acadêmicos
    'ENSINO MÉDIO COMPLETO': 1, 'ENSINO SUPERIOR INCOMPLETO': 2, 'ENSINO SUPERIOR COMPLETO': 3,
    'PÓS-GRADUAÇÃO': 4, 'MESTRADO': 4, 'DOUTORADO': 4
}

def get_level_score(level_str: str) -> float:
    if level_str is None:
        return 0.0
    return LEVEL_MAP.get(str(level_str).upper().strip(), 0.0)

def calculate_structured_score(job_val, app_val):
    job_score = get_level_score(job_val)
    app_score = get_level_score(app_val)
    
    if job_score == 0: return 1.0 # Requirement not specified
    if app_score == 0: return 0.0 # Candidate missing info
    
    return min(app_score / job_score, 1.0)

def load_data() -> pl.DataFrame:
    print("Loading Data...")
    try:
        # Using feature store if available as per notebook, or recreating from curated
        # Notebook used: '../data/feature_store/resultado_final.parquet'
        # Let's try to use the raw curated files to ensure we have the raw text for embeddings
        # But we need the joined structure.
        
        # Re-using the join logic from behavioral script for consistency
        jobs = pl.read_parquet(DATA_DIR / "jobs.parquet")
        applicants = pl.read_parquet(DATA_DIR / "applicants.parquet")
        prospects = pl.read_parquet(DATA_DIR / "prospects.parquet")

        # Prefix
        applicants = applicants.select([pl.col(c).alias(f"app_{c}") for c in applicants.columns])
        jobs = jobs.select([pl.col(c).alias(f"job_{c}") for c in jobs.columns])
        
        df = prospects.join(applicants, left_on="p_codigo", right_on="app_codigo_candidato", how="inner")
        df = df.join(jobs, left_on="codigo_vaga", right_on="job_codigo_vaga", how="inner")
        
        return df
    except Exception as e:
        print(f"Error loading: {e}")
        # Fallback for running in root
        return pl.DataFrame() # Should handle this better but assuming path is correct now

def run_skills_pipeline(df: pl.DataFrame):
    print("Running Skills/Technical Scoring Pipeline...")
    
    # 1. Structured Scoring
    # professional_level_score (Weight 0.6)
    # academic_level_score (Weight 0.2)
    # english_level_score (Weight 0.2)
    
    # Check column names
    # Job levels: job_pv_nivel_profissional, job_pv_nivel_academico, job_pv_nivel_ingles
    # App levels: app_ip_senioridade_aparente (or from CV?), app_fei_nivel_academico, app_fei_nivel_ingles
    # Note: 'app_senioridade_aparente' was used in notebook. In schema it might be 'app_ip_nivel_profissional' or similar.
    # Schema check from previous step: 'ip_nivel_profissional' exists.
    
    df = df.with_columns([
        pl.struct(["job_pv_nivel_profissional", "app_ip_nivel_profissional"])
          .map_elements(lambda x: calculate_structured_score(x['job_pv_nivel_profissional'], x['app_ip_nivel_profissional']), return_dtype=pl.Float64)
          .alias("score_senioridade"),
          
        pl.struct(["job_pv_nivel_academico", "app_fei_nivel_academico"])
          .map_elements(lambda x: calculate_structured_score(x['job_pv_nivel_academico'], x['app_fei_nivel_academico']), return_dtype=pl.Float64)
          .alias("score_academico"),
          
        pl.struct(["job_pv_nivel_ingles", "app_fei_nivel_ingles"])
          .map_elements(lambda x: calculate_structured_score(x['job_pv_nivel_ingles'], x['app_fei_nivel_ingles']), return_dtype=pl.Float64)
          .alias("score_ingles"),
    ])
    
    df = df.with_columns(
        (pl.col("score_senioridade") * 0.6 + pl.col("score_academico") * 0.2 + pl.col("score_ingles") * 0.2)
        .alias("structured_match_score")
    )
    
    # 2. Soft Match (Embeddings)
    # Concatenate skills lists
    # Job: job_pv_competencia_tecnicas_e_comportamentais (need to be careful, schema showed this)
    # App: app_ip_conhecimentos_tecnicoss (or similar)
    
    # For now, simplistic approach: text concatenation of relevant columns
    print("Generating Embeddings (Soft Match)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Helper to clean lists/text
    # We will cast to string for simplicity as some might be lists
    job_text = df.select(
        pl.concat_str([
            pl.col("job_pv_principais_atividades").fill_null(""),
            pl.col("job_pv_competencia_tecnicas_e_comportamentais").fill_null("")
        ], separator=" ")
    ).to_series().to_list()
    
    app_text = df.select(
        pl.concat_str([
            pl.col("app_ip_conhecimentos_tecnicos").fill_null(""),
            pl.col("app_cv_pt").fill_null("") # Using full CV content as efficient proxy for all skills
        ], separator=" ")
    ).to_series().to_list()
    
    # Encode
    # In production batch this by chunks. Here strict dataset size?
    # Truncating to 1000 for speed if testing
    LIMIT = 1000
    if len(job_text) > LIMIT:
        print(f"Warning: Truncating to {LIMIT} for baseline speed.")
        job_text = job_text[:LIMIT]
        app_text = app_text[:LIMIT]
        df = df.head(LIMIT)
        
    job_emb = model.encode(job_text, show_progress_bar=True)
    app_emb = model.encode(app_text, show_progress_bar=True)
    
    # Cosine Sim (Pairwise)
    # Efficient pairwise: (A . B) / (|A| |B|)
    # SentenceTransformer utils.cos_sim does this block-wise
    # But we want row-wise correspondence (Job i vs App i)
    # So dot product of normalized vectors
    
    from numpy.linalg import norm
    
    scores = []
    for i in range(len(job_emb)):
        j_v = job_emb[i]
        a_v = app_emb[i]
        cos = np.dot(j_v, a_v) / (norm(j_v) * norm(a_v))
        scores.append(cos)
        
    df = df.with_columns(pl.Series("soft_match_score", scores))
    
    # 3. Clustering (Tiering)
    print("Clustering Candidates...")
    # Combine scores
    # Overall Technical Score = 0.4 * Structured + 0.6 * Soft (Hypothesis, can be tuned)
    df = df.with_columns(
        (pl.col("structured_match_score") * 0.4 + pl.col("soft_match_score") * 0.6).alias("final_technical_score")
    )
    
    X = df.select("final_technical_score").to_numpy()
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Order clusters so 0=Low, 2=High? KMeans doesn't guarantee order.
    # We rank them by mean score
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(cluster_centers)
    rank_map = {old: new for new, old in enumerate(sorted_idx)}
    
    ranked_clusters = [rank_map[c] for c in clusters]
    
    df = df.with_columns(pl.Series("tier", ranked_clusters)) # 0=Low, 1=Mid, 2=High/Top
    
    return df

def log_experiment(df: pl.DataFrame):
    setup_mlflow()
    with mlflow.start_run():
        mlflow.log_param("model_type", "RuleBased+Embedding+KMeans")
        mlflow.log_metric("avg_technical_score", df["final_technical_score"].mean())
        
        # Log distribution
        tier_counts = df["tier"].value_counts().sort("tier")
        print("Tier Counts:")
        print(tier_counts)
        
        # Log artifacts (sample CSV)
        df.select(["p_codigo", "final_technical_score", "tier", "structured_match_score", "soft_match_score"]).write_csv("skills_scored_sample.csv")
        mlflow.log_artifact("skills_scored_sample.csv")
        
        print(f"Logged {len(df)} scored candidates.")

if __name__ == "__main__":
    df = load_data()
    if not df.is_empty():
        scored_df = run_skills_pipeline(df)
        log_experiment(scored_df)
    else:
        print("Data load failed or empty.")
