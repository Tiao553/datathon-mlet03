
import polars as pl
import mlflow
import mlflow.sklearn
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

# --- Configuration ---
DATA_DIR = Path("/home/tiao553/datathon-mlet03/data/curated")
EXPERIMENT_NAME = "Cultural_Baseline"
MLFLOW_TRACKING_URI = "file:/home/tiao553/datathon-mlet03/mlruns"

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def load_data() -> pl.DataFrame:
    print("Loading Data...")
    try:
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
        return pl.DataFrame()

def run_cultural_pipeline(df: pl.DataFrame):
    print("Running Cultural Fit Pipeline...")
    
    # Hypothesis: Cultural fit is "Soft Skills" match + "Values" match
    # Fields:
    # Job: job_pv_habilidades_comportamentais_necessarias, job_ib_objetivo_vaga
    # App: app_ib_objetivo_profissional, app_competencias_comportamentais (if available, mostly in 'app_ip_conhecimentos_tecnicos' or separate?)
    
    # We will do semantic similarity between:
    # A) Job Culture: "Habilidades Comportamentais" + "Beneficios" (proxy for culture) + "Objetivo Vaga"
    # B) App Mindset: "Objetivo Profissional" + "Comentario" (often reveals attitude)
    
    df = df.with_columns([
        pl.concat_str([
            pl.col("job_pv_habilidades_comportamentais_necessarias").fill_null(""),
            pl.col("job_ib_objetivo_vaga").fill_null("")
        ], separator=" ").alias("job_culture_text"),
        
        pl.concat_str([
            pl.col("app_ib_objetivo_profissional").fill_null(""),
            pl.col("p_comentario").fill_null("")
        ], separator=" ").alias("app_culture_text")
    ])
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    LIMIT = 1000
    if len(df) > LIMIT:
        df = df.head(LIMIT)
        
    job_emb = model.encode(df["job_culture_text"].to_list(), show_progress_bar=True)
    app_emb = model.encode(df["app_culture_text"].to_list(), show_progress_bar=True)
    
    scores = []
    for i in range(len(job_emb)):
        j_v = job_emb[i]
        a_v = app_emb[i]
        cos = np.dot(j_v, a_v) / (norm(j_v) * norm(a_v))
        scores.append(cos)
        
    df = df.with_columns(pl.Series("cultural_score", scores))
    
    return df

def log_experiment(df: pl.DataFrame):
    setup_mlflow()
    with mlflow.start_run():
        mlflow.log_param("model", "Cultural_Semantic_Sim")
        mlflow.log_metric("avg_cultural_score", df["cultural_score"].mean())
        
        df.select(["p_codigo", "cultural_score", "job_culture_text", "app_culture_text"]).write_csv("cultural_scored_sample.csv")
        mlflow.log_artifact("cultural_scored_sample.csv")
        print("Logged cultural experiment.")

if __name__ == "__main__":
    df = load_data()
    if not df.is_empty():
        scored = run_cultural_pipeline(df)
        log_experiment(scored)

