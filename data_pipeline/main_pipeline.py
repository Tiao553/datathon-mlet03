import polars as pl
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipe.utils.logger import get_logger
from pipe.features.cleanning_and_accurate import gerar_features
from pipe.scoring.skills import SkillsScorer
from pipe.scoring.behavioral import BehavioralScorer
from pipe.scoring.cultural import CulturalScorer

logger = get_logger("main_pipeline")

def main():
    logger.info("Starting Main Scoring Pipeline")
    
    # 1. Load Data
    # Assuming 'resultado_final.parquet' contains the merged data (Prospects + Applicants + Jobs)
    # OR we load raw and join. Let's assume we start from the 'curated' join logic in main_feature_engineering
    # For now, let's look for the richest available file
    input_path = "data/feature_store/resultado_final.parquet"
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        # Fallback: try to run joins? Or just fail. 
        # Assuming main_feature_engineering.py runs BEFORE this.
        return

    logger.info(f"Loading data from {input_path}")
    df = pl.read_parquet(input_path)
    
    if len(df) == 0:
        logger.warning("Empty dataframe")
        return

    # 2. Feature Engineering (General)
    # Check if features need to be generated (e.g. check for PII columns that are dropped in simple FE)
    if "app_ib_telefone" in df.columns:
        logger.info("Raw data detected. Running full feature engineering...")
        df = gerar_features(df)
    else:
        logger.info("Processed data detected. Skipping initial feature engineering.")
    
    # 3. Calculate Scores
    
    # Skills Score
    logger.info("Calculating Skills Score...")
    skills_scorer = SkillsScorer()
    
    # We need to map dataframe rows to scorer inputs
    # This loop is inefficient for python, but fits the modular design. 
    # For optimization, we would vectorise.
    
    skills_scores = []
    
    rows = df.to_dicts()
    for row in rows:
        # Extract lists. The parquet likely has them as lists or strings.
        # Assuming lists if main_feature_engineering ran with Ollama
        j_skills = row.get("job_competencias_tecnicas", []) + row.get("job_ferramentas_tecnologicas", [])
        c_skills = row.get("app_competencias_tecnicas", []) + row.get("app_ferramentas_tecnologicas", [])
        
        # Handle if they are strings
        if isinstance(j_skills, str): j_skills = [j_skills] # naive split?
        if isinstance(c_skills, str): c_skills = [c_skills]
        if isinstance(j_skills, list) and len(j_skills) > 0 and isinstance(j_skills[0], list): j_skills = [item for sublist in j_skills for item in sublist] # flatten if nested
        
        score_val = skills_scorer.get_total_score(row, row) # Passing full row as dict
        skills_scores.append(score_val)
        
    df = df.with_columns(pl.Series("score_skills", skills_scores))
    
    # Behavioral Score
    logger.info("Calculating Behavioral Score...")
    behav_scorer = BehavioralScorer()
    df_behav = behav_scorer.predict(df)
    
    # Join back using IDs
    df = df.join(df_behav, on=["codigo_candidato", "codigo_vaga"], how="left")
    
    # Cultural Score
    logger.info("Calculating Cultural Score...")
    cult_scorer = CulturalScorer()
    df_cult = cult_scorer.process_dataframe(df)
    
    df = df.join(df_cult, on=["codigo_candidato", "codigo_vaga"], how="left")
    
    # 4. Save Output
    output_path = "data/output/scored_candidates.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.write_parquet(output_path)
    
    logger.info(f"Pipeline finished. Data saved to {output_path}")
    logger.info(f"Columns: {df.columns}")
    logger.info(df.select(["codigo_candidato", "score_skills", "score_behavioral", "score_cultural"]).head())

if __name__ == "__main__":
    main()
