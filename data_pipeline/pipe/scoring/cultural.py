from sentence_transformers import SentenceTransformer, util
import polars as pl
from typing import List
import numpy as np

class CulturalScorer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Reuse the same model instance if possible in main pipeline to save RAM
        self.model = SentenceTransformer(model_name)

    def calculate_score(self, job_culture: List[str], cand_culture: List[str]) -> float:
        if not job_culture or not cand_culture:
            return 0.5 # Neutral if unknown
            
        # Filter empty
        job_culture = [x for x in job_culture if x]
        cand_culture = [x for x in cand_culture if x]
        
        if not job_culture or not cand_culture:
            return 0.5

        # Embedding similarity
        job_emb = self.model.encode(job_culture, convert_to_tensor=True)
        cand_emb = self.model.encode(cand_culture, convert_to_tensor=True)
        
        sim_matrix = util.cos_sim(job_emb, cand_emb).cpu().numpy()
        
        # Average of best matches
        mean_score = np.mean(np.max(sim_matrix, axis=1))
        return float(mean_score)

    def process_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Batched processing for cultural score.
        """
        # We need to extract lists from text/list columns
        # Assuming struct or list columns: 'job_pv_habilidades_comportamentais_necessarias' vs 'app_competencias_comportamentais'
        
        scores = []
        
        # It's better to iterate or use map_elements.
        # Use map_elements for simplicity in Polars
        # Note: This is slow for large DF. Vectorized embedding is better but requires flattening.
        
        rows = df.select([
            "codigo_candidato", 
            "codigo_vaga", 
            "job_competencias_comportamentais", 
            "app_competencias_comportamentais"
        ]).to_dicts()
        
        for row in rows:
            j_comp = row.get("job_competencias_comportamentais", [])
            c_comp = row.get("app_competencias_comportamentais", [])
            
            # Ensure they are lists
            if isinstance(j_comp, str): j_comp = [j_comp]
            if isinstance(c_comp, str): c_comp = [c_comp]
            if j_comp is None: j_comp = []
            if c_comp is None: c_comp = []
            
            score = self.calculate_score(j_comp, c_comp)
            scores.append(score)
            
        return df.with_columns(
            pl.Series("score_cultural", scores)
        ).select(["codigo_candidato", "codigo_vaga", "score_cultural"])
