import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Union

class SkillsScorer:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.level_map = {
            'NENHUM': 0, 'BÁSICO': 1, 'INTERMEDIÁRIO': 2, 'AVANÇADO': 3, 'FLUENTE': 4, 'TÉCNICO': 1.5,
            'JÚNIOR': 1, 'JUNIOR': 1, 'PLENO': 2, 'SÊNIOR': 3, 'SENIOR': 3, 'ESPECIALISTA': 4, 'LÍDER': 5,
            'ENSINO MÉDIO COMPLETO': 1, 'ENSINO SUPERIOR INCOMPLETO': 2, 'ENSINO SUPERIOR COMPLETO': 3,
            'PÓS-GRADUAÇÃO': 4, 'MESTRADO': 4, 'DOUTORADO': 4
        }
        self.weights = {'professional': 0.6, 'academic': 0.2, 'english': 0.2}

    def calculate_embedding_score(self, job_skills: List[str], candidate_skills: List[str], threshold: float = 0.5) -> float:
        if not job_skills or not candidate_skills:
            return 0.0
            
        # Filter empty strings
        job_skills = [s for s in job_skills if s]
        candidate_skills = [s for s in candidate_skills if s]
        
        if not job_skills or not candidate_skills:
            return 0.0

        job_emb = self.model.encode(job_skills, convert_to_tensor=True)
        cand_emb = self.model.encode(candidate_skills, convert_to_tensor=True)
        
        # Cosine similarity matrix
        sim_matrix = util.cos_sim(job_emb, cand_emb).cpu().numpy()
        
        # For each job skill, find max match in candidate skills
        best_matches = np.max(sim_matrix, axis=1)
        
        # Filter matches above threshold
        qualified_matches = [score if score >= threshold else 0.0 for score in best_matches]
        
        return float(np.mean(qualified_matches))

    def _get_level_score(self, level: str) -> float:
        return self.level_map.get(str(level).upper().strip(), 0.0)

    def calculate_structured_score(self, job_levels: Dict[str, str], cand_levels: Dict[str, str]) -> float:
        """
        Calculates structured score based on professional, academic, and english levels.
        levels dict keys: 'professional', 'academic', 'english'
        """
        scores = []
        
        for key, weight in self.weights.items():
            job_lvl_str = job_levels.get(key, '')
            cand_lvl_str = cand_levels.get(key, '')
            
            job_score = self._get_level_score(job_lvl_str)
            cand_score = self._get_level_score(cand_lvl_str)
            
            if job_score == 0:
                similarity = 1.0
            elif cand_score == 0:
                similarity = 0.0
            else:
                similarity = min(cand_score / job_score, 1.0)
                
            scores.append(similarity * weight)
            
        return sum(scores)

    def get_total_score(self, job_data: Dict, cand_data: Dict) -> float:
        # 1. Semantic Skill Match
        job_skills = job_data.get('competencias_tecnicas', []) + job_data.get('ferramentas_tecnologicas', [])
        cand_skills = cand_data.get('competencias_tecnicas', []) + cand_data.get('ferramentas_tecnologicas', [])
        
        semantic_score = self.calculate_embedding_score(job_skills, cand_skills)
        
        # 2. Structured Match (Assuming these keys exist or are mapped)
        job_levels = {
            'professional': job_data.get('senioridade_aparente', ''),
            'academic': job_data.get('nivel_formacao', ''),
            'english': job_data.get('idiomas', {}).get('ingles', '') # Example mapping
        }
        cand_levels = {
            'professional': cand_data.get('senioridade_aparente', ''),
            'academic': cand_data.get('nivel_formacao', ''),
            'english': cand_data.get('idiomas', {}).get('ingles', '')
        }
        
        structured_score = self.calculate_structured_score(job_levels, cand_levels)
        
        # Final technical score (weighted average, e.g., 70% semantic, 30% structured)
        return 0.7 * semantic_score + 0.3 * structured_score
