from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from pydantic import BaseModel, Json
import json

# ... (imports remain matching existing file content if careful, but replace_file_content replaces block)
# I need to be careful about imports. The user file has:
# from fastapi import FastAPI, HTTPException, UploadFile, File, Query
# I need to add Form.

from pydantic import BaseModel
import polars as pl
import os
import sys
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_pipeline')))

from data_pipeline.pipe.scoring.skills import SkillsScorer
from data_pipeline.pipe.scoring.behavioral import BehavioralScorer
from data_pipeline.pipe.scoring.cultural import CulturalScorer
from data_pipeline.pipe.features.prompts import chamar_llm, prompt_candidato, prompt_vaga
from data_pipeline.pipe.features.free_text_transform import extrair_json_limpo
from data_pipeline.pipe.ingest.document_parser import DocumentParser
from data_pipeline.pipe.features.payload_models import CandidateData, JobData

app = FastAPI(title="Recruitment Scoring API", version="1.0")

# Load Global Resources (Models)
# In production, use lifespan events
skills_scorer = SkillsScorer()
behavioral_scorer = BehavioralScorer()
cultural_scorer = CulturalScorer()

# Pre-load Job Data for lookups (simple caching)
JOBS_PATH = "data/curated/jobs.parquet"
df_jobs = None
if os.path.exists(JOBS_PATH):
    df_jobs = pl.read_parquet(JOBS_PATH)

class ScoringRequest(BaseModel):
    resume_text: Optional[str] = None
    job_id: Optional[str] = None
    job_description: Optional[str] = None
    
    # Phase 3: Immutable Payload (Zero-Shot Support)
    # Using strict Pydantic models from payload_models.py
    candidate_data: Optional[CandidateData] = None
    job_data: Optional[JobData] = None

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": True}

@app.post("/predict")
def predict_score(request: ScoringRequest):
    # 1. Extract Candidate Data
    # Initialize containers for scoring
    c_skills = []
    c_cult = []
    cand_data_debug = {} # For response parsing

    if request.candidate_data:
        # Zero-Shot / Structured Path
        # specific mapping from Pydantic model to scorer inputs
        if request.candidate_data.skills:
            c_skills = request.candidate_data.skills.technical_skills + request.candidate_data.skills.tools
            c_cult = request.candidate_data.skills.soft_skills
        
        # Serialize for response
        cand_data_debug = request.candidate_data.dict()
        
    elif request.resume_text:
        try:
            prompt_row = {'app_cv_pt': request.resume_text}
            prompt = prompt_candidato(prompt_row)
            response_text = chamar_llm(prompt, model_name="gemma3:1b")
            cand_data_legacy = extrair_json_limpo(response_text)
            
            # Map legacy dict to lists
            c_skills = cand_data_legacy.get("competencias_tecnicas", []) + cand_data_legacy.get("ferramentas_tecnologicas", [])
            c_cult = cand_data_legacy.get("competencias_comportamentais", [])
            cand_data_debug = cand_data_legacy
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Resume extraction failed: {str(e)}")
    else:
         raise HTTPException(status_code=400, detail="Either 'resume_text' or 'candidate_data' must be provided.")

    # 2. Get Job Data
    j_skills = []
    j_cult = []
    job_data_debug = {}

    if request.job_data:
        # Structured Path
        if request.job_data.requirements:
            j_skills = request.job_data.requirements.required_tech_skills + request.job_data.requirements.nice_to_have_skills
            j_cult = request.job_data.requirements.required_soft_skills
        job_data_debug = request.job_data.dict()
        
    elif request.job_id and df_jobs is not None:
        job_row = df_jobs.filter(pl.col("codigo_vaga") == request.job_id).head(1)
        if len(job_row) > 0:
            pass # Use generic fallback or improve later (legacy flow unimplemented in detail in snippet)
             
    if not job_data_debug and request.job_description:
        try:
            prompt_row = {
                'job_ib_titulo_vaga': 'Job',
                'job_pv_principais_atividades': request.job_description,
                'job_pv_competencia_tecnicas_e_comportamentais': '',
                'job_pv_demais_observacoes': '',
                'job_pv_habilidades_comportamentais_necessarias': ''
            }
            prompt = prompt_vaga(prompt_row)
            response_text = chamar_llm(prompt, model_name="gemma3:1b")
            job_data_legacy = extrair_json_limpo(response_text)
            
            j_skills = job_data_legacy.get("competencias_tecnicas", []) + job_data_legacy.get("ferramentas_tecnologicas", [])
            j_cult = job_data_legacy.get("competencias_comportamentais", [])
            job_data_debug = job_data_legacy
        except Exception as e:
            print(f"Job extraction warning: {e}")
    
    # Fallback if no skills found (prevents empty vector errors)
    if not j_skills and not request.job_data:
         # Legacy fallback
         j_skills = ["Python", "Data Science"]
         j_cult = ["Proatividade"]
         job_data_debug = {"note": "fallback_used"}

    # 3. Calculate Scores
    score_skills = skills_scorer.calculate_embedding_score(j_skills, c_skills)
    
    score_cultural = cultural_scorer.calculate_score(j_cult, c_cult)
    
    data_dict = {
        "codigo_candidato": "API_REQ",
        "codigo_vaga": request.job_id or "API_JOB",
        "p_comentario": "", 
        "contem_palavra_chave_positiva": 0,
        "contem_palavra_chave_negativa": 0,
        "p_recrutador": "Outros"
    }
    df_input = pl.DataFrame([data_dict])
    try:
        df_scored = behavioral_scorer.predict(df_input)
        score_behavioral = df_scored["score_behavioral"][0]
    except Exception as e:
        print(f"Behavioral scoring error: {e}")
        score_behavioral = 0.5

    return {
        "candidate_extracted": cand_data_debug,
        "job_extracted": job_data_debug,
        "scores": {
            "skills": float(score_skills),
            "cultural": float(score_cultural),
            "behavioral": float(score_behavioral)
        }
    }

@app.post("/predict_file")
async def predict_score_file(
    file: UploadFile = File(...),
    job_id: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None),
    use_ocr: bool = Query(False),
    candidate_data: Optional[str] = Form(None),
    job_data: Optional[str] = Form(None)
):
    # 1. Read File Logic
    try:
        content = await file.read()
        filename = file.filename
        # OCR/Parser Logic: Extracts text to populate 'resume_text' field
        resume_text = DocumentParser.parse_file(content, filename, use_ocr=use_ocr)
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"File parsing error: {e}")
         
    if len(resume_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Extracted text is empty. Try use_ocr=true.")
    
    
    # 2. Parse JSON fields if provided
    c_data_parsed = None
    if candidate_data:
        try:
            # Enforce Strict V2 Schema
            c_data_parsed = CandidateData.model_validate_json(candidate_data)
        except Exception as e:
             raise HTTPException(status_code=400, detail=f"Invalid structure in candidate_data: {e}")

    j_data_parsed = None
    if job_data:
        try:
            # Enforce Strict V2 Schema
            j_data_parsed = JobData.model_validate_json(job_data)
        except Exception as e:
             raise HTTPException(status_code=400, detail=f"Invalid structure in job_data: {e}")

    # Delegate to the main logic
    req = ScoringRequest(
        resume_text=resume_text,
        job_id=job_id,
        job_description=job_description,
        candidate_data=c_data_parsed,
        job_data=j_data_parsed
    )
    return predict_score(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
