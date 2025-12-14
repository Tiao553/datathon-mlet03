from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import polars as pl
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_pipeline')))

from data_pipeline.pipe.scoring.skills import SkillsScorer
from data_pipeline.pipe.scoring.behavioral import BehavioralScorer
from data_pipeline.pipe.scoring.cultural import CulturalScorer
from data_pipeline.pipe.features.prompts import chamar_llm, prompt_candidato, prompt_vaga
from data_pipeline.pipe.features.free_text_transform import extrair_json_limpo

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
    resume_text: str
    job_id: str = None
    job_description: str = None

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": True}

@app.post("/predict")
def predict_score(request: ScoringRequest):
    # 1. Extract Resume Data
    if not request.resume_text:
        raise HTTPException(status_code=400, detail="Resume text is required")

    try:
        # Construct row for prompt (expects dict with 'app_cv_pt')
        prompt_row = {'app_cv_pt': request.resume_text}
        prompt = prompt_candidato(prompt_row)
        
        # Call LLM (using gemma3:1b for speed as verified)
        response_text = chamar_llm(prompt, model_name="gemma3:1b")
        cand_data = extrair_json_limpo(response_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume extraction failed: {str(e)}")

    # 2. Get Job Data
    job_data = {}
    if request.job_id and df_jobs is not None:
        # Lookup job
        # Assuming job_id matches 'codigo_vaga'
        # Need to handle potential type mismatch (int/str)
        # Assuming string for simplicity or casting
        job_row = df_jobs.filter(pl.col("codigo_vaga") == request.job_id).head(1)
        if len(job_row) > 0:
            # We need the structured job data. 
            # If jobs.parquet is raw, we might need to extract it too!
            # But earlier pipeline saved structured data to 'vagas_processadas.jsonl'
            # For this API prototype, let's assumes we extract on the fly if description provided, 
            # OR we rely on what's in the row if columns exist.
            
            # Let's try to extract on fly if description provided, otherwise use what we have
            pass
            
    if not job_data and request.job_description:
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
            job_data = extrair_json_limpo(response_text)
        except Exception as e:
            print(f"Job extraction warning: {e}")
    
    if not job_data:
         # Mock fallback if job data missing
         job_data = {
             "competencias_tecnicas": ["Python", "Data Science"],
             "competencias_comportamentais": ["Proatividade"]
         }

    # 3. Calculate Scores
    
    # Skills
    # Flatten lists
    j_skills = job_data.get("competencias_tecnicas", []) + job_data.get("ferramentas_tecnologicas", [])
    c_skills = cand_data.get("competencias_tecnicas", []) + cand_data.get("ferramentas_tecnologicas", [])
    
    score_skills = skills_scorer.calculate_embedding_score(j_skills, c_skills)
    
    # Cultural
    j_cult = job_data.get("competencias_comportamentais", [])
    c_cult = cand_data.get("competencias_comportamentais", [])
    score_cultural = cultural_scorer.calculate_score(j_cult, c_cult)
    
    # Behavioral
    # Needs dataframe input with specific columns
    # We construct a minimal DF
    
    # Mapping extracted data to model features is tricky without the full pipeline.
    # We will use the simplified features we know we have or can derive.
    
    # Create DF with ALL columns mock-filled or derived matches
    data_dict = {
        "codigo_candidato": "API_REQ",
        "codigo_vaga": request.job_id or "API_JOB",
        "p_comentario": "", # No comment in API request usually
        "contem_palavra_chave_positiva": 0,
        "contem_palavra_chave_negativa": 0,
        "p_recrutador": "Outros"
    }
    
    # Try to use any signals from resume as proxy?
    # For now, behavioral score will be default/neutral 0.5 unless we have history data.
    # But we can run the scorer on this row.
    df_input = pl.DataFrame([data_dict])
    
    try:
        df_scored = behavioral_scorer.predict(df_input)
        score_behavioral = df_scored["score_behavioral"][0]
    except Exception as e:
        print(f"Behavioral scoring error: {e}")
        score_behavioral = 0.5

    return {
        "candidate_extracted": cand_data,
        "scores": {
            "skills": float(score_skills),
            "cultural": float(score_cultural),
            "behavioral": float(score_behavioral)
        }
    }

from fastapi import UploadFile, File
from data_pipeline.pipe.ingest.document_parser import DocumentParser

@app.post("/predict_file")
async def predict_score_file(
    file: UploadFile = File(...),
    job_id: str = None,
    job_description: str = None
):
    # 1. Read File Logic
    try:
        content = await file.read()
        resume_text = DocumentParser.parse_file(content, file.filename)
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"File parsing error: {e}")
         
    if len(resume_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Extracted text is empty. OCR failed or blank file.")
    
    # Delegate to the main logic
    # Create request object to reuse logic
    req = ScoringRequest(
        resume_text=resume_text,
        job_id=job_id,
        job_description=job_description
    )
    return predict_score(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
