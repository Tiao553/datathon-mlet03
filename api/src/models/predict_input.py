from pydantic import BaseModel, Field

class PredictInput(BaseModel):
    skill_level: int = Field(..., ge=1, le=10, description="Skill técnico de 1 a 10")
    cultural_fit_score: float = Field(..., ge=0.0, le=1.0, description="Fit cultural de 0 a 1")
    motivation_score: float = Field(..., ge=0.0, le=1.0, description="Motivação de 0 a 1")