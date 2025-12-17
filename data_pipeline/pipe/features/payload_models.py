from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum

class Seniority(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    SPECIALIST = "specialist"
    UNKNOWN = "unknown"

class EducationLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    PHD = "phd"
    NOT_DECLARED = "not_declared"

class CandidateProfile(BaseModel):
    resume_text: Optional[str] = Field(None, description="Texto cru do currículo")
    years_experience_range: str = "unknown"
    seniority_inferred: Seniority = Seniority.UNKNOWN
    education_level: EducationLevel = EducationLevel.NOT_DECLARED
    field_of_study: Optional[str] = None
    has_degree: bool = False
    languages: List[str] = []
    availability: str = "unspecified"

    # Make resume_text optional here to support legacy flow where it might not be in the struct yet,
    # or keep it required if we want to enforce it. 
    # The doc says "Rejeitar se vazio" fallback, but usually for "Zero-Shot" pure payload it is required.
    # I'll make it Optional for flexibility in the API wrapper, or strict?
    # Let's keep strict if possible, but API request might be partial.
    # Actually doc schema says: resume_text: str = Field(...) required.
    # I will stick to doc for now, but user might send partial data? 
    # If the user sends "candidate_data" they likely have the struct.
    # But wait, in predict_file, we construct this? No, we receive it. 
    # I'll make it Optional to avoid validation errors if users behave weirdly, 
    # but strictly it should be there. I'll stick to 'Optional' for safety in 'ScoringRequest' composition,
    # or override the field if I want loose validation.
    # Actually, let's follow the doc: `resume_text` is `str` (required).
    # IF the client sends the object, it usually should have it. 
    # However, for `ScoringRequest`, we accept `resume_text` at top level.
    # If I use this model, I might need to populate it from top level.
    
class CandidateSkills(BaseModel):
    technical_skills: List[str] = []
    soft_skills: List[str] = []
    tools: List[str] = []
    certifications: List[str] = []

class QualitySignals(BaseModel):
    has_email: bool = False
    has_phone: bool = False
    has_linkedin: bool = False
    completeness_score: float = 0.0
    is_local_to_job: bool = False

class CandidateBehavioral(BaseModel):
    days_since_profile_update: int = -1
    days_in_process: int = -1
    recruiter_touchpoints: int = 0
    sentiment_score: int = 0

class CandidateData(BaseModel):
    profile: Optional[CandidateProfile] = None
    skills: Optional[CandidateSkills] = None
    quality_signals: Optional[QualitySignals] = None
    behavioral_signals: Optional[CandidateBehavioral] = None
    embeddings: Dict[str, List[float]] = {}

class JobMetadata(BaseModel):
    job_title: str
    department: str = "general"
    location: str = "remote"
    contract_type: str = "unspecified"

class JobRequirements(BaseModel):
    required_tech_skills: List[str] = []
    required_soft_skills: List[str] = []
    target_seniority: Seniority = Seniority.UNKNOWN
    nice_to_have_skills: List[str] = []

class JobData(BaseModel):
    metadata: Optional[JobMetadata] = None
    requirements: Optional[JobRequirements] = None
    embeddings: Dict[str, List[float]] = {}

class ImmutablePayloadV2(BaseModel):
    request_id: str
    candidate: CandidateData
    job_context: JobData
