# Phase III – Immutable Payload Schema & Signal Mapping

Este documento detalha o contrato `ImmutablePayloadV2` descrito em `docs/ADR-0.md` e mapeia os ~30 sinais necessários para operar o pipeline Zero-Shot sem dependência de `job_id`. Cada campo indica a fonte prevista (pipeline existente, enriquecimento LLM ou cálculo derivado) e o comportamento de fallback para manter o payload **imutável** mesmo com dados faltantes.

---

## 1. Estrutura de Alto Nível

```json
{
  "request_id": "req_123456",
  "candidate": {
    "profile": {...},
    "skills": {...},
    "quality_signals": {...},
    "behavioral_signals": {...},
    "embeddings": {...}
  },
  "job_context": {
    "metadata": {...},
    "requirements": {...},
    "embeddings": {...}
  }
}
```

- **Todos os blocos são obrigatórios**, porém cada campo interno define fallback explícito (`null`, lista vazia, escalar default).
- O schema será versionado via `ImmutablePayloadV2` (Pydantic + JSON Schema) antes de liberar o endpoint `/v2/match`.

---

## 2. Mapeamento de Sinais (Candidate)

| # | Campo | Tipo | Fonte / Derivação | Fallback / Observações |
| :- | :--- | :--- | :--- | :--- |
| 1 | `candidate.profile.resume_text` | `str` | Upload (`/predict_file`) ou payload direto | Rejeitar se vazio |
| 2 | `candidate.profile.years_experience_range` | `Enum(str)` | LLM (`prompt_candidato`) normalizado | `"unknown"` |
| 3 | `candidate.profile.seniority_inferred` | `Enum(str)` | Heurística + LLM (faixas do CV) | `"unknown"` |
| 4 | `candidate.profile.education_level` | `Enum(str)` | Campos `app_fei_*` | `"not_declared"` |
| 5 | `candidate.profile.field_of_study` | `str` | `app_fei_cursos` / extração CV | `null` |
| 6 | `candidate.profile.has_degree` | `bool` | `education_level` ∈ {bacharelado+} | `false` |
| 7 | `candidate.profile.languages` | `list[str]` | `app_fei_nivel_ingles/espanhol/outro` | lista vazia |
| 8 | `candidate.profile.language_proficiency` | `dict` | Mesmos campos (níveis padronizados) | `{"english":"unknown"}` |
| 9 | `candidate.profile.availability` | `str` | texto livre CV (`disponível`, `PJ`, `CLT`) | `"unspecified"` |
| 10 | `candidate.skills.technical_skills` | `list[str]` | `competencias_tecnicas` do LLM | lista vazia |
| 11 | `candidate.skills.soft_skills` | `list[str]` | `competencias_comportamentais` | lista vazia |
| 12 | `candidate.skills.tools` | `list[str]` | `ferramentas_tecnologicas` | lista vazia |
| 13 | `candidate.skills.certifications` | `list[str]` | `certificacoes` (pipeline + LLM) | lista vazia |
| 14 | `candidate.quality_signals.has_email` | `bool` | Flags `ind_app_email` | `false` |
| 15 | `candidate.quality_signals.has_phone` | `bool` | Flags `ind_app_telefone` | `false` |
| 16 | `candidate.quality_signals.has_linkedin` | `bool` | Flags `ind_app_linkedin` | `false` |
| 17 | `candidate.quality_signals.has_address` | `bool` | Flags `ind_app_endereco` | `false` |
| 18 | `candidate.quality_signals.has_social_profile` | `bool` | Flags `ind_app_facebook` | `false` |
| 19 | `candidate.quality_signals.completeness_score` | `float` | `% campos != null` (Seção 3.3 do FE doc) | `0.0` |
| 20 | `candidate.quality_signals.is_local_to_job` | `bool` | `ind_mesma_localidade` (rapidfuzz) | `false` |
| 21 | `candidate.behavioral_signals.days_since_profile_update` | `int` | `dias_desde_ultima_atualizacao` | `-1` |
| 22 | `candidate.behavioral_signals.days_in_process` | `int` | `dias_no_processo` | `-1` |
| 23 | `candidate.behavioral_signals.application_velocity` | `int` | `dias_para_se_candidatar` | `-1` |
| 24 | `candidate.behavioral_signals.recruiter_touchpoints` | `int` | Contagem `p_recrutador`/eventos | `0` |
| 25 | `candidate.behavioral_signals.sentiment_score` | `int` | Regex positivos/negativos (`feature_engineering_analysis.md`) | `0` |
| 26 | `candidate.embeddings.semantic_vector` | `list[float]` | Client-side ou `EmbeddingsService` interno | Normalizar para zeros se ausente |

---

## 3. Mapeamento de Sinais (Job Context)

| # | Campo | Tipo | Fonte / Derivação | Fallback / Observações |
| :- | :--- | :--- | :--- | :--- |
| 27 | `job_context.metadata.job_title` | `str` | `job_ib_titulo_vaga` | obrigatório |
| 28 | `job_context.metadata.department` | `str` | `job_ib_empresa_divisao` / extração | `"general"` |
| 29 | `job_context.metadata.recruiter_id` | `str` | `job_ib_requisitante` (`Outros` se <10) | `"outros"` |
| 30 | `job_context.metadata.contract_type` | `Enum(str)` | `job_ib_tipo_contratacao` | `"unspecified"` |
| 31 | `job_context.metadata.days_since_opening` | `int` | `job_ib_data_requicisao` | `-1` |
| 32 | `job_context.metadata.location` | `str` | `job_pv_cidade/bairro` concatenado | `"remote"` |
| 33 | `job_context.requirements.required_tech_skills` | `list[str]` | `job_pv_competencia_tecnicas_e_comportamentais` (split) | lista vazia |
| 34 | `job_context.requirements.required_soft_skills` | `list[str]` | Mesmo campo, filtrando soft skills | lista vazia |
| 35 | `job_context.requirements.target_seniority` | `Enum(str)` | `job_pv_nivel_profissional` | `"unspecified"` |
| 36 | `job_context.requirements.nice_to_have_skills` | `list[str]` | `job_pv_demais_observacoes` (LLM parse) | lista vazia |
| 37 | `job_context.requirements.tools` | `list[str]` | `job_pv_equipamentos_necessarios` | lista vazia |
| 38 | `job_context.requirements.soft_constraints` | `list[str]` | `job_pv_habilidades_comportamentais_necessarias` | lista vazia |
| 39 | `job_context.embeddings.description_vector` | `list[float]` | Vetorização do JD (mesma dimensão do candidato) | zeros se ausente |
| 40 | `job_context.embeddings.requirements_vector` | `list[float]` | Fusão texto de requisitos | zeros se ausente |

---

## 4. Próximos Passos

1. **JSON Schema + Pydantic**: gerar `ImmutablePayloadV2` (sem tocar no FastAPI até aprovação).
2. **Serviços Derivados**: implementar utilitários (`calc_completeness_score`, `infer_seniority`, `EmbeddingService`) dentro de `data_pipeline/pipe/features/`.
3. **Validação e Testes**: criar testes unitários para cada derivação e para o schema (via `pytest`).
4. **Endpoint `/v2/match`**: somente após validação do schema; seguirá com feature flag solicitada pelo time.

Qualquer alteração nessa lista de sinais ou no contrato precisa de aprovação do time (especialmente para campos adicionais que impactem clientes).

## 5. Modelos Pydantic (Implementação de Referência)

Abaixo, a definição formal das estruturas de dados prevista para o SDK/API.

```python
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
    resume_text: str = Field(..., description="Texto cru do currículo")
    years_experience_range: str = "unknown"
    seniority_inferred: Seniority = Seniority.UNKNOWN
    education_level: EducationLevel = EducationLevel.NOT_DECLARED
    field_of_study: Optional[str] = None
    has_degree: bool = False
    languages: List[str] = []
    availability: str = "unspecified"

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
    profile: CandidateProfile
    skills: CandidateSkills
    quality_signals: QualitySignals
    behavioral_signals: CandidateBehavioral
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
    metadata: JobMetadata
    requirements: JobRequirements
    embeddings: Dict[str, List[float]] = {}

class ImmutablePayloadV2(BaseModel):
    request_id: str
    candidate: CandidateData
    job_context: JobData
```
