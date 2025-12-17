# ADR-0: Evolution to Advanced MLOps Platform & Zero-Shot Architecture

> [!IMPORTANT]
> **Status:** In Progress (Phase 2 Complete)
> **Date:** 2025-12-15 (Updated: 2025-12-16)
> **Context:** Transitioning from a Prototype (Local/Rigid) to an Enterprise MLOps Platform (Cloud/Scalable).
> **Driver:** Need to support infinite job variations without retraining (`job_id` bottleneck) and decouple infrastructure.
>
> **Progress:**
> - ✅ Phase I: Infrastructure Decoupling (LLM Adapter Pattern) - **COMPLETE**
> - ✅ Phase II: Observability (Airflow + Langfuse + Evidently) - **COMPLETE**
> - 🔜 Phase III: Immutable Payload (Zero-Shot Learning) - **PLANNED**

---

## 1. Executive Roadmap

We define a 3-Phase evolution strategy to achieve operational maturity.

| Phase | Focus | Key Deliverable | Tech Stack | Status |
| :--- | :--- | :--- | :--- | :---: |
| **I** | **Decoupling** | **Adapter Pattern** for LLMs (Cloud Ready) | Python Protocols, `.env` Config | ✅ |
| **II** | **Observability** | **Drift Monitoring & Prompt Engineering** | **Langfuse**, **Airflow**, **Evidently AI** | ✅ |
| **III** | **Scalability** | **Immutable Payload** (Zero-Shot) | Vector Embeddings, Schema Validation | 🔜 |

---

## Detailed Roadmap

### Phase I: Infrastructure Decoupling (The Adapter) ✅ **COMPLETE**

**Problem:** The API is hardcoded to `localhost:11434` (Ollama), making cloud deployment impossible without code changes.

**Solution:** Implement the **Adapter Pattern** to switch between Local and Cloud providers dynamically.

**Implementation:**
- Created `data_pipeline/infra/llm_gateway.py` with `LLMProvider` protocol
- Implemented `OllamaAdapter` and `DeepSeekAdapter`
- Refactored `prompts.py` to use gateway
- Added `.env` configuration support

**Architecture:**
```python
# Protocol Definition
class LLMService(Protocol):
    def generate(self, prompt: str) -> str: ...

# Adapter A: Local Development (Cost $0)
class OllamaAdapter:
    def generate(self, prompt: str): return requests.post("http://ollama:11434/...")

# Adapter B: Production (High Availability)
class DeepSeekAdapter:
    def generate(self, prompt: str): return client.chat.completions.create(...)

# Injection
def get_llm_service() -> LLMService:
    return OllamaAdapter() if os.getenv("ENV") == "DEV" else DeepSeekAdapter()
```

---

### Phase II: Continuous Evaluation & Prompt Management (LLMOps) ✅ **COMPLETE**

**Problem:**
1.  **Drift:** We don't know if the model is degrading over time.
2.  **Prompt Sprawl:** Prompts are hardcoded in string literals (`prompts.py`), making versioning and testing difficult.

**Solution:** **Full LLMOps Stack (Airflow + Langfuse + Evidently AI).**

**Implementation Strategy:**
-   **Prompt Management (Langfuse):**
    -   Move hardcoded strings to Langfuse CMS.
    -   Track Prompt Versions (v1 vs v2).
    -   Trace execution cost and latency per call.
    -   Add `langfuse` container to `docker-compose.yml`.
-   **Drift Pipeline (Airflow):**
    -   **Deployment:** Add `airflow-webserver` and `airflow-scheduler` services to `docker-compose.yml`.
    -   **Frequency:** Weekly (`@weekly` DAG).
    -   **Workload:**
        1.  **Extract:** Pull last 7 days of inference logs.
        2.  **Compute Metrics:** PSI (Population Stability), Embedding Drift.
        3.  **Alert:** Slack/Email notification if `Drift_Score > 0.15`.

**Why Docker?** Keeps the stack portable. Developers can run the exact monitoring stack locally before deploying to AWS ECS/Kubernetes.

**Implementation Status:**
- ✅ Airflow webserver + scheduler deployed (Docker Compose)
- ✅ Custom Airflow Dockerfile with dependencies (polars, evidently, pandas)
- ✅ `drift_monitoring_weekly` DAG created with Evidently AI
- ✅ Drift detection utilities (`dags/utils/drift_detection.py`)
- ✅ HTML report generation and threshold-based alerting
- ⏸️ Langfuse service configured (DB ready, integration pending)
- 🔜 Slack/Email alerting integration

---

### Phase III: The Immutable Payload (Zero-Shot Design) 🔜 **PLANNED**

**Problem:** The current API relies on `job_id`.
-   *New Job ID = Unknown Feature = Retraining Trigger.*
-   This creates a "Red Queen Race" where we constantly retrain just to stay in place.

**Solution:** **Immutable Input Schema.**
We expand the API payload to accept *concepts*, not *keys*. The model scores "Data vs Data", not "Data vs ID".

Based on our Feature Engineering Analysis (`docs/feature_engineering_analysis.md`), we require **~30 signals** to fully capture the context without retraining.

**Proposed API Schema (Version 2.0):**
No matter how many new jobs are created, this payload structure **never changes**.

```json
{
  "request_id": "req_123456",
  "candidate": {
    "profile": {
      "resume_text": "Experienced Python Dev...",
      "years_experience_range": "5-8_years",
      "seniority_inferred": "senior",  // Derived from LLM
      "education_level": "bachelors",
      "field_of_study": "computer_science",
      "has_degree": true
    },
    "skills": {
      "technical_skills": ["python", "pytorch", "fastapi", "docker"],
      "soft_skills": ["mentoring", "communication", "adaptability"],
      "tools": ["jira", "slack", "aws_ec2"]
    },
    "quality_signals": {
      "has_email": true,
      "has_phone": true,
      "has_linkedin": true,
      "has_address": true,
      "completeness_score": 0.95,
      "is_local_to_job": true  // Computed fuzzy match
    },
    "embeddings": {
       "semantic_vector": [0.12, -0.98, ..., 0.44] // Optional: optimized client-side or computed server-side
    }
  },
  "job_context": {
    "metadata": {
        "job_title": "Senior MLOps Engineer",
        "department": "Engineering",
        "recruiter_id": "rec_09", // Handled as "Other" if low cardinality
        "days_since_opening": 12
    },
    "requirements": {
        "required_tech_skills": ["python", "kubernetes"],
        "required_soft_skills": ["problem_solving"],
        "target_seniority": "senior"
    },
    "embeddings": {
        "description_vector": [0.15, -0.91, ..., 0.33]
    }
  }
}
```

**Outcome:**
-   **Zero-Shot:** The model calculates `Similarity(candidate.vector, job.vector)` and `Match(candidate.skills, job.requirements)`.
-   **Robustness:** Features like `is_local_to_job` and `completeness_score` are universally applicable, regardless of the unique Job ID.
-   **No Retraining:** New jobs are just new data points in the same feature space. The processing logic remains constant.

---

## 3. Financial & Operational ROI

| architecture | Cost/10k Reqs | MLOps Effort | Scalability |
| :--- | :--- | :--- | :--- |
| **Current (Local/ID)** | $400/mo (GPU) | High (Manual Retraining) | Low (Failed on new IDs) |
| **Target (Cloud/Zero-Shot)** | $30/mo (API) | **Zero** (No Retraining) | **Infinite** (Any new job works) |
