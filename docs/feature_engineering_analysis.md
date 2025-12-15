# Feature Engineering & Data Pipeline Deep Dive

This document provides a comprehensive technical analysis of the feature engineering pipeline, covering data ingestion, cleaning, transformation, and model-specific preprocessing.

## 1. Data Ingestion & Flattening Strategy
*Found in: `jobs_feature_enginerring.ipynb`*

The pipeline begins by ingesting three core hierarchical JSON datasets and flattening them into a tabular structure.

- **Jobs (`jobs.json`)**: Flattened with prefix `job_`.
- **Applicants (`applicants.json`)**: Flattened with prefix `app_`.
- **Prospects (`prospects.json`)**: Flattened with prefix `p_`.
- **Merge Logic**:
    - Central Table: `prospects` (representing the application event).
    - Joins: Inner joins on `codigo_candidato` (Applicant) and `codigo_vaga` (Job).

## 2. Privacy & PII Management (Data Cleaning)
*Found in: `jobs_feature_enginerring.ipynb`*

A "Signal Preservation" strategy is used where sensitive data is removed but the *fact* of its existence is preserved as a feature.

### 2.1. Feature Extraction (Before Dropping)
Binary flags created to indicate profile completeness/quality:
| Feature Name | Derived From (Dropped Column) | Logic |
| :--- | :--- | :--- |
| `ind_app_telefone` | `app_ib_telefone` | Not Null & Not Empty |
| `ind_app_email` | `app_ib_email` | Not Null & Not Empty |
| `ind_app_linkedin` | `app_ip_url_linkedin` | Not Null & Not Empty |
| `ind_app_endereco` | `app_ip_endereco` | Not Null & Not Empty |
| `ind_app_facebook` | `app_ip_facebook` | Not Null & Not Empty |

### 2.2. Locality Matching (Fuzzy Logic)
- **Goal**: Determine if candidate is local to the job.
- **Method**: `rapidfuzz` partial ratio matching.
- **Comparison**: Cross-compares Candidate Locations (`app_ib_local`, `app_ip_endereco`) vs. Job Locations (`job_pv_cidade`, `job_pv_bairro`, etc.).
- **Threshold**: Similarity Score $\ge 70$.
- **Output**: `ind_mesma_localidade` (Boolean).

## 3. Model-Ready Feature Engineering
*Found in: `Model_engagement_score_final.py`*

Crucial transformations happen immediately prior to model training (LightGBM).

### 3.1. Target Variable Definition (`engajado`)
The target `engajado` (Engaged) is a binary classification (1/0) derived from `p_situacao_candidado`.
*   **Positive Class (1)**:
    *   "CONTRATADO PELA DECISION"
    *   "CONTRATADO COMO HUNTING"
    *   "DOCUMENTAÇÃO PJ"
    *   "ENCAMINHADO AO REQUISITANTE"
    *   "APROVADO"
    *   "ENTREVISTA TÉCNICA"
*   **Negative Class (0)**: All other statuses.

### 3.2. Behavioral & Sentiment Features (Regex)
Text analysis on recruiter comments (`p_comentario`) creates powerful behavioral signals.

*   **Negative Sentiment Regex**:
    *   Pattern: `(?i)(n(a|ã)o\sresponde|desisti|sem\sinteresse|n(a|ã)o\sretorna|n(a|ã)o\satende)`
    *   Feature: `contem_palavra_chave_negativa`
*   **Positive Sentiment Regex**:
    *   Pattern: `(?i)(proativ|interessad|bom\sperfil|responsiv|gostou|avan(ç|c)ou|performou\sbem)`
    *   Feature: `contem_palavra_chave_positiva`
*   **Sentiment Score**: `sentimento_comentario_score` (+1 if positive match, -1 if negative, 0 otherwise).

### 3.3. Profile Completeness Index
A composite score `percentual_perfil_completo` is calculated as the % of non-null fields from this specific list:
1.  `app_ib_objetivo_profissional`
2.  `app_ib_local`
3.  `app_ip_sexo`
4.  `app_ip_estado_civil`
5.  `app_ip_pcd`
6.  `app_ip_remuneracao`
7.  `app_fei_nivel_academico`
8.  `app_fei_nivel_ingles`

### 3.4. Temporal Features (Time Deltas)
All dates converted to epoch for numerical difference calculation:
-   **`dias_para_se_candidatar`**: Application Date - Job Requisition Date. (Measures reactivity to new market openings).
-   **`dias_desde_ultima_atualizacao`**: Current Date - Last Update Date. (Measures staleness).
-   **`dias_no_processo`**: Last Update Date - Application Date. (Measures process duration).

### 3.5. Categorical Grouping (Cardinality Reduction)
-   **Field**: `p_recrutador`
-   **Threshold**: Recruiters with `< 10` occurrences are grouped into category `"Outros"` to prevent overfitting on rare labels.

## 4. Preprocessing & Model Architecture

### 4.1. Preprocessing Pipeline (`ColumnTransformer`)
-   **Numerical/Binary Features**: Scaled using `StandardScaler` (Z-score normalization).
    -   Includes: `tamanho_cv`, time deltas, sentiment scores, binary flags.
-   **Categorical Features**: Encoded using `OneHotEncoder` (`handle_unknown='ignore'`).
    -   Includes: `p_recrutador_tratado`.

### 4.2. Model Selection
-   **Algorithm**: `LightGBM` (`LGBMClassifier`)
-   **Handling Imbalance**: `is_unbalance=True` (Critical for hiring data where hires are minority events).
-   **Optimization**: `RandomizedSearchCV` (50 iterations, 5-fold CV) optimizing for **ROC-AUC**.
