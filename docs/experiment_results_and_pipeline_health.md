# Relatório de Experimentos e Saúde do Pipeline

## 1. Visão Geral dos Experimentos
Executamos uma rodada tripla de experimentos para validar os pilares da IA de Recrutamento. Todos os modelos foram versionados via MLflow e salvaram artefatos reproduzíveis.

| Modelo | Objetivo | Resultado (Métrica) | Status |
| :--- | :--- | :--- | :--- |
| **Comportamental** | Prever probabilidade de engajamento | **AUC: 0.7246** | Validado (Baseline) |
| **Técnico (Skills)** | Classificar fit técnico (Hard Skills) | **3 Tiers (Clustering)** | Validado |
| **Cultural** | Medir alinhamento de valores/soft skills | **Score de Similaridade** | Validado |

## 2. Detalhes dos Modelos e Dados

### 2.1 Modelo Comportamental (`run_behavioral_baseline.py`)
- **Dados de Entrada**: Histórico de candidatos (`prospects`), candidaturas (`applicants`) e vagas (`jobs`).
- **Feature Engineering Validada**:
    - **Remoção de PII**: Dados sensíveis (Email, Telefone, CPF) foram removidos *antes* da modelagem, mantendo apenas indicadores binários (`ind_app_telefone`, etc.).
    - **Análise de Sentimento (Regex)**: Captura de padrões como "não responde", "interessado", "bom perfil" nos comentários.
    - **Features Temporais**: `dias_no_processo`, `recencia_atualizacao`.
- **Algoritmo**: LightGBM (`is_unbalance=True`) com otimização `RandomizedSearchCV`.

### 2.2 Modelo Técnico / Skills (`run_skills_baseline.py`)
- **Lógica Híbrida**:
    1.  **Match Estruturado (60%)**: Comparação direta de níveis de senioridade, escolaridade e inglês.
    2.  **Soft Match (40%)**: Embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) comparando listas de tecnologias da vaga vs. candidato.
- **Resultado**: Clusterização K-Means gerou 3 grupos claros (Baixo, Médio, Alto Fit).

### 2.3 Modelo Cultural (`run_cultural_baseline.py`)
- **Abordagem**: Similaridade semântica entre "Objetivos/Benefícios da Vaga" e "Objetivos/Comentários do Candidato".

## 3. Diagnóstico do Pipeline de Dados

Respondendo às questões críticas sobre a saúde dos dados:

### 3.1 Estamos Normalizando os Dados?
**SIM.**
- Nos scripts de experimento (`models/experiments`), implementamos explicitamente:
    - `StandardScaler` para features numéricas (`percentual_perfil_completo`, `tamanho_cv`, scores).
    - `OneHotEncoder` para variáveis categóricas (`recrutador`).
    - **Observação**: Esta normalização ocorre *dentro* do pipeline de treinamento (`sklearn.pipeline.Pipeline` ou `ColumnTransformer`). Para inferência, o pipeline salvo (pickle/mlflow) já contém essas transformações, garantindo que novos dados sejam normalizados com a mesma média/desvio padrão do treino.

### 3.2 Estamos Processando Corretamente para os Modelos?
**SIM, com ressalvas corrigidas.**
- **Correção de Chaves**: Identificamos e corrigimos uma falha crítica no join entre `Prospects` (`p_codigo`) e `Applicants` (`id` -> `codigo_candidato`).
- **Prefixação**: Implementamos a renomeação automática de colunas (`job_`, `app_`) para evitar colisões e garantir que as features corretas sejam alimentadas no modelo.
- **Tratamento de Nulos**: Features numéricas como `tamanho_cv` e `percentual_perfil_completo` tratam nulos como zero ou via imputação implícita nos cálculos.

### 3.3 Estamos Tomando Cuidados com Data Drift?
**PARCIALMENTE (Ainda Manual).**
- **Estado Atual**: Não existe um monitoramento *automatizado* (ex: Evidently AI ou WhyLogs) rodando em produção.
- **Mitigação Atual**:
    - O treino utiliza validação cruzada (`StratifiedKFold`) para garantir generalização.
    - O uso de `StandardScaler` mitiga pequenas flutuações nas escalas das variáveis.
- **Risco Identificado**: Mudanças no comportamento dos recrutadores (ex: novos campos de texto, mudanças na forma de preencher "comentários") podem degradar o modelo de sentimento (Regex) silenciosamente.
- **Recomendação**: Implementar um passo de "Data Validation" no pipeline de retreino que compare as distribuições estatísticas das novas cargas com o dataset de referência (Baseline).
