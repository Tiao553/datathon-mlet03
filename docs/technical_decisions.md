# Documentação de Decisões Técnicas e Arquitetura

## 1. Arquitetura do Sistema
O diagrama abaixo detalha o fluxo completo da solução, desde a infraestrutura (Docker) até a camada de aplicação e modelagem.

```mermaid
graph TD
    subgraph "Infraestrutura (Docker Compose)"
        API[FastAPI Service<br>:8000]
        MLflow[MLflow Server<br>:5000]
        ES[Elasticsearch<br>:9200]
        LS[Logstash<br>:50000]
        Kibana[Kibana<br>:5601]
        
        API -->|Logs (TCP)| LS
        LS -->|Index| ES
        ES -->|Visualize| Kibana
        API -->|Track Metrics| MLflow
    end

    subgraph "Camada de Aplicação (API)"
        ReqFile[Upload PDF/DOCX] -->|/predict_file| OCR(DocumentParser)
        ReqJson[JSON Payload] -->|/predict| Text(Resume Text)
        
        OCR -->|Extracted Text| Text
        
        Text --> LLM_Ext[LLM Extractor<br>(Ollama/Gemma3)]
        LLM_Ext -->|Struct JSON| Features
        
        Features --> SkillsEng[Skills Engine<br>SBERT + Regras]
        Features --> BehavEng[Behavioral Engine<br>LightGBM]
        Features --> CultEng[Cultural Engine<br>SBERT]
    end

    subgraph "Pipeline de Treinamento (Offline)"
        RawData[Raw JSON] -->|Polars ETL| Curated[Curated Parquet]
        Curated -->|Transforms| FeatureStore[Feature Store]
        
        FeatureStore --> Exp[Run Experiment<br>run_expanded_exp.py]
        Exp -->|Log Model/Metrics| MLflow
        MLflow -->|Load Artifact| BehavEng
    end

    SkillsEng --> Scores
    BehavEng --> Scores
    CultEng --> Scores
    Scores --> Resp[JSON Response]
```

## 2. Decisões de Modelagem (Os 3 Scores)

### 2.1 Score de Skills (Técnico)
**Abordagem**: Híbrida (Semântica + Regras).
*   **Modelo de Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
*   **Lógica**:
    1.  Vetorização das skills da vaga e do candidato.
    2.  Cálculo de Similaridade de Cosseno (Max-Mean) para capturar sinônimos.
    3.  Regras rígidas para Senioridade (peso 60%), Acadêmico (20%) e Idiomas (20%).
*   **Justificativa**: Garante flexibilidade para termos técnicos ("React" == "React.js") e rigor para requisitos mandatórios.

### 2.2 Score Comportamental (Engagement)
**Modelo**: `LightGBM Classifier`.
**Objetivo**: Prever sucesso/engajamento no processo seletivo.
**Features Chave** (Expandidas na Fase 2):
*   **NLP**: Análise de sentimento (Regex) em comentários de recrutadores anteriores.
*   **Temporal**: Dias entre criação do perfil e candidatura (proatividade).
*   **Compliance**: Nível de preenchimento do cadastro.
*   **Normalização**: Uso de pipelines de transformação (`curated_transform`) para garantir consistência treino-inferência.
**Métricas**: AUC ~0.81. Otimizado para Recall na classe minoritária (Contratados) usando pesos balanceados (`is_unbalance=True`).

### 2.3 Score de Fit Cultural
**Abordagem**: Comparação Semântica de Soft Skills.
**Lógica**: Mede a distância vetorial entre "Competências Comportamentais" exigidas pela vaga e declaradas pelo candidato.

## 3. Infraestrutura e Observabilidade

### 3.1 Docker & Microserviços
A solução é totalmente containerizada:
*   **Orquestração**: `docker-compose`.
*   **Isolamento**: A API roda em seu próprio container baseado em `python:3.10-slim`.
*   **Monitoramento (ELK)**:
    *   Logs da aplicação são enviados assincronamente (TCP) para o **Logstash**.
    *   Armazenados no **Elasticsearch** para busca e auditoria.
    *   **Kibana** (opcional) para dashboards de erro e latência.
*   **Experiment Tracking**: **MLflow** centraliza o versionamento de modelos e métricas, permitindo reprodutibilidade.

### 3.2 OCR e Ingestão de Documentos
Para suportar upload de arquivos:
*   **PDF**: Processamento em duas etapas via `pypdf`.
    1.  Tentativa de extração de camada de texto (rápido).
    2.  Fallback para OCR (`tesseract`) se o texto for insuficiente (< 50 chars), resolvendo casos de currículos digitalizados como imagem.
*   **DOCX**: Extração via `python-docx`.

## 4. Tecnologias
*   **Linguagem**: Python 3.10
*   **Processamento**: Polars (ETL), Pydantic (Validação API), pypdf/Tesseract (OCR).
*   **ML/AI**: LightGBM, Sentence-Transformers, Ollama (Local LLM).
*   **Infra**: Docker, Elasticsearch, Logstash, MLflow.
