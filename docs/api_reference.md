# Referência da API

## 1. Introdução
A **Recruitment Scoring API** é um serviço RESTful desenvolvido em FastAPI. Ela serve como a interface de inferência para o sistema de triagem inteligente, permitindo que sistemas externos (ATS, Portais de Vagas) obtenham scores de candidatos em tempo real.

## 2. Especificação Técnica
*   **Protocolo**: HTTP/1.1
*   **Formato de Dados**: JSON (Application/JSON)
*   **Porta Padrão**: 8000
*   **Autenticação**: *Aberta para MVP (Recomenda-se OAuth2 ou API Key para produção)*.

## 3. Endpoints

### 3.1 Health Check
Verifica se o serviço e os modelos estão carregados e operacionais.

*   **URL**: `/health`
*   **Método**: `GET`
*   **Resposta de Sucesso (200 OK)**:
    ```json
    {
      "status": "ok",
      "models_loaded": true
    }
    ```

### 3.2 Predict Score
Calcula os scores de Skills, Comportamental e Cultural para um dado currículo.

*   **URL**: `/predict`
*   **Método**: `POST`
*   **Corpo da Requisição (Request Body)**:
    ```json
    {
      "resume_text": "Texto completo do currículo...",
      "job_id": "12345 (Opcional: Busca dados da vaga no banco)",
      "job_description": "Descrição da vaga (Opcional: Se não passar ID)"
    }
    ```
    *Nota: Pelo menos `job_id` OU `job_description` deve ser fornecido para um cálculo preciso do Score de Skills.*

*   **Resposta de Sucesso (200 OK)**:
    ```json
    {
      "candidate_extracted": {
        "competencias_tecnicas": ["Python", "Docker", "AWS"],
        "senioridade_aparente": "Sênior",
        "experiencia_anos": "5-8 anos",
        "nivel_formacao": "Superior Completo"
      },
      "scores": {
        "skills": 0.88,       // float: 0.0 - 1.0
        "behavioral": 0.65,   // float: 0.0 - 1.0 (0.5 = neutro/sem histórico)
        "cultural": 0.72      // float: 0.0 - 1.0
      }
    }
    ```

*   **Códigos de Erro**:
    *   `400 Bad Request`: Body inválido ou `resume_text` vazio.
    *   `500 Internal Server Error`: Falha na extração (LLM Timeout) ou erro interno no cálculo dos scores.

## 4. Exemplos de Uso (CURL)

### Calcular score comparando com descrição de vaga ad-hoc
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "resume_text": "Desenvolvedor Python com 5 anos de experiência em Django e Flask. Conhecimento em AWS.", 
           "job_description": "Vaga para Backend Python. Necessário experiência com APIs REST e Cloud Computing."
         }'
```

### Calcular score para uma vaga existente (pelo ID)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "resume_text": "Candidato a Analista de Dados. Domínio de SQL e PowerBI.", 
           "job_id": "VAGA_DATA_001" 
         }'
```

## 5. Guia de Implantação (Deployment)
Para ambiente de produção, não utilize o servidor de desenvolvimento (`uvicorn ... --reload`). Utilize um gerenciador de processos como Gunicorn com Uvicorn workers.

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker serving.api:app --bind 0.0.0.0:8000
```
*   **Workers (-w 4)**: Ajuste conforme o número de cores da CPU.
*   **Timeouts**: Como a extração via LLM pode demorar, configure timeouts generosos no Gunicorn (`--timeout 120`).
