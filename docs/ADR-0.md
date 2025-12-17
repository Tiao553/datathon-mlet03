# ADR-0: Evolução para Plataforma MLOps Avançada & Arquitetura Zero-Shot

> [!IMPORTANT]
> **Status:** Em Andamento (Fase 2 Completa)
> **Data:** 15-12-2025 (Atualizado: 16-12-2025)
> **Contexto:** Transição de um Protótipo (Local/Rígido) para uma Plataforma MLOps Empresarial (Cloud/Escalável).
> **Driver:** Necessidade de suportar infinitas variações de vagas sem retreinamento (gargalo do `job_id`) e desacoplar a infraestrutura.
>
> **Progresso:**
>
> - ✅ Fase I: Desacoplamento de Infraestrutura (Padrão Adapter para LLM) - **CONCLUÍDO**
> - ✅ Fase II: Observabilidade (Airflow + Langfuse + Evidently) - **CONCLUÍDO**
> - ✅ Fase III: Payload Imutável (Aprendizado Zero-Shot) - **PROJETADO** (Ver `docs/phase3_payload_schema.md`)

---

## 1. Roadmap Executivo

Definimos uma estratégia de evolução em 3 fases para atingir a maturidade operacional.

| Fase | Foco | Principal Entregável | Tech Stack | Status |
| :--- | :--- | :--- | :--- | :---: |
| **I** | **Desacoplamento** | **Padrão Adapter** para LLMs (Pronto para Cloud) | Protocolos Python, Config `.env` | ✅ |
| **II** | **Observabilidade** | **Monitoramento de Drift & Engenharia de Prompt** | **Langfuse**, **Airflow**, **Evidently AI** | ✅ |
| **III** | **Escalabilidade** | **Payload Imutável** (Zero-Shot) | Embeddings Vetoriais, Validação de Schema | 🔜 |

---

## Detalhamento do Roadmap

### Fase I: Desacoplamento de Infraestrutura (O Adapter) ✅ **CONCLUÍDO**

**Problema:** A API estava hardcoded para `localhost:11434` (Ollama), impossibilitando o deploy em nuvem sem alterações no código.

**Solução:** Implementar o **Padrão Adapter** para alternar entre provedores Local e Cloud dinamicamente.

**Implementação:**

- Criado `data_pipeline/infra/llm_gateway.py` com protocolo `LLMProvider`
- Implementados `OllamaAdapter` e `DeepSeekAdapter`
- Refatorado `prompts.py` para usar o gateway
- Adicionado suporte a configuração via `.env`

**Arquitetura:**

```python
# Definição do Protocolo
class LLMService(Protocol):
    def generate(self, prompt: str) -> str: ...

# Adapter A: Desenvolvimento Local (Custo R$0)
class OllamaAdapter:
    def generate(self, prompt: str): return requests.post("http://ollama:11434/...")

# Adapter B: Produção (Alta Disponibilidade)
class DeepSeekAdapter:
    def generate(self, prompt: str): return client.chat.completions.create(...)

# Injeção
def get_llm_service() -> LLMService:
    return OllamaAdapter() if os.getenv("ENV") == "DEV" else DeepSeekAdapter()
```

---

### Fase II: Avaliação Contínua & Gerenciamento de Prompt (LLMOps) ✅ **CONCLUÍDO**

**Problema:**

1. **Drift:** Não sabemos se o modelo está degradando ao longo do tempo.
2. **Proliferação de Prompts:** Prompts hardcoded em strings (`prompts.py`), dificultando versionamento e testes.

**Solução:** **Stack Completa de LLMOps (Airflow + Langfuse + Evidently AI).**

**Estratégia de Implementação:**

- **Gerenciamento de Prompt (Langfuse):**
  - Mover strings hardcoded para o CMS do Langfuse.
  - Rastrear Versões de Prompt (v1 vs v2).
  - Tracejar custo de execução e latência por chamada.
  - Adicionar container `langfuse` ao `docker-compose.yml`.
- **Pipeline de Drift (Airflow):**
  - **Deploy:** Adicionar serviços `airflow-webserver` e `airflow-scheduler` ao `docker-compose.yml`.
  - **Frequência:** Semanal (DAG `@weekly`).
  - **Carga de Trabalho:**
        1. **Extrair:** Puxar logs de inferência dos últimos 7 dias.
        2. **Calcular Métricas:** PSI (Estabilidade Populacional), Drift de Embeddings.
        3. **Alertar:** Notificação Slack/Email se `Drift_Score > 0.15`.

**Por que Docker?** Mantém a stack portátil. Desenvolvedores podem rodar a stack exata de monitoramento localmente antes de fazer deploy para AWS ECS/Kubernetes.

**Status da Implementação:**

- ✅ Airflow webserver + scheduler deployados (Docker Compose)
- ✅ Dockerfile customizado do Airflow com dependências (polars, evidently, pandas)
- ✅ DAG `drift_monitoring_weekly` criada com Evidently AI
- ✅ Utilitários de detecção de drift (`dags/utils/drift_detection.py`)
- ✅ Geração de relatório HTML e alertas baseados em limiares
- ⏸️ Serviço Langfuse configurado (DB pronto, integração pendente)
- 🔜 Integração de alertas Slack/Email

---

### Fase III: O Payload Imutável (Design Zero-Shot) 🔜 **PLANEJADO**

**Problema:** A API atual depende de `job_id`.

- *Novo Job ID = Feature Desconhecida = Gatilho de Retreinamento.*
- Isso cria uma "Corrida da Rainha Vermelha" onde constantemente retreinamos apenas para permanecer no lugar.

**Solução:** **Schema de Input Imutável.**
Expandimos o payload da API para aceitar *conceitos*, não *chaves*. O modelo pontua "Dados vs Dados", não "Dados vs ID".

Baseado em nossa Análise de Engenharia de Features (`docs/feature_engineering_analysis.md`), precisamos de **~30 sinais** para capturar totalmente o contexto sem retreinamento.

**Schema de API Proposto (Versão 2.0):**
Não importa quantas novas vagas sejam criadas, essa estrutura de payload **nunca muda**.

```json
{
  "request_id": "req_123456",
  "candidate": {
    "profile": {
      "resume_text": "Desenvolvedor Python Experiente...",
      "years_experience_range": "5-8_years",
      "seniority_inferred": "senior",  // Derivado de LLM
      "education_level": "bachelors",
      "field_of_study": "computer_science",
      "has_degree": true
    },
    "skills": {
      "technical_skills": ["python", "pytorch", "fastapi", "docker"],
      "soft_skills": ["mentoria", "comunicacao", "adaptabilidade"],
      "tools": ["jira", "slack", "aws_ec2"]
    },
    "quality_signals": {
      "has_email": true,
      "has_phone": true,
      "has_linkedin": true,
      "has_address": true,
      "completeness_score": 0.95,
      "is_local_to_job": true  // Match fuzzy computado
    },
    "embeddings": {
       "semantic_vector": [0.12, -0.98, ..., 0.44] // Opcional: otimizado no client-side ou computado no server-side
    }
  },
  "job_context": {
    "metadata": {
        "job_title": "Engenheiro MLOps Senior",
        "department": "Engenharia",
        "recruiter_id": "rec_09", // Tratado como "Outros" se baixa cardinalidade
        "days_since_opening": 12
    },
    "requirements": {
        "required_tech_skills": ["python", "kubernetes"],
        "required_soft_skills": ["resolucao_problemas"],
        "target_seniority": "senior"
    },
    "embeddings": {
        "description_vector": [0.15, -0.91, ..., 0.33]
    }
  }
}
```

**Resultado:**

- **Zero-Shot:** O modelo calcula `Similaridade(candidate.vector, job.vector)` e `Match(candidate.skills, job.requirements)`.
- **Robustez:** Features como `is_local_to_job` e `completeness_score` são universalmente aplicáveis, independente do Job ID único.
- **Sem Retreinamento:** Novas vagas são apenas novos pontos de dados no mesmo espaço de features. A lógica de processamento permanece constante.

---

## 3. ROI Financeiro & Operacional

| arquitetura | Custo/10k Reqs | Esforço MLOps | Escalabilidade |
| :--- | :--- | :--- | :--- |
| **Atual (Local/ID)** | $400/mês (GPU) | Alto (Retreinamento Manual) | Baixa (Falha em novos IDs) |
| **Alvo (Cloud/Zero-Shot)** | $30/mês (API) | **Zero** (Sem Retreinamento) | **Infinita** (Qualquer nova vaga funciona) |
