# Arquitetura de Dados e Feature Engineering

## 1. Fluxo de Dados (Pipeline ETL)

O pipeline de dados segue a arquitetura "Medallion" (Bronze/Silver/Gold) adaptada:

1.  **Raw (Bronze)**: Arquivos JSON originais aninhados (`Jobs.json`, `Applicants.json`, `Prospects.json`).
    *   *Desafio*: Estruturas complexas sem schema definido.
2.  **Curated (Silver)**: Dados "achatados" (flattened) em formato tabular `.parquet`.
    *   Colunas renomeadas com prefixos (`job_`, `app_`, `p_`) para evitar colisão em joins.
    *   Tipagem forte (String, Int, Datetime) aplicada via Polars.
3.  **Feature Store (Gold)**: Dados prontos para modelagem.
    *   Features calculadas (`dias_processo`, `score_sentimento`).
    *   Dados textuais estruturados via LLM (Currículos parseados).
    *   PII (Informação Pessoal Identificável) removida ou ofuscada.

## 2. Tratamento de PII (Privacidade)
Antes da modelagem, o módulo `cleanning_and_accurate.py` remove dados sensíveis para conformidade com LGPD/GDPR:
*   **Removidos**: `p_nome`, `app_ib_telefone`, `app_ib_email`, `app_ip_cpf`, `app_ip_endereco`, links de redes sociais.
*   **Preservados (Features Derivadas)**: Em vez do dado bruto, criamos flags binárias indicando presença:
    *   `ind_app_email`: 1 se tem email, 0 caso contrário.
    *   `ind_app_linkedin`: 1 se tem LinkedIn, 0 caso contrário.
    *   `ind_mesma_localidade`: Comparação fuzzy (0-100) entre cidade do candidato e da vaga, sem expor o endereço.

## 3. Catálogo de Dados (Dicionário Simplificado)

### 3.1 Tabela de Vagas (`jobs.parquet`)
| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `codigo_vaga` | String | ID único da vaga. |
| `job_competencias_tecnicas` | List[str] | Lista limpa de hard skills exigidas. |
| `job_competencias_comportamentais` | List[str] | Lista limpa de soft skills exigidas. |
| `job_nivel_senioridade` | String | Senioridade (Junior, Pleno, Senior). |
| `job_ib_data_criacao` | Datetime | Data de abertura da vaga. |

### 3.2 Tabela de Candidatos (`applicants.parquet`)
| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `codigo_candidato` | String | ID único do candidato. |
| `app_cv_pt` | String | Texto completo (OCR/Raw) do currículo em português. |
| `app_competencias_tecnicas` | List[str] | Extraídas do CV via LLM. |
| `app_senioridade_aparente` | String | Senioridade inferida pelo LLM. |
| `app_formacao_academica` | Bool | Flag indicando se possui ensino superior. |

## 4. Engenharia de Features Detalhada

### 4.1 Extração de Currículos (LLM)
Transformamos o texto não estruturado em dados tabulares usando `Ollama (Gemma3:1b)`.
**Prompting**: Utilizamos Few-Shot prompting implícito (nas instruções) para garantir formato JSON estrito.
**Batching**: Processamento paralelo local com `ThreadPoolExecutor` para maximizar I/O, salvo em `.jsonl` incremental para tolerância a falhas.

### 4.2 Features Temporais (`time_features`)
Calculadas para capturar a urgência e o "timing" da candidatura:
*   `temp_criacao_candidatura`: Dias entre criação do perfil e candidatura (mede proatividade imediata).
*   `dias_vaga_aberta`: Quanto tempo a vaga ficou aberta até o match.
*   `candidatura_antes_abertura`: Flag de erro ou candidatura interna (data candidata < data abertura).

### 4.3 Features de Texto e Sentimento
Para o score comportamental, analisamos os comentários históricos dos recrutadores (`p_comentario`):
*   **Lógica**: Regex com dicionário de palavras.
*   **Positivas**: "interessado", "motivado", "excelente", "aprovado".
*   **Negativas**: "desistiu", "fraco", "sem experiência", "recusou".
*   **Feature Resultante**: `sentimento_comentario_score` (-1, 0, 1).
