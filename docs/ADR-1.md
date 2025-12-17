# ADR-1: Arquitetura em Nuvem & Estratégia DevOps (Fase 3+)

## Status

Aceito

## Contexto

Mover de um MVP local para um ambiente de produção escalável requer uma arquitetura em nuvem robusta. Precisamos definir pipelines de armazenamento, computação, orquestração, serving e CI/CD usando serviços AWS e Infraestrutura como Código (IaC).

## Decisões

### 1. Armazenamento & Gerenciamento de Dados

- **Data Lake**: **AWS S3**
  - `s3://datathon-mlet03-raw/`: Arquivos brutos (PDFs, docs).
  - `s3://datathon-mlet03-curated/`: Arquivos Parquet processados.
  - `s3://datathon-mlet03-artifacts/`: Artefatos de modelo (backend MLflow).
- **Feature Store**: **Amazon SageMaker Feature Store**
  - *Decisão*: Adotar SageMaker Feature Store para ingestão e recuperação gerenciada de features, substituindo lookups locais de Parquet em produção.
  - *Trade-off*: Custo mais alto vs. consistência e baixa latência.

### 2. Treinamento de Modelo & Registro

- **Computação de Treino**: **Amazon SageMaker Training Jobs**
  - Desacoplado da orquestração. Permite uso de instâncias GPU apenas quando necessário.
- **Model Registry**: **MLflow na AWS** (Gerenciado ou Self-hosted no SageMaker/EC2)
  - *Decisão*: Avaliar o uso profundo do SageMaker Experiments ou hospedar um servidor MLflow leve no ECS/Fargate com RDS (metadados) e S3 (artefatos).

### 3. Orquestração (Airflow)

- **Opção A (Custo Otimizado)**: Airflow no **EC2 (Bastion/Micro)**.
  - *Prós*: Controle de custos por falha, controle total.
  - *Contras*: Overhead de manutenção (atualizações, patches de segurança).
- **Opção B (Gerenciado)**: **Amazon MWAA** (Managed Workflows for Apache Airflow).
  - *Prós*: Manutenção zero, auto-scaling, segurança integrada.
  - *Contras*: Custo base mínimo (~$300/mês).
- **Recomendação**: Começar com **Opção A (EC2)** para dev/test para economizar custos. Migrar para **MWAA** se a contagem de DAGs ou tamanho do time crescer significativamente.

### 4. Serving de Modelo (API)

- **Arquitetura**: **Serverless (AWS Lambda + API Gateway)**
  - *Container*: Dockerizar aplicação FastAPI.
  - *Computação*: AWS Lambda (Suporte a Imagem de Container).
  - *Gateway*: HTTP API Gateway para roteamento.
- **Modelos**:
  - **LLM**: Usar **Amazon Bedrock** (Claude/Titan) ou **SageMaker Endpoints** (DeepSeek/Llama) via APIs padrão para evitar hospedar pesos pesados no Lambda.
  - **Scorers**: Modelos leves Scikit-Learn/LightGBM carregados dentro do Lambda (ou armazenados no EFS se grandes).
- **OCR**:
  - Usar **Textract** (Gerenciado) ou **PaddleOCR** otimizado em uma task maior do Lambda/Fargate se o custo permitir.

### 5. Infraestrutura como Código (IaC)

- **Ferramenta**: **Terraform**
  - Todos os recursos (S3, IAM, SageMaker, Lambda, API Gateway) definidos em módulos HCL.
  - Estado armazenado em backend remoto S3 com locking DynamoDB.

### 6. DevOps & CI/CD

- **Pipeline**: **GitHub Actions**
- **Estágios**:
    1. **Lint & Teste Unitário**: Pytest, Flake8.
    2. **Build**: Build Docker & push para ECR.
    3. **Plano Infra**: `terraform plan`.
    4. **Deploy (Staging)**: `terraform apply` + Sincronização Delta para S3.
    5. **Teste de Integração**: Rodar `test_phase3.py` contra endpoint da API de Staging.
    6. **Release**: Aprovação manual -> Promover para Prod.

### 7. Observabilidade & Logging

- **Logs**: **Amazon CloudWatch Logs**
  - *Decisão*: Adotar CloudWatch como a solução de logging centralizada para tasks Lambda, API Gateway e ECS/SageMaker.
  - *Racional*: Integração nativa com stack AWS Serverless. Elimina a necessidade de gerenciar uma stack ELK self-hosted para depuração padrão.
  - *Tracing*: **AWS X-Ray** para rastreamento distribuído através de Lambda e API Gateway.

### 8. LLM Ops & Gerenciamento de Prompt

- **Decisão**: **Langfuse** (Recomendado)
- **Análise de Trade-off**:
  - **Opção A: Langfuse**
    - *Prós*: **Agnóstico ao Modelo** (funciona perfeitamente com SageMaker Endpoints, Bedrock, OpenAI e LLMs Locais), **Tracing/Observabilidade** superior (quebra de latência, rastreamento de custo por usuário) e Código Aberto (sem vendor lock-in).
    - *Contras*: Requer configuração de ECS/RDS para self-hosting (overhead operacional).
  - **Opção B: Amazon Bedrock Prompt Management**
    - *Prós*: Totalmente gerenciado (Zero ops), integração nativa IAM.
    - *Contras*: **Fortemente acoplado ao Amazon Bedrock**. Como podemos usar modelos no SageMaker (Open Source/Fine-tuned) ou misturar provedores, a ferramenta do Bedrock pode restringir nossa flexibilidade para rastrear essas chamadas externas.
- **Estratégia de Deploy (Langfuse na AWS)**:
  - **Computação**: **Amazon ECS (Fargate)** garante gerenciamento de container serverless sem manutenção de EC2.
  - **Banco de Dados**: **Amazon RDS (PostgreSQL)** para armazenamento persistente de traces e prompts.
  - *Acesso*: PrivateLink/VPC para tráfego interno seguro.

## Consequências

- **Escalabilidade**: API Serverless escala a zero e se adapta a picos.
- **Manutenibilidade**: Infraestrutura é código versionado.
- **Custo**: Pay-per-use para API e Computação. Feature Store e MWAA têm custos fixos para monitorar.
