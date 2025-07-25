# Datathon Machine Learning Engineering

[![Assista à apresentação no YouTube](https://img.shields.io/badge/YouTube-Apresentação-red?logo=youtube)](https://youtu.be/v03U9tBDizg)

Clique no badge acima ou no link abaixo para assistir à apresentação do projeto:

🔗 [Apresentação no YouTube](https://youtu.be/v03U9tBDizg)

---

## Contexto

Este projeto foi desenvolvido como parte do Datathon Pós Tech, com o objetivo de aplicar Inteligência Artificial para solucionar desafios reais de uma empresa do setor de bodyshop e recrutamento, a **Decision**. A empresa busca otimizar o processo de recrutamento e seleção, conectando talentos qualificados às necessidades dos clientes, principalmente no setor de TI, onde agilidade e precisão no “match” entre candidatos(as) e vagas são essenciais.

## Desafios da Empresa

- Falta de padronização em entrevistas, gerando perda de informações valiosas.
- Dificuldade em identificar o real engajamento dos candidatos(as).
- Necessidade de alinhar habilidades técnicas, fit cultural e motivação dos candidatos(as) às vagas.
- Processo manual e pouco escalável para encontrar o(a) candidato(a) ideal em tempo hábil.

## Objetivo do Projeto

Desenvolver uma solução baseada em IA para automatizar e aprimorar o processo de recrutamento, propondo algoritmos e ferramentas que:

- Padronizem e otimizem entrevistas.
- Identifiquem padrões de candidatos(as) de sucesso.
- Realizem o “match” entre perfis e vagas de forma eficiente e baseada em dados.
- Disponibilizem o modelo de forma produtiva via API.

## Solução Proposta

A solução contempla:

- **Pipeline completo de Machine Learning**: feature engineering, pré-processamento, treinamento, validação e salvamento do modelo.
- **API para deployment**: endpoint `/predict` para receber dados e retornar previsões.
- **Empacotamento com Docker**: garantindo portabilidade e reprodutibilidade.
- **Deploy local ou em nuvem**: execução do modelo em ambiente isolado.
- **Testes unitários**: para garantir a qualidade e robustez do código.
- **Monitoramento contínuo**: logs e painel para acompanhamento de drift do modelo.

## Exemplos de Casos de Uso

- Agente de IA para entrevistas automatizadas, utilizando dados históricos para simular o papel do entrevistador.
- Otimização do processo de entrevistas, aprendendo padrões de sucesso em candidatos(as) anteriores.
- Identificação de atributos-chave em candidatos(as) de sucesso via algoritmos de clusterização.

## Requisitos Técnicos

- **Pipeline de treinamento**: feature engineering, pré-processamento, treinamento, validação e salvamento do modelo (pickle/joblib).
- **API**: Flask ou FastAPI, com endpoint `/predict`.
- **Empacotamento**: Dockerfile para API e dependências.
- **Deploy**: local ou em nuvem (AWS, Google Cloud Run, Heroku, etc).
- **Testes unitários**: para todos os componentes da pipeline.
- **Monitoramento**: logs e painel de acompanhamento.

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repo.git
   cd seu-repo
   ```
2. (Opcional) Crie um ambiente virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r data_pipeline/requirements.txt
   ```
4. Execute o pipeline de treinamento:
   ```bash
   python data_pipeline/main_feature_engineering.py
   python data_pipeline/main_curated.py
   ```
5. Inicie a API (exemplo com FastAPI ou Flask):
   ```bash
   # Exemplo com FastAPI
   uvicorn serving.api:app --reload
   # Exemplo com Flask
   python serving/api.py
   ```
6. Teste o endpoint `/predict` usando Postman ou cURL.

7. (Opcional) Empacote e rode com Docker:
   ```bash
   docker build -t decision-api .
   docker run -p 8000:8000 decision-api
   ```

## Estrutura do Repositório

- `data/` - Base de dados bruta e processada.
- `data_pipeline/` - Scripts de pipeline, engenharia de features, validação e requirements.
- `model_registry/` - Modelos treinados.
- `monitoring/` - Logs e monitoramento de qualidade.
- `notebooks/` - Notebooks de EDA, engenharia de features, treinamento e avaliação.
- `output/` - Resultados dos processamentos.
- `serving/` - Scripts para servir modelos ou APIs.

## Tecnologias Utilizadas

- Python 3.x
- Pandas, NumPy, Scikit-learn
- Jupyter Notebook
- Flask ou FastAPI
- Docker
- Logging, PyYAML, json, parquet

## Entregáveis

1. Código-fonte organizado e documentado neste repositório.
2. Link para a API de predição.
3. Vídeo de até 5 minutos explicando a estratégia de modelo e deploy ([link no topo](https://youtu.be/v03U9tBDizg)).

## Monitoramento e Testes

- Logs de execução e qualidade disponíveis em `monitoring/`.
- Testes unitários implementados para os principais componentes.
- Painel de acompanhamento de drift do modelo (em desenvolvimento).

## Contato

Para dúvidas ou sugestões, entre em contato com o responsável pelo projeto.
