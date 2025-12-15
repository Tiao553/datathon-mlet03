# Resumo de Experimentos e Análises (EDA & Feature Engineering)

Este documento resume as análises e testes encontrados nos notebooks `docs/archive_notebooks/EDA.ipynb` e `docs/archive_notebooks/jobs_feature_enginerring.ipynb`.

## 1. Análise Exploratória de Dados (`EDA.ipynb`)

O foco principal deste notebook foi entender a estrutura dos dados brutos (JSON) e avaliar a qualidade/completude antes da modelagem.

### Principais Testes e Análises:
- **Análise Estrutural (Schemas)**:
    - Mapeamento da estrutura aninhada dos arquivos `Jobs.json`, `Prospects.json` e `Applicants.json`.
    - Conversão de tipos complexos (`Struct`, `List`) para tabelas "achatadas" (*flat*) usando Polars.
- **Análise Qualitativa**:
    - **Nulos**: Cálculo de contagem e porcentagem de valores nulos por coluna. Identificou-se alta taxa de nulos em campos como `pv_habilidades_comportamentais_necessarias` (~80%).
    - **Cardinalidade**: Contagem de valores únicos para distinguir entre campos categóricos (ex: `ib_tipo_contratacao`) e texto livre.
    - **Análise Textual**: Estatísticas de comprimento (min, max, média) para colunas de string, ajudando a identificar descrições longas vs. códigos curtos.
- **Identificação de PII**: Mapeamento inicial de campos sensíveis para remoção posterior.

## 2. Engenharia de Features (`jobs_feature_enginerring.ipynb`)

Este notebook foca na transformação dos dados para consumo por modelos, com ênfase em limpeza e criação de novas variáveis.

### Principais Testes e Implementações:
- **Achatamento de Dados (Flattening)**:
    - Uso intensivo de `polars.DataFrame.unnest` e `explode` para transformar estruturas aninhadas em colunas diretas (ex: `job_ib_titulo_vaga`, `app_ib_objetivo_profissional`).
- **Remoção de PII (Privacidade)**:
    - Remoção de colunas sensíveis como telefones, emails, documentos e nomes completos.
    - Criação de **flags binárias** (0/1) indicando a presença desses dados antes da remoção (ex: `ind_app_linkedin`, `ind_app_telefone`), preservando a informação de "completo" sem expor o dado.
- **Match de Localidade (Fuzzy Matching)**:
    - Implementação de comparação aproximada de texto (`rapidfuzz`) entre o endereço do candidato e o local da vaga.
    - Criação da feature `ind_mesma_localidade` baseada em um *threshold* de similaridade (ex: 70%).
- **Tratamento de Datas**:
    - Parsing de strings para objetos de data/hora (ex: `app_ip_data_aceite`).
- **Criação de Flags de Texto**:
    - Identificação de padrões simples em textos (Regex) para categorizar ou filtrar registros.

## Conclusão para Próximos Passos
Os experimentos validaram o uso de **Polars** para processamento eficiente e destacaram a necessidade de:
1.  **Imputar ou tratar nulos** em campos críticos de texto.
2.  **Utilizar flags de preenchimento** como proxy de "engajamento" ou "qualidade do perfil".
3.  **Features de Similaridade**: O uso de fuzzy match para locais sugere que embeddings ou distâncias textuais serão cruciais para comparar *skills* e descrições de vagas.
