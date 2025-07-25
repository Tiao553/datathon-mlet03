# Data View

## 📂 Visão Geral

Este projeto tem como objetivo construir um pipeline de dados que integre e trate os dados oriundos de três fontes principais em formato **JSON**: `Jobs.json`, `Prospects.json` e `Applicants.json`.

Os dados **raw** estão organizados na camada **`/raw/`**. A partir desses dados, serão geradas duas visões principais:

* **Camada Tratada (Curated)**: com limpeza, normalização e estruturação das informações
* **Feature Store**: com variáveis analíticas derivadas e prontas para consumo por modelos de machine learning ou dashboards analíticos

⚠️ **Observação sobre a privacidade dos dados**:
Todos os dados sensíveis (clientes, candidatos e analistas) foram anonimizados utilizando nomes, números de telefone e e-mails aleatórios.

---

## 🧱 Estrutura dos Arquivos

### `Jobs.json` – Informações sobre a vaga

* Chave primária: `codigo_vaga`
* Agrupado em três blocos principais:

  * `informacoes_basicas`: dados administrativos
  * `perfil_vaga`: requisitos técnicos e comportamentais
  * `beneficios`: aspectos comerciais

#### Schema

```json
{
  "codigo_vaga": "String",
  "informacoes_basicas": {
    "data_requicisao": "String",
    "limite_esperado_para_contratacao": "String",
    "titulo_vaga": "String",
    "vaga_sap": "String",
    "cliente": "String",
    "solicitante_cliente": "String",
    "empresa_divisao": "String",
    "requisitante": "String",
    "analista_responsavel": "String",
    "tipo_contratacao": "String",
    "prazo_contratacao": "String",
    "objetivo_vaga": "String",
    "prioridade_vaga": "String",
    "origem_vaga": "String",
    "superior_imediato": "String",
    "nome": "String",
    "telefone": "String",
    "data_inicial": "String",
    "data_final": "String"
  },
  "perfil_vaga": {
    "pais": "String",
    "estado": "String",
    "cidade": "String",
    "bairro": "String",
    "regiao": "String",
    "local_trabalho": "String",
    "vaga_especifica_para_pcd": "String",
    "faixa_etaria": "String",
    "horario_trabalho": "String",
    "nivel profissional": "String",
    "nivel_academico": "String",
    "nivel_ingles": "String",
    "nivel_espanhol": "String",
    "outro_idioma": "String",
    "areas_atuacao": "String",
    "principais_atividades": "String",
    "competencia_tecnicas_e_comportamentais": "String",
    "demais_observacoes": "String",
    "viagens_requeridas": "String",
    "equipamentos_necessarios": "String",
    "habilidades_comportamentais_necessarias": "String"
  },
  "beneficios": {
    "valor_venda": "String",
    "valor_compra_1": "String",
    "valor_compra_2": "String"
  }
}
```

### `Prospects.json` – Lista de candidatos por vaga

* Chave: `codigo_vaga`
* Contém lista de prospecções para cada vaga
* Cada prospecção possui informações individuais do candidato na vaga

#### Schema

```json
{
  "codigo_vaga": "String",
  "titulo": "String",
  "modalidade": "String",
  "prospects": {
    "list_of": {
      "nome": "String",
      "codigo": "String",
      "situacao_candidado": "String",
      "data_candidatura": "String",
      "ultima_atualizacao": "String",
      "comentario": "String",
      "recrutador": "String"
    }
  }
}
```

### `Applicants.json` – Informações completas dos candidatos

* Chave: `codigo_candidato`
* Detalhamento completo de cada candidato, dividido em blocos temáticos:

  * Informações básicas
  * Informações pessoais
  * Informações profissionais
  * Formação e idiomas
  * CVs (pt/en)

#### Schema

```json
{
  "codigo_candidato": "String",
  "infos_basicas": {
    "telefone_recado": "String",
    "telefone": "String",
    "objetivo_profissional": "String",
    "data_criacao": "String",
    "inserido_por": "String",
    "email": "String",
    "local": "String",
    "sabendo_de_nos_por": "String",
    "data_atualizacao": "String",
    "codigo_profissional": "String",
    "nome": "String"
  },
  "informacoes_pessoais": {
    "data_aceite": "String",
    "download_cv": "String",
    "nome": "String",
    "cpf": "String",
    "fonte_indicacao": "String",
    "email": "String",
    "email_secundario": "String",
    "data_nascimento": "String",
    "telefone_celular": "String",
    "telefone_recado": "String",
    "sexo": "String",
    "estado_civil": "String",
    "pcd": "String",
    "endereco": "String",
    "skype": "String",
    "url_linkedin": "String",
    "facebook": "String"
  },
  "informacoes_profissionais": {
    "titulo_profissional": "String",
    "area_atuacao": "String",
    "conhecimentos_tecnicos": "String",
    "certificacoes": "String",
    "outras_certificacoes": "String",
    "remuneracao": "String",
    "nivel_profissional": "String"
  },
  "formacao_e_idiomas": {
    "nivel_academico": "String",
    "instituicao_ensino_superior": "String",
    "cursos": "String",
    "ano_conclusao": "String",
    "nivel_ingles": "String",
    "nivel_espanhol": "String",
    "outro_idioma": "String"
  },
  "cargo_atual": {},
  "cv_pt": "String",
  "cv_en": "String"
}
```

---

## 🔗 Estrutura de Join dos Dados

Abaixo está o grafo textual representando o fluxo de integração entre as fontes de dados:

```
(Jobs.json) <--- codigo_vaga ---> (Prospects.json)
                                 |
                                 |-- codigo (código do candidato)
                                 v
                        (Applicants.json)
```

### Explicação:

1. **Jobs.json** é a base principal contendo os detalhes da vaga (`codigo_vaga`)
2. **Prospects.json** associa cada vaga a múltiplos candidatos prospectados (`codigo`)
3. **Applicants.json** traz os dados completos de cada candidato (`codigo_candidato`)

---

## 🧹 Visões Derivadas

### 1. Visão Tratada (`curated.jobs_applicants_view`)

Contém:

* Campos selecionados e renomeados
* Dados normalizados e tipados
* Redução de nulos e colunas descartadas com 100% ausentes
* Flags auxiliares para preenchimento de dados

### 2. Visão Feature Store (`features.jobs_applicants`)

Contém:

* Features categóricas e numéricas derivadas
* Colunas vetorizadas e clusterizadas (conhecimentos, certificações, etc.)
* Encoding de níveis de idioma, faixa salarial, título profissional
* Flags de canais de origem, presença de CV, etc.

---

## ✅ Exemplo de Caso Real

Para a vaga `10976` (em `Jobs.json`), temos:

* 25 prospecções (em `Prospects.json`) associadas a essa vaga
* O candidato `"Sr. Thales Freitas"` com código `41496` (em `Applicants.json`) foi o contratado

---
