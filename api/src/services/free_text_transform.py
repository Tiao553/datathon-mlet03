import os
from typing import List
import polars as pl
from pydantic import BaseModel
import json
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm
from api.src.services.prompts import prompt_vaga, prompt_candidato, chamar_llm, chamar_deepseek
from api.src.utils.logger import get_logger
import time

logger = get_logger("feature_engineering_process")

###########################################################
# schemas
###########################################################


class VagaEstruturada(BaseModel):
    ferramentas_tecnologicas: List[str]
    competencias_comportamentais: List[str]
    competencias_tecnicas: List[str]
    experiencia_anos: str
    senioridade_aparente: str
    formacao_academica: str
    nivel_formacao: str
    area_formacao: str


class CandidatoEstruturado(BaseModel):
    principais_ferramentas_tecnologicas: List[str]
    ferramentas_tecnologicas: List[str]
    competencias_tecnicas: List[str]
    competencias_comportamentais: List[str]
    experiencia_anos: str
    senioridade_aparente: str
    formacao_academica: str
    nivel_formacao: str
    area_formacao: str

###########################################################
    # utils
###########################################################


def extrair_json_limpo(resposta_modelo: str) -> dict:
    linhas = resposta_modelo.strip().split('\n')
    if linhas and linhas[0].strip() == '```json':
        if len(linhas) > 1 and linhas[-1].strip() == '```':
            json_string = '\n'.join(linhas[1:-1])
        else:
            json_string = resposta_modelo.strip()
    else:
        json_string = resposta_modelo.strip()
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON: {e}")
        return {}


def salvar_jsonl(lista: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for linha in lista:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")


def carregar_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def salvar_jsonl_append(lista: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for linha in lista:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")


def append_parquet_safely(novo_df: pl.DataFrame, path: Path):
    if path.exists():
        df_existente = pl.read_parquet(path)
        df_final = pl.concat([df_existente, novo_df], how="vertical_relaxed")
    else:
        df_final = novo_df
    df_final.write_parquet(path)


###########################################################
    # process
###########################################################


def chamar_llm_com_retry(
    gerar_prompt_fn,
    row: dict,
    schema_cls: BaseModel,
    tipo: str,
    cod: str,
    respostas_brutas: list,
    max_retries: int = 3,
    delay: float = 0.2
) -> tuple[str, dict]:
    """
    Tenta gerar resposta válida via LLM até max_retries vezes.
    Em caso de falha, utiliza fallback com chamar_deepseek().

    Retorna: (resposta bruta, dados validados como dict)
    """
    for tentativa in range(1, max_retries + 1):
        try:
            prompt = gerar_prompt_fn(row)
            resposta = chamar_llm(prompt)
            dados_dict = extrair_json_limpo(resposta)
            respostas_brutas.append({
                "modelo": "llm",
                "tipo": tipo,
                "codigo": cod,
                "resposta": resposta
            })
            dados_validos = schema_cls.model_validate_json(
                json.dumps(dados_dict))
            logger.info(
                f"[OK {tipo.upper()}] Validação bem-sucedida na tentativa {tentativa}")
            return resposta, dados_validos.model_dump()
        except Exception as e:
            logger.warning(
                f"[RETRY {tipo.upper()}] Tentativa {tentativa} falhou: {e}")
            time.sleep(delay)

    # Fallback com modelo alternativo (deepseek)
    logger.warning(
        f"[LLM FALLBACK] Iniciando fallback com DeepSeek após {max_retries} falhas com modelo primário.")

    try:
        prompt = gerar_prompt_fn(row)
        resposta = chamar_deepseek(prompt)
        dados_dict = extrair_json_limpo(resposta)
        respostas_brutas.append({
            "modelo": "deepseek",
            "tipo": tipo,
            "codigo": cod,
            "resposta": resposta
        })
        dados_validos = schema_cls.model_validate_json(json.dumps(dados_dict))
        logger.info(f"[OK {tipo.upper()} - DEEPSEEK] Validação bem-sucedida")
        return resposta, dados_validos.model_dump()
    except Exception as e:
        logger.error(f"[FALHA {tipo.upper()} - DEEPSEEK] Erro: {e}")
        raise RuntimeError(
            f"[FALHA {tipo.upper()}] Não foi possível validar com nenhum modelo após {max_retries} tentativas.")


def processar_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Iniciando processamento de DataFrame com LLM")

    DIR_OUT = Path("data/feature_store")
    ARQ_CANDIDATOS = DIR_OUT / "candidatos_processados.jsonl"
    ARQ_VAGAS = DIR_OUT / "vagas_processadas.jsonl"
    ARQ_RAW_RESPONSES = DIR_OUT / \
        f"respostas_brutas_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
    ARQ_FINAL_PARQUET = DIR_OUT / "resultado_final.parquet"

    logger.info("→ Carregando histórico de candidatos e vagas processados")
    candidatos_ja_processados = {
        x["codigo_candidato"]: x["dados"] for x in carregar_jsonl(ARQ_CANDIDATOS)}
    vagas_ja_processadas = {x["codigo_vaga"]: x["dados"]
                            for x in carregar_jsonl(ARQ_VAGAS)}
    respostas_brutas = []

    registros = df.to_dicts()
    logger.info(f"→ Total de registros a processar: {len(registros)}")

    try:
        for row in tqdm(registros, desc="Processando registros"):
            cod_vaga = row.get("codigo_vaga")
            cod_candidato = row.get("codigo_candidato")

            if cod_vaga == cod_candidato:
                cod_candidato = f"{cod_candidato}_cand"

            # VAGA
            if cod_vaga not in vagas_ja_processadas:
                try:
                    resposta, dados = chamar_llm_com_retry(
                        prompt_vaga, row, VagaEstruturada, tipo="vaga", cod=cod_vaga, respostas_brutas=respostas_brutas)
                    vagas_ja_processadas[cod_vaga] = dados
                    salvar_jsonl_append(
                        [{"codigo_vaga": cod_vaga, "dados": dados}], ARQ_VAGAS)
                    logger.debug(f"[OK VAGA] {cod_vaga}")
                except Exception as e:
                    logger.error(f"[ERRO VAGA] {cod_vaga}: {e}")
                    continue

            # CANDIDATO
            if cod_candidato not in candidatos_ja_processados:
                try:
                    resposta, dados = chamar_llm_com_retry(
                        prompt_candidato, row, CandidatoEstruturado, tipo="candidato", cod=cod_candidato, respostas_brutas=respostas_brutas)
                    candidatos_ja_processados[cod_candidato] = dados
                    salvar_jsonl_append(
                        [{"codigo_candidato": cod_candidato, "dados": dados}], ARQ_CANDIDATOS)
                    salvar_jsonl_append(
                        [{"tipo": "candidato", "codigo": cod_candidato, "resposta": resposta}], ARQ_RAW_RESPONSES)
                    logger.debug(f"[OK CANDIDATO] {cod_candidato}")
                except Exception as e:
                    logger.error(f"[ERRO CANDIDATO] {cod_candidato}: {e}")
                    continue

            # Atualiza linha com dados prefixados
            dados_vaga = {f"job_{k}": v for k,
                          v in vagas_ja_processadas.get(cod_vaga, {}).items()}
            dados_candidato = {f"app_{k}": v for k, v in candidatos_ja_processados.get(
                cod_candidato, {}).items()}
            row.update(dados_vaga)
            row.update(dados_candidato)

            # Salva linha no Parquet incrementalmente
            append_parquet_safely(pl.DataFrame([row]), ARQ_FINAL_PARQUET)

    except Exception as e:
        logger.error(f"⛔ ERRO DETECTADO: {e}")

    logger.info("✅ Processamento concluído")
    logger.info(f"→ Dados salvos incrementalmente em: {ARQ_FINAL_PARQUET}")


if __name__ == "__main__":
    
    json_exemplo = """{
  "candidato": {
    "nome_completo": "Juliana Vasconcelos",
    "data_nascimento": "1994-08-15",
    "contato": {
      "email": "juliana.vasconcelos.dev@email.com",
      "telefone": "+55 (11) 98765-4321",
      "linkedin": "https://linkedin.com/in/julianavasconcelos"
    },
    "resumo_profissional": "Desenvolvedora de Software com mais de 6 anos de experiência, especializada em desenvolvimento backend com Python e seus frameworks, como Django e FastAPI. Tenho um perfil proativo, focado em resolver problemas complexos e construir soluções escaláveis. Busco uma oportunidade para aplicar minhas habilidades em um ambiente dinâmico e colaborativo, contribuindo para projetos inovadores.",
    "experiencia_profissional": [
      {
        "cargo": "Desenvolvedora Backend Pleno",
        "empresa": "Soluções Digitais Tech Inova",
        "periodo": "Janeiro 2022 - Presente",
        "descricao_atividades": "Responsável pelo desenvolvimento e manutenção de APIs RESTful para um sistema de gestão de logística. Utilizei FastAPI para garantir alta performance e Python para criar scripts de automação que reduziram o tempo de processamento de relatórios em 30%. Colaborei em um time ágil, participando de todo o ciclo de vida do produto, desde o planejamento até o deploy em ambiente de nuvem (AWS)."
      },
      {
        "cargo": "Desenvolvedora de Software Júnior",
        "empresa": "Web Creations Ltda.",
        "periodo": "Junho 2019 - Dezembro 2021",
        "descricao_atividades": "Atuei no desenvolvimento de um e-commerce utilizando Django e Python. Fui responsável pela criação de novos módulos, correção de bugs e integração com gateways de pagamento. Aprendi a trabalhar com bancos de dados PostgreSQL e a utilizar Docker para a criação de ambientes de desenvolvimento padronizados."
      }
    ],
    "educacao": [
      {
        "instituicao": "Universidade Federal de São Carlos (UFSCar)",
        "curso": "Bacharelado em Ciência da Computação",
        "periodo": "2014 - 2018"
      }
    ],
    "habilidades_tecnicas": [
      "Python",
      "FastAPI",
      "Django",
      "SQL (PostgreSQL, MySQL)",
      "Docker",
      "Git",
      "AWS (EC2, S3)",
      "APIs RESTful",
      "Metodologias Ágeis"
    ]
  },
  "vaga": {
    "titulo_vaga": "Desenvolvedor(a) de IA - Processamento de Linguagem Natural",
    "empresa_nome": "DataMind Analytics",
    "publicada_em": "2025-07-25",
    "localizacao": "São Paulo, SP",
    "modalidade": "Híbrido (3 dias no escritório)",
    "descricao_empresa": "A DataMind Analytics é uma startup líder em soluções de inteligência artificial para o mercado financeiro. Nosso objetivo é transformar grandes volumes de dados em insights acionáveis para nossos clientes. Valorizamos um ambiente de trabalho inovador, com autonomia e muitos desafios.",
    "resumo_da_vaga": "Estamos em busca de um(a) Desenvolvedor(a) de IA talentoso(a) e apaixonado(a) por Processamento de Linguagem Natural (PLN) para se juntar à nossa equipe. Você trabalhará no desenvolvimento de modelos capazes de extrair informações, classificar textos e analisar sentimentos de grandes volumes de documentos não estruturados, como relatórios e notícias.",
    "responsabilidades_atribuicoes": [
      "Pesquisar, projetar e implementar modelos de Machine Learning e Deep Learning com foco em PLN.",
      "Realizar o pré-processamento e a limpeza de dados textuais para treinamento de modelos.",
      "Desenvolver e otimizar pipelines de dados para alimentar os sistemas de IA.",
      "Criar APIs para disponibilizar os modelos como serviço para outras equipes e produtos.",
      "Manter-se atualizado sobre as últimas tendências e técnicas em IA e PLN.",
      "Colaborar com engenheiros de dados, cientistas de dados e especialistas de negócio."
    ],
    "requisitos_qualificacoes": {
      "obrigatorios": [
        "Sólida experiência em Python e bibliotecas de IA (ex: scikit-learn, spaCy, NLTK).",
        "Experiência prática com frameworks de Deep Learning como PyTorch ou TensorFlow.",
        "Conhecimento profundo de arquiteturas de modelos de PLN, como Transformers (BERT, GPT, etc.).",
        "Experiência com o ciclo de vida de projetos de Machine Learning, do desenvolvimento ao deploy (MLOps).",
        "Familiaridade com ambientes de nuvem (AWS, GCP ou Azure)."
      ],
      "diferenciais": [
        "Mestrado ou Doutorado em Ciência da Computação, Inteligência Artificial ou área relacionada.",
        "Experiência com LLMs (Large Language Models) e técnicas de fine-tuning.",
        "Publicações em conferências de IA/PLN.",
        "Conhecimento em Docker e Kubernetes."
      ]
    },
    "beneficios": [
      "Salário competitivo",
      "Plano de Saúde e Odontológico",
      "Vale Refeição e Vale Alimentação",
      "Horário Flexível",
      "Auxílio Home Office",
      "Plano de desenvolvimento de carreira"
    ]
  }
}"""
    import io
    dados_dict = json.loads(json_exemplo)

    json_bytes = json.dumps(dados_dict).encode('utf-8')

    json_buffer = io.BytesIO(json_bytes)

    df_exemplo = pl.read_json(json_buffer)
    processar_dataframe(df_exemplo)