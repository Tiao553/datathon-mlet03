import os
from typing import List
import polars as pl
from pydantic import BaseModel
import json
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm
from pipe.features.prompts import prompt_vaga, prompt_candidato, chamar_llm, chamar_deepseek
from pipe.utils.logger import get_logger
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
