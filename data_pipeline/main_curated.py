import polars as pl
from pathlib import Path

from pipe.utils.logger import get_logger
from pipe.utils.audit import save_quality_issues
from pipe.ingest.read_raw import load_json_to_df, get_file_path
from pipe.transform.curated_transform import flatten_struct_columns, normalize_dataframe
from pipe.validation.schema_check import assert_valid_schema
from pipe.validation.quality_rules import (
    check_required_columns,
    check_duplicates,
    get_invalid_rows_for_required_columns,
)
from pipe.validation.schemas_curated.curated_jobs_schema import CuratedJobRecord
from pipe.validation.schemas_curated.curated_prospects_schema import CuratedProspectRecord
from pipe.validation.schemas_curated.curated_applicants_schema import CuratedApplicantRecord

logger = get_logger("curated_main")
OUTPUT_DIR = Path("data/curated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------- JOBS ------------------------- #
def process_jobs() -> None:
    logger.info("Processando: jobs.json")
    df = load_json_to_df(get_file_path("jobs.json", "jobs"), "codigo_vaga")
    df = flatten_struct_columns(df)

    assert_valid_schema(df, CuratedJobRecord, label="jobs",
                        id_field="codigo_vaga")

    issues = []
    issues += check_required_columns(df, ["codigo_vaga"])
    issues += check_duplicates(df, ["codigo_vaga"])

    if issues:
        save_quality_issues(issues, label="jobs")
        logger.error(
            f"[JOBS] {len(issues)} problemas de qualidade identificados:")
        for issue in issues:
            logger.error(f"[JOBS] {issue}")
        raise ValueError(
            "[JOBS] Falha na checagem de qualidade. Verifique os logs.")

    df = normalize_dataframe(df, date_columns=[
        "ib_data_requicisao", "ib_data_inicial", "ib_data_final", "ib_limite_esperado_para_contratacao"
    ])
    df.write_parquet(OUTPUT_DIR / "jobs.parquet")
    logger.info("jobs.parquet gerado com sucesso.")


# ----------------------- PROSPECTS ---------------------- #
def process_prospects() -> None:
    logger.info("Processando: prospects.json")
    df = load_json_to_df(get_file_path(
        "prospects.json", "prospects"), "codigo_vaga")
    df = flatten_struct_columns(df)

    assert_valid_schema(df, CuratedProspectRecord,
                        label="prospects", id_field="codigo_vaga")

    # ----------- Tratamento de registros inválidos por campos obrigatórios nulos ----------- #
    issues = []
    invalid_idx = get_invalid_rows_for_required_columns(
        df, ["codigo_vaga", "p_codigo"])
    if invalid_idx:
        issues.append(
            f"{len(invalid_idx)} registros removidos por campos nulos ou inválidos.")
        valid_idx = sorted(set(range(len(df))) - set(invalid_idx))
        df = df[valid_idx]
        logger.warning(f"[PROSPECTS] {issues[-1]}")
        # Log apenas os removidos
        save_quality_issues(issues, label="prospects")

    # ----------- Checagens permanentes com erro em caso de falha ----------- #
    schema_issues = []
    schema_issues += check_required_columns(df, ["codigo_vaga", "p_codigo"])
    schema_issues += check_duplicates(df, ["codigo_vaga", "p_codigo"])

    if schema_issues:
        save_quality_issues(schema_issues, label="prospects")
        logger.error(
            f"[PROSPECTS] {len(schema_issues)} problemas de qualidade identificados:")
        for issue in schema_issues:
            logger.error(f"[PROSPECTS] {issue}")
        raise ValueError(
            "[PROSPECTS] Falha na checagem de qualidade. Verifique os logs.")

    df = normalize_dataframe(df, date_columns=[
        "p_data_candidatura", "p_ultima_atualizacao"
    ])
    df.write_parquet(OUTPUT_DIR / "prospects.parquet")
    logger.info("prospects.parquet gerado com sucesso.")


# ---------------------- APPLICANTS ---------------------- #
def process_applicants() -> None:
    logger.info("Processando: applicants.json")
    df = load_json_to_df(get_file_path(
        "applicants.json", "applicants"), "codigo_candidato")
    df = flatten_struct_columns(df)

    assert_valid_schema(df, CuratedApplicantRecord,
                        label="applicants", id_field="codigo_candidato")

    issues = []
    issues += check_required_columns(df, ["codigo_candidato", "ib_nome"])
    issues += check_duplicates(df, ["codigo_candidato"])

    if issues:
        save_quality_issues(issues, label="applicants")
        logger.error(
            f"[APPLICANTS] {len(issues)} problemas de qualidade identificados:")
        for issue in issues:
            logger.error(f"[APPLICANTS] {issue}")
        raise ValueError(
            "[APPLICANTS] Falha na checagem de qualidade. Verifique os logs.")

    df = normalize_dataframe(df,
                             datetime_columns=[
                                 "ib_data_atualizacao", "ib_data_criacao"],
                             date_columns=["ip_data_nascimento"],
                             )
    df.write_parquet(OUTPUT_DIR / "applicants.parquet")
    logger.info("applicants.parquet gerado com sucesso.")


# ------------------------- EXECUÇÃO ------------------------- #
if __name__ == "__main__":
    logger.info("Iniciando pipeline de transformação raw → curated")
    process_jobs()
    process_prospects()
    process_applicants()
    logger.info(
        "Pipeline finalizada com sucesso. Dados disponíveis em data/curated")
