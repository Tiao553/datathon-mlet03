import polars as pl
from pipe.utils.logger import get_logger
from pipe.features.free_text_transform import processar_dataframe
from pipe.features.cleanning_and_accurate import gerar_features

logger = get_logger("main_feature_engineering")

if __name__ == "__main__":
    logger.info("Início do pipeline de engenharia de atributos")

    logger.info("→ Lendo arquivos Parquet da camada curated")
    df_job = pl.read_parquet("data/curated/jobs.parquet")
    df_applicants = pl.read_parquet("data/curated/applicants.parquet")
    df_prospects = pl.read_parquet("data/curated/prospects.parquet")

    logger.info(
        f"→ Linhas carregadas: jobs={len(df_job)}, applicants={len(df_applicants)}, prospects={len(df_prospects)}")

    logger.info("→ Renomeando colunas para evitar conflitos de namespace")
    df_applicants = df_applicants.rename(
        {col: f"app_{col}" for col in df_applicants.columns if col != "codigo_candidato"})
    df_job = df_job.rename(
        {col: f"job_{col}" for col in df_job.columns if col != "codigo_vaga"})

    logger.info("→ Realizando joins entre prospects, applicants e jobs")
    df = (
        df_prospects
        .rename({"p_codigo": "codigo_candidato"})
        .join(df_applicants, on="codigo_candidato", how="inner")
        .join(df_job, on="codigo_vaga", how="inner")
    )
    logger.info(f"→ Total de registros após joins: {len(df)}")
    logger.debug(f"Colunas finais do DataFrame: {df.columns}")

    logger.info("→ Iniciando etapa de geração de features estruturadas")
    df = gerar_features(df)

    logger.info(
        "→ Iniciando etapa de processamento de texto livre (ex: embeddings)")
    processar_dataframe(df[:1000])  # Teste com 1000 registros
    logger.info(f"→ Processamento finalizado com")  # shape: {df_final.shape}")
