from rapidfuzz.fuzz import partial_ratio
from pipe.utils.logger import get_logger
import polars as pl

logger = get_logger("feature_engineering")

# Função auxiliar para gerar indicador booleano


def gerar_indicador(nome: str, col_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(col_expr.is_not_null() & (col_expr != ""))
        .then(True)
        .otherwise(False)
        .alias(nome)
    )

# Executa fuzzy matching fora do Polars


def comparar_locais_com_fuzzy(locais_candidato: list[str], locais_vaga: list[str], threshold: int = 90) -> bool:
    for loc_candidato in locais_candidato:
        for loc_vaga in locais_vaga:
            if loc_candidato and loc_vaga:
                score = partial_ratio(loc_candidato.upper(), loc_vaga.upper())
                if score >= threshold:
                    return True
    return False


def remover_pii_e_engineering(df: pl.DataFrame) -> pl.DataFrame:
    pii_cols = [
        "p_nome",
        "app_ib_telefone_recado",
        "app_ib_telefone",
        "app_ib_email",
        "app_ip_nome",
        "app_ip_cpf",
        "app_ip_email",
        "app_ip_email_secundario",
        "app_ip_data_nascimento",
        "app_ip_telefone_celular",
        "app_ip_telefone_recado",
        "app_ip_endereco",
        "app_ip_skype",
        "app_ip_url_linkedin",
        "app_ip_facebook",
    ]

    # Adiciona indicadores binários
    df = df.with_columns([
        gerar_indicador("ind_app_telefone", pl.col("app_ib_telefone")),
        gerar_indicador("ind_app_email", pl.col("app_ib_email")),
        gerar_indicador("ind_app_linkedin", pl.col("app_ip_url_linkedin")),
        gerar_indicador("ind_app_endereco", pl.col("app_ip_endereco")),
        gerar_indicador("ind_app_facebook", pl.col("app_ip_facebook")),
    ])

    # Extrai registros para processamento externo
    registros = df.select([
        "app_ib_local", "app_ip_endereco",
        "job_pv_estado", "job_pv_cidade", "job_pv_bairro", "job_pv_regiao", "job_pv_local_trabalho"
    ]).to_dicts()

    # Gera coluna fuzzy de localidade
    matches = []
    for row in registros:
        locais_candidato = [row["app_ib_local"], row["app_ip_endereco"]]
        locais_vaga = [
            row["job_pv_estado"], row["job_pv_cidade"], row["job_pv_bairro"],
            row["job_pv_regiao"], row["job_pv_local_trabalho"]
        ]
        matches.append(comparar_locais_com_fuzzy(
            locais_candidato, locais_vaga, threshold=70))

    # Adiciona coluna ao DataFrame final
    df = df.with_columns([
        pl.Series("ind_mesma_localidade", matches)
    ])

    # Remove colunas PII
    df = df.drop([col for col in pii_cols if col in df.columns])

    return df


def classificar_prioridade_vaga(df: pl.DataFrame) -> pl.DataFrame:
    prioridade_col = (
        pl.col("job_ib_prioridade_vaga")
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )

    return df.with_columns(
        pl.when(
            prioridade_col.str.contains(r"(ALTA|HIGH|URGENTE|URGENCY)")
        ).then(pl.lit("ALTA"))
        .when(
            prioridade_col.str.contains(
                r"(MEDIA|MÉDIA|MEDIUM|INTERMEDIARIA|INTERMEDIÁRIA)")
        ).then(pl.lit("MEDIA"))
        .when(
            prioridade_col.str.contains(r"(BAIXA|LOW|BAIXA PRIORIDADE)")
        ).then(pl.lit("BAIXA"))
        .otherwise(pl.lit("DESCONHECIDO"))
        .alias("job_ib_prioridade_vaga")
    )


def extrair_lista(df: pl.DataFrame, coluna: str, spliter: str) -> pl.DataFrame:
    """
    Cria nova coluna com lista padronizada e limpa, removendo elementos vazios e substituindo por 'DESCONHECIDO' se necessário.
    """
    # Pré-processamento e transformação dos elementos
    col_limpa = (
        pl.col(coluna)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.split(spliter)
        .list.eval(
            pl.element()
            .str.strip_chars()
            .str.strip_chars(" ")
            .str.to_uppercase()
        )
    )

    # Lista com elementos vazios removidos
    col_filtrada = col_limpa.list.eval(pl.element().filter(pl.element() != ""))

    return df.with_columns(
        pl.when(
            (col_filtrada.list.len() == 0)
        ).then(
            pl.lit(["DESCONHECIDO"])
        ).otherwise(
            col_filtrada
        ).alias(f"{coluna}_cleaned_list")
    )


def gerar_features_temporais(df: pl.DataFrame) -> pl.DataFrame:
    max_data_candidatura = df.select(pl.col("p_data_candidatura").max()).item()
    hoje = pl.lit(max_data_candidatura)

    def dias(col1, col2, nome, abs_val=False):
        expr = (col1 - col2).dt.total_days()
        if abs_val:
            expr = expr.abs()
        return (
            pl.when(col1.is_not_null() & col2.is_not_null())
            .then(expr)
            .otherwise(-1)
            .alias(nome)
            .clip(0, None)  # ✅ CORRETO
        )

    return df.with_columns([
        dias(pl.col("app_ib_data_criacao"), pl.col("p_data_candidatura"),
             "tempo_entre_criacao_e_candidatura", abs_val=True),
        dias(pl.col("p_data_candidatura"), pl.col("job_ib_data_inicial"),
             "tempo_entre_job_inicial_e_candidatura"),
        dias(pl.col("job_ib_limite_esperado_para_contratacao"), pl.col(
            "job_ib_data_requicisao"), "tempo_para_contratacao"),
        dias(pl.col("app_ib_data_atualizacao"), pl.col(
            "app_ib_data_criacao"), "tempo_ultima_atualizacao_aplicacao"),
        dias(pl.col("job_ib_data_final"), pl.col(
            "job_ib_data_inicial"), "tempo_vaga_aberta"),
        (
            pl.when(pl.col("job_ib_data_final").is_not_null())
            .then((pl.col("job_ib_data_final") > hoje).cast(pl.Int8))
            .otherwise(-1)
            .alias("vaga_em_aberto_no_momento")
        ),
        (
            pl.when(pl.col("p_data_candidatura").is_not_null() &
                    pl.col("job_ib_data_inicial").is_not_null())
            .then((pl.col("p_data_candidatura") < pl.col("job_ib_data_inicial")).cast(pl.Int8))
            .otherwise(-1)
            .alias("candidatura_antes_da_abertura_oficial")
        ),
        (
            pl.when(pl.col("job_ib_data_requicisao").is_not_null())
            .then((hoje - pl.col("job_ib_data_requicisao")).dt.total_days())
            .otherwise(-1)
            .alias("dias_desde_requisicao")
            .clip(0, None)
        ),
        (
            pl.when(pl.col("job_ib_data_requicisao").is_not_null())
            .then(((hoje - pl.col("job_ib_data_requicisao")).dt.total_days() < 30).cast(pl.Int8))
            .otherwise(-1)
            .alias("vaga_requisitada_recente")
        )
    ])


def gerar_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Função principal para gerar features a partir do DataFrame de entrada.
    """
    logger.info("Início da geração de features")

    logger.info("→ Etapa: remoção de PII e engenharia inicial")
    df = remover_pii_e_engineering(df)

    logger.info("→ Etapa: classificação de prioridade de vaga")
    df = classificar_prioridade_vaga(df)

    list_to_extract = [
        ("job_ib_tipo_contratacao", ","),
        ("job_pv_areas_atuacao", "-"),
    ]

    logger.info("→ Etapa: extração de listas de colunas configuradas")
    for coluna, spliter in list_to_extract:
        logger.debug(f"Extraindo coluna: {coluna} com separador: '{spliter}'")
        df = extrair_lista(df, coluna, spliter)

    logger.info("→ Etapa: geração de features temporais")
    df = gerar_features_temporais(df)

    logger.info("→ Etapa: criação da coluna 'job_ind_beneficios_declarados'")
    df = df.with_columns([
        (
            (pl.col("job_b_valor_venda").is_not_null() & (pl.col("job_b_valor_venda") != "")) |
            (pl.col("job_b_valor_compra_1").is_not_null() & (pl.col("job_b_valor_compra_1") != "")) |
            (pl.col("job_b_valor_compra_2").is_not_null()
             & (pl.col("job_b_valor_compra_2") != ""))
        ).alias("job_ind_beneficios_declarados")
    ])

    logger.info("→ Etapa: criação da coluna 'match_pcd'")
    df = df.with_columns([
        pl.when(
            (pl.col("job_pv_vaga_especifica_para_pcd") == "SIM") &
            (pl.col("app_ip_pcd").is_not_null()) &
            (pl.col("app_ip_pcd") != "")
        )
        .then(True)
        .otherwise(False)
        .alias("match_pcd")
    ])

    logger.info("Finalização da geração de features")
    return df
