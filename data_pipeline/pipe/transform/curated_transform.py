import polars as pl
import re
import unicodedata


def get_initials(name: str) -> str:
    s1 = re.sub(r'([A-Z])', r' \1', name).strip()
    s2 = re.sub(r'[-_]', ' ', s1)
    return "".join(word[0] for word in s2.split()).lower()


def clean_column_name(name: str) -> str:
    """
    Remove acentos, substitui espaços e caracteres especiais por "_",
    transforma em caixa baixa e remove duplicações de "_".
    """
    name = unicodedata.normalize('NFKD', name).encode(
        'ASCII', 'ignore').decode()
    name = re.sub(r"[^\w\s]", "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"__+", "_", name)
    return name.lower().strip("_")


def flatten_struct_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Desestrutura colunas do tipo Struct ou List[Struct], aplicando prefixo baseado em iniciais do nome.
    Higieniza todos os nomes das colunas ao final.
    """
    for col_name in df.columns:
        dtype = df.schema[col_name]

        if isinstance(dtype, pl.Struct):
            prefix = get_initials(col_name)
            field_names = [f.name for f in dtype.fields]
            new_names = [f"{prefix}_{field}" for field in field_names]
            df = df.with_columns(pl.col(col_name).struct.rename_fields(
                new_names)).unnest(col_name)

        elif isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct):
            prefix = get_initials(col_name)
            df = df.explode(col_name)
            field_names = [f.name for f in dtype.inner.fields]
            new_names = [f"{prefix}_{field}" for field in field_names]
            df = df.with_columns(pl.col(col_name).struct.rename_fields(
                new_names)).unnest(col_name)

    # Higieniza os nomes de colunas
    sanitized = {col: clean_column_name(col) for col in df.columns}
    return df.rename(sanitized)


def clean_string_column(col: pl.Series) -> pl.Series:
    return (
        col.cast(str)
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"[^\w\s@.,\-:/]", "")
        .str.strip_chars()
        .str.to_uppercase()
    )


def parse_date_columns(df: pl.DataFrame, date_columns: list[str]) -> pl.DataFrame:
    for col in date_columns:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).str.to_date("%d-%m-%Y", strict=False).alias(col)
            )
    return df


def parse_datetime_columns(df: pl.DataFrame, date_columns: list[str]) -> pl.DataFrame:
    for col in date_columns:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).str.strptime(pl.Datetime,
                                         "%d-%m-%Y %H:%M:%S", strict=False).alias(col)
            )
    return df


def normalize_dataframe(df: pl.DataFrame, date_columns: list[str] = [], datetime_columns: list[str] = []) -> pl.DataFrame:
    # Limpar colunas string
    string_cols = [col for col, dtype in df.schema.items()
                   if dtype == pl.String]
    df = df.with_columns([clean_string_column(
        pl.col(col)).alias(col) for col in string_cols])

    # Converter datas
    df = parse_date_columns(df, date_columns)

    # Converter datetimes
    df = parse_datetime_columns(df, datetime_columns)

    return df
