import polars as pl
from typing import List, Optional, Tuple
import re


# ------------------------------------------------------------------
# 1. Campos obrigatórios não nulos
# ------------------------------------------------------------------
def check_required_columns(df: pl.DataFrame, columns: List[str]) -> List[str]:
    """
    Verifica valores obrigatórios, considerando nulos, vazios, espaços ou "-".
    """
    issues = []
    for col in columns:
        if col not in df.columns:
            issues.append(f"Coluna obrigatória ausente: {col}")
            continue

        series = df[col]
        if series.dtype == pl.String:
            invalid = series.is_null() | series.str.strip_chars().is_in(
                ["", "-", " "])
        else:
            invalid = series.is_null()

        n_invalid = invalid.sum()
        if n_invalid > 0:
            issues.append(
                f"Coluna {col} contém {n_invalid} valores inválidos (nulo, vazio, '-', espaço)")
    return issues


def get_invalid_rows_for_required_columns(df: pl.DataFrame, required_cols: List[str]) -> List[int]:
    """
    Retorna os índices das linhas que possuem campos obrigatórios nulos,
    vazios, apenas espaços ou contendo apenas hífens ("-").
    """
    invalid_mask = None

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")

        col_clean = (
            pl.col(col)
            .cast(pl.String)
            .str.strip_chars()
            .str.strip_chars("-")
            .str.strip_chars()
        )

        mask = col_clean.is_null() | (col_clean == "") | (col_clean.str.len_chars() == 0)
        invalid_mask = mask if invalid_mask is None else invalid_mask | mask

    # Cria coluna auxiliar de índices
    df_with_idx = df.with_columns(pl.arange(0, df.height).alias("_row_idx"))
    invalid_rows = df_with_idx.filter(invalid_mask)
    return invalid_rows["_row_idx"].to_list()


# ------------------------------------------------------------------
# 2. Duplicações
# ------------------------------------------------------------------
def check_duplicates(df: pl.DataFrame, subset: List[str]) -> List[str]:
    dup_count = df.select(pl.col(subset)).is_duplicated().sum()
    if dup_count > 0:
        return [f"[DUPLICATE] {dup_count} registros duplicados encontrados com base nas colunas {subset}."]
    return []


# ------------------------------------------------------------------
# 3. Valores permitidos (domínio fechado)
# ------------------------------------------------------------------
def check_value_domain(df: pl.DataFrame, column: str, allowed_values: List[str]) -> List[str]:
    if column not in df.columns:
        return [f"[DOMAIN] Coluna '{column}' ausente."]

    unique_vals = df[column].unique().to_list()
    invalid = [
        val for val in unique_vals if val not in allowed_values and val is not None]
    if invalid:
        return [f"[DOMAIN] Valores inválidos na coluna '{column}': {invalid}"]
    return []


# ------------------------------------------------------------------
# 4. Verificação por expressão regular
# ------------------------------------------------------------------
def check_regex_format(df: pl.DataFrame, column: str, pattern: str) -> List[str]:
    if column not in df.columns:
        return [f"[REGEX] Coluna '{column}' ausente."]

    regex = re.compile(pattern)
    invalid_count = df.filter(~pl.col(column).cast(
        str).apply(lambda x: bool(regex.match(x)))).height
    if invalid_count > 0:
        return [f"[REGEX] {invalid_count} valores em '{column}' não seguem o padrão '{pattern}'."]
    return []


# ------------------------------------------------------------------
# 5. Ranges numéricos
# ------------------------------------------------------------------
def check_numeric_range(df: pl.DataFrame, column: str, min_value: float, max_value: float) -> List[str]:
    if column not in df.columns:
        return [f"[RANGE] Coluna '{column}' ausente."]

    min_invalid = df.filter(pl.col(column) < min_value).height
    max_invalid = df.filter(pl.col(column) > max_value).height
    issues = []
    if min_invalid > 0:
        issues.append(
            f"[RANGE] {min_invalid} valores menores que {min_value} em '{column}'.")
    if max_invalid > 0:
        issues.append(
            f"[RANGE] {max_invalid} valores maiores que {max_value} em '{column}'.")
    return issues


# ------------------------------------------------------------------
# 6. Colunas com alta cardinalidade
# ------------------------------------------------------------------
def check_high_cardinality(df: pl.DataFrame, column: str, threshold: int = 1000) -> List[str]:
    if column not in df.columns:
        return [f"[CARDINALITY] Coluna '{column}' ausente."]

    unique_count = df[column].n_unique()
    if unique_count > threshold:
        return [f"[CARDINALITY] Coluna '{column}' possui {unique_count} valores distintos (>{threshold})."]
    return []
