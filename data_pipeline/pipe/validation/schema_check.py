from typing import Type, Optional
from pydantic import BaseModel, ValidationError
import polars as pl
from pipe.utils.logger import get_logger

logger = get_logger("schema_check")


def validate_schema(df: pl.DataFrame, model: Type[BaseModel]) -> list[str]:
    """
    Valida linha a linha do DataFrame com o schema Pydantic.
    Retorna lista de mensagens de erro (não interrompe na 1ª falha).
    """
    errors = []
    for i, row in enumerate(df.to_dicts()):
        try:
            model(**row)
        except ValidationError as e:
            errors.append(f"Linha {i}: {e}")
    return errors


def assert_valid_schema(df: pl.DataFrame, model: Type[BaseModel], label: str = "", id_field: Optional[str] = None):
    """
    Valida e emite log com todos os erros encontrados.
    Se `id_field` for passado, inclui identificador nas mensagens de erro.
    """
    logger.info(f"[{label}] Validando schema com {len(df)} registros")
    raw_errors = []
    for i, row in enumerate(df.to_dicts()):
        try:
            model(**row)
        except ValidationError as e:
            id_val = row.get(
                id_field, f"linha {i}") if id_field else f"linha {i}"
            error_msg = f"[{label}] Registro inválido (id={id_val}): {e}"
            raw_errors.append(error_msg)

    if raw_errors:
        for err in raw_errors:
            logger.error(err)
        logger.error(
            f"[{label}] Total de erros encontrados: {len(raw_errors)}")
        raise ValueError(
            f"[{label}] Falha na validação de schema: veja logs para detalhes.")

    logger.info(f"[{label}] Schema válido ✅")
