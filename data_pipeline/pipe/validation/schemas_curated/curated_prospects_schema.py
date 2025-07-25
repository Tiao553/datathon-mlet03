from pydantic import BaseModel
from typing import List, Optional


class CuratedProspectRecord(BaseModel):
    codigo_vaga: str
    titulo: Optional[str]
    modalidade: Optional[str]
    # prospects
    p_nome: Optional[str]
    p_codigo: Optional[str]
    p_situacao_candidado: Optional[str]
    p_data_candidatura: Optional[str]
    p_ultima_atualizacao: Optional[str]
    p_comentario: Optional[str]
    p_recrutador: Optional[str]
