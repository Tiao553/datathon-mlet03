from pydantic import BaseModel
from typing import Optional, Dict


class CuratedApplicantRecord(BaseModel):
    codigo_candidato: str

    # infos_basicas: InfosBasicas
    ib_telefone_recado: Optional[str]
    ib_telefone: Optional[str]
    ib_objetivo_profissional: Optional[str]
    ib_data_criacao: Optional[str]
    ib_inserido_por: Optional[str]
    ib_email: Optional[str]
    ib_local: Optional[str]
    ib_sabendo_de_nos_por: Optional[str]
    ib_data_atualizacao: Optional[str]
    ib_codigo_profissional: Optional[str]
    ib_nome: Optional[str]

    # informacoes_pessoais: InformacoesPessoais
    ip_data_aceite: Optional[str]
    ip_download_cv: Optional[str]
    ip_nome: Optional[str]
    ip_cpf: Optional[str]
    ip_fonte_indicacao: Optional[str]
    ip_email: Optional[str]
    ip_email_secundario: Optional[str]
    ip_data_nascimento: Optional[str]
    ip_telefone_celular: Optional[str]
    ip_telefone_recado: Optional[str]
    ip_sexo: Optional[str]
    ip_estado_civil: Optional[str]
    ip_pcd: Optional[str]
    ip_endereco: Optional[str]
    ip_skype: Optional[str]
    ip_url_linkedin: Optional[str]
    ip_facebook: Optional[str]

    # informacoes_profissionais: InformacoesProfissionais
    ip_titulo_profissional: Optional[str]
    ip_area_atuacao: Optional[str]
    ip_conhecimentos_tecnicos: Optional[str]
    ip_certificacoes: Optional[str]
    ip_outras_certificacoes: Optional[str]
    ip_remuneracao: Optional[str]
    ip_nivel_profissional: Optional[str]

    # formacao_e_idiomas: FormacaoIdiomas
    fei_nivel_academico: Optional[str]
    fei_instituicao_ensino_superior: Optional[str]
    fei_cursos: Optional[str]
    fei_ano_conclusao: Optional[str]
    fei_nivel_ingles: Optional[str]
    fei_nivel_espanhol: Optional[str]
    fei_outro_idioma: Optional[str]

    cv_pt: Optional[str]
    cv_en: Optional[str]
