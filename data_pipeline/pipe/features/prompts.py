
import time
import os
import sys

# Add project root to path if needed (though usually handled by execution context)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import the new Infrastructure Gateway
from data_pipeline.infra.llm_gateway import get_llm_provider

###########################################################
# Prompt Functions
###########################################################


def extrair_texto(row, key):
    valor = row.get(key, "")
    if isinstance(valor, list):
        valor = valor[0] if valor else ""
    return str(valor).strip() if valor else ""


def prompt_vaga(row):
    return f"""
Você é um especialista em Recursos Humanos com foco em análise de descrições de vagas.
Sua tarefa é extrair e estruturar todas as informações relevantes.

Caso algum dado não esteja presente, use “Não mencionado”.
Título: {extrair_texto(row, 'job_ib_titulo_vaga')}
Atividades: {extrair_texto(row, 'job_pv_principais_atividades')}
Competências: {extrair_texto(row, 'job_pv_competencia_tecnicas_e_comportamentais')}
Observações: {extrair_texto(row, 'job_pv_demais_observacoes')}
Habilidades comportamentais: {extrair_texto(row, 'job_pv_habilidades_comportamentais_necessarias')}

---
Com base nessas informações, responda somente este JSON, sem deixar campos genéricos:
json
{{
  "ferramentas_tecnologicas": [
    "liste ao menos 10 e no maximo 13 ferramentas, plataformas, linguagens de programação, frameworks, ambientes, bancos de dados, sistemas ou metodologias técnicas mencionadas no currículo.
    - explícita ou implicitamente.
    - Utilize apenas 1 ou 2 palavras por item.
    - Cada item deve conter apenas UMA tecnologia ou ferramenta.
    - Caso encontre nomes agrupados, separe em itens individuais.
    - se tiver em ingles, traduza para português.
  ],
  "competencias_tecnicas": [
    - liste ao menos 10 e no maximo 13 competências técnicas (hard skills), tais como linguagens, frameworks, ferramentas, metodologias, certificações e especialidades técnicas.
    - Utilize apenas 1 ou 2 palavras por item.
    - Caso encontre nomes agrupados, separe em itens individuais.
    - Não inclua nomes de empresas, cidades ou cargos.
    - se tiver em ingles, traduza para português.
  ],
  "competencias_comportamentais": [
    - liste ao menos 10 e no maximo 13 competências comportamentais (soft skills) e características profissionais.
    - incluindo tanto menções explícitas quanto habilidades inferidas do contexto do currículo.
    - (exemplo: proatividade, liderança, adaptabilidade, comunicação, trabalho em equipe, pensamento analítico, resiliência, criatividade, foco em resultados, ética).
    - Utilize apenas 1 ou 2 palavras por item.
    - Caso encontre nomes agrupados, separe em itens individuais.
  ],
  "experiencia_anos": "Informe uma das opções padronizadas: '0-2 anos', '2-5 anos', '5-8 anos', '8-10 anos', '10+ anos' ou 'Não mencionado'. Baseie-se em datas, frases ou tempo total estimado.",
  "senioridade_aparente": "Classifique com base nas experiências e termos utilizados no currículo. Opções: Estágio, Júnior, Pleno, Sênior, Especialista, Não mencionado.",
  "formacao_academica": "ESCREVA UMA STRING Indicando 'true' se há formação acadêmica mencionada no currículo, ou 'false' se não houver qualquer menção.",
  "nivel_formacao": "Escolha entre: Ensino Médio, Tecnólogo, Superior Completo, Pós-graduação, Mestrado, Doutorado, Não mencionado.",
  "area_formacao": "Informe a área principal de formação, como TI, Engenharia, Administração, ou 'Não mencionado'."
}}
Regras:
- Não invente dados.
- Fundamente inferências em padrões de mercado.
- Saída: SOMENTE o JSON, sem comentários, códigos ou explicações adicionais.
- Responda em português.
- GARANTA que o JSON esteja completo e siga o modelo fornecido.
""".strip()


def prompt_candidato(row):
    return f"""
Você é um especialista em análise de currículos para posições técnicas de TI. Analise cuidadosamente o currículo abaixo:

{extrair_texto(row, 'app_cv_pt')}

---
Devolva apenas um objeto em formato JSON conforme o modelo abaixo, preenchendo todos os campos solicitados:
json
{{
  "principais_ferramentas_tecnologicas" : [
    - a princiapal ferramentas tecnológicas.
    - Exemplo Linguamgem pricipal, e ferramentas e serviços mais relevantes mencionadas no currículo.
    - separadas por vírgula.
    - Utilize apenas 1 ou 2 palavras por item.
  ],
  "ferramentas_tecnologicas": [
    "liste ao menos 10 e no maximo 13 ferramentas, plataformas, linguagens de programação, frameworks, ambientes, bancos de dados, sistemas ou metodologias técnicas mencionadas no currículo.
    - explícita ou implicitamente.
    - Utilize apenas 1 ou 2 palavras por item.
    - Cada item deve conter apenas UMA tecnologia ou ferramenta.
    - Caso encontre nomes agrupados, separe em itens individuais.
    - se tiver em ingles, traduza para português.
  ],
  "competencias_tecnicas": [
    - liste ao menos 10 e no maximo 13 competências técnicas (hard skills), tais como linguagens, frameworks, ferramentas, metodologias, certificações e especialidades técnicas.
    - Utilize apenas 1 ou 2 palavras por item.
    - Caso encontre nomes agrupados, separe em itens individuais.
    - Não inclua nomes de empresas, cidades ou cargos.
    - se tiver em ingles, traduza para português.
  ],
  "competencias_comportamentais": [
    - liste ao menos 10 e no maximo 13 competências comportamentais (soft skills) e características profissionais.
    - incluindo tanto menções explícitas quanto habilidades inferidas do contexto do currículo.
    - (exemplo: proatividade, liderança, adaptabilidade, comunicação, trabalho em equipe, pensamento analítico, resiliência, criatividade, foco em resultados, ética).
    - Utilize apenas 1 ou 2 palavras por item.
    - Caso encontre nomes agrupados, separe em itens individuais.
  ],
  "experiencia_anos": "Informe uma das opções padronizadas: '0-2 anos', '2-5 anos', '5-8 anos', '8-10 anos', '10+ anos' ou 'Não mencionado'. Baseie-se em datas, frases ou tempo total estimado.",
  "senioridade_aparente": "Classifique com base nas experiências e termos utilizados no currículo. Opções: Estágio, Júnior, Pleno, Sênior, Especialista, Não mencionado.",
  "formacao_academica": "ESCREVA UMA STRING Indicando 'true' se há formação acadêmica mencionada no currículo, ou 'false' se não houver qualquer menção.",
  "nivel_formacao": "Escolha entre: Ensino Médio, Tecnólogo, Superior Completo, Pós-graduação, Mestrado, Doutorado, Não mencionado.",
  "area_formacao": "Informe a área principal de formação, como TI, Engenharia, Administração, ou 'Não mencionado'."
}}
Regras:
- Não invente dados.
- Fundamente inferências em padrões de mercado.
- Saída: SOMENTE o JSON, sem comentários, códigos ou explicações adicionais.
- Responda em português.
- GARANTA que o JSON esteja completo e siga o modelo fornecido.
""".strip()

# Função para análise do currículo via LLM Gateway (Adapter Pattern)


def chamar_llm(prompt, model_name=None):
    """
    Chama o LLM através do Gateway configurado.
    O gateway decide se usa Ollama (Local) ou DeepSeek (Cloud) baseado em env vars.
    """
    provider = get_llm_provider()
    
    # Se model_name não for passado, o adapter usa o default do env ou da classe
    # Se for passado (como 'gemma3:4b'), o adapter tenta honrar se possível/relevante
    try:
        return provider.generate(prompt, model_name=model_name)
    except Exception as e:
        print(f"Erro na chamada do LLM: {e}")
        return ""


def chamar_llm_com_retry(prompt: str, logger, max_retries: int = 3, delay: int = 2) -> str:
    for tentativa in range(1, max_retries + 1):
        try:
            resposta = chamar_llm(prompt)
            if resposta and resposta.strip():
                return resposta
            else:
                logger.warning(
                    f"[LLM RETRY] Resposta vazia na tentativa {tentativa}")
        except Exception as e:
            logger.warning(f"[LLM RETRY] Erro na tentativa {tentativa}: {e}")
        time.sleep(delay)
    raise RuntimeError(
        "Falha ao obter resposta do LLM após múltiplas tentativas.")


def chamar_deepseek(prompt: str) -> str:
    """
    Wrapper legado para manter compatibilidade, mas agora usa o Gateway.
    Força o uso do provider DeepSeek se instanciado explicitamente, 
    ou confia no gateway se a config estiver certa.
    
    Para garantir comportamento, aqui podemos instanciar o adapter direto se necessário,
    mas idealmente usamos o env var LLM_PROVIDER='deepseek'.
    """
    # Opção A: Usar o gateway (respeita .env)
    return chamar_llm(prompt, model_name="deepseek-chat")
    
    # Opção B (Se quisermos forçar DeepSeek independente do env):
    # from data_pipeline.infra.llm_gateway import DeepSeekAdapter
    # return DeepSeekAdapter().generate(prompt)
