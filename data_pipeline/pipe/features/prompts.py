
import requests
import json
import time
import psutil
import time
from openai import OpenAI
import os

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

# Função para análise do currículo via Ollama API


def chamar_llm(prompt, model_name="gemma3:4b"):
    start_time = time.time()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().used

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options":  {
            "num_predict": 1536,        # menor contexto ajuda a manter foco
            "temperature": 0.1,        # zero = determinístico
            "top_p": 0.8,              # restringe variedade
        }
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().used
    exec_time = time.time() - start_time

    print(f"Tempo de execução: {exec_time:.2f} segundos")
    print(f"Uso de CPU: {cpu_after - cpu_before:.2f}%")
    print(f"Uso de memória: {mem_after - mem_before:.2f} bytes")

    result_json = response.json()

    return result_json.get('response', '').strip()


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
    Chama o modelo DeepSeek para gerar uma resposta baseada no prompt.
    A API Key é carregada automaticamente de DEEPSEEK_API_KEY se definida como variável de ambiente.
    """
    client = OpenAI(api_key="sk-", base_url="https://api.deepseek.com/v1")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ocorreu um erro ao chamar o DeepSeek: {e}"
