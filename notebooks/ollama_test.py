import requests
import json
import time
import psutil

# Função para análise do currículo via Ollama API


def analisar_curriculo(prompt, model_name="gemma3:4b"):  # gemma3:4b
    start_time = time.time()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().used

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 768, "temperature": 0.1, "top_p": 0.95}
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().used
    exec_time = time.time() - start_time

    result_json = response.json()
    return {
        "resposta": result_json.get('response', '').strip(),
        "tempo_execucao": exec_time,
        "uso_cpu": cpu_after - cpu_before,
        "uso_ram_mb": (mem_after - mem_before) / 1024 / 1024
    }


prompt = '''
Você é um especialista em Recursos Humanos com foco em análise de descrições de vagas.
Sua tarefa é extrair e estruturar todas as informações relevantes a partir de uma descrição de vaga de forma completa e minuciosa para gerar um "perfil estruturado da vaga".
seguem os dados abaixo:

Caso algum dado não esteja presente, use “Não mencionado”.
Título: CONSULTOR CONTROL M
Atividades: - EXPERIÊNCIA COMPROVADA EM PROJETOS DE CONTROL-M
Competências: - EXPERIÊNCIA COMPROVADA EM PROJETOS DE CONTROL-M
Observações: CONTRATAÇÃO PJ PROJETO PONTUAL DE 2 A 3 MESES CLIENTE: CAPGEMINI PROJETO: LIGHT / RIO DEJANEIRO LOCAL DE TRABALHO: HIBRIDA: REMOTA E PRESENCIAL NA LIGHT NO RJ QUANDO SOLICITADO
Habilidades comportamentais:
---
Com base nessas informações, responda somente este JSON, sem deixar campos genéricos:
json
{{
  "ferramentas_tecnologicas": [
    "liste ao menos 10 ferramentas, plataformas, linguagens de programação, frameworks, ambientes, bancos de dados, sistemas ou metodologias técnicas mencionadas no currículo.
    - explícita ou implicitamente.
    - Utilize apenas 1 ou 2 palavras por item.
    - Cada item deve conter apenas UMA tecnologia ou ferramenta.
    - Caso encontre nomes agrupados, separe em itens individuais.
    - se tiver em ingles, traduza para português.
  ],
  "competencias_tecnicas": [
    - liste ao menos 10 competências técnicas (hard skills), tais como linguagens, frameworks, ferramentas, metodologias, certificações e especialidades técnicas.
    - Utilize apenas 1 ou 2 palavras por item.
    - Caso encontre nomes agrupados, separe em itens individuais.
    - Não inclua nomes de empresas, cidades ou cargos.
    - se tiver em ingles, traduza para português.
  ],
  "competencias_comportamentais": [
    - liste ao menos 10 competências comportamentais (soft skills) e características profissionais.
    - incluindo tanto menções explícitas quanto habilidades inferidas do contexto do currículo.
    - (exemplo: proatividade, liderança, adaptabilidade, comunicação, trabalho em equipe, pensamento analítico, resiliência, criatividade, foco em resultados, ética).
    - Utilize apenas 1 ou 2 palavras por item.
    - Caso encontre nomes agrupados, separe em itens individuais.
  ],
  "experiencia_anos": "Informe uma das opções padronizadas: '0-2 anos', '2-5 anos', '5-8 anos', '8-10 anos', '10+ anos' ou 'Não mencionado'. Baseie-se em datas, frases ou tempo total estimado.",
  "senioridade_aparente": "Classifique com base nas experiências e termos utilizados no currículo. Opções: Estágio, Júnior, Pleno, Sênior, Especialista, Não mencionado.",
  "formacao_academica": "Indique true se há formação acadêmica mencionada no currículo, ou false se não houver qualquer menção.",
  "nivel_formacao": "Escolha entre: Ensino Médio, Tecnólogo, Superior Completo, Pós-graduação, Mestrado, Doutorado, Não mencionado.",
  "area_formacao": "Informe a área principal de formação, como TI, Engenharia, Administração, ou 'Não mencionado'."
}}
Regras:
- Não invente dados.
- Fundamente inferências em padrões de mercado.
- Saída: SOMENTE o JSON, sem comentários, códigos ou explicações adicionais.
- Responda em português.
'''

# Executando a análise
resultado = analisar_curriculo(prompt)


# Exibindo o resultado
print("✅ Resultado do modelo:\n")
print(resultado["resposta"])
print(f"\n⏱️ Tempo de execução: {resultado['tempo_execucao']:.2f} segundos")
print(f"🧠 Uso de CPU: {resultado['uso_cpu']:.2f}%")
print(f"💾 Uso de RAM: {resultado['uso_ram_mb']:.2f} MB")
