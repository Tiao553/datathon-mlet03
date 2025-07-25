import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuração do modelo
model_id = "google/gemma-2b-it"
token = "asad"

# Verifica se há GPU disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(
    model_id, token=token, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    token=token,
    trust_remote_code=True
)
prompt = '''Você é um especialista em análise de currículos com foco em recrutamento técnico e comportamental para posições de TI e correlatas. Sua tarefa é:
Ler atentamente o conteúdo abaixo, extraído do currículo (texto livre).

Conteúdo do currículo (texto livre):
DADOS PESSOAIS ESTADO CIVIL: CASADO IDADE: 33 ANOS OBJETIVO LEVAR À EMPRESA O MEU CONHECIMENTO, TORNANDO-ME UM MEMBRO A MAIS PARA O SEU DESENVOLVIMENTO. FORMAÇÃO ACADÊMICA UNIVERSIDADE ESTÁCIO DE SÁ  RJ MBA EM CIÊNCIA DE DADOS E BIG DATA ANALYTICS - CONCLUÍDO UNIVERSIDADE ESTÁCIO DE SÁ  RJ SUPERIOR EM ANÁLISE E DESENVOLVIMENTO DE SISTEMAS - CONCLUÍDO ETE JUSCELINO KUBITSCHEK FAETEC TÉCNICO EM ELETRICIDADE - CONCLUÍDO CERTIFICADOS  CERTIFICADO ITIL FOUNDADION V3 CURSOS DE APERFEIÇOAMENTO MONTAGEM E MANUTENÇÃO DE MICROS, DESENVOLVIMENTO DELPHI PLENO, LINGUAGEM PL/SQL, DESENVOLVEDOR JAVA WEB PLENO, SCRUM FUNDAMENTALS SFC , ITIL FOUNDATION V3, DB2 LUW - DATABASE ADMINISTRATION  CERTIFICATION WORKSHOP IDIOMAS ESPANHOL: AVANÇADO INGLÊS: AVANÇADO EXPERIÊNCIA ANALISTA DE SISTEMAS SÊNIOR - POR CSC, PROJETO PETROBRÁS DEZ/2020 - PRESENTE. ALOCADO NO CLIENTE PETROBRAS, FAÇO PARTE DA EQUIPE RESPONSÁVEL POR ADMINISTRAR A FERRAMENTA BMC CONTROL-M NA PARTE DE SUPORTE E PRODUÇÃO INSTALAÇÃO E CONFIGURAÇÃO DE AGENT, CMS, FIXES E UPGRADE DO CONTROL-M EM SERVIDORES UNIX, LINUX E WINDOWS. CRIAÇÃO DE JOBS UTILIZANDO ROTINAS EM SHELL SCRIPT E/OU BATS. CONFIGURAÇÃO DE JOBS DE SAP ETL BI DATA WAREHOUSE, BD, JAVA, WEB SERVICES, FILE TRANSFER AFT, ETC... NO CONTROL-M DE ACORDO COM A REGRA DE NEGÓCIO DO CLIENTE. EXTRAÇÃO DE RELATÓRIOS E OTIMIZAÇÃO DO AMBIENTE DE PRODUÇÃO UTILIZANDO O PL/SQL UTILIZAÇÃO DO IBM CONNECT:DIRECT PARA A TRANSFERÊNCIA DE ARQUIVOS INTERNO E EXTERNO NO CLIENTE. RESPONSÁVEL PELA FERRAMENTA IBM B2B STERLING INTEGRATOR, NA QUAL ADMINISTRO E PRESTO SUPORTE PARA ÁREA FINANCEIRA DA EMPRESA. SUPORTE AOS PROCESSOS DE CONCILIAÇÃO BANCÁRIA EDI, GARANTINDO O FUNCIONAMENTO NAS TRANSAÇÕES FINANCEIRAS DA EMPRESA VIA CONTROL_M JUNTO AS VANS E BANCOS. ÚNICO RESPONSÁVEL PELO AMBIENTE MAINFRAME COM FOCO EM SISTEMAS DE PLANO DE SAÚDE, ATUANDO NO DESENVOLVIMENTO DE NOVOS PROCESSOS JCL, ANALISE E CRIAÇÃO DE GDGS, MANUTENÇÃO DE TABLESPACES DB2, SUSTENTAÇÃO DO CICS, SUSTENTAÇÃO DO MVS PARA COMUNICAÇÃO COM BAIXA PLATAFORMA E DMZS PARA TRANSFERENCIA DE ARQUIVOS FINANCEIROS PARA OS BANCOS. ANALISTA DE PRODUÇÃO PLENO - POR COONECTCOM, PROJETO PETROBRÁS JUN/19 - DEZ/2020. SUPORTE A PRODUÇÃO 3º E 4º NIVEL EM AMBIENTES MULTIPLATAFORMA E MAINFRAME. MULTIPLATAFORMA DESENVOLVIMENTO E AGILIDADE DE JOBS VIA CONTROL-M, PROCESSOS AUTOMÁTICOS DE BACKUP, SHELL SCRIPTS, QUERYS, PROGRAMAS ABAPSAP, BI, DATA WAREHOUSE, WEB SERVICES, FILE TRANSFER AFT, ETL, FILEWATCH, BD, BATS, HADOOP, WORKFLOWS EXTRAÇÃO DE DADOS PL/SQL UTILIZAÇÃO DO CONNECT DIRECT PARA TRANSMISSÃO DE DADOS, UTILIZADO PARA ENVIO E RECEBIMENTO DE ARQUIVOS MANUTENÇÃO DE AGENTES DO CONTROL-M SISTEMAS OPERACIONAIS: LINUX, WINDOWS SERVER E UNIX. SUPORTE NOS PROCESSOS FINANCEIROS EDI DE TRANSAÇÕES BANCÁRIAS, AUTOMATIZADAS VIA CONTROL-M, CONNECT DIRETECT E APOIO AO CLIENTE JUNTO COM OS BANCOS E VANS. MAINFRAME SISTEMAS OPERACIONAIS: Z/NM, Z/OS, ZLINUX:  CUSTOMIZAÇÃO, HOMOLOGAÇÃO, ADMINISTRAÇÃO E IMPLANTAÇÃO DE SOFTWARES DA BMC PARA IBM MAINFRAME FAMÍLIA IOA, BMC CONTROL M/O/R/D  EXECUÇÃO DE POCS PROOFS OF CONCEPF PARA HOMOLOGAÇÃO DE SOFTWARE, E CONFECÇÃO DE RELATÓRIOS GERENCIAIS  CODIFICAÇÃO E MANUTENÇÃO DE JCLS, CARTÕES DE CONTROLE, PROCEDURES E OUTROS APLICATIVOS PERTENCENTES AOS SISTEMAS DE INFRAESTRUTURA NA PRODUÇÃO DO AMBIENTE CONTROL-M MAINFRAME  SUPORTE E ADMINISTRAÇÃO DE ROTINAS DE PRODUÇÃO PARA COLETA E SALVA DE REGISTROS DE SMF:  SUPORTE E ADMINISTRAÇÃO DE PROGRAMAS, SUB-ROTINAS, JCL, CARTÕES DE CONTROLE, PROCEDURES E OUTROS OBJETOS PERTENCENTES AOS SISTEMAS DE INFRAESTRUTURA DO AMBIENTE MAINFRAME  SUPORTE E ADMINISTRAÇÃO DO GERENCIADOR DE FITOTECA CA1  APLICAÇÕES/SOFTWARES:  TSO/SDSF  ROSCOE  RPF  JES2  NJE  SSH  SFTP  DFHSM  FTP - LINGUAGENS DE PROGRAMAÇÃO:  JCL  AUTO EDIT ANALISTA DE PRODUÇÃO V - PREPOSTO, POR GP EM CAIXA ECONÔMICA FEDERAL JUN/18 - JAN/2019. ATUANDO COMO PREPOSTO NA PLATAFORMA MAINFRAME USANDO OS SISTEMAS ZOS/360, SDSF, IPSF E CONTROL-M PARA MONITORAÇÃO E TRATAMENTO DE ERROS NO AMBIENTE DE PRODUÇÃO, RECUPERAÇÃO DE ARQUIVOS EM DISCO E EM FITAS DE BACKUP, EXECUÇÃO DE PACOTES E JCL VIA ENDEVOR, EXPERIÊNCIA TAMBÉM EM CICS, IDMS, DB2 E SQL. ATUANDO TAMBÉM EM PLATAFORMA DE AMBIENTE DISTRIBUÍDO NA EXECUÇÃO DE PROGRAMAS EM SHELLSCRIPT, SQL, JAVA, VIA CONTROL-M ENTERPRISE E POWERCENTER. MONITORAÇÃO DE SCRIPTS E JOBS AUTOMÁTICOS, MANUTENÇÃO E TRATAMENTO DE ERROS DOS MESMOS, EXECUÇÕES DE ARQUIVOS B2B, IMPORTAÇÃO DE MAPAS VIA CLEARCASE E NEXUS. ATUANDO COM AS SEGUINTES FERRAMENTAS POWERCENTER WORKFLOW MONITOR, MANAGER E REPOSITORY. EM AMBIENTES WINDOWS E LINUX. CONHECIMENTO EM SHELLSCRIPT E PL/SQL. AUXILIANDO NA COORDENAÇÃO DA EQUIPE EM AMBAS PLATAFORMAS ALTA E BAIXA, RESPONDENDO SLM E COMPARECENDO A REUNIÕES COM O CLIENTE. ANALISTA DE SUPORTE A PRODUÇÃO, POR INFOREGIS EM GÁS NATURAL FENOSACEG JAN/15  JUN/18. DESENVOLVIMENTO DE PROCESSOS JCL PARA INTEGRAÇÃO COM O CONTROL-M, EM AMBIENTE MAINFRAME. EXPERIÊNCIA EM CONTROL-M 6.0, CONTROL-M 8.0, CONTROL-D, ADMINISTRAÇÃO E MANUTENÇÃO DE JOBS EM JCL, UTILITÁRIOS DE RECUPERAÇÃO DE ARQUIVOS, GERENCIAMENTO DE IMPRESSORAS E PERIFÉRICOS SOB O Z/OS OS/390 ADMINISTRADOS PELO SUBSISTEMA JES3. GERENCIAMENTO DE FLUXOS DE JOBS CONTROL-M, AUTOMAÇÃO DE PROCESSOS BATCH ATRAVÉS DE SCRIPTS SHELL/SQL, ACOMPANHAMENTO DE DESEMPENHO DA PRODUÇÃO, MONITORAMENTO DE PROCESSOS ATRAVÉS DO CONTROL-M ENTERPRISE, SÓLIDOS CONHECIMENTOS EM IMPLANTAÇÃO DE JCL AUTOMATIZADOS NO CONTROL-M EM AMBIENTE MAINFRAME, IMPLEMENTAÇÃO DE PROCESSOS EM SHELLSCRIPT, PL/SQL PARA PROCESSAMENTO AUTOMÁTICO CONTROL-M. MONITORAMENTO NAGIOS DE SISTEMAS DE REDE. PARTICIPAÇÃO NO PROJETO ZEUS DESENVOLVIDO COM FINALIDADE DE IMPLANTAR UM SISTEMA INTEGRANDO SIEBEL E SGC, EM ALTA E BAIXA PLATAFORMA. ATUANDO TAMBÉM NA COORDENAÇÃO TÉCNICA DA EQUIPE DE SCHEDULER, FORNECENDO SUPORTE QUANDO NECESSÁRIO. INFORMÁTICA CONHECIMENTOS EM JCL, SHELL SCRIPT, HTML, DELPHI, SQL, MODELAGEM DE DADOS, SCRUM, ITIL, JAVA, C, ENDEVOR, TSO/ISPF, SDSF, MAINFRAME IBM, CONTROL-M, Z/OS, POWER CENTER, CLEARCASE, B2B, LINGUAGEM R, MVS, JBOSS, JEE, JAVA, CONNECT DIRECT, CICS, DB2.

Devolva apenas um objeto em formato JSON conforme o modelo abaixo, preenchendo todos os campos solicitados:
json
{{
  "ferramentas_tecnologicas": [
    "Liste todas as ferramentas, plataformas, linguagens de programação, frameworks, ambientes, bancos de dados, sistemas ou metodologias técnicas mencionadas no currículo, explícita ou implicitamente. Cada item deve conter apenas UMA tecnologia ou ferramenta. Caso encontre nomes agrupados, separe em itens individuais. Caso não haja menção, escreva 'Não mencionado'."
  ],
  "competencias_tecnicas": [
    "Liste o máximo de competências técnicas (hard skills), tais como linguagens, frameworks, ferramentas, metodologias, certificações e especialidades técnicas, relacionadas à área de atuação. Utilize apenas 1 ou 2 palavras por item. Não inclua nomes de empresas, cidades ou cargos. Caso não haja menção, escreva 'Não mencionado'."
  ],
  "competencias_comportamentais": [
    "Liste ao menos 10 competências comportamentais (soft skills) e características profissionais, incluindo tanto menções explícitas quanto habilidades inferidas do contexto do currículo (exemplo: proatividade, liderança, adaptabilidade, comunicação, trabalho em equipe, pensamento analítico, resiliência, criatividade, foco em resultados, ética). Caso não haja menção, escreva 'Não mencionado'."
  ],
  "experiencia_anos": "Identifique, a partir do texto, quantos anos de experiência profissional o candidato possui. Sempre retorne UMA das faixas padronizadas abaixo, conforme o valor explícito ou implícito: '0-2 anos', '2-5 anos', '5-8 anos', '8-10 anos', '10+ anos'. Caso não seja possível determinar, escreva 'Não mencionado'. Para menções como '10 anos', '10+', 'Mais de 10 anos', '20 anos', '20+', utilize sempre a faixa '10+ anos'.",
  "senioridade_aparente": "Classifique objetivamente o nível de senioridade do candidato com base nas experiências apresentadas. Escolha entre: Estágio, Júnior, Pleno, Sênior, Especialista, ou 'Não mencionado'.",
  "formacao_academica": "Indique true se há formação acadêmica mencionada no currículo, ou false se não houver qualquer menção.",
  "nivel_formacao": "Informe o nível de formação acadêmica mais elevado alcançado (exemplo: Ensino Médio, Tecnólogo, Superior Completo, Pós-graduação, Mestrado, Doutorado, Não mencionado).",
  "area_formacao": "Informe a principal área de formação acadêmica (exemplo: TI, Administração, Engenharia, Direito, Não mencionado).",
  "nivel_ingles": "Classifique o nível de inglês conforme explicitado ou inferido: Básico, Intermediário, Avançado, Fluente, Nativo ou 'Não mencionado'."
}}

Atenção:
- Seja fiel ao texto; não invente informações.
- Ao inferir, fundamente sua resposta de acordo com padrões de mercado (exemplo: domínio de ferramentas avançadas pode indicar senioridade maior).
- Não inclua explicações fora do JSON.
**A saída deve ser SOMENTE o objeto JSON, sem qualquer texto adicional, introdução, explicação ou bloco de código.**
'''

# Aplica estrutura de chat para o modelo
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt").to(device)

# Monitoramento antes da inferência
start_time = time.time()
cpu_before = psutil.cpu_percent(interval=None)
mem_before = psutil.virtual_memory().used

# Geração da resposta
with torch.no_grad():
    output_ids = model.generate(
        inputs,
        max_new_tokens=4096,
        do_sample=False,
        temperature=0.7
    )
decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Monitoramento após a inferência
cpu_after = psutil.cpu_percent(interval=None)
mem_after = psutil.virtual_memory().used
exec_time = time.time() - start_time

# Exibe resultados
print(decoded)
print(f"\nTempo de execução: {exec_time:.2f} segundos")
print(f"Uso de CPU: {cpu_after - cpu_before:.2f}%")
print(f"Uso de RAM: {(mem_after - mem_before) / 1024 / 1024:.2f} MB")
