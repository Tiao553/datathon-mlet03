# --- 1. Configuração e Carregamento ---
# O print inicial serve para marcar o começo do processo.
import joblib  # Para salvar nosso modelo final em um arquivo.
# Para definir as faixas de busca aleatória dos hiperparâmetros.
from scipy.stats import randint, uniform
# O nosso modelo avançado de Machine Learning (LightGBM).
import lightgbm as lgb
import datetime  # Para trabalhar com datas, como a data atual.
import matplotlib.pyplot as plt  # Para gerar os gráficos.
# Para preparar os dados para o modelo (normalizar números e codificar categorias).
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Para aplicar diferentes preparações a diferentes colunas.
from sklearn.compose import ColumnTransformer
# Para avaliar a performance do modelo.
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay
# Para dividir os dados e otimizar o modelo.
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# Para operações numéricas, muito usado por outras bibliotecas.
import numpy as np
import polars as pl  # Para manipulação de dados em alta velocidade.
print("--- Iniciando o processo de construção do modelo final ---")

# Importa as bibliotecas (ferramentas) que vamos precisar para o projeto.
# A linha abaixo foi removida pois Pipeline não é mais necessário na abordagem final e mais rápida.
# from sklearn.pipeline import Pipeline

# --- Etapa 1: Carregamento dos Dados ---
# Imprime no console em qual etapa estamos, para facilitar o acompanhamento.
print("\n[Etapa 1/5] Carregando a base de dados...")

# Define uma variável com o caminho completo até o arquivo de dados.
caminho_base = "/home/tnsantos/datathon-recrutamento-ia/teste4/resultado_final.parquet"

# Este é um bloco de segurança `try-except`. O Python tentará executar o código dentro do `try`.
try:
    # `pl.read_parquet` lê o arquivo no formato Parquet (ótimo para grandes volumes) e o carrega em um DataFrame do Polars.
    df = pl.read_parquet(caminho_base)
    # Se o carregamento for bem-sucedido, imprime uma mensagem de confirmação com o formato do DataFrame (linhas, colunas).
    print(f"Base de dados carregada com sucesso. Shape: {df.shape}")
# Se o Python não encontrar o arquivo no caminho especificado (gerando um `FileNotFoundError`), ele executa o bloco `except`.
except FileNotFoundError:
    # Imprime uma mensagem de erro clara e amigável.
    print(
        f"ERRO: O arquivo não foi encontrado em '{caminho_base}'. Verifique o caminho.")
    # Encerra a execução do script para evitar outros erros.
    exit()

# --- Etapa 2: Definição da Variável Alvo (y) ---
# Informa o progresso da execução.
print("\n[Etapa 2/5] Criando a variável alvo 'engajado'...")
# Criamos uma lista em Python com todos os status de candidato que, juntos, definimos como um "sucesso" no processo seletivo.
status_sucesso = [
    "CONTRATADO PELA DECISION", "CONTRATADO COMO HUNTING", "DOCUMENTAÇÃO PJ",
    "ENCAMINHADO AO REQUISITANTE", "APROVADO", "ENTREVISTA TÉCNICA"
]
# `df.with_columns` é o comando do Polars para adicionar ou modificar colunas no DataFrame.
df = df.with_columns(
    # A lógica a seguir cria a nova coluna "engajado":
    # `pl.when(...)` inicia uma condição.
    # `pl.col("p_situacao_candidado").is_in(status_sucesso)` verifica, para cada linha, se o valor da coluna "p_situacao_candidado" está contido na nossa lista `status_sucesso`.
    engajado=pl.when(pl.col("p_situacao_candidado").is_in(status_sucesso))
    # Se a condição for verdadeira, `.then(1)` define o valor da nova coluna "engajado" como 1.
    .then(1)
    # Se a condição for falsa, `.otherwise(0)` define o valor como 0.
    .otherwise(0)
)
print("Variável alvo 'engajado' criada com sucesso.")


# --- Etapa 3: Engenharia de Features ---
# Informa o início da etapa mais complexa, a criação de "pistas" para o modelo.
print("\n[Etapa 3/5] Executando a engenharia de features avançada...")

# `regex` são padrões de texto para encontrar palavras-chave. `(?i)` no início torna a busca insensível a maiúsculas/minúsculas.
regex_palavras_negativas = r"(?i)(n(a|ã)o\sresponde|desisti|sem\sinteresse|n(a|ã)o\sretorna|n(a|ã)o\satende)"
regex_palavras_positivas = r"(?i)(proativ|interessad|bom\sperfil|responsiv|gostou|avan(ç|c)ou|performou\sbem)"

# Define uma lista com os nomes das colunas que usaremos para medir a completude do perfil de um candidato.
colunas_perfil_app = [
    'app_ib_objetivo_profissional', 'app_ib_local', 'app_ip_sexo',
    'app_ip_estado_civil', 'app_ip_pcd', 'app_ip_remuneracao',
    'app_fei_nivel_academico', 'app_fei_nivel_ingles',
]

# --- Limpeza e Agrupamento de Features ---
# Garante que não há valores nulos na coluna `p_recrutador`, substituindo-os por "Desconhecido". Isso evita erros futuros.
df = df.with_columns(pl.col("p_recrutador").fill_null("Desconhecido"))
# Define um limite mínimo de aparições para um recrutador ser considerado "comum".
min_frequency = 10
# Conta quantas vezes cada recrutador aparece na base de dados. `pl.len()` é a forma moderna de `pl.count()`.
recruiter_counts = df.group_by("p_recrutador").agg(pl.len().alias("count"))
# Filtra a contagem para obter uma lista apenas com os recrutadores que aparecem `min_frequency` vezes ou mais.
common_recruiters = recruiter_counts.filter(
    pl.col("count") >= min_frequency)["p_recrutador"]

# --- Bloco Principal de Criação de Features ---
# `with_columns` permite criar várias colunas de uma vez de forma eficiente.
df = df.with_columns(
    # `sum_horizontal` soma os resultados de uma expressão para cada linha. Aqui, contamos quantos campos do perfil não são nulos e dividimos pelo total para obter uma porcentagem.
    percentual_perfil_completo=(pl.sum_horizontal(pl.col(c).is_not_null(
    ) for c in colunas_perfil_app) / len(colunas_perfil_app)).fill_null(0),
    # Cria uma flag (coluna com 0 ou 1) que é 1 se o objetivo profissional foi preenchido. `.cast(pl.Int8)` converte True/False para 1/0.
    tem_objetivo_profissional=(pl.col('app_ib_objetivo_profissional').is_not_null(
    ) & (pl.col('app_ib_objetivo_profissional') != "")).cast(pl.Int8),
    tem_remuneracao_definida=(pl.col('app_ip_remuneracao').is_not_null() & (
        pl.col('app_ip_remuneracao') != "")).cast(pl.Int8),
    # Mede o tamanho do CV contando o número de caracteres. `.fill_null(0)` garante que CVs vazios tenham tamanho 0.
    tamanho_cv=pl.col('app_cv_pt').str.len_chars().fill_null(0),
    # Usa o regex negativo para verificar se o comentário contém palavras-chave de desinteresse.
    contem_palavra_chave_negativa=pl.col('p_comentario').str.contains(
        regex_palavras_negativas).fill_null(False).cast(pl.Int8),
    # `.dt.epoch` converte datas para um número (segundos/dias desde 1970), permitindo cálculos matemáticos como subtração. Aqui, calculamos a diferença em dias.
    dias_para_se_candidatar=(pl.col('p_data_candidatura').dt.epoch(
        time_unit="d") - pl.col('job_ib_data_requicisao').dt.epoch(time_unit="d")).fill_null(-1),
    # Usa a flag que já existia na base, garantindo que valores nulos se tornem 0.
    candidatura_proativa=pl.col(
        'candidatura_antes_da_abertura_oficial').fill_null(0).cast(pl.Int8),
    # Calcula há quantos dias foi a última atualização, usando a data de hoje (`datetime.datetime.now`). `timezone.utc` garante consistência.
    dias_desde_ultima_atualizacao=(pl.lit(datetime.datetime.now(datetime.timezone.utc)).dt.epoch(
        time_unit="d") - pl.col('p_ultima_atualizacao').dt.epoch(time_unit="d")).fill_null(-1),
    # Usa o regex positivo para verificar se o comentário contém palavras-chave de interesse.
    contem_palavra_chave_positiva=pl.col('p_comentario').str.contains(
        regex_palavras_positivas).fill_null(False).cast(pl.Int8),
    # Calcula a duração do processo para um candidato (da candidatura até a última atualização).
    dias_no_processo=(pl.col('p_ultima_atualizacao').dt.epoch(
        time_unit="d") - pl.col('p_data_candidatura').dt.epoch(time_unit="d")).fill_null(-1),
    # Cria a coluna de recrutador com os valores raros agrupados em "Outros". `pl.lit("Outros")` garante que "Outros" é tratado como um valor de texto, e não um nome de coluna.
    p_recrutador_tratado=pl.when(pl.col("p_recrutador").is_in(
        common_recruiters)).then(pl.col("p_recrutador")).otherwise(pl.lit("Outros"))
)
# Com as flags de palavras-chave criadas, agora criamos um score de sentimento simples: +1 para positivo, -1 para negativo, 0 para neutro/ambos.
df = df.with_columns(
    sentimento_comentario_score=pl.when(pl.col('contem_palavra_chave_positiva') == 1).then(
        1).when(pl.col('contem_palavra_chave_negativa') == 1).then(-1).otherwise(0)
)
print("Engenharia de features concluída.")


# --- Etapa 4: Otimização de Hiperparâmetros ---
# Informa o início da etapa de machine learning.
print("\n[Etapa 4/5] Otimizando hiperparâmetros do modelo LightGBM...")

# Define as listas de features que serão usadas pelo pré-processador.
features_para_escalar = [  # Colunas numéricas e binárias que serão padronizadas.
    'percentual_perfil_completo', 'tamanho_cv', 'dias_para_se_candidatar',
    'dias_desde_ultima_atualizacao', 'dias_no_processo', 'sentimento_comentario_score',
    'tem_objetivo_profissional', 'tem_remuneracao_definida',
    'contem_palavra_chave_negativa', 'candidatura_proativa',
    'contem_palavra_chave_positiva'
]
# Colunas de texto/categoria que serão codificadas.
features_categoricas = ['p_recrutador_tratado']
# Lista com todas as features.
features_finais_df = features_para_escalar + features_categoricas
target = 'engajado'  # O nome da nossa coluna alvo.

# --- Preparação Final dos Dados ---
# Convertemos do formato Polars para Pandas, que tem integração mais estável com o scikit-learn.
df_pandas = df.to_pandas()
X = df_pandas[features_finais_df]  # X (maiúsculo) são nossas features.
y = df_pandas[target]           # y (minúsculo) é nosso alvo.

# `train_test_split` divide os dados em um conjunto para treinar (80%) e outro para testar (20%).
# `random_state=42` garante que a divisão seja sempre a mesma, para reprodutibilidade.
# `stratify=y` garante que a proporção de 0s e 1s no alvo seja a mesma nos conjuntos de treino e teste.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# --- Pré-processamento (Abordagem Robusta) ---
# O `ColumnTransformer` é um "organizador" que aplica os passos corretos às colunas certas.
preprocessor = ColumnTransformer(
    transformers=[
        # Para as features numéricas/binárias, `StandardScaler` as padroniza (média 0, desvio padrão 1).
        ('scaler', StandardScaler(), features_para_escalar),
        # Para a feature categórica, `OneHotEncoder` a transforma em colunas de 0s e 1s.
        # `handle_unknown='ignore'` evita erros se uma nova categoria aparecer nos dados de teste.
        ('cat', OneHotEncoder(handle_unknown='ignore',
         sparse_output=False), features_categoricas)
    ],
    remainder='drop'  # Ignora qualquer outra coluna que não foi especificada.
)
# Aplicamos o pré-processamento UMA ÚNICA VEZ antes da busca para otimizar a velocidade.
# `fit_transform` aprende os padrões dos dados de treino e os transforma.
X_train_processed = preprocessor.fit_transform(X_train)
# `transform` apenas aplica a mesma transformação aprendida nos dados de teste.
X_test_processed = preprocessor.transform(X_test)
print("Dados pré-processados e prontos para o modelo.")

# --- Definição da Busca ---
# Define o modelo base que queremos otimizar. `is_unbalance=True` ajuda o modelo a lidar com dados desbalanceados.
# `verbose=-1` "silencia" o LightGBM para evitar o excesso de logs.
model_base = lgb.LGBMClassifier(is_unbalance=True, random_state=42, verbose=-1)

# Define o "cardápio" de hiperparâmetros que a busca vai testar, com as faixas de valores.
param_distributions = {
    'n_estimators': randint(50, 300),        # Número de árvores
    'learning_rate': uniform(0.01, 0.1),     # Taxa de aprendizado
    'num_leaves': randint(10, 40),           # Complexidade da árvore
    'max_depth': [-1, 10, 20],               # Profundidade máxima
    'reg_alpha': uniform(0, 0.5),            # Regularização L1
    'reg_lambda': uniform(0, 0.5),           # Regularização L2
    # Fração de features a usar por árvore
    'colsample_bytree': uniform(0.7, 0.3)
}

# Configura a busca aleatória (`RandomizedSearchCV`).
random_search = RandomizedSearchCV(
    estimator=model_base,                      # O modelo a ser otimizado.
    param_distributions=param_distributions,   # O "cardápio" de parâmetros.
    # Número de combinações a testar.
    n_iter=50,
    # Validação cruzada com 5 "dobras".
    cv=5,
    scoring='roc_auc',                         # Métrica a ser maximizada.
    random_state=42,                           # Para reprodutibilidade.
    # Execução sequencial para evitar problemas.
    n_jobs=1,
    verbose=1                                  # Mostra o progresso da busca.
)

# Inicia a busca. Este é o passo mais demorado.
print("\nExecutando a busca pelos melhores hiperparâmetros...")
random_search.fit(X_train_processed, y_train)

# Imprime os resultados da busca.
print("\nBusca concluída!")
print(
    f"Melhor Score AUC (em validação cruzada): {random_search.best_score_:.4f}")
print("Melhores Hiperparâmetros encontrados:")
print(random_search.best_params_)

# --- Etapa 5: Avaliação do Modelo Final Otimizado ---
# Informa o início da última etapa.
print("\n[Etapa 5/5] Avaliando o melhor modelo no conjunto de teste...")

# Pega o "modelo campeão" com os melhores hiperparâmetros encontrados na busca.
best_model = random_search.best_estimator_

# `joblib.dump` salva o modelo treinado em um arquivo.
nome_arquivo_modelo = 'engagement_score_model.pkl'
joblib.dump(best_model, nome_arquivo_modelo)
print(f"Modelo final salvo com sucesso em: '{nome_arquivo_modelo}'")

# Usa o modelo final para prever as probabilidades no conjunto de teste.
# O `[:, 1]` pega especificamente a probabilidade da classe 1 (engajado), que é o nosso score.
y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
# `predict` dá a classificação final (0 ou 1).
y_pred_class = best_model.predict(X_test_processed)

# Calcula e imprime as métricas de performance finais.
final_auc = roc_auc_score(y_test, y_pred_proba)
final_report = classification_report(y_test, y_pred_class, zero_division=0)
print(f"\nAUC-ROC Score FINAL: {final_auc:.4f}")
print("\nRelatório de Classificação FINAL:")
print(final_report)

# Gera a Matriz de Confusão para visualizar os acertos e erros.
print("Gerando Matriz de Confusão FINAL...")
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test_processed, y_test, ax=ax, cmap='Blues')
    ax.set_title("Matriz de Confusão - Modelo Final Otimizado")
    plt.show()
except Exception as e:
    print(f"Não foi possível gerar a Matriz de Confusão: {e}")

# Mensagem final de conclusão.
print("\n--- Processo concluído ---")
