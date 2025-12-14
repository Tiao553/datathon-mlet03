import polars as pl
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Carrega o modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

# Função para unir listas de campos técnicos em texto


def concat_feature_text(row):
    candidato = (row.get('app_competencias_tecnicas', []) +
                 row.get('app_ferramentas_tecnologicas', []) +
                 row.get('app_principais_ferramentas_tecnologicas', []))
    vaga = (row.get('job_competencias_tecnicas', []) +
            row.get('job_ferramentas_tecnologicas', []))
    return " ".join(candidato), " ".join(vaga)


if __name__ == "__main__":
    # Load the dataset
    df_as = pl.read_parquet('./data/feature_store/resultado_final.parquet')

    # Coleta os textos a partir do Polars DataFrame
    textos_candidato = []
    textos_vaga = []

    for row in df_as.to_dicts():
        txt_candidato, txt_vaga = concat_feature_text(row)
        textos_candidato.append(txt_candidato)
        textos_vaga.append(txt_vaga)

    # Embeddings
    emb_candidato = model.encode(textos_candidato, convert_to_tensor=True)
    emb_vaga = model.encode(textos_vaga, convert_to_tensor=True)

    # Similaridade de cosseno
    scores = util.cos_sim(emb_candidato, emb_vaga).diagonal().cpu().numpy()

    # Adiciona a nova coluna no DataFrame Polars
    df = df_as.with_columns([
        pl.Series(name="score_tecnico_similaridade", values=scores)
    ])

    # Extrai os scores como array numpy
    scores_np = df.select(
        "score_tecnico_similaridade").to_numpy().reshape(-1, 1)

    # Clusterização com k = 3
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(scores_np)

    # Adiciona ao Polars DataFrame
    df = df.with_columns([
        pl.Series(name="cluster_tecnico", values=labels)
    ])

    print(df.group_by("cluster_tecnico").agg([
        pl.col("score_tecnico_similaridade").mean().alias(
            "media_score_cluster")
    ]).sort("cluster_tecnico").head())
