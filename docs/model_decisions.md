# Decisões de Modelagem e Algoritmos

Este documento detalha as escolhas científicas e de engenharia por trás dos modelos preditivos do sistema.

## 1. Score Comportamental (Machine Learning)

### 1.1 O Modelo: LightGBM
O **LightGBM (Light Gradient Boosting Machine)** foi escolhido como o motor principal para o score comportamental.
*   **Motivo - Velocidade**: Utiliza o algoritmo *histogram-based* para split de árvores, tornando o treino até 20x mais rápido que o XGBoost tradicional em CPUs.
*   **Motivo - Leaf-wise Growth**: Cresce as árvores por folha (best-first) em vez de por nível (depth-wise). Isso geralmente resulta em menor erro loss, embora exija controle de overfitting (`max_depth`).
*   **Motivo - Dados Faltantes**: O LightGBM decide automaticamente a direção padrão para valores `NaN` durante o split, eliminando a necessidade de imputação (média/mediana) que poderia introduzir ruído.

### 1.2 Tratamento de Desbalanceamento de Classes
Em recrutamento, o número de candidatos "contratados" (classe positiva) é muito menor que o número de rejeitados.
*   **Estratégia**: Utilizamos o parâmetro `is_unbalance=True`.
*   **Efeito**: O algoritmo ajusta automaticamente os pesos da função de perda (Loss Function), penalizando mais os erros na classe minoritária (contratados). Isso maximiza o Recall, garantindo que não percamos talentos promissores.

### 1.3 Análise de Importância das Features (Feature Importance)
Baseado nos experimentos (`model/experiments/run_behavioral_exp.py`), as features mais impactantes foram:
1.  **`sentimento_comentario_score`**: Crítica. Comentários positivos de triagens anteriores são o maior preditor de sucesso futuro.
2.  **`percentual_perfil_completo`**: Candidatos que preenchem todo o perfil demonstram maior interesse/engajamento.
3.  **`dias_desde_ultima_atualizacao`**: Candidatos "frescos" (recência) têm maior probabilidade de resposta.

## 2. Scores Semânticos (NLP/Embeddings)

### 2.1 Modelo de Linguagem: SBERT
Para os scores Técnico e Cultural, utilizamos o `sentence-transformers/all-MiniLM-L6-v2`.
*   **Arquitetura**: Baseado no BERT, treinado com *Contrastive Learning* em mais de 1 bilhão de pares de sentenças.
*   **Output**: Vetor denso de 384 dimensões.
*   **Performance**: Processa ~14.000 sentenças por segundo (em GPU T4), garantindo escalabilidade.

### 2.2 Estratégia "Cold Start" (Início a Frio)
O que acontece quando um candidato novo chega sem histórico?
*   **Score Comportamental**: O modelo recebe `Score Sentimento = 0` (Neutro) e `Dias Atualização = 0` (Hoje). O score resultante tende para a média (~0.5), evitando penalizar injustamente o candidato novo.
*   **Score Técnico**: Depende puramente do CV. Se o CV for muito curto/vazio, a extração do LLM retorna listas vazias, resultando em Score 0.0. Isso é intencional: sem dados, não há match.

## 3. Hiperparâmetros Finais

### LightGBM (Behavioral)
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': True,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'max_depth': 15,
    'num_leaves': 31,
    'feature_fraction': 0.8, # Subsample de colunas para reduzir overfitting
    'random_state': 42
}
```

### Thresholds de Similaridade (Skills)
*   **Corte (Threshold)**: `0.5`
*   **Explicação**: Apenas pares (Skill Vaga, Skill Candidato) com similaridade coseno > 0.5 são considerados no cálculo da média. Isso filtra correspondências irrelevantes (ex: "Java" vs "Javascript" pode ter similaridade vetorial baixa mas não nula; o threshold remove esse ruído).
