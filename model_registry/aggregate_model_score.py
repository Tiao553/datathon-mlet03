def modelo_agregador_v1(score_tecnico, score_cultural, score_engajamento):
    """
    Agrega os scores técnico, cultural e de engajamento para gerar um match geral e um status para o candidato.
    Implementa uma abordagem híbrida com um motor de regras (veto, destaque) e média ponderada.

    Args:
        score_tecnico (float): Score de 0 a 1 para o fit técnico.
        score_cultural (float): Score de 0 a 1 para o fit cultural.
        score_engajamento (float): Score de 0 a 1 para o nível de engajamento.

    Returns:
        dict: Um dicionário contendo o 'overall_match' (float) e o 'status' (str) do candidato.
    """
    
    # --- Bloco 1: Definição de Parâmetros e Thresholds (Limites) ---
    # Manter todos os parâmetros aqui facilita a manutenção e ajuste futuro do modelo.
    
    VETO_CULTURAL_THRESHOLD = 0.4        # Define o valor mínimo para o score cultural. Abaixo disso, é vetado.
    VETO_TECNICO_THRESHOLD = 0.3         # Define o valor mínimo para o score técnico. Abaixo disso, é vetado.
    
    DESTAQUE_TECNICO_THRESHOLD = 0.85    # Define o valor mínimo do score técnico para ser considerado um destaque.
    DESTAQUE_CULTURAL_THRESHOLD = 0.8    # Define o valor mínimo do score cultural para ser considerado um destaque.

    PESO_TECNICO = 0.5                   # Define o peso do score técnico na média ponderada (50%).
    PESO_CULTURAL = 0.3                  # Define o peso do score cultural na média ponderada (30%).
    PESO_ENGAGAMENTO = 0.2               # Define o peso do score de engajamento na média ponderada (20%).

    BUCKET_ALTO_POTENCIAL = 0.8          # Define a nota de corte para o status "Alto Potencial".
    BUCKET_POTENCIAL = 0.6               # Define a nota de corte para o status "Potencial".

    # --- Bloco 2: Motor de Regras (Heurísticas de Veto e Destaque) ---
    # O motor de regras verifica primeiro as condições mais críticas (exceções).
    
    # Regra de Veto 1: Verifica se o fit cultural é um fator eliminatório.
    if score_cultural < VETO_CULTURAL_THRESHOLD: # Se o score cultural for menor que o limite de veto...
        return {                                 # ...retorna imediatamente o resultado, ignorando o resto da lógica.
            'overall_match': score_cultural,     # O match geral pode refletir o score que causou o veto.
            'status': 'Incompatível Culturalmente' # O status indica claramente o motivo da reprovação.
        }                                        # A função para aqui.

    # Regra de Veto 2: Verifica se o fit técnico é um fator eliminatório.
    if score_tecnico < VETO_TECNICO_THRESHOLD:   # Se o score técnico for menor que o limite de veto...
        return {                                 # ...retorna imediatamente o resultado.
            'overall_match': score_tecnico,      # O match geral reflete o score técnico baixo.
            'status': 'Não Qualificado Tecnicamente' # O status indica o motivo.
        }                                        # A função para aqui.

    # Regra de Destaque: Identifica candidatos excepcionais.
    if score_tecnico >= DESTAQUE_TECNICO_THRESHOLD and score_cultural >= DESTAQUE_CULTURAL_THRESHOLD: # Se ambos os scores (técnico e cultural) estiverem acima dos limites de destaque...
        return {                                 # ...retorna um status especial para priorização.
            'overall_match': (score_tecnico + score_cultural) / 2, # Pode-se usar a média dos scores de destaque ou um valor fixo como 0.95.
            'status': 'Alto Potencial+'         # O status especial "+" sinaliza um candidato excepcional.
        }                                        # A função para aqui.

    # --- Bloco 3: Lógica Padrão (Média Ponderada) ---
    # Este bloco só é executado se nenhuma das regras de exceção (veto/destaque) for acionada.
    
    # Calcula o score final usando a média ponderada com os pesos definidos no Bloco 1.
    overall_match = (score_tecnico * PESO_TECNICO) + \
                    (score_cultural * PESO_CULTURAL) + \
                    (score_engajamento * PESO_ENGAGAMENTO)
    
    # Arredonda o resultado para 4 casas decimais para melhor legibilidade.
    overall_match = round(overall_match, 4)

    # Define o status final com base no score calculado, usando os buckets definidos no Bloco 1.
    status_final = ''                                     # Inicializa a variável que guardará o status final.
    if overall_match >= BUCKET_ALTO_POTENCIAL:            # Se o score final for maior ou igual ao limite de "Alto Potencial"...
        status_final = 'Alto Potencial'                   # ...define o status correspondente.
    elif overall_match >= BUCKET_POTENCIAL:               # Senão, se o score final for maior ou igual ao limite de "Potencial"...
        status_final = 'Potencial'                        # ...define o status correspondente.
    else:                                                 # Caso contrário (se for menor que todos os limites anteriores)...
        status_final = 'Baixo Potencial'                  # ...define o status como "Baixo Potencial".
        
    # Retorna o dicionário final com o score calculado e o status definido.
    return {
        'overall_match': overall_match,
        'status': status_final
    }

# --- Bloco 4: Exemplos de Uso ---
# Esta parte do código só roda quando o script é executado diretamente.
# Serve para testar e demonstrar o funcionamento da função com diferentes cenários.
if __name__ == "__main__":
    print("--- Testando o Modelo Agregador ---")

    candidato_veto_cultural = modelo_agregador_v1(score_tecnico=1.0, score_cultural=0.0, score_engajamento=0.7)
    print(f"\n1. Candidato Veto Cultural: {candidato_veto_cultural}")
