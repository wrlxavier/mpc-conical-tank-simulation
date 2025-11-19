"""
parametros_sistema.py

Centraliza todos os parâmetros físicos, operacionais e de controle do sistema
de 5 tanques tronco-cônicos para simulação de controle MPC.

Baseado na Entrega 3 - Proposta da Estrutura de Controle
"""

import numpy as np

# ==============================================================================
# PARÂMETROS DE SIMULAÇÃO
# ==============================================================================

# Tempo total de simulação (segundos)
TEMPO_TOTAL = 3000.0  # ~50 minutos

# Passo de integração para o modelo não-linear (segundos)
DT_INTEGRACAO = 0.5  # Refinado para capturar dinâmica rápida

# Período de amostragem do controlador MPC (segundos)
TS_CONTROLADOR = 5.0  # Adequado para tau_h = 961s e tau_C = 370s


# ==============================================================================
# PARÂMETROS GEOMÉTRICOS DOS TANQUES
# ==============================================================================

# Tanques de processo C, D, E (tronco-cônicos, todos idênticos)
TANQUES_PROCESSO = {
    "C": {
        "nome": "Tanque C",
        "raio_inferior": 0.75,  # m
        "raio_superior": 1.25,  # m
        "altura_maxima": 3.00,  # m
        "tipo": "tronco_conico",
    },
    "D": {
        "nome": "Tanque D",
        "raio_inferior": 0.75,  # m
        "raio_superior": 1.25,  # m
        "altura_maxima": 3.00,  # m
        "tipo": "tronco_conico",
    },
    "E": {
        "nome": "Tanque E",
        "raio_inferior": 0.75,  # m
        "raio_superior": 1.25,  # m
        "altura_maxima": 3.00,  # m
        "tipo": "tronco_conico",
    },
}

# Reservatórios de utilidades A, B (cilíndricos)
TANQUES_UTILIDADES = {
    "A": {
        "nome": "Tanque A (Água)",
        "raio": 1.75,  # m
        "altura_maxima": 3.00,  # m
        "tipo": "cilindrico",
    },
    "B": {
        "nome": "Tanque B (Salmoura)",
        "raio": 1.75,  # m
        "altura_maxima": 3.00,  # m
        "tipo": "cilindrico",
    },
}


# ==============================================================================
# COEFICIENTES DE DESCARGA E GANHOS DE ATUADORES
# ==============================================================================

# Coeficientes de descarga das válvulas (lei de Torricelli: Q = kv * u * sqrt(h))
# Para tanques de processo C, D, E
KV_VALVULAS = {
    "C": 0.016,  # m^(5/2)/s
    "D": 0.016,  # m^(5/2)/s
    "E": 0.016,  # m^(5/2)/s
}

# Ganhos das bombas de água (Q = kp_A * u)
KP_BOMBAS_AGUA = {
    "C": 0.008,  # m³/s (vazo máxima = 0.008 m³/s quando u=1)
    "D": 0.008,  # m³/s
    "E": 0.008,  # m³/s
}

# Ganhos das bombas de salmoura (Q = kp_B * u)
KP_BOMBAS_SALMOURA = {"C": 0.008, "D": 0.008, "E": 0.008}  # m³/s  # m³/s  # m³/s

# Ganhos das válvulas de suprimento dos reservatórios
KV_SUPRIMENTO_A = 0.048  # m³/s (vazo máxima quando uA=1)
KV_SUPRIMENTO_B = 0.048  # m³/s (vazo máxima quando uB=1)


# ==============================================================================
# PONTO DE OPERAÇÃO NOMINAL
# ==============================================================================

# Estados de equilíbrio
PONTO_OPERACAO = {
    # Reservatórios
    "hA_eq": 1.50,  # m - nível tanque A
    "hB_eq": 1.50,  # m - nível tanque B
    # Tanques de processo (todos idênticos no ponto de operação)
    "h_eq": 1.50,  # m - nível dos tanques C, D, E
    "C_eq": 180.0,  # kg/m³ - concentração dos tanques C, D, E
    # Controles de equilíbrio
    "uA_eq": 0.306,  # abertura válvula suprimento A
    "uB_eq": 0.306,  # abertura válvula suprimento B
    # Para cada tanque de processo (C, D, E)
    "u1_eq": 0.6125,  # comando bomba água
    "u2_eq": 0.6125,  # comando bomba salmoura
    "u3_eq": 0.50,  # abertura válvula descarga (50%)
    # Vazões de equilíbrio
    "Q_agua_eq": 0.0049,  # m³/s
    "Q_salmoura_eq": 0.0049,  # m³/s
}

# Concentração da salmoura (constante)
CB = 360.0  # kg/m³


# ==============================================================================
# PARÂMETROS DO MODELO LINEARIZADO
# ==============================================================================

# Autovalores do sistema linearizado (elementos da diagonal de A)
AUTOVALORES = {
    "lambda_h": -0.00104,  # s^-1 (dinâmica de nível)
    "lambda_C": -0.00270,  # s^-1 (dinâmica de concentração)
}

# Constantes de tempo
CONSTANTES_TEMPO = {"tau_h": 961.0, "tau_C": 370.0}  # s (~16 minutos)  # s (~6 minutos)

# Matriz A (8x8) - diagonal por blocos, estados: [hA, hB, hC, CC, hD, CD, hE, CE]
# Apenas elementos diagonais não-nulos
A_CONTINUA = np.diag(
    [
        0.0,  # hA (reservatório A)
        0.0,  # hB (reservatório B)
        AUTOVALORES["lambda_h"],  # hC
        AUTOVALORES["lambda_C"],  # CC
        AUTOVALORES["lambda_h"],  # hD
        AUTOVALORES["lambda_C"],  # CD
        AUTOVALORES["lambda_h"],  # hE
        AUTOVALORES["lambda_C"],  # CE
    ]
)

# Matriz B (8x11) - ganha de cada entrada sobre cada estado
# Entradas: [uA, uB, uC1, uC2, uC3, uD1, uD2, uD3, uE1, uE2, uE3]
B_CONTINUA = np.array(
    [
        [
            0.00499,
            0.0,
            -0.000831,
            0.0,
            0.0,
            -0.000831,
            0.0,
            0.0,
            -0.000831,
            0.0,
            0.0,
        ],  # hA
        [
            0.0,
            0.00499,
            0.0,
            -0.000831,
            0.0,
            0.0,
            -0.000831,
            0.0,
            0.0,
            -0.000831,
            0.0,
        ],  # hB
        [0.0, 0.0, 0.00255, 0.00255, -0.00624, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # hC
        [0.0, 0.0, -0.396, 0.396, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # CC
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.00255, 0.00255, -0.00624, 0.0, 0.0, 0.0],  # hD
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.396, 0.396, 0.0, 0.0, 0.0, 0.0],  # CD
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00255, 0.00255, -0.00624],  # hE
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.396, 0.396, 0.0],  # CE
    ]
)

# Matriz C (8x8) - identidade (todos os estados são medidos)
C_CONTINUA = np.eye(8)

# Matriz D (8x11) - nula (sem transmissão direta)
D_CONTINUA = np.zeros((8, 11))


# ==============================================================================
# LIMITES FÍSICOS DOS ATUADORES E VARIÁVEIS
# ==============================================================================

# Limites de nível
LIMITES_NIVEL = {
    "h_min": 0.2,  # m - nível mínimo seguro
    "h_max": 2.8,  # m - nível máximo seguro (margem antes de 3.0m)
}

# Limites de concentração (informativo)
LIMITES_CONCENTRACAO = {
    "C_min": 0.0,  # kg/m³
    "C_max": 360.0,  # kg/m³ (concentração da salmoura)
}

# Limites dos atuadores (normalizados 0-1 ou vazão máxima)
LIMITES_ATUADORES = {
    # Bombas de água (comando 0-1)
    "u1_min": 0.0,
    "u1_max": 1.0,
    # Bombas de salmoura (comando 0-1)
    "u2_min": 0.0,
    "u2_max": 1.0,
    # Válvulas de descarga (abertura 0-1)
    "u3_min": 0.0,
    "u3_max": 1.0,
    # Válvulas de suprimento reservatórios
    "uA_min": 0.0,
    "uA_max": 1.0,
    "uB_min": 0.0,
    "uB_max": 1.0,
}

# Limites de taxa de variação (slew rate) para suavidade
LIMITES_VARIACAO = {
    "delta_u1_max": 0.2,  # variação máxima por amostra (bomba água)
    "delta_u2_max": 0.2,  # variação máxima por amostra (bomba salmoura)
    "delta_u3_max": 0.15,  # variação máxima por amostra (válvula)
}


# ==============================================================================
# PARÂMETROS DO CONTROLADOR MPC
# ==============================================================================

# Horizontes de predição e controle
MPC_HORIZONTES = {
    "Np": 50,  # Horizonte de predição (200s = 40 * 5s)
    "Nc": 20,  # Horizonte de controle (100s = 20 * 5s)
}

# Pesos da função custo (matrizes diagonais Q, R, I)
# Q penaliza desvio das saídas em relação à referência
# R penaliza esforço de controle
# I penaliza erro acumulado (termo integral para offset-free)

MPC_PESOS = {
    # Para cada tanque de processo (C, D, E)
    "C": {
        "Q": np.diag([250.0, 180.0]),
        "R": np.diag([300.0, 300.0, 600.0]),
        "I": np.diag([60.0, 40.0]),
    },
    "D": {
        "Q": np.diag([250.0, 180.0]),
        "R": np.diag([300.0, 300.0, 600.0]),
        "I": np.diag([60.0, 40.0]),
    },
    "E": {
        "Q": np.diag([250.0, 180.0]),
        "R": np.diag([300.0, 300.0, 600.0]),
        "I": np.diag([60.0, 40.0]),
    },
}

# Pergunta: como fazer o mpc ser mais potente no controle do nivel e concentracao?
# Resposta: aumentar os pesos Q e I, principalmente em C (mais crítico)

LIMITE_OVERSHOOT = 0.08  # valor final permitido no overshoot
LIMITE_UNDERSHOOT = 0.08  # valor final permitido no undershoot

# Nota: Peso maior em R[2] (u3) para suavizar ação da válvula e evitar chattering
# Pesos Q e I podem ser ajustados durante sintonia para atender requisitos R1-R4


# ==============================================================================
# REQUISITOS DE DESEMPENHO (R1-R4)
# ==============================================================================

REQUISITOS = {
    "R1": {
        "nome": "Velocidade de Resposta",
        "criterio": "Tempo de acomodação (5%) < 288s (10x mais rápido que malha aberta)",
        "t_settling_max": 288.0,  # segundos
    },
    "R2": {
        "nome": "Erro em Regime Permanente",
        "criterio": "Erro em regime <= 5% do valor final",
        "erro_max_percentual": 5.0,  # %
    },
    "R3": {
        "nome": "Sobressinal (Overshoot)",
        "criterio": "Overshoot < 10% em nível e concentração",
        "overshoot_max_percentual": 10.0,  # %
    },
    "R4": {
        "nome": "Respeito às Restrições Físicas",
        "criterio": "Nunca violar limites de h, C e atuadores",
    },
}


# ==============================================================================
# CONDIÇÕES INICIAIS PADRÃO
# ==============================================================================

CONDICOES_INICIAIS = {
    # Reservatórios (em equilíbrio)
    "hA_0": PONTO_OPERACAO["hA_eq"],
    "hB_0": PONTO_OPERACAO["hB_eq"],
    # Tanques de processo (em equilíbrio)
    "hC_0": PONTO_OPERACAO["h_eq"],
    "CC_0": PONTO_OPERACAO["C_eq"],
    "hD_0": PONTO_OPERACAO["h_eq"],
    "CD_0": PONTO_OPERACAO["C_eq"],
    "hE_0": PONTO_OPERACAO["h_eq"],
    "CE_0": PONTO_OPERACAO["C_eq"],
    # Controles iniciais
    "uA_0": PONTO_OPERACAO["uA_eq"],
    "uB_0": PONTO_OPERACAO["uB_eq"],
    "uC1_0": PONTO_OPERACAO["u1_eq"],
    "uC2_0": PONTO_OPERACAO["u2_eq"],
    "uC3_0": PONTO_OPERACAO["u3_eq"],
    "uD1_0": PONTO_OPERACAO["u1_eq"],
    "uD2_0": PONTO_OPERACAO["u2_eq"],
    "uD3_0": PONTO_OPERACAO["u3_eq"],
    "uE1_0": PONTO_OPERACAO["u1_eq"],
    "uE2_0": PONTO_OPERACAO["u2_eq"],
    "uE3_0": PONTO_OPERACAO["u3_eq"],
}


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================


def calcular_area_tronco_conico(h, r_inferior, r_superior, h_maxima):
    """
    Calcula a área da seção transversal de um tanque tronco-cônico
    em função da altura do líquido.

    Args:
        h: altura do líquido (m)
        r_inferior: raio da base inferior (m)
        r_superior: raio da base superior (m)
        h_maxima: altura máxima do tanque (m)

    Returns:
        Área da seção transversal (m²)
    """
    dr_dh = (r_superior - r_inferior) / h_maxima
    r_h = r_inferior + dr_dh * h
    return np.pi * r_h**2


def calcular_volume_tronco_conico(h, r_inferior, r_superior, h_maxima):
    """
    Calcula o volume de líquido em um tanque tronco-cônico.

    Args:
        h: altura do líquido (m)
        r_inferior: raio da base inferior (m)
        r_superior: raio da base superior (m)
        h_maxima: altura máxima do tanque (m)

    Returns:
        Volume (m³)
    """
    dr_dh = (r_superior - r_inferior) / h_maxima
    r_h = r_inferior + dr_dh * h

    # V = (pi*h/3) * (R0² + R0*Rh + Rh²)
    return (np.pi * h / 3.0) * (r_inferior**2 + r_inferior * r_h + r_h**2)


def validar_limites(valor, minimo, maximo):
    """
    Garante que um valor esteja dentro dos limites especificados.

    Args:
        valor: valor a ser limitado
        minimo: limite inferior
        maximo: limite superior

    Returns:
        Valor saturado dentro dos limites
    """
    return np.clip(valor, minimo, maximo)


# ==============================================================================
# INFORMAÇÕES ADICIONAIS
# ==============================================================================

INFO_SISTEMA = {
    "autor": "Sistema de Controle MPC para Tanques Tronco-Cônicos",
    "disciplina": "Técnicas de Controle de Processos Industriais",
    "versao": "1.0",
    "data": "Novembro 2025",
    "descricao": "Simulação programada de 5 tanques (2 cilíndricos + 3 tronco-cônicos) "
    "com controle MPC multivarável de nível e concentração.",
}

if __name__ == "__main__":
    # Testes básicos
    print("=" * 70)
    print(INFO_SISTEMA["descricao"])
    print("=" * 70)
    print(f"\nTempo total de simulação: {TEMPO_TOTAL}s")
    print(f"Período de amostragem MPC: {TS_CONTROLADOR}s")
    print(f"Passo de integração: {DT_INTEGRACAO}s")
    print(f"\nConstantes de tempo:")
    print(
        f"  - Nível (τ_h): {CONSTANTES_TEMPO['tau_h']}s (~{CONSTANTES_TEMPO['tau_h']/60:.1f} min)"
    )
    print(
        f"  - Concentração (τ_C): {CONSTANTES_TEMPO['tau_C']}s (~{CONSTANTES_TEMPO['tau_C']/60:.1f} min)"
    )
    print(f"\nPonto de operação nominal:")
    print(f"  - Nível: {PONTO_OPERACAO['h_eq']}m")
    print(f"  - Concentração: {PONTO_OPERACAO['C_eq']}kg/m³")
    print(f"\nHorizontes MPC:")
    print(
        f"  - Predição (Np): {MPC_HORIZONTES['Np']} amostras ({MPC_HORIZONTES['Np']*TS_CONTROLADOR}s)"
    )
    print(
        f"  - Controle (Nc): {MPC_HORIZONTES['Nc']} amostras ({MPC_HORIZONTES['Nc']*TS_CONTROLADOR}s)"
    )
    print("\n" + "=" * 70)
