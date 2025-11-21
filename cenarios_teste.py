"""
cenarios_teste.py

Define cenários de teste programados para validação do sistema de controle MPC.

Cada cenário especifica:
- Perfis de referência (setpoints) para níveis e concentrações ao longo do tempo
- Instantes de aplicação de degraus e mudanças de referência
- Perturbações planejadas no sistema

Os cenários são projetados para verificar os requisitos R1-R4:
- R1: Velocidade de resposta (t_settling < 288s)
- R2: Erro em regime permanente (< 5%)
- R3: Overshoot (< 10%)
- R4: Respeito às restrições físicas

Baseado na Entrega 3 - Proposta da Estrutura de Controle
"""

import numpy as np
from typing import Dict, List, Tuple
import parametros_sistema as params


# ==============================================================================
# FUNÇÕES AUXILIARES PARA GERAÇÃO DE PERFIS
# ==============================================================================


def gerar_degrau(
    tempo_total: float,
    dt: float,
    t_degrau: float,
    valor_inicial: float,
    valor_final: float,
) -> np.ndarray:
    """
    Gera um perfil de referência com degrau único.

    Args:
        tempo_total: duração total da simulação (s)
        dt: passo de tempo (s)
        t_degrau: instante do degrau (s)
        valor_inicial: valor antes do degrau
        valor_final: valor após o degrau

    Returns:
        Array com perfil de referência ao longo do tempo
    """
    n_pontos = int(tempo_total / dt) + 1
    perfil = np.ones(n_pontos) * valor_inicial
    idx_degrau = int(t_degrau / dt)
    perfil[idx_degrau:] = valor_final
    return perfil


def gerar_rampa(
    tempo_total: float,
    dt: float,
    t_inicio: float,
    t_fim: float,
    valor_inicial: float,
    valor_final: float,
) -> np.ndarray:
    """
    Gera um perfil de referência com rampa suave.

    Args:
        tempo_total: duração total da simulação (s)
        dt: passo de tempo (s)
        t_inicio: instante de início da rampa (s)
        t_fim: instante de fim da rampa (s)
        valor_inicial: valor antes da rampa
        valor_final: valor após a rampa

    Returns:
        Array com perfil de referência ao longo do tempo
    """
    n_pontos = int(tempo_total / dt) + 1
    perfil = np.ones(n_pontos) * valor_inicial

    idx_inicio = int(t_inicio / dt)
    idx_fim = int(t_fim / dt)

    if idx_fim > idx_inicio:
        n_rampa = idx_fim - idx_inicio
        perfil[idx_inicio:idx_fim] = np.linspace(valor_inicial, valor_final, n_rampa)
        perfil[idx_fim:] = valor_final

    return perfil


def gerar_multiplos_degraus(
    tempo_total: float, dt: float, instantes: List[float], valores: List[float]
) -> np.ndarray:
    """
    Gera um perfil com múltiplos degraus em diferentes instantes.

    Args:
        tempo_total: duração total da simulação (s)
        dt: passo de tempo (s)
        instantes: lista de instantes de mudança (s)
        valores: lista de valores correspondentes (deve ter len(instantes)+1)

    Returns:
        Array com perfil de referência ao longo do tempo
    """
    n_pontos = int(tempo_total / dt) + 1
    perfil = np.ones(n_pontos) * valores[0]

    for i, t_mudanca in enumerate(instantes):
        idx = int(t_mudanca / dt)
        if idx < n_pontos:
            perfil[idx:] = valores[i + 1]

    return perfil


# ==============================================================================
# DEFINIÇÃO DOS CENÁRIOS DE TESTE
# ==============================================================================


def cenario_1_degrau_nivel_unico(
    tempo_total: float = 3000.0, dt: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Cenário 1: Degrau único no setpoint de nível do Tanque C

    Objetivo: Verificar R1 (velocidade), R2 (erro permanente) e R3 (overshoot)
    para variação isolada de nível.

    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência para todos os tanques
    """
    # Ponto de operação nominal
    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]

    # Degrau de +0.2m no nível do tanque C em t=30s
    hC_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=30.0,
        valor_inicial=h_eq,
        valor_final=min(h_eq + 0.2, 1.7),
    )

    # Mantém concentração e demais tanques em equilíbrio
    CC_ref = np.ones_like(hC_ref) * C_eq
    hD_ref = np.ones_like(hC_ref) * h_eq
    CD_ref = np.ones_like(hC_ref) * C_eq
    hE_ref = np.ones_like(hC_ref) * h_eq
    CE_ref = np.ones_like(hC_ref) * C_eq

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 1: Degrau Único em Nível (Tanque C)",
        "descricao": "Degrau de +0.2m em hC em t=30s. Verifica R1, R2, R3 para nível.",
    }


def cenario_2_degrau_concentracao_unico(
    tempo_total: float = 3000.0, dt: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Cenário 2: Degrau único no setpoint de concentração do Tanque D

    Objetivo: Verificar R1, R2, R3 para variação isolada de concentração,
    aproveitando a dinâmica mais rápida (τ_C = 370s vs τ_h = 961s).

    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência
    """
    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]

    # Degrau de +20 kg/m³ na concentração do tanque D em t=30s
    CD_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=30.0,
        valor_inicial=C_eq,
        valor_final=min(C_eq + 20.0, 200.0),
    )

    # Mantém demais variáveis em equilíbrio
    hC_ref = np.ones_like(CD_ref) * h_eq
    CC_ref = np.ones_like(CD_ref) * C_eq
    hD_ref = np.ones_like(CD_ref) * h_eq
    hE_ref = np.ones_like(CD_ref) * h_eq
    CE_ref = np.ones_like(CD_ref) * C_eq

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 2: Degrau Único em Concentração (Tanque D)",
        "descricao": "Degrau de +20kg/m³ em CD em t=30s. Verifica dinâmica de concentração.",
    }


def cenario_3_mudancas_combinadas(
    tempo_total: float = 3000.0, dt: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Cenário 3: Mudanças combinadas em múltiplos tanques

    Objetivo: Verificar desempenho MIMO com mudanças simultâneas em nível
    e concentração de diferentes tanques.

    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência
    """
    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]

    # Tanque C: degrau em nível em t=60s
    hC_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=60.0,
        valor_inicial=h_eq,
        valor_final=min(h_eq + 0.2, 1.7),
    )

    # Tanque C: degrau em concentração em t=360s
    CC_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=360.0,
        valor_inicial=C_eq,
        valor_final=min(C_eq + 20.0, 200.0),
    )

    # Tanque D: ambos simultaneamente em t=720s
    hD_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=720.0,
        valor_inicial=h_eq,
        valor_final=max(h_eq - 0.2, 1.3),
    )
    CD_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=720.0,
        valor_inicial=C_eq,
        valor_final=max(C_eq - 20.0, 150.0),
    )

    # Tanque E: rampa suave em t=1440s até t=1500s
    hE_ref = gerar_rampa(
        tempo_total,
        dt,
        t_inicio=1440.0,
        t_fim=1500.0,
        valor_inicial=h_eq,
        valor_final=h_eq + 0.20,
    )
    CE_ref = np.ones(int(tempo_total / dt) + 1) * C_eq

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 3: Mudanças Combinadas MIMO",
        "descricao": "Múltiplos degraus em tanques diferentes. Testa coordenação MIMO.",
    }


def cenario_3_degrau_nivel_concentracao(
    tempo_total: float = 3000.0, dt: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Cenário 3: Mudança de nível e concentração no Tanque C

    Objetivo: Verificar a resposta do sistema a mudanças simultâneas de nível
    e concentração no Tanque C.

    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência
    """
    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]

    # Tanque C: degrau em nível em t=30s
    hC_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=30.0,
        valor_inicial=h_eq,
        valor_final=min(h_eq + 0.2, 1.7),
    )

    # Tanque C: degrau negativo em concentração em t=30s
    CC_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=30.0,
        valor_inicial=C_eq,
        valor_final=min(C_eq - 20.0, 150.0),
    )

    # Mantém demais variáveis em equilíbrio
    hD_ref = np.ones(int(tempo_total / dt) + 1) * h_eq
    CD_ref = np.ones(int(tempo_total / dt) + 1) * C_eq
    hE_ref = np.ones(int(tempo_total / dt) + 1) * h_eq
    CE_ref = np.ones(int(tempo_total / dt) + 1) * C_eq

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 3: Mudança de Nível e Concentração no Tanque C",
        "descricao": "Múltiplos degraus em tanques diferentes. Testa coordenação MIMO.",
    }


def cenario_4_teste_limites(
    tempo_total: float = 3000.0, dt: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Cenário 4: Teste de respeito às restrições físicas (R4)

    Objetivo: Verificar que o MPC respeita os limites de nível e concentração
    mesmo com referências agressivas próximas aos limites operacionais.

    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência
    """
    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]
    h_max_seguro = params.LIMITES_NIVEL["h_max"]
    h_min_seguro = params.LIMITES_NIVEL["h_min"]

    # Tanque C: referência próxima ao limite máximo
    hC_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=300.0,
        valor_inicial=h_eq,
        valor_final=h_max_seguro - 0.4,
    )
    CC_ref = np.ones(int(tempo_total / dt) + 1) * C_eq

    # Tanque D: referência próxima ao limite mínimo
    hD_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=300.0,
        valor_inicial=h_eq,
        valor_final=h_min_seguro + 0.4,
    )
    CD_ref = np.ones(int(tempo_total / dt) + 1) * C_eq

    # Tanque E: concentração próxima aos limites
    hE_ref = np.ones(int(tempo_total / dt) + 1) * h_eq
    CE_ref = gerar_multiplos_degraus(
        tempo_total, dt, instantes=[500.0, 1500.0], valores=[C_eq, 50.0, 320.0]
    )

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 4: Teste de Limites Operacionais",
        "descricao": "Referências próximas aos limites físicos. Verifica R4 (restrições).",
    }


def cenario_4_nivel_concentracao_multiplos_tanques(
    tempo_total: float = 3000.0, dt: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Cenário 4: Mudanças simultâneas de nível e concentração em múltiplos tanques

    Objetivo: Verificar a resposta do sistema a mudanças simultâneas de nível
    e concentração em diferentes tanques.

    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência
    """
    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]

    # Tanque C: degrau em nível em t=30s
    hC_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=30.0,
        valor_inicial=h_eq,
        valor_final=min(h_eq + 0.2, 1.7),
    )

    # Tanque C: degrau em concentração em t=30s
    CC_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=30.0,
        valor_inicial=C_eq,
        valor_final=min(C_eq + 20.0, 200.0),
    )

    # Tanque D: ambos simultaneamente em t=30s
    hD_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=30.0,
        valor_inicial=h_eq,
        valor_final=max(h_eq - 0.2, 1.3),
    )
    CD_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=30.0,
        valor_inicial=C_eq,
        valor_final=max(C_eq - 20.0, 150.0),
    )

    # Tanque E: degrau em nível em t=30s
    hE_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=30.0,
        valor_inicial=h_eq,
        valor_final=min(h_eq + 0.3, 1.7),
    )

    # Tanque E: degrau em concentração em t=30s
    CE_ref = gerar_degrau(
        tempo_total,
        dt,
        t_degrau=30.0,
        valor_inicial=C_eq,
        valor_final=max(C_eq - 20.0, 150.0),
    )

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 4: Mudanças Simultâneas de Nível e Concentração",
        "descricao": "Verifica a resposta do sistema a mudanças simultâneas em múltiplos tanques.",
    }


def cenario_5_rejeicao_perturbacao(
    tempo_total: float = 3000.0, dt: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Cenário 5: Teste de rastreamento com múltiplas mudanças de referência

    Objetivo: Simular perfil de produção com múltiplas mudanças de setpoint,
    verificando capacidade de rastreamento e erro em regime (R2).

    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência
    """
    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]

    # Sequência de setpoints para simular receita de produção (ajustados para limites)
    instantes_h = [30.0, 130.0, 230.0, 330.0]
    valores_h = [h_eq, 1.7, 1.3, 1.7, 1.5]

    instantes_C = [40.0, 140.0, 240.0, 340.0]
    valores_C = [C_eq, 200.0, 150.0, 200.0, 180.0]

    # Tanque C com perfil variável
    hC_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_h, valores_h)
    CC_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_C, valores_C)

    # Tanques D e E seguem perfil mais simples
    hD_ref = gerar_degrau(
        tempo_total, dt, t_degrau=30.0, valor_inicial=h_eq, valor_final=1.7
    )
    CD_ref = gerar_degrau(
        tempo_total, dt, t_degrau=60.0, valor_inicial=C_eq, valor_final=195.0
    )

    hE_ref = np.ones(int(tempo_total / dt) + 1) * h_eq
    CE_ref = np.ones(int(tempo_total / dt) + 1) * C_eq

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 5: Perfil de Produção Variável",
        "descricao": "Múltiplas mudanças sequenciais. Testa rastreamento e offset-free.",
    }


def cenario_6_validacao_completa(
    tempo_total: float = 3000.0, dt: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Cenário 6: Validação completa de todos os requisitos

    Objetivo: Cenário integrado que exercita todos os aspectos do sistema
    para demonstração final de desempenho.

    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência
    """
    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]

    # Perfis coordenados para todos os tanques
    # Tanque C: variações moderadas em ambas as variáveis
    hC_ref = gerar_multiplos_degraus(
        tempo_total, dt, instantes=[400.0, 1200.0], valores=[h_eq, 1.65, 1.35]
    )
    CC_ref = gerar_multiplos_degraus(
        tempo_total, dt, instantes=[700.0, 1800.0], valores=[C_eq, 200.0, 160.0]
    )

    # Tanque D: foco em concentração
    hD_ref = np.ones(int(tempo_total / dt) + 1) * h_eq
    CD_ref = gerar_rampa(
        tempo_total,
        dt,
        t_inicio=500.0,
        t_fim=800.0,
        valor_inicial=C_eq,
        valor_final=220.0,
    )

    # Tanque E: foco em nível
    hE_ref = gerar_multiplos_degraus(
        tempo_total,
        dt,
        instantes=[300.0, 1000.0, 2000.0],
        valores=[h_eq, 1.75, 1.25, 1.60],
    )
    CE_ref = np.ones(int(tempo_total / dt) + 1) * C_eq

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 6: Validação Completa R1-R4",
        "descricao": "Cenário integrado exercitando todos os requisitos de desempenho.",
    }


# ==============================================================================
# FUNÇÃO PRINCIPAL: SELEÇÃO DE CENÁRIOS
# ==============================================================================


def obter_cenario(
    numero_cenario: int, tempo_total: float = 3000.0, dt: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Retorna o cenário de teste especificado.

    Args:
        numero_cenario: número do cenário (1-6)
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência e metadados do cenário
    """
    cenarios = {
        1: cenario_1_degrau_nivel_unico,
        2: cenario_2_degrau_concentracao_unico,
        3: cenario_3_degrau_nivel_concentracao,
        4: cenario_4_nivel_concentracao_multiplos_tanques,
        5: cenario_5_rejeicao_perturbacao,
        # 3: cenario_3_mudancas_combinadas,
        # 4: cenario_4_teste_limites,
        # 5: cenario_5_rejeicao_perturbacao,
        # 6: cenario_6_validacao_completa,
    }

    if numero_cenario not in cenarios:
        raise ValueError(f"Cenário {numero_cenario} não existe. Escolha entre 1 e 6.")

    return cenarios[numero_cenario](tempo_total, dt)


def listar_cenarios():
    """Exibe a lista de cenários disponíveis com descrições fiéis aos sinais."""
    print("=" * 70)
    print("CENÁRIOS DE TESTE DISPONÍVEIS")
    print("=" * 70)
    print("\n1. Degrau Único em Nível (Tanque C)")
    print("   → Degrau de +0.2m em hC em t=30s .")
    print("      CC, hD, CD, hE, CE permanecem constantes.\n")

    print("2. Degrau Único em Concentração (Tanque D)")
    print("   → Degrau de +20kg/m³ em CD em t=30s .")
    print("      hC, CC, hD, hE, CE permanecem constantes.\n")

    print("3. Mudança de Nível e Concentração no Tanque C")
    print("   → hC: degrau de +0.2m em t=30s; CC: degrau de -20kg/m³ em t=30s.")
    print("      hD, CD, hE, CE permanecem constantes.\n")

    print("4. Mudança Simultânea de Nível e Concentração em Múltiplos Tanques")
    print("   → hC: degrau de +0.2m em t=30s; CC: degrau de +20kg/m³ em t=30s.")
    print("      hD: degrau de -0.2m em t=30s; CD: degrau de -20kg/m³ em t=30s.")
    print("      hE: degrau de +0.3m em t=30s; CE: degrau de -20kg/m³ em t=30s.\n")

    print("5. Perfil de Produção Variável")
    print("   → hC: múltiplos degraus (1.7, 1.3, 1.7, 1.5m) em t=30, 130, 230, 330s.")
    print(
        "      CC: múltiplos degraus (200, 150, 200, 180kg/m³) em t=40, 140, 240, 340s."
    )
    print("      hD: degrau para 1.7m em t=30s; CD: degrau para 195kg/m³ em t=60s.")
    print("      hE, CE permanecem constantes.\n")

    # print("3. Mudanças Combinadas MIMO")
    # print("   → hC: degrau de +0.2m em t=200s; CC: degrau de +20kg/m³ em t=800s.")
    # print("      hD: degrau de -0.2m em t=1400s; CD: degrau de -20kg/m³ em t=1400s.")
    # print("      hE: rampa de +0.3m de t=2000s a t=2200s; CE constante.\n")

    # print("4. Teste de Limites Operacionais")
    # print("   → hC: degrau para próximo do limite máximo (h_max - 0.4) em t=300s.")
    # print("      hD: degrau para próximo do limite mínimo (h_min + 0.4) em t=300s.")
    # print("      CE: degraus para 50kg/m³ em t=500s e 320kg/m³ em t=1500s.")
    # print("      CC, CD, hE constantes.\n")

    # print("5. Perfil de Produção Variável")
    # print(
    #     "   → hC: múltiplos degraus (1.7, 1.3, 1.7, 1.5m) em t=300, 900, 1500, 2100s."
    # )
    # print(
    #     "      CC: múltiplos degraus (200, 150, 200, 180kg/m³) em t=400, 1000, 1600, 2200s."
    # )
    # print("      hD: degrau para 1.7m em t=600s; CD: degrau para 195kg/m³ em t=1200s.")
    # print("      hE, CE constantes.\n")

    # print("6. Validação Completa R1-R4")
    # print("   → hC: degraus para 1.65m em t=400s e 1.35m em t=1200s.")
    # print("      CC: degraus para 200kg/m³ em t=700s e 160kg/m³ em t=1800s.")
    # print("      hD constante; CD: rampa de C_eq até 220kg/m³ de t=500s a t=800s.")
    # print(
    #     "      hE: degraus para 1.75m, 1.25m, 1.60m em t=300, 1000, 2000s; CE constante.\n"
    # )

    print("=" * 70)


# ==============================================================================
# TESTES
# ==============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("TESTE DOS CENÁRIOS DE REFERÊNCIA")
    print("=" * 70)

    # Lista cenários disponíveis
    listar_cenarios()

    # Gera e plota o Cenário 3 como exemplo
    print("\nGerando Cenário 3 (Mudanças Combinadas)...")
    cenario = obter_cenario(3, tempo_total=3000.0, dt=0.5)

    tempo = np.arange(0, 3000.0 + 0.5, 0.5)

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(cenario["nome"], fontsize=14, fontweight="bold")

    tanques = ["C", "D", "E"]
    for i, tanque in enumerate(tanques):
        # Nível
        axes[i, 0].plot(tempo, cenario[f"h{tanque}_ref"], "b-", linewidth=2)
        axes[i, 0].set_ylabel(f"h{tanque} [m]", fontsize=11)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axhline(
            y=params.LIMITES_NIVEL["h_max"],
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Limite máx",
        )
        axes[i, 0].axhline(
            y=params.LIMITES_NIVEL["h_min"],
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Limite mín",
        )

        # Concentração
        axes[i, 1].plot(tempo, cenario[f"C{tanque}_ref"], "g-", linewidth=2)
        axes[i, 1].set_ylabel(f"C{tanque} [kg/m³]", fontsize=11)
        axes[i, 1].grid(True, alpha=0.3)

    axes[2, 0].set_xlabel("Tempo [s]", fontsize=11)
    axes[2, 1].set_xlabel("Tempo [s]", fontsize=11)

    plt.tight_layout()
    plt.show()

    print(f"\n{cenario['descricao']}")
    print("=" * 70)
