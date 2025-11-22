"""
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
"""

import numpy as np
from typing import Dict, List, Tuple
import parametros_sistema as params


# FUNÇÕES AUXILIARES PARA GERAÇÃO DE PERFIS


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


def gerar_senoidal(
    tempo_total: float,
    dt: float,
    frequencia: float,
    amplitude: float,
    fase: float = 0.0,
) -> np.ndarray:
    """
    Gera um perfil senoidal.

    Args:
        tempo_total: duração total da simulação (s)
        dt: passo de tempo (s)
        frequencia: frequência da onda senoidal (Hz)
        amplitude: amplitude da onda senoidal
        fase: fase da onda senoidal (radianos)

    Returns:
        Array com perfil de referência ao longo do tempo
    """
    t = np.arange(0, tempo_total + dt, dt)
    return amplitude * np.sin(2 * np.pi * frequencia * t + fase)


# DEFINIÇÃO DOS CENÁRIOS DE TESTE


def cenario_1(tempo_total: float = 3000.0, dt: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Cenário 1: Degrau único no setpoint de nível do Tanque C (+0.2m), concentração do Tanque D (+20kg/m³) e nível do Tanque E (-0.2m)
    Objetivo: Verificar R1, R2, R3 para variação isolada de nível e concentração.
    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)
    Returns:
        Dicionário com perfis de referência
    """

    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]

    # Degrau em hC: +0.2m a partir de t=30s, sem retorno ao equilíbrio
    instantes_hC = [30.0]
    valores_hC = [h_eq, h_eq + 0.2]
    hC_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_hC, valores_hC)

    # Degrau em CD: +20kg/m³ a partir de t=30s, sem retorno ao equilíbrio
    instantes_CD = [30.0]
    valores_CD = [C_eq, C_eq + 20.0]
    CD_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_CD, valores_CD)

    # Degrau em hE: -0.2m a partir de t=30s, sem retorno ao equilíbrio
    instantes_hE = [30.0]
    valores_hE = [h_eq, h_eq - 0.2]
    hE_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_hE, valores_hE)

    # Demais variáveis constantes no ponto de equilíbrio
    CC_ref = np.ones_like(hC_ref) * C_eq
    hD_ref = np.ones_like(hC_ref) * h_eq
    CE_ref = np.ones_like(hC_ref) * C_eq

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 1: Degraus em hC (+0.2m), CD (+20kg/m³), hE (-0.2m)",
        "descricao": "Degraus em hC (+0.2m), CD (+20kg/m³) e hE (-0.2m) em t=30s. Demais variáveis mantidas no ponto de equilíbrio.",
    }


def cenario_2(tempo_total: float = 3000.0, dt: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Cenário 2: Senoidal no Nível do Tanque C e Concentração do Tanque D

    Objetivo: Verificar R1, R2, R3 para variação isolada de nível e concentração.

    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência
    """

    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]

    frequencia = 1 / 360.0  # Período de 6 minutos (360s)

    senoide_hC = gerar_senoidal(
        tempo_total=tempo_total,
        dt=dt,
        frequencia=frequencia,
        amplitude=0.2,
        fase=0.0,
    )

    senoide_cD = gerar_senoidal(
        tempo_total=tempo_total,
        dt=dt,
        frequencia=frequencia,
        amplitude=20.0,
        fase=0.0,
    )

    rampa_hE = gerar_rampa(
        tempo_total=tempo_total,
        dt=dt,
        valor_inicial=h_eq,
        valor_final=h_eq + 0.2,
        t_inicio=30.0,
        t_fim=400.0,
    )

    hC_ref = h_eq + senoide_hC
    CC_ref = np.ones_like(hC_ref) * C_eq
    hD_ref = np.ones_like(hC_ref) * h_eq
    CD_ref = C_eq + senoide_cD
    hE_ref = rampa_hE
    CE_ref = np.ones_like(hC_ref) * C_eq

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 2: Senoidal no Nível do Tanque C e Concentração do Tanque D",
        "descricao": "Sinal senoidal em hC e CD. Verifica resposta dinâmica.",
    }


def cenario_3(tempo_total: float = 3000.0, dt: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Cenário 3: Mudança de Nível e Concentração nos Tanques C, D e E

    Objetivo: Testar coordenação MIMO com múltiplos degraus em tanques diferentes.

    Args:
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência
    """

    h_eq = params.PONTO_OPERACAO["h_eq"]
    C_eq = params.PONTO_OPERACAO["C_eq"]

    instantes_hC = [30.0, 400.0]
    valores_hC = [h_eq, (h_eq + 0.2), h_eq]

    instantes_CC = [30.0, 400.0]
    valores_CC = [C_eq, (C_eq + 20.0), C_eq]

    hC_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_hC, valores_hC)
    CC_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_CC, valores_CC)

    instantes_hD = [30.0, 400.0]
    valores_hD = [h_eq, (h_eq - 0.2), h_eq]

    instantes_CD = [30.0, 400.0]
    valores_CD = [C_eq, (C_eq - 20.0), C_eq]

    hD_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_hD, valores_hD)
    CD_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_CD, valores_CD)

    instantes_hE = [30.0, 400.0]
    valores_hE = [h_eq, (h_eq + 0.2), h_eq]

    instantes_CE = [30.0, 400.0]
    valores_CE = [C_eq, (C_eq - 20.0), C_eq]
    hE_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_hE, valores_hE)
    CE_ref = gerar_multiplos_degraus(tempo_total, dt, instantes_CE, valores_CE)

    return {
        "hC_ref": hC_ref,
        "CC_ref": CC_ref,
        "hD_ref": hD_ref,
        "CD_ref": CD_ref,
        "hE_ref": hE_ref,
        "CE_ref": CE_ref,
        "nome": "Cenário 3: Mudança de Nível e Concentração nos Tanques C, D e E",
        "descricao": "Múltiplos degraus em tanques diferentes. Testa coordenação MIMO.",
    }


# FUNÇÃO PRINCIPAL: SELEÇÃO DE CENÁRIOS


def obter_cenario(
    numero_cenario: int, tempo_total: float = 3000.0, dt: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Retorna o cenário de teste especificado.

    Args:
        numero_cenario: número do cenário (1-3)
        tempo_total: duração da simulação (s)
        dt: passo de integração (s)

    Returns:
        Dicionário com perfis de referência e metadados do cenário
    """
    cenarios = {
        1: cenario_1,
        2: cenario_2,
        3: cenario_3,
    }

    if numero_cenario not in cenarios:
        raise ValueError(f"Cenário {numero_cenario} não existe. Escolha entre 1 e 3.")

    return cenarios[numero_cenario](tempo_total, dt)


def listar_cenarios():
    """Exibe a lista de cenários disponíveis com descrições fiéis aos sinais."""
    print("=" * 70)
    print("CENÁRIOS DE TESTE DISPONÍVEIS")
    print("=" * 70)

    print("1. Degraus em hC (+0.2m), CD (+20kg/m³) e hE (-0.2m) em t=30s")
    print("   → hC: degrau de +0.2m em t=30s (mantido); CC: constante.")
    print("      hD: constante; CD: degrau de +20kg/m³ em t=30s (mantido).")
    print("      hE: degrau de -0.2m em t=30s (mantido); CE: constante.\n")

    print("2. Senoidal no Nível do Tanque C e Concentração do Tanque D")
    print("   → hC: sinal senoidal com amplitude de 0.2m e período de 360s.")
    print("      CD: sinal senoidal com amplitude de 20kg/m³ e período de 360s.")
    print("      hE: rampa de +0.2m entre t=30s e t=400s.\n")

    print("3. Mudança de Nível e Concentração nos Tanques C, D e E")
    print("   → hC: degrau de +0.2m em t=30s; CC: degrau de +20kg/m³ em t=30s.")
    print("      hD: degrau de -0.2m em t=30s; CD: degrau de -20kg/m³ em t=30s.")
    print("      hE: degrau de +0.2m em t=30s; CE: degrau de -20kg/m³ em t=30s.\n")

    print("=" * 70)
