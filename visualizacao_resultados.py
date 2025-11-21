"""
visualizacao_resultados.py

Gera gráficos e análises visuais dos resultados da simulação MPC.

Funções principais:
- Plotar evolução temporal de estados (níveis e concentrações)
- Plotar ações de controle (bombas e válvulas)
- Calcular métricas de desempenho (tempo de acomodação, overshoot, erro permanente)
- Verificar atendimento aos requisitos R1-R4
- Gerar relatório consolidado de desempenho
- Exportar dados para CSV

Baseado na Entrega 3 - Requisitos de Desempenho
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import pandas as pd
import parametros_sistema as params
import os


# ==============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# ==============================================================================

# Estilo dos gráficos
plt.style.use("seaborn-v0_8-darkgrid")
CORES = {
    "C": "#1f77b4",  # Azul
    "D": "#ff7f0e",  # Laranja
    "E": "#2ca02c",  # Verde
    "ref": "#d62728",  # Vermelho
    "limite": "#7f7f7f",  # Cinza
}

GRAFICOS_DIR = os.path.join(os.path.dirname(__file__), "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)


# ==============================================================================
# CÁLCULO DE MÉTRICAS DE DESEMPENHO
# ==============================================================================


def calcular_tempo_acomodacao(
    tempo: np.ndarray,
    sinal: np.ndarray,
    referencia: np.ndarray,
    tolerancia: float = 0.05,
    t_inicio_degrau: float = None,
) -> float:
    """
    Calcula o tempo de acomodação (settling time) para uma banda de ±5%.

    Args:
        tempo: vetor de tempo (s)
        sinal: sinal de resposta
        referencia: sinal de referência
        tolerancia: banda de tolerância (padrão 5% = 0.05)
        t_inicio_degrau: instante do degrau (s). Se None, detecta automaticamente

    Returns:
        Tempo de acomodação em segundos (ou np.inf se não acomodou)
    """
    # Detecta instante do degrau (primeira mudança significativa na referência)
    if t_inicio_degrau is None:
        diff_ref = np.abs(np.diff(referencia))
        idx_degrau = np.where(diff_ref > 1e-6)[0]
        if len(idx_degrau) == 0:
            return 0.0  # Sem degrau
        idx_inicio = idx_degrau[0] + 1
    else:
        idx_inicio = np.argmin(np.abs(tempo - t_inicio_degrau))

    # Valor final da referência
    ref_final = referencia[-1]

    # Banda de tolerância
    banda_superior = ref_final * (1 + tolerancia)
    banda_inferior = ref_final * (1 - tolerancia)

    # Procura o último ponto que saiu da banda
    dentro_da_banda = (sinal >= banda_inferior) & (sinal <= banda_superior)

    # Identifica quando entra definitivamente na banda (não sai mais)
    for i in range(len(sinal) - 1, idx_inicio, -1):
        if not dentro_da_banda[i]:
            # Encontrou o último ponto fora da banda
            if i + 1 < len(tempo):
                return tempo[i + 1] - tempo[idx_inicio]
            else:
                return tempo[i] - tempo[idx_inicio]

    # Se nunca saiu da banda após o degrau, acomodou imediatamente
    if dentro_da_banda[idx_inicio:].all():
        return 0.0

    # Se não acomodou até o final
    return np.inf


def calcular_overshoot(
    sinal: np.ndarray, referencia: np.ndarray, t_inicio_degrau_idx: int = None
) -> float:
    """
    Calcula o overshoot percentual máximo.

    Args:
        sinal: sinal de resposta
        referencia: sinal de referência
        t_inicio_degrau_idx: índice do início do degrau

    Returns:
        Overshoot em percentual (0-100)
    """
    if t_inicio_degrau_idx is None:
        diff_ref = np.abs(np.diff(referencia))
        idx_degrau = np.where(diff_ref > 1e-6)[0]
        if len(idx_degrau) == 0:
            return 0.0
        t_inicio_degrau_idx = idx_degrau[0] + 1

    ref_inicial = referencia[t_inicio_degrau_idx - 1]
    ref_final = referencia[-1]
    variacao_ref = ref_final - ref_inicial

    if abs(variacao_ref) < 1e-6:
        return 0.0

    # Overshoot para degrau positivo
    if variacao_ref > 0:
        pico = np.max(sinal[t_inicio_degrau_idx:])
        overshoot = ((pico - ref_final) / variacao_ref) * 100.0
    # Overshoot para degrau negativo (undershoot)
    else:
        vale = np.min(sinal[t_inicio_degrau_idx:])
        overshoot = ((ref_final - vale) / abs(variacao_ref)) * 100.0

    return max(0.0, overshoot)


def calcular_erro_regime_permanente(
    sinal: np.ndarray, referencia: np.ndarray, percentual_final: float = 0.2
) -> float:
    """
    Calcula o erro em regime permanente como percentual do valor final.

    Args:
        sinal: sinal de resposta
        referencia: sinal de referência
        percentual_final: fração final da simulação a considerar (padrão 20%)

    Returns:
        Erro percentual em relação ao valor final da referência
    """
    n_final = int(len(sinal) * percentual_final)
    sinal_regime = sinal[-n_final:]
    ref_regime = referencia[-n_final:]

    ref_final = ref_regime[-1]

    if abs(ref_final) < 1e-6:
        return 0.0

    erro_medio = np.mean(np.abs(sinal_regime - ref_regime))
    erro_percentual = (erro_medio / abs(ref_final)) * 100.0

    return erro_percentual


def verificar_violacao_restricoes(hist_estados: Dict, hist_controles: Dict) -> Dict:
    """
    Verifica se houve violação de restrições físicas durante a simulação.

    Returns:
        Dicionário com violações detectadas
    """
    violacoes = {
        "niveis_min": [],
        "niveis_max": [],
        "concentracoes_min": [],
        "concentracoes_max": [],
        "controles_min": [],
        "controles_max": [],
    }

    # Verifica níveis
    for tanque in ["C", "D", "E"]:
        h = hist_estados[f"h{tanque}"]
        if np.any(h < params.LIMITES_NIVEL["h_min"]):
            violacoes["niveis_min"].append(tanque)
        if np.any(h > params.LIMITES_NIVEL["h_max"]):
            violacoes["niveis_max"].append(tanque)

    # Verifica concentrações
    for tanque in ["C", "D", "E"]:
        C = hist_estados[f"C{tanque}"]
        if np.any(C < params.LIMITES_CONCENTRACAO["C_min"]):
            violacoes["concentracoes_min"].append(tanque)
        if np.any(C > params.LIMITES_CONCENTRACAO["C_max"]):
            violacoes["concentracoes_max"].append(tanque)

    # Verifica controles
    for chave, valores in hist_controles.items():
        if chave.endswith("1") or chave.endswith("2"):  # Bombas
            if np.any(valores < 0.0) or np.any(valores > 1.0):
                violacoes["controles_min"].append(chave)
        elif chave.endswith("3"):  # Válvulas
            if np.any(valores < 0.0) or np.any(valores > 1.0):
                violacoes["controles_min"].append(chave)

    return violacoes


# ==============================================================================
# FUNÇÕES DE PLOTAGEM
# ==============================================================================


def plotar_niveis(resultados: Dict, salvar: bool = True, nome_arquivo: str = None):
    """
    Plota a evolução temporal dos níveis de todos os tanques.
    Salva a imagem em /graficos.
    """
    tempo = resultados["estados"]["tempo"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "Evolução dos Níveis dos Tanques de Processo", fontsize=16, fontweight="bold"
    )

    tanques = ["C", "D", "E"]
    for i, tanque in enumerate(tanques):
        # Estado
        axes[i].plot(
            tempo,
            resultados["estados"][f"h{tanque}"],
            color=CORES[tanque],
            linewidth=2,
            label=f"h{tanque} (medido)",
        )

        # Referência
        axes[i].plot(
            tempo,
            resultados["referencias"][f"h{tanque}_ref"],
            color=CORES["ref"],
            linestyle="--",
            linewidth=1.5,
            label="Referência",
            alpha=0.8,
        )

        # Limites
        axes[i].axhline(
            y=params.LIMITES_NIVEL["h_max"],
            color=CORES["limite"],
            linestyle=":",
            alpha=0.5,
            label="Limite máx",
        )
        axes[i].axhline(
            y=params.LIMITES_NIVEL["h_min"],
            color=CORES["limite"],
            linestyle=":",
            alpha=0.5,
            label="Limite mín",
        )

        axes[i].set_ylabel(f"h{tanque} [m]", fontsize=12, fontweight="bold")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=9)
        axes[i].set_ylim([0, 3.2])

    axes[2].set_xlabel("Tempo [s]", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if salvar:
        if nome_arquivo is None:
            nome_arquivo = os.path.join(GRAFICOS_DIR, "niveis_tanques.png")
        else:
            nome_arquivo = os.path.join(GRAFICOS_DIR, nome_arquivo)
        plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plotar_concentracoes(
    resultados: Dict, salvar: bool = True, nome_arquivo: str = None
):
    """
    Plota a evolução temporal das concentrações de todos os tanques.
    Salva a imagem em /graficos.
    """
    tempo = resultados["estados"]["tempo"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "Evolução das Concentrações dos Tanques de Processo",
        fontsize=16,
        fontweight="bold",
    )

    tanques = ["C", "D", "E"]
    for i, tanque in enumerate(tanques):
        # Estado
        axes[i].plot(
            tempo,
            resultados["estados"][f"C{tanque}"],
            color=CORES[tanque],
            linewidth=2,
            label=f"C{tanque} (medido)",
        )

        # Referência
        axes[i].plot(
            tempo,
            resultados["referencias"][f"C{tanque}_ref"],
            color=CORES["ref"],
            linestyle="--",
            linewidth=1.5,
            label="Referência",
            alpha=0.8,
        )

        # Limites
        axes[i].axhline(
            y=params.CB,
            color=CORES["limite"],
            linestyle=":",
            alpha=0.5,
            label=f"CB máx ({params.CB})",
        )

        axes[i].set_ylabel(f"C{tanque} [kg/m³]", fontsize=12, fontweight="bold")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=9)
        axes[i].set_ylim([0, params.CB + 20])

    axes[2].set_xlabel("Tempo [s]", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if salvar:
        if nome_arquivo is None:
            nome_arquivo = os.path.join(GRAFICOS_DIR, "concentracoes_tanques.png")
        else:
            nome_arquivo = os.path.join(GRAFICOS_DIR, nome_arquivo)
        plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plotar_controles(resultados: Dict, salvar: bool = True, nome_arquivo: str = None):
    """
    Plota as ações de controle (bombas e válvulas) ao longo do tempo.
    Salva a imagem em /graficos.
    """
    tempo = resultados["estados"]["tempo"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    fig.suptitle(
        "Ações de Controle (Bombas e Válvulas)", fontsize=16, fontweight="bold"
    )

    tanques = ["C", "D", "E"]
    labels = ["Bomba Água (u1)", "Bomba Salmoura (u2)", "Válvula Descarga (u3)"]

    for i, tanque in enumerate(tanques):
        for j, sufixo in enumerate(["1", "2", "3"]):
            chave = f"u{tanque}{sufixo}"
            axes[i, j].plot(
                tempo, resultados["controles"][chave], color=CORES[tanque], linewidth=2
            )
            axes[i, j].axhline(y=0.0, color="k", linestyle=":", alpha=0.3)
            axes[i, j].axhline(y=1.0, color="k", linestyle=":", alpha=0.3)
            axes[i, j].set_ylabel(f"{tanque}-{labels[j]}", fontsize=10)
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].set_ylim([-0.1, 1.1])

    for j in range(3):
        axes[2, j].set_xlabel("Tempo [s]", fontsize=11, fontweight="bold")

    plt.tight_layout()

    if salvar:
        if nome_arquivo is None:
            nome_arquivo = os.path.join(GRAFICOS_DIR, "controles_tanques.png")
        else:
            nome_arquivo = os.path.join(GRAFICOS_DIR, nome_arquivo)
        plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plotar_resumo_completo(
    resultados: Dict, salvar: bool = True, nome_arquivo: str = None
):
    """
    Plota um resumo consolidado com estados e controles.
    Salva a imagem em /graficos.
    """
    tempo = resultados["estados"]["tempo"]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(
        f"Simulação MPC - {resultados['parametros']['cenario']}",
        fontsize=16,
        fontweight="bold",
    )

    tanques = ["C", "D", "E"]

    # Primeira linha: Níveis
    for i, tanque in enumerate(tanques):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(
            tempo,
            resultados["estados"][f"h{tanque}"],
            color=CORES[tanque],
            linewidth=2,
            label="Medido",
        )
        ax.plot(
            tempo,
            resultados["referencias"][f"h{tanque}_ref"],
            color=CORES["ref"],
            linestyle="--",
            linewidth=1.5,
            label="Ref",
            alpha=0.7,
        )
        ax.set_ylabel(f"h{tanque} [m]", fontsize=10, fontweight="bold")
        ax.set_title(f"Tanque {tanque} - Nível", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Segunda linha: Concentrações
    for i, tanque in enumerate(tanques):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(
            tempo,
            resultados["estados"][f"C{tanque}"],
            color=CORES[tanque],
            linewidth=2,
            label="Medido",
        )
        ax.plot(
            tempo,
            resultados["referencias"][f"C{tanque}_ref"],
            color=CORES["ref"],
            linestyle="--",
            linewidth=1.5,
            label="Ref",
            alpha=0.7,
        )
        ax.set_ylabel(f"C{tanque} [kg/m³]", fontsize=10, fontweight="bold")
        ax.set_title(f"Tanque {tanque} - Concentração", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Terceira linha: Bombas
    for i, tanque in enumerate(tanques):
        ax = fig.add_subplot(gs[2, i])
        ax.plot(
            tempo,
            resultados["controles"][f"u{tanque}1"],
            color="blue",
            linewidth=1.5,
            label="Bomba Água",
            alpha=0.7,
        )
        ax.plot(
            tempo,
            resultados["controles"][f"u{tanque}2"],
            color="orange",
            linewidth=1.5,
            label="Bomba Salmoura",
            alpha=0.7,
        )
        ax.set_ylabel(f"Bombas {tanque}", fontsize=10, fontweight="bold")
        ax.set_title(f"Tanque {tanque} - Bombas", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim([0, 1])

    # Quarta linha: Válvulas
    for i, tanque in enumerate(tanques):
        ax = fig.add_subplot(gs[3, i])
        ax.plot(
            tempo,
            resultados["controles"][f"u{tanque}3"],
            color="green",
            linewidth=1.5,
            label="Válvula",
        )
        ax.set_ylabel(f"Válvula {tanque}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Tempo [s]", fontsize=10, fontweight="bold")
        ax.set_title(f"Tanque {tanque} - Válvula Descarga", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim([0, 1])

    if salvar:
        if nome_arquivo is None:
            nome_arquivo = os.path.join(GRAFICOS_DIR, "resumo_completo.png")
        else:
            nome_arquivo = os.path.join(GRAFICOS_DIR, nome_arquivo)
        plt.savefig(nome_arquivo, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# ANÁLISE DE DESEMPENHO E RELATÓRIO
# ==============================================================================


def analisar_desempenho(
    resultados: Dict,
    verbose: bool = True,
    salvar: bool = False,
    nome_arquivo: str = "relatorio_desempenho.txt",
) -> Dict:
    """
    Analisa o desempenho do sistema e verifica atendimento aos requisitos R1-R4.

    Returns:
        Dicionário com métricas calculadas e status de conformidade
    """
    tempo = resultados["estados"]["tempo"]
    metricas = {}

    tanques = ["C", "D", "E"]
    for tanque in tanques:
        metricas[tanque] = {}

        # Métricas de nível
        h = resultados["estados"][f"h{tanque}"]
        h_ref = resultados["referencias"][f"h{tanque}_ref"]

        t_settling_h = calcular_tempo_acomodacao(tempo, h, h_ref)
        overshoot_h = calcular_overshoot(h, h_ref)
        erro_regime_h = calcular_erro_regime_permanente(h, h_ref)

        metricas[tanque]["nivel"] = {
            "t_settling": t_settling_h,
            "overshoot": overshoot_h,
            "erro_regime": erro_regime_h,
        }

        # Métricas de concentração
        C = resultados["estados"][f"C{tanque}"]
        C_ref = resultados["referencias"][f"C{tanque}_ref"]

        t_settling_C = calcular_tempo_acomodacao(tempo, C, C_ref)
        overshoot_C = calcular_overshoot(C, C_ref)
        erro_regime_C = calcular_erro_regime_permanente(C, C_ref)

        metricas[tanque]["concentracao"] = {
            "t_settling": t_settling_C,
            "overshoot": overshoot_C,
            "erro_regime": erro_regime_C,
        }

    # Verifica violações de restrições
    violacoes = verificar_violacao_restricoes(
        resultados["estados"], resultados["controles"]
    )
    metricas["violacoes"] = violacoes

    # Avaliação dos requisitos
    conformidade = avaliar_requisitos(metricas)
    metricas["conformidade"] = conformidade

    if verbose:
        imprimir_relatorio_desempenho(metricas)
    if salvar:
        salvar_relatorio_texto(metricas, nome_arquivo=nome_arquivo)

    return metricas


def avaliar_requisitos(metricas: Dict) -> Dict:
    """
    Avalia se os requisitos R1-R4 foram atendidos.
    """
    conformidade = {
        "R1": {"status": True, "detalhes": []},
        "R2": {"status": True, "detalhes": []},
        "R3": {"status": True, "detalhes": []},
        "R4": {"status": True, "detalhes": []},
    }

    t_max_permitido = params.REQUISITOS["R1"]["t_settling_max"]
    erro_max_permitido = params.REQUISITOS["R2"]["erro_max_percentual"]
    overshoot_max_permitido = params.REQUISITOS["R3"]["overshoot_max_percentual"]

    for tanque in ["C", "D", "E"]:
        # R1: Velocidade
        for variavel in ["nivel", "concentracao"]:
            t_s = metricas[tanque][variavel]["t_settling"]
            if t_s > t_max_permitido:
                conformidade["R1"]["status"] = False
                conformidade["R1"]["detalhes"].append(
                    f"Tanque {tanque} ({variavel}): t_s={t_s:.1f}s > {t_max_permitido}s"
                )

        # R2: Erro em regime
        for variavel in ["nivel", "concentracao"]:
            erro = metricas[tanque][variavel]["erro_regime"]
            if erro > erro_max_permitido:
                conformidade["R2"]["status"] = False
                conformidade["R2"]["detalhes"].append(
                    f"Tanque {tanque} ({variavel}): erro={erro:.2f}% > {erro_max_permitido}%"
                )

        # R3: Overshoot
        for variavel in ["nivel", "concentracao"]:
            os = metricas[tanque][variavel]["overshoot"]
            if os > overshoot_max_permitido:
                conformidade["R3"]["status"] = False
                conformidade["R3"]["detalhes"].append(
                    f"Tanque {tanque} ({variavel}): overshoot={os:.2f}% > {overshoot_max_permitido}%"
                )

    # R4: Restrições
    violacoes = metricas["violacoes"]
    if any(violacoes.values()):
        conformidade["R4"]["status"] = False
        for chave, lista in violacoes.items():
            if lista:
                conformidade["R4"]["detalhes"].append(f"{chave}: {lista}")

    return conformidade


def imprimir_relatorio_desempenho(metricas: Dict):
    """
    Imprime relatório formatado de desempenho.
    """
    print("\n" + "=" * 70)
    print("RELATÓRIO DE DESEMPENHO DO SISTEMA MPC")
    print("=" * 70 + "\n")

    # Métricas por tanque
    for tanque in ["C", "D", "E"]:
        print(f"--- Tanque {tanque} ---")
        print(f"  Nível:")
        print(
            f"    Tempo de acomodação: {metricas[tanque]['nivel']['t_settling']:.1f}s"
        )
        print(f"    Overshoot: {metricas[tanque]['nivel']['overshoot']:.2f}%")
        print(f"    Erro em regime: {metricas[tanque]['nivel']['erro_regime']:.2f}%")
        print(f"  Concentração:")
        print(
            f"    Tempo de acomodamento: {metricas[tanque]['concentracao']['t_settling']:.1f}s"
        )
        print(f"    Overshoot: {metricas[tanque]['concentracao']['overshoot']:.2f}%")
        print(
            f"    Erro em regime: {metricas[tanque]['concentracao']['erro_regime']:.2f}%"
        )
        print()

    # Conformidade com requisitos
    print("=" * 70)
    print("VERIFICAÇÃO DE REQUISITOS (R1-R4)")
    print("=" * 70 + "\n")

    for req in ["R1", "R2", "R3", "R4"]:
        status = (
            "✓ ATENDIDO"
            if metricas["conformidade"][req]["status"]
            else "✗ NÃO ATENDIDO"
        )
        criterio = params.REQUISITOS[req]["criterio"]
        print(f"{req} - {params.REQUISITOS[req]['nome']}: {status}")
        print(f"   Critério: {criterio}")
        if not metricas["conformidade"][req]["status"]:
            print(f"   Detalhes:")
            for detalhe in metricas["conformidade"][req]["detalhes"]:
                print(f"     - {detalhe}")
        print()

    print("=" * 70 + "\n")


def salvar_relatorio_texto(
    metricas: Dict, nome_arquivo: str = "relatorio_desempenho.txt"
):
    """
    Salva o relatório de desempenho em um arquivo de texto.
    """
    with open(nome_arquivo, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RELATÓRIO DE DESEMPENHO DO SISTEMA MPC\n")
        f.write("=" * 70 + "\n\n")

        # Métricas por tanque
        for tanque in ["C", "D", "E"]:
            f.write(f"--- Tanque {tanque} ---\n")
            f.write(f"  Nível:\n")
            f.write(
                f"    Tempo de acomodação: {metricas[tanque]['nivel']['t_settling']:.1f}s\n"
            )
            f.write(f"    Overshoot: {metricas[tanque]['nivel']['overshoot']:.2f}%\n")
            f.write(
                f"    Erro em regime: {metricas[tanque]['nivel']['erro_regime']:.2f}%\n"
            )
            f.write(f"  Concentração:\n")
            f.write(
                f"    Tempo de acomodamento: {metricas[tanque]['concentracao']['t_settling']:.1f}s\n"
            )
            f.write(
                f"    Overshoot: {metricas[tanque]['concentracao']['overshoot']:.2f}%\n"
            )
            f.write(
                f"    Erro em regime: {metricas[tanque]['concentracao']['erro_regime']:.2f}%\n"
            )
            f.write("\n")

        # Conformidade com requisitos
        f.write("=" * 70 + "\n")
        f.write("VERIFICAÇÃO DE REQUISITOS (R1-R4)\n")
        f.write("=" * 70 + "\n\n")

        for req in ["R1", "R2", "R3", "R4"]:
            status = (
                "✓ ATENDIDO"
                if metricas["conformidade"][req]["status"]
                else "✗ NÃO ATENDIDO"
            )
            criterio = params.REQUISITOS[req]["criterio"]
            f.write(f"{req} - {params.REQUISITOS[req]['nome']}: {status}\n")
            f.write(f"   Critério: {criterio}\n")
            if not metricas["conformidade"][req]["status"]:
                f.write(f"   Detalhes:\n")
                for detalhe in metricas["conformidade"][req]["detalhes"]:
                    f.write(f"     - {detalhe}\n")
            f.write("\n")

    print(f"Relatório salvo em: {nome_arquivo}")


def exportar_para_csv(resultados: Dict, nome_arquivo: str = "resultados_simulacao.csv"):
    """
    Exporta os dados da simulação para arquivo CSV.
    """
    # Garante que o diretório existe
    diretorio = os.path.dirname(nome_arquivo)
    if diretorio and not os.path.exists(diretorio):
        os.makedirs(diretorio, exist_ok=True)

    # Cria DataFrame combinando estados, controles e referências
    df = pd.DataFrame(
        {
            "tempo": resultados["estados"]["tempo"],
            **resultados["estados"],
            **resultados["controles"],
            **resultados["referencias"],
        }
    )

    df.to_csv(nome_arquivo, index=False)
    print(f"Dados exportados para: {nome_arquivo}")


# ==============================================================================
# TESTE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MÓDULO DE VISUALIZAÇÃO - Execute simulacao_principal.py primeiro")
    print("=" * 70)
