"""
controlador_mpc.py

Implementa o controlador MPC (Model Predictive Control) para os tanques de processo
C, D e E, baseado em modelo linearizado em espaço de estados com discretização ZOH.

Características principais:
- MPC linear com horizontes de predição (Np) e controle (Nc)
- Função custo quadrática com pesos Q, R e termo integral I (offset-free)
- Restrições em estados (níveis, concentrações) e controles (bombas, válvulas)
- Otimização convexa via CVXPY

Baseado na Entrega 3 e no trabalho do semestre 2025/1
"""

import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
from typing import Tuple, Dict
import parametros_sistema as params


# ==============================================================================
# DISCRETIZAÇÃO DO MODELO LINEAR
# ==============================================================================


def discretizar_modelo(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, Ts: float
) -> Tuple[np.ndarray, ...]:
    """
    Discretiza o modelo contínuo em tempo usando Zero-Order Hold (ZOH).

    Sistema contínuo:
        dx/dt = A*x + B*u
        y = C*x + D*u

    Sistema discreto:
        x[k+1] = Ad*x[k] + Bd*u[k]
        y[k] = C*x[k] + D*u[k]

    Args:
        A: matriz de estado contínua
        B: matriz de entrada contínua
        C: matriz de saída contínua
        D: matriz de transmissão direta contínua
        Ts: período de amostragem (s)

    Returns:
        Tupla (Ad, Bd, Cd, Dd) - matrizes discretizadas
    """
    sys_discreto = cont2discrete((A, B, C, D), Ts, method="zoh")

    Ad = np.array(sys_discreto[0])
    Bd = np.array(sys_discreto[1])
    Cd = np.array(sys_discreto[2])
    Dd = np.array(sys_discreto[3])

    return Ad, Bd, Cd, Dd


def extrair_modelo_tanque(nome_tanque: str, Ts: float) -> Tuple[np.ndarray, ...]:
    """
    Extrai e discretiza o modelo linearizado de um tanque de processo.

    Para cada tanque (C, D ou E), o modelo tem 2 estados [h, C] e 3 entradas [u1, u2, u3].

    Args:
        nome_tanque: 'C', 'D' ou 'E'
        Ts: período de amostragem (s)

    Returns:
        Tupla (Ad, Bd, Cd, Dd, idx_estados, idx_controles)
    """
    # Mapeamento dos índices de estados no modelo global (8 estados)
    indices_estados = {
        "C": [2, 3],  # hC, CC
        "D": [4, 5],  # hD, CD
        "E": [6, 7],  # hE, CE
    }

    # Mapeamento dos índices de controles no modelo global (11 entradas)
    indices_controles = {
        "C": [2, 3, 4],  # uC1, uC2, uC3
        "D": [5, 6, 7],  # uD1, uD2, uD3
        "E": [8, 9, 10],  # uE1, uE2, uE3
    }

    idx_estados = indices_estados[nome_tanque]
    idx_controles = indices_controles[nome_tanque]

    # Extrai submatrizes do modelo global
    A_tanque = params.A_CONTINUA[np.ix_(idx_estados, idx_estados)]
    B_tanque = params.B_CONTINUA[np.ix_(idx_estados, idx_controles)]
    C_tanque = np.eye(2)  # Todos os estados são medidos
    D_tanque = np.zeros((2, 3))

    # Discretiza
    Ad, Bd, Cd, Dd = discretizar_modelo(A_tanque, B_tanque, C_tanque, D_tanque, Ts)

    return Ad, Bd, Cd, Dd, idx_estados, idx_controles


# ==============================================================================
# CLASSE: CONTROLADOR MPC POR TANQUE
# ==============================================================================


class ControladorMPC:
    """
    Controlador MPC para um tanque de processo individual.

    Resolve o problema de otimização:
        min  Σ ||y[k] - r[k]||²_Q + Σ ||u[k]||²_R + Σ ||e_int[k]||²_I
        s.t. x[k+1] = Ad*x[k] + Bd*u[k]
             y[k] = Cd*x[k]
             x_min <= x[k] <= x_max
             u_min <= u[k] <= u_max
             Δu_min <= u[k] - u[k-1] <= Δu_max
    """

    def __init__(self, nome_tanque: str, Ts: float):
        """
        Args:
            nome_tanque: 'C', 'D' ou 'E'
            Ts: período de amostragem (s)
        """
        self.nome = nome_tanque
        self.Ts = Ts

        # Extrai modelo discretizado
        self.Ad, self.Bd, self.Cd, self.Dd, _, _ = extrair_modelo_tanque(
            nome_tanque, Ts
        )

        # Dimensões
        self.nx = self.Ad.shape[0]  # número de estados (2: h, C)
        self.nu = self.Bd.shape[1]  # número de entradas (3: u1, u2, u3)
        self.ny = self.Cd.shape[0]  # número de saídas (2: h, C)

        # Horizontes
        self.Np = params.MPC_HORIZONTES["Np"]
        self.Nc = params.MPC_HORIZONTES["Nc"]

        # Pesos
        self.Q = params.MPC_PESOS[nome_tanque]["Q"]
        self.R = params.MPC_PESOS[nome_tanque]["R"]
        self.I = params.MPC_PESOS[nome_tanque]["I"]

        # Limites
        self.h_min = params.LIMITES_NIVEL["h_min"]
        self.h_max = params.LIMITES_NIVEL["h_max"]
        self.C_min = params.LIMITES_CONCENTRACAO["C_min"]
        self.C_max = params.LIMITES_CONCENTRACAO["C_max"]

        self.u_min = np.array(
            [
                params.LIMITES_ATUADORES["u1_min"],
                params.LIMITES_ATUADORES["u2_min"],
                params.LIMITES_ATUADORES["u3_min"],
            ]
        )

        self.u_max = np.array(
            [
                params.LIMITES_ATUADORES["u1_max"],
                params.LIMITES_ATUADORES["u2_max"],
                params.LIMITES_ATUADORES["u3_max"],
            ]
        )

        self.delta_u_max = np.array(
            [
                params.LIMITES_VARIACAO["delta_u1_max"],
                params.LIMITES_VARIACAO["delta_u2_max"],
                params.LIMITES_VARIACAO["delta_u3_max"],
            ]
        )

        # Estados internos
        self.erro_integral = np.zeros(self.ny)  # Termo integral para offset-free
        self.u_anterior = np.array(
            [
                params.PONTO_OPERACAO["u1_eq"],
                params.PONTO_OPERACAO["u2_eq"],
                params.PONTO_OPERACAO["u3_eq"],
            ]
        )

        # Ponto de operação (para trabalhar com desvios)
        self.x_eq = np.array(
            [params.PONTO_OPERACAO["h_eq"], params.PONTO_OPERACAO["C_eq"]]
        )

        self.u_eq = self.u_anterior.copy()

        print(f"[MPC-{nome_tanque}] Inicializado: Np={self.Np}, Nc={self.Nc}, Ts={Ts}s")

    def calcular_controle(
        self, x_medido: np.ndarray, referencia: np.ndarray
    ) -> np.ndarray:
        """
        Resolve o problema de otimização MPC e retorna a ação de controle ótima.

        Args:
            x_medido: estado atual medido [h, C]
            referencia: referência desejada [h_ref, C_ref]

        Returns:
            Ação de controle ótima u = [u1, u2, u3]
        """
        # Trabalha com desvios em relação ao ponto de operação
        x_dev = x_medido - self.x_eq
        r_dev = referencia - self.x_eq

        # Variáveis de decisão do CVXPY
        x = cp.Variable((self.nx, self.Np + 1))  # estados futuros
        u = cp.Variable((self.nu, self.Nc))  # controles futuros
        e_int = cp.Variable((self.ny, self.Np + 1))  # erros integrais

        # Adiciona variáveis de folga para restrições de nível
        slack_h = cp.Variable(
            (self.Np + 1,), nonneg=True
        )  # Folga para restrições de nível
        peso_slack = 1e6  # Penalização alta para desencorajar uso

        # Função custo
        custo = 0.0

        # Penalização das folgas
        custo += peso_slack * cp.sum(slack_h)

        # Restrições
        restricoes = [
            x[:, 0] == x_dev,  # condição inicial
            e_int[:, 0] == self.erro_integral,  # integral inicial
        ]

        # Horizonte de predição
        for k in range(self.Np):
            # Índice de controle (após Nc, mantém u constante)
            k_u = min(k, self.Nc - 1)

            # Predição de estado
            restricoes.append(x[:, k + 1] == self.Ad @ x[:, k] + self.Bd @ u[:, k_u])

            # Saída predita
            y_pred = self.Cd @ x[:, k + 1]

            # Erro de rastreamento
            erro = y_pred - r_dev

            # Atualização do erro integral
            restricoes.append(e_int[:, k + 1] == e_int[:, k] + erro)

            # Custo de rastreamento
            custo += cp.quad_form(erro, self.Q)

            # Custo de esforço de controle (apenas dentro do horizonte Nc)
            if k < self.Nc:
                custo += cp.quad_form(u[:, k], self.R)

            # Custo do termo integral (offset-free)
            custo += cp.quad_form(e_int[:, k + 1], self.I)

            # Restrições nos estados (com margem de segurança)
            restricoes.append(x[0, k + 1] + self.x_eq[0] >= self.h_min - slack_h[k + 1])
            restricoes.append(x[0, k + 1] + self.x_eq[0] <= self.h_max + slack_h[k + 1])
            restricoes.append(x[1, k + 1] + self.x_eq[1] >= self.C_min)
            restricoes.append(x[1, k + 1] + self.x_eq[1] <= self.C_max)

            # Restrição anti-overshoot (limita o nível máximo baseado na referência)
            if np.any(r_dev > 0):  # Se há degrau positivo
                restricoes.append(
                    x[0, k + 1] <= r_dev[0] * (1 + params.LIMITE_OVERSHOOT)
                )  # Max de overshoot em nível

            # Restrição anti-undershoot (limita o nível mínimo baseado na referência) - DEGRAU NEGATIVO
            if np.any(r_dev < 0):  # Se há degrau negativo
                restricoes.append(
                    x[0, k + 1] >= r_dev[0] * (1 - params.LIMITE_UNDERSHOOT)
                )  # Max de undershoot em nível (evita cair demais)

            # Restrição anti-overshoot (concentração)
            if r_dev[1] > 0:  # Se há degrau positivo na concentração
                restricoes.append(
                    x[1, k + 1] <= r_dev[1] * (1 + params.LIMITE_OVERSHOOT)
                )  # Max de overshoot em concentração

            # Restrição anti-undershoot (concentração) - DEGRAU NEGATIVO
            if r_dev[1] < 0:  # Se há degrau negativo na concentração
                restricoes.append(
                    x[1, k + 1] >= r_dev[1] * (1 - params.LIMITE_UNDERSHOOT)
                )  # Max de undershoot em concentração (evita cair demais)

        # Restrições nos controles
        for k in range(self.Nc):
            restricoes.append(u[:, k] >= self.u_min - self.u_eq)
            restricoes.append(u[:, k] <= self.u_max - self.u_eq)

            # Restrição de taxa de variação
            if k == 0:
                delta_u = u[:, k] - (self.u_anterior - self.u_eq)
            else:
                delta_u = u[:, k] - u[:, k - 1]

            restricoes.append(delta_u >= -self.delta_u_max)
            restricoes.append(delta_u <= self.delta_u_max)

        # Monta e resolve o problema
        problema = cp.Problem(cp.Minimize(custo), restricoes)

        try:
            # Solução do problema (antiga)
            # problema.solve(solver=cp.OSQP, warm_start=True, verbose=False)

            # problema.solve(
            #     solver=cp.OSQP,
            #     warm_start=True,
            #     verbose=False,
            #     max_iter=10000,  # Aumenta limite de iterações
            #     eps_abs=1e-4,  # Tolerância absoluta
            #     eps_rel=1e-4,  # Tolerância relativa
            #     polish=True,  # Refinamento da solução
            # )

            problema.solve(
                solver=cp.CLARABEL,
                verbose=False,
            )

            if problema.status in ["optimal", "optimal_inaccurate"]:
                # Extrai primeiro controle da sequência ótima (horizonte deslizante)
                u_otimo_dev = u[:, 0].value
                u_otimo = u_otimo_dev + self.u_eq

                # Atualiza estado integral
                y_atual = self.Cd @ x_dev
                erro_atual = y_atual - r_dev
                self.erro_integral += erro_atual

                # Atualiza controle anterior
                self.u_anterior = u_otimo.copy()

                return np.clip(u_otimo, self.u_min, self.u_max)

            else:
                print(
                    f"[MPC-{self.nome}] AVISO: Otimização falhou ({problema.status}). "
                    f"Tentando estratégia de recuperação..."
                )

                # Estratégia 1: Reduzir horizontes temporariamente
                Np_reduzido = max(10, self.Np // 2)
                Nc_reduzido = max(5, self.Nc // 2)
                try:
                    # Variáveis de decisão para horizontes reduzidos
                    x_r = cp.Variable((self.nx, Np_reduzido + 1))
                    u_r = cp.Variable((self.nu, Nc_reduzido))
                    e_int_r = cp.Variable((self.ny, Np_reduzido + 1))
                    slack_h_r = cp.Variable((Np_reduzido + 1,), nonneg=True)
                    custo_r = 1e6 * cp.sum(slack_h_r)
                    restricoes_r = [
                        x_r[:, 0] == x_dev,
                        e_int_r[:, 0] == self.erro_integral,
                    ]
                    for k in range(Np_reduzido):
                        k_u = min(k, Nc_reduzido - 1)
                        restricoes_r.append(
                            x_r[:, k + 1] == self.Ad @ x_r[:, k] + self.Bd @ u_r[:, k_u]
                        )
                        y_pred_r = self.Cd @ x_r[:, k + 1]
                        erro_r = y_pred_r - r_dev
                        restricoes_r.append(e_int_r[:, k + 1] == e_int_r[:, k] + erro_r)
                        custo_r += cp.quad_form(erro_r, self.Q)
                        if k < Nc_reduzido:
                            custo_r += cp.quad_form(u_r[:, k], self.R)
                        custo_r += cp.quad_form(e_int_r[:, k + 1], self.I)
                        restricoes_r.append(
                            x_r[0, k + 1] + self.x_eq[0]
                            >= self.h_min - slack_h_r[k + 1]
                        )
                        restricoes_r.append(
                            x_r[0, k + 1] + self.x_eq[0]
                            <= self.h_max + slack_h_r[k + 1]
                        )
                        restricoes_r.append(x_r[1, k + 1] + self.x_eq[1] >= self.C_min)
                        restricoes_r.append(x_r[1, k + 1] + self.x_eq[1] <= self.C_max)
                    for k in range(Nc_reduzido):
                        restricoes_r.append(u_r[:, k] >= self.u_min - self.u_eq)
                        restricoes_r.append(u_r[:, k] <= self.u_max - self.u_eq)
                        if k == 0:
                            delta_u_r = u_r[:, k] - (self.u_anterior - self.u_eq)
                        else:
                            delta_u_r = u_r[:, k] - u_r[:, k - 1]
                        restricoes_r.append(delta_u_r >= -self.delta_u_max)
                        restricoes_r.append(delta_u_r <= self.delta_u_max)
                    problema_r = cp.Problem(cp.Minimize(custo_r), restricoes_r)
                    problema_r.solve(solver=cp.CLARABEL, verbose=False)
                    if problema_r.status in ["optimal", "optimal_inaccurate"]:
                        u_otimo_dev_r = u_r[:, 0].value
                        u_otimo_r = u_otimo_dev_r + self.u_eq
                        self.u_anterior = u_otimo_r.copy()
                        print(
                            f"[MPC-{self.nome}] Recuperação com horizonte reduzido bem-sucedida."
                        )
                        return np.clip(u_otimo_r, self.u_min, self.u_max)
                    else:
                        print(
                            f"[MPC-{self.nome}] Recuperação com horizonte reduzido falhou ({problema_r.status})."
                        )
                except Exception as e_r:
                    print(f"[MPC-{self.nome}] ERRO na recuperação reduzida: {e_r}")

                # Estratégia 2: Aplicar controle conservador
                print(f"[MPC-{self.nome}] Aplicando controle conservador.")
                fator_conservador = 0.9
                u_conservador = self.u_anterior * fator_conservador + self.u_eq * (
                    1 - fator_conservador
                )
                return np.clip(u_conservador, self.u_min, self.u_max)

        except Exception as e:
            print(
                f"[MPC-{self.nome}] ERRO na otimização: {e}. Mantendo controle anterior."
            )
            return self.u_anterior

    def resetar_integrador(self):
        """Reseta o termo integral (útil ao trocar de referência)."""
        self.erro_integral = np.zeros(self.ny)
        print(f"[MPC-{self.nome}] Integrador resetado.")


# ==============================================================================
# CONTROLADORES AUXILIARES (TANQUES A E B)
# ==============================================================================


class ControladorPID:
    """
    Controlador PID simples para os reservatórios de utilidades (A e B).

    Usa apenas ação proporcional e integral (PI) para manter níveis constantes.
    """

    def __init__(self, nome_reservatorio: str, Kp: float = 0.5, Ki: float = 0.1):
        """
        Args:
            nome_reservatorio: 'A' ou 'B'
            Kp: ganho proporcional
            Ki: ganho integral
        """
        self.nome = nome_reservatorio
        self.Kp = Kp
        self.Ki = Ki
        self.erro_integral = 0.0
        self.setpoint = (
            params.PONTO_OPERACAO["hA_eq"]
            if nome_reservatorio == "A"
            else params.PONTO_OPERACAO["hB_eq"]
        )

        print(
            f"[PID-{nome_reservatorio}] Inicializado: Kp={Kp}, Ki={Ki}, SP={self.setpoint:.2f}m"
        )

    def calcular_controle(self, nivel_medido: float, Ts: float) -> float:
        """
        Calcula a ação de controle PI.

        Args:
            nivel_medido: nível atual do reservatório (m)
            Ts: período de amostragem (s)

        Returns:
            Comando da válvula de suprimento [0, 1]
        """
        erro = self.setpoint - nivel_medido

        # Termo proporcional
        P = self.Kp * erro

        # Termo integral (com anti-windup simples)
        self.erro_integral += erro * Ts
        self.erro_integral = np.clip(self.erro_integral, -10.0, 10.0)
        I = self.Ki * self.erro_integral

        # Ação de controle
        u = P + I

        # Satura nos limites físicos
        return np.clip(u, 0.0, 1.0)

    def resetar(self):
        """Reseta o integrador."""
        self.erro_integral = 0.0


# ==============================================================================
# SISTEMA DE CONTROLE COMPLETO
# ==============================================================================


class SistemaControle:
    """
    Gerencia todos os controladores do sistema:
    - MPCs para tanques C, D, E
    - PIDs para reservatórios A, B
    """

    def __init__(self, Ts: float):
        """
        Args:
            Ts: período de amostragem do sistema (s)
        """
        self.Ts = Ts

        # Controladores MPC
        self.mpc_C = ControladorMPC("C", Ts)
        self.mpc_D = ControladorMPC("D", Ts)
        self.mpc_E = ControladorMPC("E", Ts)

        # Controladores PID
        self.pid_A = ControladorPID("A", Kp=15.0, Ki=0.25)
        self.pid_B = ControladorPID("B", Kp=15.0, Ki=0.25)

        print(f"\n{'='*70}")
        print("SISTEMA DE CONTROLE COMPLETO INICIALIZADO")
        print(f"{'='*70}\n")

    def calcular_acoes(self, estados: dict, referencias: dict) -> dict:
        """
        Calcula todas as ações de controle do sistema.

        Args:
            estados: dicionário com estados medidos
                {'hA', 'hB', 'hC', 'CC', 'hD', 'CD', 'hE', 'CE'}
            referencias: dicionário com referências desejadas
                {'hC_ref', 'CC_ref', 'hD_ref', 'CD_ref', 'hE_ref', 'CE_ref'}

        Returns:
            Dicionário com ações de controle
                {'uA', 'uB', 'uC1', 'uC2', 'uC3', 'uD1', 'uD2', 'uD3', 'uE1', 'uE2', 'uE3'}
        """
        # Controle dos reservatórios (mantém níveis constantes)
        uA = self.pid_A.calcular_controle(estados["hA"], self.Ts)
        uB = self.pid_B.calcular_controle(estados["hB"], self.Ts)

        # Controle MPC dos tanques de processo
        x_C = np.array([estados["hC"], estados["CC"]])
        r_C = np.array([referencias["hC_ref"], referencias["CC_ref"]])
        u_C = self.mpc_C.calcular_controle(x_C, r_C)

        x_D = np.array([estados["hD"], estados["CD"]])
        r_D = np.array([referencias["hD_ref"], referencias["CD_ref"]])
        u_D = self.mpc_D.calcular_controle(x_D, r_D)

        x_E = np.array([estados["hE"], estados["CE"]])
        r_E = np.array([referencias["hE_ref"], referencias["CE_ref"]])
        u_E = self.mpc_E.calcular_controle(x_E, r_E)

        return {
            "uA": uA,
            "uB": uB,
            "uC1": u_C[0],
            "uC2": u_C[1],
            "uC3": u_C[2],
            "uD1": u_D[0],
            "uD2": u_D[1],
            "uD3": u_D[2],
            "uE1": u_E[0],
            "uE2": u_E[1],
            "uE3": u_E[2],
        }


# ==============================================================================
# TESTES
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE DO CONTROLADOR MPC")
    print("=" * 70)

    # Cria controlador MPC para tanque C
    Ts = params.TS_CONTROLADOR
    mpc = ControladorMPC("C", Ts)

    # Estado inicial (no ponto de operação)
    x_atual = np.array([params.PONTO_OPERACAO["h_eq"], params.PONTO_OPERACAO["C_eq"]])

    # Referência (pequeno degrau)
    ref = np.array([1.7, 200.0])  # +0.2m em nível, +20kg/m³ em concentração

    print(f"\nEstado inicial: h={x_atual[0]:.2f}m, C={x_atual[1]:.1f}kg/m³")
    print(f"Referência: h={ref[0]:.2f}m, C={ref[1]:.1f}kg/m³")

    # Calcula ação de controle
    u_otimo = mpc.calcular_controle(x_atual, ref)

    print(f"\nAção de controle ótima:")
    print(f"  u1 (bomba água): {u_otimo[0]:.4f}")
    print(f"  u2 (bomba salmoura): {u_otimo[1]:.4f}")
    print(f"  u3 (válvula descarga): {u_otimo[2]:.4f}")
    print("\n" + "=" * 70)
