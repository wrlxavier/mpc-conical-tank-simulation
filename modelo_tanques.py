"""
Implementa o modelo fenomenológico não-linear dos tanques do sistema:
- 2 reservatórios cilíndricos (A e B) para utilidades
- 3 tanques tronco-cônicos (C, D, E) de processo

Cada tanque possui métodos para calcular suas propriedades geométricas,
derivadas temporais e atualização de estados via Runge-Kutta de 4 ordem.
"""

import numpy as np
from typing import Tuple
import parametros_sistema as params


# CLASSE BASE: TANQUE CILÍNDRICO


class TanqueCilindrico:
    """
    Modelo de um tanque cilíndrico para os reservatórios de utilidades (A e B).

    Equação dinâmica (balanço de massa total):
        dh/dt = (Q_in - Q_out) / A_t

    onde A_t é a área da seção transversal (constante).
    """

    def __init__(
        self,
        nome: str,
        raio: float,
        altura_max: float,
        nivel_inicial: float = 1.5,
        concentracao: float = 0.0,
    ):
        """
        Args:
            nome: identificador do tanque ('A' ou 'B')
            raio: raio do cilindro (m)
            altura_max: altura máxima do tanque (m)
            nivel_inicial: nível inicial do líquido (m)
            concentracao: concentração (kg/m³) - para tanque B (salmoura)
        """
        self.nome = nome
        self.raio = raio
        self.altura_max = altura_max
        self.area = np.pi * raio**2  # Área constante

        # Estados
        self.nivel = nivel_inicial
        self.concentracao = concentracao  # Fixo para B, zero para A

        # Vazões
        self.vazao_entrada = 0.0  # m³/s
        self.vazao_saida = 0.0  # m³/s

    def set_vazoes(self, vazao_entrada: float, vazao_saida: float) -> None:
        """Atualiza as vazões de entrada e saída."""
        self.vazao_entrada = max(0.0, vazao_entrada)
        self.vazao_saida = max(0.0, vazao_saida)

    def derivada_nivel(self) -> float:
        """
        Calcula dh/dt para o tanque cilíndrico.

        Returns:
            Taxa de variação do nível (m/s)
        """
        return (self.vazao_entrada - self.vazao_saida) / self.area

    def atualizar(self, dt: float) -> Tuple[float, float]:
        """
        Atualiza o estado do tanque usando Runge-Kutta de 4 ordem.

        Args:
            dt: passo de tempo (s)

        Returns:
            Tupla (novo_nivel, concentracao)
        """
        # Salva estado inicial
        h0 = self.nivel

        # RK4 para o nível
        k1 = self.derivada_nivel()

        self.nivel = h0 + k1 * dt / 2.0
        k2 = self.derivada_nivel()

        self.nivel = h0 + k2 * dt / 2.0
        k3 = self.derivada_nivel()

        self.nivel = h0 + k3 * dt
        k4 = self.derivada_nivel()

        # Atualização final
        self.nivel = h0 + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.0

        # Saturação nos limites físicos
        self.nivel = np.clip(self.nivel, 0.0, self.altura_max)

        return self.nivel, self.concentracao

    def get_volume(self) -> float:
        """Retorna o volume atual de líquido (m³)."""
        return self.area * self.nivel

    def __repr__(self):
        return f"TanqueCilindrico({self.nome}): h={self.nivel:.3f}m, V={self.get_volume():.3f}m³"


# CLASSE: TANQUE TRONCO-CÔNICO


class TanqueTroncoConico:
    """
    Modelo de um tanque tronco-cônico para os tanques de processo (C, D, E).
    """

    def __init__(
        self,
        nome: str,
        raio_inferior: float,
        raio_superior: float,
        altura_max: float,
        kv_valvula: float,
        kp_bomba_agua: float,
        kp_bomba_salmoura: float,
        nivel_inicial: float = 1.5,
        concentracao_inicial: float = 180.0,
    ):
        """
        Args:
            nome: identificador do tanque ('C', 'D' ou 'E')
            raio_inferior: raio da base inferior (m)
            raio_superior: raio da base superior (m)
            altura_max: altura máxima do tanque (m)
            kv_valvula: coeficiente de descarga da válvula (m^(5/2)/s)
            kp_bomba_agua: ganho da bomba de água (m³/s)
            kp_bomba_salmoura: ganho da bomba de salmoura (m³/s)
            nivel_inicial: nível inicial (m)
            concentracao_inicial: concentração inicial (kg/m³)
        """
        self.nome = nome
        self.raio_inferior = raio_inferior
        self.raio_superior = raio_superior
        self.altura_max = altura_max
        self.kv = kv_valvula
        self.kp_agua = kp_bomba_agua
        self.kp_salmoura = kp_bomba_salmoura

        # Parâmetro geométrico (conicidade)
        self.dr_dh = (raio_superior - raio_inferior) / altura_max

        # Estados
        self.nivel = nivel_inicial
        self.concentracao = concentracao_inicial

        # Comandos de controle (atualizados externamente)
        self.u1 = 0.0  # comando bomba água [0, 1]
        self.u2 = 0.0  # comando bomba salmoura [0, 1]
        self.u3 = 0.0  # abertura válvula descarga [0, 1]

        # Concentração da salmoura (parâmetro externo)
        self.CB = params.CB

    def set_controles(self, u1: float, u2: float, u3: float) -> None:
        """
        Atualiza os comandos de controle (saturados nos limites).

        Args:
            u1: comando bomba água [0, 1]
            u2: comando bomba salmoura [0, 1]
            u3: abertura válvula descarga [0, 1]
        """
        self.u1 = np.clip(u1, 0.0, 1.0)
        self.u2 = np.clip(u2, 0.0, 1.0)
        self.u3 = np.clip(u3, 0.0, 1.0)

    def calcular_raio(self, h: float) -> float:
        """
        Calcula o raio da seção transversal na altura h.
        """
        return self.raio_inferior + self.dr_dh * h

    def calcular_area(self, h: float) -> float:
        """
        Calcula a área da seção transversal na altura h.
        """
        r_h = self.calcular_raio(h)
        return np.pi * r_h**2

    def calcular_volume(self, h: float) -> float:
        """
        Calcula o volume de líquido até a altura h.
        """
        r_h = self.calcular_raio(h)
        return (np.pi * h / 3.0) * (
            self.raio_inferior**2 + self.raio_inferior * r_h + r_h**2
        )

    def calcular_vazoes(self) -> Tuple[float, float, float]:
        """
        Calcula as vazões atuais com base nos controles.

        Returns:
            Tupla (Q_agua, Q_salmoura, Q_saida) em m³/s
        """
        Q_agua = self.kp_agua * self.u1
        Q_salmoura = self.kp_salmoura * self.u2
        Q_saida = self.kv * self.u3 * np.sqrt(max(self.nivel, 0.0))

        return Q_agua, Q_salmoura, Q_saida

    def derivada_nivel(self) -> float:
        """
        Calcula dh/dt (balanço de massa total).

        Returns:
            Taxa de variação do nível (m/s)
        """
        Q_agua, Q_salmoura, Q_saida = self.calcular_vazoes()
        A_h = self.calcular_area(self.nivel)

        if A_h < 1e-9:  # Evita divisão por zero
            return 0.0

        return (Q_agua + Q_salmoura - Q_saida) / A_h

    def derivada_concentracao(self) -> float:
        """
        Calcula dC/dt (balanço de massa de espécie).

        Returns:
            Taxa de variação da concentração (kg/(m³·s))
        """
        Q_agua, Q_salmoura, Q_saida = self.calcular_vazoes()
        V_h = self.calcular_volume(self.nivel)

        if V_h < 1e-9:
            return 0.0

        termo_entrada_sal = Q_salmoura * self.CB
        termo_saida_sal = self.concentracao * (Q_agua + Q_salmoura)

        return (termo_entrada_sal - termo_saida_sal) / V_h

    def atualizar(self, dt: float) -> Tuple[float, float]:
        """
        Atualiza os estados do tanque usando Runge-Kutta de 4 ordem (RK4).

        Args:
            dt: passo de tempo (s)

        Returns:
            Tupla (novo_nivel, nova_concentracao)
        """
        # Salva estados iniciais
        h0 = self.nivel
        C0 = self.concentracao

        # ===== RK4 para nível =====
        k1_h = self.derivada_nivel()

        self.nivel = h0 + k1_h * dt / 2.0
        k2_h = self.derivada_nivel()

        self.nivel = h0 + k2_h * dt / 2.0
        k3_h = self.derivada_nivel()

        self.nivel = h0 + k3_h * dt
        k4_h = self.derivada_nivel()

        # Atualização final do nível
        novo_nivel = h0 + (k1_h + 2 * k2_h + 2 * k3_h + k4_h) * dt / 6.0

        # RK4 para concentração
        self.nivel = h0
        self.concentracao = C0

        k1_C = self.derivada_concentracao()

        self.nivel = h0 + k1_h * dt / 2.0
        self.concentracao = C0 + k1_C * dt / 2.0
        k2_C = self.derivada_concentracao()

        self.nivel = h0 + k2_h * dt / 2.0
        self.concentracao = C0 + k2_C * dt / 2.0
        k3_C = self.derivada_concentracao()

        self.nivel = h0 + k3_h * dt
        self.concentracao = C0 + k3_C * dt
        k4_C = self.derivada_concentracao()

        nova_concentracao = C0 + (k1_C + 2 * k2_C + 2 * k3_C + k4_C) * dt / 6.0

        self.nivel = np.clip(novo_nivel, 0.0, self.altura_max)
        self.concentracao = np.clip(nova_concentracao, 0.0, self.CB)

        return self.nivel, self.concentracao

    def __repr__(self):
        Q_agua, Q_salmoura, Q_saida = self.calcular_vazoes()
        return (
            f"TanqueTroncoConico({self.nome}): h={self.nivel:.3f}m, "
            f"C={self.concentracao:.1f}kg/m³, V={self.calcular_volume(self.nivel):.3f}m³, "
            f"Q_in={Q_agua+Q_salmoura:.5f}m³/s, Q_out={Q_saida:.5f}m³/s"
        )


# CLASSE: SISTEMA COMPLETO (5 TANQUES)


class SistemaCompleto:
    """
    Encapsula o sistema completo de 5 tanques:
    - Tanques A e B (cilíndricos - utilidades)
    - Tanques C, D e E (tronco-cônicos - processo)
    """

    def __init__(self):
        """Inicializa todos os tanques com parâmetros do arquivo de configuração."""

        # Reservatórios cilíndricos
        self.tanque_A = TanqueCilindrico(
            nome="A",
            raio=params.TANQUES_UTILIDADES["A"]["raio"],
            altura_max=params.TANQUES_UTILIDADES["A"]["altura_maxima"],
            nivel_inicial=params.CONDICOES_INICIAIS["hA_0"],
            concentracao=0.0,  # água pura
        )

        self.tanque_B = TanqueCilindrico(
            nome="B",
            raio=params.TANQUES_UTILIDADES["B"]["raio"],
            altura_max=params.TANQUES_UTILIDADES["B"]["altura_maxima"],
            nivel_inicial=params.CONDICOES_INICIAIS["hB_0"],
            concentracao=params.CB,  # salmoura
        )

        # Tanques de processo tronco-cônicos
        self.tanque_C = TanqueTroncoConico(
            nome="C",
            raio_inferior=params.TANQUES_PROCESSO["C"]["raio_inferior"],
            raio_superior=params.TANQUES_PROCESSO["C"]["raio_superior"],
            altura_max=params.TANQUES_PROCESSO["C"]["altura_maxima"],
            kv_valvula=params.KV_VALVULAS["C"],
            kp_bomba_agua=params.KP_BOMBAS_AGUA["C"],
            kp_bomba_salmoura=params.KP_BOMBAS_SALMOURA["C"],
            nivel_inicial=params.CONDICOES_INICIAIS["hC_0"],
            concentracao_inicial=params.CONDICOES_INICIAIS["CC_0"],
        )

        self.tanque_D = TanqueTroncoConico(
            nome="D",
            raio_inferior=params.TANQUES_PROCESSO["D"]["raio_inferior"],
            raio_superior=params.TANQUES_PROCESSO["D"]["raio_superior"],
            altura_max=params.TANQUES_PROCESSO["D"]["altura_maxima"],
            kv_valvula=params.KV_VALVULAS["D"],
            kp_bomba_agua=params.KP_BOMBAS_AGUA["D"],
            kp_bomba_salmoura=params.KP_BOMBAS_SALMOURA["D"],
            nivel_inicial=params.CONDICOES_INICIAIS["hD_0"],
            concentracao_inicial=params.CONDICOES_INICIAIS["CD_0"],
        )

        self.tanque_E = TanqueTroncoConico(
            nome="E",
            raio_inferior=params.TANQUES_PROCESSO["E"]["raio_inferior"],
            raio_superior=params.TANQUES_PROCESSO["E"]["raio_superior"],
            altura_max=params.TANQUES_PROCESSO["E"]["altura_maxima"],
            kv_valvula=params.KV_VALVULAS["E"],
            kp_bomba_agua=params.KP_BOMBAS_AGUA["E"],
            kp_bomba_salmoura=params.KP_BOMBAS_SALMOURA["E"],
            nivel_inicial=params.CONDICOES_INICIAIS["hE_0"],
            concentracao_inicial=params.CONDICOES_INICIAIS["CE_0"],
        )

        # Dicionário para acesso fácil
        self.tanques_processo = {
            "C": self.tanque_C,
            "D": self.tanque_D,
            "E": self.tanque_E,
        }

        self.tanques_utilidades = {"A": self.tanque_A, "B": self.tanque_B}

    def atualizar_sistema(self, dt: float, controles: dict) -> dict:
        """
        Atualiza todos os tanques do sistema.

        Args:
            dt: passo de integração (s)
            controles: dicionário com comandos de controle
                {
                    'uA': float, 'uB': float,
                    'uC1': float, 'uC2': float, 'uC3': float,
                    'uD1': float, 'uD2': float, 'uD3': float,
                    'uE1': float, 'uE2': float, 'uE3': float
                }

        Returns:
            Dicionário com estados atualizados de todos os tanques
        """
        # Atualiza controles dos tanques de processo
        self.tanque_C.set_controles(
            controles["uC1"], controles["uC2"], controles["uC3"]
        )
        self.tanque_D.set_controles(
            controles["uD1"], controles["uD2"], controles["uD3"]
        )
        self.tanque_E.set_controles(
            controles["uE1"], controles["uE2"], controles["uE3"]
        )

        # Calcula demandas dos reservatórios (vazões de saída = somatório das bombas)
        Q_saida_A = (
            self.tanque_C.kp_agua * self.tanque_C.u1
            + self.tanque_D.kp_agua * self.tanque_D.u1
            + self.tanque_E.kp_agua * self.tanque_E.u1
        )

        Q_saida_B = (
            self.tanque_C.kp_salmoura * self.tanque_C.u2
            + self.tanque_D.kp_salmoura * self.tanque_D.u2
            + self.tanque_E.kp_salmoura * self.tanque_E.u2
        )

        # Vazões de entrada dos reservatórios (controladas por uA e uB)
        Q_entrada_A = params.KV_SUPRIMENTO_A * controles["uA"]
        Q_entrada_B = params.KV_SUPRIMENTO_B * controles["uB"]

        # Atualiza reservatórios
        self.tanque_A.set_vazoes(Q_entrada_A, Q_saida_A)
        self.tanque_B.set_vazoes(Q_entrada_B, Q_saida_B)

        hA, _ = self.tanque_A.atualizar(dt)
        hB, _ = self.tanque_B.atualizar(dt)

        # Atualiza tanques de processo
        hC, CC = self.tanque_C.atualizar(dt)
        hD, CD = self.tanque_D.atualizar(dt)
        hE, CE = self.tanque_E.atualizar(dt)

        # Retorna todos os estados
        return {
            "hA": hA,
            "hB": hB,
            "hC": hC,
            "CC": CC,
            "hD": hD,
            "CD": CD,
            "hE": hE,
            "CE": CE,
        }

    def get_estados(self) -> dict:
        """Retorna os estados atuais de todos os tanques."""
        return {
            "hA": self.tanque_A.nivel,
            "hB": self.tanque_B.nivel,
            "hC": self.tanque_C.nivel,
            "CC": self.tanque_C.concentracao,
            "hD": self.tanque_D.nivel,
            "CD": self.tanque_D.concentracao,
            "hE": self.tanque_E.nivel,
            "CE": self.tanque_E.concentracao,
        }

    def __repr__(self):
        return (
            f"\n{'='*70}\n"
            f"SISTEMA COMPLETO - Estado Atual\n"
            f"{'='*70}\n"
            f"Reservatórios:\n"
            f"  {self.tanque_A}\n"
            f"  {self.tanque_B}\n"
            f"Tanques de Processo:\n"
            f"  {self.tanque_C}\n"
            f"  {self.tanque_D}\n"
            f"  {self.tanque_E}\n"
            f"{'='*70}"
        )
