"""
Orquestra a simulação completa do sistema de 5 tanques com controle MPC.

Este arquivo integra:
- Modelo fenomenológico não-linear dos tanques (modelo_tanques.py)
- Controlador MPC multivarável (controlador_mpc.py)
- Cenários de teste programados (cenarios_teste.py)
- Parâmetros do sistema (parametros_sistema.py)

A simulação é do tipo OFFLINE/PROGRAMADA:
- Define intervalo de tempo total
- Aplica degraus em setpoints em instantes específicos
- Gera histórico completo de estados e controles
- Não opera em tempo real
"""

import numpy as np
from typing import Dict, Tuple
import time
from tqdm import tqdm  # Barra de progresso (opcional)

# Módulos do projeto
import parametros_sistema as params
from modelo_tanques import SistemaCompleto
from controlador_mpc import SistemaControle
import cenarios_teste


# ==============================================================================
# CLASSE PRINCIPAL DE SIMULAÇÃO
# ==============================================================================


class SimuladorMPC:
    """
    Classe que encapsula toda a lógica de simulação do sistema MPC.
    """

    def __init__(
        self,
        tempo_total: float,
        dt_integracao: float,
        Ts_controlador: float,
        numero_cenario: int = 1,
    ):
        """
        Inicializa o simulador.

        Args:
            tempo_total: duração total da simulação (s)
            dt_integracao: passo de integração do modelo (s)
            Ts_controlador: período de amostragem do MPC (s)
            numero_cenario: cenário de teste a executar (1-3)
        """
        self.controles_atuais = {}
        self.tempo_total = tempo_total
        self.dt = dt_integracao
        self.Ts = Ts_controlador

        # Valida que Ts é múltiplo de dt
        if not np.isclose(self.Ts % self.dt, 0.0):
            raise ValueError(
                f"Ts ({Ts_controlador}) deve ser múltiplo de dt ({dt_integracao})"
            )

        self.passos_controle = int(self.Ts / self.dt)

        # Vetores de tempo
        self.n_passos = int(self.tempo_total / self.dt) + 1
        self.tempo = np.arange(0, self.tempo_total + self.dt, self.dt)

        # Carrega cenário de teste
        print(f"\n{'='*70}")
        print("INICIALIZANDO SIMULAÇÃO MPC")
        print(f"{'='*70}")
        print(f"Tempo total: {tempo_total}s ({tempo_total/60:.1f} min)")
        print(f"Passo de integração: {dt_integracao}s")
        print(f"Período de amostragem MPC: {Ts_controlador}s")
        print(f"Número de passos: {self.n_passos}")

        self.cenario = cenarios_teste.obter_cenario(
            numero_cenario, tempo_total, self.dt
        )
        print(f"\nCenário: {self.cenario['nome']}")
        print(f"Descrição: {self.cenario['descricao']}")

        # Inicializa sistema físico e controlador
        print(f"\n{'='*70}")
        self.sistema = SistemaCompleto()
        self.controlador = SistemaControle(Ts_controlador)
        print(f"{'='*70}\n")

        # Históricos para armazenar dados da simulação
        self._inicializar_historicos()

        # Controles iniciais (ponto de operação)
        self.controles_atuais = {
            "uA": params.CONDICOES_INICIAIS["uA_0"],
            "uB": params.CONDICOES_INICIAIS["uB_0"],
            "uC1": params.CONDICOES_INICIAIS["uC1_0"],
            "uC2": params.CONDICOES_INICIAIS["uC2_0"],
            "uC3": params.CONDICOES_INICIAIS["uC3_0"],
            "uD1": params.CONDICOES_INICIAIS["uD1_0"],
            "uD2": params.CONDICOES_INICIAIS["uD2_0"],
            "uD3": params.CONDICOES_INICIAIS["uD3_0"],
            "uE1": params.CONDICOES_INICIAIS["uE1_0"],
            "uE2": params.CONDICOES_INICIAIS["uE2_0"],
            "uE3": params.CONDICOES_INICIAIS["uE3_0"],
        }

        # Contador de acionamentos do MPC
        self.contador_mpc = 0

    def _inicializar_historicos(self):
        """Cria estruturas para armazenar históricos de estados e controles."""
        n = self.n_passos

        # Histórico de estados
        self.hist_estados = {
            "tempo": self.tempo.copy(),
            "hA": np.zeros(n),
            "hB": np.zeros(n),
            "hC": np.zeros(n),
            "CC": np.zeros(n),
            "hD": np.zeros(n),
            "CD": np.zeros(n),
            "hE": np.zeros(n),
            "CE": np.zeros(n),
        }

        # Histórico de controles
        self.hist_controles = {
            "uA": np.zeros(n),
            "uB": np.zeros(n),
            "uC1": np.zeros(n),
            "uC2": np.zeros(n),
            "uC3": np.zeros(n),
            "uD1": np.zeros(n),
            "uD2": np.zeros(n),
            "uD3": np.zeros(n),
            "uE1": np.zeros(n),
            "uE2": np.zeros(n),
            "uE3": np.zeros(n),
        }

        # Histórico de referências
        self.hist_referencias = {
            "hC_ref": self.cenario["hC_ref"].copy(),
            "CC_ref": self.cenario["CC_ref"].copy(),
            "hD_ref": self.cenario["hD_ref"].copy(),
            "CD_ref": self.cenario["CD_ref"].copy(),
            "hE_ref": self.cenario["hE_ref"].copy(),
            "CE_ref": self.cenario["CE_ref"].copy(),
        }

        # Armazena estados iniciais
        estados_iniciais = self.sistema.get_estados()
        for chave, valor in estados_iniciais.items():
            self.hist_estados[chave][0] = valor

        # Armazena controles iniciais
        for chave, valor in self.controles_atuais.items():
            self.hist_controles[chave][0] = valor

    def executar(self, verbose: bool = True, usar_barra_progresso: bool = True):
        """
        Executa a simulação completa.

        Args:
            verbose: se True, imprime informações durante a simulação
            usar_barra_progresso: se True, exibe barra de progresso

        Returns:
            Dicionário com históricos de estados, controles e referências
        """
        print(f"{'='*70}")
        print("INICIANDO SIMULAÇÃO")
        print(f"{'='*70}\n")

        tempo_inicio = time.time()

        # Configura barra de progresso
        iterador = range(1, self.n_passos)
        if usar_barra_progresso:
            try:
                iterador = tqdm(
                    iterador, desc="Simulação", unit="passo", ncols=80, colour="green"
                )
            except:
                pass  # Se tqdm não disponível, usa range normal

        # Loop principal de simulação
        for k in iterador:
            tempo_atual = k * self.dt

            # ===== CONTROLE MPC (executado a cada Ts) =====
            if k % self.passos_controle == 0:
                # Lê estados atuais
                estados_atuais = self.sistema.get_estados()

                # Lê referências atuais do cenário
                referencias_atuais = {
                    "hC_ref": self.hist_referencias["hC_ref"][k],
                    "CC_ref": self.hist_referencias["CC_ref"][k],
                    "hD_ref": self.hist_referencias["hD_ref"][k],
                    "CD_ref": self.hist_referencias["CD_ref"][k],
                    "hE_ref": self.hist_referencias["hE_ref"][k],
                    "CE_ref": self.hist_referencias["CE_ref"][k],
                }

                # Calcula ações de controle via MPC
                self.controles_atuais = self.controlador.calcular_acoes(
                    estados_atuais, referencias_atuais
                )

                self.contador_mpc += 1

                # Verbose (a cada 10 acionamentos do MPC)
                if verbose and self.contador_mpc % 10 == 0:
                    self._imprimir_status(
                        tempo_atual, estados_atuais, referencias_atuais
                    )

            # ===== ATUALIZAÇÃO DO MODELO (a cada dt) =====
            estados_novos = self.sistema.atualizar_sistema(
                self.dt, self.controles_atuais
            )

            # ===== ARMAZENAMENTO DE DADOS =====
            for chave, valor in estados_novos.items():
                self.hist_estados[chave][k] = valor

            for chave, valor in self.controles_atuais.items():
                self.hist_controles[chave][k] = valor

        # Finalização
        tempo_fim = time.time()
        tempo_execucao = tempo_fim - tempo_inicio

        print(f"\n{'='*70}")
        print("SIMULAÇÃO CONCLUÍDA")
        print(f"{'='*70}")
        print(f"Tempo de execução: {tempo_execucao:.2f}s")
        print(f"Acionamentos do MPC: {self.contador_mpc}")
        print(f"Tempo simulado: {self.tempo_total}s ({self.tempo_total/60:.1f} min)")
        print(f"{'='*70}\n")

        self.verificar_integridade_historicos()
        return self.obter_resultados()

    def _imprimir_status(self, tempo: float, estados: dict, referencias: dict):
        """Imprime status da simulação (chamado periodicamente)."""
        print(f"\n[t={tempo:6.1f}s] Estados:")
        print(
            f"  Tanque C: h={estados['hC']:.3f}m (ref={referencias['hC_ref']:.3f}), "
            f"C={estados['CC']:.1f}kg/m³ (ref={referencias['CC_ref']:.1f})"
        )
        print(
            f"  Tanque D: h={estados['hD']:.3f}m (ref={referencias['hD_ref']:.3f}), "
            f"C={estados['CD']:.1f}kg/m³ (ref={referencias['CD_ref']:.1f})"
        )
        print(
            f"  Tanque E: h={estados['hE']:.3f}m (ref={referencias['hE_ref']:.3f}), "
            f"C={estados['CE']:.1f}kg/m³ (ref={referencias['CE_ref']:.1f})"
        )

    def obter_resultados(self) -> Dict:
        """
        Retorna todos os dados da simulação.

        Returns:
            Dicionário contendo históricos, parâmetros e metadados
        """
        return {
            "estados": self.hist_estados,
            "controles": self.hist_controles,
            "referencias": self.hist_referencias,
            "parametros": {
                "tempo_total": self.tempo_total,
                "dt": self.dt,
                "Ts": self.Ts,
                "cenario": self.cenario["nome"],
            },
            "sistema": self.sistema,
            "controlador": self.controlador,
        }

    def verificar_integridade_historicos(self):
        n = self.n_passos
        for nome, arr in self.hist_estados.items():
            assert len(arr) == n, f"Histórico de estado {nome} desalinhado"
        for nome, arr in self.hist_controles.items():
            assert len(arr) == n, f"Histórico de controle {nome} desalinhado"
        for nome, arr in self.hist_referencias.items():
            assert len(arr) == n, f"Histórico de referência {nome} desalinhado"


# ==============================================================================
# FUNÇÃO AUXILIAR: EXECUÇÃO RÁPIDA
# ==============================================================================


def executar_simulacao(
    numero_cenario: int = 1, tempo_total: float = None, verbose: bool = True
) -> Dict:
    """
    Função auxiliar para executar simulação de forma simplificada.

    Args:
        numero_cenario: número do cenário de teste (1-6)
        tempo_total: duração da simulação (s). Se None, usa padrão dos parâmetros
        verbose: se True, imprime informações durante execução

    Returns:
        Dicionário com resultados da simulação
    """
    # Usa parâmetros padrão
    if tempo_total is None:
        tempo_total = params.TEMPO_TOTAL

    dt = params.DT_INTEGRACAO
    Ts = params.TS_CONTROLADOR

    # Cria e executa simulador
    simulador = SimuladorMPC(tempo_total, dt, Ts, numero_cenario)
    resultados = simulador.executar(verbose=verbose, usar_barra_progresso=True)

    return resultados
