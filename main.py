"""
main.py

Ponto de entrada principal para a simulação do sistema de controle MPC
de tanques tronco-cônicos.

Este arquivo fornece uma interface simples e interativa para:
- Selecionar e executar cenários de teste
- Visualizar resultados
- Gerar relatórios de desempenho
- Exportar dados para análise posterior

Uso:
    python main.py

Projeto: Sistema de Controle MPC para Tanques Tronco-Cônicos
Disciplina: Técnicas de Controle de Processos Industriais
Baseado na Entrega 3 - Proposta da Estrutura de Controle
"""

import os
import sys
from typing import Optional
import warnings

# Suprime warnings de otimização do CVXPY (opcional)
warnings.filterwarnings("ignore")

# Importa módulos do projeto
import parametros_sistema as params
import cenarios_teste
from simulacao_principal import executar_simulacao, executar_todos_cenarios
from visualizacao_resultados import (
    plotar_niveis,
    plotar_concentracoes,
    plotar_controles,
    plotar_resumo_completo,
    analisar_desempenho,
    exportar_para_csv,
)


# ==============================================================================
# BANNER E INTERFACE
# ==============================================================================


def exibir_banner():
    """Exibe banner de apresentação do sistema."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║        SIMULAÇÃO DE CONTROLE MPC PARA TANQUES TRONCO-CÔNICOS        ║
    ║                                                                      ║
    ║  Sistema de 5 tanques (2 cilíndricos + 3 tronco-cônicos)            ║
    ║  Controle MPC multivarável de nível e concentração                  ║
    ║                                                                      ║
    ║  Disciplina: Técnicas de Controle de Processos Industriais          ║
    ║  Universidade Federal de Minas Gerais - UFMG                        ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def exibir_menu_principal():
    """Exibe menu de opções do sistema."""
    print("\n" + "=" * 70)
    print("MENU PRINCIPAL")
    print("=" * 70)
    print("\n1. Executar cenário específico")
    print("2. Executar todos os cenários (1-6)")
    print("3. Listar cenários disponíveis")
    print("4. Configurar parâmetros de simulação")
    print("5. Sobre o sistema")
    print("0. Sair")
    print("\n" + "=" * 70)


def listar_cenarios_menu():
    """Lista cenários disponíveis de forma interativa."""
    cenarios_teste.listar_cenarios()
    input("\nPressione ENTER para voltar ao menu...")


def exibir_sobre():
    """Exibe informações sobre o sistema."""
    print("\n" + "=" * 70)
    print("SOBRE O SISTEMA")
    print("=" * 70)
    print(f"\n{params.INFO_SISTEMA['descricao']}")
    print(f"\nVersão: {params.INFO_SISTEMA['versao']}")
    print(f"Data: {params.INFO_SISTEMA['data']}")
    print(f"Disciplina: {params.INFO_SISTEMA['disciplina']}")
    print("\nArquitetura do Sistema:")
    print("  - parametros_sistema.py: Parâmetros físicos e de controle")
    print("  - modelo_tanques.py: Modelo fenomenológico não-linear")
    print("  - controlador_mpc.py: Controlador MPC com otimização convexa")
    print("  - cenarios_teste.py: Cenários de validação R1-R4")
    print("  - simulacao_principal.py: Orquestrador da simulação")
    print("  - visualizacao_resultados.py: Análise e gráficos")
    print("  - main.py: Interface de usuário")
    print("\nConstantes de Tempo:")
    print(
        f"  - Nível (τ_h): {params.CONSTANTES_TEMPO['tau_h']}s "
        f"(~{params.CONSTANTES_TEMPO['tau_h']/60:.1f} min)"
    )
    print(
        f"  - Concentração (τ_C): {params.CONSTANTES_TEMPO['tau_C']}s "
        f"(~{params.CONSTANTES_TEMPO['tau_C']/60:.1f} min)"
    )
    print("\nHorizontes MPC:")
    print(
        f"  - Predição (Np): {params.MPC_HORIZONTES['Np']} amostras "
        f"({params.MPC_HORIZONTES['Np']*params.TS_CONTROLADOR}s)"
    )
    print(
        f"  - Controle (Nc): {params.MPC_HORIZONTES['Nc']} amostras "
        f"({params.MPC_HORIZONTES['Nc']*params.TS_CONTROLADOR}s)"
    )
    print("\n" + "=" * 70)
    input("\nPressione ENTER para voltar ao menu...")


# ==============================================================================
# FUNÇÕES DE EXECUÇÃO
# ==============================================================================


def executar_cenario_especifico():
    """Executa um cenário específico escolhido pelo usuário."""
    print("\n" + "=" * 70)
    print("EXECUÇÃO DE CENÁRIO ESPECÍFICO")
    print("=" * 70)

    # Lista cenários
    cenarios_teste.listar_cenarios()

    # Solicita escolha
    try:
        num_cenario = int(input("\nEscolha o número do cenário (1-6): "))
        if num_cenario < 1 or num_cenario > 6:
            print("❌ Número de cenário inválido!")
            return
    except ValueError:
        print("❌ Entrada inválida!")
        return

    # Solicita tempo de simulação
    usar_padrao = (
        input(f"\nUsar tempo padrão ({params.TEMPO_TOTAL}s)? (S/n): ").strip().lower()
    )

    if usar_padrao in ["n", "nao", "não"]:
        try:
            tempo_total = float(input("Digite o tempo total de simulação (s): "))
        except ValueError:
            print("❌ Valor inválido! Usando tempo padrão.")
            tempo_total = params.TEMPO_TOTAL
    else:
        tempo_total = params.TEMPO_TOTAL

    # Executa simulação
    print(f"\n{'='*70}")
    print(f"EXECUTANDO CENÁRIO {num_cenario}")
    print(f"{'='*70}\n")

    resultados = executar_simulacao(
        numero_cenario=num_cenario, tempo_total=tempo_total, verbose=False
    )

    # Menu de pós-processamento
    pos_processar_resultados(resultados, num_cenario)


def executar_todos_cenarios_menu():
    """Executa todos os cenários disponíveis."""
    print("\n" + "=" * 70)
    print("EXECUÇÃO DE TODOS OS CENÁRIOS")
    print("=" * 70)

    # Solicita tempo de simulação
    usar_padrao = (
        input(f"\nUsar tempo padrão ({params.TEMPO_TOTAL}s) para todos? (S/n): ")
        .strip()
        .lower()
    )

    if usar_padrao in ["n", "nao", "não"]:
        try:
            tempo_total = float(input("Digite o tempo total de simulação (s): "))
        except ValueError:
            print("❌ Valor inválido! Usando tempo padrão.")
            tempo_total = params.TEMPO_TOTAL
    else:
        tempo_total = params.TEMPO_TOTAL

    # Confirmação
    confirmar = (
        input(
            f"\nIsto executará 6 simulações (~{6*tempo_total/60:.1f} min). Continuar? (S/n): "
        )
        .strip()
        .lower()
    )

    if confirmar in ["n", "nao", "não"]:
        print("Operação cancelada.")
        return

    # Executa todos os cenários
    resultados_todos = executar_todos_cenarios(tempo_total=tempo_total, verbose=False)

    print("\n" + "=" * 70)
    print("RESUMO DE DESEMPENHO - TODOS OS CENÁRIOS")
    print("=" * 70)

    # Analisa cada cenário
    for i in range(1, 7):
        print(f"\n>>> Cenário {i} <<<")
        metricas = analisar_desempenho(resultados_todos[i], verbose=False)

        # Exibe apenas status de conformidade
        print("Requisitos:")
        for req in ["R1", "R2", "R3", "R4"]:
            status = "✓" if metricas["conformidade"][req]["status"] else "✗"
            print(f"  {req}: {status}")

    print("\n" + "=" * 70)

    # Opção de salvar
    salvar = (
        input("\nDeseja exportar todos os resultados para CSV? (s/N): ").strip().lower()
    )
    if salvar in ["s", "sim"]:
        for i in range(1, 7):
            exportar_para_csv(resultados_todos[i], f"cenario_{i}_resultados.csv")
        print("✓ Todos os resultados foram exportados!")

    input("\nPressione ENTER para voltar ao menu...")


def pos_processar_resultados(resultados: dict, num_cenario: int):
    """Menu de pós-processamento após simulação."""
    while True:
        print("\n" + "=" * 70)
        print("PÓS-PROCESSAMENTO DOS RESULTADOS")
        print("=" * 70)
        print("\n1. Visualizar gráficos de níveis")
        print("2. Visualizar gráficos de concentrações")
        print("3. Visualizar gráficos de controles")
        print("4. Visualizar resumo completo")
        print("5. Gerar relatório de desempenho")
        print("6. Exportar dados para CSV")
        print("7. Executar todas as análises")
        print("0. Voltar ao menu principal")
        print("\n" + "=" * 70)

        opcao = input("\nEscolha uma opção: ").strip()

        if opcao == "1":
            plotar_niveis(resultados)
        elif opcao == "2":
            plotar_concentracoes(resultados)
        elif opcao == "3":
            plotar_controles(resultados)
        elif opcao == "4":
            plotar_resumo_completo(resultados)
        elif opcao == "5":
            analisar_desempenho(resultados, verbose=True)
            input("\nPressione ENTER para continuar...")
        elif opcao == "6":
            nome_arquivo = f"cenario_{num_cenario}_resultados.csv"
            exportar_para_csv(resultados, nome_arquivo)
            input("\nPressione ENTER para continuar...")
        elif opcao == "7":
            print("\nExecutando todas as análises...\n")
            plotar_resumo_completo(resultados)
            analisar_desempenho(resultados, verbose=True)
            exportar_para_csv(resultados, f"cenario_{num_cenario}_resultados.csv")
            input("\nPressione ENTER para continuar...")
        elif opcao == "0":
            break
        else:
            print("❌ Opção inválida!")


# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================


def main():
    """Função principal do programa."""
    # Limpa terminal (opcional)
    os.system("cls" if os.name == "nt" else "clear")

    # Exibe banner
    exibir_banner()

    # Loop principal do menu
    while True:
        exibir_menu_principal()

        opcao = input("\nEscolha uma opção: ").strip()

        if opcao == "1":
            executar_cenario_especifico()
        elif opcao == "2":
            executar_todos_cenarios_menu()
        elif opcao == "3":
            listar_cenarios_menu()
        elif opcao == "4":
            print(
                "\n⚠️  Configuração de parâmetros deve ser feita editando parametros_sistema.py"
            )
            input("\nPressione ENTER para continuar...")
        elif opcao == "5":
            exibir_sobre()
        elif opcao == "0":
            print("\n" + "=" * 70)
            print("Encerrando o sistema. Até logo!")
            print("=" * 70 + "\n")
            sys.exit(0)
        else:
            print("\n❌ Opção inválida! Tente novamente.")


# ==============================================================================
# EXECUÇÃO DIRETA
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário (Ctrl+C)")
        print("Encerrando o sistema...\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        print("Por favor, verifique os arquivos de configuração e dependências.")
        import traceback

        traceback.print_exc()
        sys.exit(1)
