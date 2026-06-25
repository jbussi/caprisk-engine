import numpy as np
from src.schemas.finance_inputs import CompanySimulationRequest
from src.simulation.macro_models import MacroSimulator
from src.finance.metrics import DebtPortfolioEvaluator

class CapriskEngine:
    def __init__(self, request: CompanySimulationRequest):
        self.request = request
        self.macro_sim = MacroSimulator(
            n_simulations=request.cenarios_monte_carlo, 
            n_months=request.prazo_simulacao_meses
        )
        self.debt_evaluator = DebtPortfolioEvaluator(
            portfolio=request  # O request contém a lista de dívidas e o equity
        )

    def run_simulation(self, current_selic: float, current_ipca: float) -> dict:
        """
        Executa a simulação integrada de ALM e apura indicadores operacionais e financeiros.
        Tudo calculado de forma vetorizada em moeda nominal, preparando para deflação real posterior.
        """
        n_sim = self.request.cenarios_monte_carlo
        n_months = self.request.prazo_simulacao_meses
        dt = 1 / 12

        # 1. Gera os cenários das taxas de mercado (Selic e IPCA)
        macro_scenarios = self.macro_sim.simulate(r0=current_selic, i0=current_ipca)
        
        # 2. Gera os fluxos de caixa de TODOS os passivos (Serviço da Dívida acumulado por cenário/mês)
        # Shape: (n_simulations, n_months + 1)
        debt_service_matrix = self.debt_evaluator.project_all_portfolio(macro_scenarios)

        # 3. Inicializa matrizes para o Ativo / Operacional
        ebitda_matrix = np.zeros((n_sim, n_months + 1))
        lucro_liquido_matrix = np.zeros((n_sim, n_months + 1))
        caixa_acumulado_matrix = np.zeros((n_sim, n_months + 1))

        # Condições iniciais (Mês 0)
        receita_mensal_base = self.request.operacional.receita_anual_base / 12
        ebitda_inicial = receita_mensal_base * self.request.operacional.margem_ebitda_base
        
        ebitda_matrix[:, 0] = ebitda_inicial
        caixa_acumulado_matrix[:, 0] = self.request.equity_atual  # Ponto de partida de liquidez

        # Choques operacionais intrínsecos independentes da macroeconomia para cada mês
        vol_op_mensal = self.request.operacional.volatilidade_operacional * np.sqrt(dt)
        choques_operacionais = np.random.normal(0, vol_op_mensal, (n_sim, n_months + 1))

        # 4. Evolução Temporal da Empresa (Mês a Mês)
        for t in range(1, n_months + 1):
            # Índices de inflação e juros do período anterior para o reajuste
            ipca_simulado = macro_scenarios["ipca"][:, t-1]
            selic_simulada = macro_scenarios["selic"][:, t-1]

            # Evolução do EBITDA considerando repasse de inflação e choque do negócio
            # Em versões futuras, o PIB/Demanda entrará explicitamente aqui multiplicando a elasticidade
            fator_reajuste = 1 + (ipca_simulado * self.request.operacional.repasse_ipca * dt)
            choque_negocio = np.exp(choques_operacionais[:, t])
            
            ebitda_matrix[:, t] = ebitda_matrix[:, t-1] * fator_reajuste * choque_negocio

            # Serviço da dívida cobrado neste mês específico
            servico_divida_t = debt_service_matrix[:, t]

            # Lucro Líquido Simplificado (EBITDA - Serviço da Dívida) antes de impostos
            # Nota: O WACC dinâmico e o benefício fiscal da dívida (tax shield) incidirão aqui nas próximas versões
            lucro_liquido_matrix[:, t] = ebitda_matrix[:, t] - servico_divida_t

            # Dinâmica do Caixa Líquido Acumulado (Moeda Nominal)
            caixa_acumulado_matrix[:, t] = caixa_acumulado_matrix[:, t-1] + lucro_liquido_matrix[:, t]

        # 5. Apuração Estatística de Insolvência e Eficiência
        # Uma empresa entra em insolvência técnica se o caixa acumulado quebra a barreira de zero
        passou_pela_insolvencia = np.any(caixa_acumulado_matrix < 0, axis=1)
        probabilidade_insolvencia = np.mean(passou_pela_insolvencia)

        # Mapeamento dos cenários de quebra para análise de estresse
        # Captura as taxas médias que causaram a ruína da empresa
        taxas_selic_na_quebra = macro_scenarios["selic"][passou_pela_insolvencia, :]
        selic_critica_media = np.mean(taxas_selic_na_quebra) if np.sum(passou_pela_insolvencia) > 0 else None

        return {
            "probabilidade_insolvencia": float(probabilidade_insolvencia),
            "selic_critica_media": float(selic_critica_media) if selic_critica_media else "Nenhum cenário de quebra",
            "ebitda_medio_final": float(np.mean(ebitda_matrix[:, -1])),
            "lucro_liquido_medio_final": float(np.mean(lucro_liquido_matrix[:, -1])),
            "matrizes": {
                "caixa": caixa_acumulado_matrix,
                "selic": macro_scenarios["selic"],
                "ipca": macro_scenarios["ipca"]
            }
        }