import numpy as np
from pydantic import BaseModel, Field
from typing import Literal, List

# --- SCHEMAS DE VALIDAÇÃO (Pydantic) ---

class DebtContractInput(BaseModel):
    id: str
    principal: float = Field(..., gt=0, description="Valor total captado da dívida")
    indexador: Literal["CDI", "IPCA", "PRE"]
    spread_anual: float = Field(..., ge=0, description="Spread ou taxa fixa anual (ex: 0.025 para 2.5%)")
    amortizacao: Literal["SAC", "PRICE", "BULLET"]
    prazo_meses: int = Field(..., gt=0, le=120, description="Prazo total do contrato em meses")
    carencia_meses: int = Field(0, ge=0, description="Meses de carência para o início da amortização do principal")


class CapitalStructureInput(BaseModel):
    equity: float = Field(..., ge=0, description="Montante de Capital Próprio investido")
    ke_anual: float = Field(..., gt=0, description="Custo de oportunidade do Capital Próprio (Taxa requerida)")
    dividas: List[DebtContractInput]

# --- MOTOR DE CÁLCULO FINANCEIRO ---

class DebtPortfolioEvaluator:
    def __init__(self, portfolio: CapitalStructureInput):
        self.portfolio = portfolio

    def project_contract_cash_flow(self, contract: DebtContractInput, macro_scenarios: dict) -> np.ndarray:
        """
        Projeta o Serviço da Dívida (Juros + Amortização) de um contrato específico
        para TODOS os cenários simulados de Monte Carlo simultaneamente.
        
        Output: np.ndarray de shape (n_simulations, n_months + 1)
        """
        n_simulations, n_months = macro_scenarios["selic"].shape
        n_months -= 1  # Ajuste para desconsiderar o mês 0
        
        # Matriz para guardar as saídas de caixa do serviço da dívida (Mês 0 é zero)
        debt_service = np.zeros((n_simulations, n_months + 1))
        
        # Inicializa matrizes auxiliares para controlar o Saldo Devedor de cada cenário
        saldo_devedor = np.zeros((n_simulations, n_months + 1))
        saldo_devedor[:, 0] = contract.principal
        
        # Converte o spread anual para mensal (taxa equivalente)
        rate_spread_mensal = (1 + contract.spread_anual) ** (1/12) - 1
        
        for t in range(1, n_months + 1):
            if t > contract.prazo_meses:
                break
                
            # 1. Determinar a taxa de juros do mês corrente para cada cenário
            if contract.indexador == "CDI":
                # Selic/CDI simulada para o mês convertido em taxa mensal equivalente
                rate_macro_mensal = (1 + macro_scenarios["selic"][:, t-1]) ** (1/12) - 1
                rate_total_mensal = rate_macro_mensal + rate_spread_mensal
            elif contract.indexador == "IPCA":
                # IPCA simulado para o mês convertido em taxa mensal equivalente
                rate_macro_mensal = (1 + macro_scenarios["ipca"][:, t-1]) ** (1/12) - 1
                rate_total_mensal = rate_macro_mensal + rate_spread_mensal
            else:  # PRE
                rate_total_mensal = rate_spread_mensal
                
            # 2. Atualiza o Saldo Devedor com a correção monetária/juros antes do pagamento
            juros_do_mes = saldo_devedor[:, t-1] * rate_total_mensal
            
            # 3. Cálculo da Amortização do Principal dependendo do sistema e da carência
            meses_decorridos = t
            meses_restantes_amortizacao = contract.prazo_meses - contract.carencia_meses
            
            amortizacao_do_mes = np.zeros(n_simulations)
            
            if meses_decorridos > contract.carencia_meses:
                if contract.amortizacao == "SAC":
                    # Amortização constante sobre o principal original
                    amortizacao_do_mes[:] = contract.principal / meses_restantes_amortizacao
                elif contract.amortizacao == "BULLET" and meses_decorridos == contract.prazo_meses:
                    # Todo o principal pago no último mês
                    amortizacao_do_mes[:] = contract.principal
                elif contract.amortizacao == "PRICE":
                    # Fórmula da tabela PRICE estocástica adaptada para o saldo remanescente
                    n_p = contract.prazo_meses - meses_decorridos + 1
                    # Evita divisão por zero caso a taxa zere
                    pmt = np.where(
                        rate_total_mensal > 0,
                        saldo_devedor[:, t-1] * (rate_total_mensal * (1 + rate_total_mensal)**n_p) / ((1 + rate_total_mensal)**n_p - 1),
                        saldo_devedor[:, t-1] / n_p
                    )
                    amortizacao_do_mes = pmt - juros_do_mes
            
            # Ajuste de segurança para não amortizar mais do que o saldo devedor existente
            amortizacao_do_mes = np.minimum(amortizacao_do_mes, saldo_devedor[:, t-1])
            
            # 4. Total do Serviço da Dívida e Atualização do Saldo Devedor Final do Mês
            debt_service[:, t] = juros_do_mes + amortizacao_do_mes
            saldo_devedor[:, t] = saldo_devedor[:, t-1] - amortizacao_do_mes
            
        return debt_service

    def project_all_portfolio(self, macro_scenarios: dict) -> np.ndarray:
        """Soma o fluxo de caixa de todas as dívidas do passivo"""
        n_simulations, n_months = macro_scenarios["selic"].shape
        total_portfolio_service = np.zeros((n_simulations, n_months))
        
        for debt in self.portfolio.dividas:
            total_portfolio_service += self.project_contract_cash_flow(debt, macro_scenarios)
            
        return total_portfolio_service