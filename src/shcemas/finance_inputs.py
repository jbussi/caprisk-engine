from pydantic import BaseModel, Field
from typing import List, Optional
from src.finance.metrics import DebtContractInput # Importa o que já criamos para o passivo

class OperationalInputs(BaseModel):
    receita_anual_base: float = Field(..., gt=0, description="Receita bruta dos últimos 12 meses (LTM)")
    margem_ebitda_base: float = Field(..., gt=0, lt=1, description="Margem EBITDA atual (ex: 0.20 para 20%)")
    
    # Sensibilidades Macroeconômicas
    elasticidade_pib: float = Field(1.0, description="Sensibilidade da receita ao PIB. >1 significa setor cíclico")
    repasse_ipca: float = Field(0.8, ge=0, le=1.5, description="Capacidade de repassar inflação aos preços (1.0 = repasse integral)")
    
    # Risco Intrínseco do Negócio
    volatilidade_operacional: float = Field(..., gt=0, lt=0.5, description="Desvio padrão do choque operacional anual")
    
    # Alíquota de Impostos para benefício fiscal
    aliquota_imposto: float = Field(0.34, ge=0, le=0.5, description="Alíquota efetiva de IR/CSLL (padrão 34% para Lucro Real)")

class CompanySimulationRequest(BaseModel):
    """
    Payload completo que a API receberá para rodar a otimização
    unindo Ativos (Operacional) e Passivos (Estrutura de Capital).
    """
    nome_empresa: str
    cenarios_monte_carlo: int = Field(10000, ge=1000, le=50000, description="Número de caminhos a simular")
    prazo_simulacao_meses: int = Field(36, ge=12, le=60, description="Horizonte preditivo")
    
    operacional: OperationalInputs
    equity_atual: float = Field(..., ge=0, description="Capital próprio atual na mesa")
    ke_proposto: float = Field(..., gt=0, description="Retorno requerido pelo acionista (Custo de Equity)")
    dividas_propostas: List[DebtContractInput]