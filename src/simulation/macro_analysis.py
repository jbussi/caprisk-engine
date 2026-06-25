import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from pydantic import BaseModel

class IndexWeightConfig(BaseModel):
    peso_ipca: float
    peso_igpm: float
    # Possibilidade de expandir para outros índices setoriais (INCC, etc.)

class InflationIndexBuilder:
    """Constroi o índice de inflação personalizado para deflacionar o caixa da empresa"""
    def __init__(self, config: IndexWeightConfig):
        self.config = config

    def calculate_custom_index(self, matrix_ipca: np.ndarray, matrix_igpm: np.ndarray) -> np.ndarray:
        """Combina as matrizes estocásticas de inflação no índice customizado"""
        return (matrix_ipca * self.config.peso_ipca) + (matrix_igpm * self.config.peso_igpm)


class MacroTimeSeriesFitter:
    """Modelo VAR para estimar relações macroeconômicas e projetar bandas de estresse"""
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.model = None
        self.results = None

    def fit(self, history_data: pd.DataFrame):
        """
        Treina o modelo de séries temporais com dados históricos.
        history_data deve conter colunas: ['selic', 'ipca', 'igpm', 'desemprego']
        """
        # Alta interpretabilidade: O VAR encontra automaticamente as defasagens ideais (AIC/BIC)
        self.model = VAR(history_data)
        self.results = self.model.fit(maxlags=3, ic='aic')
        print(self.results.summary()) # Transparência total dos coeficientes econômicos

    def generate_stress_bounds(self, last_observations: np.ndarray, steps: int) -> dict:
        """
        Em vez de gerar trajetórias aleatórias puras, gera os limites matemáticos
        do intervalo de confiança (ex: 95%) para servirem como cenários de estresse.
        """
        # forecast_interval calcula analiticamente os limites superior e inferior da projeção
        alpha = 1.0 - self.confidence_level
        mid, lower, upper = self.results.forecast_interval(last_observations, steps=steps, alpha=alpha)
        
        # O retorno nos dá o "Pior Cenário Plausível" (Cenário de Cauda) para cada variável
        return {
            "cenario_medio": mid,
            "estresse_superior": upper, # Ex: Selic e Inflação explodindo
            "estresse_inferior": lower  # Ex: Recessão/Deflação
        }