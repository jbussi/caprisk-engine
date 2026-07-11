import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Optional


class PredictorConfig(BaseModel):
    """Configuração estrita de parâmetros para o motor econométrico."""
    max_lags: int = Field(default=3, ge=1, description="Número máximo de defasagens (lags) testadas pelo AIC.")
    horizonte_projeção_meses: int = Field(default=36, ge=1, description="Prazo da projeção futura do fluxo de caixa.")
    significancia_banda: float = Field(default=0.05, description="Nível de significância alfa (0.05 = 95% de confiança).")


class MacroPredictorEngine:
    """
    Motor econométrico baseado em Vetores Autorregressivos (VAR).
    Responsável por projetar cenários macroeconômicos e validar a cobertura 
    estatística dos intervalos de confiança para gestão de risco de capital.
    """
    def __init__(self, config: Optional[PredictorConfig] = None):
        self.config = config or PredictorConfig()
        self.model_fitted = None
        self.var_names: List[str] = []
        self.lags_otimos: int = 1

    def selecionar_e_ajustar_modelo(self, df_universo: pd.DataFrame) -> List[str]:
        """
        Executa a triagem de lags via AIC e ajusta o modelo VAR definitivo.
        Garante a integridade dos tipos numéricos antes do ajuste.
        """
        df_trabalho = df_universo[df_universo.index >= '1994-08-01'].copy()
        
        for col in df_trabalho.columns:
            df_trabalho[col] = pd.to_numeric(df_trabalho[col], errors='coerce')
            
        df_trabalho = df_trabalho.dropna(how='all', axis=1).dropna()
        df_trabalho.columns = [str(col).strip().replace('\r', '').replace('\n', '') for col in df_trabalho.columns]
        
        self.var_names = list(df_trabalho.columns)
        
        if len(self.var_names) == 0:
            raise ValueError("Erro Crítico: Todas as colunas do DataFrame foram descartadas por não serem numéricas.")
        
        model = VAR(df_trabalho)
        
        selecao_lag = model.select_order(maxlags=self.config.max_lags)
        self.lags_otimos = selecao_lag.selected_orders['aic']
        if self.lags_otimos == 0: 
            self.lags_otimos = 1 
        
        self.model_fitted = model.fit(maxlags=self.lags_otimos)
        return self.var_names
    
    def gerar_projeções_estresse(self, df_historico: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Gera cenário base e as bandas analíticas usando a covariância dos resíduos."""
        if self.model_fitted is None:
            raise ValueError("O modelo precisa ser ajustado antes de projetar.")
            
        # Garante o mesmo tratamento de limpeza e colunas string que o fit aplicou
        df_trabalho = df_historico.copy()
        for col in df_trabalho.columns:
            df_trabalho[col] = pd.to_numeric(df_trabalho[col], errors='coerce')
        df_trabalho = df_trabalho.dropna(how='all', axis=1).dropna()
        df_trabalho.columns = [str(col).strip().replace('\r', '').replace('\n', '') for col in df_trabalho.columns]
        
        valores_iniciais = df_trabalho.values[-self.lags_otimos:]
        
        projeção_bruta = self.model_fitted.forecast(y=valores_iniciais, steps=self.config.horizonte_projeção_meses)
        
        erros_projeção = self.model_fitted.forecast_interval(
            y=valores_iniciais, 
            steps=self.config.horizonte_projeção_meses, 
            alpha=self.config.significancia_banda
        )
        projeção_inferior, _, projeção_superior = erros_projeção
        
        index_futuro = pd.date_range(
            start=df_trabalho.index[-1] + pd.DateOffset(months=1), 
            periods=self.config.horizonte_projeção_meses, 
            freq='MS'
        )
        
        cenarios_por_variavel = {}
        for i, nome_var in enumerate(self.var_names):
            df_var_cenarios = pd.DataFrame(index=index_futuro)
            df_var_cenarios["CENARIO_BASE"] = projeção_bruta[:, i]
            df_var_cenarios["CENARIO_ESTRESSE_SUPERIOR"] = projeção_superior[:, i]
            df_var_cenarios["CENARIO_ESTRESSE_INFERIOR"] = projeção_inferior[:, i]
            
            df_var_cenarios = df_var_cenarios.clip(lower=-0.02)
            cenarios_por_variavel[nome_var] = df_var_cenarios
            
        return cenarios_por_variavel

    def rodar_teste_ruido_branco(self) -> dict:
        """Valida se os resíduos do VAR são ruído branco (Portmanteau Test)."""
        if self.model_fitted is None:
            raise ValueError("O modelo precisa ser ajustado primeiro.")
        
        # O statsmodels não aceita o parâmetro 'significance' aqui
        teste = self.model_fitted.test_whiteness(nlags=10)
        
        return {
            "estatistica": teste.test_statistic,
            "p_valor": teste.pvalue,
            # Fazemos a checagem lógica usando o config diretamente no retorno
            "passou": teste.pvalue > self.config.significancia_banda
        }

    def gerar_projeções_benchmark_sarima(self, df_historico: pd.DataFrame, coluna_alvo: str) -> pd.DataFrame:
        """
        Gera uma projeção univariada de benchmark (SARIMA) com bandas de confiança analíticas
        para servir de termo comparativo contra a estrutura multivariada do VAR.
        """
        # Limpeza rápida local
        serie = pd.to_numeric(df_historico[coluna_alvo], errors='coerce').dropna()
        serie = serie[serie.index >= '1994-08-01']
        
        # Ajusta um modelo SARIMA básico (1,1,1) x (1,0,0,12) para capturar inércia e sazonalidade anual
        modelo_sarima = SARIMAX(serie, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12), enforce_stationarity=False)
        resultado_sarima = modelo_sarima.fit(disp=False)
        
        # Realiza o forecast
        projeção = resultado_sarima.get_forecast(steps=self.config.horizonte_projeção_meses)
        df_sarima = pd.DataFrame(index=projeção.predicted_mean.index)
        
        df_sarima["SARIMA_BASE"] = projeção.predicted_mean
        
        # Coleta os limites do intervalo de confiança conforme a significância configurada
        intervalo = projeção.conf_int(alpha=self.config.significancia_banda)
        df_sarima["SARIMA_ESTRESSE_SUPERIOR"] = intervalo.iloc[:, 1]
        df_sarima["SARIMA_ESTRESSE_INFERIOR"] = intervalo.iloc[:, 0]
        
        return df_sarima.clip(lower=-0.02)

    def executar_backtesting_cobertura(self, df_universo: pd.DataFrame, coluna_alvo: str, meses_teste: int = 36) -> dict:
        """
        Executa um teste de cobertura macro (Backtesting de Cauda) via Walk-Forward Validation.
        Mede a eficácia real das bandas em conter os eventos históricos fora da amostra (out-of-sample).
        """
        df_trabalho = df_universo.copy()
        for col in df_trabalho.columns:
            df_trabalho[col] = pd.to_numeric(df_trabalho[col], errors='coerce')
        df_trabalho = df_trabalho.dropna(how='all', axis=1).dropna()
        df_trabalho.columns = [str(col).strip().replace('\r', '').replace('\n', '') for col in df_trabalho.columns]
        
        coluna_limpa = str(coluna_alvo).strip().replace('\r', '').replace('\n', '')
        idx_alvo = list(df_trabalho.columns).index(coluna_limpa)
        
        total_pontos = 0
        violacoes_superior = 0
        violacoes_inferior = 0
        
        # Validação móvel passo a passo
        for i in range(meses_teste - 3):
            df_treino_movel = df_trabalho.iloc[:-(meses_teste - i)].copy()
            if len(df_treino_movel) < 50:
                continue
                
            model_temp = VAR(df_treino_movel)
            res_temp = model_temp.fit(maxlags=self.lags_otimos)
            
            # Previsão de 3 passos à frente
            stderr_temp = res_temp.forecast_interval(df_treino_movel.values[-self.lags_otimos:], steps=3, alpha=self.config.significancia_banda)
            
            lim_inf = stderr_temp[1][:, idx_alvo]
            lim_sup = stderr_temp[2][:, idx_alvo]
            
            reais = df_trabalho.iloc[len(df_treino_movel):len(df_treino_movel)+3][coluna_limpa].values
            
            for r, inf, sup in zip(reais, lim_inf, lim_sup):
                total_pontos += 1
                if r > sup:
                    violacoes_superior += 1
                elif r < inf:
                    violacoes_inferior += 1
                    
        taxa_violacao = (violacoes_superior + violacoes_inferior) / total_pontos if total_pontos > 0 else 0
        
        return {
            "total_pontos": total_pontos,
            "violacoes_teto": violacoes_superior,
            "violacoes_piso": violacoes_inferior,
            "taxa_violacao_real": taxa_violacao,
            "taxa_alvo_esperada": self.config.significancia_banda,
            "aprovado": taxa_violacao <= (self.config.significancia_banda * 1.5) # tolerância aceitável de mercado
        }