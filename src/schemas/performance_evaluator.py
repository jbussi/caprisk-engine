# -*- coding: utf-8 -*-
"""
Módulo de Avaliação de Desempenho e Risco Estocástico (ALM Engine).

Este módulo implementa a esteira de processamento quantitativo para cálculo de indicadores 
financeiros clássicos e avançados (WACC, ROE, ROI, IL) sob múltiplas réguas de desconto.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union
from pydantic import BaseModel


# ------------------------------------------------------------------------------
# 1. CONTÊINERES DE DADOS (DATA TRANSFER OBJECTS)
# ------------------------------------------------------------------------------

class EmpresaConfigDTO(BaseModel):
    """
    Data Transfer Object para isolar as premissas estáticas de estrutura da firma.
    """
    ativo_inicial: float
    passivo_inicial: float
    pl_inicial: float
    aliquota_imposto: float = 0.34  # IR + CSLL padrão no Brasil (34%)
    beta_desalavancado_setor: float


class ResultadoCenárioDTO(BaseModel):
    """
    Armazena a distribuição final e vetores estocásticos dos indicadores calculados.
    """
    vpl_selic: np.ndarray
    vpl_acionista: np.ndarray
    vpl_wacc: np.ndarray
    vpl_real_ipca: np.ndarray
    vpl_real_igpm: np.ndarray
    vpl_real_custom: np.ndarray
    roe: np.ndarray
    roi: np.ndarray
    il: np.ndarray
    wacc_medio_cenarios: np.ndarray

    class Config:
        arbitrary_types_allowed = True


# ------------------------------------------------------------------------------
# 2. PERSPECTIVAS DE DESCONTO TEMPORAL (STRATEGY PATTERN)
# ------------------------------------------------------------------------------

class BaseDiscountStrategy:
    """Classe base abstrata para cálculo de fatores de desconto cumulativos."""
    def calcular_fatores(self, horizonte: int, trajetoria_macro: np.ndarray, index_map: Dict[str, int]) -> np.ndarray:
        raise NotImplementedError


class FinanceiraSelicStrategy(BaseDiscountStrategy):
    """Desconto puro pela taxa livre de risco (Selic)."""
    def calcular_fatores(self, horizonte: int, trajetoria_macro: np.ndarray, index_map: Dict[str, int]) -> np.ndarray:
        factors = np.ones(horizonte)
        selic_anual = trajetoria_macro[:, index_map['selic']] / 100
        log_acumulado = 0.0
        for t in range(horizonte):
            r_m = (1 + selic_anual[t]) ** (1/12) - 1
            log_acumulado += np.log(1 + r_m)
            factors[t] = np.exp(-log_acumulado)
        return factors


class CAPMStrategy(BaseDiscountStrategy):
    """Desconto pelo Custo de Capital do Acionista (Ke) via CAPM Estocástico Dinâmico."""
    def __init__(self, beta_leverage: float, premio_mercado: float = 0.05):
        self.beta = beta_leverage
        self.premio = premio_mercado

    def calcular_fatores(self, horizonte: int, trajetoria_macro: np.ndarray, index_map: Dict[str, int]) -> np.ndarray:
        factors = np.ones(horizonte)
        selic_anual = trajetoria_macro[:, index_map['selic']] / 100
        log_acumulado = 0.0
        for t in range(horizonte):
            ke_anual = selic_anual[t] + (self.beta * self.premio)
            r_m = (1 + ke_anual) ** (1/12) - 1
            log_acumulado += np.log(1 + r_m)
            factors[t] = np.exp(-log_acumulado)
        return factors


class InflationDiscountStrategy(BaseDiscountStrategy):
    """Desconto monetário bruto para extração de ganho real de Poder de Compra."""
    def __init__(self, tipo_indice: str, custom_weights: Dict[str, float] = None):
        self.tipo = tipo_indice.lower()
        self.weights = custom_weights if custom_weights else {"ipca": 1.0, "igpm": 0.0}

    def calcular_fatores(self, horizonte: int, trajetoria_macro: np.ndarray, index_map: Dict[str, int]) -> np.ndarray:
        factors = np.ones(horizonte)
        log_acumulado = 0.0
        for t in range(horizonte):
            if self.tipo == "ipca":
                inf_mensal = trajetoria_macro[t, index_map['ipca']] / 100
            elif self.tipo == "igpm":
                inf_mensal = trajetoria_macro[t, index_map['igpm']] / 100
            else:
                inf_mensal = (trajetoria_macro[t, index_map['ipca']] / 100 * self.weights.get('ipca', 0.5)) + \
                             (trajetoria_macro[t, index_map['igpm']] / 100 * self.weights.get('igpm', 0.5))
            
            log_acumulado += np.log(1 + inf_mensal)
            factors[t] = np.exp(-log_acumulado)
        return factors


# ------------------------------------------------------------------------------
# 3. METRICAS DE RISCO DE CAUDA (METRICS STRATEGY)
# ------------------------------------------------------------------------------

class RiskMeasureEvaluator:
    """Calculador isolado das medidas de volatilidade e risco assimétrico de mercado."""
    
    @staticmethod
    def calcular_volatilidade(dados: np.ndarray) -> float:
        return float(np.std(dados))

    @staticmethod
    def calcular_downside_risk(dados: np.ndarray, benchmark: float = 0.0) -> float:
        sub_benchmark = dados[dados < benchmark]
        if len(sub_benchmark) == 0:
            return 0.0
        return float(np.sqrt(np.sum((sub_benchmark - benchmark) ** 2) / len(dados)))

    @staticmethod
    def calcular_value_at_risk(dados: np.ndarray, confianca: float = 0.95) -> float:
        percentil = (1.0 - confianca) * 100
        return float(np.percentile(dados, percentil))


# ------------------------------------------------------------------------------
# 4. MOTOR CENTRAL DE AVALIAÇÃO DE DESEMPENHO (ORCHESTRATOR)
# ------------------------------------------------------------------------------

class PerformanceEvaluatorEngine:
    """
    Orquestrador POO do módulo de relatórios analíticos de desempenho ALM.
    Responsável por rodar o balanço, reavancar betas e consolidar os outputs estruturados.
    """
    
    def __init__(self, config: EmpresaConfigDTO, colunas_macro: List[str] = None):
        self.config = config
        # Fallback inteligente para garantir compatibilidade com as chamadas do app.py
        colunas = colunas_macro if colunas_macro else ["ipca", "igpm", "selic"]
        self.colunas = [c.lower() for c in colunas]
        self._mapear_indices_colunas()
        self.beta_reavancado = self._reavancar_beta_firma()

    def _mapear_indices_colunas(self):
        self.idx_map = {
            "selic": self.colunas.index("selic"),
            "ipca": self.colunas.index("ipca"),
            "igpm": self.colunas.index("igpm")
        }

    def _reavancar_beta_firma(self) -> float:
        """Aplica a metodologia de Hamada para reavancagem de risco do Beta."""
        if self.config.pl_inicial <= 0:
            return self.config.beta_desalavancado_setor * 5.0
        razao_alavancagem = self.config.passivo_inicial / self.config.pl_inicial
        eficiencia_fiscal = 1.0 - self.config.aliquota_imposto
        return self.config.beta_desalavancado_setor * (1.0 + eficiencia_fiscal * razao_alavancagem)

    def calcular_wacc_periodo(self, selic_atual: float, spread_captacao: float) -> float:
        """Mapeia o custo marginal ponderado de capital para um período específico."""
        kd_bruto = selic_atual + spread_captacao
        kd_liquido = kd_bruto * (1.0 - self.config.aliquota_imposto)
        
        ke_periodo = selic_atual + (self.beta_reavancado * 0.05)
        
        peso_passivo = self.config.passivo_inicial / self.config.ativo_inicial
        peso_pl = self.config.pl_inicial / self.config.ativo_inicial
        
        return (peso_passivo * kd_liquido) + (peso_pl * ke_periodo)

    def processar_metricas_estocasticas(self, caixas_brutos_ativo: np.ndarray, 
                                        fluxo_servico_divida: np.ndarray, 
                                        trajetorias_macro: np.ndarray,
                                        spread_balanco: float,
                                        custom_weights: Dict[str, float] = None) -> ResultadoCenárioDTO:
        """
        Roda a esteira matricial cenário por cenário, isolando as métricas clássicas e de desconto.
        """
        n_simulacoes, horizonte, _ = trajetorias_macro.shape
        
        vpl_selic = np.zeros(n_simulacoes)
        vpl_acionista = np.zeros(n_simulacoes)
        vpl_wacc = np.zeros(n_simulacoes)
        vpl_real_ipca = np.zeros(n_simulacoes)
        vpl_real_igpm = np.zeros(n_simulacoes)
        vpl_real_custom = np.zeros(n_simulacoes)
        roe_vec = np.zeros(n_simulacoes)
        roi_vec = np.zeros(n_simulacoes)
        il_vec = np.zeros(n_simulacoes)
        wacc_medios = np.zeros(n_simulacoes)

        strat_selic = FinanceiraSelicStrategy()
        strat_capm = CAPMStrategy(beta_leverage=self.beta_reavancado)
        strat_ipca = InflationDiscountStrategy("ipca")
        strat_igpm = InflationDiscountStrategy("igpm")
        strat_custom = InflationDiscountStrategy("custom", custom_weights=custom_weights)

        for s in range(n_simulacoes):
            matriz_macro_cenario = trajetorias_macro[s, :, :]
            
            fluxo_op_ativo = caixas_brutos_ativo[s, :].copy()
            fluxo_op_ativo[-1] += self.config.ativo_inicial
            
            fluxo_liq_pl = fluxo_op_ativo - fluxo_servico_divida[s, :]

            # 1. Cálculo do WACC do mês
            wacc_do_mes = np.array([self.calcular_wacc_periodo(matriz_macro_cenario[t, self.idx_map['selic']]/100, spread_balanco) for t in range(horizonte)])
            wacc_medios[s] = np.mean(wacc_do_mes)

            # 2. Fatores de desconto sobre a matriz bidi do cenário core
            f_selic = strat_selic.calcular_fatores(horizonte, matriz_macro_cenario, self.idx_map)
            f_capm = strat_capm.calcular_fatores(horizonte, matriz_macro_cenario, self.idx_map)
            f_ipca = strat_ipca.calcular_fatores(horizonte, matriz_macro_cenario, self.idx_map)
            f_igpm = strat_igpm.calcular_fatores(horizonte, matriz_macro_cenario, self.idx_map)
            f_custom = strat_custom.calcular_fatores(horizonte, matriz_macro_cenario, self.idx_map)

            f_wacc = np.exp(-np.cumsum(np.log(1 + wacc_do_mes)))
            
            # 3. Integração de Valores Presentes Líquidos (VPL) - Valores Absolutos
            vpl_selic[s] = np.sum(fluxo_liq_pl * f_selic) - self.config.pl_inicial
            vpl_acionista[s] = np.sum(fluxo_liq_pl * f_capm) - self.config.pl_inicial
            vpl_wacc[s] = np.sum(fluxo_op_ativo * f_wacc) - self.config.ativo_inicial
            
            vpl_real_ipca[s] = np.sum(fluxo_liq_pl * f_ipca) - self.config.pl_inicial
            vpl_real_igpm[s] = np.sum(fluxo_liq_pl * f_igpm) - self.config.pl_inicial
            vpl_real_custom[s] = np.sum(fluxo_liq_pl * f_custom) - self.config.pl_inicial

            # 4. Indicadores Percentuais Líquidos Correção (Subtraindo o capital inicial / 1.0)
            # ROE Líquido = (Valor Presente dos Fluxos do Acionista - PL Inicial) / PL Inicial
            roe_vec[s] = vpl_acionista[s] / self.config.pl_inicial
            
            # ROI Líquido = (Valor Presente dos Fluxos Operacionais - Ativo Inicial) / Ativo Inicial
            roi_vec[s] = vpl_wacc[s] / self.config.ativo_inicial
            
            # Índice de Lucratividade (IL) tradicional = VP dos Fluxos / Investimento Inicial
            il_vec[s] = np.sum(fluxo_liq_pl * f_capm) / self.config.pl_inicial

        return ResultadoCenárioDTO(
            vpl_selic=vpl_selic, vpl_acionista=vpl_acionista, vpl_wacc=vpl_wacc,
            vpl_real_ipca=vpl_real_ipca, vpl_real_igpm=vpl_real_igpm, vpl_real_custom=vpl_real_custom,
            roe=roe_vec, roi=roi_vec, il=il_vec, wacc_medio_cenarios=wacc_medios
        )

    def gerar_relatorio_executivo(self, res: ResultadoCenárioDTO, medida_risco: str = "volatilidade") -> Dict:
        """Consolida as métricas financeiras finais ajustadas ao risco para exibição."""
        selic_medio_rf = 0.105
        roe_medio = float(np.mean(res.roe))
        
        if medida_risco.lower() == "volatilidade":
            risco = RiskMeasureEvaluator.calcular_volatilidade(res.roe)
        elif medida_risco.lower() == "downside":
            risco = RiskMeasureEvaluator.calcular_downside_risk(res.roe, benchmark=selic_medio_rf)
        else:
            risco = abs(RiskMeasureEvaluator.calcular_value_at_risk(res.roe, confianca=0.95))

        sharpe_balanco = (roe_medio - selic_medio_rf) / risco if risco > 0 else 0.0

        return {
            "Metricas_Retorno": {
                "ROE_Medio": roe_medio,
                "ROI_Medio": float(np.mean(res.roi)),
                "IL_Medio": float(np.mean(res.il)),
                "WACC_Medio_Global": float(np.mean(res.wacc_medio_cenarios))
            },
            "Criacao_Valor_Mediana": {
                "VPL_Régua_Selic": float(np.median(res.vpl_selic)),
                "VPL_Régua_Acionista_Ke": float(np.median(res.vpl_acionista)),
                "VPL_Régua_Firma_WACC": float(np.median(res.vpl_wacc))
            },
            "Poder_Compra_Mediana": {
                "VPL_Real_IPCA": float(np.median(res.vpl_real_ipca)),
                "VPL_Real_IGPM": float(np.median(res.vpl_real_igpm)),
                "VPL_Real_Customizado": float(np.median(res.vpl_real_custom))
            },
            "Performance_Risco": {
                "Medida_Risco_Utilizada": medida_risco,
                "Valor_Risco_Denominador": risco,
                "Indice_Sharpe_Do_Balanco": sharpe_balanco,
                "Probabilidade_VPL_Negativo_Acionista": float(np.mean(res.vpl_acionista < 0))
            }
        }