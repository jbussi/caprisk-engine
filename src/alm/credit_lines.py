# -*- coding: utf-8 -*-
"""
Módulo de Gestão de Passivos: Modelagem de Linhas de Crédito e Amortização.
"""

import numpy as np


class BasePricingPolicy:
    """Interface abstrata para definição de políticas de spread comercial."""
    def calcular_spread(self, volume: float, alavancagem_atual: float = None) -> float:
        raise NotImplementedError


class TabelaFaixasPricingPolicy(BasePricingPolicy):
    """Política de precificação comercial baseada em faixas estáticas de volume."""
    def __init__(self, faixas: list):
        self.faixas = sorted(faixas, key=lambda x: x['limite'])
        
    def calcular_spread(self, volume: float, alavancagem_atual: float = None) -> float:
        for faixa in self.faixas:
            if volume <= faixa['limite']:
                return faixa['spread']
        return self.faixas[-1]['spread']


class FuncaoContinuaPricingPolicy(BasePricingPolicy):
    """Política de precificação dinâmica contínua integrada ao risco de crédito."""
    def __init__(self, spread_minimo: float = 0.015, spread_maximo: float = 0.08, 
                 elasticidade_volume: float = 0.04, gatilho_alavancagem: float = 1.5):
        self.spread_minimo = spread_minimo
        self.spread_maximo = spread_maximo
        self.elasticidade_volume = elasticidade_volume
        self.gatilho_alavancagem = gatilho_alavancagem
        
    def calcular_spread(self, volume: float, alavancagem_atual: float = None) -> float:
        # Efeito Escala: O volume reduz o spread logaritmicamente
        efeito_escala = self.elasticidade_volume * np.log10(max(volume, 1.0))
        spread_base = max(self.spread_maximo - efeito_escala, self.spread_minimo)
        
        # Efeito Risco: Se a alavancagem fura o gatilho, aplica penalização linear
        if alavancagem_atual and alavancagem_atual > self.gatilho_alavancagem:
            penalizacao = 0.01 * (alavancagem_atual - self.gatilho_alavancagem)
            return min(spread_base + penalizacao, self.spread_maximo)
            
        return spread_base


class LinhaCreditoDisponivel:
    """Fábrica e simuladora de contratos de dívida corporativa estruturada."""
    def __init__(self, nome: str, indexador: str, tipo_amortizacao: str, 
                 prazo_maximo: int, politica_pricing: BasePricingPolicy):
        self.nome = nome
        self.indexador = indexador.lower()
        self.tipo_amortizacao = tipo_amortizacao.upper()
        self.prazo_maximo = prazo_maximo
        self.politica_pricing = politica_pricing

    def simular_fluxo_caixa_contrato(self, volume: float, prazo_meses: int, 
                                     trajetoria_indexador_cenario: np.ndarray, 
                                     spread_anual: float) -> np.ndarray:
        """
        Calcula o fluxo mensal de saídas de caixa (Parcela = Amortização + Juros) para um cenário.
        """
        horizonte = len(trajetoria_indexador_cenario)
        fluxo_caixa = np.zeros(horizonte)
        
        saldo_devedor = volume
        amortizacao_constante = volume / prazo_meses if prazo_meses > 0 else 0
        
        for t in range(min(prazo_meses, horizonte)):
            # Coleta a taxa de juros nominal do período (Indexador do mês + Spread)
            taxa_anual = (trajetoria_indexador_cenario[t] / 100) + spread_anual
            taxa_mensal = (1 + taxa_anual) ** (1/12) - 1
            
            juros_do_mes = saldo_devedor * taxa_mensal
            
            if self.tipo_amortizacao == "SAC":
                amort_mes = amortizacao_constante
                parcela = amort_mes + juros_do_mes
                saldo_devedor -= amort_mes
                
            elif self.tipo_amortizacao == "PRICE":
                # Recálculo dinâmico da fórmula Price adaptada para juros flutuantes
                prazo_restante = prazo_meses - t
                if taxa_mensal > 0:
                    parcela = volume * (taxa_mensal / (1 - (1 + taxa_mensal) ** (-prazo_restante)))
                else:
                    parcela = amortizacao_constante
                amort_mes = parcela - juros_do_mes
                saldo_devedor -= amort_mes
                
            elif self.tipo_amortizacao == "BULLET":
                # Amortiza tudo apenas no último mês
                if t == (prazo_meses - 1):
                    amort_mes = volume
                    parcela = amort_mes + juros_do_mes
                    saldo_devedor = 0
                else:
                    parcela = juros_do_mes
            else:
                parcela = amortizacao_constante + juros_do_mes
                saldo_devedor -= amortizacao_constante
                
            fluxo_caixa[t] = parcela
            
        return fluxo_caixa