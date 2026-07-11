# -*- coding: utf-8 -*-
"""
Módulo de Gestão de Passivos: Modelagem de Linhas de Crédito, 
Amortização e Equivalência de Valor Presente na Origem.
"""

import numpy as np
from scipy.optimize import brentq


class BasePricingPolicy:
    """Interface abstrata para definição de políticas de spread comercial."""
    def calcular_spread(self, volume: float, alavancagem_dl_ebitda: float = None) -> float:
        raise NotImplementedError


class FuncaoContinuaPricingPolicy(BasePricingPolicy):
    """
    Política de precificação baseada em Risco Setorial (Beta),
    Escala Controlada e Alavancagem Financeira (Dívida Líquida / EBITDA).
    
    Retorna o SPREAD ALVO anualizado que o banco exige para o risco do cliente.
    """
    def __init__(self, 
                 spread_ancora: float = 0.03,        # 3% para risco médio (Beta = 1.0)
                 beta_setor: float = 1.0,            # Beta do setor do devedor
                 gatilho_dl_ebitda: float = 2.5,     # Limite de conforto da alavancagem
                 fator_punicao_linear: float = 0.008, # +0.8% de spread por unidade acima do gatilho
                 volume_referencia: float = 50_000_000): # Volume base para ganho de escala
        
        self.spread_ancora = spread_ancora
        self.beta_setor = beta_setor
        self.gatilho_dl_ebitda = gatilho_dl_ebitda
        self.fator_punicao_linear = fator_punicao_linear
        self.volume_referencia = volume_referencia

    def calcular_spread(self, volume: float, alavancagem_dl_ebitda: float = None) -> float:
        # 1. Risco Setorial via Beta (Risco Sistêmico)
        spread_base = self.spread_ancora * self.beta_setor
        
        # 2. Risco de Crédito Idiosincrático (Dívida Líquida / EBITDA)
        if alavancagem_dl_ebitda and alavancagem_dl_ebitda > self.gatilho_dl_ebitda:
            excesso = alavancagem_dl_ebitda - self.gatilho_dl_ebitda
            spread_base += excesso * self.fator_punicao_linear
        elif alavancagem_dl_ebitda and alavancagem_dl_ebitda < (self.gatilho_dl_ebitda * 0.4):
            spread_base *= 0.85 # Desconto de "bom pagador" (baixo endividamento)
            
        # 3. Efeito Escala Suavizado (Tratamento logarítmico controlado)
        razao_volume = volume / self.volume_referencia
        efeito_escala = 0.008 * (1.0 / (1.0 + np.log10(max(razao_volume, 0.1) + 1)))
        
        # Garante limitadores de mercado para o spread (mínimo de 1% e máximo de 18% a.a.)
        return float(np.clip(spread_base - efeito_escala, 0.01, 0.18))


class LinhaCreditoDisponivel:
    """
    Fábrica e simuladora de contratos de dívida corporativa estruturada.
    Garante equivalência matemática de Valor Presente na largada usando a BULLET como régua.
    """
    def __init__(self, nome: str, indexador: str, tipo_amortizacao: str, 
                 prazo_maximo: int, politica_pricing: BasePricingPolicy):
        self.nome = nome
        self.indexador = indexador.lower()
        self.tipo_amortizacao = tipo_amortizacao.upper()
        self.prazo_maximo = prazo_maximo
        self.politica_pricing = politica_pricing

    def _gerar_fluxo_nominal(self, volume: float, prazo_meses: int, 
                             trajetoria_indexador: np.ndarray, taxa_spread_ano: float) -> np.ndarray:
        """Gera o fluxo nominal bruto de parcelas (Amortização + Juros) dada uma taxa de spread."""
        horizonte = len(trajetoria_indexador)
        fluxo = np.zeros(horizonte)
        saldo_devedor = volume
        amort_constante = volume / prazo_meses if prazo_meses > 0 else 0
        
        for t in range(min(prazo_meses, horizonte)):
            # Juros nominais do mês: (Indexador do mês + Spread simulado)
            taxa_anual_composta = (trajetoria_indexador[t] / 100) + taxa_spread_ano
            taxa_mensal = (1 + taxa_anual_composta) ** (1/12) - 1
            juros = saldo_devedor * taxa_mensal
            
            if self.tipo_amortizacao == "SAC":
                amort = amort_constante
                parcela = amort + juros
                saldo_devedor -= amort
                
            elif self.tipo_amortizacao == "PRICE":
                prazo_restante = prazo_meses - t
                if taxa_mensal > 0 and prazo_restante > 0:
                    # Fórmula da Price recalculada mensalmente sobre o saldo restante
                    parcela = saldo_devedor * (taxa_mensal / (1 - (1 + taxa_mensal) ** (-prazo_restante)))
                    amort = parcela - juros
                else:
                    amort = amort_constante
                    parcela = amort + juros
                saldo_devedor -= amort
                
            elif self.tipo_amortizacao == "BULLET":
                if t == (prazo_meses - 1):
                    amort = saldo_devedor
                    parcela = amort + juros
                    saldo_devedor = 0
                else:
                    amort = 0
                    parcela = juros
            else:
                amort = amort_constante
                parcela = amort + juros
                saldo_devedor -= amort
                
            fluxo[t] = parcela
            
        return fluxo

    def simular_fluxo_caixa_contrato(self, volume: float, prazo_meses: int, 
                                     trajetoria_indexador_cenario: np.ndarray, 
                                     alavancagem_dl_ebitda: float) -> np.ndarray:
        """
        Calcula o fluxo mensal calibrando a taxa de juros da tabela (SAC ou PRICE)
        para que o seu Valor Presente seja equivalente à BULLET com a taxa de risco alvo.
        """
        # 1. Calcula o spread de risco anual alvo da empresa
        spread_alvo_ano = self.politica_pricing.calcular_spread(volume, alavancagem_dl_ebitda)
        
        # 2. Define a taxa de desconto de referência (Indexador Médio + Spread Alvo)
        taxa_ref_anual = (np.mean(trajetoria_indexador_cenario) / 100) + spread_alvo_ano
        taxa_desconto_mensal = (1 + taxa_ref_anual) ** (1/12) - 1
        
        # 3. Constrói o Benchmark de Valor Presente usando a BULLET com a taxa alvo
        fluxo_bullet_ref = self._gerar_fluxo_nominal(volume, prazo_meses, trajetoria_indexador_cenario, spread_alvo_ano)
        vp_alvo_benchmark = sum(p / ((1 + taxa_desconto_mensal) ** (t + 1)) for t, p in enumerate(fluxo_bullet_ref[:prazo_meses]))

        # Se a linha já for BULLET, não precisa de calibração cruzada
        if self.tipo_amortizacao == "BULLET":
            return fluxo_bullet_ref

        # 4. Função Objetivo: Encontrar o spread nominal que zera a diferença de Valor Presente
        def objetivo_vpl(spread_teste):
            fluxo_teste = self._gerar_fluxo_nominal(volume, prazo_meses, trajetoria_indexador_cenario, spread_teste)
            vp_teste = sum(p / ((1 + taxa_desconto_mensal) ** (t + 1)) for t, p in enumerate(fluxo_teste[:prazo_meses]))
            return vp_teste - vp_alvo_benchmark

        # Otimizador Brentq busca o spread exato para emparelhar os VPLs
        try:
            spread_nominal_calibrado = brentq(objetivo_vpl, -0.05, 0.40)
        except ValueError:
            # Fallback de segurança operacional
            spread_nominal_calibrado = spread_alvo_ano

        # Retorna o fluxo nominal final perfeito, com a taxa ajustada para equivalência na largada
        return self._gerar_fluxo_nominal(volume, prazo_meses, trajetoria_indexador_cenario, spread_nominal_calibrado)