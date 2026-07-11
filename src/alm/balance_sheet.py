# -*- coding: utf-8 -*-
"""
Módulo Core de ALM (Asset Liability Management): Projeção Estocástica de Balanço.
"""

import numpy as np
from typing import List, Dict


class ALMEngine:
    """Motor de Processamento Quantitativo e Consolidação Estrutural do Balanço."""
    def __init__(self, valor_ativo_inicial: float, taxa_retorno_ativo_anual: float, 
                 nomes_variaveis: Dict[str, str]):
        self.valor_ativo_inicial = valor_ativo_inicial
        self.taxa_ativo_mensal = (1 + taxa_retorno_ativo_anual) ** (1/12) - 1
        self.nomes_vars = {k.lower(): v for k, v in nomes_variaveis.items()}

    def projetar_caixas_ativo(self, horizonte: int, n_simulacoes: int) -> np.ndarray:
        """Projeta a entrada de caixa bruta gerada pela operação física do Ativo."""
        caixas_ativo = np.zeros((n_simulacoes, horizonte))
        # O Ativo cresce de forma contínua com base na sua eficiência interna constante
        for t in range(horizonte):
            caixas_ativo[:, t] = self.valor_ativo_inicial * self.taxa_ativo_mensal * ((1 + self.taxa_ativo_mensal) ** t)
        return caixas_ativo

    def consolidar_carteira_passivos(self, trajetorias_macro: np.ndarray, 
                                     colunas_macro: List[str], 
                                     alocacao_linhas: List[Dict]) -> np.ndarray:
        """
        Varre o portfólio de dívidas configurado e gera a matriz tridimensional 
        consolidada do serviço da dívida para todos os cenários.
        """
        n_simulacoes, horizonte, _ = trajetorias_macro.shape
        colunas_lower = [c.lower() for c in colunas_macro]
        fluxo_passivo_consolidado = np.zeros((n_simulacoes, horizonte))

        for alocacao in alocacao_linhas:
            linha = alocacao['linha_objeto']
            peso = alocacao['peso_no_passivo']
            prazo = alocacao['prazo_meses']
            volume_linha = alocacao['volume_total_captado'] * peso
            
            idx_indexador = colunas_lower.index(linha.indexador)
            spread_linha = alocacao.get('spread_fixado', 0.035)

            for s in range(n_simulacoes):
                trajetoria_indexador = trajetorias_macro[s, :, idx_indexador]
                
                # Invoca a inteligência matemática do credit_lines.py
                fluxo_contrato = linha.simular_fluxo_caixa_contrato(
                    volume=volume_linha,
                    prazo_meses=prazo,
                    trajetoria_indexador_cenario=trajetoria_indexador,
                    spread_anual=spread_linha
                )
                fluxo_passivo_consolidado[s, :] += fluxo_contrato

        return fluxo_passivo_consolidado