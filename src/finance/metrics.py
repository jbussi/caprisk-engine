import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional


class TipoAmortizacao(str, Enum):
    PRICE = "PRICE"
    SAC = "SAC"
    BULLET = "BULLET"


class TipoIndexador(str, Enum):
    PREFIXADO = "PREFIXADO"
    SELIC = "SELIC"
    IPCA = "IPCA"
    IGPM = "IGPM"
    DOLAR = "DOLAR"


class ContratoDividaInput(BaseModel):
    """
    Schema estrito para validação de cada contrato de passivo enviado pelo cliente.
    """
    id_divida: str = Field(..., description="Identificador único do contrato ou banco emissor.")
    valor_principal: float = Field(..., ge=0, description="Valor nominal captado (Saldo devedor inicial).")
    taxa_nominal_ano: float = Field(..., ge=0, description="Spread ou taxa fixa contratual ao ano (ex: 0.05 para 5% a.a.).")
    tipo_amortizacao: TipoAmortizacao = Field(..., description="Sistema de amortização: PRICE, SAC ou BULLET.")
    indexador: TipoIndexador = Field(..., description="Indexador macroeconômico ao qual a dívida está atrelada.")
    prazo_total_meses: int = Field(..., ge=1, description="Prazo total do contrato em meses.")
    carencia_meses: int = Field(default=0, ge=0, description="Meses de carência onde há apenas pagamento de juros ou acúmulo.")


class PassivoCalculatorEngine:
    """
    Motor financeiro responsável por projetar e estressar cronogramas de pagamento.
    Calcula parcelas sob tabelas PRICE, SAC e BULLET integrando indexadores estocásticos.
    """
    def __init__(self, contrato: ContratoDividaInput):
        self.contrato = contrato
        # Converte a taxa anual nominal do contrato para a taxa equivalente mensal (juros compostos)
        self.taxa_mensal_contratual = (1 + contrato.taxa_nominal_ano) ** (1 / 12) - 1

    def calcular_cronograma_estressado(self, serie_indexador_projetada: pd.Series) -> pd.DataFrame:
        """
        Gera o fluxo de caixa completo da dívida (Parcela, Juros, Amortização, Saldo Devedor)
        recalculado dinamicamente com base nas projeções mensais do indexador econômico.
        """
        prazo = self.contrato.prazo_total_meses
        carencia = self.contrato.carencia_meses
        saldo_devedor = self.contrato.valor_principal
        
        # Garante que a série do indexador cobre todo o horizonte do contrato
        if len(serie_indexador_projetada) < prazo:
            raise ValueError(f"A série do indexador projetada possui tamanho ({len(serie_indexador_projetada)}) menor que o prazo da dívida ({prazo}).")
            
        # Listas para acumular o fluxo de caixa
        fluxo_datas = []
        fluxo_parcelas = []
        fluxo_juros = []
        fluxo_amortizacoes = []
        fluxo_saldo_devedor = []
        
        for mes in range(1, prazo + 1):
            # 1. Captura a taxa do indexador para o mês corrente (ex: IPCA do mês ou Selic do mês)
            # Se for Prefixado, o indexador não adiciona custo (taxa_indexador = 0)
            taxa_indexador_mes = 0.0
            if self.contrato.indexador != TipoIndexador.PREFIXADO:
                taxa_indexador_mes = serie_indexador_projetada.iloc[mes - 1]
            
            # Taxa total ponderada do mês = Juros Contratual + Variação do Indexador (Aproximação linear padrão de mercado)
            taxa_total_mes = self.taxa_mensal_contratual + taxa_indexador_mes
            
            # 2. Atualiza o saldo devedor pela inflação/indexador antes do cálculo da parcela (Atualização Monetária)
            if self.contrato.indexador in [TipoIndexador.IPCA, TipoIndexador.IGPM, TipoIndexador.DOLAR]:
                saldo_devedor = saldo_devedor * (1 + taxa_indexador_mes)
                # Nesse caso, a taxa de juros do mês incide apenas sobre o saldo já corrigido
                juros_do_mes = saldo_devedor * self.taxa_mensal_contratual
            else:
                # Para Selic ou Pré, os juros correm sobre a taxa cheia acumulada
                juros_do_mes = saldo_devedor * taxa_total_mes

            # 3. Tratamento do Período de Carência
            if mes <= carencia:
                amortizacao_do_mes = 0.0
                parcela_do_mes = juros_do_mes
                saldo_devedor_fim = saldo_devedor
            else:
                # Prazo restante efetivo para amortizar a dívida
                prazo_restante = prazo - max(carencia, mes - 1)
                
                # 4. Motores de Amortização de Mercado
                if self.contrato.tipo_amortizacao == TipoAmortizacao.SAC:
                    amortizacao_do_mes = saldo_devedor / prazo_restante
                    parcela_do_mes = amortizacao_do_mes + juros_do_mes
                    saldo_devedor_fim = saldo_devedor - amortizacao_do_mes
                    
                elif self.contrato.tipo_amortizacao == TipoAmortizacao.PRICE:
                    # Fórmula francesa adaptada dinamicamente para o saldo atualizado e taxa corrente
                    if taxa_total_mes > 0:
                        fator = ((1 + taxa_total_mes) ** prazo_restante) - 1
                        parcela_do_mes = saldo_devedor * (taxa_total_mes * ((1 + taxa_total_mes) ** prazo_restante)) / fator
                    else:
                        parcela_do_mes = saldo_devedor / prazo_restante
                        
                    amortizacao_do_mes = parcela_do_mes - juros_do_mes
                    saldo_devedor_fim = saldo_devedor - amortizacao_do_mes
                    
                elif self.contrato.tipo_amortizacao == TipoAmortizacao.BULLET:
                    # No modelo Bullet, amortiza tudo no último mês do contrato
                    if mes == prazo:
                        amortizacao_do_mes = saldo_devedor
                        parcela_do_mes = amortizacao_do_mes + juros_do_mes
                        saldo_devedor_fim = 0.0
                    else:
                        amortizacao_do_mes = 0.0
                        parcela_do_mes = juros_do_mes
                        saldo_devedor_fim = saldo_devedor

            # Garante consistência matemática contra saldos residuais negativos por arredondamento
            if saldo_devedor_fim < 0 or np.isclose(saldo_devedor_fim, 0):
                saldo_devedor_fim = 0.0

            # Aloca métricas calculadas nas listas da esteira
            fluxo_parcelas.append(parcela_do_mes)
            fluxo_juros.append(juros_do_mes)
            fluxo_amortizacoes.append(amortizacao_do_mes)
            fluxo_saldo_devedor.append(saldo_devedor_fim)
            
            # Avança o saldo devedor para a abertura do próximo mês
            saldo_devedor = saldo_devedor_fim

        # Consolida tudo num DataFrame indexado pelo mês de pagamento
        df_cronograma = pd.DataFrame({
            "MES": list(range(1, prazo + 1)),
            "PARCELA": fluxo_parcelas,
            "JUROS": fluxo_juros,
            "AMORTIZACAO": fluxo_amortizacoes,
            "SALDO_DEVEDOR_FIM": fluxo_saldo_devedor
        }).set_index("MES")
        
        return df_cronograma