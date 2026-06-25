import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import Dict

class CustomIndexConfig(BaseModel):
    """Representa o dicionário de pesos dinâmicos que o usuário insere"""
    # Exemplo de input: {"IPCA": 0.40, "IGPM": 0.60}
    pesos: Dict[str, float]

    def validar_pesos(self):
        if not np.isclose(sum(self.pesos.values()), 1.0):
            raise ValueError("A soma das proporções dos custos deve ser exatamente 1.0 (100%)")


class MacroDataFetcher:
    """Coleta dados estruturados diretamente das APIs oficiais (SGS/Banco Central)"""
    
    @staticmethod
    def get_bcb_series(series_code: int, start_date: str) -> pd.DataFrame:
        """Coleta séries temporais do Sistema Gerenciador de Séries Temporais do BCB"""
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados?formato=json&dataInicial={start_date}"
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.DataFrame(response.json())
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = pd.to_numeric(df['valor']) / 100  # Converte percentual para decimal
        df = df.set_index('data')
        return df


class DynamicInflationScraper:
    """
    Scraper flexível para capturar dados de tabelas de inflação/preços na web.
    Pode ser adaptado para ler qualquer estrutura de tabela HTML de índices setoriais.
    """
    def __init__(self, target_url: str):
        self.url = target_url

    def scrape_table_to_dataframe(self, table_id: str = None) -> pd.DataFrame:
        """
        Varre uma página web, localiza uma tabela de índices e a converte em DataFrame.
        Útil para capturar índices que não possuem API pública amigável.
        """
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(self.url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Encontra a tabela especificada
        table = soup.find('table', {'id': table_id}) if table_id else soup.find('table')
        if not table:
            raise ValueError("Não foi possível encontrar a tabela de dados na página informada.")
            
        # Converte a tabela HTML diretamente para uma lista de DataFrames do Pandas
        df_list = pd.read_html(str(table))
        df = df_list[0]
        
        # Tratamento genérico de datas e valores (deve ser refinado conforme o site alvo)
        # Supondo coluna 0 como data e coluna 1 como valor do índice
        df.columns = ['data', 'valor_indice']
        return df


class InflationIndexEngine:
    """Une os dados brutos e gera o indicador sintético customizado baseado em pesos"""
    def __init__(self, config: CustomIndexConfig):
        config.validar_pesos()
        self.config = config

    def build_sintetic_index(self, historical_matrix: pd.DataFrame) -> pd.Series:
        """
        Recebe um DataFrame contendo colunas com os índices brutos (ex: 'IPCA', 'IGPM')
        e calcula o índice dinâmico ponderado histórico.
        """
        sintetico = pd.Series(0.0, index=historical_matrix.index)
        
        for indice, peso in self.config.pesos.items():
            if indice in historical_matrix.columns:
                sintetico += historical_matrix[indice] * peso
            else:
                raise KeyError(f"O índice {indice} exigido nos custos não foi encontrado na base de dados.")
                
        return sintetico


if __name__ == "__main__":
    # Teste rápido de coleta de dados reais do Banco Central
    print("Coletando dados reais Selic e IPCA do BCB...")
    
    # Códigos SGS: 11 (Selic acumulada no mês % a.m), 433 (IPCA variação mensal % a.m)
    try:
        selic_historica = MacroDataFetcher.get_bcb_series(11, "01/01/2015")
        ipca_historico = MacroDataFetcher.get_bcb_series(433, "01/01/2015")
        
        # Junta os dados pela data comum
        database = pd.DataFrame(index=ipca_historico.index)
        database['IPCA'] = ipca_historico['valor']
        database['SELIC'] = selic_historica['valor']
        
        print("\nBase de dados macro montada com sucesso (Primeiras linhas):\n", database.dropna().head())
        
        # Testando o cálculo do índice customizado
        config_custos = CustomIndexConfig(pesos={"IPCA": 0.30, "SELIC": 0.70}) # Empresa muito indexada a juros
        engine_indice = InflationIndexEngine(config_custos)
        
        indice_personalizado = engine_indice.build_sintetic_index(database.dropna())
        print("\nÍndice customizado gerado para a empresa (Poder de Compra):\n", indice_personalizado.head())
        
    except Exception as e:
        print(f"Erro na execução do teste: {e}")