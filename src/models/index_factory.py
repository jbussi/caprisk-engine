import pandas as pd
from typing import Dict, Tuple

class IndexFactory:
    """
    Fábrica adaptada para a nova Single Source of Truth estática.
    Orquestra a leitura da matriz unificada e gera a ponderação de inflação da firma.
    """
    
    @staticmethod
    def ler_matriz_unificada(caminho_csv: str = "dataset_macro_final.csv") -> pd.DataFrame:
        """
        Carrega de forma limpa o CSV unificado eliminando qualquer resíduo numérico 
        ou de texto presente no rodapé do arquivo.
        """
        # 1. Lê o arquivo bruto
        df = pd.read_csv(caminho_csv, sep=';')
        
        # 2. Força a conversão da coluna Data. 
        # O errors='coerce' transforma strings inválidas (como "1", "Fonte", etc.) em NaT (Not a Time)
        df['Data'] = pd.to_datetime(df['Data'], format='%m/%Y', errors='coerce')
        
        # 3. Elimina as linhas que viraram nulas (passa o rodo no rodapé)
        df = df.dropna(subset=['Data'])
        
        return df.set_index('Data')

    @staticmethod
    def build_corporate_macro_matrix(pesos_custos: Dict[str, float], caminho_csv: str = "dataset_macro_final.csv") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Gera a combinação linear ponderada para criar o índice sintético do cliente
        baseado estritamente nas colunas do arquivo final.
        """
        df_universo = IndexFactory.ler_matriz_unificada(caminho_csv)
        
        # Filtra apenas o corte temporal estável do Plano Real para o cálculo
        df_universo = df_universo[df_universo.index >= '1994-08-01']
        
        # Calcula o índice customizado multiplicando o peso pelas colunas correspondentes
        indice_sintetico = pd.Series(0.0, index=df_universo.index)
        
        for coluna, peso in pesos_custos.items():
            if coluna in df_universo.columns:
                indice_sintetico += df_universo[coluna] * peso
                
        return df_universo, indice_sintetico