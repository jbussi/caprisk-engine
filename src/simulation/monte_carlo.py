import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MacroMonteCarloSimulator:
    """
    Simulador de Monte Carlo Paramétrico Multivariado acoplado ao motor VAR.
    Preserva a estrutura de covariância dos resíduos e a dinâmica temporal (lags).
    Suporta análise profunda e integrada de todos os indicadores do sistema.
    """
    def __init__(self, engine, regras_normalizacao, config_indicador_custom):
        self.engine = engine
        self.regras_normalizacao = regras_normalizacao
        self.config_indicador_custom = config_indicador_custom
        
        if not hasattr(engine, 'model_fitted') or engine.model_fitted is None:
            raise ValueError("O motor (engine) fornecido não possui um modelo VAR ajustado.")
            
        self.model_var = engine.model_fitted
        self.names = list(self.model_var.names)
        self.lags = engine.lags_otimos
        self.horizonte = engine.config.horizonte_projeção_meses
        self.sigma_u = self.model_var.sigma_u  
        self.coefs = self.model_var.coefs      
        self.intercept = self.model_var.k_trend 
        
    def rodar_simulacao(self, df_historico_limpo, n_simulacoes=10000, seed=42):
        if seed is not None:
            np.random.seed(seed)
            
        n_vars = len(self.names)
        df_var_input = df_historico_limpo[self.names].astype(float)
        valores_iniciais = df_var_input.values[-self.lags:]
        
        trajetorias_brutas = np.zeros((n_simulacoes, self.horizonte, n_vars))
        
        print(f"🎲 Iniciando Monte Carlo Estocástico Multivariado | N = {n_simulacoes} | H = {self.horizonte} meses")
        
        for s in range(n_simulacoes):
            choques = np.random.multivariate_normal(np.zeros(n_vars), self.sigma_u, size=self.horizonte)
            historico_simulado = list(valores_iniciais.copy())
            
            for t in range(self.horizonte):
                Y_pred = np.zeros(n_vars)
                if self.intercept is not None:
                    Y_pred += self.intercept
                    
                for lag_idx in range(self.lags):
                    Y_pred += np.dot(self.coefs[lag_idx], historico_simulado[-1 - lag_idx])
                
                Y_atual = Y_pred + choques[t]
                trajetorias_brutas[s, t, :] = Y_atual
                historico_simulado.append(Y_atual)
                
        return trajetorias_brutas

    def consolidar_indicador_customizado(self, trajetorias_brutas):
        n_simulacoes = trajetorias_brutas.shape[0]
        trajetorias_custom = np.zeros((n_simulacoes, self.horizonte))
        
        for s in range(n_simulacoes):
            acumulador_normalizado = np.zeros(self.horizonte)
            for col, peso in self.config_indicador_custom.items():
                if col in self.names:
                    idx_col = self.names.index(col)
                    mu = self.regras_normalizacao[col]["media"]
                    sigma = self.regras_normalizacao[col]["desvio"]
                    
                    sim_bruta = trajetorias_brutas[s, :, idx_col]
                    sim_norm = (sim_bruta - mu) / sigma
                    acumulador_normalizado += sim_norm * peso
                    
            trajetorias_custom[s, :] = acumulador_normalizado
            
        return trajetorias_custom

    def extrair_metricas_risco_serie(self, serie_trajetorias):
        """Calcula os percentis estatísticos para uma matriz [N, Horizonte] específica."""
        df_metricas = pd.DataFrame(index=[f"Mês +{t+1}" for t in range(self.horizonte)])
        df_metricas["Mediana"] = np.percentile(serie_trajetorias, 50, axis=0)
        df_metricas["Inf_90"] = np.percentile(serie_trajetorias, 10, axis=0)
        df_metricas["Sup_90"] = np.percentile(serie_trajetorias, 90, axis=0)
        df_metricas["Inf_95"] = np.percentile(serie_trajetorias, 5, axis=0)
        df_metricas["Sup_95"] = np.percentile(serie_trajetorias, 95, axis=0)
        df_metricas["Inf_99"] = np.percentile(serie_trajetorias, 1, axis=0)
        df_metricas["Sup_99"] = np.percentile(serie_trajetorias, 99, axis=0)
        return df_metricas

    def gerar_relatorio_consolidado_var(self, trajetorias_brutas, trajetorias_custom, mes_alvo=12):
        """
        Gera uma tabela executiva comparando o risco de cauda (VaR) de todas as variáveis 
        em um determinado horizonte de planejamento (ex: Mês +12).
        """
        idx_mes = mes_alvo - 1
        linhas_relatorio = []
        
        # 1. Avalia as variáveis originais do sistema
        for idx_col, nome_col in enumerate(self.names):
            dados_mes = trajetorias_brutas[:, idx_mes, idx_col]
            linhas_relatorio.append({
                "Indicador": nome_col,
                "Tipo": "Original (Bruto)",
                "Mediana": np.median(dados_mes),
                "VaR 90% (Sup)": np.percentile(dados_mes, 90),
                "VaR 95% (Sup)": np.percentile(dados_mes, 95),
                "VaR 99% (Sup)": np.percentile(dados_mes, 99),
                "Volatilidade Sim.": np.std(dados_mes)
            })
            
        # 2. Avalia o indicador customizado integrado
        dados_cust_mes = trajetorias_custom[:, idx_mes]
        linhas_relatorio.append({
            "Indicador": "Indicador Customizado",
            "Tipo": "Ponderado (Z-Score)",
            "Mediana": np.median(dados_cust_mes),
            "VaR 90% (Sup)": np.percentile(dados_cust_mes, 90),
            "VaR 95% (Sup)": np.percentile(dados_cust_mes, 95),
            "VaR 99% (Sup)": np.percentile(dados_cust_mes, 99),
            "Volatilidade Sim.": np.std(dados_cust_mes)
        })
        
        return pd.DataFrame(linhas_relatorio).set_index("Indicador")

    def plotar_comparativo_distribuicoes(self, trajetorias_brutas, trajetorias_custom, mes_alvo=12, colunas_foco=None, caminho_salvar=None):
        """
        Plota gráficos de densidade (KDE) para comparar visualmente o comportamento e
        achatamento das caudas das variáveis focadas em relação ao indicador customizado.
        Sálva o arquivo fisicamente antes de limpar o buffer de exibição.
        """
        if colunas_foco is None:
            colunas_foco = self.names[:3]
            
        idx_mes = mes_alvo - 1
        fig = plt.figure(figsize=(12, 6), dpi=100)
        
        # Plot da densidade do indicador customizado
        sns.kdeplot(trajetorias_custom[:, idx_mes], label="Indicador Customizado (Z-Score)", color="#2c3e50", linewidth=3, fill=True, alpha=0.1)
        
        # Plot das densidades das variáveis macro selecionadas (normalizadas para justa comparação)
        for col in colunas_foco:
            if col in self.names:
                idx_col = self.names.index(col)
                mu = self.regras_normalizacao[col]["media"]
                sigma = self.regras_normalizacao[col]["desvio"]
                
                dados_norm = (trajetorias_brutas[:, idx_mes, idx_col] - mu) / sigma
                sns.kdeplot(dados_norm, label=f"{col} (Z-Score Equivalente)", linestyle="--", linewidth=1.5)
                
        plt.title(f"Análise de Sensibilidade e Densidade de Risco no Mês +{mes_alvo}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Desvios Padrão em Relação à Média (Escala Z-Score)", fontsize=11)
        plt.ylabel("Densidade Probabilística", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="upper right", frameon=True, facecolor="white")
        plt.tight_layout()
        
        # Salvamento preventivo antes do plt.show()
        if caminho_salvar:
            plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
            print(f"🖼️ Gráfico de densidade exportado com sucesso: {caminho_salvar}")
            
        plt.show()
        plt.close(fig)

    def plotar_painel_mc(self, df_metricas, trajetorias_custom, nome_indicador="Indicador Customizado", caminho_salvar=None):
        """
        Gera o Fan Chart estocástico clássico com bandas de confiança.
        Salva o arquivo fisicamente antes de limpar o buffer de exibição.
        """
        fig = plt.figure(figsize=(14, 6), dpi=100)
        meses = np.arange(1, self.horizonte + 1)
        
        for i in range(15):
            plt.plot(meses, trajetorias_custom[i, :], color="#7f8c8d", alpha=0.15, linewidth=1)
            
        plt.fill_between(meses, df_metricas["Inf_99"], df_metricas["Sup_99"], color="#c0392b", alpha=0.10, label="Risco Extremo (IC 99%)")
        plt.fill_between(meses, df_metricas["Inf_95"], df_metricas["Sup_95"], color="#e67e22", alpha=0.18, label="Volatilidade Alta (IC 95%)")
        plt.fill_between(meses, df_metricas["Inf_90"], df_metricas["Sup_90"], color="#f1c40f", alpha=0.25, label="Volatilidade Padrão (IC 90%)")
        
        plt.plot(meses, df_metricas["Mediana"], color="#2c3e50", linestyle="-", linewidth=2.5, label="Mediana Estocástica")
        
        plt.title(f"Fan Chart Probabilístico de Estresse: {nome_indicador}", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Horizonte (Meses)", fontsize=11)
        plt.ylabel("Z-Score", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        plt.xticks(meses)
        plt.legend(loc="upper left", frameon=True, facecolor="white")
        plt.tight_layout()
        
        # Salvamento preventivo antes do plt.show()
        if caminho_salvar:
            plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
            print(f"🖼️ Fan Chart exportado com sucesso: {caminho_salvar}")
            
        plt.show()
        plt.close(fig)