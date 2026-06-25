import numpy as np
import pandas as pd

class MacroSimulator:
    def __init__(self, n_simulations: int, n_months: int, dt: float = 1/12):
        """
        Gerador de Cenários Macroeconômicos Estocásticos Correlacionados.
        dt = 1/12 representa passos mensais.
        """
        self.n_simulations = n_simulations
        self.n_months = n_months
        self.dt = dt
        
        # Parâmetros Calibrados (Exemplo com médias históricas do mercado brasileiro)
        # Selic (Vasicek)
        self.kappa_r = 0.3   # Velocidade de reversão à média
        self.theta_r = 0.10  # Meta de juros de longo prazo (10%)
        self.sigma_r = 0.03  # Volatilidade dos juros
        
        # IPCA (Vasicek modificado)
        self.kappa_i = 0.4   # Velocidade de reversão
        self.theta_i = 0.045 # Meta de inflação de longo prazo (4.5%)
        self.sigma_i = 0.02  # Volatilidade da inflação
        
        # Matriz de Correlação dos Choques (Selic vs IPCA)
        # Historicamente, choques de inflação alta correlacionam com subida de juros
        self.rho = 0.25 
        
    def _generate_correlated_shocks(self) -> tuple[np.ndarray, np.ndarray]:
        """Gera choques brownianos correlacionados usando Cholesky"""
        # Matriz de covariância simplificada para 2 variáveis
        # [ 1    rho ]
        # [ rho   1  ]
        # A decomposição de Cholesky resulta em L onde L * L^T = Correl
        # L = [ 1             0          ]
        #     [ rho   sqrt(1 - rho^2)   ]
        
        Z1 = np.random.normal(0, 1, (self.n_simulations, self.n_months))
        Z2 = np.random.normal(0, 1, (self.n_simulations, self.n_months))
        
        # Aplicando a correlação
        W_r = Z1
        W_i = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
        
        return W_r, W_i

    def simulate(self, r0: float, i0: float) -> dict[str, np.ndarray]:
        """
        Roda a simulação de Monte Carlo para Selic e IPCA.
        Retorna matrizes de shape (n_simulations, n_months + 1)
        """
        # Inicializa as matrizes de cenários (linhas = caminhos, colunas = meses)
        selic = np.zeros((self.n_simulations, self.n_months + 1))
        ipca = np.zeros((self.n_simulations, self.n_months + 1))
        
        # Condições iniciais (cenário atual do mercado)
        selic[:, 0] = r0
        ipca[:, 0] = i0
        
        # Gera os choques para todo o período
        W_r, W_i = self._generate_correlated_shocks()
        
        # Evolução temporal via Euler-Maruyama
        for t in range(1, self.n_months + 1):
            r_old = selic[:, t-1]
            i_old = ipca[:, t-1]
            
            # Equação da Selic (Vasicek)
            dr = self.kappa_r * (self.theta_r - r_old) * self.dt + self.sigma_r * np.sqrt(self.dt) * W_r[:, t-1]
            selic[:, t] = np.clip(r_old + dr, 0.02, 0.25) # Clip para evitar juros absurdos (min 2%, max 25%)
            
            # Equação do IPCA
            di = self.kappa_i * (self.theta_i - i_old) * self.dt + self.sigma_i * np.sqrt(self.dt) * W_i[:, t-1]
            ipca[:, t] = np.clip(i_old + di, -0.02, 0.15) # Evita deflação extrema ou hiperinflação fora do modelo
            
        return {
            "selic": selic,
            "ipca": ipca
        }

if __name__ == "__main__":
    # Teste rápido de sanidade do motor macro
    sim = MacroSimulator(n_simulations=5, n_months=36)
    cenarios = sim.simulate(r0=0.105, i0=0.04) # Começando com Selic a 10.5% e IPCA a 4.0%
    print("Shape da matriz gerada (Cenários, Meses):", cenarios["selic"].shape)
    print("Exemplo de 1 trajetória da Selic nos primeiros 6 meses:\n", cenarios["selic"][0, :6])