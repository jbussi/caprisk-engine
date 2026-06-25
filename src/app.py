import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.schemas.finance_inputs import CompanySimulationRequest
from src.simulation.monte_carlo import CapriskEngine

# 1. Inicialização do App com metadados profissionais
app = FastAPI(
    title="Caprisk Engine API",
    description="Motor quantitativo de simulação de estrutura de capital, ALM e risco de insolvência.",
    version="0.1.0"
)

# 2. Configuração de CORS (Essencial para o seu frontend se comunicar com a API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringir para o domínio real do frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Endpoint de Sanidade (Healthcheck)
@app.get("/health", status_code=status.HTTP_200_OK, tags=["Monitoramento"])
def health_check():
    return {"status": "healthy", "engine": "Caprisk Engine v0.1.0"}

# 4. Endpoint Principal de Simulação e Estresse de Capital
@app.post("/api/v1/simulate", status_code=status.HTTP_200_OK, tags=["Análise Quantitativa"])
def run_capital_simulation(request: CompanySimulationRequest):
    """
    Recebe a estrutura de ativos, custos e passivos da firma.
    Roda o motor de Monte Carlo estruturado e retorna o diagnóstico de risco de insolvência.
    """
    try:
        # Instancia o motor utilizando o padrão Orientado a Objetos baseado no request
        engine = CapriskEngine(request=request)
        
        # Executa a simulação integrando a macroeconomia atual do mercado brasileiro
        # Em versões futuras, esses valores iniciais de Selic e IPCA podem vir direto da esteira de dados
        resultado_simulacao = engine.run_simulation(current_selic=0.105, current_ipca=0.04)
        
        # Remove as matrizes brutas pesadas para não sobrecarregar o tráfego HTTP do JSON de resposta,
        # mantendo apenas os agregados estatísticos e os outputs analíticos estruturados.
        # (As matrizes podem ser salvas em cache ou parquet caso o front queira plotar todas as linhas)
        if "matrizes" in resultado_simulacao:
            del resultado_simulacao["matrizes"]
            
        return {
            "sucesso": True,
            "empresa": request.nome_empresa,
            "horizonte_meses": request.prazo_simulacao_meses,
            "cenarios_calculados": request.cenarios_monte_carlo,
            "diagnostico": resultado_simulacao
        }
        
    except ValueError as val_err:
        # Captura erros de validação financeira ou consistência de pesos
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Erro de consistência financeira nos parâmetros fornecidos: {str(val_err)}"
        )
    except Exception as e:
        # Fallback de segurança para erros genéricos no motor interno
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha interna no processamento do motor de Monte Carlo: {str(e)}"
        )

if __name__ == "__main__":
    # Inicializa o servidor local na porta 8000 com reload automático para desenvolvimento
    uvicorn.run("src.app.py:app", host="0.0.0.0", port=8000, reload=True)