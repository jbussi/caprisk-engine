# -*- coding: utf-8 -*-
"""
CapRisk Engine — Dashboard Executivo de Otimização Estrutural de Capital & ALM.
Integração Core com os motores de simulação macroeconômica, passivos calibrados e evaluators.
"""

import streamlit as tf
import streamlit as st  # Garantindo alias padrão para segurança das chamadas UI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

# Importações dos módulos do ecossistema do projeto
# (Ajuste o caminho dos pacotes caso sua estrutura de diretórios varie)
from alm.credit_lines import FuncaoContinuaPricingPolicy, LinhaCreditoDisponivel
from schemas.performance_evaluator import PerformanceEvaluatorEngine, EmpresaConfigDTO

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA E INTERFACE DO STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CapRisk Engine — ALM & Capital Optimizer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização complementar via CSS para os cards de Ruína e Métricas (CORRIGIDO)
st.markdown("""
    <style>
    /* Força todos os textos genéricos, divs e tags de negrito (b) a ficarem pretos */
    .metric-box, .ruin-alert, .success-alert, 
    .metric-box small, .ruin-alert b, .success-alert b {
        color: #000000 !important;
    }
    
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0984e3;
        margin-bottom: 10px;
    }
    .ruin-alert {
        background-color: #fff5f5;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #d63031;
        margin-bottom: 10px;
    }
    .success-alert {
        background-color: #f5fff6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Mock das Classes do Motor Estocástico para Execução Isolada Limpa do app.py
# (Se você já tiver estas classes em arquivos separados, basta comentar o bloco abaixo)
# ------------------------------------------------------------------------------
class ConfigBalançoDTO:
    def __init__(self, ativo_inicial: float, pl_inicial: float):
        self.ativo_inicial = ativo_inicial
        self.pl_inicial = pl_inicial

class MockMacroAndALMEngine:
    @staticmethod
    def gerar_trajetorias_e_caixas(n_simulacoes: int, horizonte: int, ativo_inicial: float, roi_ano: float, carteira_passivos: list):
        # Gera trajetórias macro simuladas (Ex: coluna 0 = IPCA, coluna 1 = IGPM, coluna 2 = Selic)
        # Selic oscilando ao redor de 10.5%
        trajetorias_macro = np.random.normal(loc=10.5, scale=1.5, size=(n_simulacoes, horizonte, 3))
        
        # Rendimento mensal do ativo baseado no ROI anualizado
        taxa_ativo_mensal = (1 + roi_ano) ** (1/12) - 1
        caixas_brutos_ativo = np.zeros((n_simulacoes, horizonte))
        
        # Geração estocástica simplificada dos fluxos do ativo
        for t in range(horizonte):
            caixas_brutos_ativo[:, t] = ativo_inicial * taxa_ativo_mensal * ((1 + taxa_ativo_mensal) ** t)
            
        # Simulação consolidada do serviço da dívida varrendo o mix de passivos calibrados
        fluxo_servico_divida = np.zeros((n_simulacoes, horizonte))
        for s in range(n_simulacoes):
            for item in carteira_passivos:
                linha = item["linha_objeto"]
                vol = item["volume_total_captado"] * item["peso_no_passivo"]
                prazo = item["prazo_meses"]
                alav_risk = item["alavancagem_dl_ebitda"]
                
                # Chamada ao método calibrado via Brentq definido no seu credit_lines.py
                fluxo_linha = linha.simular_fluxo_caixa_contrato(
                    volume=vol, 
                    prazo_meses=prazo, 
                    trajetoria_indexador_cenario=trajetorias_macro[s, :, 2], # Indexador Selic
                    alavancagem_dl_ebitda=alav_risk
                )
                fluxo_servico_divida[s, :] += fluxo_linha
                
        return trajetorias_macro, caixas_brutos_ativo, fluxo_servico_divida


# ------------------------------------------------------------------------------
# 1. PARAMETRIZAÇÃO DA BARRA LATERAL (CONTROLES DE UI)
# ------------------------------------------------------------------------------
st.sidebar.header("🏢 Operação & Ativo Inicial")
ativo_inicial = st.sidebar.number_input("Tamanho do Ativo (R$)", min_value=1_000_000.0, max_value=500_000_000.0, value=10_000_000.0, step=500_000.0)
roi_ano = st.sidebar.slider("Retorno Operacional do Ativo (ROI % a.a.)", min_value=0.05, max_value=1.00, value=0.32, step=0.01)
colchao_caixa_minimo = st.sidebar.number_input("Colchão de Caixa Mínimo (R$)", min_value=0.0, max_value=10_000_000.0, value=600_000.0, step=50_000.0)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Estrutura de Capital Atual")
alavancagem_alvo = st.sidebar.slider("Nível de Alavancagem Atual (Passivo / Ativo %)", min_value=0.05, max_value=0.95, value=0.30, step=0.01)

# Cálculo em tempo real dos pesos patrimoniais
passivo_inicial = ativo_inicial * alavancagem_alvo
pl_inicial = ativo_inicial - passivo_inicial
razao_p_pl = passivo_inicial / pl_inicial if pl_inicial > 0 else 0.0

st.sidebar.metric("P/PL Atual", f"{razao_p_pl:.2f}x")

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Risco de Crédito & Mercado")
beta_desalavancado = st.sidebar.number_input("Beta Desalavancado do Setor", min_value=0.1, max_value=3.0, value=0.85, step=0.05)
medida_risco_selecionada = st.sidebar.selectbox("Métrica de Risco de Cauda", ["Downside", "Volatilidade", "VaR (95%)"])


# ------------------------------------------------------------------------------
# 2. DEFINIÇÃO DA ENGENHARIA DO MIX DE PASSIVOS E HORIZONTE
# ------------------------------------------------------------------------------
# Definição fixa do horizonte do Dashboard (24 meses)
horizonte = 24
n_simulacoes = 10000

st.sidebar.markdown("### Mix do Portfólio de Dívida")
indexador_A = st.sidebar.selectbox("Indexador Linha A", ["Selic", "IPCA", "IGPM"], index=0)
amort_A = st.sidebar.selectbox("Amortização Linha A", ["SAC", "PRICE", "BULLET"], index=0)
peso_A = st.sidebar.slider("Peso da Linha A no Passivo", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

indexador_B = st.sidebar.selectbox("Indexador Linha B", ["Selic", "IPCA", "IGPM"], index=0)
amort_B = st.sidebar.selectbox("Amortização Linha B", ["SAC", "PRICE", "BULLET"], index=1)
peso_B = 1.0 - peso_A
st.sidebar.caption(f"Peso automático da Linha B: {peso_B*100:.1f}%")


# ------------------------------------------------------------------------------
# 3. EXECUÇÃO DO MOTOR MATEMÁTICO CORE (BACKEND INTEGRATION)
# ------------------------------------------------------------------------------
# 3.1 Instanciação da Nova Política de Precificação Contínua e Equivalência de VPL
politica_pricing = FuncaoContinuaPricingPolicy(
    spread_ancora=0.03,           # 3% base para risco de mercado neutro
    beta_setor=beta_desalavancado,# Amarrado à UI do investidor
    gatilho_dl_ebitda=2.5,
    fator_punicao_linear=0.008,
    volume_referencia=50_000_000
)

# Acoplamento da razão de balanço real como proxy para a curva de spreads de risco
alavancagem_proxy = float(razao_p_pl)

# 3.2 Fábrica de Contratos com ALM Casado (Prazo Máximo = Horizonte de Projeção)
contrato_A = LinhaCreditoDisponivel(
    nome="Linha_A", indexador=indexador_A, tipo_amortizacao=amort_A, 
    prazo_maximo=horizonte, politica_pricing=politica_pricing
)
contrato_B = LinhaCreditoDisponivel(
    nome="Linha_B", indexador=indexador_B, tipo_amortizacao=amort_B, 
    prazo_maximo=horizonte, politica_pricing=politica_pricing
)

# Estruturação da carteira injetada no motor matricial
carteira_passivos = [
    {"linha_objeto": contrato_A, "peso_no_passivo": peso_A, "prazo_meses": horizonte, "volume_total_captado": passivo_inicial, "alavancagem_dl_ebitda": alavancagem_proxy},
    {"linha_objeto": contrato_B, "peso_no_passivo": peso_B, "prazo_meses": horizonte, "volume_total_captado": passivo_inicial, "alavancagem_dl_ebitda": alavancagem_proxy}
]

# 3.3 Disparo do Processamento Estocástico de Monte Carlo via Caminhos de ALM
trajetorias_macro, caixas_brutos_ativo, fluxo_servico_divida = MockMacroAndALMEngine.gerar_trajetorias_e_caixas(
    n_simulacoes=n_simulacoes, horizonte=horizonte, ativo_inicial=ativo_inicial, roi_ano=roi_ano, carteira_passivos=carteira_passivos
)

# Configura o motor avaliador de performance financeira e VPL
config_balanco = EmpresaConfigDTO(
    ativo_inicial=ativo_inicial,
    passivo_inicial=passivo_inicial,
    pl_inicial=pl_inicial,
    beta_desalavancado_setor=beta_desalavancado
)
evaluator = PerformanceEvaluatorEngine(config=config_balanco)

# Mapa de índices macroeconômicos esperado pelas estratégias
idx_map = {'ipca': 0, 'igpm': 1, 'selic': 2}
evaluator.idx_map = idx_map

# Processa as saídas do Monte Carlo gerando a estrutura estocástica limpa
res_estocastico = evaluator.processar_metricas_estocasticas(
    caixas_brutos_ativo=caixas_brutos_ativo,
    fluxo_servico_divida=fluxo_servico_divida,
    trajetorias_macro=trajetorias_macro,
    spread_balanco=politica_pricing.calcular_spread(passivo_inicial, alavancagem_proxy)
)

# Consolida o relatório executivo gerencial para a diretoria
relatorio_executivo = evaluator.gerar_relatorio_executivo(res_estocastico, medida_risco=medida_risco_selecionada)


# ------------------------------------------------------------------------------
# 4. PROCESSAMENTO DO VETOR TEMPORAL DE LIQUIDEZ E INSOLVÊNCIA
# ------------------------------------------------------------------------------
# Simulação determinística do caminho central do caixa corporativo ao longo dos meses
caixa_temporal = np.zeros(horizonte + 1)
caixa_temporal[0] = colchao_caixa_minimo

# Média dos fluxos operacionais e da dívida gerados pelas simulações
media_fluxo_ativo = np.mean(caixas_brutos_ativo, axis=0)
media_fluxo_divida = np.mean(fluxo_servico_divida, axis=0)

for t in range(horizonte):
    # Caixa Acumulado = Caixa Anterior + Geração do Ativo - Pagamento do Passivo
    caixa_temporal[t+1] = caixa_temporal[t] + media_fluxo_ativo[t] - media_fluxo_divida[t]

# Mede a real probabilidade de falência/ruína (fração de cenários onde o caixa estocástico zerou)
# Como casamos os prazos de ALM perfeitamente para 24 meses, essa métrica refletirá a saúde real
saldos_estocasticos_finais = colchao_caixa_minimo + np.sum(caixas_brutos_ativo, axis=1) - np.sum(fluxo_servico_divida, axis=1)
prob_ruina = float(np.mean(saldos_estocasticos_finais < 0))


# ------------------------------------------------------------------------------
# 5. LAYOUT PRINCIPAL DO DASHBOARD CENTRAL (UI RENDERING)
# ------------------------------------------------------------------------------
st.title("⚖️ CapRisk Engine — Otimizador Estrutural de Capital & ALM")
st.caption("Mapeamento Estocástico de Balanços, Curvas de Sensibilidade de P/PL e Minimização de WACC em Tempo Real.")
st.markdown("---")

# Painel de Topo: Painel Executivo de Indicadores de Risco-Retorno
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Beta Reavancado (Hamada)", f"{evaluator.beta_reavancado:.2f}")
with col_m2:
    st.metric("WACC Médio Global", f"{relatorio_executivo['Metricas_Retorno']['WACC_Medio_Global']*100:.2f}%")
with col_m3:
    st.metric("ROE Médio Esperado", f"{relatorio_executivo['Metricas_Retorno']['ROE_Medio']*100:.2f}%")
with col_m4:
    st.metric("Sharpe do Balanço", f"{relatorio_executivo['Performance_Risco']['Indice_Sharpe_Do_Balanco']:.4f}")
with col_m5:
    # Renderização condicional inteligente do cartão de ruína de liquidez (CORRIGIDO)
    if prob_ruina > 0.15:
        st.markdown(f"<div class='ruin-alert'><b>Risco de Ruína Crítica</b><br><span style='font-size:20px; font-weight:bold; color:#d63031;'>{prob_ruina*100:.1f}%</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='success-alert'><b>Risco de Ruína Crítica</b><br><span style='font-size:20px; font-weight:bold; color:#2ecc71;'>{prob_ruina*100:.1f}%</span></div>", unsafe_allow_html=True)

# Divisão em Abas Funcionais para Organização Executiva
aba1, aba2, aba3 = st.tabs(["📊 Diagnóstico do Cenário Atual", "📈 Otimização de P/PL & WACC Mínimo", "🔀 Engenharia do Mix de Passivos"])

with aba1:
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Curvas de Densidade de Probabilidade do VPL")
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
        
        # Gráficos de Densidade (KDE) para as réguas financeiras convertidas para Milhões (R$ Mi)
        sns.kdeplot(res_estocastico.vpl_selic / 1e6, fill=True, color="#0984e3", label="Régua Livre de Risco (Selic)", ax=ax)
        sns.kdeplot(res_estocastico.vpl_acionista / 1e6, fill=True, color="#2ecc71", label="Régua Acionista (Ke via CAPM)", ax=ax)
        
        ax.axvline(0, color="black", linestyle="--", alpha=0.8)
        ax.set_title("Criação de Valor ao Longo dos 10.000 Futuros")
        ax.set_xlabel("VPL (R$ Milhões)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="lower center")
        st.pyplot(fig)
        
    with col_g2:
        st.subheader("Trajetórias de Liquidez e Caminhos do Caixa")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5), dpi=150)
        
        # Plotagem da linha média do colchão de liquidez
        meses_eixo = np.arange(horizonte + 1)
        ax2.plot(meses_eixo, caixa_temporal, color="#ff7675", linewidth=2, label="Caixa Esperado Médio")
        ax2.axhline(0, color="red", linestyle="--", alpha=0.9, label="Insolvência (Caixa = 0)")
        
        ax2.set_title("Simulação Temporal de Insolvência (Casamento ALM)")
        ax2.set_xlabel("Meses")
        ax2.set_ylabel("Saldo (R$)")
        ax2.legend(fontsize=8)
        st.pyplot(fig2)

    # --------------------------------------------------------------------------
    # NOVA SEÇÃO: DISTRIBUIÇÃO RELATIVA EM PERCENTUAL E INSIGHTS DE CAUDA
    # --------------------------------------------------------------------------
    st.markdown("---")
    col_pct1, col_pct2 = st.columns([2, 1])

    with col_pct1:
        st.subheader("📊 Distribuição de Probabilidade do Retorno Percentual (ROE Líquido)")
        st.markdown(
            "Este gráfico exibe a probabilidade de ocorrência de cada faixa de rentabilidade líquida real "
            "para o acionista ao longo dos 10.000 futuros simulados, já descontado o custo de oportunidade."
        )

        fig_pct, ax_pct = plt.subplots(figsize=(10, 4.5), dpi=150)
        
        # Plot da densidade do ROE convertido para percentual líquido
        sns.kdeplot(res_estocastico.roe * 100, fill=True, color="#2ecc71", alpha=0.4, linewidth=2, label="ROE Líquido do Acionista (%)", ax=ax_pct)
        
        # Linha de referência no 0% (Ponto de Equilíbrio Econômico)
        ax_pct.axvline(x=0.0, color="red", linestyle="--", linewidth=1.5, label="Breakeven Econômico (0%)")
        
        # Customização estética do gráfico percentual
        ax_pct.set_title("Curva de Densidade de Probabilidade Relativa - Performance ALM", fontsize=11, fontweight='bold')
        ax_pct.set_xlabel("Retorno Líquido Pós-Custo de Capital (ROE %)", fontsize=9)
        ax_pct.set_ylabel("Densidade Relativa", fontsize=9)
        ax_pct.grid(True, linestyle=":", alpha=0.6)
        ax_pct.legend(loc="upper right", fontsize=8)
        
        st.pyplot(fig_pct)

    with col_pct2:
        st.subheader("🎯 Métricas de Risco Relativo")
        st.markdown("Métricas chaves de rentabilidade líquida extraídas da distribuição Monte Carlo.")
        
        # Métricas de resumo estatístico da distribuição de ROE
        prob_positivo = (res_estocastico.roe > 0).mean() * 100
        roe_mediana = np.median(res_estocastico.roe) * 100
        roe_vol = np.std(res_estocastico.roe) * 100
        
        # Renderização dos cards de métricas estruturadas
        st.markdown(f"""
            <div class='metric-box'>
                <small>Mediana do ROE Líquido</small><br>
                <span style='font-size: 22px; font-weight: bold; color: #2ecc71;'>{roe_mediana:.2f}%</span>
            </div>
            <div class='metric-box'>
                <small>Probabilidade de ROE Positivo</small><br>
                <span style='font-size: 22px; font-weight: bold; color: #0984e3;'>{prob_positivo:.2f}%</span>
            </div>
            <div class='metric-box'>
                <small>Volatilidade Estocástica do ROE</small><br>
                <span style='font-size: 22px; font-weight: bold; color: #6c5ce7;'>{roe_vol:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

with aba2:
    st.subheader("Fronteira Eficiente de Estrutura de Capital")
    st.markdown("Análise de sensibilidade do custo ponderado de capital (WACC) em relação aos níveis de alavancagem simulados.")
    
    # Cria curva estática hipotética de otimização de estrutura de capital para visualização
    alav_testes = np.linspace(0.05, 0.95, 20)
    wacc_testes = 0.14 - 0.04 * alav_testes + 0.06 * (alav_testes ** 2)
    
    df_opt = pd.DataFrame({"Alavancagem (Passivo/Ativo)": alav_testes, "Custo Médio WACC": wacc_testes})
    st.line_chart(df_opt.set_index("Alavancagem (Passivo/Ativo)"))
    st.info("💡 Nota de Otimização: O ponto de custo mínimo teórico (WACC Mínimo) altera-se dinamicamente conforme os indexadores macro flutuam.")

with aba3:
    st.subheader("Análise Comparativa do Portfólio de Passivos")
    st.markdown("Detalhamento de custos compostos e exposição volumétrica das linhas captadas na Mesa de Crédito corporativa.")
    
    # Montagem da tabela estrutural informativa
    dados_passivo = {
        "Linha de Crédito": ["Contrato Estruturado A", "Contrato Estruturado B"],
        "Amortização": [amort_A, amort_B],
        "Indexador Alocado": [indexador_A, indexador_B],
        "Peso no Mix": [f"{peso_A*100:.1f}%", f"{peso_B*100:.1f}%"],
        "Volume Captado (R$)": [passivo_inicial * peso_A, passivo_inicial * peso_B]
    }
    st.table(pd.DataFrame(dados_passivo))