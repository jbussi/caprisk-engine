# ⚖️ CapRisk Engine — Dashboard Executivo de Otimização Estrutural de Capital & ALM

O **CapRisk Engine** é uma plataforma analítica de alta performance desenvolvida para a simulação estocástica de balanços corporativos, gestão de ativos e passivos (ALM - *Asset and Liability Management*) e otimização da estrutura de capital. 

O sistema integra modelos macroeconômicos stochásticos (Monte Carlo com 10.000 cenários) e a formulação de desalavancagem/reavancagem de risco patrimonial via **Metodologia de Hamada**, permitindo mapear com precisão cirúrgica a probabilidade de insolvência e a real criação de valor econômico para o acionista.

---

## 🚀 Funcionalidades Principais (v1.0.0)

*   **Simulação estocástica de Monte Carlo:** Geração de 10.000 trajetórias macroeconômicas simultâneas estressando indexadores chaves de mercado (**Selic**, **IPCA**, **IGPM**).
*   **Gestão Algorítmica de Passivos (ALM):** Suporte a estruturas completas de amortização (`SAC`, `PRICE`, `BULLET`) com curvas de spreads dinâmicas e calibração via equivalência de Valor Presente Líquido (VPL).
*   **Métricas de Risco de Cauda & Performance:** Custo Ponderado de Capital (**WACC Médio Global**), **Beta Reavancado de Hamada**, **Sharpe do Balanço** (Alfa sobre volatilidade do ROE) e mapeamento dinâmico de **Risco de Ruína Crítica** (liquidez imediata do caixa).
*   **Fronteira Eficiente de Capital:** Mapeamento visual e estatístico do ponto ótimo de alavancagem ($P/PL$) focado na minimização do custo de capital.

---

## 🛠️ Arquitetura do Projeto

O ecossistema está estruturado seguindo os princípios de Design Orientado a Objetos (POO) e forte separação de responsabilidades (Engines de Cálculo vs. Camada de UI):

```text
caprisk-engine/
├── src/
│   ├── alm/
│   │   ├── __init__.py
│   │   └── credit_lines.py          # Políticas de pricing contínuo e fluxos de contratos
│   ├── reports/
│   │   ├── __init__.py
│   │   └── performance_evaluator.py # Motor matricial de performance e VPL estocástico
│   └── app.py                       # Interface executiva centralizada (Streamlit UI)
├── schemas/
│   └── performance_evaluator.py     # DTOs e contratos de dados estruturados
├── requirements.txt                 # Dependências homologadas do ecossistema
└── README.md
```

## 📦 Configuração e Instalação

### Pré-requisitos:
* Python 3.10 ou superior
* Ambiente isolado (recomendado: venv ou conda)

## 📦 Configuração e Instalação

### Pré-requisitos
*   Python 3.10 ou superior
*   Ambiente isolado (recomendado: `venv` ou `conda`)

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/joaobussi/caprisk-engine.git](https://github.com/joaobussi/caprisk-engine.git)
   cd caprisk-engine
   ```
2. **Crie e ative um ambiente virtual:**
   ```bash
   python -m venv venv
    # No Windows:
    .\venv\Scripts\activate
    # No Linux/Mac:
    source venv/bin/activate
    ```
3. **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Execute o Dashboard:**
    ```bash
    streamlit run src/app.py
    ```

## 📈 Formulação e Conceitos Chaves

### 1. Reavancagem de Risco (Equação de Hamada)
O modelo ajusta dinamicamente o Beta do setor para refletir a alavancagem financeira introduzida pela estrutura de passivos escolhida na UI:
$$\beta_L = \beta_U \left[ 1 + (1 - T)\left(\frac{P}{PL}\right) \right]$$

### 2. Sharpe do Balanço
Calculado como a eficiência do retorno real do investidor subtraído do seu custo de oportunidade (CAPM), ponderado pela volatilidade estocástica gerada no Monte Carlo:
$$\text{Sharpe do Balanço} = \frac{\text{Mediana do ROE Líquido}}{\sigma_{\text{ROE}}}$$

## 🎯 Roadmap Estrutural (Próximos Passos v1.1.0)

* [ ] **Tratamento Customizável de Ativos e Passivos:** Módulos flexíveis para indexadores mistos (ex: CDI + Spread fixo), carências personalizadas e inserção estocástica na geração operacional do Ativo.
* [ ] **Dashboard de Alta Resolução Analítica:** Inclusão de relatórios GAPs de prazos/taxas (ALM Mismatch), gráficos de sensibilidade sob estresse macroeconômico e detalhamento do VaR do balanço.
* [ ] **Bankruptcy Trapping:** Implementação de travas de falência no motor de simulação para congelar fluxos residuais e penalizar cenários pós-ruína de liquidez.

---

## 👥 Créditos

* **João Bussi**