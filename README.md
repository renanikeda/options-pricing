# Options Pricing - B3 Data Collection and Mathematical Models

Este projeto realiza o download e tratamento de dados de negociações de opções da B3 (Brasil, Bolsa, Balcão), também implementa e compara modelos matemáticos para precificação de opções, como o Black-Scholes, Kou Jump Diffusion e Heston Stochastic Volatility.

## 📁 Estrutura do Projeto

```
Options-Pricing/
├── models/                           # Modelos de precificação
│   ├── black_scholes.py              # Modelo Black-Scholes-Merton
│   ├── brownian_motion.py            # Simulação de Movimento Browniano
│   ├── heston_model.py               # Modelo de Volatilidade Estocástica de Heston
│   ├── kou_jump_diffusion.py         # Modelo de Difusão com Saltos de Kou
│   └── calibration.py                # Script para calibração dos modelos
│
├── data/                             # Coleta e processamento de dados
│   ├── B3_options_negotiation.py     # Processamento de histórico de negociações XML da B3
│   ├── B3_options_information.py     # Coleta dados das opções como Strike e Maturity
│   ├── opcoes_net.py                 # Coleta de dados do opcoes.net.br
│   └── Histórico B3/                 # Dados processados da B3
│       ├── Negociações 20200102.csv
│       ├── Negociações 20200103.csv
│       └── ... (arquivos diários)
├── utils.py                           # Funções utilitárias
├── requirements.txt                   # Dependências do projeto
└── README.md
```

## 🔧 Módulos de Coleta de Dados

### B3 - Brasil, Bolsa, Balcão

#### [`B3_options_negotiation.py`](data_collection/B3_options_negotiation.py)
- **Função**: Processamento de dados históricos em formato XML da B3
- **Entrada**: Arquivos XML históricos da B3
- **Processamento**: Conversão e limpeza de dados XML
- **Output**: Arquivos CSV estruturados no diretório `Histórico B3/`
- **Dados**: Negociações, preços, volumes, strikes, vencimentos

#### [`opcoes_net.py`](data_collection/opcoes_net.py)
- **Função**: Coleta de histórico de preços por ticker
- **Fonte**: opcoes.net.br
- **Input**: Ticker da opção
- **Output**: Dados históricos de preços e métricas

## 🧮 Modelos Matemáticos

### 1. Black-Scholes-Merton ([`black_scholes.py`](models/black_scholes.py))

**Modelo Clássico de Precificação**

- ✅ Precificação de opções europeias (call e put)
- ✅ Cálculo de gregas (Delta, Gamma, Vega, Theta, Rho)
- ✅ Volatilidade implícita via método de Newton-Raphson
- ✅ Simulação Monte Carlo para validação
- 📊 Visualização de superfícies de volatilidade

**Funções principais**:
```python
black_scholes_price(S, K, r, sigma, T, option_type)
implied_volatility(S, K, r, T, market_price, option_type)
calculate_greeks(S, K, r, sigma, T, option_type)
```

### 2. Movimento Browniano ([`brownian_motion.py`](models/brownian_motion.py))

**Processos Estocásticos Base**

- ✅ Movimento Browniano padrão
- ✅ Movimento Browniano Geométrico (GBM)
- ✅ Movimentos Brownianos correlacionados
- ✅ Simulação de trajetórias múltiplas
- 📊 Visualização de trajetórias

**Funções principais**:
```python
brownian_motion(T, dt, M)  # M trajetórias
geometric_brownian_motion(S0, T, dt, r, sigma, M)
cov_brownian_motion_diff(T, dt, rho, N, M)  # N dimensões correlacionadas
```

### 3. Modelo de Heston ([`heston_model.py`](models/heston_model.py))

**Volatilidade Estocástica**

- ✅ Simulação do modelo de Heston (Full Truncation scheme)
- ✅ Precificação via Monte Carlo
- ✅ Precificação semi-analítica via função característica
- ✅ Formulação numericamente estável ("Little Trap")
- ✅ Calibração a preços de mercado
- 📊 Visualização de trajetórias de preços e volatilidade

**Parâmetros do modelo**:
- `kappa`: velocidade de reversão à média
- `theta`: nível de volatilidade de longo prazo
- `sigma`: volatilidade da volatilidade (vol of vol)
- `rho`: correlação entre preço e volatilidade (efeito leverage)
- `v0`: volatilidade inicial

**Funções principais**:
```python
heston_model(S0, v0, rho, kappa, theta, sigma, r, T, dt, M)
heston_option_price_mc(S0, K, v0, rho, kappa, theta, sigma, r, lambd, T, dt, M)
heston_price_stable(S0, K, v0, kappa, theta, sigma, rho, lambd, T, r)
characteristic_function(phi, S0, v0, kappa, theta, sigma, rho, lambd, T, r)
```

### 4. Modelo de Kou ([`kou_jump_diffusion.py`](models/kou_jump_diffusion.py))

**Difusão com Saltos (Jump Diffusion)**

- ✅ Processos de Poisson para modelagem de saltos
- ✅ Distribuição dupla exponencial para tamanhos de saltos
- ✅ Simulação de trajetórias com saltos
- ✅ Precificação via Monte Carlo
- ✅ Precificação semi-analítica via função Upsilon
- ✅ Otimização com Numba para calibração rápida
- 📊 Visualização de PDFs de saltos e trajetórias

**Parâmetros do modelo**:
- `lambda`: intensidade dos saltos
- `p`: probabilidade de salto positivo
- `eta1`: parâmetro para saltos positivos (> 1)
- `eta2`: parâmetro para saltos negativos (> 0)
- `sigma`: volatilidade da difusão

**Funções principais**:
```python
kou_process(S0, mu, sigma, T, dt, eta1, eta2, p, lambd, M)
kou_option_price_mc(S0, K, r, sigma, T, dt, eta1, eta2, p, lambd, M)
kou_option_price_numba(S0, K, r, sigma, T, eta1, eta2, p, lambd)  # Versão otimizada
Upsilon_numba(x, T, mu, sigma, lambd, eta1, eta2, p)  # Precificação analítica
```

## 🚀 Setup e Uso

### 1. Configuração do Ambiente

```bash
# Configurar Python 3.13
pyenv install 3.13
pyenv local 3.13

# Instalar dependências
pip install -r requirements.txt
```

### 2. Coleta de Dados

**Processar histórico XML da B3**:
```bash
python data_collection/B3_options_negotiation.py
```

**Coletar dados do opcoes.net.br**:
```bash
python data_collection/opcoes_net.py
```

**Obter taxas do Tesouro Direto**:
```bash
python data_collection/tesouro_direto.py
```

### 3. Exemplos de Uso dos Modelos

#### Black-Scholes-Merton

```python
from models.black_scholes import black_scholes_price, implied_volatility, calculate_greeks

# Precificar opção
S0 = 100      # Preço spot
K = 100       # Strike
r = 0.05      # Taxa livre de risco
sigma = 0.2   # Volatilidade
T = 1.0       # Tempo até vencimento
option_type = OptionType.CALL

price = black_scholes_price(S0, K, r, sigma, T, option_type)

# Calcular volatilidade implícita
market_price = 10.45
iv = implied_volatility(S0, K, r, T, market_price, option_type)

# Calcular gregas
greeks = calculate_greeks(S0, K, r, sigma, T, option_type)
```

#### Modelo de Heston

```python
from models.heston_model import heston_model, heston_option_price_mc

# Parâmetros
v0 = 0.04       # Volatilidade inicial
rho = -0.7      # Correlação (efeito leverage)
kappa = 3.0     # Velocidade de reversão
theta = 0.04    # Vol. longo prazo
sigma = 0.6     # Vol of vol
lambd = 0.0     # Risk premium

# Simular trajetórias
t, S, v = heston_model(S0, v0, rho, kappa, theta, sigma, r, T, dt=0.01, M=1000)

# Precificar opção
price = heston_option_price_mc(S0, K, v0, rho, kappa, theta, sigma, r, lambd, T, dt=0.01, M=100000)
```

#### Modelo de Kou

```python
from models.kou_jump_diffusion import kou_process, kou_option_price_numba

# Parâmetros
eta1 = 10.0     # Saltos positivos
eta2 = 5.0      # Saltos negativos
p = 0.4         # Prob. salto positivo
lambd = 1.0     # Intensidade de saltos

# Simular trajetórias
t, S = kou_process(S0, r, sigma, T, dt=0.01, eta1, eta2, p, lambd, M=1000)

# Precificar opção (versão rápida com Numba)
price = kou_option_price_numba(S0, K, r, sigma, T, eta1, eta2, p, lambd)
```

## 📈 Aplicações

Este projeto implementa um sistema completo para:

### Análise de Mercado
- 📊 Coleta automatizada de dados históricos da B3
- 📈 Processamento e limpeza de dados de negociação
- 💹 Análise de volumes e preços de opções

### Modelagem Quantitativa
- 🎲 **Black-Scholes-Merton**: Modelo base para opções europeias
- 📉 **Heston**: Captura smile/skew de volatilidade
- 🔀 **Kou**: Modela eventos extremos e descontinuidades

### Precificação e Hedging
- 💰 Precificação de opções europeias
- 🔢 Cálculo de gregas para gestão de risco
- 🎯 Calibração a preços de mercado
- ⚖️ Estratégias de hedging

### Análise de Risco
- 📊 Análise de volatilidade implícita
- 🌡️ Superfícies de volatilidade
- 📈 Simulações Monte Carlo
- 🎲 Análise de cenários extremos (tail risk)

## 🔬 Performance e Otimizações

### Numba JIT Compilation
- ⚡ Speedup de 10-100x em funções críticas
- 🔄 Compilação automática na primeira execução
- 🚀 Ideal para calibração de modelos

### Vetorização NumPy
- 📊 Operações matriciais eficientes
- 🔢 Broadcast para múltiplas simulações
- 💾 Uso eficiente de memória

### Paralelização
- 🔀 Processamento paralelo com `prange`
- 💻 Aproveitamento de múltiplos cores
- ⚙️ Batch processing de múltiplos strikes

## 📋 Requisitos

### Software
- Python 3.13
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- Numba >= 0.57.0 (para otimizações)

### Hardware Recomendado
- CPU: Multi-core para paralelização
- RAM: 8GB+ para grandes simulações
- Armazenamento: SSD recomendado para I/O de dados

## 📚 Referências

### Artigos Científicos
- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- Kou, S. G. (2002). "A Jump-Diffusion Model for Option Pricing"
- Albrecher, H., et al. (2007). "The Little Heston Trap"

### Livros
- Hull, J. C. "Options, Futures, and Other Derivatives"
- Wilmott, P. "Paul Wilmott on Quantitative Finance"

## 🤝 Contribuindo

Este projeto faz parte de um TCC (Trabalho de Conclusão de Curso) sobre análise quantitativa do mercado de opções brasileiro.

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos e de pesquisa.

---

*Desenvolvido para análise quantitativa do mercado de opções brasileiro, integrando coleta de dados reais com modelos matemáticos avançados de precificação e gestão de risco.*