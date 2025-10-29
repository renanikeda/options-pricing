# Options Pricing - B3 Data Collection and Mathematical Models

Este projeto realiza o download e tratamento de dados de negociações de opções da B3 (Brasil, Bolsa, Balcão), além de implementar modelos matemáticos para precificação de opções.

## 📁 Estrutura do Projeto

```
├── B3_negotiation_hist_xml.py         # Processamento de histórico XML da B3 (atual)
├── B3_negotiation_hist_xml_old.py     # Versão antiga do processamento XML
├── opcoes_net_hist_price.py           # Coleta de dados do opcoes.net.br
├── black_scholes_merton.py            # Modelo Black-Scholes-Merton
├── brownian_motion.py                 # Simulação de Movimento Browniano
├── kou_jump_diffusion.py              # Modelo de Difusão com Saltos de Kou
├── utils.py                           # Funções utilitárias
├── interested_merged_deals.csv        # Dados consolidados de interesse
├── requirements.txt                   # Dependências do projeto
├── README.md
└── Histórico B3/                      # Dados processados da B3
    ├── Negociações 20200102.csv
    ├── Negociações 20200103.csv
    └── ... (arquivos diários)
```

## 🔧 Módulos de Coleta de Dados

### [`B3_negotiation_hist_xml.py`](B3_negotiation_hist_xml.py)
- **Função**: Processamento de dados históricos em formato XML da B3
- **Entrada**: Arquivos XML históricos da B3
- **Processamento**: Conversão e limpeza de dados XML
- **Output**: Arquivos CSV estruturados no diretório [`Histórico B3/`](Histórico%20B3/)

### [`opcoes_net_hist_price.py`](opcoes_net_hist_price.py)
- **Função**: Coleta de histórico de preços por ticker
- **Fonte**: opcoes.net.br
- **Input**: Ticker da opção
- **Output**: Dados históricos de preços

## 🧮 Modelos Matemáticos

### [`black_scholes_merton.py`](black_scholes_merton.py)
- **Modelo**: Black-Scholes-Merton
- **Função**: Precificação clássica de opções europeias
- **Features**: Cálculo de preços e gregas

### [`brownian_motion.py`](brownian_motion.py)
- **Modelo**: Movimento Browniano Geométrico
- **Função**: Simulação de trajetórias de preços de ativos
- **Aplicação**: Base para simulações Monte Carlo

### [`kou_jump_diffusion.py`](kou_jump_diffusion.py)
- **Modelo**: Difusão com Saltos de Kou
- **Função**: Modelagem de preços com descontinuidades (saltos)
- **Aplicação**: Precificação mais realista considerando eventos extremos

## 🛠️ Utilitários

### [`utils.py`](utils.py)
- **Função**: Funções auxiliares compartilhadas
- **Conteúdo**: Utilitários matemáticos e de processamento de dados

## 📊 Dados

- **[`interested_merged_deals.csv`](interested_merged_deals.csv)**: Dados consolidados de negociações de interesse
- **[`Histórico B3/`](Histórico%20B3/)**: Diretório com arquivos CSV diários de negociações processadas
- Formato dos arquivos: `Negociações YYYYMMDD.csv`

## Setup

1. **Configurar ambiente Python**:
   ```bash
   pyenv install 3.13
   pyenv local 3.13
   ```

2. **Instalar dependências**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Como Usar

### Coleta de Dados

1. **Para processar histórico XML da B3**:
   ```bash
   python B3_negotiation_hist_xml.py
   ```

2. **Para dados do opcoes.net.br**:
   ```bash
   python opcoes_net_hist_price.py
   ```

### Modelos de Precificação

1. **Modelo Black-Scholes-Merton**:
   ```bash
   python black_scholes_merton.py
   ```

2. **Simulação Browniana**:
   ```bash
   python brownian_motion.py
   ```

3. **Modelo com Saltos de Kou**:
   ```bash
   python kou_jump_diffusion.py
   ```

## 📈 Aplicação

Este projeto implementa um sistema completo para:
- **Coleta de dados**: Download e processamento de dados históricos da B3
- **Modelagem matemática**: Implementação de modelos de precificação de opções
- **Análise quantitativa**: Ferramentas para análise de volatilidade e risco
- **Simulações**: Geração de cenários para precificação e hedging

## 📋 Requisitos

- Python 3.13
- Bibliotecas científicas (NumPy, Pandas, etc. - ver [`requirements.txt`](requirements.txt))
- Acesso à internet para download de dados

---

*Este projeto faz parte do desenvolvimento de ferramentas para análise quantitativa do mercado de opções brasileiro, integrando coleta de dados reais com modelos matemáticos avançados.*