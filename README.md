# Options Pricing - B3 Data Collection and Processing

Este projeto realiza o download e tratamento de dados de negociações de opções da B3 (Brasil, Bolsa, Balcão), coletando informações de diferentes fontes para análise de precificação de opções.

## 📁 Estrutura do Projeto

```
├── B3_download_options.py          # Download de opções B3 do último mês
├── B3_negotiation_hist_xml.py      # Processamento de histórico XML da B3
├── opcoes_net_hist_price.py        # Coleta de dados do opcoes.net.br
├── README.md
├── Histórico B3/                   # Dados processados da B3
│   ├── merged_deals.csv
│   └── Negociações YYYYMMDD.csv
└── Histórico B3 old/               # Arquivos históricos antigos
```

## 🔧 Módulos

### [`B3_download_options.py`](B3_download_options.py)
- **Função**: Download de negociações de opções da B3 do último mês
- **Fonte**: API/site oficial da B3
- **Output**: Arquivos CSV com dados recentes de negociações

### [`B3_negotiation_hist_xml.py`](B3_negotiation_hist_xml.py)
- **Função**: Processamento de dados históricos em formato XML da B3
- **Entrada**: Arquivos XML históricos da B3
- **Processamento**: Conversão e limpeza de dados XML
- **Output**: Arquivos CSV estruturados no diretório `Histórico B3/`

### [`opcoes_net_hist_price.py`](opcoes_net_hist_price.py)
- **Função**: Coleta de histórico de preços por ticker
- **Fonte**: opcoes.net.br
- **Input**: Ticker da opção
- **Output**: Dados históricos de preços

## 📊 Dados Coletados

Os arquivos CSV gerados contêm informações sobre:
- Negociações diárias de opções
- Preços históricos
- Volume de negociação
- Dados consolidados em [`merged_deals.csv`](Histórico%20B3/merged_deals.csv)


## Setup
1. pyenv install 3.13
2. pyenv local 3.13
3. pip install -r requirements.txt

## 🚀 Como Usar

1. **Para dados recentes da B3**:
   ```bash
   python B3_download_options.py
   ```

2. **Para processar histórico XML**:
   ```bash
   python B3_negotiation_hist_xml.py
   ```

3. **Para dados do opcoes.net.br**:
   ```bash
   python opcoes_net_hist_price.py
   ```

## 📈 Aplicação

Este projeto é parte de um sistema de análise e precificação de opções, fornecendo dados históricos e atuais necessários para:
- Modelagem de preços de opções
- Análise de volatilidade
- Estudos de mercado de derivativos

## 📋 Requisitos

- Python 3.x
- Bibliotecas para processamento XML e CSV
- Acesso à internet para download de dados

---

*Este projeto faz parte do desenvolvimento de ferramentas para análise quantitativa do mercado de opções brasileiro.*