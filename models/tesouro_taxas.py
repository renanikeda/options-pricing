import pandas as pd
from enum import Enum
from datetime import datetime
from typing import Optional
from bcb import sgs


url_taxas = 'https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv'

## example tesouro_taxas.loc[('Tesouro Prefixado', '2025-09-01', '2027-01-01')] # Tipo Tesouro Prefixado Data Base 2025-09-01 Data Vencimento 2027-01-01
#selic = sgs.get(('selic', 432), start = '2020-01-01')


class TipoTitulo(Enum):
    PREFIXADO = "Tesouro Prefixado"
    Selic = "Tesouro Selic"
    IPCA = "Tesouro IPCA+"

class TesouroTaxas:
    def __init__(self):
        self.tesouro_taxas = self.get_tesouro_taxas()

    def get_tesouro_taxas(self):
        tesouro_taxas  = pd.read_csv(url_taxas, sep=';', decimal=',')
        tesouro_taxas['Data Vencimento'] = pd.to_datetime(tesouro_taxas['Data Vencimento'], dayfirst=True)
        tesouro_taxas['Data Base']       = pd.to_datetime(tesouro_taxas['Data Base'], dayfirst=True)
        cols = tesouro_taxas.columns.tolist()
        # Assuming the columns are in positions that need swapping
        tesouro_taxas = tesouro_taxas[['Tipo Titulo', 'Data Base', 'Data Vencimento'] + [col for col in cols if col not in ['Data Base', 'Tipo Titulo', 'Data Vencimento']]]

        multi_indice = pd.MultiIndex.from_frame(tesouro_taxas.iloc[:, :3])
        tesouro_taxas = tesouro_taxas.set_index(multi_indice).iloc[: , 3:]
        return tesouro_taxas
    
    def valid_date(self, date_text: str):
        try:
            datetime.strptime(date_text, '%Y-%m-%d')
            return
        except ValueError:
            ValueError("Dates must be in 'YYYY-MM-DD' format")
        
    def get_taxa(self, tipo_titulo: TipoTitulo, data_base: str, data_vencimento: Optional[str] = None):
        self.valid_date(data_base)
        if data_vencimento: self.valid_date(data_vencimento)
        try:
            rate = self.tesouro_taxas.loc[(tipo_titulo.value, data_base, data_vencimento)]['Taxa Compra Manha'] if data_vencimento else self.tesouro_taxas.loc[(tipo_titulo.value, data_base)]['Taxa Compra Manha']
            return rate
        except KeyError:
            return None
        
    def get_tesouro_vencimentos(self, tipo_titulo: TipoTitulo, data_base: str):
        self.valid_date(data_base)
        try:
            vencimentos = self.tesouro_taxas.loc[(tipo_titulo.value, data_base)].index.tolist()
            return vencimentos
        except KeyError:
            return []
        
    def get_selic(self, date: str):
        self.valid_date(date)
        return sgs.get(('selic', 432), start = date, end = date)['selic'].iloc[0]

if __name__ == "__main__":
    tesouro = TesouroTaxas()
    taxa = tesouro.get_taxa(TipoTitulo.PREFIXADO, '2025-01-03')
    print(f'Prefixado:\n{taxa}')
    vencimentos = tesouro.get_tesouro_vencimentos(TipoTitulo.PREFIXADO, '2025-01-03')
    print(f'Vencimentos:\n{vencimentos}')
    selic_rate = tesouro.get_selic('2025-01-03')
    print(f'Selic Rate:\n{selic_rate}')