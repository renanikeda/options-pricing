from B3_options_information import get_options_info_treated
from B3_negotiation_hist_xml import get_negotiation_treated
import os
import pandas as pd
from datetime import datetime, timedelta


interested_tickers = [r'IBOV.*', r'PETR.*', r'VALE.*', r'BOVA11.*']

def gen_date_list(ini_date: str, end_date: str):
  start_date = datetime.strptime(ini_date, '%Y-%m-%d')
  end_date = datetime.strptime(end_date, '%Y-%m-%d')
  delta = timedelta(days=1)

  date_list = []
  while start_date <= end_date:
    date_list.append(start_date.strftime('%Y%m%d'))
    start_date += delta

  return date_list


if __name__ == "__main__":

  date_ini = '2020-07-01'
  date_end = '2021-01-01'
  # database = '20200901'
  output = 'Histórico B3'

  databases = list(filter(lambda database: not  os.path.exists(f'{output}/Negociações {database}.csv'), gen_date_list(date_ini, date_end)))
  # databases = gen_date_list(date_ini, date_end)
  print(databases)
  for database in databases:
    negotiation_df = get_negotiation_treated(database)
    negotiation_df.rename(columns={'TradeDate': 'Data Base'}, inplace=True)
    infos_df = get_options_info_treated(database)
    if negotiation_df.empty or infos_df.empty:
        continue
    result = (negotiation_df.join(infos_df.set_index(['Data Base', 'Ticker']), on =['Data Base', 'Ticker']))


    result.to_csv(f'{output}/Negociações {database}.csv', index=False)
    # writer = pd.ExcelWriter(f'{output}/Negociações {database}.xlsx', engine = 'openpyxl', mode = 'w')
    # negotiation_df.to_excel(writer, sheet_name="Negociação", index=False)
    # infos_df.to_excel(writer, sheet_name="Infos", index=False)
    # result.to_excel(writer, sheet_name="Join", index=False)
    # writer.close()