
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from random import randint
from time import sleep
import os 

def get_options_info(date: str, tipo: str = 'Empresa'):
  """
  Get options information from B3 for a given date and type.
  Strike and Maturity Date
  """
  
  url = f'https://www.b3.com.br/json/{date}/Posicoes/{tipo}/SI_C_OPCPOSAB{tipo.upper()[:3]}.json'
  try:
    sleep(randint(10, 100)/1000) 
    headers = {}
    response = requests.get(url, headers=headers, verify=True)
    
    #result date %Y%m%d
    return response.json().get(tipo)
  except Exception as e:
    print(f"Error fetching options info for date {date}: {e}")
    return None

def treat_options_info(dict_info: Dict[str, List[Dict]], database: str):
  interested_tickers = [r'IBOV.*', r'PETR.*', r'VALE.*', r'BOVA.*']
  result_list = []
  if isinstance(dict_info, dict):
    for value in dict_info.values():
      result_list.extend(value)
  else:
    result_list = dict_info
  df = pd.DataFrame(result_list)
  df = df[['dtVen', 'prEx', 'ser']]
  df.insert(0, 'Data Base', datetime.strptime(database, '%Y%m%d').strftime('%Y-%m-%d'))
  df.columns = ['Data Base', 'Maturity Date', 'Strike', 'Ticker']
  df['Maturity Date'] = pd.to_datetime(df['Maturity Date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
  df = df[df['Ticker'].str.contains('|'.join(interested_tickers), regex=True)] 
  return df 

def get_options_info_treated(database: str):
  final_df_info = pd.DataFrame()
  for tipo in ['Empresa', 'Indice']:
    dict_info = get_options_info(database, tipo)
    if dict_info is None: continue
    df_info = treat_options_info(dict_info, database)
    final_df_info = pd.concat([final_df_info, df_info], ignore_index=True)  
  
  return final_df_info

def merge_all_deals(root_path: str, output_path: str):
    all_files = [os.path.join(root_path, filename) for filename in os.listdir(root_path) if filename.endswith('.csv') and 'Infos' in filename]
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file, sep=',', encoding='latin1', decimal='.')
            if df.empty:
                continue
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.sort_values(by=['Data Base'], inplace=True)
        merged_df.to_csv(output_path, index=False)
        return
    else:
        print("No CSV files found or all files failed to read.")
        return pd.DataFrame()

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
  date_ini = '2025-10-01'
  date_end = '2025-12-01'
  # database = '20200901'
  output = 'Histórico B3'

  databases = list(filter(lambda database: not  os.path.exists(f'{output}/Infos {database}.csv'), gen_date_list(date_ini, date_end)))

  for database in databases:
    final_df_info = pd.DataFrame()
    for tipo in ['Empresa', 'Indice']:
      dict_info = get_options_info(database, tipo)
      if dict_info is None: continue
      df_info = treat_options_info(dict_info, database)
      final_df_info = pd.concat([final_df_info, df_info], ignore_index=True)
    if not final_df_info.empty: final_df_info.to_csv(f'{output}/Infos {database}.csv', index=False)

  merge_all_deals(output, 'b3_options_info.csv')


