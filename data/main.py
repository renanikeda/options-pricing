from B3_options import get_options_treated
import os
import pandas as pd
from datetime import datetime, timedelta


interested_tickers = [r'IBOV.*', r'PETR.*', r'VALE.*', r'BOVA11.*']

def gen_date_list(ini_date: str, end_date: str):
    """
    Generate list of dates between ini_date and end_date, excluding weekends.
    
    Parameters:
    ini_date (str): start date in format 'YYYY-MM-DD'
    end_date (str): end date in format 'YYYY-MM-DD'
    
    Returns:
    list: list of dates in format 'YYYYMMDD' (weekdays only)
    """
    start_date = datetime.strptime(ini_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    delta = timedelta(days=1)

    date_list = []
    while start_date <= end_date:
        if start_date.weekday() < 5:  # Monday=0, Friday=4
            date_list.append(start_date.strftime('%Y%m%d'))
        start_date += delta

    return date_list


if __name__ == "__main__":

  date_ini = '2025-01-01'
  date_end = '2026-01-01'
  # database = '20200901'
  output = 'Histórico B3'

  if os.path.exists(output) == False:
    os.mkdir(output)

  databases = list(filter(lambda database: not  os.path.exists(f'{output}/Negociações {database}.csv'), gen_date_list(date_ini, date_end)))
  # databases = gen_date_list(date_ini, date_end)
  print(databases)
  for database in databases:
    result = get_options_treated(database)
    if not result.empty:  result.to_csv(f'{output}/Negociações {database}.csv', index=False)
