from B3_options import get_options_treated
import os
import subprocess
from datetime import datetime, timedelta
from time import sleep


interested_tickers = [r'IBOV.*', r'PETR.*', r'VALE.*', r'BOVA.*']

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

def compress_data():
  command = r'tar -czvf "Histórico B3.tar.gz" "Histórico B3"'
  result = subprocess.run(command, shell=True, capture_output=True, text=True)

  if result.returncode == 0:
      print("Archive created successfully!")
  else:
      print(f"Error: {result.stderr}")

if __name__ == "__main__":

  date_ini = '2026-03-25'
  date_end = '2026-03-30'
  # database = '20200901'
  output = 'Histórico B3'

  if os.path.exists(output) == False:
    os.mkdir(output)
  
  databases = list(filter(lambda database: not  os.path.exists(f'{output}/Negociações {database}.csv'), gen_date_list(date_ini, date_end)))
  # databases = gen_date_list(date_ini, date_end)
  print(databases)
  for database in databases:
    sleep(0.15)  # To avoid overloading the server with requests
    result = get_options_treated(database)
    if not result.empty:
        print(f'Saving in {output}/Negociações {database}.csv')
        result.to_csv(f'{output}/Negociações {database}.csv', index=False)
    else:
        print("Empty result")
#   compress_data() 