import time
from enum import Enum
from datetime import datetime, timedelta
class OptionType(Enum):
    CALL = "call"
    PUT = "put"

def classify_option(ticker: str):
    call_maturities = ['A', 'B', 'C' ,'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    put_maturities = ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
    if ticker[4] in call_maturities:
        return OptionType.CALL
    elif ticker[4] in put_maturities:
        return OptionType.PUT
    else:
        raise ValueError("Ticker does not specify option type.")

colors = ['black', 'red', 'green', 'blue', 'olive', 'purple', 'orange', 'brown', 'pink', 'gray']

options_data = lambda database: f'../market options/Histórico B3/Negociações {database}.csv'


def gen_date_list(ini_date: str, end_date: str):
  start_date = datetime.strptime(ini_date, '%Y-%m-%d')
  end_date = datetime.strptime(end_date, '%Y-%m-%d')
  delta = timedelta(days=1)

  date_list = []
  while start_date <= end_date:
    date_list.append(start_date.strftime('%Y%m%d'))
    start_date += delta

  return date_list

def ndays(database:str, ndays: int):
    start_date = datetime.strptime(database, '%Y-%m-%d')

    return (start_date + timedelta(days=ndays)).strftime('%Y-%m-%d')

def measure(func):
    start_time = time.time()
    res = func()
    end_time = time.time()
    diff = end_time - start_time
    if diff < 60 * 2:
        print(f"Elapsed time for {func.__name__}: {round(diff, 2)} seconds")
    else:
        print(f"Elapsed time for {func.__name__}: {round((diff)/60, 2)} minutes")
    return res