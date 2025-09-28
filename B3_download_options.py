from datetime import datetime, timedelta
from random import randint
from time import sleep
import requests
import json
import os


url = "https://arquivos.b3.com.br/bdi/table/export/csv?lang=pt-BR"


def gen_date_list(ini_date: str, end_date: str):
  start_date = datetime.strptime(ini_date, '%Y-%m-%d')
  end_date = datetime.strptime(end_date, '%Y-%m-%d')
  delta = timedelta(days=1)

  date_list = []
  while start_date <= end_date:
    date_list.append(start_date.strftime('%Y-%m-%d'))
    start_date += delta

  return date_list

def save_options_to_file(session: requests.Session, option: str, date: str, filename: str):
  sleep(randint(100, 3000)/1000) 
  payload = json.dumps({
    "Name": option,
    "Date": date,
    "FinalDate": date,
    "ClientId": "",
    "Filters": {}
  })
  headers = {
    'Content-Type': 'application/json',
  }

  response = session.post(url, headers=headers, data=payload)
  response.encoding = 'UTF-8'
  response_text = response.text
  response_text = response_text.split('Negócios Realizados;')[-1]
  response_text = response_text.split("(*) Lote de mil")[0].strip()

  os.makedirs(os.getcwd() + '/' + date, exist_ok=True)
  with open(filename, 'w', encoding='utf-8') as file:
    file.write(response_text)


# dates = ['2025-08-14']
dates = gen_date_list('2025-08-25', '2025-09-10')

# Create a session object
with requests.Session() as session:
  for date in dates:
    for option in ['SellingIndixesOptions', 'SellingOptions', 'PurchaseOptions', 'PurchaseIndixesOptions']:
      save_options_to_file(session, option, date, f'{date}/{option}.csv')