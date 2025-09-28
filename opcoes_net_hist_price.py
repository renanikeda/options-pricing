import requests
from random import randint
from time import sleep

# url = "https://opcoes.net.br/chartData/json/IBOVV139?c=2025-9-13-2"
url = "https://opcoes.net.br/chartData/json"
# response [Data Hora negociação, Premio, Preço ativo base, Vol implicita]

def get_hist_options_price(session: requests.Session, option: str):
  sleep(randint(100, 3000)/1000) 
  payload = {}
  headers = {
    'accept': 'application/json, */*; q=0.01',
    'accept-language': 'pt-BR,pt;q=0.9',
    'priority': 'u=1, i',
    'referer': f'https://opcoes.net.br/{option}',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Linux"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'x-requested-with': 'XMLHttpRequest'
  }
  response = session.get(f'{url}/{option}', headers=headers, data=payload)

  return response.json().get('chartData')

with requests.Session() as session:
  print(get_hist_options_price(session, 'IBOVV139'))