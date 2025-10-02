import requests
import zipfile
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import os
import shutil

def gen_date_list_code(ini_date: str, end_date: str):
  start_date = datetime.strptime(ini_date, '%Y-%m-%d')
  end_date = datetime.strptime(end_date, '%Y-%m-%d')
  delta = timedelta(days=1)

  date_list = []
  while start_date <= end_date:
    date_list.append("PR"+start_date.strftime('%y%m%d'))
    start_date += delta

  return date_list

def parse_xml(xml_path: str):
    """
    Parse an XML file and return its root element
    
    Args:
        xml_path: Path to the XML file
    
    Returns:
        Element: Root element of the parsed XML tree
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return root
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except IOError as e:
        print(f"Error reading file: {e}")
        return None

def parse_xml_to_df(xml_path: str):
    
    root = parse_xml(xml_path)
    if root is None: return pd.DataFrame()  # Return empty DataFrame on parse error
    ns = {
        "bvmf052": "urn:bvmf.052.01.xsd",
        "head": "urn:iso:std:iso:20022:tech:xsd:head.001.001.01",
        "bvmf217": "urn:bvmf.217.01.xsd"
    }

    rows = []

    # Loop through all price reports
    for pricrpt in root.findall(".//bvmf217:PricRpt", ns):
        record = {}
        
        # Extract fields
        record["TradeDate"] = pricrpt.findtext("bvmf217:TradDt/bvmf217:Dt", namespaces=ns)
        record["Ticker"] = pricrpt.findtext("bvmf217:SctyId/bvmf217:TckrSymb", namespaces=ns)
        record["FirstPrice"] = pricrpt.findtext("bvmf217:FinInstrmAttrbts/bvmf217:FrstPric", namespaces=ns)
        record["MinPrice"] = pricrpt.findtext("bvmf217:FinInstrmAttrbts/bvmf217:MinPric", namespaces=ns)
        record["MaxPrice"] = pricrpt.findtext("bvmf217:FinInstrmAttrbts/bvmf217:MaxPric", namespaces=ns)
        record["LastPrice"] = pricrpt.findtext("bvmf217:FinInstrmAttrbts/bvmf217:LastPric", namespaces=ns)
        record["AvgPrice"] = pricrpt.findtext("bvmf217:FinInstrmAttrbts/bvmf217:TradAvrgPric", namespaces=ns)
        record["OscnPctg"] = pricrpt.findtext("bvmf217:FinInstrmAttrbts/bvmf217:OscnPctg", namespaces=ns)
        record["TradeQty"] = pricrpt.findtext("bvmf217:TradDtls/bvmf217:TradQty", namespaces=ns)
        record["TradeAmount"] = pricrpt.findtext("bvmf217:FinInstrmAttrbts/bvmf217:RglrTraddCtrcts", namespaces=ns)
        rows.append(record)

    # Convert to DataFrame
    return pd.DataFrame(rows)


def download_and_save_zipfile(file_code: str, save_path: str = None):
    """
    Download zip file from B3 and save it locally
    
    Args:
        file_code: The file code (e.g., 'PR250401')
        save_path: Path to save the zip file (optional, defaults to current directory)
    
    Returns:
        str: Path to the saved zip file if successful, None if failed
    """
    headers = {
        'referer': 'https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/boletins-diarios/pesquisa-por-pregao/pesquisa-por-pregao/',
    }
    
    url = f'https://www.b3.com.br/pesquisapregao/download?filelist={file_code}.zip,'
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        if save_path is None:
            save_path = f'{file_code}.zip'
        elif os.path.isdir(save_path):
            save_path = os.path.join(save_path, f'{file_code}.zip')
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        with open(save_path, 'wb') as file:
            file.write(response.content)
        
        print(f"Successfully downloaded and saved: {save_path}")
        return save_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None
    except IOError as e:
        print(f"Error saving file: {e}")
        return None

def unzip_file(zip_path: str, extract_to: str = None, remove_zip: bool = True):
    """
    Extract all files from a zip archive
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract files to (optional, defaults to same directory as zip)
    
    Returns:
        list: List of extracted file paths if successful, None if failed
    """
    try:
        if extract_to is None:
            extract_to = os.path.dirname(zip_path) or '.'
        
        # os.makedirs(extract_to, exist_ok=True)
        
        extracted_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Extracting {zip_path} to {extract_to}")
            print("Zip file contents:")
            if zip_ref.namelist():
                os.makedirs(extract_to, exist_ok=True)
            for filename in zip_ref.namelist():
                print(f"  - {filename}")
                
            zip_ref.extractall(extract_to)
            
            for filename in zip_ref.namelist():
                extracted_file_path = os.path.join(extract_to, filename)
                print("extracted_file_path: ", extracted_file_path)
                if '.zip' in extracted_file_path:
                    print("Found nested zip file, extracting it...")
                    extracted_file_path = unzip_file(extracted_file_path, '/'.join(extracted_file_path.split('/')[:-1]))
                    print(extracted_file_path)
                extracted_files.append(extracted_file_path) if isinstance(extracted_file_path, str) else extracted_files.extend(extracted_file_path)
                
        os.remove(zip_path) if remove_zip else None
        return extracted_files
        
    except zipfile.BadZipFile:
        print("Error: File is not a valid zip file")
        return None
    except IOError as e:
        print(f"Error extracting file: {e}")
        return None


def merge_all_deals(root_path: str, output_path: str, ticker_regex:str = ''):
    all_files = [os.path.join(root_path, filename) for filename in os.listdir(root_path) if filename.endswith('.csv') and 'Negociações' in filename]
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file, sep=',', encoding='latin1', decimal='.')
            if df.empty:
                continue
            filtered_df = df[df['Ticker'].str.contains(ticker_regex, regex=True)] if ticker_regex else df
            filtered_df = filtered_df[filtered_df['TradeQty'] > 0]
            df_list.append(filtered_df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.sort_values(by=['TradeDate'], inplace=True)
        merged_df.to_csv(output_path, index=False)
        return
    else:
        print("No CSV files found or all files failed to read.")
        return pd.DataFrame()


interested_tickers = [r'IBOV.*', r'PETR.*', r'VALE.*', r'BOVA11.*']
date_ini = '2025-01-01'
date_end = '2025-10-01'
output = 'Histórico B3'

codes = list(filter(lambda code: not  os.path.exists(f'{output}/{code.replace('PR', 'Negociações 20')}.csv'), gen_date_list_code(date_ini, date_end)))

print(codes)

for code in codes:
    zip_path = download_and_save_zipfile(code)
    extracted_files = unzip_file(zip_path, f"{output}/{code}") if zip_path else None
    print(extracted_files)
    if not extracted_files:
        continue
    file = extracted_files[-1]
    if file.endswith('.xml'):
        df = parse_xml_to_df(file)
        date = code.replace('PR', '20')
        output_file = f"{output}/Negociações {date}.csv"
        df.to_csv(output_file, index=False)
    shutil.rmtree('/'.join(file.split('/')[:-1]))

output_path = 'interested_merged_deals.csv'
merge_all_deals(output, output_path, '|'.join(interested_tickers))