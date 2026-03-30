import requests
import zipfile
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Literal
import os
from time import sleep
import shutil

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

def parse_products_xml_to_df(xml_path: str) -> pd.DataFrame:
    
    root = parse_xml(xml_path)
    if root is None: return pd.DataFrame()  # Return empty DataFrame on parse error
    ns = {
        "bvmf052": "urn:bvmf.052.01.xsd",
        "head": "urn:iso:std:iso:20022:tech:xsd:head.001.001.01",
        "bvmf100": "urn:bvmf.100.02.xsd"
    }

    rows = []

    # Loop through all price reports
    for pricrpt in root.findall(".//bvmf100:Instrm", ns):
        record = {}
        # Extract fields
        record["Ticker"] = pricrpt.findtext("bvmf100:InstrmInf/bvmf100:OptnOnEqtsInf/bvmf100:TckrSymb", namespaces=ns)
        if not record["Ticker"]: continue
        record["Ação"] = pricrpt.findtext("bvmf100:FinInstrmAttrCmon/bvmf100:Asst", namespaces=ns)
        record["ID ação"] = pricrpt.findtext("bvmf100:InstrmInf/bvmf100:OptnOnEqtsInf/bvmf100:UndrlygInstrmId/bvmf100:OthrId/bvmf100:Id", namespaces=ns)
        record["Style"] = pricrpt.findtext("bvmf100:InstrmInf/bvmf100:OptnOnEqtsInf/bvmf100:OptnStyle", namespaces=ns)
        record["Type"] = pricrpt.findtext("bvmf100:InstrmInf/bvmf100:OptnOnEqtsInf/bvmf100:OptnTp", namespaces=ns)
        record["Strike"] = pricrpt.findtext("bvmf100:InstrmInf/bvmf100:OptnOnEqtsInf/bvmf100:ExrcPric", namespaces=ns)
        record["Maturity"] = pricrpt.findtext("bvmf100:InstrmInf/bvmf100:OptnOnEqtsInf/bvmf100:XprtnDt", namespaces=ns)
        record["Descrição"] = pricrpt.findtext("bvmf100:FinInstrmAttrCmon/bvmf100:Desc", namespaces=ns)
        rows.append(record)

    # Convert to DataFrame
    return pd.DataFrame(rows)

def parse_negotiation_xml_to_df(xml_path: str) -> pd.DataFrame:
    
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
        record["ID"] = pricrpt.findtext("bvmf217:FinInstrmId/bvmf217:OthrId/bvmf217:Id", namespaces=ns)
        record["TradeDate"] = pricrpt.findtext("bvmf217:TradDt/bvmf217:Dt", namespaces=ns)
        record["Ticker"] = pricrpt.findtext("bvmf217:SctyId/bvmf217:TckrSymb", namespaces=ns)
        record["FirstPrice"] = pricrpt.findtext("bvmf217:FinInstrmAttrbts/bvmf217:FrstPric", namespaces=ns)
        if not record["FirstPrice"]: continue
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
        response = requests.get(url, headers=headers, timeout=10)
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
            # print("Zip file contents:")
            # if zip_ref.namelist():
            #     os.makedirs(extract_to, exist_ok=True)
            # for filename in zip_ref.namelist():
            #     print(f"  - {filename}")
                
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
        # return None
        raise zipfile.BadZipFile
    except IOError as e:
        print(f"Error extracting file: {e}")
        return None


def merge_all_deals(root_path: str, output_path: str, ticker_regex:str = ''):
    all_files = [os.path.join(root_path, filename) for filename in os.listdir(root_path) if filename.endswith('.csv') and 'Negociações' in filename]
    df_list = []
    for file in all_files[:10]:
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

def get_b3_info(database: str, output: str = '.', type: Literal['negotiation', 'product'] = 'negotiation', retries = 0):
    try:
        interested_tickers = [r'IBOV.*', r'PETR.*', r'VALE.*', r'BOVA.*']
        ticker_regex = '|'.join(interested_tickers)
        prefix = 'PR' if type == 'negotiation' else 'IN'
        code = prefix + database[2:]
        print(code)
        zip_path = download_and_save_zipfile(code)
        extracted_files = unzip_file(zip_path, f"{output}/{code}") if zip_path else None
        print(extracted_files)
        result = pd.DataFrame()
        if not extracted_files: 
            return result 
        file = extracted_files[-1]
        if file.endswith('.xml'):
            result = parse_negotiation_xml_to_df(file) if type == 'negotiation' else parse_products_xml_to_df(file)
        shutil.rmtree('/'.join(file.split('/')[:-1]))
        result = result[result['Ticker'].str.contains(ticker_regex, regex=True)]
        return result
    except zipfile.BadZipFile:
        if retries > 3:
            print('Max retries reached. Exiting.')
            return pd.DataFrame()
        retries += 1 
        print(f'Bad zip file encountered. Trying again {retries}...')
        sleep_time = 0.5*retries
        sleep(sleep_time)
        return get_b3_info(database, output, type, retries)
    except:
        return pd.DataFrame()

def get_options_treated(database: str, output: str = '.'):
    try:
        negotiations = get_b3_info(database, output, 'negotiation')
        # print(negotiations)
        if negotiations.empty:
            return pd.DataFrame()
        products = get_b3_info(database, output, 'product')
        # print(products)
        if products.empty:
            return pd.DataFrame()
        products = products.merge(negotiations[['ID', 'Ticker']], left_on='ID ação', right_on='ID', how='left')
        products.rename(columns={'Ticker_x': 'Ticker', 'Ticker_y': 'Asset Ticker'}, inplace=True)
        final_df = negotiations.merge(products[['Ticker', 'Style', 'Type', 'Strike', 'Maturity', 'Asset Ticker']], on='Ticker', how='left')
        final_df.drop(columns=['ID'], inplace=True)
        final_df.loc[final_df['Ticker'].str.contains('IBOV') & ~final_df['Strike'].isna(), 'Asset Ticker'] = 'BOVA11'
        return final_df

    except:
        return pd.DataFrame()

if __name__ == "__main__":
    database = '20250616'
    output = '.'
    result = get_options_treated(database, output)
    result.to_csv(f"./Negociações {database}.csv", index=False)

# if __name__ == "__main__":
#     code = 'IN250616'
#     output = '.'
#     zip_path = download_and_save_zipfile(code)
#     extracted_files = unzip_file(zip_path, f"{output}/{code}", True)
#     print(extracted_files)
#     file = extracted_files[-1]
#     if file.endswith('.xml'):
#         df_negotiation = parse_information_xml_to_df(file)
#         date = code.replace('IN', '20')
#         output_file = f"./Informações {date}.csv"
#         df_negotiation.to_csv(output_file, index=False)

#     code = 'PR250616'
#     output = '.'
#     zip_path = download_and_save_zipfile(code)
#     extracted_files = unzip_file(zip_path, f"{output}/{code}", True)
#     print(extracted_files)
#     file = extracted_files[-1]
#     if file.endswith('.xml'):
#         df_negotiation = parse_negotiation_xml_to_df(file)
#         date = code.replace('PR', '20')
#         output_file = f"./Negociações {date}.csv"
#         df_negotiation.to_csv(output_file, index=False)