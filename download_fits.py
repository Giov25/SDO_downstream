import drms
import os
import requests
import re
from datetime import datetime, timedelta


email_address = "pippo.baudo0125@gmail.com"

client = drms.Client(email=email_address)
#choosing the series
ic = client.series(r'hmi.Ic_720s')[0]
m = client.series(r'hmi.M_720s')[0]
euv = client.series(r'aia.lev1_euv_12s')[0]
uv = client.series(r'aia.lev1_uv_24s')[0]
#collecting them in to a list
dati = [ic, m, euv, uv]
data_12min = []
data_24sec = []
data_12sec = []
for i in range(1,4):
    t_start = "2011.05.22_00:00:10"
    t_end = "2025.02.01_00:00:00"
    start_time = datetime.strptime(t_start, "%Y.%m.%d_%H:%M:%S")
    end_time = datetime.strptime(t_end, "%Y.%m.%d_%H:%M:%S")
    if i==1:
        while start_time <= end_time:
            period_end_time = start_time + timedelta(days=0) + timedelta(minutes=5)
            period_str = f"[{start_time.strftime('%Y.%m.%d_%H:%M:%S')}-{period_end_time.strftime('%Y.%m.%d_%H:%M:%S')}]"
            data_12min.append(period_str)
            start_time = start_time + timedelta(days=1)
    if i==2:
        while start_time <= end_time:
            period_end_time = start_time + timedelta(days=0) + timedelta(seconds=10)
            period_str = f"[{start_time.strftime('%Y.%m.%d_%H:%M:%S')}-{period_end_time.strftime('%Y.%m.%d_%H:%M:%S')}]"
            data_24sec.append(period_str)
            start_time = start_time + timedelta(days=1)
    
    if i==3:
        while start_time <= end_time:
            period_end_time = start_time + timedelta(days=0) + timedelta(seconds=0)
            period_str = f"[{start_time.strftime('%Y.%m.%d_%H:%M:%S')}-{period_end_time.strftime('%Y.%m.%d_%H:%M:%S')}]"
            data_12sec.append(period_str)
            start_time = start_time + timedelta(days=1)

print(len(data_12min))
def reformat_string(stringa):
    end_index = stringa.index("{")
    stringa = stringa[:end_index]
    part1 = stringa.split('[')[0].replace('.', '_').replace('_TAI', '') + '.fits'
    
    date_pattern = re.search(r'\[([0-9]{4}[-.][0-9]{2}[-.][0-9]{2})(T|_)[0-9]{2}:[0-9]{2}:[0-9]{2}(Z|_TAI)?\]', stringa)
    if len(stringa)>40:
        date_part = date_pattern.group(1).replace('.', '-') + stringa[date_pattern.end() - 11:date_pattern.end()].replace('_TAI', '')
    else:
            date_part = date_pattern.group(1).replace('.', '-') + 'T' + stringa[date_pattern.end() - 13:date_pattern.end()].replace('_TAI', 'Z')
    if stringa[0:6]=="hmi.Ic":
        part2 = '[' + date_part + '[Ic]'
    elif stringa[0:10]=="hmi.M_720s":
        part2 = '[' + date_part + '[M]'
    else:
        part2 = '[' + date_part + '[' + stringa.split('[')[2].split(']')[0] + ']'
    return part2 + part1



def process_export_requests(export_requests, Dic):
    """
    Funzione che processa le richieste di esportazione e popola il dizionario Dic.
    
    Parametri:
    - export_requests: Lista di tuple con richieste e tipi di dati associati.
    - Dic: Dizionario che memorizza i record e gli URL.
    """
    for export_request in export_requests:
        try:
            for url, record in zip(export_request.urls.url, export_request.data.record):
                if not url.endswith("spikes.fits"):
                    record = reformat_string(record)
                    Dic[record] = url
        except:continue

# Lista dei suffissi per le richieste
requests_data = [
    (ic + data_12min[i], data_12min[i]),
    (m + data_12min[i], data_12min[i]),
    (uv + data_24sec[i], data_24sec[i]),
    (euv + data_12sec[i], data_12sec[i])
]

Dic = {}
for i in range(len(data_12min)):
    export_requests = [
        client.export(ic + data_12min[i]),
        client.export(m + data_12min[i]),
        client.export(uv + data_24sec[i]),
        client.export(euv + data_12sec[i])
    ]
    
    process_export_requests(export_requests, Dic)
    
    
    
    
    
def download_file(url, file_name, folder_path):
    try:
        if not file_name.endswith(".fits"):
            file_name += ".fits"
        full_path = os.path.join(folder_path, file_name)
        response = requests.get(url)
        
        # Verifica che la richiesta sia andata a buon fine (status code 200)
        if response.status_code == 200:
            with open(full_path, 'wb') as file:
                file.write(response.content)
            
            print(f" {full_path}")
        else:
            print(f"Errore nello scaricare il file da {url}: Status code {response.status_code}")
    
    except Exception as e:
        print(f"Si è verificato un errore durante il download: {e}")

folder_path = "nuovo_donwload"
if not folder_path in os.listdir(os.getcwd()):
    output_dir=os.mkdir(os.path.join(os.getcwd(),folder_path))

for file_name, url in Dic.items():
    download_file(url, file_name, folder_path)
