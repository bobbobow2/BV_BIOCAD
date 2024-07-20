import requests
import os
links = pd.read_csv(r"C:\Users\edmeh\Downloads\opig.stats.ox.ac.uk_4th_Jul_2024.csv")
spis = []
baze = 'https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/'
for i in range(34610):
    if (baze in links.iloc[i,0]) and ("?raw=true" in links.iloc[i,0]):
        spis.append(links.iloc[i,0])

urls = spis
download_folder = r"C:\Users\edmeh\Downloads\Datasets_bio01\Structure"
os.makedirs(download_folder, exist_ok=True) 
for url in urls:
    filename = url.split('/')[-2]
    print(filename)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(download_folder, filename), 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f'Файл {filename} скачан.')
    else:
        print(f'Ошибка при скачивании файла {filename}.')
