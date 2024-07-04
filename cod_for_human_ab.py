import requests
import os
import pandas as pd
import numpy as np
import gzip
## рабочая
def ustanovka_data_human_antibody(spis):
    def unzip_gz(archive_path, extract_path):
        with gzip.open(archive_path, 'rb') as f_in:
            with open(os.path.join(extract_path, os.path.basename(archive_path).replace('.gz', '')), 'wb') as f_out:
                f_out.write(f_in.read())
    data_it = pd.DataFrame() # Создаем пустой DataFrame для склеивания
    for i, url in enumerate(spis): # Используем enumerate для доступа к индексу и значению
        response = requests.get(url)
        if response.status_code == 200:
            file_Path = os.path.join(r"C:\Users\edmeh\Downloads\Datasets_bio01\zip_bd", f"{i}.gz")
            file_zip_Path = os.path.join(r"C:\Users\edmeh\Downloads\Datasets_bio01\unpacking_bd", str(i))
            print(file_zip_Path)
            os.makedirs(file_zip_Path, exist_ok=True)
            with open(file_Path, 'wb') as file:
                file.write(response.content)
            unzip_gz(file_Path, file_zip_Path)
            data = pd.read_csv(os.path.join(file_zip_Path, f"{i}"))
            # Обработка данных из файла
            data = data.reset_index()
            data = data.iloc[:, :len(data.columns)- 1]
            col = data.iloc[0].tolist()
            data = data.iloc[1:,:]
            data.columns = col
            data = data[['sequence_alignment_aa_light', 'sequence_alignment_aa_heavy']]
            data_it = pd.concat([data_it, data], ignore_index=True)
            print(f'Файл {i} успешно загружен, разархивирован и обработан')
            print('------------------------->>>>   ', str(i)*20)
        else:
            print(f'Не удалось загрузить файл {i}')
    data_it.to_csv(r'C:\Users\edmeh\Downloads\Datasets_bio01\data_human_antibody.csv')
    return data_it
