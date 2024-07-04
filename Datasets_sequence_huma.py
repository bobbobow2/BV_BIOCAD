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
            data_ful_sq = data
            data_ful_sq = data_ful_sq.reset_index()
            data_ful_sq = data_ful_sq.iloc[:, :len(data.columns)- 1]
            col = data_ful_sq.iloc[0].tolist()
            data_ful_sq = data_ful_sq.iloc[1:,:]
            data_ful_sq.columns = col
            data_ful_sq= data_ful_sq[[
            'cdr1_aa_heavy',
            'cdr2_aa_heavy',
            'cdr3_aa_heavy',
            'cdr1_aa_light',
            'cdr2_aa_light',
            'cdr3_aa_light', 'sequence_alignment_aa_light', 'sequence_alignment_aa_heavy']]
            data_ful_sq
            data_ful_sq.columns = ['cdr1_h', 'cdr2_h', 'cdr3_h', 'cdr1_l', 'cdr2_l', 'cdr3_l', 'seq_l', 'seq_h']
            data_ful_sq
            data_ful_sq
            def find_start_end(text, substring):
                start = text.find(substring)
                if start == -1:
                    return None, None
                end = start + len(substring)
                return (start, end)
            spis_cdr_l = []
            spis_cdr_h = []
            data_ful_sq = data_ful_sq.dropna()
            data_ful_sq
            spis_column_index_heavy = []
            spis_column_index_light = []
            for i in range(len(data_ful_sq)):
                # print(data_ful_sq['cdr1_h'].iloc[i])
                cdr1_h = find_start_end(data_ful_sq['seq_h'].iloc[i],data_ful_sq['cdr1_h'].iloc[i])
                spis_cdr_h.append(cdr1_h)

                cdr2_h = find_start_end(data_ful_sq['seq_h'].iloc[i],data_ful_sq['cdr2_h'].iloc[i])
                spis_cdr_h.append(cdr2_h)

                cdr3_h = find_start_end(data_ful_sq['seq_h'].iloc[i],data_ful_sq['cdr3_h'].iloc[i])
                spis_cdr_h.append(cdr3_h)



                cdr1_l = find_start_end(data_ful_sq['seq_l'].iloc[i],data_ful_sq['cdr1_l'].iloc[i])
                spis_cdr_l.append(cdr1_l)

                cdr2_l = find_start_end(data_ful_sq['seq_l'].iloc[i],data_ful_sq['cdr2_l'].iloc[i])
                spis_cdr_l.append(cdr2_l)

                cdr3_l = find_start_end(data_ful_sq['seq_l'].iloc[i],data_ful_sq['cdr3_l'].iloc[i])
                spis_cdr_l.append(cdr3_l)
                # print(spis_cdr_l)
                # print(spis_cdr_h)
                # print(data_ful_sq['seq_h'].iloc[i])
                # print(data_ful_sq['seq_l'].iloc[i])
                # print('<---------->', i)
                spis_column_index_heavy.append(spis_cdr_h)
                spis_column_index_light.append(spis_cdr_l)
                spis_cdr_l = []
                spis_cdr_h = []


            data_ful_sq['index_cdr_h'] = spis_column_index_heavy
            data_ful_sq['index_cdr_l'] = spis_column_index_light
            data_ful_sq = data_ful_sq[['seq_l', 'seq_h', 'index_cdr_h', 'index_cdr_l']]
            data_it = pd.concat([data_it, data_ful_sq], ignore_index=True)
            print(f'Файл {i} успешно загружен, разархивирован и обработан')
            print('------------------------->>>>   ', str(i)*20)
        else:
            print(f'Не удалось загрузить файл {i}')
    data_it[[*data_it.columns[:-2], data_it.columns[-1], data_it.columns[-2]]].to_csv(r'C:\Users\edmeh\Downloads\Datasets_bio01\data_human_antibody.csv')
    return data_it[[*data_it.columns[:-2], data_it.columns[-1], data_it.columns[-2]]]




ustanovka_data_human_antibody(spis)
