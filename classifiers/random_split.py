import os
import re
import time
import pandas as pd
import random
from joblib import delayed, Parallel
from compression import compress
from calculate_size import calculate_size


def remove_commas_inside_quotes(match):
    return match.group().replace(',', '')


def random_split(n_clus, data_path, name, file_name, delimiter, column):
    t_random_split = 0
    with open(data_path, encoding='utf-8') as dp:
        if file_name == 'econbiz':
            raw_data = dp.readlines()[1:]
        else:
            raw_data = dp.readlines()
    result = []
    for i in range(n_clus):
        result.append([])
        for j in range(column):
            result[i].append([])
    for i in range(len(raw_data)):
        regex_list = re.sub(r'"[^"]*"', remove_commas_inside_quotes, raw_data[i]).split(delimiter)
        for j in range(len(regex_list)):
            split_start = time.time()
            result[random.randint(0, n_clus - 1)][j].append(regex_list[j])
            split_end = time.time()
            t_random_split += (split_end - split_start)
    for i in range(n_clus):
        for j in range(len(result[i])):
            output = pd.DataFrame(result[i][j])
            # Check if the file exists before trying to remove it
            if os.path.exists(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv'):
                os.remove(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv')
            output.to_csv(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv', mode='a', index=False, header=False)
    return t_random_split


def main_compress(n_clus, name, file_name, column):
    gzip_start = time.time()
    compress('gzip', n_clus, name, file_name, column)
    gzip_end = time.time()
    t_gzip = gzip_end - gzip_start

    lz4_start = time.time()
    compress('lz4', n_clus, name, file_name, column)
    lz4_end = time.time()
    t_lz4 = lz4_end - lz4_start

    zstd_start = time.time()
    compress('zst', n_clus, name, file_name, column)
    zstd_end = time.time()
    t_zstd = zstd_end - zstd_start
    return t_gzip, t_lz4, t_zstd

def main_calculate_size(n_clus, name, file_name, column):
    gzip_size = calculate_size('gzip', n_clus, name, file_name, column)
    lz4_size = calculate_size('lz4', n_clus, name, file_name, column)
    zstd_size = calculate_size('zstd', n_clus, name, file_name, column)
    return gzip_size, lz4_size, zstd_size


def parallel_random_split(batch, n_clus, column, delimiter, file_name, name):
    result = []
    for i in range(n_clus):
        result.append([])
        for j in range(column):
            result[i].append([])
    for i in range(len(batch)):
        regex_list = re.sub(r'"[^"]*"', remove_commas_inside_quotes, batch[i]).split(delimiter)
        for j in range(len(regex_list)):
            result[i % n_clus][j].append(regex_list[j])
    for i in range(n_clus):
        for j in range(len(result[i])):
            output = pd.DataFrame(result[i][j])
            output.to_csv(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv', mode='a', index=False, header=False)


def create_tasks(raw_data, n_clus, column, delimiter, file_name, name, n_jobs=10):
    tasks = []
    batch_size = len(raw_data) // n_jobs
    for i in range(n_jobs):
        start = i * batch_size
        end = (i + 1) * batch_size if i != n_jobs - 1 else len(raw_data)
        batch = raw_data[start:end]
        tasks.append(delayed(parallel_random_split)(batch, n_clus, column, delimiter, file_name, name))
    return tasks


def main():
    name = 'RandomSplit'
    file_names = ['DS_001', 'DS_002', 'customer']
    columns = [8, 8, 8]
    delimiters = ['|', '|', '|']
    n_clus = 20
    for i in range(len(file_names)):
        size = os.stat(f'./test_data/{file_names[i]}/{file_names[i]}.csv').st_size / (1024 * 1024)  # MB
        with open(f'./test_data/{file_names[i]}/{file_names[i]}.csv', encoding='utf-8') as dp:
            raw_data = dp.readlines()
        start_time = time.time()
        tasks = create_tasks(raw_data, n_clus, columns[i], delimiters[i], file_names[i], name)
        Parallel(n_jobs=-1)(tasks)
        end_time = time.time()
        t_random_split = end_time - start_time
        # t_random_split = random_split(n_clus, name, file_names[i], delimiters[i], columns[i])
        t_gzip, t_lz4, t_zstd = main_compress(n_clus, name, file_names[i], columns[i])
        gzip_size, lz4_size, zstd_size = main_calculate_size(n_clus, name, file_names[i], columns[i])
        with open(f'./results/{file_names[i]}/{file_names[i]}_{n_clus}_parallel_classification_results.txt', 'a') as r:
            r.write(name + '\n')
            r.write(f'Random split time: {t_random_split:.5f} s\n')
            r.write(f'Gzip: compression time: {t_gzip:.5f} s, compression ratio: {(size / gzip_size):.5f}\n')
            r.write(f'LZ4: compression time: {t_lz4:.5f} s, compression ratio: {(size / lz4_size):.5f}\n')
            r.write(f'Zstandard: compression time: {t_zstd:.5f} s, compression ratio: {(size / zstd_size):.5f}\n\n')


if __name__ == '__main__':
    main()

