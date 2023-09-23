import os
import time
from compression import compress
from calculate_size import calculate_size
from parallel_compression import create_tasks
from split_data import split_data
from joblib import Parallel, delayed


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


def main():
    names = ['DecisionTree', 'RandomForest', 'AdaBoost', 'QDA', 'MLP', 'GaussianNB', 'KNN', 'LogisticRegression']
    file_names = ['DS_001', 'DS_002', 'customer']
    columns = [8, 8, 8]
    delimiters = ['|', '|', '|']
    n_clus = 20
    for i in range(len(file_names)):
        size = os.stat(f'./test_data/{file_names[i]}/{file_names[i]}.csv').st_size / (1024 * 1024)  # MB
        for name in names:
            split_data(n_clus, f'./test_data/{file_names[i]}/{file_names[i]}.csv',
                       f'./test_data/{file_names[i]}/{file_names[i]}_{n_clus}_predictions_{name}.csv', name, file_names[i], delimiters[i], columns[i])
            # t_gzip, t_lz4, t_zstd = main_compress(n_clus, name, file_names[i], columns[i])

            tasks = create_tasks('gzip', n_clus, name, file_names[i], columns[i])
            gzip_start_time = time.time()
            Parallel(n_jobs=-1)(tasks)  # Use all CPUs
            gzip_end_time = time.time()
            t_gzip = gzip_end_time - gzip_start_time

            tasks = create_tasks('lz4', n_clus, name, file_names[i], columns[i])
            lz4_start_time = time.time()
            Parallel(n_jobs=-1)(tasks)  # Use all CPUs
            lz4_end_time = time.time()
            t_lz4 = lz4_end_time - lz4_start_time

            tasks = create_tasks('zstd', n_clus, name, file_names[i], columns[i])
            zstd_start_time = time.time()
            Parallel(n_jobs=-1)(tasks)  # Use all CPUs
            zstd_end_time = time.time()
            t_zstd = zstd_end_time - zstd_start_time

            gzip_size, lz4_size, zstd_size = main_calculate_size(n_clus, name, file_names[i], columns[i])
            with open(f'./results/{file_names[i]}/{file_names[i]}_{n_clus}_parallel_classification_results.txt', 'a') as r:
                r.write(name + '\n')
                r.write(f'Gzip: compression time: {t_gzip:.5f} s, compression ratio: {(size / gzip_size):.5f}\n')
                r.write(f'LZ4: compression time: {t_lz4:.5f} s, compression ratio: {(size / lz4_size):.5f}\n')
                r.write(f'Zstandard: compression time: {t_zstd:.5f} s, compression ratio: {(size / zstd_size):.5f}\n\n')


if __name__ == '__main__':
    main()

