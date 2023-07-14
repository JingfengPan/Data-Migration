import time
import pandas as pd
import random
from compression import compress
from calculate_size import calculate_size
from calculate_throughput import calculate_random_split


def read_data(path):
    with open(path) as f:
        raw_data = f.readlines()
    return raw_data


def random_split(n_clus, name, path):
    raw_data = read_data(path)
    split_start = time.time()
    result = []
    for i in range(n_clus):
        result.append([])
        for j in range(len(raw_data[0].replace(',', ' ').split('|'))):
            result[i].append([])
    for i in range(len(raw_data)):
        list = raw_data[i].replace(',', ' ').split('|')
        for j in range(len(list)):
            result[random.randint(0, n_clus - 1)][j].append(list[j])
    split_end = time.time()
    t_split = split_end - split_start
    print('Random split time (s):', t_split)
    for i in range(n_clus):
        for j in range(len(result[i])):
            output = pd.DataFrame(result[i][j])
            output.to_csv('./data/csv/{}_{}_{}.csv'.format(name, i, j), mode='a', index=False, header=False)
    return t_split


def main_compress(n_clus, rs_name, columns):
    gzip_rs_start = time.time()
    compress('gzip', n_clus, rs_name, columns)
    gzip_rs_end = time.time()
    t_rs_gzip = gzip_rs_end - gzip_rs_start

    lz4_rs_start = time.time()
    compress('lz4', n_clus, rs_name, columns)
    lz4_rs_end = time.time()
    t_rs_lz4 = lz4_rs_end - lz4_rs_start

    zstd_rs_start = time.time()
    compress('zst', n_clus, rs_name, columns)
    zstd_rs_end = time.time()
    t_rs_zstd = zstd_rs_end - zstd_rs_start

    return t_rs_gzip, t_rs_lz4, t_rs_zstd


def main_calculate_size(n_clus, rs_name, columns):
    rs_gzip_size = calculate_size('gzip', n_clus, rs_name, columns)
    rs_lz4_size = calculate_size('lz4', n_clus, rs_name, columns)
    rs_zstd_size = calculate_size('zstandard', n_clus, rs_name, columns)
    return rs_gzip_size, rs_lz4_size, rs_zstd_size


def main(n_clus, rs_name, n_data, path, columns):
        t_split = random_split(n_clus, rs_name, path)
        t_rs_gzip, t_rs_lz4, t_rs_zstd = main_compress(n_clus, rs_name, columns)
        rs_gzip_size, rs_lz4_size, rs_zstd_size = main_calculate_size(n_clus, rs_name, columns)
        network_speeds = [2 * 1024, 4 * 1024, 6 * 1024, 8 * 1024]
        for network_speed in network_speeds:
            print('Gzip')
            calculate_random_split(t_split, t_rs_gzip, rs_gzip_size, n_clus, n_data, network_speed, path)
            print('LZ4')
            calculate_random_split(t_split, t_rs_lz4, rs_lz4_size, n_clus, n_data, network_speed, path)
            print('Zstandard')
            calculate_random_split(t_split, t_rs_zstd, rs_zstd_size, n_clus, n_data, network_speed, path)


if __name__ == '__main__':
    main(20, 'random_001_20', 10000, './test_data/DS_001/DS_001.csv', 8)

