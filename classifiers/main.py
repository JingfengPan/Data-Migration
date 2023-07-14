import time
from compression import compress
from calculate_size import calculate_size
from calculate_throughput import calculate_throughput
from split_data import split_data


def main_compress(n_clus, _name, columns):
    gzip_start = time.time()
    compress('gzip', n_clus, _name, columns)
    gzip_end = time.time()
    t_gzip = gzip_end - gzip_start

    lz4_start = time.time()
    compress('lz4', n_clus, _name, columns)
    lz4_end = time.time()
    t_lz4 = lz4_end - lz4_start

    zstd_start = time.time()
    compress('zst', n_clus, _name, columns)
    zstd_end = time.time()
    t_zstd = zstd_end - zstd_start

    return t_gzip, t_lz4, t_zstd

def main_calculate_size(n_clus, _name, columns):
    gzip_size = calculate_size('gzip', n_clus, _name, columns)
    lz4_size = calculate_size('lz4', n_clus, _name, columns)
    zstd_size = calculate_size('zstandard', n_clus, _name, columns)
    return gzip_size, lz4_size, zstd_size


def main(n_clus, name, t_clus, t_first_full, n_data, path, columns, abbr):
    split_data(n_clus, path, './test_data/DS_001/DS_001_{}_predictions_{}.csv'.format(n_clus, abbr), name)
    t_gzip, t_lz4, t_zstd = main_compress(n_clus, name, columns)
    gzip_size, lz4_size, zstd_size = main_calculate_size(n_clus, name, columns)
    network_speeds = [2 * 1024, 4 * 1024, 6 * 1024, 8 * 1024]
    for network_speed in network_speeds:
        print('Gzip')
        calculate_throughput(t_clus, t_first_full, t_gzip, gzip_size, n_clus, n_data, network_speed, path)
        print('LZ4')
        calculate_throughput(t_clus, t_first_full, t_lz4, lz4_size, n_clus, n_data, network_speed, path)
        print('Zstandard')
        calculate_throughput(t_clus, t_first_full, t_zstd, zstd_size, n_clus, n_data, network_speed, path)


if __name__ == '__main__':
    abbrs = ['df', 'rf', 'aba', 'qda', 'mlp', 'gnb']
    t_clus = [0.4010, 2.7217, 9.8250, 2.8392, 1.5160, 1.5103]
    for i in range(len(t_clus)):
        main(20, '20_{}_001'.format(abbrs[i]), t_clus[i], t_clus[i] / 20, 10000, './test_data/DS_001/DS_001_{}_shuffle.csv'.format(20), 8, abbrs[i])