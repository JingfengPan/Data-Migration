import time
from incremental_k_prototypes import incremental_k_prototypes
from compression import compress
from calculate_size import calculate_size
from random_split import random_split
from calculate_throughput import calculate_throughput
import matplotlib
import matplotlib.pyplot as plt


def output_results(n_clus, oc_name, n_data, network_speed, path):
    t_clus, t_first_full = incremental_k_prototypes(n_clus, [0, 1, 2, 4], [3], oc_name, path)  # coeff

    gzip_oc_start = time.time()
    compress('gzip', n_clus, oc_name)
    gzip_oc_end = time.time()
    t_oc_gzip = gzip_oc_end - gzip_oc_start
    # print('Online Clustering Gzip Compression Time (s):', t_oc_gzip)

    lz4_oc_start = time.time()
    compress('lz4', n_clus, oc_name)
    lz4_oc_end = time.time()
    t_oc_lz4 = lz4_oc_end - lz4_oc_start
    # print('Online Clustering LZ4 Compression Time (s):', t_oc_lz4)

    zstd_oc_start = time.time()
    compress('zst', n_clus, oc_name)
    zstd_oc_end = time.time()
    t_oc_zstd = zstd_oc_end - zstd_oc_start
    # print('Online Clustering Zstandard Compression Time (s):', t_oc_zstd)

    oc_gzip_size = calculate_size('gzip', n_clus, oc_name)
    oc_lz4_size = calculate_size('lz4', n_clus, oc_name)
    oc_zstd_size = calculate_size('zstandard', n_clus, oc_name)

    oc_throughput_gzip = calculate_throughput(t_clus, t_first_full, t_oc_gzip, oc_gzip_size, n_clus, n_data, network_speed, path)
    print('----------------------------------------')
    oc_throughput_lz4 = calculate_throughput(t_clus, t_first_full, t_oc_lz4, oc_lz4_size, n_clus, n_data, network_speed, path)
    print('----------------------------------------')
    oc_throughput_zstd = calculate_throughput(t_clus, t_first_full, t_oc_zstd, oc_zstd_size, n_clus, n_data, network_speed, path)
    print('----------------------------------------')

    return oc_throughput_gzip, oc_throughput_lz4, oc_throughput_zstd


def main():
    network_speed = [2 * 1024, 4 * 1024, 6 * 1024, 8 * 1024]
    for ns in network_speed:
        print(output_results(20, '10000_20_001_{}'.format(ns), 10000, ns, './test_data/DS_001.csv'))


if __name__ == "__main__":
    main()
