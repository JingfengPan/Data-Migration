import time
from incremental_k_prototypes import incremental_k_prototypes
from compression import compress
from calculate_size import calculate_size
from random_split import random_split
from calculate_throughput import calculate_throughput
from column_wise import column_wise
import matplotlib
import matplotlib.pyplot as plt


def calculate_compress_size(n_clus, oc_name, columns, algorithm):
    time_start = time.time()
    compress(algorithm, n_clus, oc_name, columns)
    time_end = time.time()
    compression_time = time_end - time_start
    # print('Online Clustering Compression Time (s):', time_oc)
    size = calculate_size(algorithm, n_clus, oc_name, columns)
    return compression_time, size


def main(n_clus, oc_name, path, columns):
    t_clus, t_first_full = incremental_k_prototypes(n_clus, [0, 1, 2, 4], [3], oc_name, path)  # coeff
    column_wise(n_clus, oc_name, columns)
    algorithms = ['gzip', 'lz4', 'zstd']
    for algorithm in algorithms:
        print('Compression Algorithm:', algorithm)
        compression_time, size = calculate_compress_size(n_clus, oc_name, columns, algorithm)
        print(t_clus + compression_time)
        print(155.16 * 1024 / size)


if __name__ == "__main__":
    main(20, '10000_20_001', './test_data/DS_001.csv', 8)
