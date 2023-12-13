import os
import re
import time
from joblib import Parallel, delayed
import gzip
import lz4.frame
import zstandard as zstd
from incremental_k_prototypes import incremental_k_prototypes
from preprocess_data import read_data


def column_wise_split(n_clus, data_path, labels, delimiter):
    with open(data_path, encoding='utf-8') as dp:
        data = dp.readlines()

    column_count = len(re.sub(r'"[^"]*"', lambda match: match.group().replace(',', ''), data[0]).split(delimiter))
    result = [[[] for _ in range(column_count)] for _ in range(n_clus)]

    for i, line in enumerate(data):
        regex_list = re.sub(r'"[^"]*"', lambda match: match.group().replace(',', ''), line).split(delimiter)
        for j, value in enumerate(regex_list):
            result[int(labels[i])][j].append(value)

    return result


def compress_column(column_data, format):
    column_bytes = "".join(column_data).encode('utf-8')
    if format == 'gzip':
        compressed = gzip.compress(column_bytes)
    elif format == 'lz4':
        compressed = lz4.frame.compress(column_bytes)
    elif format == 'zstd':
        compressed = zstd.compress(column_bytes)
    else:
        raise ValueError("Unsupported compression format")
    compressed_size = len(compressed)
    return compressed_size


def parallel_column_wise_compression(data, data_path, format, n_processes):
    tasks = [(cluster[column_index], format) for cluster in data for column_index in range(len(cluster))]
    start_time = time.time()
    results = Parallel(n_jobs=n_processes)(delayed(compress_column)(*task) for task in tasks)
    end_time = time.time()
    original_size = os.stat(data_path).st_size

    compressed_size = sum(results)
    compression_time = end_time - start_time
    compression_ratio = original_size / compressed_size

    return original_size, compressed_size, compression_ratio, compression_time


name = 'DS_001'
num_indexs = [0, 5]
cate_indexs = [3, 6]
n_clus = 5
test_data_path = f'./test_data/{name}.csv'
dataset = read_data(name, n_clus, num_indexs, cate_indexs, '|')
t_clus, clus_labels = incremental_k_prototypes(dataset, n_clus, num_indexs, cate_indexs, name)
data = column_wise_split(n_clus, test_data_path, clus_labels, '|')
n_processors_list = [1, 2, 5, 10]
print(f"Total Original Size: 155.15867 MB")
for n_processors in n_processors_list:
    print(f"Number of processors: {n_processors}")
    for format in ('gzip', 'lz4', 'zstd'):
        original_size, compressed_size, compression_ratio, compression_time = parallel_column_wise_compression(data, test_data_path, format, n_processors)
        print(f"Compression Format: {format}")
        print(f"Total Compressed Size: {compressed_size/(1024*1024):.5f} MB")
        print(f"Compression Ratio: {compression_ratio:.5f}")
        print(f"Compression time: {compression_time:.5f} s")
