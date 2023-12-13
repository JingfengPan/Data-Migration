import time
from joblib import Parallel, delayed
import gzip
import lz4.frame
import zstandard as zstd
from incremental_k_prototypes import incremental_k_prototypes
from preprocess_data import read_data


def compress_cluster(cluster_bytes, format):
    if format == 'gzip':
        compressed = gzip.compress(cluster_bytes)
    elif format == 'lz4':
        compressed = lz4.frame.compress(cluster_bytes)
    elif format == 'zstd':
        compressed = zstd.compress(cluster_bytes)
    else:
        raise ValueError("Unsupported compression format")

    return len(compressed)


def parallel_row_based_compression(n_clus, data_path, labels, format, n_processes):
    with open(data_path, encoding='utf-8') as dp:
        data = dp.readlines()

    result = [[] for _ in range(n_clus)]
    for i, line in enumerate(data):
        result[int(labels[i])].append(line)

    cluster_bytes_list = ["".join(cluster).encode('utf-8') for cluster in result]
    original_size = sum([len(cluster_bytes) for cluster_bytes in cluster_bytes_list])

    start_time = time.time()
    results = Parallel(n_jobs=n_processes)(delayed(compress_cluster)(cluster_bytes, format) for cluster_bytes in cluster_bytes_list)
    end_time = time.time()
    compression_time = end_time - start_time

    compressed_size = sum(results)
    compression_ratio = original_size / compressed_size

    return original_size, compressed_size, compression_ratio, compression_time


name = 'DS_001'
num_indexs = [0, 5]
cate_indexs = [3, 6]
n_clus = 5
test_data_path = f'./test_data/{name}.csv'
dataset = read_data(name, n_clus, num_indexs, cate_indexs, '|')
t_clus, clus_labels = incremental_k_prototypes(dataset, n_clus, num_indexs, cate_indexs, name)
n_processors_list = [1, 2, 5, 10]
print(f"Total Original Size: 155.15867 MB")
for n_processors in n_processors_list:
    print(f"Number of processors: {n_processors}")
    for format in ('gzip', 'lz4', 'zstd'):
        original_size, compressed_size, compression_ratio, compression_time = parallel_row_based_compression(n_clus, test_data_path, clus_labels, format, n_processors)
        print(f"Compression Format: {format}")
        print(f"Total Compressed Size: {compressed_size/(1024*1024):.5f} MB")
        print(f"Compression Ratio: {compression_ratio:.5f}")
        print(f"Compression time: {compression_time:.5f} s")
