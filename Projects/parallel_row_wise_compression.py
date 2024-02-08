import os
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


def parallel_row_wise_compression(n_clus, data_path, labels, format):
    with open(data_path, encoding='utf-8') as dp:
        data = dp.readlines()

    result = [[] for _ in range(n_clus)]
    for i, line in enumerate(data):
        result[int(labels[i])].append(line)

    cluster_bytes_list = ["".join(cluster).encode('utf-8') for cluster in result]
    original_size = os.stat(data_path).st_size

    start_time = time.time()
    results = Parallel(n_jobs=n_clus)(delayed(compress_cluster)(cluster_bytes, format) for cluster_bytes in cluster_bytes_list)
    end_time = time.time()
    compression_time = end_time - start_time

    compressed_size = sum(results)
    compression_ratio = original_size / compressed_size

    return original_size, compressed_size, compression_ratio, compression_time
