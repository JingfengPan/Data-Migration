import os
import re
import time
from joblib import Parallel, delayed
import gzip
import lz4.frame
import zstandard as zstd


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


def parallel_column_wise_compression(n_clus, data_path, labels, format, delimiter):
    data = column_wise_split(n_clus, data_path, labels, delimiter)
    tasks = [(cluster[column_index], format) for cluster in data for column_index in range(len(cluster))]
    start_time = time.time()
    results = Parallel(n_jobs=n_clus)(delayed(compress_column)(*task) for task in tasks)
    end_time = time.time()
    original_size = os.stat(data_path).st_size

    compressed_size = sum(results)
    compression_time = end_time - start_time
    compression_ratio = original_size / compressed_size

    return original_size, compressed_size, compression_ratio, compression_time

