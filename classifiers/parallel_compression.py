from joblib import Parallel, delayed
import os
import gzip
import lz4.frame
import shutil
import zstandard as zstd


# Step 1: Break down into smaller functions
def compress_gzip(i, j, n_clus, name, file_name):
    with open(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv', 'rb') as f_in:
        if os.path.exists(f'./data/gzip/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.gz'):
            os.remove(f'./data/gzip/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.gz')
        with gzip.open(f'./data/gzip/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def compress_lz4(i, j, n_clus, name, file_name):
    with open(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv', 'rb') as f_in:
        if os.path.exists(f'./data/lz4/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.lz4'):
            os.remove(f'./data/lz4/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.lz4')
        with lz4.frame.open(f'./data/lz4/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.lz4', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def compress_zstd(i, j, n_clus, name, file_name):
    with open(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv', 'rb') as f_in:
        if os.path.exists(f'./data/zstd/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.zst'):
            os.remove(f'./data/zstd/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.zst')
        with zstd.open(f'./data/zstd/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.zst', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

# Step 2: Create an iterable of tasks
def create_tasks(format, n_clus, name, file_name, column):
    tasks = []
    for i in range(n_clus):
        for j in range(column):
            if format == 'gzip':
                tasks.append(delayed(compress_gzip)(i, j, n_clus, name, file_name))
            elif format == 'lz4':
                tasks.append(delayed(compress_lz4)(i, j, n_clus, name, file_name))
            else:
                tasks.append(delayed(compress_zstd)(i, j, n_clus, name, file_name))
    return tasks


