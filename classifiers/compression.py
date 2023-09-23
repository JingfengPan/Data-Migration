import gzip
import os
import shutil
import lz4.frame
import zstandard as zstd


def compress(format, n_clus, name, file_name, column):
    if format == 'gzip':
        # gzip
        for i in range(n_clus):
            for j in range(column):
                with open(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv', 'rb') as f_in:
                    if os.path.exists(f'./data/gzip/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.gz'):
                        os.remove(f'./data/gzip/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.gz')
                    with gzip.open(f'./data/gzip/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.gz', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
    elif format == 'lz4':
        # lz4
        for i in range(n_clus):
            for j in range(column):
                with open(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv', 'rb') as f_in:
                    if os.path.exists(f'./data/gzip/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.lz4'):
                        os.remove(f'./data/gzip/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.lz4')
                    with lz4.frame.open(f'./data/lz4/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.lz4', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
    else:
        # zstandard
        for i in range(n_clus):
            for j in range(column):
                with open(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv', 'rb') as f_in:
                    if os.path.exists(f'./data/gzip/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.zst'):
                        os.remove(f'./data/gzip/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.zst')
                    with zstd.open(f'./data/zstd/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.zst'.format(name, i, j), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
