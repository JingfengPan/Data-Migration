import gzip
import shutil
import lz4.frame
import zstandard as zstd


def compress(format, n_clus, name, columns):
    if format == 'gzip':
        # gzip
        for i in range(n_clus):
            for j in range(columns):
                with open('./data/csv/{}_{}_{}.csv'.format(name, i, j), 'rb') as f_in:
                    with gzip.open('./data/gzip/{}_{}_{}.gz'.format(name, i, j), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
    elif format == 'lz4':
        # lz4
        for i in range(n_clus):
            for j in range(columns):
                with open('./data/csv/{}_{}_{}.csv'.format(name, i, j), 'rb') as f_in:
                    with lz4.frame.open('./data/lz4/{}_{}_{}.lz4'.format(name, i, j), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
    else:
        # zstandard
        for i in range(n_clus):
            for j in range(columns):
                with open('./data/csv/{}_{}_{}.csv'.format(name, i, j), 'rb') as f_in:
                    with zstd.open('./data/zstandard/{}_{}_{}.zst'.format(name, i, j), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
