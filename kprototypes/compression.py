import gzip
import shutil
import time
import lz4.frame
import zstandard as zstd


def compress(format, n_clus, name):
    if format == 'gzip':
        # gzip
        for i in range(n_clus):
            with open('./data/csv/{}_{}.csv'.format(name, i), 'rb') as f_in:
                with gzip.open('./data/gzip/{}_{}.gz'.format(name, i), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    elif format == 'lz4':
        # lz4
        for i in range(n_clus):
            with open('./data/csv/{}_{}.csv'.format(name, i), 'rb') as f_in:
                with lz4.frame.open('./data/lz4/{}_{}.lz4'.format(name, i), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    else:
        # zstandard
        for i in range(n_clus):
            with open('./data/csv/{}_{}.csv'.format(name, i), 'rb') as f_in:
                with zstd.open('./data/zstandard/{}_{}.zst'.format(name, i), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
