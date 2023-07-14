import os


def calculate_size(dir, n_clus, name, columns):
    if dir == 'gzip':
        extension = 'gz'
    elif dir == 'lz4':
        extension = 'lz4'
    else:
        extension = 'zst'
    sum = 0
    for i in range(n_clus):
        for j in range(columns):
            size = os.path.getsize('./data/{}/{}_{}_{}.{}'.format(dir, name, i, j, extension))
            sum = sum + size / 1024
    # print(dir, sum, 'KB')
    return sum

