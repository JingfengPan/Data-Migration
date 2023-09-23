import os


def calculate_size(dir, n_clus, name, file_name, column):
    if dir == 'gzip':
        extension = 'gz'
    elif dir == 'lz4':
        extension = 'lz4'
    else:
        extension = 'zst'
    sum = 0
    for i in range(n_clus):
        for j in range(column):
            size = os.path.getsize(f'./data/{dir}/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.{extension}')
            sum = sum + size / (1024 * 1024)  # MB
    # print(dir, sum, 'KB')
    return sum

