import os


def read_data_size(path):
    with open(path, encoding='utf-8') as f:
        size = os.stat(path).st_size / (1024 * 1024)  # MB
    return size


def calculate_compression_ratio(dir, path, n_clus, class_name, name, column):
    if dir == 'gzip':
        extension = 'gz'
    elif dir == 'lz4':
        extension = 'lz4'
    else:
        extension = 'zst'
    sum = 0
    for i in range(n_clus):
        for j in range(column):
            size = os.path.getsize(f'./data/{dir}/{name}/{name}_{n_clus}_{class_name}_{i}_{j}.{extension}')
            sum = sum + size / (1024 * 1024)  # MB
    # print(dir, sum, 'KB')
    original_size = os.stat(path).st_size / (1024 * 1024)
    comp_ratio = original_size / sum
    return comp_ratio

