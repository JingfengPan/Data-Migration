import os


def read_data(path):
    with open(path, 'r') as f:
        lines = len(f.readlines())
        size = os.stat(path).st_size / (1024 * 1024)  # MB
    return lines, size

def calculate_compression_time(t_comp, n_clus, n_data, lines):
    t_avg = (t_comp + t_comp / n_clus) / 2
    return n_data / lines * t_avg


def calculate_online_clustering_t_tran(t_first_full, t_comp, t_tran, size):
    throughput = size / (t_first_full + t_comp + t_tran)
    return throughput


def calculate_online_clustering_t_clus(t_clus, t_comp, t_tran, n_data, size, lines):
    throughput = size / (t_clus + t_comp + (n_data / lines) * t_tran)
    return throughput


def calculate_random_split(t_split, t_comp, rs_compressed_size, n_clus, n_data, network_speed, path):
    lines, size = read_data(path)
    t_tran = rs_compressed_size / network_speed
    random_split_throughput = size / ((n_clus * n_data / lines) * t_split + t_comp + t_tran)
    print('Random Split Throughput (MB/s):', random_split_throughput)
    r_s = size * 1024 / rs_compressed_size
    print('Random Split Compression Ratio:', r_s)
    return random_split_throughput


def calculate_limit(oc_compressed_size, t_clus, network_speed, t_first_full, t_split, r_s, r_c):
    limit_network_speed = oc_compressed_size / t_clus
    limit_input_size = network_speed * (t_first_full - t_split) / (1 / r_s - 1 / r_c)
    return limit_network_speed, limit_input_size


def calculate_throughput(t_clus, t_first_full, t_oc_comp, oc_compressed_size, n_clus, n_data, network_speed, path):
    lines, size = read_data(path)
    t_oc_comp = calculate_compression_time(t_oc_comp, n_clus, n_data, lines)
    t_tran = oc_compressed_size / network_speed
    if t_clus > t_tran:
        online_clustering_throughput = calculate_online_clustering_t_clus(t_clus, t_oc_comp, t_tran, n_data, size, lines)
        print('Online Clustering Throughput (MB/s):', online_clustering_throughput)
    else:
        online_clustering_throughput = calculate_online_clustering_t_tran(t_first_full, t_oc_comp, t_tran, size)
        print('Online Clustering Throughput (MB/s):', online_clustering_throughput)
    r_c = size * 1024 / oc_compressed_size
    print('Online Clustering Compression Ratio:', r_c)
    return online_clustering_throughput
