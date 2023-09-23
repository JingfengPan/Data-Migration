import os
from draw_figures import draw_figures

def read_data(path):
    with open(path, encoding='utf-8') as f:
        lines = len(f.readlines())
        size = os.stat(path).st_size / (1024 * 1024)  # MB
    return lines, size


def calculate_compression_time(t_comp, n_clus, n_data, lines):
    t_avg = (t_comp + t_comp / n_clus) / 2
    return n_data / lines * t_avg


def calculate_t_tran(t_first_full, t_comp, t_tran, size):
    throughput = size / (t_first_full + t_comp + t_tran)
    return throughput


def calculate_t_class(t_class, t_comp, t_tran, n_data, size, lines):
    throughput = size / (t_class + t_comp + (n_data / lines) * t_tran)
    return throughput


def calculate_total_throughput(t_class, t_comp, compression_ratio, network_speed, path):
    lines, size = read_data(path)
    t_tran = (size / compression_ratio) / network_speed
    throughput = size / (t_class + t_comp + t_tran)
    # print('Random Split Throughput (MB/s):', random_split_throughput)
    return throughput


def calculate_theoretical_throughput(t_class, t_comp, compression_ratio, n_clus, n_data, network_speed, path):
    lines, size = read_data(path)
    t_comp = calculate_compression_time(t_comp, n_clus, n_data, lines)
    t_tran = (size / compression_ratio) / network_speed
    if t_class > t_tran:
        throughput = calculate_t_class(t_class, t_comp, t_tran, n_data, size, lines)
        # print('Throughput (MB/s):', throughput)
    else:
        t_first_full = ((n_clus * n_data) / lines) * t_class
        throughput = calculate_t_tran(t_first_full, t_comp, t_tran, size)
        # print('Throughput (MB/s):', throughput)
    return throughput


def calculate_parts_throughput(t_class, t_comp, compression_ratio, network_speed, path):
    _, data_size = read_data(path)
    total_time = t_class + t_comp
    class_throughput = data_size / total_time
    network_throughput = network_speed * compression_ratio
    throughput = min(class_throughput, network_throughput)
    return throughput


def main():
    file_names = ['econbiz', 'orders']
    names = ['DecisionTree', 'RandomForest', 'AdaBoost', 'QDA', 'MLP', 'GaussianNB', 'KNN', 'LogisticRegression', 'RandomSplit']
    t_class = [[2.40383, 2.34413, 2.58102, 0.60245, 0.47043, 0.52235, 3.21202, 0.27944, 2.18530],
              [5.96454, 58.91170, 4.06493, 1.61460, 0.85257, 1.19410, 5.44181, 0.44169, 2.24518],
              [0.24692, 1.35546, 2.02869, 0.69506, 2.02560, 1.52124, 3.31763, 0.84883, 1.09149]]

    t_comp = [[2.48864, 1.53393, 1.55639, 1.49644, 7.13268, 1.57800, 1.54223, 7.13478, 1.59083],  # econbiz gzip
              [0.14414, 0.12353, 0.16506, 0.14143, 0.41362, 0.18728, 0.12062, 0.45480, 0.17516],  # econbiz lz4
              [0.21906, 0.24659, 0.23712, 0.26417, 0.63064, 0.26220, 0.24859, 0.62376, 0.25884],  # econbiz zstd
              [3.77744, 3.71960, 14.03708, 14.76348, 14.76009, 14.87257, 3.55366, 14.76977, 3.80144],  # orders gzip
              [0.20815, 0.16907, 0.37525, 0.37789, 0.35920, 0.37025, 0.16874, 0.37425, 0.17487],  # orders lz4
              [0.26761, 0.26309, 0.44613, 0.40829, 0.42646, 0.42346, 0.26960, 0.44334, 0.26061],  # orders zstd
              [1.44872, 1.43309, 1.44153, 1.40229, 1.67355, 1.49300, 1.42264, 3.34910, 1.53045],  # partsupp gzip
              [0.11731, 0.10768, 0.12205, 0.19608, 0.14194, 0.10548, 0.12304, 0.22793, 0.12052],  # partsupp lz4
              [0.16745, 0.18261, 0.18154, 0.16505, 0.19387, 0.16162, 0.17517, 0.26062, 0.20807]]  # partsupp zstd

    compression_ratio = [[3.32104, 3.32104, 3.32107, 3.32093, 3.32650, 3.32109, 3.32109, 3.32640, 3.20617],  # econbiz gzip
                         [1.99612, 1.99611, 1.99612, 1.99600, 1.99809, 1.99609, 1.99607, 1.99809, 1.90738],  # econbiz lz4
                         [3.51445, 3.51444, 3.51445, 3.51349, 3.51268, 3.51441, 3.51428, 3.51268, 3.30855],  # econbiz zstd
                         [4.48436, 4.48785, 4.51027, 4.53305, 4.53446, 4.53412, 4.40655, 4.53413, 4.38250],  # orders gzip
                         [2.23699, 2.23670, 2.24179, 2.24717, 2.24723, 2.24714, 2.21587, 2.24723, 2.20844],  # orders lz4
                         [4.13522, 4.12803, 4.17738, 4.27872, 4.30014, 4.30072, 3.91483, 4.30014, 3.88646],  # orders zstd
                         [4.78286, 4.78286, 4.78294, 4.78121, 4.76592, 4.78281, 4.78170, 4.60875, 4.50047],  # partsupp gzip
                         [2.38784, 2.38784, 2.38789, 2.38736, 2.38497, 2.38785, 2.38760, 2.34325, 2.31208],  # partsupp lz4
                         [4.36317, 4.36295, 4.36354, 4.35957, 4.31201, 4.36316, 4.36210, 4.02510, 3.89580]]  # partsupp zstd
    n_clus = 20
    network_speeds = [4, 8, 12, 16, 20]
    compression_algorithms = ['Gzip', 'LZ4', 'Zstandard']
    for i in range(len(file_names)):
        theoretical_throughput_list = []
        lines, _ = read_data(f'./test_data/{file_names[i]}/{file_names[i]}.csv')
        n_data = lines // 10
        # total_throughput_list = []
        # parts_throughput_list = []
        with open(f'./results/{file_names[i]}/{file_names[i]}_{n_clus}_parallel_throughput_results.txt', 'a') as r:
            for j in range(len(compression_algorithms)):
                theoretical_throughput_list.append([])
                # total_throughput_list.append([])
                # parts_throughput_list.append([])
                r.write(compression_algorithms[j] + '\n')
                for k in range(len(names)):
                    theoretical_throughput_list[j].append([])
                    # total_throughput_list[j].append([])
                    # parts_throughput_list[j].append([])
                    r.write(names[k] + '\n')
                    for network_speed in network_speeds:
                        theoretical_throughput = calculate_theoretical_throughput(t_class[i][k], t_comp[i*3+j][k], compression_ratio[i*3+j][k], n_clus, n_data, network_speed, f'./test_data/{file_names[i]}/{file_names[i]}.csv')
                        theoretical_throughput_list[j][k].append(theoretical_throughput)
                        # total_throughput = calculate_total_throughput(t_class[i][k], t_comp[i*3+j][k], compression_ratio[i*3+j][k], network_speed, f'./test_data/{file_names[i]}/{file_names[i]}.csv')
                        # total_throughput_list[j][k].append(total_throughput)
                        # parts_throughput = calculate_parts_throughput(t_class[i][k], t_comp[i*3+j][k], compression_ratio[i*3+j][k], network_speed, f'./test_data/{file_names[i]}/{file_names[i]}.csv')
                        # parts_throughput_list[j][k].append(parts_throughput)
                        r.write(f'Network speed: {network_speed} MB/s\n')
                        r.write(f'Theoretical throughput: {theoretical_throughput:.5f}\n')
                        # r.write(f'Total throughput: {total_throughput:.5f}\n')
                        # r.write(f'Parts throughput: {parts_throughput:.5f}\n\n')
        draw_figures(theoretical_throughput_list, file_names[i], 'Theoretical')
        # draw_figures(total_throughput_list, file_names[i], 'Total')
        # draw_figures(parts_throughput_list, file_names[i], 'Parts')


if __name__ == '__main__':
    main()

