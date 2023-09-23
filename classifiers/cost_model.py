import os
from draw_figures import draw_figures


def read_data(path):
    with open(path, encoding='utf-8') as f:
        size = os.stat(path).st_size / (1024 * 1024)  # MB
    return size

def cost_model(
        block_size: int,  # the size of each block in the data stream of migration process, MB
        time_limit: int,  # time limit of the data migration, seconds
        num_cores: int,  # number of cpu cores in the source system of data migration
        net_speed: float,  # the network bandwidth from the source system to the target system in data migration, MB/s
        process_speed: float,  # including preprocess and compression, MB/s
        # non_process_speed: float,  # no process, MB/s
        process_compression_ratio: float,  # final compression ratio
        # non_process_compression_ratio: float,  # compression ratio without preprocessing, to compare
        price_cpu: float = 0.048 / 3600,  # the price of using one CPU core per second
        price_net: float = 0.05 / 1024  # the price of transforming data per megabyte
        # need_process: bool,     # true if cpu is needed for preprocessing and compressing, otherwise
        # process_time: float,    # total time used for preprocess and compression
        # file_size: float,       # file size before compression
):
    process_compression_ratio = max(process_compression_ratio, block_size / (time_limit * net_speed), 1)  # compression ratio limitation
    if process_speed < (block_size / (time_limit * num_cores)):
        print('The processing speed should be faster due to the time limit!')
        return

    # cost without any preprocessing, apply compression algorithm directly
    # cost_without_pre = block_size * price_cpu / non_process_speed + block_size * price_net / non_process_compression_ratio

    # cost with preprocessing
    cost_with_pre = block_size * price_cpu / process_speed + block_size * price_net / process_compression_ratio

    # print the costs
    # print(f'The cost with preprocessing is {cost_with_pre:.5f} dollars')  # \nThe cost without preprcessing is {cost_without_pre:.4f} dollars'

    return cost_with_pre  # , cost_without_pre


def main():
    # withp, withoutp = cost_model(block_size=155.16, time_limit=20, num_cores=16, net_speed=8.0, process_speed=7.65695,
    # non_process_speed=112.92659, process_compression_ratio=4.33050, non_process_compression_ratio=2.65687)
    file_names = ['econbiz', 'orders', 'partsupp']
    names = ['DecisionTree', 'RandomForest', 'AdaBoost', 'QDA', 'MLP', 'GaussianNB', 'KNN', 'LogisticRegression',
             'RandomSplit']
    t_class = [[0.12269, 1.23916, 5.29191, 1.10362, 0.70425, 0.84558, 14.15683, 0.24637, 2.97385],
               [1.96403, 5.07535, 7.65821, 2.46244, 1.56459, 1.95713, 31.95065, 0.37635, 8.74105],
               [0.10469, 0.79657, 4.19124, 1.12619, 0.72333, 0.90137, 13.95879, 0.17967, 2.60037]]

    t_comp = [[2.48864, 1.53393, 1.55639, 1.49644, 7.13268, 1.57800, 1.54223, 7.13478, 1.59083],  # econbiz gzip
              [0.14414, 0.12353, 0.16506, 0.14143, 0.41362, 0.18728, 0.12062, 0.45480, 0.17516],  # econbiz lz4
              [0.21906, 0.24659, 0.23712, 0.26417, 0.63064, 0.26220, 0.24859, 0.62376, 0.25884],  # econbiz zstd
              [3.77744, 3.71960, 14.03708, 14.76348, 14.76009, 14.87257, 3.55366, 14.76977, 3.80144],  # orders gzip
              [0.20815, 0.16907, 0.37525, 0.37789, 0.35920, 0.37025, 0.16874, 0.37425, 0.17487],  # orders lz4
              [0.26761, 0.26309, 0.44613, 0.40829, 0.42646, 0.42346, 0.26960, 0.44334, 0.26061],  # orders zstd
              [1.44872, 1.43309, 1.44153, 1.40229, 1.67355, 1.49300, 1.42264, 3.34910, 1.53045],  # partsupp gzip
              [0.11731, 0.10768, 0.12205, 0.19608, 0.14194, 0.10548, 0.12304, 0.22793, 0.12052],  # partsupp lz4
              [0.16745, 0.18261, 0.18154, 0.16505, 0.19387, 0.16162, 0.17517, 0.26062, 0.20807]]  # partsupp zstd

    compression_ratio = [[3.32104, 3.32104, 3.32107, 3.32093, 3.32650, 3.32109, 3.32109, 3.32640, 3.20617],
                         # econbiz gzip
                         [1.99612, 1.99611, 1.99612, 1.99600, 1.99809, 1.99609, 1.99607, 1.99809, 1.90738],
                         # econbiz lz4
                         [3.51445, 3.51444, 3.51445, 3.51349, 3.51268, 3.51441, 3.51428, 3.51268, 3.30855],
                         # econbiz zstd
                         [4.48436, 4.48785, 4.51027, 4.53305, 4.53446, 4.53412, 4.40655, 4.53413, 4.38250],
                         # orders gzip
                         [2.23699, 2.23670, 2.24179, 2.24717, 2.24723, 2.24714, 2.21587, 2.24723, 2.20844],
                         # orders lz4
                         [4.13522, 4.12803, 4.17738, 4.27872, 4.30014, 4.30072, 3.91483, 4.30014, 3.88646],
                         # orders zstd
                         [4.78286, 4.78286, 4.78294, 4.78121, 4.76592, 4.78281, 4.78170, 4.60875, 4.50047],
                         # partsupp gzip
                         [2.38784, 2.38784, 2.38789, 2.38736, 2.38497, 2.38785, 2.38760, 2.34325, 2.31208],
                         # partsupp lz4
                         [4.36317, 4.36295, 4.36354, 4.35957, 4.31201, 4.36316, 4.36210, 4.02510, 3.89580]]
                         # partsupp zstd
    n_clus = 20
    network_speeds = [4, 8, 12, 16, 20]
    compression_algorithms = ['Gzip', 'LZ4', 'Zstandard']
    for i in range(len(file_names)):
        block_size = read_data(f'./test_data/{file_names[i]}/{file_names[i]}.csv')
        cost_list = []
        with open(f'./results/{file_names[i]}_{n_clus}_cost_results.txt', 'a') as r:
            for j in range(len(compression_algorithms)):
                cost_list.append([])
                r.write(compression_algorithms[j] + '\n')
                for k in range(len(names)):
                    cost_list[j].append([])
                    r.write(names[k] + '\n')
                    for network_speed in network_speeds:
                        total_time = t_class[i][k] + t_comp[i*3+j][k]
                        class_throughput = block_size / total_time
                        cost = cost_model(block_size=block_size, time_limit=100, num_cores=16, net_speed=network_speed,
                                           process_speed=class_throughput, process_compression_ratio=compression_ratio[i*3+j][k])
                        cost_list[j][k].append(cost)
                        r.write(f'Network speed: {network_speed} MB/s\n')
                        r.write(f'Cost: ${cost:.5f}\n\n')
        draw_figures(cost_list, file_names[i])


if __name__ == '__main__':
    main()
