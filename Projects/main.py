import os
import pandas as pd
from incremental_k_prototypes import incremental_k_prototypes
from preprocess_data import read_data
from random_split import random_split
from classifiers import split_train_test_set, train_test_classification_models
from parallel_column_wise_compression import parallel_column_wise_compression
from parallel_row_wise_compression import parallel_row_wise_compression
from calculate_throughput import calculate_throughput
from cost_model import cost_model
from draw_figures import draw_figures


def create_new_table(name, indices):
    new_data = []
    with open(f'test_data/{name}.csv', encoding='utf-8') as dp:
        raw_data = dp.readlines()
        for index in indices:
            new_data.append(raw_data[index].strip('\n'))
        if not os.path.exists(f'./data/{name}'):
            os.makedirs(f'./data/{name}')
        output = pd.DataFrame(new_data)
        new_data_path = f'./data/{name}/{name}.csv'
        output.to_csv(new_data_path, mode='w', index=False, header=False)
    return new_data_path


def output_column_wise_results(labels, format, n_clus, name, t_clus, network_speeds, path, delimiter):
    throughputs = []
    original_size, compressed_size, comp_ratio, t_comp = parallel_column_wise_compression(n_clus, path, labels, format, delimiter)
    with open(f'./results/{name}/{name}_results_figures.txt', 'a') as r:
        r.write(f'{format}\n')
        r.write(f'Compression time: {t_comp:.5f} s, Compression ratio: {comp_ratio:.5f}\n')
        for network_speed in network_speeds:
            r.write(f'Network speed: {network_speed} MB/s\n')
            throughput, bottleneck = calculate_throughput(t_clus, t_comp, comp_ratio, network_speed, path)
            throughputs.append(throughput)
            r.write(f'Bottleneck: {bottleneck}, with throughput: {throughput:.5f} MB/s\n')
        cost = cost_model(t_clus, t_comp, comp_ratio, path, n_clus)
        r.write(f'Cost: $ {cost:.5f} /TB\n\n')
    return throughputs, cost


def output_row_wise_results(labels, format, n_clus, name, t_clus, network_speeds, path):
    throughputs = []
    original_size, compressed_size, comp_ratio, t_comp = parallel_row_wise_compression(n_clus, path, labels, format)
    with open(f'./results/{name}/{name}_results_figures.txt', 'a') as r:
        r.write(f'{format}\n')
        r.write(f'Compression time: {t_comp:.5f} s, Compression ratio: {comp_ratio:.5f}\n')
        for network_speed in network_speeds:
            r.write(f'Network speed: {network_speed} MB/s\n')
            throughput, bottleneck = calculate_throughput(t_clus, t_comp, comp_ratio, network_speed, path)
            throughputs.append(throughput)
            r.write(f'Bottleneck: {bottleneck}, with throughput: {throughput:.5f} MB/s\n')
        cost = cost_model(t_clus, t_comp, comp_ratio, path, n_clus)
        r.write(f'Cost: $ {cost:.5f} /TB\n\n')
    return throughputs, cost


def main():
    names = ['DS_001', 'DS_002', 'DS_003',  'orders', 'partsupp']
    num_indexs = [[0, 5], [0, 5], [0, 5], [0, 1, 3, 6], [0, 1, 2, 3]]
    cate_indexs = [[3, 6], [3, 6], [3, 6], [2, 5], []]
    delimiter = '|'
    n_clusters = [20]
    buffer_sizes = [10000]
    train_sizes = [0.8]
    test_sizes = [0.1]
    class_names = ['DecisionTree', 'QDA', 'MLP', 'GaussianNB', 'LogisticRegression']  # 'RandomForest', 'AdaBoost', 'KNN'
    network_speeds = [5, 10, 15, 20]
    for i in range(len(names)):
        column_throughputs_list = []
        column_cost_list = []
        row_throughputs_list = []
        row_cost_list = []
        print('dataset:', names[i])
        test_data_path = f'./test_data/{names[i]}.csv'
        result_path = f'./results/{names[i]}/{names[i]}_results_figures.txt'
        dataset = read_data(names[i], num_indexs[i], cate_indexs[i], delimiter)
        if not os.path.exists(f'./results/{names[i]}'):
            os.makedirs(f'./results/{names[i]}')
        if os.path.exists(result_path):
            os.remove(result_path)
        with open(result_path, 'a') as r:
            for n_clus in n_clusters:
                print('n_clus:', n_clus)
                r.write(f'n_clus: {n_clus}\n')
                t_random_split, random_labels = random_split(dataset, n_clus, names[i])
                r.write('\ncolumn-wise\n')
                r.flush()
                gzip_throughputs, gzip_cost = output_column_wise_results(random_labels, 'gzip', n_clus, names[i], t_random_split, network_speeds, test_data_path, delimiter)
                lz4_throughputs, lz4_cost = output_column_wise_results(random_labels, 'lz4', n_clus, names[i], t_random_split, network_speeds, test_data_path, delimiter)
                zstd_throughputs, zstd_cost = output_column_wise_results(random_labels, 'zstd', n_clus, names[i], t_random_split, network_speeds, test_data_path, delimiter)
                column_throughputs_list.append(gzip_throughputs)
                column_throughputs_list.append(lz4_throughputs)
                column_throughputs_list.append(zstd_throughputs)
                column_cost_list.append(gzip_cost)
                column_cost_list.append(lz4_cost)
                column_cost_list.append(zstd_cost)
                r.write('\nrow-wise\n')
                r.flush()
                gzip_throughputs, gzip_cost = output_row_wise_results(random_labels, 'gzip', n_clus, names[i], t_random_split, network_speeds, test_data_path)
                lz4_throughputs, lz4_cost = output_row_wise_results(random_labels, 'lz4', n_clus, names[i], t_random_split, network_speeds, test_data_path)
                zstd_throughputs, zstd_cost = output_row_wise_results(random_labels, 'zstd', n_clus, names[i], t_random_split, network_speeds, test_data_path)
                row_throughputs_list.append(gzip_throughputs)
                row_throughputs_list.append(lz4_throughputs)
                row_throughputs_list.append(zstd_throughputs)
                row_cost_list.append(gzip_cost)
                row_cost_list.append(lz4_cost)
                row_cost_list.append(zstd_cost)
                r.write('-' * 50 + '\n')
                r.flush()
                for buffer_size in buffer_sizes:
                    print('buffer_size:', buffer_size)
                    r.write(f'\nbuffer_size: {buffer_size}\n')
                    r.flush()
                    t_clus, clus_labels = incremental_k_prototypes(dataset, n_clus, num_indexs[i], cate_indexs[i], buffer_size, names[i])
                    r.write('\ncolumn-wise\n')
                    r.flush()
                    gzip_throughputs, gzip_cost = output_column_wise_results(clus_labels, 'gzip', n_clus, names[i], t_clus, network_speeds, test_data_path, delimiter)
                    lz4_throughputs, lz4_cost = output_column_wise_results(clus_labels, 'lz4', n_clus, names[i], t_clus, network_speeds, test_data_path, delimiter)
                    zstd_throughputs, zstd_cost = output_column_wise_results(clus_labels, 'zstd', n_clus, names[i], t_clus, network_speeds, test_data_path, delimiter)
                    column_throughputs_list.append(gzip_throughputs)
                    column_throughputs_list.append(lz4_throughputs)
                    column_throughputs_list.append(zstd_throughputs)
                    column_cost_list.append(gzip_cost)
                    column_cost_list.append(lz4_cost)
                    column_cost_list.append(zstd_cost)
                    r.write('\nrow-wise\n')
                    r.flush()
                    gzip_throughputs, gzip_cost = output_row_wise_results(clus_labels, 'gzip', n_clus, names[i], t_clus, network_speeds, test_data_path)
                    lz4_throughputs, lz4_cost = output_row_wise_results(clus_labels, 'lz4', n_clus, names[i], t_clus, network_speeds, test_data_path)
                    zstd_throughputs, zstd_cost = output_row_wise_results(clus_labels, 'zstd', n_clus, names[i], t_clus, network_speeds, test_data_path)
                    row_throughputs_list.append(gzip_throughputs)
                    row_throughputs_list.append(lz4_throughputs)
                    row_throughputs_list.append(zstd_throughputs)
                    row_cost_list.append(gzip_cost)
                    row_cost_list.append(lz4_cost)
                    row_cost_list.append(zstd_cost)
                    r.write('-' * 50 + '\n')
                    r.flush()
                    for train_size in train_sizes:
                        train_data, test_data, train_labels, test_labels, indices = split_train_test_set(dataset, clus_labels, train_size, 0, 'no_overlap')
                        new_data_path = create_new_table(names[i], indices)
                        r.write(f'\nNo overlap: {train_size * 100}%\n')
                        r.flush()
                        for j in range(len(class_names)):
                            test_time, predictions = train_test_classification_models(train_data, test_data, train_labels, test_labels,
                                                                                        names[i], class_names[j], n_clus, train_size, 'no_overlap', buffer_size)
                            r.write('\ncolumn-wise\n')
                            r.flush()
                            gzip_throughputs, gzip_cost = output_column_wise_results(predictions, 'gzip', n_clus, names[i], test_time, network_speeds, new_data_path, delimiter)
                            lz4_throughputs, lz4_cost = output_column_wise_results(predictions, 'lz4', n_clus, names[i], test_time, network_speeds, new_data_path, delimiter)
                            zstd_throughputs, zstd_cost = output_column_wise_results(predictions, 'zstd', n_clus, names[i], test_time, network_speeds, new_data_path, delimiter)
                            column_throughputs_list.append(gzip_throughputs)
                            column_throughputs_list.append(lz4_throughputs)
                            column_throughputs_list.append(zstd_throughputs)
                            column_cost_list.append(gzip_cost)
                            column_cost_list.append(lz4_cost)
                            column_cost_list.append(zstd_cost)
                            r.write('\nrow-wise\n')
                            r.flush()
                            gzip_throughputs, gzip_cost = output_row_wise_results(predictions, 'gzip', n_clus, names[i], test_time, network_speeds, new_data_path)
                            lz4_throughputs, lz4_cost = output_row_wise_results(predictions, 'lz4', n_clus, names[i], test_time, network_speeds, new_data_path)
                            zstd_throughputs, zstd_cost = output_row_wise_results(predictions, 'zstd', n_clus, names[i], test_time, network_speeds, new_data_path)
                            row_throughputs_list.append(gzip_throughputs)
                            row_throughputs_list.append(lz4_throughputs)
                            row_throughputs_list.append(zstd_throughputs)
                            row_cost_list.append(gzip_cost)
                            row_cost_list.append(lz4_cost)
                            row_cost_list.append(zstd_cost)
                            r.write('-' * 50 + '\n')
                            r.flush()
                        if os.path.exists(new_data_path):
                            os.remove(new_data_path)
                    for train_size in train_sizes:
                        train_data, test_data, train_labels, test_labels, _ = split_train_test_set(dataset, clus_labels, train_size, 0, 'insert')
                        r.write(f'\nInsert: {train_size * 100}%\n')
                        r.flush()
                        for j in range(len(class_names)):
                            test_time, predictions = train_test_classification_models(train_data, test_data, train_labels, test_labels,
                                                                                        names[i], class_names[j], n_clus, train_size, 'insert', buffer_size)
                            r.write('\ncolumn-wise\n')
                            r.flush()
                            gzip_throughputs, gzip_cost = output_column_wise_results(predictions, 'gzip', n_clus, names[i], test_time, network_speeds, test_data_path, delimiter)
                            lz4_throughputs, lz4_cost = output_column_wise_results(predictions, 'lz4', n_clus, names[i], test_time, network_speeds, test_data_path, delimiter)
                            zstd_throughputs, zstd_cost = output_column_wise_results(predictions, 'zstd', n_clus, names[i], test_time, network_speeds, test_data_path, delimiter)
                            column_throughputs_list.append(gzip_throughputs)
                            column_throughputs_list.append(lz4_throughputs)
                            column_throughputs_list.append(zstd_throughputs)
                            column_cost_list.append(gzip_cost)
                            column_cost_list.append(lz4_cost)
                            column_cost_list.append(zstd_cost)
                            r.write('\nrow-wise\n')
                            r.flush()
                            gzip_throughputs, gzip_cost = output_row_wise_results(predictions, 'gzip', n_clus, names[i], test_time, network_speeds, test_data_path)
                            lz4_throughputs, lz4_cost = output_row_wise_results(predictions, 'lz4', n_clus, names[i], test_time, network_speeds, test_data_path)
                            zstd_throughputs, zstd_cost = output_row_wise_results(predictions, 'zstd', n_clus, names[i], test_time, network_speeds, test_data_path)
                            row_throughputs_list.append(gzip_throughputs)
                            row_throughputs_list.append(lz4_throughputs)
                            row_throughputs_list.append(zstd_throughputs)
                            row_cost_list.append(gzip_cost)
                            row_cost_list.append(lz4_cost)
                            row_cost_list.append(zstd_cost)
                            r.write('-' * 50 + '\n')
                            r.flush()

                    for test_size in test_sizes:
                        train_data, test_data, train_labels, test_labels, indices = split_train_test_set(dataset, clus_labels, 0, test_size, 'update')
                        new_data_path = create_new_table(names[i], indices)
                        r.write(f'\nUpdate: {test_size * 100}%\n')
                        r.flush()
                        for j in range(len(class_names)):
                            test_time, predictions = train_test_classification_models(train_data, test_data, train_labels, test_labels,
                                                                                        names[i], class_names[j], n_clus, test_size, 'update', buffer_size)
                            r.write('\ncolumn-wise\n')
                            r.flush()
                            gzip_throughputs, gzip_cost = output_column_wise_results(predictions, 'gzip', n_clus, names[i], test_time, network_speeds, new_data_path, delimiter)
                            lz4_throughputs, lz4_cost = output_column_wise_results(predictions, 'lz4', n_clus, names[i], test_time, network_speeds, new_data_path, delimiter)
                            zstd_throughputs, zstd_cost = output_column_wise_results(predictions, 'zstd', n_clus, names[i], test_time, network_speeds, new_data_path, delimiter)
                            column_throughputs_list.append(gzip_throughputs)
                            column_throughputs_list.append(lz4_throughputs)
                            column_throughputs_list.append(zstd_throughputs)
                            column_cost_list.append(gzip_cost)
                            column_cost_list.append(lz4_cost)
                            column_cost_list.append(zstd_cost)
                            r.write('\nrow-wise\n')
                            r.flush()
                            gzip_throughputs, gzip_cost = output_row_wise_results(predictions, 'gzip', n_clus, names[i], test_time, network_speeds, new_data_path)
                            lz4_throughputs, lz4_cost = output_row_wise_results(predictions, 'lz4', n_clus, names[i], test_time, network_speeds, new_data_path)
                            zstd_throughputs, zstd_cost = output_row_wise_results(predictions, 'zstd', n_clus, names[i], test_time, network_speeds, new_data_path)
                            row_throughputs_list.append(gzip_throughputs)
                            row_throughputs_list.append(lz4_throughputs)
                            row_throughputs_list.append(zstd_throughputs)
                            row_cost_list.append(gzip_cost)
                            row_cost_list.append(lz4_cost)
                            row_cost_list.append(zstd_cost)
                            r.write('-' * 50 + '\n')
                            r.flush()
                        if os.path.exists(new_data_path):
                            os.remove(new_data_path)


if __name__ == '__main__':
    main()
