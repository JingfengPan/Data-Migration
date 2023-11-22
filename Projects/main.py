import os
import pandas as pd
from incremental_k_prototypes import incremental_k_prototypes
from preprocess_data import read_data
from random_split import random_split
from classifiers import split_train_test_set, train_test_classification_models
from split_data import column_wise_split
from parallel_compression import parallel_compression
from calculate_compression_ratio import calculate_compression_ratio
from calculate_throughput import calculate_throughput
from cost_model import cost_model
from draw_figures import draw_figures


def create_new_table(name, indices):
    new_data = []
    with open(f'test_data/{name}.csv', encoding='utf-8') as dp:
        if name == 'econbiz':
            raw_data = dp.readlines()[1:]
        else:
            raw_data = dp.readlines()
        for index in indices:
            new_data.append(raw_data[index].strip('\n'))
        if not os.path.exists(f'./data/csv/{name}'):
            os.makedirs(f'./data/csv/{name}')
        output = pd.DataFrame(new_data)
        output.to_csv(f'./data/csv/{name}/{name}.csv', mode='w', index=False, header=False)


def output_results(format, n_clus, name, class_name, t_clus, network_speeds, column, path, n_processes=10, pipelines=1):
    t_comp = parallel_compression(format, n_clus, class_name, name, column, n_processes)
    comp_ratio = calculate_compression_ratio(format, path,  n_clus, class_name, name, column)
    empty_directory(f'./data/{format}/{name}')
    with open(f'./results/{name}/{name}_{n_clus}_results.txt', 'a') as r:
        r.write(f'\n{format}\n')
        r.write(f'Compression time: {t_comp:.5f} s, Compression ratio: {comp_ratio:.5f}\n')
        for network_speed in network_speeds:
            r.write(f'Network speed: {network_speed} MB/s')
            throughput = calculate_throughput(t_clus, t_comp, comp_ratio, network_speed, path, pipelines)
            r.write(f'Throughput: {throughput:.5f} MB/s\n')
        cost = cost_model(t_clus, t_comp, comp_ratio, path, pipelines)
        r.write(f'Cost: $ {cost:.5f} /TB\n')


def empty_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)


def main():
    names = ['partsupp']  # 'DS_001', , 'orders', 'partsupp'
    num_indexs = [[0, 1, 2, 3]]  # [0, 5], [0, 1, 3, 6], [0, 1, 2, 3]
    cate_indexs = [[]]  # [3, 6], [2, 5], []
    delimiters = ['|']  # '|', '|', '|'
    n_clus_list = [20]
    buffer_sizes = [10000, 20000, 30000, 40000]
    train_sizes = [0.9]
    test_sizes = [0.1]
    class_names = ['DecisionTree', 'RandomForest', 'AdaBoost', 'QDA', 'MLP', 'GaussianNB', 'KNN', 'LogisticRegression']
    n_processors = [5, 10, 15, 20]
    network_speeds = [5, 10, 15, 20]
    pipelines = [1, 5, 10, 15, 20]
    for i in range(len(names)):
        data_path = f'./data/csv/{names[i]}/{names[i]}.csv'
        test_data_path = f'./test_data/{names[i]}.csv'
        for n_clus in n_clus_list:
            dataset = read_data(names[i], n_clus, num_indexs[i], cate_indexs[i], delimiters[i])

            if not os.path.exists(f'./results/{names[i]}'):
                os.makedirs(f'./results/{names[i]}')
            if os.path.exists(f'./results/{names[i]}/{names[i]}_{n_clus}_results.txt'):
                os.remove(f'./results/{names[i]}/{names[i]}_{n_clus}_results.txt')
            with open(f'./results/{names[i]}/{names[i]}_{n_clus}_results.txt', 'a') as r:
                t_clus, clus_labels = incremental_k_prototypes(dataset, n_clus, num_indexs[i], cate_indexs[i], names[i])
                column = column_wise_split(n_clus, test_data_path, clus_labels, 'clus', names[i], delimiters[i])
                output_results('gzip', n_clus, names[i], 'clus', t_clus, network_speeds, column, test_data_path)
                output_results('lz4', n_clus, names[i], 'clus', t_clus, network_speeds, column, test_data_path)
                output_results('zstd', n_clus, names[i], 'clus', t_clus, network_speeds, column, test_data_path)
                r.write('-' * 50 + '\n')
                r.flush()
                empty_directory(f'./data/csv/{names[i]}')

                t_random_split, random_labels = random_split(dataset, n_clus, names[i])
                column = column_wise_split(n_clus, test_data_path, random_labels, 'random', names[i], delimiters[i])
                output_results('gzip', n_clus, names[i], 'random', t_random_split, network_speeds, column, test_data_path)
                output_results('lz4', n_clus, names[i], 'random', t_random_split, network_speeds, column, test_data_path)
                output_results('zstd', n_clus, names[i], 'random', t_random_split, network_speeds, column, test_data_path)
                r.write('-' * 50 + '\n')
                r.flush()
                empty_directory(f'./data/csv/{names[i]}')

                for train_size in train_sizes:
                    train_data, test_data, train_labels, test_labels, indices = split_train_test_set(dataset, clus_labels, train_size, 0, 'no_overlap')
                    create_new_table(names[i], indices)
                    r.write(f'\nNo overlap: {train_size* 100}%\n')
                    r.flush()
                    for j in range(len(class_names)):
                        test_time, predictions = train_test_classification_models(train_data, test_data, train_labels, test_labels,
                                                                                    names[i], class_names[j], n_clus, train_size, 'no_overlap', buffer_size=10000)
                        column = column_wise_split(n_clus, data_path, predictions, class_names[j], names[i], delimiters[i])
                        output_results('gzip', n_clus, names[i], class_names[j], test_time, network_speeds, column, data_path)
                        output_results('lz4', n_clus, names[i], class_names[j], test_time, network_speeds, column, data_path)
                        output_results('zstd', n_clus, names[i], class_names[j], test_time, network_speeds, column, data_path)
                        r.write('-' * 50 + '\n')
                        r.flush()
                    empty_directory(f'./data/csv/{names[i]}')

                for train_size in train_sizes:
                    train_data, test_data, train_labels, test_labels, _ = split_train_test_set(dataset, clus_labels, train_size, 0, 'insert')
                    r.write(f'\nInsert: {train_size * 100}%\n')
                    r.flush()
                    for j in range(len(class_names)):
                        test_time, predictions = train_test_classification_models(train_data, test_data, train_labels, test_labels,
                                                                                    names[i], class_names[j], n_clus, train_size, 'insert', buffer_size=10000)
                        column = column_wise_split(n_clus, test_data_path, predictions, class_names[j], names[i], delimiters[i])
                        output_results('gzip', n_clus, names[i], class_names[j], test_time, network_speeds, column, test_data_path)
                        output_results('lz4', n_clus, names[i], class_names[j], test_time, network_speeds, column, test_data_path)
                        output_results('zstd', n_clus, names[i], class_names[j], test_time, network_speeds, column, test_data_path)
                        r.write('-' * 50 + '\n')
                        r.flush()
                    empty_directory(f'./data/csv/{names[i]}')

                for test_size in test_sizes:
                    train_data, test_data, train_labels, test_labels, indices = split_train_test_set(dataset, clus_labels, 0, test_size, 'update')
                    create_new_table(names[i], indices)
                    r.write(f'\nUpdate: {test_size * 100}%\n')
                    r.flush()
                    for j in range(len(class_names)):
                        test_time, predictions = train_test_classification_models(train_data, test_data, train_labels, test_labels,
                                                                                    names[i], class_names[j], n_clus, train_size, 'update', buffer_size=10000)
                        column = column_wise_split(n_clus, data_path, predictions, class_names[j], names[i], delimiters[i])
                        output_results('gzip', n_clus, names[i], class_names[j], test_time, network_speeds, column, data_path)
                        output_results('lz4', n_clus, names[i], class_names[j], test_time, network_speeds, column, data_path)
                        output_results('zstd', n_clus, names[i], class_names[j], test_time, network_speeds, column, data_path)
                        r.write('-' * 50 + '\n')
                        r.flush()
                    empty_directory(f'./data/csv/{names[i]}')


if __name__ == '__main__':
    main()

