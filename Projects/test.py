from incremental_k_prototypes import incremental_k_prototypes
from preprocess_data import read_data
from classifiers import split_train_test_set, train_test_classification_models
from parallel_column_wise_compression import parallel_column_wise_compression
from parallel_row_based_compression import parallel_row_based_compression
from calculate_throughput import calculate_throughput
from cost_model import cost_model


def create_new_table(name, indices):
    new_data = []
    with open(f'test_data/{name}.csv', encoding='utf-8') as dp:
        raw_data = dp.readlines()
        for index in indices:
            new_data.append(raw_data[index].strip('\n'))
        return new_data


def output_column_wise_results(format, n_clus, name, t_clus, network_speeds, path, n_processes):
    original_size, compressed_size, comp_ratio, t_comp = parallel_column_wise_compression(n_clus, path, clus_labels, format, n_processes)
    with open(f'./results/{name}/{name}_column_wise_results.txt', 'a') as r:
        r.write(f'{format}\n')
        r.write(f'Compression time: {t_comp:.5f} s, Compression ratio: {comp_ratio:.5f}\n')
        for network_speed in network_speeds:
            r.write(f'Network speed: {network_speed} MB/s\n')
            throughput = calculate_throughput(t_clus, t_comp, comp_ratio, network_speed, path)
            r.write(f'Throughput: {throughput:.5f} MB/s\n')
        cost = cost_model(t_clus, t_comp, comp_ratio, path)
        r.write(f'Cost: $ {cost:.5f} /TB\n\n')


def output_row_based_results(format, n_clus, name, t_clus, network_speeds, path, n_processes):
    original_size, compressed_size, comp_ratio, t_comp = parallel_row_based_compression(n_clus, path, clus_labels, format, n_processes)
    with open(f'./results/{name}/{name}_row_based_results.txt', 'a') as r:
        r.write(f'{format}\n')
        r.write(f'Compression time: {t_comp:.5f} s, Compression ratio: {comp_ratio:.5f}\n')
        for network_speed in network_speeds:
            r.write(f'Network speed: {network_speed} MB/s\n')
            throughput = calculate_throughput(t_clus, t_comp, comp_ratio, network_speed, path)
            r.write(f'Throughput: {throughput:.5f} MB/s\n')
        cost = cost_model(t_clus, t_comp, comp_ratio, path)
        r.write(f'Cost: $ {cost:.5f} /TB\n\n')


names = ['DS_001', 'orders', 'partsupp']
num_indexs = [[0, 5], [0, 1, 3, 6], [0, 1, 2, 3]]
cate_indexs = [[3, 6], [2, 5], []]
n_clus_list = [5, 10, 15]
network_speeds = [10, 20, 30]
class_names = ['DecisionTree', 'RandomForest', 'AdaBoost', 'QDA', 'MLP', 'GaussianNB', 'KNN', 'LogisticRegression']
train_size = 0.5
for i in range(len(names)):
    print(names[i])
    test_data_path = f'./test_data/{names[i]}.csv'
    for n_clus in n_clus_list:
        print(n_clus)
        dataset = read_data(names[i], n_clus, num_indexs[i], cate_indexs[i], '|')
        t_clus, clus_labels = incremental_k_prototypes(dataset, n_clus, num_indexs[i], cate_indexs[i], names[i])
        train_data, test_data, train_labels, test_labels, indices = split_train_test_set(dataset, clus_labels, train_size, 0, 'no_overlap')
        test_dataset = create_new_table(names[i], indices)
        for j in range(len(class_names)):
            test_time, predictions = train_test_classification_models(train_data, test_data, train_labels, test_labels,
                                                                      names[i], class_names[j], n_clus, train_size,
                                                                      'no_overlap', buffer_size=10000)

            with open(f'./results/{names[i]}/{names[i]}_row_based_results.txt', 'a') as r:
                r.write(f'Clusters: {n_clus}\n')
                r.flush()
            output_row_based_results('gzip', n_clus, names[i], t_clus, network_speeds, test_data_path)
            output_row_based_results('lz4', n_clus, names[i], t_clus, network_speeds, test_data_path)
            output_row_based_results('zstd', n_clus, names[i], t_clus, network_speeds, test_data_path)

            with open(f'./results/{names[i]}/{names[i]}_column_wise_results.txt', 'a') as r:
                r.write(f'Clusters: {n_clus}\n')
                r.flush()
            output_column_wise_results('gzip', n_clus, names[i], t_clus, network_speeds, test_data_path)
            output_column_wise_results('lz4', n_clus, names[i], t_clus, network_speeds, test_data_path)
            output_column_wise_results('zstd', n_clus, names[i], t_clus, network_speeds, test_data_path)
