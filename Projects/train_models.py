import os
import pandas as pd
from incremental_k_prototypes import incremental_k_prototypes
from preprocess_data import read_data
from classifiers import split_train_test_set, train_and_save_model, load_and_test_model


def main():
    names = ['orders', 'partsupp']  # 'DS_001', 'DS_002', 'DS_003'
    num_indexs = [[0, 1, 3, 6], [0, 1, 2, 3]]  # [0, 5], [0, 5], [0, 5]
    cate_indexs = [[2, 5], []]  # [3, 6], [3, 6], [3, 6]
    delimiter = '|'
    n_clus = 20
    buffer_size = 10000
    train_size = 0.8
    test_size = 0.1
    class_names = ['DecisionTree', 'QDA', 'MLP', 'GaussianNB', 'LogisticRegression']
    for i in range(len(names)):
        dataset = read_data(names[i], num_indexs[i], cate_indexs[i], delimiter)
        t_clus, clus_labels = incremental_k_prototypes(dataset, n_clus, num_indexs[i], cate_indexs[i], buffer_size, names[i])

        train_data, test_data, train_labels, test_labels, indices = split_train_test_set(dataset, clus_labels, train_size, 0, 'no_overlap', names[i])
        for j in range(len(class_names)):
            model_path, scaler_path, train_time = train_and_save_model(train_data, train_labels, names[i], class_names[j], 'no_overlap', train_size)
            load_and_test_model(test_data, test_labels, model_path, scaler_path)

        train_data, test_data, train_labels, test_labels, _ = split_train_test_set(dataset, clus_labels, train_size, 0, 'insert', names[i])
        for j in range(len(class_names)):
            model_path, scaler_path, train_time = train_and_save_model(train_data, train_labels, names[i], class_names[j], 'insert', train_size)
            load_and_test_model(test_data, test_labels, model_path, scaler_path)

        train_data, test_data, train_labels, test_labels, indices = split_train_test_set(dataset, clus_labels, 0, test_size, 'update', names[i])
        for j in range(len(class_names)):
            model_path, scaler_path, train_time = train_and_save_model(train_data, train_labels, names[i], class_names[j], 'update', test_size)
            load_and_test_model(test_data, test_labels, model_path, scaler_path)


if __name__ == '__main__':
    main()
