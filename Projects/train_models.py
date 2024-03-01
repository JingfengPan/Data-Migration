import pandas as pd
from incremental_k_prototypes import incremental_k_prototypes
from preprocess_data import read_data
from classifiers import split_train_test_set, train_and_save_model, load_and_test_model
from generate_test_set import generate_test_set
from k_prototypes import k_prototypes


def main():
    names = ['DS_001', 'DS_002', 'DS_003', 'orders', 'partsupp']
    n_clus = 10
    class_names = ['DecisionTree', 'QDA', 'MLP', 'GaussianNB', 'LogisticRegression']
    for i in range(len(names)):
        pre_dataset = pd.read_csv(f'./pre_datasets/{names[i]}_preprocess.csv', header=None)
        clus_labels = pd.read_csv(f'./labels/{names[i]}/{names[i]}_{n_clus}_clus.csv', header=None)

        train_data, test_data, train_labels, test_labels, indices = split_train_test_set(pre_dataset, clus_labels, 0.2, 0, 'no_overlap', names[i])
        for j in range(len(class_names)):
            model_path, _ = train_and_save_model(train_data, train_labels, names[i], class_names[j], 'no_overlap', 0.2)
            generate_test_set(names[i], 'no_overlap', 0.2)
            # load_and_test_model(test_data, test_labels, model_path, scaler_path)

        train_data, test_data, train_labels, test_labels, _ = split_train_test_set(pre_dataset, clus_labels, 0.8, 0, 'insert', names[i])
        for j in range(len(class_names)):
            model_path, _ = train_and_save_model(train_data, train_labels, names[i], class_names[j], 'insert', 0.8)
            # load_and_test_model(test_data, test_labels, model_path, scaler_path)

        train_data, test_data, train_labels, test_labels, indices = split_train_test_set(pre_dataset, clus_labels, 0, 0.1, 'update', names[i])
        for j in range(len(class_names)):
            model_path, _ = train_and_save_model(train_data, train_labels, names[i], class_names[j], 'update', 0.1)
            generate_test_set(names[i], 'update', 0.1)
            # load_and_test_model(test_data, test_labels, model_path, scaler_path)


if __name__ == '__main__':
    main()
