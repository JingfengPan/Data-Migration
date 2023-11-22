import pandas as pd
from incremental_k_prototypes import incremental_k_prototypes
from preprocess_data import read_data
from classifiers import split_train_test_set, train_test_classification_models

# dataset = read_data('orders', 20, [0, 1, 3, 4, 6], [2, 5, 7], '|')
# t_clus, clus_labels = incremental_k_prototypes(dataset, 20, [0, 1, 3, 4, 6], [2, 5, 7], 'orders')
# train_data, test_data, train_labels, test_labels, indices = split_train_test_set(dataset, clus_labels, 0.9, 0, 'no_overlap')

