import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def select_attributes(data_path, labels_path, train_size, file_name, n_clus):
    with open(data_path) as f:
        data = f.readlines()
    features = []
    for i in range(len(data)):
        temp_list = data[i].split('|')
        if temp_list[6].replace(' ', '') == 'FURNITURE':
            temp_list[6] = 0
        elif temp_list[6].replace(' ', '') == 'BUILDING':
            temp_list[6] = 1
        elif temp_list[6].replace(' ', '') == 'HOUSEHOLD':
            temp_list[6] = 2
        elif temp_list[6].replace(' ', '') == 'MACHINERY':
            temp_list[6] = 3
        elif temp_list[6].replace(' ', '') == 'AUTOMOBILE':
            temp_list[6] = 4
        features.append([temp_list[3], float(temp_list[5]), temp_list[6]])
    features = np.array(features)
    labels = pd.read_csv(labels_path, header=None)
    features_train, _, labels_train, _ = train_test_split(features, labels, train_size=train_size, random_state=0)
    features_train = features_train.T
    train_selected_attr = pd.DataFrame(features_train).T
    train_selected_attr.to_csv(f'./test_data/{file_name}/{file_name}_{n_clus}_train.csv', mode='a', index=False, header=False)
    train_labels = pd.DataFrame(labels_train)
    train_labels.to_csv(f'./test_data/{file_name}/{file_name}_{n_clus}_train_labels.csv', mode='a', index=False, header=False)
    features = features.T
    test_selected_attr = pd.DataFrame(features).T
    test_selected_attr.to_csv(f'./test_data/{file_name}/{file_name}_{n_clus}_test.csv', mode='a', index=False, header=False)


def create_prediction_tasks(test_data, clf, n_jobs=10):
    # Split the test data into n_jobs batches
    tasks = []
    batch_size = len(test_data) // n_jobs
    for i in range(n_jobs):
        start = i * batch_size
        end = (i + 1) * batch_size if i != n_jobs - 1 else len(test_data)
        batch = test_data[start:end]
        tasks.append(delayed(lambda x: clf.predict(x))(batch))
    return tasks


def main():
    file_names = ['DS_001', 'DS_002', 'customer']  # 'econbiz', 'orders', 'partsupp'
    n_clus = 20
    delimiter = ','
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(n_estimators=10, n_jobs=-1),
        'AdaBoost': AdaBoostClassifier(n_estimators=10, estimator=DecisionTreeClassifier(max_depth=10)),
        'QDA': QuadraticDiscriminantAnalysis(),
        'MLP': MLPClassifier(hidden_layer_sizes=(10,)),
        'GaussianNB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_jobs=-1),
        'LogisticRegression': LogisticRegression(n_jobs=-1)
    }
    for file_name in file_names:
        print(f'Processing {file_name} dataset...')
        select_attributes(f'./test_data/{file_name}/{file_name}.csv', f'./test_data/{file_name}/{file_name}_{n_clus}_labels.csv', 0.9, file_name, n_clus)
        train_data = pd.read_csv(f'./test_data/{file_name}/{file_name}_{n_clus}_train.csv', delimiter=delimiter, header=None)
        train_labels = pd.read_csv(f'./test_data/{file_name}/{file_name}_{n_clus}_train_labels.csv', header=None)
        test_data = pd.read_csv(f'./test_data/{file_name}/{file_name}_{n_clus}_test.csv', delimiter=delimiter, header=None)
        test_labels = pd.read_csv(f'./test_data/{file_name}/{file_name}_{n_clus}_labels.csv', header=None)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        for name, clf in classifiers.items():
            # Training
            print(f'Training {name} classifier...')
            train_start = time.time()
            clf.fit(train_scaled, train_labels.values.ravel())
            train_end = time.time()

            # Parallel prediction
            print(f'Testing {name} classifier...')
            test_start = time.time()
            prediction_tasks = create_prediction_tasks(test_scaled, clf)
            batch_predictions = Parallel(n_jobs=-1)(prediction_tasks)
            test_end = time.time()
            predictions = np.concatenate(batch_predictions)

            accuracy = np.mean(predictions == test_labels.values.ravel())

            with open(f'./results/{file_name}/{file_name}_{n_clus}_parallel_classification_results.txt', 'a') as r:
                r.write(name + '\n')
                r.write(f'Accuracy: {accuracy:.5f}\n')
                r.write(f'Training time: {(train_end - train_start):.5f} s\n')
                r.write(f'Testing time: {(test_end - test_start):.5f} s\n\n')

            with open(f'./test_data/{file_name}/{file_name}_{n_clus}_predictions_{name}.csv', 'w') as p:
                for j in range(len(predictions)):
                    p.write(str(predictions[j]) + '\n')


if __name__ == '__main__':
    main()
