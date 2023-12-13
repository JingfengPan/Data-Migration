import os
import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_train_test_set(dataset, labels, train_size, test_size, case):
    indices = np.arange(len(dataset))
    features = np.array(dataset)

    if case == 'no_overlap':
        features_train, features_test, labels_train, labels_test, _, indices_test = train_test_split(features, labels, indices, train_size=train_size, random_state=42)
        indices = indices_test
    elif case == 'insert':
        features_train, _, labels_train, _ = train_test_split(features, labels, train_size=train_size, random_state=42)
        features_test = features
        labels_test = labels

    elif case == 'update':
        features_same, features_diff, labels_same, labels_diff, indices_same, indices_diff = train_test_split(features, labels, indices, test_size=test_size*2, random_state=42)
        features_before, features_after, labels_before, labels_after, _, indices_after = train_test_split(features_diff, labels_diff, indices_diff, test_size=0.5, random_state=42)
        features_train = np.concatenate((features_same, features_before))
        features_test = np.concatenate((features_same, features_after))
        labels_train = np.concatenate((labels_same, labels_before))
        labels_test = np.concatenate((labels_same, labels_after))
        indices = np.concatenate((indices_same, indices_after))

    return features_train, features_test, labels_train, labels_test, indices


def create_prediction_tasks(test_data, clf, buffer_size):
    # Split the test data into subsets that the maximum size of each subset is buffer_size
    tasks = []
    task_nums = len(test_data) // buffer_size
    for i in range(task_nums):
        start = i * buffer_size
        end = (i + 1) * buffer_size if i != task_nums - 1 else len(test_data)
        batch = test_data[start:end]
        tasks.append(delayed(lambda x: clf.predict(x))(batch))
    return tasks


def train_test_classification_models(train_data, test_data, train_labels, test_labels, name, class_name, n_clus, percentage, case, buffer_size):
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10),
        'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=10)),
        'QDA': QuadraticDiscriminantAnalysis(),
        'MLP': MLPClassifier(hidden_layer_sizes=(10,)),
        'GaussianNB': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression()
    }

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    clf = classifiers[class_name]
    # Training
    print(f'Training {class_name} classifier...')
    train_start = time.time()
    clf.fit(train_scaled, train_labels)
    train_end = time.time()
    train_time = train_end - train_start

    # Parallel prediction
    print(f'Testing {class_name} classifier...')
    test_start = time.time()
    prediction_tasks = create_prediction_tasks(test_scaled, clf, buffer_size)
    batch_predictions = Parallel(n_jobs=-1)(prediction_tasks)
    test_end = time.time()
    test_time = test_end - test_start
    predictions = np.concatenate(batch_predictions)
    accuracy = np.mean(predictions == np.array(test_labels))

    with open(f'./results/{name}/{name}_{n_clus}_results.txt', 'a') as r:
        r.write(f'\n{case} | {percentage * 100}% | {class_name}\n')
        r.write(f'Accuracy: {accuracy:.5f}\n')
        r.write(f'Training time: {train_time:.5f} s\n')
        r.write(f'Testing time: {test_time:.5f} s\n')

    # with open(f'./test_data/{name}/{class_name}_{n_clus}_predictions.csv', 'w') as p:
    #     for j in range(len(predictions)):
    #         p.write(str(predictions[j]) + '\n')

    return test_time, predictions
