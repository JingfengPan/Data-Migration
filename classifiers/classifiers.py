import time
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def read_data(data_path, labels_path):
    data = pd.read_csv(data_path, header=None)
    labels = pd.read_csv(labels_path, header=None)
    return data, labels


def train_classifiers(name, train_data, train_labels):
    Names = ['Decision Tree', 'Random Forest', 'AdaBoost', 'QDA', 'MLP', 'Gaussian NB']
    classifiers = [
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=10, n_jobs=-1),
        AdaBoostClassifier(n_estimators=10, estimator=DecisionTreeClassifier(max_depth=10)),
        QuadraticDiscriminantAnalysis(),
        MLPClassifier(hidden_layer_sizes=(10,)),
        GaussianNB()
    ]
    for i in range(len(Names)):
        if Names[i] == name:
            clf = classifiers[i]
            break
    clf.fit(train_data, train_labels)
    return clf


def test_classifiers(clf, test_data, test_labels):
    predictions = clf.predict(test_data)
    accuracy = clf.score(test_data, test_labels)
    return predictions, accuracy


def main():
    Names = ['Decision Tree', 'Random Forest', 'AdaBoost', 'QDA', 'MLP', 'Gaussian NB']
    n_clus = [5, 10, 15, 20]
    for i in n_clus:
        for name in Names:
            train_data, train_labels = read_data('./test_data/DS_001/DS_001_train.csv', './test_data/DS_001/DS_001_{}_labels.csv'.format(i))
            train_start = time.time()
            clf = train_classifiers(name, train_data, train_labels)
            train_end = time.time()
            test_data, test_labels = read_data('./test_data/DS_001/DS_001_{}_test.csv'.format(i), './test_data/DS_001/DS_001_{}_shuffle_labels.csv'.format(i))
            test_start = time.time()
            predictions, accuracy = test_classifiers(clf, test_data, test_labels)
            test_end = time.time()
            print('Accuracy:', accuracy)
            print('Training time:', train_end - train_start)
            print('Testing time:', test_end - test_start)
            if name == 'Decision Tree':
                abbr = 'df'
            elif name == 'Random Forest':
                abbr = 'rf'
            elif name == 'AdaBoost':
                abbr = 'aba'
            elif name == 'QDA':
                abbr = 'qda'
            elif name == 'MLP':
                abbr = 'mlp'
            else:
                abbr = 'gnb'
            with open('./test_data/DS_001/DS_001_{}_predictions_{}.csv'.format(i, abbr), 'w') as f:
                for j in range(len(predictions)):
                    f.write(str(predictions[j]) + '\n')


if __name__ == '__main__':
    main()
