import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes


def read_data(path, num_index):
    with open(path) as f:
        raw_data = f.readlines()
        dataset = []
        for i in range(len(raw_data)):
            pre = raw_data[i].split('|')
            dataset.append(pre[3:7])
            for j in range(len(dataset[i])):
                if j in num_index:
                    dataset[i][j] = float(dataset[i][j])
    return dataset


#Invoke the k-prototypes algorithm to initialize the centroids
def k_prototypes(n_clus, cate_index, num_index, name, path):
    data = read_data(path, num_index)
    kp = KPrototypes(n_jobs=-1, n_clusters=n_clus, max_iter=5, n_init=5)
    clusters = kp.fit_predict(np.array(data), categorical=cate_index)
    # centroids = kp.cluster_centroids_
    # gamma = kp.gamma
    output = pd.DataFrame(clusters)
    output.to_csv('./test_data/DS_001/{}.csv'.format(name), index=False, header=False)


def main():
    k_prototypes(20, [0, 1, 3], [2], 'DS_001_20_labels', './test_data/DS_001/DS_001.csv')


if __name__ == '__main__':
    main()
