import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans

def read_data(path):
    with open(path) as f:
        raw_data = f.readlines()
        dataset = []
        for i in range(len(raw_data)):
            pre = raw_data[i].split('|')
            temp_list = [pre[3], float(pre[5]), pre[6]]
            dataset.append(temp_list)
    return dataset


#Invoke the k-prototypes algorithm to initialize the centroids
def k_prototypes(n_clus, cate_index, file_name, path):
    data = read_data(path)
    # kp = KMeans(n_clusters=n_clus)
    kp = KPrototypes(n_clusters=n_clus, n_jobs=-1, max_iter=5, n_init=5)
    clusters = kp.fit_predict(np.array(data), categorical=cate_index)
    # centroids = kp.cluster_centroids_
    # gamma = kp.gamma
    output = pd.DataFrame(clusters)
    output.to_csv(f'./test_data/{file_name}/{file_name}_20_labels.csv', index=False, header=False)


def main():
    file_names = ['DS_001', 'DS_002', 'customer']
    for file_name in file_names:
        print(f'Running k-prototypes on {file_name}...')
        k_prototypes(20, [0, 2], file_name, f'./test_data/{file_name}/{file_name}.csv')


if __name__ == '__main__':
    main()
