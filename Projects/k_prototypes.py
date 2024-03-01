import os
import time
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from preprocess_data import read_data


def k_prototypes(name, n_clus, num_index, cate_index):
    pre_dataset = pd.read_csv(f'./pre_datasets/{name}_preprocess.csv', header=None)
    # New indices for categorical columns will start after the last numerical column
    new_cate_index = list(range(len(num_index), len(num_index) + len(cate_index)))

    st = time.time()

    kp = KMeans(n_clusters=n_clus)
    print(f'{name} K-Prototypes Clustering...')
    clusters = kp.fit_predict(pre_dataset)

    et = time.time()
    t_clus = et - st

    output = pd.DataFrame(clusters)
    if not os.path.exists(f'./labels/{name}'):
        os.makedirs(f'./labels/{name}')
    if os.path.exists(f'./labels/{name}/{name}_{n_clus}_clus.csv'):
        os.remove(f'./labels/{name}/{name}_{n_clus}_clus.csv')
    output.to_csv(f'./labels/{name}/{name}_{n_clus}_clus.csv', index=False, header=False)

    return t_clus, clusters
