import time

import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from collections import Counter


# n_data = amount of data to process each time
# n_clus = number of clusters
# ini = 'Cao', 'Huang', 'random'
# cat_index = index list of columns of category data
# name = output file name
def incremental_k_prototypes(n_data, n_clus, ini, cat_index, name, cluster_size, path):  # coeff
    clus_start = time.time()
    with open(path) as f:
        raw_data = f.readlines()
        dataset = []
        for i in range(len(raw_data)):
            pre = raw_data[i].split('|')  # [2:]
            # pre[4] = pre[4].strip()
            # pre[5] = pre[5].strip('\n')
            dataset.append(pre[2:7])

    result = []
    for i in range(n_clus):
        result.append([])

    num_index = []
    for i in range(len(dataset[0])):
        if i not in cat_index:
            num_index.append(i)

    index = 0
    # initialize the lock
    # lock = np.zeros(n_clus)
    # count = np.zeros(n_clus)
    while index < len(dataset):
        if index + n_data <= len(dataset):
            step = n_data
        else:
            step = len(dataset) - index
        data = dataset[index: index + step]
        raw = raw_data[index: index + step]

        if index == 0:
            kp = KPrototypes(n_jobs=-1, n_clusters=n_clus, init=ini)
            clusters = list(kp.fit_predict(np.array(data), categorical=cat_index))
            centroids = kp.cluster_centroids_
            gamma = kp.gamma
            print('Gamma:', gamma)
            for i in range(len(clusters)):
                result[clusters[i]].append(raw[i])

        else:
            clusters.clear()
            for i in range(step):
                cost_list = cost_function(data[i], centroids, gamma, cat_index, num_index)
                # check if the cluster is locked
                # locked = np.where(lock == 1)[0]
                # cost_list[locked] = np.Inf
                # count[locked] += 1
                # for j in locked:
                #    if count[j] == 1000:
                #        count[j] = 0
                #        lock[j] = 0
                #        break
                # add punitive coefficient to cluster size
                dist_list = []
                for j in range(n_clus):
                    dist_list.append(cost_list[j])  # + coeff * gamma * len(result[j]) / cluster_size
                clus_index = np.argmin(dist_list)
                clusters.append(clus_index)
                result[clus_index].append(raw[i])
                if len(result[clus_index]) == cluster_size:
                    clus_end = time.time()
                    print('A cluster is full at:', clus_end - clus_start, 's')
                    output = pd.DataFrame(result[clus_index])
                    output.to_csv('./data/csv/{}_{}.csv'.format(name, clus_index), mode='a', index=False, header=False)
                    result[clus_index].clear()
                    # set the lock
                    # lock[clus_index] = 1
            update_centroids(n_clus, centroids, data, clusters, cat_index, num_index)
        print('Centroids:', centroids)
        index += n_data
    for i in range(n_clus):
        output = pd.DataFrame(result[i])
        output.to_csv('./data/csv/{}_{}.csv'.format(name, i), mode='a', index=False, header=False)

def cost_function(tuple, centroids, gamma, cat_index, num_index):
    cost_list = []
    for i in range(len(centroids)):
        distance = 0
        for j in range(len(num_index)):
            distance += (float(centroids[i, j]) - float(tuple[num_index[j]]))**2
        similarity = 0
        for j in range(len(cat_index)):
            if centroids[i, j + len(num_index)] == tuple[cat_index[j]]:
                similarity += 1
        cost = distance + gamma * similarity
        cost_list.append(cost)
    return np.array(cost_list)

def update_centroids(n_clus, centroids, data, clusters, cat_index, num_index):
    count = []
    for i in range(n_clus):
        count.append(clusters.count(i))
    for i in range(len(num_index)):
        sum = np.zeros(n_clus)
        for j in range(len(data)):
            sum[clusters[j]] += float(data[j][num_index[i]])
        mean = sum / count
        for j in range(n_clus):
            centroids[j, i] = mean[j]

    for i in range(len(cat_index)):
        cat = []
        for j in range(n_clus):
            cat.append([])
        for j in range(len(data)):
            cat[clusters[j]].append(data[j][cat_index[i]])
        for j in range(n_clus):
            centroids[j, i + len(num_index)] = list(Counter(cat[j]).keys())[0]
