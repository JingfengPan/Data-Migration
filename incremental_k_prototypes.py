import random
import time
import pandas as pd
import numpy as np
# from kmodes.kprototypes import KPrototypes
from collections import Counter
from numba import njit


# n_clus = number of clusters
# cate_index = index list of columns of category data
# num_index = index list of columns of numeric data
# name = output file name
# path = path of the dataset
# n_data = amount of data to process each time
# cluster_size = max size of each cluster

def read_data(path, num_index):
    with open(path) as f:
        raw_data = f.readlines()
        dataset = []
        for i in range(len(raw_data)):
            pre = raw_data[i].split('|')
            dataset.append(pre[2:7])
            for j in range(len(dataset[i])):
                if j in num_index:
                    dataset[i][j] = float(dataset[i][j])
    return dataset, raw_data


def incremental_k_prototypes(n_clus, cate_index, num_index, name, path, n_data=10000, cluster_size=10000):  # coeff
    dataset, raw_data = read_data(path, num_index)

    clus_start = time.time()
    first_full = 1

    result = []
    for i in range(n_clus):
        result.append([])

    index = 0
    while index < len(dataset):
        if index + n_data <= len(dataset):
            step = n_data
        else:
            step = len(dataset) - index
        data = dataset[index: index + step]
        raw = raw_data[index: index + step]

        if index == 0:
            # Invoke the k-prototypes algorithm to initialize the centroids
            # kp = KPrototypes(n_jobs=-1, n_clusters=n_clus)
            # clusters = list(kp.fit_predict(np.array(data), categorical=cate_index))
            # centroids = kp.cluster_centroids_
            # gamma = kp.gamma
            # for i in range(len(clusters)):
            #    result[clusters[i]].append(raw[i])
            for i in range(n_clus):
                samples = random.sample(range(n_data), n_clus)
            centroids = []
            for i in range(len(samples)):
                centroids.append(data[samples[i]])
            print('Centroids:', centroids)
            temp = np.array(data)
            gamma = 0.5 * np.mean(np.array(temp[:, num_index], dtype=float).std(axis=0))
            print('Gamma:', gamma)
            clusters = []
            for i in range(step):
                cost_list = cost_function(data[i], centroids, gamma, cate_index, num_index)
                clus_index = np.argmin(cost_list)
                clusters.append(clus_index)
                result[clus_index].append(raw[i])
        else:
            clusters.clear()
            temp = np.array(data)
            gamma = 0.5 * np.mean(np.array(temp[:, num_index], dtype=float).std(axis=0))
            print('Gamma:', gamma)
            for i in range(step):
                cost_list = cost_function(data[i], centroids, gamma, cate_index, num_index)
                # add punitive coefficient to cluster size
                # dist_list = []
                # for j in range(n_clus):
                #    dist_list.append(cost_list[j] + coeff * gamma * len(result[j]) / cluster_size)
                clus_index = np.argmin(cost_list)
                clusters.append(clus_index)
                result[clus_index].append(raw[i])
                if len(result[clus_index]) == cluster_size:
                    if first_full == 1:
                        first_full_time = time.time()
                        first_full = 0
                    output = pd.DataFrame(result[clus_index])
                    output.to_csv('./data/csv/{}_{}.csv'.format(name, clus_index), mode='a', index=False, header=False)
                    result[clus_index].clear()
            update_centroids(n_clus, centroids, data, clusters, cate_index, num_index)
            print('Centroids:', centroids)
        index += n_data

    for i in range(n_clus):
        output = pd.DataFrame(result[i])
        output.to_csv('./data/csv/{}_{}.csv'.format(name, i), mode='a', index=False, header=False)
    clus_end = time.time()
    print('First Cluster Full Time (s):', first_full_time - clus_start)
    print('Clustering Time (s):', clus_end - clus_start)


def cost_function(tuple, centroids, gamma, cate_index, num_index):
    cost_list = []
    for i in range(len(centroids)):
        distance = 0.0
        for j in range(len(num_index)):
            centroids_num = centroids[i][num_index[j]]
            tuple_num = tuple[num_index[j]]
            distance += calculate_distance(centroids_num, tuple_num)
        similarity = 0
        for j in range(len(cate_index)):
            if centroids[i][cate_index[j]] == tuple[cate_index[j]]:
                similarity += 1
        cost = distance + gamma * similarity
        cost_list.append(cost)
    return np.array(cost_list)


@njit
def calculate_distance(centroid_num, tuple_num):
    return (centroid_num - tuple_num)**2


def update_centroids(n_clus, centroids, data, clusters, cate_index, num_index):
    count = []
    for i in range(n_clus):
        count.append(clusters.count(i))
    for i in range(len(num_index)):
        sum = np.zeros(n_clus)
        for j in range(len(data)):
            sum[clusters[j]] += data[j][num_index[i]]
        mean = sum / count
        for j in range(n_clus):
            centroids[j][num_index[i]] = float(mean[j])

    for i in range(len(cate_index)):
        cate = []
        for j in range(n_clus):
            cate.append([])
        for j in range(len(data)):
            cate[clusters[j]].append(data[j][cate_index[i]])
        for j in range(n_clus):
            centroids[j][cate_index[i]] = list(Counter(cate[j]).keys())[0]
