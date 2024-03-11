import os
import random
import time
import pandas as pd
import numpy as np
from collections import Counter
from numba import njit
from joblib import delayed, Parallel


# n_clus = number of clusters
# cate_index = index list of columns of category data
# num_index = index list of columns of numeric data
# name = output file name
# path = path of the dataset
# n_data = amount of data to process each time, simulate the form of data stream
# cluster_size = max size of each cluster
def incremental_k_prototypes(dataset, n_clus, nums, cates, buffer_size, name):
    num_index = []
    cate_index = []
    for i in range(len(nums)):
        num_index.append(i)
    for i in range(len(cates)):
        cate_index.append(len(nums) + i)

    clus_start = time.time()

    samples = random.sample(range(buffer_size), n_clus)
    centroids = []
    for i in range(len(samples)):
        centroids.append(dataset[:buffer_size][samples[i]])
    # print('Centroids:', centroids)
    output = []
    for index in range(0, len(dataset), buffer_size):
        data = dataset[index: index + min(buffer_size, len(dataset) - index)]

        temp = np.array(data)
        gamma = 0.5 * np.mean(np.array(temp[:, num_index], dtype=float).std(axis=0))
        # print('Gamma:', gamma)
        # clusters = Parallel(n_jobs=-1)(delayed(assign_cluster)(tuple, centroids, gamma, cate_index, num_index) for tuple in data)
        clusters = []
        for i in range(len(data)):
            cost_list = cost_function(data[i], centroids, gamma, cate_index, num_index)
            clusters.append(np.argmin(cost_list))
        update_centroids(n_clus, centroids, data, clusters, cate_index, num_index)
        output.extend(clusters)
    clus_end = time.time()
    # output = pd.DataFrame(clusters)
    # if os.path.exists(f'./test_data/{name}/{name}_{n_clus}_random_labels.csv'):
    #     os.remove(f'./test_data/{name}/{name}_{n_clus}_random_labels.csv')
    # output.to_csv(f'./test_data/{name}/{name}_{n_clus}_clustering_labels.csv', mode='w', index=False, header=False)

    t_clus = clus_end - clus_start
    # print('Clustering Time (s):', t_clus)

    # with open(f'./results/{name}/{name}_results.txt', 'a') as r:
    #     r.write(f'Online Clustering\n')
    #     r.write(f'Clustering time: {t_clus:.5f} s\n')
    return t_clus, output


# def assign_cluster(tuple, centroids, gamma, cate_index, num_index):
#     cost_list = cost_function(tuple, centroids, gamma, cate_index, num_index)
#     return np.argmin(cost_list)


def cost_function(tuple, centroids, gamma, cate_index, num_index):
    cost_list = []
    for i in range(len(centroids)):
        distance = 0
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
    count = [clusters.count(i) for i in range(n_clus)]

    for i in range(len(num_index)):
        sum = np.zeros(n_clus)
        for j in range(len(data)):
            sum[clusters[j]] += data[j][num_index[i]]

        sum_clean = np.nan_to_num(sum, nan=0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        count_clean = np.nan_to_num(count, nan=1, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = np.where(count_clean != 0, np.divide(sum_clean, count_clean), 0)

        for j in range(n_clus):
            centroids[j][num_index[i]] = float(mean[j])
    for i in range(len(cate_index)):
        cate = [[] for _ in range(n_clus)]
        for j in range(len(data)):
            cate[clusters[j]].append(data[j][cate_index[i]])
        for j in range(n_clus):
            if cate[j]:
                centroids[j][cate_index[i]] = list(Counter(cate[j]).keys())[0]
            else:
                centroids[j][cate_index[i]] = 0
