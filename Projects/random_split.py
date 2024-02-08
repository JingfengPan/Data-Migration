import os
import time
import pandas as pd
from joblib import delayed, Parallel


def random_split(dataset, n_clus, name):
    clusters = []
    split_start = time.time()
    for i in range(len(dataset)):
        clusters.append(i % n_clus)
    split_end = time.time()
    t_random_split = split_end - split_start
    # output = pd.DataFrame(clusters)
    # if os.path.exists(f'./test_data/{name}/{name}_{n_clus}_random_labels.csv'):
    #     os.remove(f'./test_data/{name}/{name}_{n_clus}_random_labels.csv')
    # output.to_csv(f'./test_data/{name}/{name}_{n_clus}_random_labels.csv', mode='w', index=False, header=False)
    with open(f'./results/{name}/{name}_results.txt', 'a') as r:
        r.write(f'\nRandom Split\n')
        r.write(f'Random split time: {t_random_split:.5f} s\n')
    return t_random_split, clusters


'''
def parallel_random_split(batch, n_clus, name):
    clusters = []
    for i in range(len(batch)):
        clusters.append(i % n_clus)
    output = pd.DataFrame(clusters)
    output.to_csv(f'./test_data/{name}/{name}_{n_clus}_random_labels.csv', mode='a', index=False, header=False)


def create_tasks(n_clus, name, buffer_size):
    with open(f'./test_data/{name}/{name}.csv', encoding='utf-8') as f:
        raw_data = f.readlines()
    tasks = []
    task_nums = len(raw_data) // buffer_size
    for i in range(task_nums):
        start = i * buffer_size
        end = (i + 1) * buffer_size if i != task_nums - 1 else len(raw_data)
        batch = raw_data[start:end]
        tasks.append(delayed(parallel_random_split)(batch, n_clus, name))
    return tasks
'''
