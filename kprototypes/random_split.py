import time
import pandas as pd
import random


def read_data(path):
    with open(path) as f:
        raw_data = f.readlines()
    return raw_data


def random_split(n_clus, name, path):
    raw_data = read_data(path)
    split_start = time.time()
    result = []
    for i in range(n_clus):
        result.append([])
    for i in range(len(raw_data)):
        result[random.randint(0, n_clus - 1)].append(raw_data[i])  # i % n_clus
    split_end = time.time()
    t_split = split_end - split_start
    print('Random split time (s):', t_split)
    for i in range(n_clus):
        csv = pd.DataFrame(result[i])
        csv.to_csv('./data/csv/{}_{}.csv'.format(name, i), index=False, header=False)
    return t_split

