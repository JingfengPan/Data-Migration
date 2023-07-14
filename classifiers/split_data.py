import pandas as pd


def split_data(n_clus, data_path, label_path, name):
    with open(data_path) as dp:
        raw_data = dp.readlines()
    with open(label_path) as lp:
        labels = lp.readlines()
    result = []
    for i in range(n_clus):
        result.append([])
        for j in range(len(raw_data[0].replace(',', ' ').split('|'))):
            result[i].append([])
    for i in range(len(raw_data)):
        list = raw_data[i].replace(',', ' ').split('|')
        for j in range(len(list)):
            result[int(labels[i])][j].append(list[j])
    for i in range(n_clus):
        for j in range(len(result[i])):
            output = pd.DataFrame(result[i][j])
            output.to_csv('./data/csv/{}_{}_{}.csv'.format(name, i, j), mode='a', index=False, header=False)