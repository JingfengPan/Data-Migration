import pandas as pd


def column_wise(n_clus, name, columns):
    for i in range(n_clus):
        result = []
        for j in range(columns):
            result.append([])
        with open('./data/csv/{}_{}.csv'.format(name, i)) as dp:
            raw_data = dp.readlines()
        for j in range(len(raw_data)):
            list = raw_data[j].replace(',', ' ').replace('"', '').split('|')
            for k in range(len(list)):
                result[k].append(list[k])
        for j in range(columns):
            output = pd.DataFrame(result[j])
            output.to_csv('./data/csv/{}_{}_{}.csv'.format(name, i, j), mode='a', index=False, header=False)
