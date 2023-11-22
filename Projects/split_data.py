import os
import re
import pandas as pd


def remove_commas_inside_quotes(match):
    return match.group().replace(',', '')


def column_wise_split(n_clus, data_path, labels, class_name, name, delimiter):
    with open(data_path, encoding='utf-8') as dp:
        if name == 'econbiz':
            data = dp.readlines()[1:]
        else:
            data = dp.readlines()
    column = len(re.sub(r'"[^"]*"', remove_commas_inside_quotes, data[0]).split(delimiter))
    result = []
    for i in range(n_clus):
        result.append([])
        for j in range(column):
            result[i].append([])
    for i in range(len(data)):
        regex_list = re.sub(r'"[^"]*"', remove_commas_inside_quotes, data[i]).split(delimiter)
        for j in range(len(regex_list)):
            result[int(labels[i])][j].append(regex_list[j])
    for i in range(n_clus):
        if not os.path.exists(f'./data/csv/{name}'):
            os.makedirs(f'./data/csv/{name}')
        for j in range(len(result[i])):
            output = pd.DataFrame(result[i][j])
            output.to_csv(f'./data/csv/{name}/{name}_{n_clus}_{class_name}_{i}_{j}.csv', mode='w', index=False, header=False)
    return column
