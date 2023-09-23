import re
import pandas as pd


def remove_commas_inside_quotes(match):
    return match.group().replace(',', '')


def split_data(n_clus, data_path, label_path, name, file_name, delimiter, column):
    with open(data_path, encoding='utf-8') as dp:
        if file_name == 'econbiz':
            raw_data = dp.readlines()[1:]
        else:
            raw_data = dp.readlines()
    with open(label_path) as lp:
        labels = lp.readlines()
    result = []
    for i in range(n_clus):
        result.append([])
        for j in range(column):
            result[i].append([])
    for i in range(len(raw_data)):
        regex_list = re.sub(r'"[^"]*"', remove_commas_inside_quotes, raw_data[i]).split(delimiter)
        for j in range(len(regex_list)):
            result[int(labels[i])][j].append(regex_list[j])
    for i in range(n_clus):
        for j in range(len(result[i])):
            output = pd.DataFrame(result[i][j])
            output.to_csv(f'./data/csv/{file_name}/{file_name}_{n_clus}_{name}_{i}_{j}.csv', mode='a', index=False, header=False)
