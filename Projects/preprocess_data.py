import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def remove_commas_inside_quotes(match):
    return match.group().replace(',', '')


# DS_001, 002, 003: num_index = [0, 5], cate_index = [3, 6]
# orders: num_index = [0, 1, 3, 6], cate_index = [2, 5]
# partsupp: num_index = [0, 1, 2, 3], cate_index = []


def read_data(name, num_index, cate_index, delimiter):
    with open(f'./datasets/{name}.csv', encoding='utf-8') as f:
        raw_data = f.readlines()
        num_data = []  # To collect numerical data
        cate_data = []  # To collect categorical data

        for i in range(len(raw_data)):
            pre = re.sub(r'"[^"]*"', remove_commas_inside_quotes, raw_data[i]).split(delimiter)

            if name in ('DS_001', 'DS_002', 'DS_003'):
                if pre[6].replace(' ', '') == 'FURNITURE':
                    pre[6] = 0
                elif pre[6].replace(' ', '') == 'BUILDING':
                    pre[6] = 1
                elif pre[6].replace(' ', '') == 'HOUSEHOLD':
                    pre[6] = 2
                elif pre[6].replace(' ', '') == 'MACHINERY':
                    pre[6] = 3
                elif pre[6].replace(' ', '') == 'AUTOMOBILE':
                    pre[6] = 4

            elif name == 'orders':
                if pre[5] == '1-URGENT':
                    pre[5] = 1
                elif pre[5] == '2-HIGH':
                    pre[5] = 2
                elif pre[5] == '3-MEDIUM':
                    pre[5] = 3
                elif pre[5] == '4-NOT SPECIFIED':
                    pre[5] = 4
                elif pre[5] == '5-LOW':
                    pre[5] = 5
                if pre[2] == 'O':
                    pre[2] = 0
                elif pre[2] == 'F':
                    pre[2] = 1
                elif pre[2] == 'P':
                    pre[2] = 2
                pre[6] = int(pre[6][10:])

            # Collecting numerical and categorical data separately
            num_list = [pre[j] for j in num_index]
            cate_list = [pre[j] for j in cate_index]
            num_data.append(num_list)
            cate_data.append(cate_list)

        # Convert lists to numpy arrays for processing
        num_features = np.array(num_data)
        cate_features = np.array(cate_data)

        # Scale the numerical features
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num_features)

        # Combine scaled numerical and original categorical features
        pre_dataset = np.hstack((num_scaled, cate_features))

        output = pd.DataFrame(pre_dataset)
        if not os.path.exists(f'./pre_datasets'):
            os.makedirs(f'./pre_datasets')
        if os.path.exists(f'./pre_datasets/{name}_preprocess.csv'):
            os.remove(f'./pre_datasets/{name}_preprocess.csv')
        output.to_csv(f'./pre_datasets/{name}_preprocess.csv', mode='a', index=False, header=False)

    return pre_dataset
