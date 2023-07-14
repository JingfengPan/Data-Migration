import random


def combine_labels(data_path, labels_path):
    with open(data_path) as dp:
        raw_data = dp.readlines()
    with open(labels_path) as lp:
        labels = lp.readlines()
    data = []
    for i in range(len(raw_data)):
        str = raw_data[i].strip('\n') + ' || ' + labels[i]
        data.append(str)
    return data


def shuffle_data(data_path, labels_path):
    data = combine_labels(data_path, labels_path)
    random.shuffle(data)
    return data


def split_labels(data_path, labels_path, data_output_path, labels_output_path):
    data = shuffle_data(data_path, labels_path)
    data_output = []
    labels_output = []
    for i in range(len(data)):
        output = data[i].split(' || ')
        data_output.append(output[0] + '\n')
        labels_output.append(output[1])
    with open(data_output_path, 'w') as dop:
        dop.writelines(data_output)
    with open(labels_output_path, 'w') as lop:
        lop.writelines(labels_output)


def preprocess_data(data_path):
    with open(data_path) as dp:
        raw_data = dp.readlines()
    data = []
    for i in range(len(raw_data)):
        preprocess_str = raw_data[i].replace(',', ' ').replace('|', ',')
        data.append(preprocess_str)
    return data


def select_attributes(data_path, output_path):
    data = preprocess_data(data_path)
    output = []
    for i in range(len(data)):
        list = data[i].split(',')
        if list[6].replace(' ', '') == 'FURNITURE':
            list[6] = 0
        elif list[6].replace(' ', '') == 'MACHINERY':
            list[6] = 1
        elif list[6].replace(' ', '') == 'BUILDING':
            list[6] = 2
        elif list[6].replace(' ', '') == 'HOUSEHOLD':
            list[6] = 3
        elif list[6].replace(' ', '') == 'AUTOMOBILE':
            list[6] = 4
        selected_attr = list[3] + ',' + list[5] + ',' + str(list[6]) + '\n'
        output.append(selected_attr)
    with open(output_path, 'w') as op:
        op.writelines(output)


def main():
    split_labels('./test_data/DS_001/DS_001.csv', './test_data/DS_001/DS_001_labels_20.csv', './test_data/DS_001/DS_001_20_shuffle.csv', './test_data/DS_001/DS_001_20_shuffle_labels.csv')
    # select_attributes('./test_data/DS_001/DS_001.csv', './test_data/DS_001/DS_001_train.csv')
    select_attributes('./test_data/DS_001/DS_001_20_shuffle.csv', './test_data/DS_001/DS_001_20_test.csv')


if __name__ == '__main__':
    main()

