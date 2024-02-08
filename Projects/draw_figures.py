import matplotlib
import matplotlib.pyplot as plt


def draw_figures(output_list, file_name):
    network_speeds = [5, 10, 15, 20]
    names = ['DecisionTree', 'QDA', 'MLP', 'GaussianNB', 'LogisticRegression', 'RandomSplit', 'KPrototypes']  # 'RandomForest', 'AdaBoost', 'KNN'

    font = {'size': 16}
    matplotlib.rc('font', **font)
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(file_name)
    plt.subplot(131)

    for i in range(len(names)):
        if i != len(names) - 1:
            plt.plot(range(len(network_speeds)), output_list[0][i], label=names[i], marker='o')
        else:
            plt.plot(range(len(network_speeds)), output_list[0][i], label=names[i], linestyle='dashed', marker='*')
    plt.xlabel('Network Speed (MB/s)')
    plt.ylabel('Cost ($/TB)')
    plt.xticks(range(len(network_speeds)), network_speeds)
    plt.title('Gzip')

    plt.legend(bbox_to_anchor=(2.8, 1.6), ncol=3)

    plt.subplot(132)
    for i in range(len(names)):
        if i != len(names) - 1:
            plt.plot(range(len(network_speeds)), output_list[1][i], label=names[i], marker='o')
        else:
            plt.plot(range(len(network_speeds)), output_list[1][i], label=names[i], linestyle='dashed', marker='*')
    plt.xlabel('Network Speed (MB/s)')
    plt.ylabel('Cost ($/TB)')
    plt.xticks(range(len(network_speeds)), network_speeds)
    plt.title('LZ4')

    plt.subplot(133)
    for i in range(len(names)):
        if i != len(names) - 1:
            plt.plot(range(len(network_speeds)), output_list[2][i], label=names[i], marker='o')
        else:
            plt.plot(range(len(network_speeds)), output_list[2][i], label=names[i], linestyle='dashed', marker='*')
    plt.xlabel('Network Speed (MB/s)')
    plt.ylabel('Cost ($/TB)')
    plt.xticks(range(len(network_speeds)), network_speeds)
    plt.title('Zstandard')

    plt.show()

