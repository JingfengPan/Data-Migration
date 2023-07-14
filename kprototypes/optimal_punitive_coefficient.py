import time
import numpy as np
from incremental_k_prototypes import incremental_k_prototypes
from compression import compress
from calculate_size import calculate_size
from matplotlib import pyplot as plt


def optimal_punitive_coefficient(n_clus, name):
    time_list = []
    gzip_list = []
    lz4_list = []
    zst_list = []
    for i in np.arange(0.5, 2, 0.1):
        start = time.time()
        # random_cluster(n_clus, '../../test_data/customer.csv', name)
        incremental_k_prototypes(10000, n_clus, 'Cao', [0, 1, 3, 5, 6], name + str(i), 10000, '../../test_data/DS_001.csv', i)
        compress(n_clus, name + str(i))
        end = time.time()
        print('Time (s):', end - start)
        time_list.append(end - start)
        gzip_size = calculate_size('gzip', n_clus, name + str(i))
        gzip_list.append(gzip_size)
        lz4_size = calculate_size('lz4', n_clus, name + str(i))
        lz4_list.append(lz4_size)
        zst_size = calculate_size('zstandard', n_clus, name + str(i))
        zst_list.append(zst_size)

    print('time list:', time_list)
    plt.figure("optimal time")
    plt.plot(np.arange(0.5, 2, 0.1), time_list)
    plt.xticks(np.arange(0.5, 2, 0.1))
    plt.xlabel('punitive coefficient')
    plt.ylabel('time (s)')
    plt.show()

    print('gzip list:', gzip_list)
    plt.figure("optimal gzip")
    plt.plot(np.arange(0.5, 2, 0.1), gzip_list)
    plt.xticks(np.arange(0.5, 2, 0.1))
    plt.xlabel('punitive coefficient')
    plt.ylabel('file size (KB)')
    plt.show()

    print('lz4 list:', lz4_list)
    plt.figure("optimal lz4")
    plt.plot(np.arange(0.5, 2, 0.1), lz4_list)
    plt.xticks(np.arange(0.5, 2, 0.1))
    plt.xlabel('punitive coefficient')
    plt.ylabel('file size (KB)')
    plt.show()

    print('zst list:', zst_list)
    plt.figure("optimal zst")
    plt.plot(np.arange(0.5, 2, 0.1), zst_list)
    plt.xticks(np.arange(0.5, 2, 0.1))
    plt.xlabel('punitive coefficient')
    plt.ylabel('file size (KB)')
    plt.show()


optimal_punitive_coefficient(15, '10000_15_Cao_001_optimal')
