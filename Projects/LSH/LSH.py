from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from parallel_column_wise_compression import parallel_column_wise_compression


# Shingles and One-hot encoding Functions
def build_shingles(sentence: str, k: int):
    shingles = []
    for i in range(len(sentence) - k):
        shingles.append(sentence[i:i + k])
    return set(shingles)


def build_vocab_high_rate(shingle_sets: list, threshold: int):
    print("Build Vocabulary...")
    vocab = {}
    for set_ in shingle_sets:
        for item in set_:
            if item not in vocab:
                vocab[item] = 0
            vocab[item] += 1
    new_vocab = {}
    i = 0
    ks = vocab.keys()
    for k in ks:
        if vocab[k] > threshold:
            new_vocab[k] = i
            i += 1
    return new_vocab


def one_hot(shingles: set, vocab: dict):
    vec = np.zeros(len(vocab))
    for shingle in shingles:
        if shingle in vocab:
            idx = vocab[shingle]
            vec[idx] = 1
    return vec


# Minhash functions
def minhash_arr(vocab: dict, resolution: int):
    length = len(vocab.keys())
    arr = np.zeros((resolution, length))

    print("Minhash Transforming...")
    for i in tqdm(range(resolution)):
        permutation = np.random.permutation(len(vocab)) + 1
        arr[i, :] = permutation.copy()

    return arr.astype(int)


def get_signature(minhash, vector):
    # get index locations of every 1 value in vector
    idx = np.nonzero(vector)[0].tolist()
    # use index locations to pull only +ve positions in minhash
    shgles = minhash[:, idx]
    # find minimum value in each hash vector
    try:
        signature = np.min(shgles, axis=1)
    except Exception:
        signature = np.zeros(40)
    return signature


def read_sentences(path, column, delimiter='|'):
    print("Read Sentences...")
    sentences = []
    with open(path) as f:
        for line in f:
            sentence = line.split(delimiter)[column]
            sentences.append(sentence.strip())
    return sentences


def process_sentences(sentences, k, resolution):
    # Build shingles
    shingles = [build_shingles(sentence, k) for sentence in tqdm(sentences)]
    # Build vocab
    vocab = build_vocab_high_rate(shingles, 1)
    print(f'Vocab Size: {len(vocab)}')
    # One-hot encode our shingles
    arr = minhash_arr(vocab, resolution)
    signatures = [get_signature(arr, one_hot(shingle_set, vocab)) for shingle_set in tqdm(shingles)]
    return np.stack(signatures)


def kmeans_cluster(data, n_clus):
    # Kmeans cluster
    print('Kmeans Clustering...')
    kmeans = KMeans(n_clusters=n_clus, random_state=42)
    labels = kmeans.fit_predict(data)

    return labels


def tune_parameters_sequentially(sentences, k_values, resolution_values, comp_formats, path, n_clus=10):
    results = {
        'k': {fmt: [] for fmt in comp_formats},
        'resolution': {fmt: [] for fmt in comp_formats},
    }

    for k in k_values:
        temp_signatures = process_sentences(sentences, k, resolution_values[0])
        temp_labels = kmeans_cluster(temp_signatures, n_clus)
        for fmt in comp_formats:
            _, _, comp_ratio, _ = parallel_column_wise_compression(n_clus, path, temp_labels, fmt, delimiter='|')
            results['k'][fmt].append((k, comp_ratio))

    optimal_k, _ = max(results['k'][comp_formats[0]], key=lambda x: x[1])

    for resolution in resolution_values:
        temp_signatures = process_sentences(sentences, optimal_k, resolution)
        temp_labels = kmeans_cluster(temp_signatures, n_clus)
        for fmt in comp_formats:
            _, _, comp_ratio, _ = parallel_column_wise_compression(n_clus, path, temp_labels, fmt, delimiter='|')
            results['resolution'][fmt].append((resolution, comp_ratio))

    optimal_resolution, _ = max(results['resolution'][comp_formats[0]], key=lambda x: x[1])

    return results



def plot_parameter_trends_sequential(results, parameter_names, comp_formats, name):
    colors = ['blue', 'green', 'red']  # Example colors for gzip, lz4, zstd
    for param in parameter_names:
        plt.figure(figsize=(10, 6))
        for fmt, color in zip(comp_formats, colors):
            parameter_values, comp_ratios = zip(*results[param][fmt])
            plt.plot(parameter_values, comp_ratios, '-o', label=f'{fmt}', color=color)
        plt.xlabel(param.capitalize())
        plt.ylabel('Compression Ratio')
        plt.title(f'{name}: Compression Ratio vs. {param.capitalize()}')
        plt.grid(True)
        plt.legend()
        plt.show()


def write_results_to_file(results, name):
    filename = f"{name}_lsh_tuning_results.txt"
    with open(filename, 'w') as f:
        for param, methods in results.items():
            f.write(f"Parameter: {param}\n")
            for fmt, values in methods.items():
                f.write(f"Compression Method: {fmt}\n")
                for value, comp_ratio in values:
                    f.write(f"{value}: {comp_ratio}\n")
            f.write("\n")


def main():
    names = ['DS_001', 'orders', 'partsupp', 'DS_002', 'DS_003']
    columns = [7, 8, 4, 7, 7]
    for name, column in zip(names, columns):
        print(f'Dataset: {name}')
        path = f'../datasets/{name}.csv'
        sentences = read_sentences(path, column=column)

        k_values = [2, 3, 4, 5, 6]
        resolution_values = [5, 10, 15, 20, 25]
        comp_formats = ['gzip', 'lz4', 'zstd']

        # Adjusted to pass comp_formats instead of path and n_clus directly
        results = tune_parameters_sequentially(sentences, k_values, resolution_values, comp_formats, path)

        # Plotting
        plot_parameter_trends_sequential(results, ['k', 'resolution'], comp_formats, name)

        # Writing results to file
        write_results_to_file(results, name)


if __name__ == "__main__":
    main()
