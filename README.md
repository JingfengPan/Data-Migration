The logical process of the program:
1. Preprocess Data -> Data with only numeric and category (The first few columns are numerical data and the last few columns are categorical data)
2. Random split -> Random split labels
3. Incremental k-prototypes clustering -> Clustering labels
4. Classifiers + Clustering labels -> Classifier labels
5. Column-wise -> Column-wise data
6. Gzip, LZ4, Zstd Compression -> Compression ratios, throughputs, costs of Random split, Incremental k-prototypes clustering and Classifiers

All the original test data is stored in the "test_data" folder, all the data that generated during the process will be temporarily stored in the "data" folder, and will be deleted automatically after using.
All the results will be output as a .txt file in the "results" folder.
Just need to run the "main.py" since all functions are intergrated into it, and all the variable parameters are listed at the top of the main() function.
The "draw_figures.py" hasn't been fully implemented yet.

Please note that I am using njit from numba to speed up the "incremental_k_prototypes.py", which only support up to Python 3.9.
