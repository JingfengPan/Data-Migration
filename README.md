Classifiers are in the classifiers folder.
Online clustering is in the kprototypes folder.
All the results are using column-wise compression.
For the k-prototypes, just run the "main.py" file.
For the classifiers: 
1. Run the "k_prototypes.py" file to label the dataset.
2. Run the "classifiers.py" file to train and test the classifiers, as well as record the results.
3. Run the "main.py" file to compress the dataset and record the results.
4. Run the "calculate_throughput.py" file and "cost_model.py" file to calculate the throughput and cost seperately. And then draw the figures.

Please note that I am using njit from numba to speed up the "incremental_k_prototypes.py", which only support up to Python 3.9.
