import os
import pandas as pd


def generate_test_set(name, case, percentage):
    # File paths
    raw_file_path = f'./datasets/{name}.csv'
    indices_file_path = f'./data/{name}_10/{name}_{case}_{percentage}_indices.csv'
    preprocessed_file_path = f'./data/{name}_10/{name}_{case}_{percentage}.csv'

    # Read the raw file, each line as a single string
    with open(raw_file_path, 'r', encoding='utf-8') as file:
        raw_lines = file.readlines()

    # Convert list of lines (strings) into a DataFrame
    raw_df = pd.DataFrame(raw_lines, columns=['RowString'])

    indices_df = pd.read_csv(indices_file_path, header=None, names=['OriginalIndex'])
    preprocessed_df = pd.read_csv(preprocessed_file_path, header=None)

    # Add the original indices to the preprocessed dataframe
    preprocessed_df['OriginalIndex'] = indices_df

    # Sort the preprocessed dataframe based on the original indices
    sorted_preprocessed_df = preprocessed_df.sort_values(by='OriginalIndex')

    # Drop the 'OriginalIndex' column to get back to the original preprocessed format
    final_sorted_preprocessed_df = sorted_preprocessed_df.drop(columns=['OriginalIndex'])
    if not os.path.exists(f'./test/{name}_10'):
        os.makedirs(f'./test/{name}_10')
    # Save the sorted preprocessed file
    test_output_path = f'./test/{name}_10/{name}_{case}_{percentage}_test.csv'
    if os.path.exists(test_output_path):
        os.remove(test_output_path)
    final_sorted_preprocessed_df.to_csv(test_output_path, index=False, header=False)

    # Sort indices_df based on 'OriginalIndex' to match the order in final_sorted_preprocessed_df
    sorted_indices_df = indices_df.sort_values(by='OriginalIndex')

    # Extract the sorted index values as a flat array
    sorted_index_values = sorted_indices_df['OriginalIndex'].to_numpy()

    # Use the sorted index values to select rows from raw_df
    selected_rows = raw_df.iloc[sorted_index_values]

    # Save the selected rows to a new CSV file
    raw_output_path = f'./test/{name}_10/{name}_{case}_{percentage}_raw.csv'
    if os.path.exists(raw_output_path):
        os.remove(raw_output_path)
    selected_rows.to_csv(raw_output_path, index=False, header=False)
