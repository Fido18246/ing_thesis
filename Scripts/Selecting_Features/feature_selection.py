import pandas as pd
import numpy as np

import datetime
import shutil
import os

from kydavra import FisherSelector


def create_directories_for_results(path, dir_name, sub_dirs_1):
    """
    Create folders for results.

    Parameters:
    - path (str): Base path.
    - dir_name (str): Directory name.
    - sub_dirs_1 (list): List of subdirectories under the first level.

    Returns:
    - str: Path to the created directory.
    """

    path = f'{path}{dir_name}/'

    if os.path.isdir(path):
        shutil.rmtree(path)

    os.mkdir(path)

    for index1, sub_name_1 in enumerate(sub_dirs_1):

        sub_path_1 = f'{path}{sub_name_1}/'
        os.mkdir(sub_path_1)

    return path


def feature_selecting(data_path, data_path_clustered, output_path, dir_name, dir_name_clustered):
    """
    Select features using FisherSelector and save the selected features.

    Parameters:
    - data_path (str): Path to the directory containing original data.
    - data_path_clustered (str): Path to the directory containing clustered data.
    - output_path (str): Path to save the selected features.
    - dir_name (str): Name of the directory for selected features.
    - dir_name_clustered (str): Name of the directory for selected clustered features.

    Returns:
    - None
    """
    
    n_features_list = [str(int(value)) for value in np.linspace(10, 120, 12)]

    list_of_names_data = os.listdir(data_path)

    output_path_clustered = create_directories_for_results(output_path, dir_name_clustered, n_features_list)
    output_path = create_directories_for_results(output_path, dir_name, n_features_list)

    N = len(n_features_list)

    Xy = pd.DataFrame()
    for i, csv_name in enumerate(list_of_names_data):
        Xy = pd.concat([Xy, pd.read_csv(f'{data_path_clustered}{csv_name}')], ignore_index=True)

    for i, k_best in enumerate(n_features_list):

        print(f'In progress: {i+1} of {N} \t | {k_best}')

        feature_selector = FisherSelector(int(k_best))
        selected_columns = feature_selector.select(Xy, 'target')

        selected_columns.append('target')

        for ii, csv_name in enumerate(list_of_names_data):

            # Full data
            Xy = pd.read_csv(f'{data_path}{csv_name}')
            Xy_selected = Xy[selected_columns]
            Xy_selected.to_csv(f'{output_path}{k_best}/{csv_name}', index=False)

            # Clustered data
            Xy = pd.read_csv(f'{data_path_clustered}{csv_name}')
            Xy_selected = Xy[selected_columns]
            Xy_selected.to_csv(f'{output_path_clustered}{k_best}/{csv_name}', index=False)


def source_feature_selecting():
    """
    Perform feature selection based on predefined paths and parameters.

    Parameters:
    - None

    Returns:
    - None
    """

    n_clusters = '1584'

    folders = ['PoC']

    for i, curr_suffix in enumerate(folders):

        t0 = datetime.datetime.now()

        DATA_PATH = f'./Results/Generating_Features/FoI_Xy_{curr_suffix}/'
        DATA_PATH_CLUSTERED = f'./Results/Clustering_Features/FoI_C_Xy_GPU_{curr_suffix}/{n_clusters}/'
        OUTPUT_PATH = f'./Results/Selecting_Features/'
        DIR_NAME = f'FoI_S_Xy_{curr_suffix}_{n_clusters}'
        DIR_NAME_CLUSTERED = f'FoI_C_S_Xy_{curr_suffix}_{n_clusters}'

        feature_selecting(DATA_PATH, DATA_PATH_CLUSTERED, OUTPUT_PATH, DIR_NAME, DIR_NAME_CLUSTERED)

        t1 = datetime.datetime.now()

        print(f'Data clustering : {t1 - t0}')


if __name__ == '__main__':

    print('Hello, home!')

    source_feature_selecting()