import cudf
import cupy as cp
from cuml import KMeans

import numpy as np


import datetime
import timeit
import shutil
import os


def cluster_data(X, n_clusters):
    """
    Reduce data size using K-means clustering.

    Parameters:
    - X (DataFrame): Features.
    - n_clusters (int): Number of clusters.

    Returns:
    - DataFrame: Reduced features.
    """

    col_names = X.columns[:-1]

    X_reduced = cudf.DataFrame()
    y_reduced = X.target.unique().values

    y = X.target
    X = X.drop(['target'], axis=1)

    for index,label in enumerate(y_reduced):

        model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1).fit(X[y==label])

        X_reduced = cudf.concat([X_reduced, model.cluster_centers_], axis=0)

    X_reduced.columns = col_names
    y_reduced = cp.repeat(y_reduced, n_clusters)

    X_reduced['target'] = y_reduced

    return X_reduced


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


def feature_clustering(data_path, output_path, dir_name):
    """
    Feature clustering using K-means clustering method.

    Parameters:
    - data_path (str): Path where CSV files are stored.
    - output_path (str): Path where the results will be saved.
    - dir_name (str): Name of the directory to store results.

    Returns:
    - None
    """

    exponents = np.linspace(0.2, 3.2, 16)
    n_clusters_list = [str(int(10**exponent)) for exponent in exponents]

    list_of_names_data = os.listdir(data_path)

    elapsed_times = cp.zeros((len(n_clusters_list), len(list_of_names_data)))

    output_path_times = output_path

    output_path = create_directories_for_results(output_path, dir_name, n_clusters_list)

    N = len(list_of_names_data)

    for i, csv_name in enumerate(list_of_names_data):

        Xy = cudf.read_csv(f'{data_path}{csv_name}')

        for ii, n_clusters in enumerate(n_clusters_list):

            print(f'{csv_name}\t:\t{n_clusters}')

            start_time = timeit.default_timer()

            Xy_clustered = cluster_data(Xy, n_clusters=int(n_clusters))

            end_time = timeit.default_timer()

            elapsed_times[ii,i] = end_time - start_time

            Xy_clustered.to_csv(f'{output_path}{n_clusters}/{csv_name}', index=False)

    elapsed_means = cp.mean(elapsed_times, axis=1)
    elapsed_stds = cp.std(elapsed_times, axis=1)

    df_times = cudf.DataFrame({'n_clusters': n_clusters_list, 'mean': elapsed_means, 'std': elapsed_stds})
    df_times.to_csv(f'{output_path_times}{dir_name}_times.csv', index=False)


def source_feature_clustering():
    """
    Main function for reducing feature data.

    Returns:
    - None
    """

    folders = ['PoC', 'ALL']
    folders = ['ALL']

    for i, curr_suffix in enumerate(folders):
        t0 = datetime.datetime.now()

        DATA_PATH = f'./Results/Generating_Features/FoI_Xy_{curr_suffix}/'
        OUTPUT_PATH = f'./Results/Clustering_Features/'
        DIR_NAME = f'FoI_C_Xy_GPU_{curr_suffix}'

        feature_clustering(DATA_PATH, OUTPUT_PATH, DIR_NAME)

        t1 = datetime.datetime.now()

        print(f'Data clustering : {t1 - t0}')


if __name__ == '__main__':

    print('Hello, home!')

    source_feature_clustering()