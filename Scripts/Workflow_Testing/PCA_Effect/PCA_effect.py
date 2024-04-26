import datetime
import timeit
import shutil
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, jaccard_score, f1_score

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def create_directory_for_results(path, dir_name):
    """
    Create folder for results.

    Parameters:
    - path (str): The base path for creating the directory.
    - dir_name (str): The name of the directory to be created.

    Returns:
    - str: The path of the created directory.
    """

    path = f'{path}{dir_name}/'

    if os.path.isdir(path):
        shutil.rmtree(path)

    os.mkdir(path)

    return path


def get_data(Xy, indexes):
    """
    Get data based on the specified indexes.

    Parameters:
    - paths (list): List of file paths.
    - indexes (list): Indexes of the files to be concatenated.

    Returns:
    - pd.DataFrame: Concatenated dataframe.
    - pd.Series: Concatenated target series.
    """
    
    X = pd.DataFrame()
    
    for index in indexes:
        X = pd.concat([X, Xy[index]], axis=0)

    y = X['target']
    X = X.drop(['target'], axis=1)

    return X, y


def get_metrics_multiclass(y, y_hat):
    """
    Calculate multiclass metrics.

    Parameters:
    - y (pd.Series): True labels.
    - y_hat (pd.Series): Predicted labels.

    Returns:
    - np.array: Array containing accuracy, Jaccard index, and F1 score.
    """

    results = np.zeros(3)

    if 0 not in y_hat or 1 not in y_hat or 2 not in y_hat:
        results[0] = 0
        results[1] = 0
        results[2] = 0

    else:
        results[0] = accuracy_score(y, y_hat)
        results[1] = jaccard_score(y, y_hat, average='macro')
        results[2] = f1_score(y, y_hat, average='macro')

    return results


def cross_validation_LinearSVC(Xy_full, Xy_clustered, target_dim):
    """
    Perform cross-validation using Linear SVC with PCA dimensionality reduction.

    Parameters:
    - Xy_full (list): List of full datasets.
    - Xy_clustered (list): List of clustered datasets.
    - target_dim (int): The target dimension for PCA.

    Returns:
    - list: List of mean and standard deviation of metrics along with the elapsed time.
    """

    k_fold = KFold(n_splits=5, shuffle=False)

    metrics = list()

    for i, (train_index, test_index) in enumerate(k_fold.split(Xy_full)):

        X_train, y_train = get_data(Xy_clustered, train_index)

        # Fitting part
        # StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        # PCA
        pca = PCA(n_components=target_dim, svd_solver='full')
        pca.fit(X_train)
        X_train = pca.transform(X_train)

        # LinearSVC - fit
        linear_svc = LinearSVC(dual=False)
        linear_svc.fit(X_train, y_train)

        # Predicting part - for loop over X_test images
        for ii, index in enumerate(test_index):
            
            # Getting data X_test, y_test
            X_test = Xy_full[index]
            y_test = X_test['target']

            X_test = X_test.drop(['target'], axis=1)

            # StandardScaler
            X_test = scaler.transform(X_test)

            # PCA
            X_test = pca.transform(X_test)

            # LinearSVC - Predicting
            y_hat = linear_svc.predict(X_test)
            
            # Metrics
            metrics.append(get_metrics_multiclass(y_test, y_hat))
            
    means = np.mean(metrics,axis=0)
    stds = np.std(metrics,axis=0)

    return list(np.concatenate((means, stds)))


def cross_validation_SVC(Xy_full, Xy_clustered, target_dim):
    """
    Perform cross-validation using SVC with PCA dimensionality reduction.

    Parameters:
    - Xy_full (list): List of full datasets.
    - Xy_clustered (list): List of clustered datasets.
    - target_dim (int): The target dimension for PCA.

    Returns:
    - list: List of mean and standard deviation of metrics along with the elapsed time.
    """

    k_fold = KFold(n_splits=5, shuffle=False)

    metrics = list()

    for i, (train_index, test_index) in enumerate(k_fold.split(Xy_full)):

        X_train, y_train = get_data(Xy_clustered, train_index)

        # Fitting part
        # StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        # PCA
        pca = PCA(n_components=target_dim, svd_solver='full')
        pca.fit(X_train)
        X_train = pca.transform(X_train)

        # SVC - fit
        svc = SVC()
        svc.fit(X_train, y_train)

        # Predicting part - for loop over X_test images
        for ii, index in enumerate(test_index):
            
            # Getting data X_test, y_test
            X_test = Xy_full[index]
            y_test = X_test['target']

            X_test = X_test.drop(['target'], axis=1)

            # StandardScaler
            X_test = scaler.transform(X_test)

            # PCA
            X_test = pca.transform(X_test)

            # SVC - Predicting
            y_hat = svc.predict(X_test)
            
            # Metrics
            metrics.append(get_metrics_multiclass(y_test, y_hat))
            
    means = np.mean(metrics,axis=0)
    stds = np.std(metrics,axis=0)

    return list(np.concatenate((means, stds)))


def cross_validation_RF(Xy_full, Xy_clustered, target_dim):
    """
    Perform cross-validation using Random Forest with PCA dimensionality reduction.

    Parameters:
    - Xy_full (list): List of full datasets.
    - Xy_clustered (list): List of clustered datasets.
    - target_dim (int): The target dimension for PCA.

    Returns:
    - list: List of mean and standard deviation of metrics along with the elapsed time.
    """

    k_fold = KFold(n_splits=5, shuffle=False)

    metrics = list()

    for i, (train_index, test_index) in enumerate(k_fold.split(Xy_full)):

        X_train, y_train = get_data(Xy_clustered, train_index)

        # Fitting part
        # StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        # PCA
        pca = PCA(n_components=target_dim, svd_solver='full')
        pca.fit(X_train)
        X_train = pca.transform(X_train)

        # RF - fit
        rf = RandomForestClassifier(n_jobs=-1, random_state=0)
        rf.fit(X_train, y_train)

        # Predicting part - for loop over X_test images
        for ii, index in enumerate(test_index):
            
            # Getting data X_test, y_test
            X_test = Xy_full[index]
            y_test = X_test['target']

            X_test = X_test.drop(['target'], axis=1)

            # StandardScaler
            X_test = scaler.transform(X_test)

            # PCA
            X_test = pca.transform(X_test)

            # RF - Predicting
            y_hat = rf.predict(X_test)
            
            # Metrics
            metrics.append(get_metrics_multiclass(y_test, y_hat))
            
    means = np.mean(metrics,axis=0)
    stds = np.std(metrics,axis=0)

    return list(np.concatenate((means, stds)))


def model_testing(data_path_full, data_path_clustered, output_path, dir_name):
    """
    Test the models using cross-validation with PCA dimensionality reduction.

    Parameters:
    - data_path_full (str): Path to the directory containing full datasets.
    - data_path_clustered (str): Path to the directory containing clustered datasets.
    - output_path (str): Path to save the results.
    - dir_name (str): Name of the directory for results.

    Returns:
    - None
    """

    output_path = create_directory_for_results(output_path,dir_name)

    CSV_names = np.asarray(os.listdir(data_path_full))
    CSV_names_clustered = np.asarray(os.listdir(data_path_clustered))

    Xy_full = list()
    Xy_clustered = list()

    # Loading data
    for i, current_csv_name in enumerate(zip(CSV_names, CSV_names_clustered)):

        Xy_full.append(pd.read_csv(f'{data_path_full}{current_csv_name[0]}'))
        Xy_clustered.append(pd.read_csv(f'{data_path_clustered}{current_csv_name[1]}'))

    # Variables for storing metrics & elapsed times
    dict_LinearSVC = dict()
    dict_SVC = dict()
    dict_RF = dict()

    # Testing PCA dimensions from 2 to 11
    dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for i, dim in enumerate(dimensions):
        print(f'In progress {i+1} of {len(dimensions)} \t {dim}')

        # LinearSVC
        t0 = timeit.default_timer()
        dict_LinearSVC[dim] = cross_validation_LinearSVC(Xy_full, Xy_clustered, dim)
        t1 = timeit.default_timer()
        dict_LinearSVC[dim].append(t1-t0)

        # SVC
        t0 = timeit.default_timer()
        dict_SVC[dim] = cross_validation_SVC(Xy_full, Xy_clustered, dim)
        t1 = timeit.default_timer()
        dict_SVC[dim].append(t1-t0)

        # RF
        t0 = timeit.default_timer()
        dict_RF[dim] = cross_validation_RF(Xy_full, Xy_clustered, dim)
        t1 = timeit.default_timer()
        dict_RF[dim].append(t1-t0)

    # Saving time & metrics as DataFrame
    df_LinearSVC = pd.DataFrame.from_dict(dict_LinearSVC, orient='index').reset_index()
    df_LinearSVC.columns = ['PCA_target_dimension', 'Accuracy_mean', 'Jaccard_index_mean', 'Dice_coefficient_mean', 'Accuracy_std', 'Jaccard_index_std', 'Dice_coefficient_std', 'Time']
    df_LinearSVC.to_csv(f'{output_path}LinearSVC.csv', index=False)

    df_SVC = pd.DataFrame.from_dict(dict_SVC, orient='index').reset_index()
    df_SVC.columns = ['PCA_target_dimension', 'Accuracy_mean', 'Jaccard_index_mean', 'Dice_coefficient_mean', 'Accuracy_std', 'Jaccard_index_std', 'Dice_coefficient_std', 'Time']
    df_SVC.to_csv(f'{output_path}SVC.csv', index=False)

    df_RF = pd.DataFrame.from_dict(dict_RF, orient='index').reset_index()
    df_RF.columns = ['PCA_target_dimension', 'Accuracy_mean', 'Jaccard_index_mean', 'Dice_coefficient_mean', 'Accuracy_std', 'Jaccard_index_std', 'Dice_coefficient_std', 'Time']
    df_RF.to_csv(f'{output_path}RF.csv', index=False)

    return None


def source_model_testing():
    """
    Main function for testing models with PCA dimensionality reduction.

    Returns:
    - None
    """

    n_clusters = '1584'
    n_features = '120'

    folders = ['PoC']

    for i, curr_suffix in enumerate(folders):

        t0 = datetime.datetime.now()

        DATA_PATH_FULL = f'./Results/Selecting_Features/FoI_S_Xy_{curr_suffix}_{n_clusters}/{n_features}/'
        DATA_PATH_CLUSTERED = f'./Results/Selecting_Features/FoI_C_S_Xy_{curr_suffix}_{n_clusters}/{n_features}/'
        OUTPUT_PATH = './Results/Workflow_Testing/'
        DIR_NAME = f'PCA_effect_exploring_{curr_suffix}_C_{n_clusters}_S_{n_features}'

        model_testing(DATA_PATH_FULL, DATA_PATH_CLUSTERED, OUTPUT_PATH, DIR_NAME)

        t1 = datetime.datetime.now()

        print(f'PCA efect testing: {t1 - t0}')


if __name__ == '__main__':

    print('Hello, home!')

    source_model_testing()