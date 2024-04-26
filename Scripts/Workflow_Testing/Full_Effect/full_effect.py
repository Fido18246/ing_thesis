import datetime
import timeit
import shutil
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

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


def get_data(paths, indexes):
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
        X = pd.concat([X, pd.read_csv(f'{paths[index]}')], axis=0)

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


def cross_validation_LinearSVC(paths):
    """
    Perform cross-validation using Linear SVC.

    Parameters:
    - paths (list): List of file paths.

    Returns:
    - list: List of mean and standard deviation of metrics along with the elapsed time.
    """

    k_fold = KFold(n_splits=5, shuffle=False)

    metrics = list()

    for i, (train_index, test_index) in enumerate(k_fold.split(paths)):
        print(f'In progress {i+1} of 5 folds \t {datetime.datetime.now()}')

        X_train, y_train = get_data(paths, train_index)

        # Fitting part
        # StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        # LinearSVC - fit
        linear_svc = LinearSVC(dual=False)
        linear_svc.fit(X_train, y_train)

        # Predicting part - for loop over X_test images
        for ii, index in enumerate(test_index):
            
            # Getting data X_test, y_test
            X_test = pd.read_csv(f'{paths[index]}')
            y_test = X_test['target']

            X_test = X_test.drop(['target'], axis=1)

            # StandardScaler
            X_test = scaler.transform(X_test)

            # LinearSVC - Predicting
            y_hat = linear_svc.predict(X_test)
            
            # Metrics
            metrics.append(get_metrics_multiclass(y_test, y_hat))
            
    means = np.mean(metrics,axis=0)
    stds = np.std(metrics,axis=0)

    return list(np.concatenate((means, stds)))


def cross_validation_SVC(paths):
    """
    Perform cross-validation using SVC.

    Parameters:
    - paths (list): List of file paths.

    Returns:
    - list: List of mean and standard deviation of metrics along with the elapsed time.
    """

    k_fold = KFold(n_splits=5, shuffle=False)

    metrics = list()

    for i, (train_index, test_index) in enumerate(k_fold.split(paths)):        
        print(f'In progress {i+1} of 5 folds \t {datetime.datetime.now()}')

        X_train, y_train = get_data(paths, train_index)

        # Fitting part
        # StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        # SVC - fit
        svc = SVC()
        svc.fit(X_train, y_train)

        # Predicting part - for loop over X_test images
        for ii, index in enumerate(test_index):
            
            # Getting data X_test, y_test
            X_test = pd.read_csv(f'{paths[index]}')
            y_test = X_test['target']

            X_test = X_test.drop(['target'], axis=1)

            # StandardScaler
            X_test = scaler.transform(X_test)

            # SVC - Predicting
            y_hat = svc.predict(X_test)
            
            # Metrics
            metrics.append(get_metrics_multiclass(y_test, y_hat))
            
    means = np.mean(metrics,axis=0)
    stds = np.std(metrics,axis=0)

    return list(np.concatenate((means, stds)))


def cross_validation_RF(paths):
    """
    Perform cross-validation using Random Forest.

    Parameters:
    - paths (list): List of file paths.

    Returns:
    - list: List of mean and standard deviation of metrics along with the elapsed time.
    """

    k_fold = KFold(n_splits=5, shuffle=False)

    metrics = list()

    for i, (train_index, test_index) in enumerate(k_fold.split(paths)):        
        print(f'In progress {i+1} of 5 folds \t {datetime.datetime.now()}')
        
        X_train, y_train = get_data(paths, train_index)

        # Fitting part
        # StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        # RF - fit
        rf = RandomForestClassifier(n_jobs=-1, random_state=0)
        rf.fit(X_train, y_train)

        # Predicting part - for loop over X_test images
        for ii, index in enumerate(test_index):
            
            # Getting data X_test, y_test
            X_test = pd.read_csv(f'{paths[index]}')
            y_test = X_test['target']

            X_test = X_test.drop(['target'], axis=1)

            # StandardScaler
            X_test = scaler.transform(X_test)

            # RF - Predicting
            y_hat = rf.predict(X_test)
            
            # Metrics
            metrics.append(get_metrics_multiclass(y_test, y_hat))
            
    means = np.mean(metrics,axis=0)
    stds = np.std(metrics,axis=0)

    return list(np.concatenate((means, stds)))


def model_testing(data_path, output_path, dir_name):
    """
    Test the models using cross-validation.

    Parameters:
    - data_path (str): Path to the directory containing datasets.
    - output_path (str): Path to save the results.
    - dir_name (str): Name of the directory for results.

    Returns:
    - None
    """

    output_path = create_directory_for_results(output_path, dir_name)

    CSV_names = np.asarray(os.listdir(data_path))

    paths = list()
    for i, name in enumerate(CSV_names):
        paths.append(f'{data_path}{name}')

    # Variables for storing metrics & elapsed times
    result_dict = dict()

    # LinearSVC
    print('LinearSVC')
    t0 = timeit.default_timer()
    result_dict['LinearSVC'] = cross_validation_LinearSVC(paths)
    t1 = timeit.default_timer()
    result_dict['LinearSVC'].append(t1-t0)

    # RF
    print('RF')
    t0 = timeit.default_timer()
    result_dict['RF'] = cross_validation_RF(paths)
    t1 = timeit.default_timer()
    result_dict['RF'].append(t1-t0)

    # SVC
    print('SVC')
    t0 = timeit.default_timer()
    result_dict['SVC'] = cross_validation_SVC(paths)
    t1 = timeit.default_timer()
    result_dict['SVC'].append(t1-t0)

    # Saving results
    result_df = pd.DataFrame.from_dict(result_dict, orient='index').reset_index()
    result_df.columns = ['Type', 'Accuracy_mean', 'Jaccard_index_mean', 'Dice_coefficient_mean', 'Accuracy_std', 'Jaccard_index_std', 'Dice_coefficient_std', 'Time']
    result_df.to_csv(f'{output_path}results.csv', index=False)

    return None


def source_model_testing():
    """
    Main function for testing models.

    Returns:
    - None
    """

    folders = ['PoC']

    for i, curr_suffix in enumerate(folders):

        t0 = datetime.datetime.now()

        DATA_PATH = f'./Results/Generating_Features/FoI_Xy_{curr_suffix}/'
        OUTPUT_PATH = './Results/Workflow_Testing/'
        DIR_NAME = f'Full_effect_exploring_{curr_suffix}'

        model_testing(DATA_PATH, OUTPUT_PATH, DIR_NAME)

        t1 = datetime.datetime.now()

        print(f'Full effect exploring: {t1 - t0}')


if __name__ == '__main__':

    print('Hello, home!')

    source_model_testing()