import os
import shutil
import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

# Ray
from ray import tune, train
from ray.tune.search.basic_variant import BasicVariantGenerator


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


def get_data(X_list, y_list, indexes):
    """
    Concatenate X and y dataframes based on given indexes.

    Parameters:
    - X_list (list): List of X dataframes.
    - y_list (list): List of y dataframes.
    - indexes (list): List of indexes specifying which dataframes to concatenate.

    Returns:
    - pd.DataFrame: Concatenated X dataframe.
    - pd.DataFrame: Concatenated y dataframe.
    """
    
    X = pd.DataFrame()
    y = pd.DataFrame()
    
    for index in indexes:
        X = pd.concat([X, X_list[index]], axis=0)
        y = pd.concat([y, y_list[index]], axis=0)

    return X, y


def cross_validation_LinearSVC(model_params, Xy_full_paths, X_clustered, y_clustered):
    """
    Perform cross-validation for LinearSVC.

    Parameters:
    - model_params (dict): Dictionary containing model hyperparameters.
    - Xy_full_paths (list): List of file paths containing full feature data.
    - X_clustered (list): List of DataFrames containing clustered feature data.
    - y_clustered (list): List of target labels.

    Returns:
    - dict: Dictionary containing the mean F1 score.
    """

    k_fold = KFold(n_splits=5, shuffle=False)

    score = list()

    # For loop over folds
    for i, (train_index, test_index) in enumerate(k_fold.split(Xy_full_paths)):
        X_train, y_train = get_data(X_clustered, y_clustered, train_index)

        # Fitting part
        # Scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        # PCA
        reducer = PCA(n_components=11)
        reducer.fit(X_train)
        X_train = reducer.transform(X_train)

        # LinearSVC - fit
        linear_svc = LinearSVC(dual=False, max_iter=1_000,
                               C=model_params['C'],
                               penalty=model_params['penalty'],
                               loss=model_params['loss'],
                               multi_class=model_params['multi_class'])
        
        linear_svc.fit(X_train, y_train.values.ravel())

        # Predicting part - for loop over X_test images
        for ii, index in enumerate(test_index):

            # Loading X_test & y_test
            X_test = pd.read_csv(f'{Xy_full_paths[index]}')

            y_test = X_test['target']

            X_test = X_test.drop(['target'], axis=1)

            # StandardScaler
            X_test = scaler.transform(X_test)

            # PCA
            X_test = reducer.transform(X_test)

            # LinearSVC - Predicting
            y_hat = linear_svc.predict(X_test)
            
            # Metrics
            score.append(f1_score(y_test, y_hat, average='macro'))

    return  {"Dice_coefficient": np.mean(score)}


def model_tuning(data_path_full, data_path_clustered, output_path, dir_name):
    """
    Tune hyperparameters for LinearSVC using Ray Tune.

    Parameters:
    - data_path_full (str): Path to full feature data.
    - data_path_clustered (str): Path to clustered feature data.
    - output_path (str): Path to store the results.
    - dir_name (str): Name of the directory for the results.

    Returns:
    - None
    """

    output_path = create_directory_for_results(output_path, dir_name)

    CSV_names_full = os.listdir(data_path_full)
    CSV_names_clustered = os.listdir(data_path_clustered)

    Xy_full_paths = list()

    X_clustered = list()
    y_clustered = list()

    # Loading full & clustered data
    for i, current_csv_name in enumerate(CSV_names_full):
        # full
        Xy_full_paths.append(os.path.abspath(f'{data_path_full}{current_csv_name}'))
        
        # clustered
        X_clustered.append(pd.read_csv(f'{data_path_clustered}{current_csv_name}'))
        y_clustered.append(X_clustered[-1]['target'])
        X_clustered[-1] = X_clustered[-1].drop(['target'], axis=1)

    # Setting trainable
    trainable = tune.with_parameters(cross_validation_LinearSVC,
                                        Xy_full_paths=Xy_full_paths,
                                        X_clustered=X_clustered, 
                                        y_clustered=y_clustered)
    
    trainable = tune.with_resources(trainable, {"cpu": 2})
    
    # Setting Search space of hyperparameters
    search_space = {
        'C': tune.grid_search(np.power(2, np.linspace(-10, 10, num=21))),
        'penalty': tune.grid_search(['l1', 'l2']),
        'loss': tune.grid_search(['hinge', 'squared_hinge']),
        'multi_class': tune.grid_search(['ovr', 'crammer_singer'])
    }

    # Setting algorithm & config
    algorithm = BasicVariantGenerator()

    tune_config = tune.TuneConfig(search_alg=algorithm, metric='Dice_coefficient', mode='max')
    run_config = train.RunConfig(log_to_file=False)

    # Tuner fit
    tuner = tune.Tuner(trainable, tune_config=tune_config, run_config=run_config, param_space=search_space)
    df_results = tuner.fit().get_dataframe()

    # Saving results
    df_results = df_results.sort_values(by=['Dice_coefficient'], ascending=False)
    df_results.to_csv(f'{output_path}Ray_Tune_results.csv', index=False)


def source_model_tuning():
    """
    Main function for tuning hyperparameters of LinearSVC.

    Returns:
    - None
    """

    t0 = datetime.datetime.now()

    DATA_PATH_FULL = f'./Results/Selecting_Features/FoI_S_Xy_ALL_1584/120/'
    DATA_PATH_CLUSTERED = f'./Results/Selecting_Features/FoI_C_S_Xy_ALL_1584/120/'
    OUTPUT_PATH = f'./Results/Hyperparameters_Tuning/'
    DIR_NAME = f'CPU_LinearSVC_Results'

    model_tuning(DATA_PATH_FULL, DATA_PATH_CLUSTERED, OUTPUT_PATH, DIR_NAME)

    t1 = datetime.datetime.now()

    print(f'Model tuning: {t1 - t0}')


if __name__ == '__main__':

    print('Hello, home!')

    source_model_tuning()