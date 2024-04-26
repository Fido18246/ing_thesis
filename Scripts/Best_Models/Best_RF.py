import os
import shutil
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
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


def cm_save(y_test, y_hat, output_path, output_name):
    """
    Save confusion matrix as an image.

    Parameters:
    - y_test (array-like): True labels.
    - y_hat (array-like): Predicted labels.
    - output_path (str): Path to save the image.
    - output_name (str): Name of the output file.

    Returns:
    - None
    """

    cm = confusion_matrix(y_test, y_hat)

    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', vmax=y_test.shape[0], vmin=0,
                xticklabels=['Pozadí', 'Cytoplazma', 'Jádro'], yticklabels=['Pozadí', 'Cytoplazma', 'Jádro'])
    
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(f'{output_path}{output_name}_cm.png')
    plt.close()


def cross_validation_RF(Xy_full_paths, X_clustered, y_clustered, output_path):
    """
    Perform cross-validation with Random Forest (RF) classifier.

    Parameters:
    - Xy_full_paths (list): List of paths to CSV files containing full data.
    - X_clustered (list): List of X dataframes.
    - y_clustered (list): List of y dataframes.
    - output_path (str): Path to save the results.

    Returns:
    - None
    """

    k_fold = KFold(n_splits=5, shuffle=False)

    score = list()
    names = list()

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

        # RF - fit
        rf = RandomForestClassifier(n_estimators=158, max_features='sqrt', bootstrap=True, random_state=0, n_jobs=-1)
        
        rf.fit(X_train, y_train.values.ravel())

        # Predicting part - for loop over X_test images
        for ii, index in enumerate(test_index):

            # Loading X_test & y_test
            X_test = pd.read_csv(f'{Xy_full_paths[index]}')

            y_test = X_test['target'].values

            X_test = X_test.drop(['target'], axis=1)

            # StandardScaler
            X_test = scaler.transform(X_test)

            # PCA
            X_test = reducer.transform(X_test)

            # RF - Predicting
            y_hat = rf.predict(X_test)
            
            # Metrics
            score.append(f1_score(y_test, y_hat, average='macro'))

            # Output_name
            output_name = Xy_full_paths[index].split('\\')[-1][:-4]

            # Confusion matrix
            cm_save(y_test, y_hat, output_path, output_name)

            # Reshape to img 500x500
            y_hat = y_hat.reshape(500, 500)
            y_test = y_test.reshape(500, 500)

            # Saving images
            plt.imsave(f'{output_path}{output_name}_prediction.png', y_hat)
            plt.imsave(f'{output_path}{output_name}_target.png', y_test)

            names.append(output_name)

    data = {'names': names, 'Dice_coefficient': score}
    df = pd.DataFrame(data)
    df.to_csv(f'{output_path}results.csv', index=False)

    return None


def best_model(data_path_full, data_path_clustered, output_path, dir_name):
    """
    Perform cross-validation with Random Forest (RF) classifier and save the results.

    Parameters:
    - data_path_full (str): Path to CSV files containing full data.
    - data_path_clustered (str): Path to CSV files containing clustered data.
    - output_path (str): Path to save the results.
    - dir_name (str): Name of the directory for results.

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

    cross_validation_RF(Xy_full_paths, X_clustered, y_clustered, output_path)


def source_best_model():
    """
    Main function.

    Returns:
    - None
    """

    t0 = datetime.datetime.now()

    DATA_PATH_FULL = f'./Results/Selecting_Features/FoI_S_Xy_ALL_1584/110/'
    DATA_PATH_CLUSTERED = f'./Results/Selecting_Features/FoI_C_S_Xy_ALL_1584/110/'
    OUTPUT_PATH = f'./Results/Best_Models/'
    DIR_NAME = f'RF_Results'

    best_model(DATA_PATH_FULL, DATA_PATH_CLUSTERED, OUTPUT_PATH, DIR_NAME)

    t1 = datetime.datetime.now()

    print(f'Best model: {t1 - t0}')


if __name__ == '__main__':

    print('Hello, home!')

    source_best_model()