import os
import cv2
import shutil
import datetime


import numpy as np
import pandas as pd


import feature_worker as fw


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


def feature_generating(data_path, label_path, output_path, dir_name):
    """
    Load image, extract features, and save CSV.

    Parameters:
    - data_path (str): Path where image files are located.
    - label_path (str): Path where TXT files with labels are stored.
    - output_path (str): Path to store the results.
    - dir_name (str): Name of the directory for the results.

    Returns:
    - None
    """

    list_of_names_img = os.listdir(data_path)
    list_of_names_labels = os.listdir(label_path)

    output_path = create_directory_for_results(output_path, dir_name)

    N = len(list_of_names_img)

    for index, (img_name,lbl_name) in enumerate(zip(list_of_names_img,list_of_names_labels)):

        print(f'In progress {index + 1} / {N} : {img_name} | {lbl_name}')

        img = cv2.imread(f'{data_path}{img_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        X = fw.get_features(img)
        y = pd.DataFrame({'target': np.loadtxt(f'{label_path}{lbl_name}', dtype=int).reshape(-1)})

        X = pd.concat([X, y], axis=1)

        df_name = img_name.split('.')[0]
        X.to_csv(f'{output_path}{df_name}.csv', index=False)
        # X.to_pickle(f'{output_path}{df_name}.pkl', compression=None)


def source_feature_generating():
    """
    Main function for feature generating.

    Returns:
    - None
    """

    folders = ['PoC', 'ALL']

    for i, curr_suffix in enumerate(folders):

        t0 = datetime.datetime.now()

        DATA_PATH = f'./Results/Resizing_Images/Images_{curr_suffix}/'
        LABEL_PATH = f'./Results/Resizing_Images/Labels_{curr_suffix}/'
        OUTPUT_PATH = f'./Results/Generating_Features/'
        DIR_NAME = f'FoI_Xy_{curr_suffix}'

        feature_generating(DATA_PATH, LABEL_PATH, OUTPUT_PATH, DIR_NAME)

        t1 = datetime.datetime.now()

        print(f'Feature generating : {t1 - t0}')


if __name__ == '__main__':

    print('Hello, home!')

    source_feature_generating()