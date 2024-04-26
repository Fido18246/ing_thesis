import datetime
import shutil
import cv2
import os


import numpy as np
import matplotlib.pyplot as plt


RESIZE_RESOLUTION = (500, 500)


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


def image_resizer(data_path, labels_path, output_path, dir_name_data, dir_name_labels, dir_name_labels_images):
    """
    Resize images and labels to a specified resolution and save them.

    Parameters:
    - data_path (str): Path to the directory containing original images.
    - labels_path (str): Path to the directory containing labeled images.
    - output_path (str): Path to the directory where resized images and labels will be saved.
    - dir_name_data (str): Name of the directory for resized images.
    - dir_name_labels (str): Name of the directory for resized labels.

    Returns:
    - None
    """

    list_of_names_data = os.listdir(data_path)
    list_of_names_labels = os.listdir(labels_path)

    output_path_data = create_directory_for_results(output_path, dir_name_data)
    output_path_labels = create_directory_for_results(output_path, dir_name_labels)
    output_path_labels_images = create_directory_for_results(output_path, dir_name_labels_images)

    N = len(list_of_names_data)

    for i, (img_name, lbl_name) in enumerate(zip(list_of_names_data, list_of_names_labels)):

        img = cv2.imread(f'{data_path}{img_name}')
        img = cv2.resize(img, RESIZE_RESOLUTION, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(f'{output_path_data}{img_name}', img)

        
        labeled_img = np.loadtxt(f'{labels_path}{lbl_name}', dtype=int)
        labeled_img = cv2.resize(labeled_img, RESIZE_RESOLUTION, interpolation=cv2.INTER_NEAREST)
        np.savetxt(f'{output_path_labels}{lbl_name}', labeled_img, fmt='%i')

        img_lbl_name = img_name.split('.')[0]
        plt.imsave(f'{output_path_labels_images}{img_lbl_name}.png', labeled_img, cmap='viridis')


def source_image_resizer():
    """
    Perform image resizing based on predefined paths and parameters.

    Parameters: 
    - None

    Returns: 
    - None
    """

    folders = ['PoC', 'ALL']


    for i, curr_suffix in enumerate(folders):

        t0 = datetime.datetime.now()

        DATA_PATH = f'./Data/Images_{curr_suffix}/'
        LABEL_PATH = f'./Data/Labels_{curr_suffix}/'
        OUTPUT_PATH = f'./Results/Resizing_Images/'
        DIR_NAME_DATA = f'Images_{curr_suffix}'
        DIR_NAME_LABELS = f'Labels_{curr_suffix}'
        DIR_NAME_LABELS_IMAGES = f'Labels_Images_{curr_suffix}'

        image_resizer(DATA_PATH, LABEL_PATH, OUTPUT_PATH, DIR_NAME_DATA, DIR_NAME_LABELS, DIR_NAME_LABELS_IMAGES)

        t1 = datetime.datetime.now()

        print(f'Data resizing : {t1 - t0}')


if __name__ == '__main__':

    print('Hello, home!')

    source_image_resizer()