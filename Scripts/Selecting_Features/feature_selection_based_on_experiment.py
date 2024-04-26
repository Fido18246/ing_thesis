import pandas as pd

import datetime
import shutil
import os


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
    Select and save features from the given data.

    Parameters:
    - data_path (str): Path to the directory containing original data.
    - data_path_clustered (str): Path to the directory containing clustered data.
    - output_path (str): Path to save the selected features.
    - dir_name (str): Name of the directory for selected features.
    - dir_name_clustered (str): Name of the directory for selected clustered features.

    Returns:
    - None
    """
    
    list_of_names_data = os.listdir(data_path)

    features_dict = {
        110: ['laplace','butterworth_R','butterworth_G','IE_FHSABP_G','IE_FHSABP_R','IE_FHSABP_B','butterworth_B','IE_GHE_R','IE_GHE_G','HSV_equalized_R','HSV_equalized_G','Lab_equalized_R','IE_GHE_B','HSV_equalized_B','IE_DSIHE_R','Lab_equalized_G','IE_DSIHE_G','Lab_A','YCrCb_Cr','IE_DSIHE_B','YUV_V','IE_BPHEME_B','dilation_B','median_B','avg_B','Total_variation_B','PIL_emboss_B','gauss_B','YUV_U','PIL_contour_B','BM3D_B','bilateral_B','RGB_B','GW_wb_B','retinex_with_adjust_B','retinex_B','standard_deviation_weighted_grey_world_B','max_white_B','grey_world_B','luminance_weighted_gray_world_B','Non_Local_Means_B','standard_deviation_and_luminance_weighted_gray_world_B','PIL_detail_B','YCrCb_Cb','Lab_B','erosion_B','PIL_emboss_G','PIL_edge_B','PIL_emboss_R','dilation_G','PIL_contour_G','HSV_V','IE_BBHE_R','PIL_contour_R','dilation_R','IE_BPHEME_G','IE_BBHE_G','IE_BPHEME_R','avg_G','Lab_L','median_G','gauss_G','Total_variation_G','automatic_color_equalization_B','YCrCb_Y','YUV_Y','bilateral_G','Lab_CLAHE_B','avg_R','BM3D_G','max_white_G','retinex_G','retinex_with_adjust_G','grey_world_G','RGB_G','Non_Local_Means_G','standard_deviation_and_luminance_weighted_gray_world_G','luminance_weighted_gray_world_G','standard_deviation_weighted_grey_world_G','GW_wb_G','median_R','gauss_R','PIL_detail_G','Total_variation_R','bilateral_R','BM3D_R','IE_BBHE_B','max_white_R','RGB_R','retinex_R','Non_Local_Means_R','standard_deviation_and_luminance_weighted_gray_world_R','grey_world_R','standard_deviation_weighted_grey_world_R','luminance_weighted_gray_world_R','GW_wb_R','retinex_with_adjust_R','PIL_detail_R','PIL_sharp_B','automatic_color_equalization_R','erosion_G','HSV_CLAHE_B','PIL_edge_G','IE_MMBEBHE_R','erosion_R','IE_MMBEBHE_G','PIL_edge_R','S_wb_G','S_wb_R','Lab_CLAHE_G','target'],
        120: ['laplace','butterworth_R','butterworth_G','IE_FHSABP_G','IE_FHSABP_R','IE_FHSABP_B','butterworth_B','IE_GHE_R','IE_GHE_G','HSV_equalized_R','HSV_equalized_G','Lab_equalized_R','IE_GHE_B','HSV_equalized_B','IE_DSIHE_R','Lab_equalized_G','IE_DSIHE_G','Lab_A','YCrCb_Cr','IE_DSIHE_B','YUV_V','IE_BPHEME_B','dilation_B','median_B','avg_B','Total_variation_B','PIL_emboss_B','gauss_B','YUV_U','PIL_contour_B','BM3D_B','bilateral_B','RGB_B','GW_wb_B','retinex_with_adjust_B','retinex_B','standard_deviation_weighted_grey_world_B','max_white_B','grey_world_B','luminance_weighted_gray_world_B','Non_Local_Means_B','standard_deviation_and_luminance_weighted_gray_world_B','PIL_detail_B','YCrCb_Cb','Lab_B','erosion_B','PIL_emboss_G','PIL_edge_B','PIL_emboss_R','dilation_G','PIL_contour_G','HSV_V','IE_BBHE_R','PIL_contour_R','dilation_R','IE_BPHEME_G','IE_BBHE_G','IE_BPHEME_R','avg_G','Lab_L','median_G','gauss_G','Total_variation_G','automatic_color_equalization_B','YCrCb_Y','YUV_Y','bilateral_G','Lab_CLAHE_B','avg_R','BM3D_G','max_white_G','retinex_G','retinex_with_adjust_G','grey_world_G','RGB_G','Non_Local_Means_G','standard_deviation_and_luminance_weighted_gray_world_G','luminance_weighted_gray_world_G','standard_deviation_weighted_grey_world_G','GW_wb_G','median_R','gauss_R','PIL_detail_G','Total_variation_R','bilateral_R','BM3D_R','IE_BBHE_B','max_white_R','RGB_R','retinex_R','Non_Local_Means_R','standard_deviation_and_luminance_weighted_gray_world_R','grey_world_R','standard_deviation_weighted_grey_world_R','luminance_weighted_gray_world_R','GW_wb_R','retinex_with_adjust_R','PIL_detail_R','PIL_sharp_B','automatic_color_equalization_R','erosion_G','HSV_CLAHE_B','PIL_edge_G','IE_MMBEBHE_R','erosion_R','IE_MMBEBHE_G','PIL_edge_R','S_wb_G','S_wb_R','Lab_CLAHE_G','automatic_color_equalization_G','IE_BHEPL_R','Lab_CLAHE_R','IE_BHEPL_G','HSV_CLAHE_G','IE_MMBEBHE_B','HSV_CLAHE_R','IE_BHEPL_B','PIL_sharp_G','PIL_sharp_R','target']
    }

    n_features_list = list(features_dict.keys())

    N = len(n_features_list)

    output_path_clustered = create_directories_for_results(output_path, dir_name_clustered, n_features_list)
    output_path = create_directories_for_results(output_path, dir_name, n_features_list)

    for i, (n_features, selected_columns) in enumerate(features_dict.items()):

        print(f'In progress: {i+1} of {N} \t | {n_features}')

        for ii, csv_name in enumerate(list_of_names_data):

            # Full data
            Xy = pd.read_csv(f'{data_path}{csv_name}')
            Xy_selected = Xy[selected_columns]
            Xy_selected.to_csv(f'{output_path}{n_features}/{csv_name}', index=False)

            # Clustered data
            Xy = pd.read_csv(f'{data_path_clustered}{csv_name}')
            Xy_selected = Xy[selected_columns]
            Xy_selected.to_csv(f'{output_path_clustered}{n_features}/{csv_name}', index=False)


def source_feature_selecting():
    """
    Perform feature selection based on predefined paths and parameters.

    Parameters:
    - None

    Returns:
    - None
    """

    n_clusters = '1584'

    folders = ['ALL']

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