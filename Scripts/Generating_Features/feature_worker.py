from colorcorrect.algorithm import grey_world, max_white, retinex, retinex_with_adjust, standard_deviation_weighted_grey_world, standard_deviation_and_luminance_weighted_gray_world, luminance_weighted_gray_world,automatic_color_equalization
from skimage.filters import roberts, sobel, scharr, prewitt, farid, laplace
from skimage.filters import butterworth, hessian
from PIL import Image, ImageFilter
from image_enhancement import image_enhancement

from skimage import img_as_float32
import skimage.restoration as sr

import pandas as pd
import numpy as np
import bm3d
import cv2


def get_features(img):
    """
    Extract various image features using different filters and algorithms.

    Parameters:
    - img (numpy.ndarray): Input image.

    Returns:
    - DataFrame: DataFrame containing all extracted features.
    """
    # ------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------ #
    # Variables of image and DataFrame

    # Float image color range <0,1>
    img_float = img_as_float32(img)

    # Grayscale image needed for edge detection and so on
    img_grayscale = (img_float[:,:,0]/3) + (img_float[:,:,1]/3) + (img_float[:,:,2]/3)

    # DataFrame of all extracted features
    df_final = pd.DataFrame()

    # ------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------ #
    # Static variables (which features we want) - right now all

    # Edge filters
    edge_filters = [roberts, sobel, scharr, prewitt, farid, laplace]

    # CCA filters 
    cca_filters_list = [grey_world, max_white, retinex, standard_deviation_weighted_grey_world, standard_deviation_and_luminance_weighted_gray_world,luminance_weighted_gray_world, retinex_with_adjust, automatic_color_equalization]

    # Image enhancement
    ie = image_enhancement.IE(img, color_space='HSV')
    img_enhancement_filters = [ie.GHE, ie.BBHE, ie.DSIHE, ie.MMBEBHE, ie.BPHEME, ie.FHSABP, ie.BHEPL, ie.RLBHE]

    # ------------------------------------------------------------------------------ #
    # ------------------------------------------------------------------------------ #
    # Extracting process

    # Gabor filter - features
    # df = gabor_filter(img_grayscale,sigmas=None, thetas=None, lambdas=None, gammas=None, psis=None)
    # df_final = pd.concat([df_final, df], axis=1)

    # Color spaces - features
    df = color_spaces(img, RGB=True, Lab=True, HSV=True, YCrCb= True, YUV=True)
    df_final = pd.concat([df_final, df], axis=1)

    # Smoothing filters - features
    df = smoothing_filters(img, average=True, median=True, gauss=True, bilateral=True)
    df_final = pd.concat([df_final, df], axis=1)

    # Edge detection - features
    df = edge_detection(img_grayscale, filters=edge_filters)
    df_final = pd.concat([df_final, df], axis=1)

    # PIL filters - features
    df = PIL_filters(img, sharpening=True, contour=True, emboss=True, edge=True, detail=True)
    df_final = pd.concat([df_final, df], axis=1)

    # ColorCorrect Algorithm - features
    df = cca_filters(img, filters=cca_filters_list)
    df_final = pd.concat([df_final, df], axis=1)    

    # Advanced filters - features
    df = advanced_filters(img_float, NLM=True, TV=True, BM=True)
    df_final = pd.concat([df_final, df], axis=1)

    # Histogram Equalization - features
    df = histogram_equalization(img, EQ=True, CL=True)
    df_final = pd.concat([df_final, df], axis=1)

    # Image enhancement library - features
    df = img_enhancement(img_enhancement_filters)
    df_final = pd.concat([df_final, df], axis=1)

    # skimage.filters library - features
    df = skimage_filters(img_float, BW=True, H=True)
    df_final = pd.concat([df_final, df], axis=1)

    # morphological filters
    df = morphological_filters(img, erosion=True, dilation=True)
    df_final = pd.concat([df_final, df], axis=1)

    # CV2 filters - features
    df = cv2_filters(img, S=True, GW=True)
    df_final = pd.concat([df_final, df], axis=1)

    return df_final


def color_spaces(img, RGB=False, Lab=False, HSV=False, YCrCb=False, YUV=False):
    """
    Extract color space features from the input image.

    https://learnopencv.com/color-spaces-in-opencv-cpp-python/

    Parameters:
    - img (numpy.ndarray): Input image.
    - RGB (bool, optional): Include RGB color space features.
    - Lab (bool, optional): Include Lab color space features.
    - HSV (bool, optional): Include HSV color space features.
    - YCrCb (bool, optional): Include YCrCb color space features.
    - YUV (bool, optional): Include YUV color space features.

    Returns:
    - DataFrame: DataFrame containing color space features.
    """

    max_value = 255.0

    df = pd.DataFrame()

    # RGB - color space
    if RGB:
        img_RGB = img_as_float32(img)

        df['RGB_R'] = img_RGB[:, :, 0].reshape(-1)
        df['RGB_G'] = img_RGB[:, :, 1].reshape(-1)
        df['RGB_B'] = img_RGB[:, :, 2].reshape(-1)

    # Lab - color space
    if Lab:
        img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_LAB = img_as_float32(img_LAB)

        df['Lab_L'] = img_LAB[:, :, 0].reshape(-1)
        df['Lab_A'] = img_LAB[:, :, 1].reshape(-1)
        df['Lab_B'] = img_LAB[:, :, 2].reshape(-1)

    # HSV - color space
    if HSV:
        img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_HSV = img_as_float32(img_HSV)

        df['HSV_H'] = img_HSV[:, :, 0].reshape(-1)
        df['HSV_S'] = img_HSV[:, :, 1].reshape(-1)
        df['HSV_V'] = img_HSV[:, :, 2].reshape(-1)

    # YCrCb - color space
    if YCrCb:
        img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        img_YCrCb = img_as_float32(img_YCrCb)

        df['YCrCb_Y'] = img_YCrCb[:, :, 0].reshape(-1)
        df['YCrCb_Cr'] = img_YCrCb[:, :, 1].reshape(-1)
        df['YCrCb_Cb'] = img_YCrCb[:, :, 2].reshape(-1)

    # YUV - color space
    if YUV:
        img_YUV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_YUV = img_as_float32(img_YUV)

        df['YUV_Y'] = img_YUV[:, :, 0].reshape(-1)
        df['YUV_U'] = img_YUV[:, :, 1].reshape(-1)
        df['YUV_V'] = img_YUV[:, :, 2].reshape(-1)

    return df


def smoothing_filters(img, average=False, median=False, gauss=False, bilateral=False):
    """
    Apply various smoothing filters to the input image.

    https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html

    Parameters:
    - img (numpy.ndarray): Input image.
    - average (bool, optional): Apply average filter.
    - median (bool, optional): Apply median filter.
    - gauss (bool, optional): Apply Gaussian filter.
    - bilateral (bool, optional): Apply bilateral filter.

    Returns:
    - DataFrame: DataFrame containing smoothed images.
    """

    df = pd.DataFrame()

    k_size = 9

    # average filter
    if average:
        img_avg = cv2.blur(img, (k_size, k_size))
        img_avg = img_as_float32(img_avg)
        df['avg_R'] = img_avg[:, :, 0].reshape(-1)
        df['avg_G'] = img_avg[:, :, 1].reshape(-1)
        df['avg_B'] = img_avg[:, :, 2].reshape(-1)

    # median filter
    if median:
        img_median = cv2.medianBlur(img, k_size)
        img_median = img_as_float32(img_median)
        df['median_R'] = img_median[:, :, 0].reshape(-1)
        df['median_G'] = img_median[:, :, 1].reshape(-1)
        df['median_B'] = img_median[:, :, 2].reshape(-1)

    # gauss filter
    if gauss:
        img_gauss = cv2.GaussianBlur(img, (k_size, k_size), sigmaX=0, sigmaY=0)
        img_gauss = img_as_float32(img_gauss)
        df['gauss_R'] = img_gauss[:, :, 0].reshape(-1)
        df['gauss_G'] = img_gauss[:, :, 1].reshape(-1)
        df['gauss_B'] = img_gauss[:, :, 2].reshape(-1)

    # bilateral filter
    if bilateral:
        img_bilateral = cv2.bilateralFilter(img, k_size, 80, 80)
        img_bilateral = img_as_float32(img_bilateral)
        df['bilateral_R'] = img_bilateral[:, :, 0].reshape(-1)
        df['bilateral_G'] = img_bilateral[:, :, 1].reshape(-1)
        df['bilateral_B'] = img_bilateral[:, :, 2].reshape(-1)

    return df


def edge_detection(img, filters=None):
    """
    Apply edge detection filters to the grayscale image.

    Parameters:
    - img (numpy.ndarray): Grayscale image.
    - filters (list, optional): List of edge detection filters.

    Returns:
    - DataFrame: DataFrame containing edge-detected images.
    """

    df = pd.DataFrame()

    if filters is None or not filters:
        return df

    for i, f in enumerate(filters):
        img_filtered = f(img).reshape(-1)
        df[f.__name__] = img_filtered

    return df


def PIL_filters(img, sharpening=False, contour=False, emboss=False, edge=False, detail=False):
    """
    Apply various filters from the Python Imaging Library (PIL) to the input image.

    https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html
    https://www.geeksforgeeks.org/python-pil-image-filter-with-imagefilter-module/

    Parameters:
    - img (numpy.ndarray): Input image.
    - sharpening (bool, optional): Apply sharpening filter.
    - contour (bool, optional): Apply contour filter.
    - emboss (bool, optional): Apply emboss filter.
    - edge (bool, optional): Apply edge enhancement filter.
    - detail (bool, optional): Apply detail filter.

    Returns:
    - DataFrame: DataFrame containing filtered images.
    """

    df = pd.DataFrame()

    # numpy array to PIL image
    img_pil = Image.fromarray(img, 'RGB')

    # Image sharpening
    if sharpening:
        img_sharpened = img_pil.filter(ImageFilter.UnsharpMask(radius = 10, percent = 300, threshold = 3))
        img_sharpened = img_as_float32(np.asarray(img_sharpened))
        df['PIL_sharp_R'] = img_sharpened[:, :, 0].reshape(-1)
        df['PIL_sharp_G'] = img_sharpened[:, :, 1].reshape(-1)
        df['PIL_sharp_B'] = img_sharpened[:, :, 2].reshape(-1)

    # Image contour
    if contour:
        img_contour = img_pil.filter(ImageFilter.CONTOUR)
        img_contour = img_as_float32(np.asarray(img_contour))
        df['PIL_contour_R'] = img_contour[:, :, 0].reshape(-1)
        df['PIL_contour_G'] = img_contour[:, :, 1].reshape(-1)
        df['PIL_contour_B'] = img_contour[:, :, 2].reshape(-1)

    # EMBOSS filter
    if emboss:
        img_emboss = img_pil.filter(ImageFilter.EMBOSS)
        img_emboss = img_as_float32(np.array(img_emboss))
        df['PIL_emboss_R'] = img_emboss[:, :, 0].reshape(-1)
        df['PIL_emboss_G'] = img_emboss[:, :, 1].reshape(-1)
        df['PIL_emboss_B'] = img_emboss[:, :, 2].reshape(-1)

    # EDGE_ENHANCE filter
    if edge:
        img_edge = img_pil.filter(ImageFilter.EDGE_ENHANCE)
        img_edge = img_as_float32(np.asarray(img_edge))
        df['PIL_edge_R'] = img_edge[:, :, 0].reshape(-1)
        df['PIL_edge_G'] = img_edge[:, :, 1].reshape(-1)
        df['PIL_edge_B'] = img_edge[:, :, 2].reshape(-1)

    # DETAIL filter
    if detail:
        img_detail = img_pil.filter(ImageFilter.DETAIL)
        img_detail = img_as_float32(np.asarray(img_detail))
        df['PIL_detail_R'] = img_detail[:, :, 0].reshape(-1)
        df['PIL_detail_G'] = img_detail[:, :, 1].reshape(-1)
        df['PIL_detail_B'] = img_detail[:, :, 2].reshape(-1)

    return df


def cca_filters(img, filters=None):
    """
    Apply various color correction algorithms to the input image.

    Parameters:
    - img (numpy.ndarray): Input image.
    - filters (list, optional): List of color correction algorithms.

    Returns:
    - DataFrame: DataFrame containing color-corrected images.
    """

    df = pd.DataFrame()

    if filters is None or not filters:
        return df

    for i, f in enumerate(filters):

        img_filtered = f(img)
        img_filtered = img_as_float32(img_filtered)

        df[f'{f.__name__}_R'] = img_filtered[:, :, 0].reshape(-1)
        df[f'{f.__name__}_G'] = img_filtered[:, :, 1].reshape(-1)
        df[f'{f.__name__}_B'] = img_filtered[:, :, 2].reshape(-1)

    return df


def advanced_filters(img, NLM=False, TV=False, BM=False):
    """
    Apply advanced filters to the input image.

    NLM
        https://www.youtube.com/watch?v=3-53P4zUkZQ
        https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf

    TV
        https://www.youtube.com/watch?v=_Ybek8eMGKU
        https://hal.archives-ouvertes.fr/hal-00437581/document

    BM
        https://www.youtube.com/watch?v=HAOeYCGFGaE
        https://webpages.tuni.fi/foi/papers/ICIP2019_Ymir.pdf

    Parameters:
    - img (numpy.ndarray): Input image.
    - NLM (bool, optional): Apply Non-Local Means denoising.
    - TV (bool, optional): Apply Total Variation denoising.
    - BM (bool, optional): Apply Block-Matching 3D denoising.

    Returns:
    - DataFrame: DataFrame containing filtered images.
    """

    df = pd.DataFrame()

    if NLM:
        sigma_est = np.mean(sr.estimate_sigma(img, channel_axis=-1))
        img_filtered = sr.denoise_nl_means(img, h=sigma_est, fast_mode=True, patch_size=5, patch_distance=3,channel_axis=-1)

        df['Non_Local_Means_R'] = img_filtered[:, :, 0].reshape(-1)
        df['Non_Local_Means_G'] = img_filtered[:, :, 1].reshape(-1)
        df['Non_Local_Means_B'] = img_filtered[:, :, 2].reshape(-1)

    if TV:
        img_filtered = sr.denoise_tv_chambolle(img, weight=0.1, eps=0.0002, max_num_iter=200, channel_axis=-1)

        df['Total_variation_R'] = img_filtered[:, :, 0].reshape(-1)
        df['Total_variation_G'] = img_filtered[:, :, 1].reshape(-1)
        df['Total_variation_B'] = img_filtered[:, :, 2].reshape(-1)

    if BM:
        img_filtered = bm3d.bm3d(img, sigma_psd=0.1, stage_arg=bm3d.BM3DStages.ALL_STAGES)

        df['BM3D_R'] = img_filtered[:, :, 0].reshape(-1)
        df['BM3D_G'] = img_filtered[:, :, 1].reshape(-1)
        df['BM3D_B'] = img_filtered[:, :, 2].reshape(-1)

    return df


def histogram_equalization(img, EQ=False, CL=False):
    """
    Apply histogram equalization techniques to the input image.

    https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial42_what_is_histogram_equalization_CLAHE.py

    Parameters:
    - img (numpy.ndarray): Input image.
    - EQ (bool, optional): Apply histogram equalization.
    - CL (bool, optional): Apply contrast limited adaptive histogram equalization.

    Returns:
    - DataFrame: DataFrame containing equalized images.
    """

    df = pd.DataFrame()

    img_Lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(img_Lab)

    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(img_HSV)

    if EQ:
        equ = cv2.equalizeHist(L)
        img_equalized = cv2.merge((equ, a, b))
        img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_LAB2RGB)
        img_equalized = img_as_float32(img_equalized)

        df['Lab_equalized_R'] = img_equalized[:, :, 0].reshape(-1)
        df['Lab_equalized_G'] = img_equalized[:, :, 1].reshape(-1)
        df['Lab_equalized_B'] = img_equalized[:, :, 2].reshape(-1)

    if EQ:
        equ = cv2.equalizeHist(V)
        img_equalized = cv2.merge((H, S, equ))
        img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_HSV2RGB)
        img_equalized = img_as_float32(img_equalized)

        df['HSV_equalized_R'] = img_equalized[:, :, 0].reshape(-1)
        df['HSV_equalized_G'] = img_equalized[:, :, 1].reshape(-1)
        df['HSV_equalized_B'] = img_equalized[:, :, 2].reshape(-1)

    if CL:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        equ = clahe.apply(L)
        img_CLAHE = cv2.merge((equ, a, b))
        img_CLAHE = cv2.cvtColor(img_CLAHE, cv2.COLOR_LAB2RGB)
        img_CLAHE = img_as_float32(img_CLAHE)

        df['Lab_CLAHE_R'] = img_CLAHE[:, :, 0].reshape(-1)
        df['Lab_CLAHE_G'] = img_CLAHE[:, :, 1].reshape(-1)
        df['Lab_CLAHE_B'] = img_CLAHE[:, :, 2].reshape(-1)

    if CL:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        equ = clahe.apply(V)
        img_CLAHE = cv2.merge((H, S, equ))
        img_CLAHE = cv2.cvtColor(img_CLAHE, cv2.COLOR_HSV2RGB)
        img_CLAHE = img_as_float32(img_CLAHE)

        df['HSV_CLAHE_R'] = img_CLAHE[:, :, 0].reshape(-1)
        df['HSV_CLAHE_G'] = img_CLAHE[:, :, 1].reshape(-1)
        df['HSV_CLAHE_B'] = img_CLAHE[:, :, 2].reshape(-1)

    return df


def img_enhancement(filters=None):
    """
    Apply image enhancement filters to the input image.

    https://pypi.org/project/image-enhancement/

    Parameters:
    - filters (list, optional): List of image enhancement filters.

    Returns:
    - DataFrame: DataFrame containing enhanced images.
    """

    df = pd.DataFrame()

    if filters is None or not filters:
        return df

    for i, f in enumerate(filters):
        img_filtered = f()
        img_filtered = img_as_float32(img_filtered)

        df[f'IE_{f.__name__}_R'] = img_filtered[:, :, 0].reshape(-1)
        df[f'IE_{f.__name__}_G'] = img_filtered[:, :, 1].reshape(-1)
        df[f'IE_{f.__name__}_B'] = img_filtered[:, :, 2].reshape(-1)

    return df


def skimage_filters(img,BW=False,H=False):
    """
    Apply filters from the skimage.filters library to the input image.

    Parameters:
    - img (numpy.ndarray): Input image.
    - BW (bool, optional): Apply Butterworth filter.
    - H (bool, optional): Apply Hessian filter.

    Returns:
    - DataFrame: DataFrame containing filtered images.
    """

    df = pd.DataFrame()

    if BW:
        img_filtered = butterworth(img)

        df['butterworth_R'] = img_filtered[:, :, 0].reshape(-1)
        df['butterworth_G'] = img_filtered[:, :, 1].reshape(-1)
        df['butterworth_B'] = img_filtered[:, :, 2].reshape(-1)
    if H:
        img_filtered = hessian(img[:, :, 0])
        df['hessian_R'] = img_filtered.reshape(-1)

        img_filtered = hessian(img[:, :, 1])
        df['hessian_G'] = img_filtered.reshape(-1)

        img_filtered = hessian(img[:, :, 2])
        df['hessian_B'] = img_filtered.reshape(-1)

    return df


def morphological_filters(img,erosion=False,dilation=False):
    """
    Apply morphological operations (erosion and dilation) to the input image channels.

    Parameters:
    - img (numpy.ndarray): Input RGB image.
    - erosion (bool, optional): Apply erosion to each color channel if True. Default is False.
    - dilation (bool, optional): Apply dilation to each color channel if True. Default is False.

    Returns:
    - DataFrame: DataFrame containing flattened arrays of pixel values after applying morphological operations.
    """

    df = pd.DataFrame()

    kernel = np.ones((5,5), dtype=np.uint8)
    r, g, b = cv2.split(img)

    if erosion:
        erosion_r = cv2.erode(r, kernel, iterations=1)
        erosion_g = cv2.erode(g, kernel, iterations=1)
        erosion_b = cv2.erode(b, kernel, iterations=1)

        df['erosion_R'] = img_as_float32(erosion_r).reshape(-1)
        df['erosion_G'] = img_as_float32(erosion_g).reshape(-1)
        df['erosion_B'] = img_as_float32(erosion_b).reshape(-1)

    if dilation:
        dilation_r = cv2.dilate(r, kernel, iterations=1)
        dilation_g = cv2.dilate(g, kernel, iterations=1)
        dilation_b = cv2.dilate(b, kernel, iterations=1)

        df['dilation_R'] = img_as_float32(dilation_r).reshape(-1)
        df['dilation_G'] = img_as_float32(dilation_g).reshape(-1)
        df['dilation_B'] = img_as_float32(dilation_b).reshape(-1)

    return df


def cv2_filters(img, S=False, GW=False):
    """
    Apply OpenCV filters to the input image.

    Parameters:
    - img (numpy.ndarray): Input image.
    - S (bool, optional): Apply Simple White Balance.
    - GW (bool, optional): Apply Grayworld White Balance.

    Returns:
    - DataFrame: DataFrame containing filtered images.
    """

    df = pd.DataFrame()

    if S:
        S_wb = cv2.xphoto.createSimpleWB()
        img_filtered = S_wb.balanceWhite(img)
        img_filtered = img_as_float32(img_filtered)

        df['S_wb_R'] = img_filtered[:, :, 0].reshape(-1)
        df['S_wb_G'] = img_filtered[:, :, 1].reshape(-1)
        df['S_wb_B'] = img_filtered[:, :, 2].reshape(-1)

    if GW:
        GW_wb = cv2.xphoto.createGrayworldWB()
        img_filtered = GW_wb.balanceWhite(img)
        img_filtered = img_as_float32(img_filtered)

        df['GW_wb_R'] = img_filtered[:, :, 0].reshape(-1)
        df['GW_wb_G'] = img_filtered[:, :, 1].reshape(-1)
        df['GW_wb_B'] = img_filtered[:, :, 2].reshape(-1)

    return df


def gabor_filter(img, sigmas=None, thetas=None, lambdas=None, gammas=None, psis=None):
    """
    Apply Gabor filter to the input image.

    https://en.wikipedia.org/wiki/Gabor_filter
    https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html
    https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
    https://inc.ucsd.edu/mplab/75/media//gabor.pdf
    https://www.baeldung.com/cs/ml-gabor-filters
    https://wisdomml.in/gabor-filter-based-edge-detection-in-image-processing/

    Parameters:
    - img (numpy.ndarray): Input image.
    - sigmas (list, optional): List of standard deviations for the Gaussian envelope.
    - thetas (list, optional): List of orientations of the normal to the parallel stripes.
    - lambdas (list, optional): List of wavelengths of the sinusoidal factor.
    - gammas (list, optional): List of aspect ratios of the Gaussian envelope.
    - psis (list, optional): List of phase offsets of the sinusoidal factor.

    Returns:
    - DataFrame: DataFrame containing Gabor-filtered images.
    """

    if sigmas is None and thetas is None and lambdas is None and gammas is None and psis is None:
        return pd.DataFrame()

    if sigmas is None:
        sigmas = np.array([0])
    if thetas is None:
        thetas = np.array([0])
    if lambdas is None:
        lambdas = np.array([0])
    if gammas is None:
        gammas = np.array([0])
    if psis is None:
        psis = np.array([0])

    k_size = 15

    n = 0
    kernels = []

    rows = img.reshape(-1).shape[0]
    cols = len(sigmas)*len(thetas)*len(lambdas)*len(gammas)*len(psis)

    gabor_features = np.zeros((rows,cols))
    gabor_labels = []

    for s in sigmas:
        for t in thetas:
            for l in lambdas:
                for g in gammas:
                    for p in psis:

                        gabor_labels.append('Gabor_' + 's_' + f'%07.3f' % s + '_' + 't_' + f'%07.3f' % t + '_' + 'l_' + f'%07.3f' % l + '_' + 'g_' + f'%07.3f' % g + '_' + 'p_' + f'%07.3f' % p)

                        gabor_kernel = cv2.getGaborKernel((k_size, k_size), s, t, l, g, p, ktype=cv2.CV_32F)
                        kernels.append(gabor_kernel)

                        gabor_features[:,n] = cv2.filter2D(img, cv2.CV_8UC3, gabor_kernel).reshape(-1)

                        n += 1

    df = pd.DataFrame(gabor_features,columns=gabor_labels)

    return df


if __name__ == '__main__':

    print('Hello, home!')