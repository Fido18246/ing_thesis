import os
import datetime
from fpdf import FPDF
import pandas as pd


def create_PDF(path_original, path_results):
    """
    Create a PDF document containing original, annotated, and predicted images side by side.

    Parameters:
    - path_original (str): Path to the directory containing original images.
    - path_results (str): Path to the directory containing annotated and predicted images.

    Returns:
    - None
    """
    
    img_names = os.listdir(path_original)
    df = pd.read_csv(f'{path_results}results.csv', index_col=0)

    width = 200
    height = 220

    space = 10
    im_width = (width - 3*space)//2

    pdf = FPDF('L', 'mm', (height,width))
    pdf.set_font("Arial", size=15)

    x1 = space
    x2 = (width/2) + (space/2)

    y2 = 25 + im_width + 15

    for i, img_name in enumerate(img_names):
        pdf.add_page()
        pdf.cell(w=0, h=0, txt=img_name, align='C')

        name = img_name[:-4]

        im1 = f'{path_original}{name}.png'
        im2 = f'{path_results}{name}_target.png'
        im3 = f'{path_results}{name}_prediction.png'
        im4 = f'{path_results}{name}_cm.png'

        pdf.set_xy(x=x1, y=10)
        pdf.cell(w=im_width, h=20, txt='Original', align='C')
        pdf.image(im1, x=x1, y=25, w=im_width)

        pdf.set_xy(x=x2, y=10)
        pdf.cell(w=im_width, h=20, txt=f"DSC = {df.at[name, 'Dice_coefficient']:.4f}", align='C')
        pdf.image(im4, x=x2, y=25, w=im_width)

        pdf.set_xy(x=x1, y=25 + im_width)
        pdf.cell(w=im_width, h=20, txt='Annotaton', align='C')
        pdf.image(im2, x=x1, y=y2, w=im_width)

        pdf.set_xy(x=x2, y=25 + im_width)
        pdf.cell(w=im_width, h=20, txt='Prediction', align='C')
        pdf.image(im3, x=x2, y=y2, w=im_width)

    if os.path.exists(f'{path_results}results.pdf'):
        os.remove(f'{path_results}results.pdf')

    pdf.output(f'{path_results}results.pdf')

    return None


def source_create_PDF():
    """
    Main function to create PDF documents.

    Returns:
    - None
    """

    folders = ['XGBClassifier_Results', 'KNeighborsClassifier_Results', 'GaussianNB_Results']

    for i, folder in enumerate(folders):

        t0 = datetime.datetime.now()

        DATA_PATH_ORIGINAL = f'./Results/Resizing_Images/Images_ALL/'
        DATA_PATH_MODEL_RESULTS = f'./Results/Other_ML_Algorithms/{folder}/'

        create_PDF(DATA_PATH_ORIGINAL, DATA_PATH_MODEL_RESULTS)

        t1 = datetime.datetime.now()

        print(f'PDF created in: {t1 - t0}')


if __name__ == '__main__':

    print('Hello, home!')

    source_create_PDF()