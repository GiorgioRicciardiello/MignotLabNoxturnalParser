"""
Author: Giorgio Ricciardiello
        giorgio.crm6@gmail.com

Receive images with a numerical text and we will extract the numbers as texts
Follow the Tesseract library implementation found in
    https://www.thepythoncode.com/article/extract-text-from-images-or-scanned-pdf-python

Alternative implementation: Train model with MNIST dataset
    https://www.openml.org/search?type=data&status=active&id=554

The class contains the run method which automatize all the necessary operations to pre-process and extract the text
from the images of the NOX sleep report. The class runs one patient at the time.

"""
import pathlib
import pickle
import cv2
import numpy as np
import pandas as pd
import PIL.Image
import pytesseract
from pytesseract import Output
from config.config import config

class TesseracModel():
    def __init__(self):
        print(f'TesseracModel constructed')

    def run(self, image:PIL.Image.Image, file_name:str) -> tuple[float, float]:
        """
        Apply the necessary signal pre-processing and text extraction from the image. An image and a file name is
        received at each run. The resulting extracted text is returned
        :param image: image containing the cropped text
        :param file_name: name of the file from where the image is taken
        :return: [txt contain in the image, certainty of the prediction]
        """

        # pil to cv2 object
        img_cv2 = self.pil_to_cv2(pil_img_object=image)
        # debug - display image
        # self.display_img(img=img_cv2)

        # Segmentation of the number
        img_cv2_gray = self.grayscale(img=img_cv2)
        img_cv2_gray_th = self.threshold(img=img_cv2_gray, min=200, max=225, binary_inv=True)
        # img_cv2_gray_th = self.threshold(img=img_cv2_gray, min=200, max=225, binary_inv=False)

        # self.display_img(img=img_cv2_gray_th)

        details= self.detect_txt(pre_proc_img=img_cv2_gray_th)

        return self.get_txt(ss_details=details, name_file=file_name)


    def open_pdf_content(self, input_dct_path: str ) -> dict:
        """Open the pdf dictionary saved as a pickle"""
        input_path = pathlib.Path(__file__).parents[1].joinpath(input_dct_path)
        if input_path.joinpath('pdf_content.pickle').exists():
            with open(input_path.joinpath('pdf_content.pickle'), 'rb') as handle:
                # self.pdf_content = pickle.load(handle)
                return pickle.load(handle)
        else:
            raise ValueError(f'Unable to locate extracted dictionary from pdf on \n {input_path}')

    def pil_to_cv2(self, pil_img_object:PIL.Image.Image) -> np.array:
        img = cv2.cvtColor(np.array(pil_img_object), cv2.COLOR_RGB2BGR)
        return img


    def display_img(self, img:PIL.Image.Image, title='Image'):
        """Displays an image on screen and maintains the output until the user presses a key"""
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.setWindowTitle('img', title)
        cv2.resizeWindow('img', 200, 200)
        # Display Image on screen
        cv2.imshow('img', img)
        # Mantain output until user presses a key
        cv2.waitKey(0)
        # Destroy windows when user presses a key
        cv2.destroyAllWindows()

    def save_result(self, output_path):
        """Once the text is extracted and the dictionary is updated, we save the updated dictionary"""
        if output_path.is_dir():
            with open(output_path.joinpath('pdf_content.pickle'), 'wb') as handle:
                pickle.dump(self.pdf_content, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if self._output_path.joinpath('pdf_content.pickle').exists():
                print(f'File saved in {self._output_path}')
        else:
            raise ValueError(f'The output path{self._output_path} does not exists \n unable to save')

    #%% preprocessing
    # Convert to grayscale
    @staticmethod
    def grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove noise
    @staticmethod
    def remove_noise(img):
        return cv2.medianBlur(img, 5)

    # Thresholding
    @staticmethod
    def threshold(img, min:int=0, max:int=225, binary_inv:bool=True):
        if binary_inv:
            return cv2.threshold(img, min, max, cv2.THRESH_BINARY_INV)[1]
        else:
            return cv2.threshold(img, min, max, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # dilation
    @staticmethod
    def dilate(img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)

    # erosion
    @staticmethod
    def erode(img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(img, kernel, iterations=1)

    # opening -- erosion followed by a dilation
    @staticmethod
    def opening(img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    @staticmethod
    def canny(img):
        return cv2.Canny(img, 100, 200)

    # skew correction
    @staticmethod
    def deskew(img):
        coords = np.column_stack(np.where(img > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # template matching
    @staticmethod
    def match_template(img, template):
        return cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    @staticmethod
    def calculate_ss_confidence(ss_details: dict):
        """Calculate the confidence score of the text grabbed from the scanned image."""
        # page_num  --> Page number of the detected text or item
        # block_num --> Block number of the detected text or item
        # par_num   --> Paragraph number of the detected text or item
        # line_num  --> Line number of the detected text or item
        # Convert the dict to dataFrame
        df = pd.DataFrame.from_dict(ss_details)
        # Convert the field conf (confidence) to numeric
        df['conf'] = pd.to_numeric(df['conf'], errors='coerce')
        # Elliminate records with negative confidence
        df = df[df.conf != -1]
        # Calculate the mean confidence by page
        conf = df.groupby(['page_num'])['conf'].mean().tolist()
        return conf[0]


    # %% text detection
    def detect_txt(self, pre_proc_img) -> dict:
        """Define the model properties that best suit the NOX images where there is only numerical text"""
        # oem --> OCR engine mode = 3 >> Legacy + LSTM mode only (LSTM neutral net mode works the best)
        # psm --> page segmentation mode = 6 >> Assume as single uniform block of text (How a page of text can be analyzed)
        config_param = r'--oem 3 --psm 6'
        details = pytesseract.image_to_data(
            pre_proc_img, output_type=Output.DICT, config=config_param, lang='eng')
        return details


    def get_txt(self, ss_details: dict, name_file:str) -> tuple[float, float]:
        """
        Extract the text found in the image and return the text with the confidence of the prediction
        :param ss_details:
        :return: [text as float, confidence of prediction]
        """
        df = pd.DataFrame.from_dict(ss_details)
        df = df.replace(r'^\s*$', np.nan, regex=True)
        if df['text'].count() > 1:
            raise ValueError(f'Multiple Numbers found in image of file {name_file} ')
        else:
            # use the row with the valyes
            df = df[df['text'].notnull()]
            df['text'] = float(df['text'])
        return df['text'].iloc[0], df['conf'].iloc[0]

if __name__ == "__main__":
    teserac_model = TesseracModel()
    # 1 open the dictionary containing the file information
    pdf_content = teserac_model.open_pdf_content(input_dct_path=config['extracted_dict_path'])

    for file_ in pdf_content.keys():
        for summ_img_key_, summ_img_value_ in pdf_content[file_]['summary_images'].item():
            img_txt, txt_conf = teserac_model.run(image=summ_img_value_, file_name=file_)
            tmp_img = pdf_content[file_]['summary_images'][summ_img_key_]
            pdf_content[file_]['summary_images'][summ_img_key_] = {'image':tmp_img, 'txt':img_txt, 'conf':txt_conf}



