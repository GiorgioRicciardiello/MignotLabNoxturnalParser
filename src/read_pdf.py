"""
Author: Giorgio Ricciardiello
        giorgio.crm6@gmail.com

Function classes are made to extract text from an independent pdf at the time, this makes
a modular program
https://www.youtube.com/watch?v=w2r2Bg42UPY
"""
import pathlib
from typing import Any, Optional
from config.config import config
import pdfplumber
import fitz
import PIL.Image
import io
import pickle


class ReadPdf:
    def __init__(self, input_path: str, output_path: str):
        self._input_path = pathlib.Path(__file__).parents[1].joinpath(input_path)
        self._output_path = pathlib.Path(__file__).parents[1].joinpath(output_path)
        self.pdf_content = {}

    def run(self):
        """Run multiple files at the same time, builts the output dictionary"""
        pdf_files = self.searching_all_files(dirpath=self._input_path)
        for file_ in pdf_files:
            file_name, file_text = self.extract_text(file_path=file_)
            self.pdf_content[file_name] = {}
            self.pdf_content[file_name]['file_text'] = file_text
            self.pdf_content[file_name]['summary_images'] = self.extract_images(file_path=file_)
        # print(self.pdf_content.keys())
        self.save_pdf_content()
        print(f'Read pdfs: {len(self.pdf_content.keys())}')

    @staticmethod
    def searching_all_files(dirpath: pathlib.Path) -> list:
        assert dirpath.is_dir()
        file_list = []
        for x in dirpath.iterdir():
            if x.is_file():
                file_list.append(x)
            # elif x.is_dir():
            #     file_list.extend(self.searching_all_files(x))
        return file_list

    def extract_text(self, file_path: pathlib.Path) -> tuple[Any, str]:
        """Return Union[keys, value] as strings Key:File name (patent ID), value: text of the pdf"""
        # file_path = pdf_files[0]
        pages = [0, 1]
        text = []
        with pdfplumber.open(file_path) as pdf:
            # pdf.pages[0].extract_text()
            for page_ in pages:
                text.append(pdf.pages[page_].extract_text())
        return file_path.name.replace('.pdf', ''), "".join(text)

    def extract_images(self, file_path: pathlib.Path) -> dict:
        """Ectract the image containing the AHO, ODI and Snore and return a dictionary"""
        # https://www.youtube.com/watch?v=w2r2Bg42UPY&ab_channel=NeuralNine
        # file_path = self.searching_all_files(dirpath=self.input_path)[0]
        summary_images = {
            'AHI': 0,
            'ODI': 0,
            'Snore': 0
        }
        img_idx_file = {1: 'AHI', 2: 'ODI', 3: 'Snore'}
        pdf = fitz.open(file_path)
        first_page = pdf[0]  # first page
        images_fp = first_page.get_images()
        # iterate over the images
        # images_fp[1] == AHI, images_fp[2] == ODI, images_fp[3] == Snore
        for idx_, image in zip(img_idx_file, images_fp[1:4]):
            # image = images_fp[3]
            base_img = pdf.extract_image(image[0])  # base image -> dictionary meta data
            image_data = base_img['image']  # image data
            img = PIL.Image.open(io.BytesIO(image_data))
            img_res = self._crop_image(image=img)
            # img_res.show()
            summary_images[img_idx_file[idx_]] = img_res
        return summary_images

    @staticmethod
    def _crop_image(image, newsize: tuple = (100, 100), newsize_bool: bool = False) -> PIL.Image.Image:
        """Crop and center the image to have only the numbers in the most size of the image"""
        # https://www.youtube.com/watch?v=TVGCHA0-1Uk&ab_channel=Finxter-CreateYourSix-FigureCodingBusiness
        # width, height = image.size
        # image.show()
        x, y = 130, 80  # coordinates of starting point to initiate crop
        width, height = 150, 70  # dimensions of the rectangle from x, y coordinates
        area = (x, y, x + width, y + height)
        # Cropped image of above dimension
        im1 = image.crop(box=area)
        if newsize_bool:
            im1 = im1.resize(newsize)  # , PIL.Image.ANTIALIAS)
        # im1.show()
        return im1

    def get_pdfs_content(self) -> dict:
        return self.pdf_content

    def save_pdf_content(self, pdf_content: Optional[dict] = None, name_save: Optional[str] = 'blank'):
        """
        Save the data extracted from the pdf, the method can also received an updated pdf data and save it in the same
        location
        :param pdf_content:
        :return:
        """
        if self._output_path.is_dir():
            if pdf_content is None:
                # save the readed content from the raw pdf
                with open(self._output_path.joinpath('pdf_content.pickle'), 'wb') as handle:
                    pickle.dump(self.pdf_content, handle, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                # save update version of the pdf content
                with open(self._output_path.joinpath(f'{name_save}.pickle'), 'wb') as handle:
                    pickle.dump(pdf_content, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if self._output_path.joinpath('pdf_content.pickle').exists():
                print(f'File saved in {self._output_path}')
            else:
                raise ValueError(f'Error in dump File NOT saved in {self._output_path}')
        else:
            raise ValueError(f'The output path{self._output_path} does not exists \n unable to save')


if __name__ == "__main__":
    readpdf = ReadPdf(input_path=config['input_path'], output_path=config['extracted_dict_path'])
    readpdf.run()
    pdfs_content = readpdf.get_pdfs_content()
    # readpdf.extract_images()
