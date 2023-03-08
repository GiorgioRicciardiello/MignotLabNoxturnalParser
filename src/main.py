"""
Author: Giorgio Ricciardiello
        giorgio.crm6@gmail.com

Main script. The pdf is read as a complete string, information is then extracted with the necessary processing tools,
a dictionary is saved where the extracted information of each subject can be analyses. Lastly, the informatio from
the saved dictionary is parsed into an excel file and merged with all the patients pdf's inserted in the input fold

HOW TO:
    1. Insert the NOX sleep reports in the folder data/input_psg_reports
    2. run the main.py script
    3. View the Excel table containing all the patients in the foler data/output_tables_report
    4. Happy sleep hunting :)
"""

from src.read_pdf import ReadPdf
from src.nox_to_excel import NoxExcel
from src.image_classifier_model_tesserac import TesseracModel
from config.config import config
from config.pdf_structure import nox_sections

if __name__ == "__main__":
    #%% read the pdfs in the input file
    readpdf = ReadPdf(input_path=config['input_path'], output_path=config['extracted_dict_path'])
    readpdf.run()
    pdfs_content = readpdf.get_pdfs_content()
    #%% Images to text
    teserac_model = TesseracModel()
    # Iterate over all the patients (keys) in the constructed dictionary and extract the text from the images
    for file_ in pdfs_content.keys():
        for summ_img_key_, summ_img_value_ in pdfs_content[file_]['summary_images'].items():
            img_txt, txt_conf = teserac_model.run(image=summ_img_value_, file_name=file_)
            tmp_img = pdfs_content[file_]['summary_images'][summ_img_key_]
            pdfs_content[file_]['summary_images'][summ_img_key_] = {'image':tmp_img, 'txt':img_txt, 'conf':txt_conf}
    readpdf.save_pdf_content(pdf_content=pdfs_content, name_save='pdf_content_imgrecogn')

    #%% Convert the pdf into an Excel
    noxexcel = NoxExcel(pdf_content=pdfs_content, structure=nox_sections, config=config)
    noxexcel.run()

