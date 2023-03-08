Author: Giorgio Ricciardiello
        giorgio.crm6@gmail.com

Project dedicated for reading and parsing pdf Noxturnal reports into Excel tables.

INPUT
Place the pdfs files as input in the folder -> data/input_psg_reports

OUTPUT
The output will be given as a single Excel table containing all the information found on each Noxturnal report,
each row is a different de-identified subject and the columns contain the data present in the Noxturnal report

CONFIGURATION
The confi folder contains the configurations files
    config.py
        contains the paths
    pdf_structure.py
        contains a nested dictionary with all the sections and subsections in the Noxtrunal report. It is important
        that all pdfs follows this structure. If the pdfs are updated with new structures, this nested dictionary
        must be config

PROGRAM
The src folder contains the different classes
    read_pdf
        Class dedicated to extract all the information from the pdf report

    nox_to_excel.py
        Class dedicated for the parsing of extracted and constructed dictionary from the pdf, including the extracted
        imageas text. The Class builds the final dataframe to save all the information available in the NOX report.

    image_classifier_model_tesserac
        The class contains the run method which automatize all the necessary operations to pre-process and extract
        the text from the images of the NOX sleep report. The class runs one patient at the time.

    main.py
        Main script. The pdf is read as a complete string, information is then extracted with the necessary processing
        tools, a dictionary is saved where the extracted information of each subject can be analyses. Lastly, the
        information from the saved dictionary is parsed into an excel file and merged with all the patients pdf's
        inserted in the input fold

