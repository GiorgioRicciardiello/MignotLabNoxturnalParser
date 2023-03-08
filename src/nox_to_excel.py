"""
Author: Giorgio Ricciardiello
        giorgio.crm6@gmail.com
Class dedicated for the parsing of extracted and constructed dictionary from the pdf, including the extracted image
as text. The Class builds the final dataframe to save all the information available in the NOX report.

Each member of the class that initiation with the name set_ is dedicated to extract the information from a specific
section of the NOX report. This allows for a modular structures and it is easier to modify.

The class received as input the dictionary containing the extracted data of all the patients. It uses a structure
variable that contains a dictionary format with all the sections and subsections of the NOX report.
"""

import pathlib
import pandas as pd
import pickle
from typing import Optional
from config.config import config
from config.pdf_structure import nox_sections
import re

class NoxExcel():
    def __init__(self, pdf_content: dict, structure: dict, config:dict):
        self.pdf_content = pdf_content
        self.nox_structure = structure
        self.config = config
        # self.patients_frame = pd.DataFrame(-1, columns=self.get_nested_keys(dictio=self.nox_structure),
        #                                    index=[*self.pdf_content.keys()])
        # self.patients_frame = pd.DataFrame(-1, columns=[],
        #                                    index=[*self.pdf_content.keys()])
        self._missing_value = None
        self.patient_df = pd.DataFrame(-1, columns=[],
                                       index=[])
        self.all_patients_frame_build = False

    def run(self):
        """Pipeline to orderly extract the text of each section of the read string of the pdf file"""
        for key_ in self.pdf_content.keys():
            # if not hasattr(self, 'all_patients_frame'):
            #     self.patient_df = pd.DataFrame(-1, columns=[], index=[key_])
            sections_idx = self.search_sections_location(patient_id=key_)
            self.set_patient_information(sections_idx=sections_idx, patient_id=key_)
            self.set_recording_information(sections_idx=sections_idx, patient_id=key_)
            self.set_summary_information(sections_idx=sections_idx, patient_id=key_)
            self.set_respiratory_parameters(sections_idx=sections_idx, patient_id=key_)
            self.set_signal_quality_percentage(sections_idx=sections_idx, patient_id=key_)
            self.set_snore(sections_idx=sections_idx, patient_id=key_)
            self.set_oxygen_saturation(sections_idx=sections_idx, patient_id=key_)
            self.set_position_and_analysis_time(sections_idx=sections_idx, patient_id=key_)
            self.set_pulse(sections_idx=sections_idx, patient_id=key_)
            self.set_cardiac_event(sections_idx=sections_idx, patient_id=key_)
            self.build_all_patients_frame(patient_id=key_)
        self.save_frame_excel()

    def save_frame_excel(self):
        """Save the frame containing all the patients in the output directory as an xlsx file"""
        if hasattr(self, 'all_patients_frame'):
            output_path = pathlib.Path(__file__).parents[1].joinpath(config['output_tables_reports'])
            self.all_patients_frame.to_excel(excel_writer=output_path.joinpath('nox_all_patients.xlsx'))
            if output_path.joinpath('nox_all_patients.xlsx').exists():
                print(f'All patients Excel saved successfully')
            else:
                raise ValueError(f'Unable to save all patients Excel')

    def build_all_patients_frame(self, patient_id:str):
        """
        Because we define all the columns after the first iteration we allocate the complete frame only once at the
        first iteration. This function must be placed once all the columns are automatically define by the code
        """
        if self.all_patients_frame_build:
            return
        else:
            self.all_patients_frame = pd.DataFrame(-1, index=self.pdf_content.keys(),
                                                   columns=[*self.patient_df.keys()])
            # Include the first patient to the all_patients_frame
            self.all_patients_frame.loc[patient_id] = self.patient_df.loc[patient_id]
            # the frame for the single patient will not be use, so we can clear memory and avoid allocation errors
            del self.patient_df
            # bool to run the function only once is set to True
            self.all_patients_frame_build = True

    def search_sections_location(self, patient_id: str) -> dict:
        sections_idx = {}
        idx_ = 0
        for secnum_, section_ in enumerate(self.nox_structure.keys()):
            sections_idx[section_] = []
            sections_idx[section_].append(
                self.pdf_content[patient_id]['file_text'].find(section_,
                                                               idx_))  # start_index_number=0, end_index_number=100)
            idx_ = sections_idx[section_][0]

        # Add the index of the next section to keep a modular work
        values = [idx[0] for idx in [*sections_idx.values()][1::]]
        values.append(len(self.pdf_content[patient_id]['file_text']))
        [sections_idx[keys_].append(val_) for keys_, val_ in zip(sections_idx.keys(), values)]
        return sections_idx

    def set_patient_information(self, sections_idx: dict, patient_id: str, key_: str = 'Patient Information'):
        # key_ = 'Patient Information'  # Section Name
        # Select the text only on the section given by key_
        patient_info_string = self.pdf_content[patient_id]['file_text'][sections_idx[key_][0]:sections_idx[key_][1]]
        # Get the indexes where a new attribute of the section is being defined from the nox_structure
        split_idx = [patient_info_string.index(patinfokeys_) for patinfokeys_ in self.nox_structure[key_].keys()]
        # Get the values of each string
        for i_ in range(0, len(split_idx) - 1):
            value = patient_info_string[split_idx[i_]:split_idx[i_ + 1] - 1].replace(
                [*self.nox_structure[key_].keys()][i_], '')
            value = value.replace(': ', '')
            if value == 'cm' or value == 'kg' or value == '':
                value = self._missing_value  # Unwanted/empty values are set to None
            # Store the values in the final dataframe
            # self.patients_frame.loc[patient_id, [*self.nox_structure[key_].keys()][i_]] = value
            # self.patient_df.loc[patient_id, [*self.nox_structure[key_].keys()][i_]] = value
            self.insert_in_output_frame(patient_id=patient_id, value=value, iterator=i_, nox_section=key_)

    def set_recording_information(self, sections_idx: dict, patient_id: str, key_: str = 'Recording Information'):
        # key_ = 'Recording Information'  # Section Name
        rec_info_string = self.pdf_content[patient_id]['file_text'][sections_idx[key_][0]:sections_idx[key_][1]]
        split_idx = [rec_info_string.index(recinfokeys_) for recinfokeys_ in self.nox_structure[key_].keys()]
        split_idx.append(len(rec_info_string))

        for i_ in range(0, len(split_idx) - 1):
            value = rec_info_string[split_idx[i_]:split_idx[i_ + 1] - 1].replace(
                [*self.nox_structure[key_].keys()][i_], '')
            value = value.replace(': ', '')
            value = value.replace('\n', '')
            if value == '':
                value = self._missing_value
            # self.patients_frame.loc[patient_id, [*self.nox_structure[key_].keys()][i_]] = value
            # self.patient_df.loc[patient_id, [*self.nox_structure[key_].keys()][i_]] = value
            self.insert_in_output_frame(patient_id=patient_id, value=value, iterator=i_, nox_section=key_)

    def set_summary_information(self, sections_idx: dict, patient_id: str, key_: str = 'Summary'):
        """Get the text and image recognition information"""
        # key_ = 'Summary'  # Section Name
        summary_info_string = self.pdf_content[patient_id]['file_text'][sections_idx[key_][0]:sections_idx[key_][1]]
        split_idx = [summary_info_string.index(infokeys_) for infokeys_ in self.nox_structure[key_].keys() if
                     'Est' in infokeys_]
        split_idx.append(summary_info_string.find('%'))
        # text from the summary
        for i_ in range(0, len(split_idx) - 1):
            value = summary_info_string[split_idx[i_]:split_idx[i_ + 1] - 1].replace(
                [*self.nox_structure[key_].keys()][i_], '')
            value = value.replace(': ', '')
            value = value.replace('\n', '')
            if value == '':
                value = self._missing_value
            # self.patients_frame.loc[patient_id, [*self.nox_structure[key_].keys()][i_]] = value
            # self.patient_df.loc[patient_id, [*self.nox_structure[key_].keys()][i_]] = value
            self.insert_in_output_frame(patient_id=patient_id, value=value, iterator=i_, nox_section=key_)

        # the key = AHI, ODI and Snore obtained from the image recognition class
        for key_, measure_ in self.pdf_content[patient_id]['summary_images'].items():
            # self.patient_df.loc[patient_id, key_] = measure_['txt']
            self.insert_in_output_frame(patient_id=patient_id, value=measure_['txt'], iterator=None,
                                        nox_section=key_)
            # include the confidence column beside the measure
            # self.patients_frame.insert(loc=self.patients_frame.columns.get_loc(key_)+1, column=key_+'_conf',
            #                            value=-1)
            # insert the confidence value
            # self.patient_df.loc[patient_id,key_+'_conf'] = measure_['conf']
            self.insert_in_output_frame(patient_id=patient_id, value=measure_['conf'], iterator=None,
                                        nox_section=key_ + '_conf')

    def set_respiratory_parameters(self, sections_idx: dict, patient_id: str, key_: str = 'Respiratory Parameters'):
        # key_ = 'Respiratory Parameters'
        resp_info_string = self.pdf_content[patient_id]['file_text'][sections_idx[key_][0]:sections_idx[key_][1]]
        resp_info_string = resp_info_string.split('\n')[1:-1]

        resp_dict = {}
        for row_ in resp_info_string:
            # row_ = resp_info_string[0]
            print(row_)
            # row_ = r'Respiration Rate (per m): 11.1 /m 10.7 /m 11.3 /m'
            if ':' in row_:
                row_name, values = row_.split(':')
            else:
                row_name, values = row_.split(')')
            if '/h' in values:
                values = values.split('/h')
            elif '/m' in values:
                values = values.split('/m')
            values = [float(val_.replace(' ', '')) for val_ in values if val_ != '']  # Total, Supine, Non-Supine, Count
            resp_dict[row_name] = values
        resp_dict = self._unwragnle_lists_dictionary(section_dictionary=resp_dict,
                                                     table_columns=['Total', 'Supine', 'Non-Supine', 'Count'])
        resp_df = pd.DataFrame(resp_dict, index=[patient_id])
        # self.patient_df = self.patient_df.join(resp_df)
        self.insert_in_output_frame(value=resp_df, patient_id=patient_id, nox_row_col=[*resp_dict.keys()], iterator=None)

    def set_signal_quality_percentage(self,sections_idx: dict, patient_id: str, key_: str = 'Respiratory Parameters'):
        signq_info_string = self.pdf_content[patient_id]['file_text'][sections_idx[key_][0]:sections_idx[key_][1]]
        signq_info_string = signq_info_string.split('%')[0:-1]

        for row_ in signq_info_string:
            if '\n' in row_:
                row_ = row_.split('\n')[1]
            row_name, value = row_.split(':')
            row_name = row_name.lstrip()  # remove leading space
            value = float(value.replace(' ', ''))
            self.insert_in_output_frame(value=value, patient_id=patient_id, nox_section=row_name,
                                        iterator=None)

    def set_snore(self,sections_idx: dict, patient_id: str, key_: str = 'Snore Total'):
        snore_info_string = self.pdf_content[patient_id]['file_text'][sections_idx[key_][0]:sections_idx[key_][1]]
        snore_info_string = snore_info_string[snore_info_string.find('\n')::]
        snore_info_string = snore_info_string.split('\n')[1:-1]
        snore_dict = {}

        for row_ in snore_info_string:
            # row_ = resp_info_string[0]
            print(row_)
            # row_ = r'Respiration Rate (per m): 11.1 /m 10.7 /m 11.3 /m'
            row_name, values = row_.split(':')
            values = values.split('%')
            values[-1] = values[-1].replace(' m', '')
            values = [float(val_.replace(' ', '')) for val_ in values if val_ != '']  # Total, Supine, Non-Supine, Count
            snore_dict[row_name] = values
        resp_dict = self._unwragnle_lists_dictionary(section_dictionary=snore_dict,
                                                     table_columns=['Total', 'Supine', 'Non-Supine', 'Duration'])
        resp_df = pd.DataFrame(resp_dict, index=[patient_id])
        # self.patient_df = self.patient_df.join(resp_df)
        self.insert_in_output_frame(value=resp_df, patient_id=patient_id, nox_row_col=[*resp_dict.keys()], iterator=None)

    def set_oxygen_saturation(self,sections_idx: dict, patient_id: str, key_: str = 'Oxygen Saturation (SpO2)'):
        oxysat_info_string = self.pdf_content[patient_id]['file_text'][sections_idx[key_][0]:sections_idx[key_][1]]
        oxysat_info_string = oxysat_info_string.split('\n')[1:-1]
        oxysat_dict = {}
        for row_ in oxysat_info_string:
            if ':' in row_:
                row_name, values = row_.split(':')
            elif 'Duration' in row_:
                values = row_[row_.find('%')+1::]
                if 'm' in values:
                    values = values.replace('m', '')
                row_name = row_[0:row_.find('%')]
            if '/h' in values:
                values = values.split('/h')
            elif '%' in values:
                values = values.split('%')
            values = [float(val_.strip()) for val_ in values if val_ != '']  # Total, Supine, Non-Supine, Count
            oxysat_dict[row_name] = values
        resp_dict = self._unwragnle_lists_dictionary(section_dictionary=oxysat_dict,
                                                     table_columns=['Total', 'Supine', 'Non-Supine', 'Duration'])
        resp_df = pd.DataFrame(resp_dict, index=[patient_id])
        # self.patient_df = self.patient_df.join(resp_df)
        self.insert_in_output_frame(value=resp_df, patient_id=patient_id, nox_row_col=[*resp_dict.keys()], iterator=None)

    def set_position_and_analysis_time(self,sections_idx: dict, patient_id: str, key_: str = 'Position & Analysis Time'):
        # key_ = 'Position & Analysis Time'
        posanly = self.pdf_content[patient_id]['file_text'][sections_idx[key_][0]:sections_idx[key_][1]]
        # posanly = posanly.split('\n')[1:-1]
        # Separate the string using the nox_structure, first we identify the indexes of each subsection
        sub_sec_idx = [posanly.find(subsec_) for subsec_, _ in self.nox_structure[key_].items()]
        sub_sec_idx.append(len(posanly))
        # Utilize the subsection indexes to split the string and extract the rwo_name and value
        posanly_dict = {}
        for i_ in range(0, len(sub_sec_idx) - 1):
            member = posanly[sub_sec_idx[i_]:sub_sec_idx[i_+1]]
            row_name, values = member.split(':')
            values = values.split('m')
            values[-1] = values[-1].replace('%', '')
            if '\n' in values[-1]:
                values[-1] = values[-1].replace('\n', '')
            values = [float(val_) for val_ in values]
            posanly_dict[row_name] = values

        posanly_dict = self._unwragnle_lists_dictionary(section_dictionary=posanly_dict,
                                                     table_columns=['Duration', 'Percentage'])
        posanly_df = pd.DataFrame(posanly_dict, index=[patient_id])
        # self.patient_df = self.patient_df.join(resp_df)
        self.insert_in_output_frame(value=posanly_df, patient_id=patient_id, nox_row_col=[*posanly_dict.keys()], iterator=None)

    def set_pulse(self,sections_idx: dict, patient_id: str, key_: str = 'Pulse'):
        # key_ = 'Pulse'
        pulse_info_string = self.pdf_content[patient_id]['file_text'][sections_idx[key_][0]:sections_idx[key_][1]]
        # posanly = posanly.split('\n')[1:-1]
        # Separate the string using the nox_structure, first we identify the indexes of each subsection
        sub_sec_idx = [pulse_info_string.index(subsec_) for subsec_, _ in self.nox_structure[key_].items()]
        sub_sec_idx.append(len(pulse_info_string))
        sub_sec_idx.sort()
        # Utilize the subsection indexes to split the string and extract the rwo_name and value
        pulse_dict = {}
        for i_ in range(0, len(sub_sec_idx) - 1):
            member = pulse_info_string[sub_sec_idx[i_]:sub_sec_idx[i_ + 1]]
            row_name, value = member.split(':')
            value = float(value)
            pulse_dict[row_name] = value

        pulse_df = pd.DataFrame(pulse_dict, index=[patient_id])
        # self.patient_df = self.patient_df.join(resp_df)
        self.insert_in_output_frame(value=pulse_df, patient_id=patient_id, nox_row_col=[*pulse_dict.keys()],
                                    iterator=None)

    def set_cardiac_event(self,sections_idx: dict, patient_id: str, key_: str = 'Cardiac Events'):
        # key_ = 'Cardiac Events'
        cardiac_info_string = self.pdf_content[patient_id]['file_text'][sections_idx[key_][0]:sections_idx[key_][1]]
        sub_sec_idx = [cardiac_info_string.index(subsec_) for subsec_, _ in self.nox_structure[key_].items()]
        sub_sec_idx.append(len(cardiac_info_string))
        sub_sec_idx.sort()

        # Utilize the subsection indexes to split the string and extract the rwo_name and value
        cardiac_dict = {}
        for i_ in range(0, len(sub_sec_idx) - 1):
            member = cardiac_info_string[sub_sec_idx[i_]:sub_sec_idx[i_ + 1]]
            first_int_idx = re.search(r"\d", member).start()  # Find the index of the first digit in a string
            row_name, values = member[0:first_int_idx-1], member[first_int_idx::]
            values = values.split('/h')
            values = [float(val_) for val_ in values]
            cardiac_dict[row_name] = values

        posanly_dict = self._unwragnle_lists_dictionary(section_dictionary=cardiac_dict,
                                                        table_columns=['Index', 'Count'])
        posanly_df = pd.DataFrame(posanly_dict, index=[patient_id])
        self.insert_in_output_frame(value=posanly_df, patient_id=patient_id, nox_row_col=[*posanly_dict.keys()],
                                    iterator=None)

    def get_nested_keys(self, dictio: dict) -> list:
        """Get all they keys from the structure dictionary. Keys will become columns for the final frame"""
        keys = []
        for (key, value) in dictio.items():
            if isinstance(value, dict):
                keys.extend(self.get_nested_keys(value))
            else:
                keys.append(key)
        return keys

    @staticmethod
    def _unwragnle_lists_dictionary(section_dictionary: dict, table_columns: list) -> dict:
        """
        Sections with rows containing multiple columns stored as a single list. This function generates a single key
        element for each list value, while naming which column the value is from
        :param section_dictionary: keys=row, value=list of elements (columns in the table)
        :param table_columns: columns table names IN ORDER AS IN THE ODF
        :return: key=row+colum name, value:single value of each row col pair
        """
        new_resp_dict = {}
        for key_ in section_dictionary.keys():
            for list_val_, type_ in zip(section_dictionary[key_], table_columns):
                new_resp_dict[key_ + '_' + type_] = list_val_
        # order alphabetical based on key
        # new_resp_dict = {key: value for key, value in sorted(new_resp_dict.items())}

        return new_resp_dict

    def insert_in_output_frame(self, value, patient_id: str, nox_section: Optional[str] = None,
                               iterator: Optional[int] = None, nox_row_col:list=None):
        """
        Function dedicated to store all the values of each section in the same dataframe and then in the complete
        pre-allocated frame. Helps for memmory effiency
        """
        if hasattr(self, 'all_patients_frame'):
            # Using the complete frame with all the rows
            if isinstance(value, pd.DataFrame):
                # self.all_patients_frame.loc[patient_id, ] = value.loc[patient_id]
                self.all_patients_frame.loc[patient_id, nox_row_col] = value.loc[patient_id]
                return
            else:
                if iterator is None:
                    self.all_patients_frame.loc[patient_id, nox_section] = value
                    return
                else:
                    # passing a single value
                    self.all_patients_frame.loc[patient_id, [*self.nox_structure[nox_section].keys()][iterator]] = value
                    return

        else:
            # First iteration when the all_patients_frame has not been jet defined
            if isinstance(value, pd.DataFrame):
                self.patient_df = self.patient_df.join(value)
                return
            if nox_section in self.nox_structure.keys() and not iterator is None:
                self.patient_df.loc[patient_id, [*self.nox_structure[nox_section].keys()][iterator]] = value
                return
            else:
                self.patient_df.loc[patient_id, nox_section] = value
                return



if __name__ == "__main__":

    input_path = pathlib.Path(__file__).parents[1].joinpath(config['extracted_dict_path'])
    if input_path.joinpath('pdf_content.pickle').exists():
        with open(input_path.joinpath('pdf_content_imgrecogn.pickle'), 'rb') as handle:
            # self.pdf_content = pickle.load(handle)
            pdfs_content = pickle.load(handle)
    else:
        raise ValueError(f'Unable to locate extracted dictionary from pdf on \n {input_path}')

    noxexcel = NoxExcel(pdf_content=pdfs_content, structure=nox_sections, config=config)
    noxexcel.run()
