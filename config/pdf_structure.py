"""
Author: Giorgio Ricciardiello
        giorgio.crm6@gmail.com
Nox report structure as a dictionary, with sections and subsections
"""
nox_sections = {'Patient Information':
                    {'Full Name': '',
                     'Patient ID': '',
                     'Height': '',
                     'Weight': '',
                     'BMI': '',
                     'Date of Birth': '',
                     'Age': '',
                     'Gender': '',
                     },
                'Recording Information': {
                    'Recording Date': '',
                    'Analysis Duration (TRT)': '',
                    'Recording Tags': '',
                    'Analysis Start Time (Lights out)': '',
                    'Device Type': '',
                    'Analysis Stop Time (Lights on)': '',
                },
                'Summary': {
                    'Est Total Sleep Time (TST)': '',
                    'Est. Sleep Efficiency': '',
                    'AHI': '',
                    'ODI': '',
                    'Snore': '',
                },
                'Respiratory Parameters': {
                    'Apneas + Hypopneas (AH)': '',
                    'Apneas': '',
                    'Obstructive (OA)': '',
                    'Mixed (MA)': '',
                    'Central (CA)': '',
                    'Hypopneas': '',
                    'Obstructive (OH)': '',
                    'Central (CH)': '',
                    'Obstructive Apnea Hypopnea (OA + MA + OH)': '',
                    'Central Apnea Hypopnea (CA + CH)': '',
                    'RDI (A+H+RERAs)': '',
                    'Hypoventilation': '',
                    'Respiration Rate (per m)': '',
                },
                'Signal Quality Percentage': {
                    'Oximeter': '',
                    'Abdomen RIP': '',
                    'Nasal Cannula': '',
                    'Thorax RIP': '',
                },
                'Snore Total': {
                    'Snore': '',
                    'Flow Limitation': '',
                    'Cheyne-Stokes Breathing': '',
                    'Paradoxical Breathing': ''
                },
                'Oxygen Saturation (SpO2)': {
                    'Oxygen Desaturation Index (ODI)': '',
                    'Average SpO2': '',
                    'Minimum SpO2': '',
                    'SpO2 Duration < 90%': '',
                    'SpO2 Duration ≤ 88%': '',
                    'SpO2 Duration < 85%': '',
                    'Average Desat Drop': '',
                },
                'Position & Analysis Time': {
                    'Supine (in TST)': '',
                    'Movement (in TST)':'',
                    'Non-Supine (in TST)': '',
                    'Invalid Data (Excluded)':'',
                    'Left (in TST)': '',
                    'Prone (in TST)': '',
                    'Right (in TST)': '',
                    'Unknown (in TST)': '',
                    'Upright (in TRT)': '',
                },
                'Pulse': {
                    'Average (in TST)': '',
                    'Max (in TST)': '',
                    'Max (in TRT)': '',
                    'Min (in TST)': '',
                    'Duration < 40 bpm': '',
                    'Duration > 100 bpm': '',
                    'Duration > 90 bpm': '',
                },
                'Cardiac Events': {
                    'Bradycardia': '',
                    'Tachycardia': '',
                    'Asystole': '',
                    'Atrial Fibrillation': ''
                }
}
