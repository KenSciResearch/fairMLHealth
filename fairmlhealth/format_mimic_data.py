# -*- coding: utf-8 -*-
"""
Formats MIMIC-III data for use with the KDD 2020 Tutorial on Measuring Fairness
for Healthcare.
To be called by Tutorial Notebook.

Contributors:
    camagallen <christine.allen@kensci.com>
"""
# Copyright (c) KenSci and contributors.
# Licensed under the MIT License.

import io
import os
import pandas as pd
import requests
from sklearn import preprocessing


def cb_round(col, base=5, sig_dec=0):
    """ Returns the column with values rounded per the custom base value

        Args:
            col (str): name of column to round
            base (float): base value to which data should be rounded (may be
                decimal)
            sig_dec (int): number of significant decimals for the custom-rounded
                value
    """
    if not base >= 0.01:
        raise ValueError(f"cannot round with base {base}."
                         + "cb_round designed for base >= 0.01."
                         )
    result = col.apply(lambda x: round(base * round(float(x)/base), sig_dec))
    return result


def load_icd_ccs_xwalk(code_type):
    """ Returns dataframe containing an ICD-9 to CCS crosswalk

        Args:
            code_type (str): key for file location (one of 'dx' or 'px')
    """
    urls = {'dx': "https://data.nber.org/ahrq/ccs/dxicd2ccsxw.csv",
            'px': "https://data.nber.org/ahrq/ccs/pricd2ccsxw.csv"
            }
    if code_type not in urls.keys():
        raise KeyError(f"Invalid code_type. Must be one of {urls.keys()}")
    s = requests.get(urls[code_type]).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    df = df.loc[:, ['icd', 'ccs', 'icddesc']]
    df.rename(columns={'icd': 'ICD9_CODE',
                       'ccs': f'{code_type}_CCS',
                       'icddesc': f'{code_type}_ICD_DESC'},
              inplace=True
              )
    return(df)


class mimic_loader():
    """ Generates a MIMIC-III data subset formatted for use with the KDD
            Tutorial
    """
    def __init__(self, data_file=""):
        self.output_file = os.path.expanduser(data_file)
        mimic_dir = os.path.dirname(self.output_file)
        if mimic_dir != "":
            if not os.path.exists(mimic_dir):
                raise OSError("Cannot load example data. " +
                f"Invalid mimic data directory passed: {mimic_dir}")
        self.data_dir = os.path.join(mimic_dir, 'zipped_files')
        if not os.path.exists(self.data_dir):
            raise OSError("MIMIC directory must contain the folder"
                          + " \'zipped_files\', which should be present in the"
                          + " raw download")

    def generate_tutorial_data(self):
        print("Generating Example Dataset from MIMIC-III Download...")
        adm_data = self.load_admit_dscg_data()
        dx_data = self.load_dxpx_data('dx')
        px_data = self.load_dxpx_data('px')
        df = adm_data.merge(dx_data, on='HADM_ID', how='inner'
                    ).merge(px_data, on='HADM_ID', how='inner')
        # Test dataset before saving
        if df['HADM_ID'].isnull().any():
            raise ValueError("Missing admission information in MIMIC data")
        if df['HADM_ID'].duplicated().any():
            raise ValueError("Duplicates found in MIMIC data")
        df.to_csv(self.output_file, index=False)
        return True

    def __load_mimic_data(self, data_key):
        """ Returns transfer data as pd dataframe after removing the ROW_ID
                column (unnecessary column that causes problems)

            :arg data_key : key for the filename of interest (see
                get_file_dict())
            :arg data_dir : the directory in which the file is stored
                (default is get_data_dir() )
        """
        file_dict = {'dx': "DIAGNOSES_ICD",
                     'ax': "ADMISSIONS",
                     'px': "PROCEDURES_ICD",
                     'rx': "PRESCRIPTIONS",
                     'pt': "PATIENTS"
                     }
        data_file = f"{self.data_dir}/{file_dict[data_key]}.csv.gz"
        df = pd.read_csv(data_file, low_memory=False)
        if 'ROW_ID' in df.columns:
            del df['ROW_ID']
        return df

    def load_admit_dscg_data(self):
        """ Loads and returns a dataframe of formatted demographic
            and length-of-stay data available through admission and
            discharge tables

            Note: drops data for patients with  age>120 y.o. or age<0
        """
        admissions = self.__load_mimic_data("ax")
        if admissions['HADM_ID'].duplicated().any():
            raise ValueError(
            "Error loading admission data: duplicate admission IDs present")
        # Calculate AGE
        adm = admissions.groupby(['SUBJECT_ID', 'HADM_ID'], as_index=False
                                 )['ADMITTIME'].min()
        dob = self.__load_mimic_data('pt')[['SUBJECT_ID', 'DOB', 'GENDER']]
        age_df = dob.merge(adm, on='SUBJECT_ID')
        age_df['DOB'] = pd.to_datetime(age_df['DOB']).dt.date
        age_df['ADMITTIME'] = pd.to_datetime(age_df['ADMITTIME']).dt.date
        age_df['AGE'] = age_df.apply(lambda e:
                                     (e['ADMITTIME'] - e['DOB']).days/365,
                                     axis=1
                                     )
        age_df.loc[age_df['AGE'].ge(5), 'AGE'] = cb_round(age_df['AGE'], base=5)
        age_df.loc[age_df['AGE'].ge(1) &
                   age_df['AGE'].lt(5), 'AGE'
                   ] = age_df['AGE'].round()
        age_df.loc[age_df['AGE'].lt(1), 'AGE'] = 0
        age_df = age_df.loc[age_df['AGE'].le(120) & age_df['AGE'].ge(0), :]
        # Attach AGE to admission information
        adm_data = admissions.merge(age_df, how='inner', on='HADM_ID')
        adm_data = adm_data[['HADM_ID', 'AGE', 'GENDER', 'INSURANCE',
                             'MARITAL_STATUS', 'ETHNICITY', 'LANGUAGE',
                             'RELIGION']
                            ]
        # Calculate and Attach Length of Stay
        ax = admissions[['HADM_ID', 'ADMITTIME', 'DISCHTIME']
                        ].sort_values(by=['HADM_ID', 'ADMITTIME']
                        ).drop_duplicates()
        ax['ADMITTIME'] = pd.to_datetime(ax['ADMITTIME'])
        ax['DISCHTIME'] = pd.to_datetime(ax['DISCHTIME'])
        ax['length_of_stay'] = \
            (ax['ADMITTIME'] - ax['DISCHTIME'])/pd.offsets.Day(-1)
        ax = ax.loc[ax['length_of_stay'].ge(0) & ax['length_of_stay'].lt(30), :]
        adm_data = adm_data.merge(ax[['HADM_ID', 'length_of_stay']], how='inner')
        # Test result before proceeding
        if not adm_data.notnull().any().any():
            raise ValueError("Missing hospital admission information after merge")
        if adm_data['HADM_ID'].duplicated().any():
            raise ValueError("Duplicate hospital admission information after merge")
        # Reformat data to one-hot encode
        adm_data.rename(columns={'MARITAL_STATUS':'MARRIED'}, inplace=True)
        dummy_cats = ['GENDER', 'ETHNICITY', 'LANGUAGE', 'INSURANCE',
                      'MARRIED', 'RELIGION']
        ohe_df = pd.get_dummies(adm_data[dummy_cats], columns=dummy_cats,
                                prefix_sep='_')
        id_df = adm_data[['HADM_ID', 'AGE', 'length_of_stay']]
        output = id_df.join(ohe_df)
        if output[f'HADM_ID'].isnull().any():
            raise ValueError("Missing hospital admission information in output")
        if output[f'HADM_ID'].duplicated().any():
            raise ValueError("Duplicate hospital admission information in output")
        return output

    def load_dxpx_data(self, feature_type):
        """ Returns a pandas dataframe with formatted diagnosis or procedure
                data, categorized in single-level CCS codes and one-hot encoded

            Args:
                feature_type (str): one of ['dx', 'px'] (i.e. ["diagnosis",
                "procedure"])
        """
        ftypes = ['dx', 'px']
        if feature_type not in ftypes:
            raise ValueError(f"Invalid code_type. Must be one of {ftypes}")
        #
        data = self.__load_mimic_data(feature_type)
        icd_map = load_icd_ccs_xwalk(code_type=feature_type
                                     )[['ICD9_CODE', f'{feature_type}_CCS']]
        df = data.merge(icd_map
                        ).drop('SEQ_NUM', axis=1
                        ).drop_duplicates(
                        ).rename(columns={'ICD9_CODE': f'{feature_type}_ICD9_CODE'})
        # Test result before proceeding
        if df[f'{feature_type}_CCS'].isnull().any():
            raise ValueError("Missing CCS information")
        if df.duplicated().any().any():
            raise ValueError("Duplicate CCS Information found")
        # Reformat data to one-hot encode
        prefix_dict = {'dx': 'DIAGNOSIS', 'px': 'PROCEDURE'}
        ohe_df = pd.get_dummies(df[f'{feature_type}_CCS'],
                                prefix=f'{prefix_dict[feature_type]}_CCS')
        ohe_df['HADM_ID'] = df['HADM_ID']
        agg_dict = {c: 'max' for c in ohe_df.columns if c != 'HADM_ID'}
        output = ohe_df.groupby('HADM_ID', as_index=False).agg(agg_dict)

        #
        fm_err = f"Error loading {feature_type}." +\
                 " Missing admissions found in formatting"
        if output[f'HADM_ID'].isnull().any():
            raise ValueError(fm_err)
        if output[f'HADM_ID'].duplicated().any():
            raise ValueError(fm_err)
        return output


if __name__ == "__main__":
    fmd = mimic_loader("~/data/MIMIC/test_data.csv")
    fmd.generate_tutorial_data()
