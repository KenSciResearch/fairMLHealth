# -*- coding: utf-8 -*-
"""
Add-ons for loading data, formatting, and generating tables as part of
KDD 2020 Tutorial on Measuring Fairness for Healthcare.
To be called by Tutorial Notebook.

Contributors:
    camagallen

"""

from IPython.display import display
import numpy as np
import os
import pandas as pd

# Metric libraries
from aif360.sklearn.metrics import *
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, accuracy_score, precision_score)

# Tutorial Libraries
from . import format_mimic_data





'''
    Global variable for backward compatibility with KDD2020 tutorial. Used to
        reduce verbosity of comparison tables.
'''
TUTORIAL_ON = False

def start_tutorial():
    global TUTORIAL_ON
    TUTORIAL_ON = True

def stop_tutorial():
    global TUTORIAL_ON
    TUTORIAL_ON = True

def is_tutorial_running():
    return TUTORIAL_ON


'''
Formatting Helpers
'''
def highlight_col(df, color = 'aquamarine'):
    return f'background-color: {color}'


def highlight_vals(df, values, colname = None, criteria = None,
                                            color = 'magenta', h_type = 'field'):
    """ Returns a list of strings setting the background color at each index of
        df where a[column] is in the list of values

    Args:
        df (pandas df): any dataframe
        values (list-like): values in colname to be highlighted
        colname (str): name of column against which to match values. Defaults
            to None.
        criteria (str): query criteria. may not . Defaults to None.
        color (str): css color name. Defaults to 'aquamarine'.
        h_type (str, optional): [description]. Defaults to 'field'.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if (criteria is not None and values is not None):
        print("Cannot process both crieteria and values. Defaulting to criteria entry")
    if not h_type in ['text', 'field']:
        raise ValueError("Wrong h_type sent")
    if not isinstance(colname, (list, tuple)):
        colname = list(colname)
    if values is None:
        values = []
    if not isinstance(values, (list, tuple)):
        values = list(values)
    #
    if criteria is None:
        criteria = f"in {values}"
    highlight = pd.Series(data = False, index = df.index)
    for col in colname:
        test_vals = values
        if criteria is not None:
            test_vals += df.query(" ".join([col, criteria]))
        highlight[col] = bool(df[col] in values)
    if h_type == 'text':
        return [f'color: {color}'
                    if highlight.any() else '' for v in highlight]
    elif h_type == 'field':
        return [f'background-color: {color}'
                    if highlight.any() else '' for v in highlight]





'''
Loaders and Printers
'''

def load_mimic3_example(mimic_dirpath):
    """ Returns a formatted MIMIC-III data subset for use in KDD Tutorial

        If formatted data file exists, loads that file. Else, generates
        formatted data and saves in mimic_dirpath.

        Args:
            mimic_dirpath (str): valid path to downloaded MIMIC data

        Returns:
            pandas dataframe of formatted MIMIC-III data
    """
    data_file = os.path.join(os.path.expanduser(mimic_dirpath),
                                "kdd_tutorial_data.csv")
    if not os.path.exists(data_file):
        formatter = format_mimic_data.mimic_loader(data_file)
        success = formatter.generate_tutorial_data()
        if not success:
            raise RuntimeError("Error generating tutorial data.")
    else:
        pass
    # Load data and restrict to only age 65+
    df = pd.read_csv(data_file)
    df['HADM_ID'] = df['HADM_ID'] + np.random.randint(10**6)
    df.rename(columns = {'HADM_ID':'ADMIT_ID'}, inplace = True)
    # Ensure that length_of_stay is at the end of the dataframe to reduce
    #   confusion for first-time tutorial users
    df = df.loc[:, [c for c in df.columns
                    if c != 'length_of_stay']+['length_of_stay']]
    return(df)


def print_feature_table(df):
    ''' Displays a table containing statistics on the features available in the
            passed df

        Args:
            df (pandas df): dataframe containing MIMIC data for the tutorial
    '''
    print(f"\n This data subset has {df.shape[0]} total observations" +
            f" and {df.shape[1]-2} input features \n")
    feat_df = pd.DataFrame({'feature':df.columns.tolist()}
                           ).query('feature not in ["ADMIT_ID","length_of_stay"]')
    feat_df['Raw Feature'] = feat_df['feature'].str.split("_").str[0]
    count_df = feat_df.groupby('Raw Feature', as_index = False)['feature'].count(
                ).rename(columns = {'feature':'Category Count (Encoded Features)'})
    display(count_df)


'''
Tutorial-Specific Helpers
'''

def simplify_tutorial_report(comparison_report_df):
    """Updates a fainress comparison report to exlude FairLearn measures. For
        use in the KDD Tutorial, which first introduces AIF360 measures before
        introducing FairLearn

        Args:
            comparison_report_df (pandas df): a fairMLHealth model_comparison
            report

        Returns:
            an updated version of the comparison_report_df
    """
    print("Note: this report has been simplified for this tutorial.",
          "For a more extensive report, omit the simplify_tutorial_report function")
    fl_measures = ["demographic_parity_difference", "demographic_parity_ratio",
                   "equalized_odds_difference", "equalized_odds_ratio"]
    ix_vals = comparison_report_df.index
    ix_vals = [v.replace(" ", "_").lower() for v in ix_vals]
    drop_meas = [ix_vals.index(v) for v in ix_vals if v in fl_measures]
    df = comparison_report_df.drop(drop_meas, axis = 0)
    return(df)

