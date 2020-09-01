"""
    Add-ons for loading data, formatting, and generating tables as part of
    KDD 2020 Tutorial on Measuring Fairness for Healthcare.
    To be called by Tutorial Notebook.

    Author: camagallen
"""

from IPython.display import display
import numpy as np
import os
import pandas as pd

# AIF360 Libraries
from aif360.sklearn.metrics import *
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, accuracy_score, precision_score)

# Tutorial Libraries
import format_mimic_data



'''
Formatting Helpers
'''
class cprint:
    ''' ANSI escape sequences for text hilghting
    '''
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    MAGENTA = '\u001b[35m'
    CYAN = '\u001b[36m'
    BLUE = '\u001b[34m'
    OFF = '\033[0m'


def highlight_col(df, color='aquamarine'):
    return f'background-color: {color}'


def highlight_row(df, colname, values, color='aquamarine', h_type='field'):
    ''' Returns a list of strings setting the background color at each index of
        df where a[column] is in the list of values

        Args:
            df (pandas df): any dataframe
            colname (str): name of column
            values (list-like): values in colname to be highlighted
            color (str): css color name
    '''
    assert h_type in ['text', 'field'], "Wrong h_type sent"
    highlight = pd.Series(data=False, index=df.index)
    highlight[colname] = df[colname] in values
    if h_type == 'text':
        return [f'color: {color}'
                    if highlight.any() else '' for v in highlight]
    elif h_type == 'field':
        return [f'background-color: {color}'
                    if highlight.any() else '' for v in highlight]



'''
Loaders and Printers
'''

def get_aif360_measures_df(X, protected_attr, y_true, y_pred, y_prob=None):
    """ Returns a dataframe containing results for the set of AIF360 measures
        used in the KDD tutorial

        Args:
            X (array-like): Sample features
            protected_attr (array-like, named): values for the protected attribute
                (note: protected attribute may also be present in X)
            y_true (array-like, 1-D): Sample targets
            y_pred (array-like, 1-D): Sample target probabilities
            protected_attr (list): list of column names or locations in
                X containing the protected attribute(s) against which
                fairness is measured
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(protected_attr, np.ndarray) or isinstance(protected_attr, pd.Series):
        protected_attr = pd.DataFrame(protected_attr)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    if isinstance(y_prob, np.ndarray):
        if len(np.shape(y_prob)) == 2:
            y_prob = y_prob[:, 1]
        y_prob = pd.Series(y_prob)
    # Format and set senstitive attributes as index for y dataframes
    pa_name = protected_attr.columns.tolist()
    protected_attr.reset_index(inplace=True, drop=True)
    y_true.reset_index(inplace=True, drop=True)
    y_true = pd.concat([protected_attr, y_true], axis=1).set_index(pa_name)
    y_pred = pd.concat([protected_attr, y_pred], axis=1).set_index(pa_name)
    y_prob = pd.concat([protected_attr,y_prob], axis=1).set_index(pa_name)
    y_pred.columns = y_true.columns
    y_prob.columns = y_true.columns
    # Generate lists of performance measures to be converted to dataframe
    scores = []
    scores.append( ['** Group Measures **', None])
    scores.append( ['Statistical Parity Difference',
                        statistical_parity_difference(y_true, y_pred,
                                    prot_attr=pa_name)] )
    scores.append( ['Disparate Impact Ratio',
                        disparate_impact_ratio(y_true, y_pred,
                                    prot_attr=pa_name)] )
    scores.append( ['Average Odds Difference',
                        average_odds_difference(y_true, y_pred,
                                    prot_attr=pa_name)] )
    scores.append( ['Equal Opportunity Difference',
                        equal_opportunity_difference(y_true, y_pred,
                                    prot_attr=pa_name)] )
    if y_prob is not None:
        scores.append( ['Positive Predictive Parity Difference',
                          difference(precision_score, y_true, y_pred,
                                     prot_attr=pa_name, priv_group=1)] )
        scores.append( ['Between-Group AUC Difference',
                        difference(roc_auc_score, y_true, y_prob,
                                   prot_attr=pa_name, priv_group=1)] )
        scores.append( ['Between-Group Balanced Accuracy Difference',
                        difference(balanced_accuracy_score, y_true, y_pred,
                                   prot_attr=pa_name, priv_group=1)] )
    else:
        pass
    scores.append( ['** Individual Measures **', None])
    scores.append( ['Consistency Score', consistency_score(X, y_pred.iloc[:,0])] )
    scores.append( ['Between-Group Generalized Entropy Error',
                        between_group_generalized_entropy_error(y_true, y_pred,
                                                            prot_attr=pa_name)])
    #
    model_scores =  pd.DataFrame(scores, columns=['Measure','Value'])
    model_scores['Value'] = model_scores.loc[:,'Value'].round(4)
    return(model_scores.fillna(""))


def load_example_data(mimic_dirpath):
    """ Returns a formatted MIMIC-III data subset for use in KDD Tutorial

        If formatted data file exists, loads that file. Else, generates
        formatted data and saves in mimic_dirpath.

        Args:
            mimic_dirpath (str): valid path to downloaded MIMIC data
    """
    data_file = os.path.join(os.path.expanduser(mimic_dirpath), "kdd_tutorial_data.csv")
    if not os.path.exists(data_file):
        formatter = format_mimic_data.mimic_loader(data_file)
        success = formatter.generate_tutorial_data()
        assert success, "Error generating tutorial data."
    else:
        pass
    # Load data and restrict to only age 65+
    df = pd.read_csv(data_file)
    df['HADM_ID'] = df['HADM_ID'] + np.random.randint(10**6)
    df.rename(columns={'HADM_ID':'ADMIT_ID'}, inplace=True)
    # Ensure that length_of_stay is at the end of the dataframe to reduce confusion for
    #   first-time tutorial users
    df = df.loc[:,[c for c in df.columns if c != 'length_of_stay']+['length_of_stay']]
    return(df)


def print_feature_table(df):
    ''' Displays a table containing statistics on the features available in the passed
        df

        Args:
            df (pandas df): dataframe containing MIMIC data for the tutorial
    '''
    print(f"\n This data subset has {df.shape[0]} total observations",
            f"and {df.shape[1]-2} input features \n")
    feat_df = pd.DataFrame({'feature':df.columns.tolist()}
                           ).query('feature not in ["ADMIT_ID","length_of_stay"]')
    feat_df['Raw Feature'] = feat_df['feature'].str.split("_").str[0]
    count_df = feat_df.groupby('Raw Feature', as_index=False)['feature'].count(
                ).rename(columns={'feature':'Category Count (Encoded Features)'})
    display(count_df)
