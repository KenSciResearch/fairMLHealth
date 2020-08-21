"""
    Add-ons for loading data and generating tables as part of KDD 2020 Tutorial on
    Measuring Fairness for Healthcare.
    To be called by Tutorial Notebook. 
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




def get_aif360_measures_df(X_test, y_test, y_pred, y_prob=None, sensitive_attr=None):
    """ Returns a dataframe containing results for the set of AIF360 measures
        used in the KDD tutorial

        Args:
            X_test (array-like): Sample features; must include sensitive attribute
            y_test (array-like, 1-D): Sample targets
            y_pred (array-like, 1-D): Sample target probabilities
            sensitive_attr (list): list of column names or locations in
                X_test containing the sensitive attribute(s) against which
                fairness is measured
    """
    assert isinstance(sensitive_attr, list) and all(
        [c in X_test.columns for c in sensitive_attr]), (
            "sensitive_attr must be list of columns in X_test")
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    if isinstance(y_prob, np.ndarray):
        if len(np.shape(y_prob)) == 2:
            y_prob = y_prob[:, 1]
        y_prob = pd.Series(y_prob)
    # Set senstitive attributes as index for y dataframes
    y_test = pd.concat([X_test.loc[:,sensitive_attr], y_test],
                           axis=1).set_index(sensitive_attr)
    y_pred = pd.concat([X_test.loc[:,sensitive_attr].reset_index(drop=True),
                            y_pred], axis=1).set_index(sensitive_attr)
    y_prob = pd.concat([X_test.loc[:,sensitive_attr].reset_index(drop=True),
                            y_prob], axis=1).set_index(sensitive_attr)
    y_pred.columns = y_test.columns
    y_prob.columns = y_test.columns
    # Generate lists of performance measures to be converted to dataframe
    scores = [["* General Performance Measures *", None],]
    scores.append( ['Selection Rate', selection_rate(y_test, y_pred)] )
    if y_prob is not None:
        scores.append( ['ROC Score', roc_auc_score(y_test, y_prob) ])
        scores.append( ['Accuracy Score', accuracy_score(y_test, y_pred) ])
        scores.append( ['Precision Score', precision_score(y_test, y_pred) ])
    else: 
        pass
    # Add spacer to metalist to separate general measures from group-specific ones
    scores.append( ["* Fairness Measures *", None]) 
    scores.append( ['Statistical Parity Difference',
                        statistical_parity_difference(y_test, y_pred,
                                    prot_attr=sensitive_attr)] )
    scores.append( ['Disparate Impact Ratio',
                        disparate_impact_ratio(y_test, y_pred,
                                    prot_attr=sensitive_attr)] )
    scores.append( ['Average Odds Difference',
                        average_odds_difference(y_test, y_pred,  
                                    prot_attr=sensitive_attr)] )
    scores.append( ['Average Odds Error',
                        average_odds_error(y_test, y_pred,
                                     prot_attr=sensitive_attr)] )
    scores.append( ['Equal Opportunity Difference',
                        equal_opportunity_difference(y_test, y_pred,
                                    prot_attr=sensitive_attr)] )
    if y_prob is not None:
        scores.append( ['Positive Predictive Parity Difference',
                          difference(precision_score, y_test, y_pred, 
                                     prot_attr=sensitive_attr, priv_group=1)] )
        scores.append( ['Between-Group AUC Difference',
                        difference(roc_auc_score, y_test, y_prob,
                                   prot_attr=sensitive_attr, priv_group=1)] )
        scores.append( ['Between-Group Balanced Accuracy Difference',
                        difference(balanced_accuracy_score, y_test, y_pred,
                                   prot_attr=sensitive_attr, priv_group=1)] )
    else:
        pass
    scores.append( ['Consistency Score', consistency_score(X_test, y_pred.iloc[:,0])] )
    scores.append( ['Generalized Entropy Error',
                        generalized_entropy_error(y_test.iloc[:,0], y_pred.iloc[:,0])] )
    scores.append( ['Between-Group Generalized Entropy Error',
                        between_group_generalized_entropy_error(y_test, y_pred,
                                    prot_attr=sensitive_attr)] )
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
    df = df.loc[df['AGE'].ge(65),:]
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
            "and {df.shape[1]-2} input features \n")
    feat_df = pd.DataFrame({'feature':df.columns.tolist()}
                           ).query('feature not in ["ADMIT_ID","length_of_stay"]')
    feat_df['Raw Feature'] = feat_df['feature'].str.split("_").str[0]
    count_df = feat_df.groupby('Raw Feature', as_index=False)['feature'].count(
                ).rename(columns={'feature':'Category Count (Encoded Features)'})
    display(count_df)

