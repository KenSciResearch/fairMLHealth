'''
Add-ons for KDD Tutorial on Fairness and Bias in Healthcare
'''

import numpy as np
import os
import pandas as pd

# AIF360 Libs
from aif360.sklearn.metrics import *

# Tutorial Libs
import format_mimic_data




def get_aif360_measures_df(X_test, y_test, y_pred, sensitive_attributes):
    ''' Returns a dataframe containing results for each AIF360 measure
    '''
    assert isinstance(sensitive_attributes, list) and all([c in X_test.columns for c in sensitive_attributes]), (
        "sensitive_attributes must be list of columns in X_test")
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    #
    y_test_lbl = pd.concat([X_test.loc[:,sensitive_attributes], y_test], axis=1).set_index(sensitive_attributes)
    y_pred_lbl = pd.concat([X_test.loc[:,sensitive_attributes].reset_index(drop=True), y_pred], axis=1).set_index(sensitive_attributes)
    y_pred_lbl.columns = y_test_lbl.columns
    #
    print("base_rate:", round(base_rate(y_test_lbl, y_pred_lbl), 4), "\n")
    scores = [['selection_rate', selection_rate(y_test_lbl, y_pred_lbl)]]
    scores.append(['disparate_impact_ratio', disparate_impact_ratio(y_test_lbl, y_pred_lbl, prot_attr=sensitive_attributes)])
    scores.append(['statistical_parity_difference', statistical_parity_difference(y_test_lbl, y_pred_lbl, prot_attr=sensitive_attributes)])
    scores.append(['average_odds_difference', average_odds_difference(y_test_lbl, y_pred_lbl, prot_attr=sensitive_attributes)])
    scores.append(['average_odds_error', average_odds_error(y_test_lbl, y_pred_lbl, prot_attr=sensitive_attributes)])
    scores.append(['equal_opportunity_difference', equal_opportunity_difference(y_test_lbl, y_pred_lbl, prot_attr=sensitive_attributes)])
    scores.append(['generalized_entropy_error', generalized_entropy_error(y_test.iloc[:,0], y_pred)])
    scores.append(['between_group_generalized_entropy_error', 
                    between_group_generalized_entropy_error(y_test_lbl, y_pred_lbl, prot_attr=sensitive_attributes)] )
    scores.append(['consistency_score', consistency_score(X_test, y_pred)])
    #
    model_scores =  pd.DataFrame(scores, columns=['measure','value'])
    return(model_scores)


def load_example_data(mimic_dirpath):
    """ 
    """
    data_file = os.path.join(os.path.expanduser(mimic_dirpath), "kdd_tutorial_data.csv")
    if not os.path.exists(data_file):
        fmd = format_mimic_data.mimic_loader(data_file)
        success = fmd.generate_tutorial_data()
        assert success, "Error generating tutorial data."
    else:
        pass
    #
    df = pd.read_csv(data_file)
    df['HADM_ID'] = df['HADM_ID'] + np.random.randint(10**6)
    df.rename(columns={'HADM_ID':'ADMIT_ID'}, inplace=True)
    df = df.loc[df['AGE'].ge(65),:]
    return(df)
