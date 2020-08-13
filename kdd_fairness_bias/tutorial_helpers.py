"""
    Add-ons for KDD 2020 Tutorial on Fairness and Bias in Healthcare
"""

from IPython.display import display
import numpy as np
import os
import pandas as pd

# AIF360 Libraries
from aif360.sklearn.metrics import *
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score

# Tutorial Libraries
import format_mimic_data




def get_aif360_measures_df(X_test, y_test, y_pred, y_prob=None, sensitive_attributes=None):
    """ Returns a dataframe containing results for the set of AIF360 measures
        used in the KDD tutorial

        Args:
            X_test (array-like): Sample features; must include sensitive attribute
            y_test (array-like, 1-D): Sample targets
            y_pred (array-like, 1-D): Sample target probabilities
            sensitive_attributes (list): list of column names or locations in
                X_test containing the sensitive attribute(s) against which
                fairness is measured
    """
    assert isinstance(sensitive_attributes, list) and all(
        [c in X_test.columns for c in sensitive_attributes]), (
            "sensitive_attributes must be list of columns in X_test")
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    if isinstance(y_prob, np.ndarray):
        if len(np.shape(y_prob)) == 2:
            y_prob = y_prob[:, 1]
        y_prob = pd.Series(y_prob)
    #
    y_test = pd.concat([X_test.loc[:,sensitive_attributes], y_test],
                           axis=1).set_index(sensitive_attributes)
    y_pred = pd.concat([X_test.loc[:,sensitive_attributes].reset_index(drop=True),
                            y_pred], axis=1).set_index(sensitive_attributes)
    y_prob = pd.concat([X_test.loc[:,sensitive_attributes].reset_index(drop=True),
                            y_prob], axis=1).set_index(sensitive_attributes)
    y_pred.columns = y_test.columns
    y_prob.columns = y_test.columns
    #
    scores = [ ['selection_rate', selection_rate(y_test, y_pred)] ]
    scores.append( ['disparate_impact_ratio',
                        disparate_impact_ratio(y_test, y_pred,
                                    prot_attr=sensitive_attributes)] )
    scores.append( ['statistical_parity_difference',
                        statistical_parity_difference(y_test, y_pred,
                                    prot_attr=sensitive_attributes)] )
    scores.append( ['average_odds_difference',
                        average_odds_difference(y_test, y_pred,
                                    prot_attr=sensitive_attributes)] )
    scores.append( ['average_odds_error',
                        average_odds_error(y_test, y_pred,
                                     prot_attr=sensitive_attributes)] )
    scores.append( ['equal_opportunity_difference',
                        equal_opportunity_difference(y_test, y_pred,
                                    prot_attr=sensitive_attributes)] )
    scores.append( ['generalized_entropy_error',
                        generalized_entropy_error(y_test.iloc[:,0], y_pred.iloc[:,0])] )
    scores.append( ['between_group_generalized_entropy_error',
                        between_group_generalized_entropy_error(y_test, y_pred,
                                    prot_attr=sensitive_attributes)] )
    scores.append( ['consistency_score', consistency_score(X_test, y_pred.iloc[:,0])] )
    if y_prob is not None:
        scores.append( ['Between-Group AUC Difference',
                        difference(roc_auc_score, y_test, y_prob,
                                   prot_attr=sensitive_attributes, priv_group=1)] )
        scores.append( ['Between-Group Balanced Accuracy Difference',
                        difference(balanced_accuracy_score, y_test, y_pred,
                                   prot_attr=sensitive_attributes, priv_group=1)] )
        scores.append( ['ROC Score', roc_auc_score(y_test, y_prob) ])
        scores.append( ['Accuracy Score', accuracy_score(y_test, y_pred) ])
    #
    model_scores =  pd.DataFrame(scores, columns=['measure','value'])
    return(model_scores)




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
    #
    df = pd.read_csv(data_file)
    df['HADM_ID'] = df['HADM_ID'] + np.random.randint(10**6)
    df.rename(columns={'HADM_ID':'ADMIT_ID'}, inplace=True)
    df = df.loc[df['AGE'].ge(65),:]
    return(df)


def print_table_of_measures():
    ''' Displays a table comparing the measures available in AIF360 vs FairLearn
    '''
    tb_dict = {
        ("General Measures","Base Rate"):["Y", "-"],
        ("General Measures","Selection Rate"):["Y","Y"],

        ("Group Fairness Measures","Demographic (Statistical) Parity Difference"):["Y","Y"],
        ("Group Fairness Measures","Disparate Impact Ratio (Demographic Parity Ratio)"):["Y","Y"],
        ("Group Fairness Measures",
          "Generalized Between-Group Predictive Disparity (eg. difference in ROC)"):["Y","Y"],
        ("Group Fairness Measures","Average Odds Difference"):["Y","-"],
        ("Group Fairness Measures","Average Odds Error"):["Y","-"],
        ("Group Fairness Measures","Equalized Odds Difference"):["-","Y"],
        ("Group Fairness Measures","Equalized Odds Ratio"):["-","Y"],

        ("Individual Fairness Measures","Between-Group Generalized Entropy Error"):["Y","-"],
        ("Individual Fairness Measures","Generalized Entropy Index"):["Y","-"],
        ("Individual Fairness Measures","Generalized Entropy Error"):["Y","-"],
        ("Individual Fairness Measures","Coefficient of Variation"):["Y","-"],
        ("Individual Fairness Measures","Consistency Score"):["Y","-"]
    }
    display(pd.DataFrame(tb_dict, index=["AIF360","FairLearn"]).transpose())


def print_table_of_algorithms():
    ''' Displays a table comparing the corrective algorithms available in
        AIF360 vs FairLearn
    '''
    tb_dict = {
        "Optimized Preprocessing (Calmon et al., 2017)":["Y","-"],
        "Disparate Impact Remover (Feldman et al., 2015)":["Y","-"],
        "Equalized Odds Postprocessing (Threshold Optimizer) (Hardt et al., 2016)":["Y","Y"],
        "Reweighing (Kamiran and Calders, 2012)":["Y","-"],
        "Reject Option Classification (Kamiran et al., 2012)":["Y","-"],
        "Prejudice Remover Regularizer (Kamishima et al., 2012)":["Y","-"],
        "Calibrated Equalized Odds Postprocessing (Pleiss et al., 2017)":["Y","-"],
        "Learning Fair Representations (Zemel et al., 2013)":["Y","-"],
        "Adversarial Debiasing (Zhang et al., 2018)":["Y","-"],
        "Meta-Algorithm for Fair Classification (Celis et al.. 2018":["Y","-"],
        "Rich Subgroup Fairness (Kearns, Neel, Roth, Wu, 2018)":["Y","-"],
        "Exponentiated Gradient (Agarwal, Beygelzimer, Dudik, Langford, Wallach, 2018)":["-","Y"],
        "Grid Search ([Agarwal, Dudik, Wu, 2019], [Agarwal, Beygelzimer, Dudik, Langford, Wallach, 2018])":["-","Y"]
    }
    display(pd.DataFrame(tb_dict, index=["AIF360","FairLearn"]).transpose())
    