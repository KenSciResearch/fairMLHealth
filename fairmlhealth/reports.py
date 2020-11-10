# -*- coding: utf-8 -*-
"""
Tools producing reports of fairness, bias, or model performance measures

Contributors:
    camagallen <christine.allen@kensci.com>
"""


from abc import ABC
import aif360.sklearn.metrics as aif_mtrc
import fairlearn.metrics as fl_mtrc
from IPython.display import HTML
import logging
import pandas as pd
import numpy as np
import sklearn.metrics as sk_metric
import warnings

# Tutorial Libraries
from . import tutorial_helpers as helper

# Temporarily hide pandas SettingWithCopy warning
warnings.filterwarnings('ignore', module='pandas')
warnings.filterwarnings('ignore', module='sklearn')

__all__ = ["classification_fairness",
           "classification_performance",
           "regression_fairness",
           "regression_performance",
           "flag_suspicious"]


def __format_fairtest_input(X, prtc_attr, y_true, y_pred, y_prob=None):
    """ Formats data for use by fairness reporting functions.

    Args:
        X (array-like): Sample features
        prtc_attr (named array-like): values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (1D array-like): Sample targets
        y_pred (1D array-like): Sample target predictions
        y_prob (1D array-like, optional): Sample target probabilities. Defaults
            to None.
    Returns:
        Tuple containing formatted versions of all passed args.

    """
    __validate_report_inputs(X, prtc_attr, y_true, y_pred, y_prob)

    # Format inputs to required datatypes
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(prtc_attr, (np.ndarray, pd.Series)):
        if isinstance(prtc_attr, pd.Series):
            prtc_attr = pd.DataFrame(prtc_attr, columns=[prtc_attr.name])
        else:
            prtc_attr = pd.DataFrame(prtc_attr)
    if isinstance(y_true, (np.ndarray, pd.Series)):
        y_true = pd.DataFrame(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred)
    if isinstance(y_prob, np.ndarray):
        y_prob = pd.DataFrame(y_prob)
    for data in [y_true, y_pred, y_prob]:
        if data is not None and (len(data.shape) > 1 and data.shape[1] > 1):
            raise TypeError("Targets and predictions must be 1-Dimensional")

    # Format and set sensitive attributes as index for y dataframes
    pa_name = prtc_attr.columns.tolist()
    prtc_attr.reset_index(inplace=True, drop=True)
    y_true = pd.concat([prtc_attr, y_true.reset_index(drop=True)], axis=1
                       ).set_index(pa_name)
    y_pred = pd.concat([prtc_attr, y_pred.reset_index(drop=True)], axis=1
                       ).set_index(pa_name)
    y_pred.columns = y_true.columns
    if y_prob is not None:
        y_prob = pd.concat([prtc_attr, y_prob.reset_index(drop=True)],
                           axis=1
                           ).set_index(pa_name)
        y_prob.columns = y_true.columns

    # Ensure that protected attributes are integer-valued
    pa_cols = prtc_attr.columns.tolist()
    for c in pa_cols:
        binary = (set(prtc_attr[c].astype(int)) == set(prtc_attr[c]))
        boolean = (prtc_attr[c].dtype == bool)
        two_valued = (set(prtc_attr[c].astype(int)) == {0, 1})
        if not two_valued and (binary or boolean):
            msg = "prtc_attr must be binary or boolean and heterogeneous"
            raise ValueError(msg)
        prtc_attr.loc[:, c] = prtc_attr[c].astype(int)
        if isinstance(c, int):
            prtc_attr.rename(columns={c: f"prtc_attribute_{c}"}, inplace=True)

    return (X, prtc_attr, y_true, y_pred, y_prob)


def __binary_group_fairness_measures(X, prtc_attr, y_true, y_pred, y_prob=None,
                                     priv_grp=1):
    """[summary]

    Args:
        X (pandas DataFrame): Sample features
        prtc_attr (named array-like): values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
        y_prob (pandas DataFrame, optional): Sample target probabilities. Defaults
            to None.

    Returns:
        [type]: [description]
    """
    pa_names = prtc_attr.columns.tolist()
    gf_vals = {}
    gf_key = 'Group Fairness'
    gf_vals['Statistical Parity Difference'] = \
        aif_mtrc.statistical_parity_difference(y_true, y_pred, prot_attr=pa_names)
    gf_vals['Disparate Impact Ratio'] = \
        aif_mtrc.disparate_impact_ratio(y_true, y_pred, prot_attr=pa_names)
    if not helper.is_tutorial_running() and not len(pa_names) > 1:
        gf_vals['Demographic Parity Difference'] = \
            fl_mtrc.demographic_parity_difference(y_true, y_pred,
                                                  sensitive_features=prtc_attr)
        gf_vals['Demographic Parity Ratio'] = \
            fl_mtrc.demographic_parity_ratio(y_true, y_pred,
                                             sensitive_features=prtc_attr)
    gf_vals['Average Odds Difference'] = \
        aif_mtrc.average_odds_difference(y_true, y_pred, prot_attr=pa_names)
    gf_vals['Equal Opportunity Difference'] = \
        aif_mtrc.equal_opportunity_difference(y_true, y_pred, prot_attr=pa_names)
    if not helper.is_tutorial_running() and not len(pa_names) > 1:
        gf_vals['Equalized Odds Difference'] = \
            fl_mtrc.equalized_odds_difference(y_true, y_pred,
                                              sensitive_features=prtc_attr)
        gf_vals['Equalized Odds Ratio'] = \
            fl_mtrc.equalized_odds_ratio(y_true, y_pred,
                                         sensitive_features=prtc_attr)
    gf_vals['Positive Predictive Parity Difference'] = \
        aif_mtrc.difference(sk_metric.precision_score, y_true,
                            y_pred, prot_attr=pa_names, priv_group=priv_grp)
    gf_vals['Balanced Accuracy Difference'] = \
        aif_mtrc.difference(sk_metric.balanced_accuracy_score, y_true,
                            y_pred, prot_attr=pa_names, priv_group=priv_grp)
    if y_prob is not None:
        gf_vals['AUC Difference'] = \
            aif_mtrc.difference(sk_metric.roc_auc_score, y_true, y_prob,
                                prot_attr=pa_names, priv_group=priv_grp)
    return (gf_key, gf_vals)


def __classification_performance_measures(y_true, y_pred):
    # Generate a model performance report
    # If more than 2 classes, return the weighted average prediction scores
    n_class = y_true.append(y_pred).iloc[:, 0].nunique()
    target_labels = [f"target = {t}" for t in set(np.unique(y_true))]
    rprt = classification_performance(y_true.iloc[:, 0], y_pred.iloc[:, 0],
                                      target_labels)
    avg_lbl = "weighted avg" if n_class > 2 else target_labels[-1]
    #
    mp_vals = {}
    c_note = "" if n_class == 2 else "(Weighted Avg)"
    mp_key = f'Model Performance {c_note}**'
    for score in ['precision', 'recall', 'f1-score']:
        mp_vals[score.title()] = rprt.loc[avg_lbl, score]
    mp_vals['Accuracy'] = rprt.loc['accuracy', 'accuracy']
    return (mp_key, mp_vals)


def __individual_fairness_measures(X, prtc_attr, y_true, y_pred):
    """
    Args:
        X (pandas DataFrame): Sample features
        prtc_attr (named array-like): values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions

    Returns:
        Tuple containing a key (header string) and a dictionary of individual
            fairness measures for the data that were passed
    """
    pa_names = prtc_attr.columns.tolist()
    # Generate dict of Individual Fairness measures
    if_vals = {}
    if_key = 'Individual Fairness'
    # consistency_score raises error if null values are present in X
    if X.notnull().all().all():
        if_vals['Consistency Score'] = \
            aif_mtrc.consistency_score(X, y_pred.iloc[:, 0])
    else:
        msg = "Cannot calculate consistency score. Null values present in data."
        logging.info(msg)
    # Other aif360 metrics (not consistency) can handle null values
    if_vals['Between-Group Generalized Entropy Error'] = \
        aif_mtrc.between_group_generalized_entropy_error(y_true, y_pred,
                                                         prot_attr=pa_names)
    return (if_key, if_vals)


def __regres_group_fairness_measures(prtc_attr, y_true, y_pred, priv_grp=1):
    pa_names = prtc_attr.columns.tolist()
    gf_vals = {}
    gf_key = 'Group Fairness'
    gf_vals['Statistical Parity Ratio'] = \
        fl_mtrc.statistical_parity_ratio(y_true, y_pred,
                                         prot_attr=prtc_attr)
    gf_vals['R2 Ratio'] = \
        aif_mtrc.ratio(sk_metric.r2_score, y_true, y_pred,
                       prot_attr=pa_names, priv_group=priv_grp)
    gf_vals['MAE Ratio'] = \
        aif_mtrc.ratio(sk_metric.mean_absolute_error, y_true, y_pred,
                       prot_attr=pa_names, priv_group=priv_grp)
    gf_vals['MSE Ratio'] = \
        aif_mtrc.ratio(sk_metric.mean_squared_error, y_true, y_pred,
                       prot_attr=pa_names, priv_group=priv_grp)
    return (gf_key, gf_vals)


def __regression_performance_measures(y_true, y_pred):
    mp_vals = {}
    mp_key = 'Model Performance'
    report = regression_performance(y_true, y_pred)
    for row in report.iterrows():
        mp_vals[row[0]] = row[1]['Score']
    return (mp_key, mp_vals)


def __validate_report_inputs(X, prtc_attr, y_true, y_pred, y_prob=None):
    valid_data_types = (pd.DataFrame, pd.Series, np.ndarray)
    for data in [X, prtc_attr, y_true, y_pred]:
        if not isinstance(data, valid_data_types):
            raise TypeError("input data is invalid type")
        if not data.shape[0] > 1:
            raise ValueError("input data is too small to measure")
    if y_prob is not None:
        if not isinstance(y_prob, valid_data_types):
            raise TypeError("y_prob is invalid type")


def classification_fairness(X, prtc_attr, y_true, y_pred, y_prob=None,
                            priv_grp=1):
    """ Returns a dataframe containing fairness measures for the model results

        Args:
            X (array-like): Sample features
            prtc_attr (array-like, named): values for the protected attribute
                (note: protected attribute may also be present in X)
            y_true (array-like, 1-D): Sample targets
            y_pred (array-like, 1-D): Sample target predictions
            y_prob (array-like, 1-D): Sample target probabilities
            priv_grp (int, optional): label of the privileged group. Defaults
                to 1.

        Returns:
            pandas dataframe
    """
    X, prtc_attr, y_true, y_pred, y_prob = \
        __format_fairtest_input(X, prtc_attr, y_true, y_pred, y_prob)

    # Generate dict of group fairness measures, if applicable
    n_class = y_true.append(y_pred).iloc[:, 0].nunique()
    if n_class == 2:
        gf_key, gf_vals = \
            __binary_group_fairness_measures(X, prtc_attr, y_true, y_pred,
                                             y_prob, priv_grp)
    else:
        raise ValueError(
            "Reporter cannot yet process multiclass classification models")
    #
    if_key, if_vals = \
        __individual_fairness_measures(X, prtc_attr, y_true, y_pred)

    #
    mp_key, mp_vals = \
        __classification_performance_measures(y_true, y_pred)

    # Convert scores to a formatted dataframe and return
    measures = {gf_key: gf_vals, if_key: if_vals, mp_key: mp_vals}
    df = pd.DataFrame.from_dict(measures, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    df['Value'] = df.loc[:, 'Value'].round(4)
    df.fillna("", inplace=True)
    return df


def classification_performance(y_true, y_pred, target_labels=None):
    """ Returns a formatted dataframe of the scikit-learn classification report

    Args:
        y_true (array): Target values. Must be compatible with model.predict().
        y_pred (array): Prediction values. Must be compatible with
            model.predict().
        target_labels (list of str): Optional labels for target values.
    """
    if target_labels is None:
        target_labels = [f"target = {t}" for t in set(y_true)]
    report = sk_metric.classification_report(y_true, y_pred, output_dict=True,
                                             target_names=target_labels)
    report = pd.DataFrame(report).transpose()
    # Move accuracy to separate row
    accuracy = report.loc['accuracy', :]
    report.drop('accuracy', inplace=True)
    report.loc['accuracy', 'accuracy'] = accuracy[0]
    return(report)


def flag_suspicious(df, caption="", as_styler=False):
    """ Generates embedded html pandas styler table containing a highlighted
        version of a model comparison dataframe

    Args:
        df (pandas dataframe): model comparison dataframe (see)
        caption (str, optional): Optional caption for table. Defaults to "".
        as_styler (bool, optional): If True, returns a pandas Styler of the
            highlighted table (to which other styles/highlights can be added).
            Otherwise, returns the table as an embedded HTML object. Defaults to
            False .

    Returns:
        embedded html or pandas.io.formats.style.Styler
    """
    if caption is None:
        caption = "Fairness Measures"
    #
    idx = pd.IndexSlice
    measures = df.index.get_level_values(1)
    ratios = df.loc[idx['Group Fairness',
                    [c.lower().endswith("ratio") for c in measures]],:].index
    difference = df.loc[idx['Group Fairness',
                        [c.lower().endswith("difference")
                         for c in measures]], :].index
    cs = df.loc[idx['Group Fairness',
                [c.lower().replace(" ", "_") == "consistency_score"
                 for c in measures]], :].index

    styled = df.style.set_caption(caption
                    ).apply(lambda x: ['color:magenta'
                            if (x.name in ratios and not 0.8 < x.iloc[0] < 1.2)
                            else '' for i in x], axis=1
                    ).apply(lambda x: ['color:magenta'
                            if (x.name in cs and x.iloc[0] < 0.5) else '' for i in x],
                            axis=1
                    )
    # Correct management of metric difference has yet to be determined for
    #   regression functions. Add style to o.o.r. difference for binary
    #   classification only
    if "MSE Ratio" not in measures:
        styled.apply(lambda x: ['color:magenta'
                     if (x.name in difference and not -0.1 < x.iloc[0] < 0.1)
                     else '' for i in x], axis=1
                     )
    if as_styler:
        return(styled)
    else:
        return(HTML(styled.render()))


def regression_fairness(X, prtc_attr, y_true, y_pred, priv_grp=1):
    """ Generates a dataframe containing fairness measures for the model results

        Args:
            X (array-like): Sample features
            prtc_attr (array-like, named): values for the protected attribute
                (note: protected attribute may also be present in X)
            y_true (array-like, 1-D): Sample targets
            y_pred (array-like, 1-D): Sample target probabilities
            priv_grp (int, optional): label of the privileged group. Defaults
                to 1.

        Returns:
            pandas dataframe
    """
    X, prtc_attr, y_true, y_pred, _ = \
        __format_fairtest_input(X, prtc_attr, y_true, y_pred)
    #
    gf_key, gf_vals = \
        __regres_group_fairness_measures(prtc_attr, y_true, y_pred,
                                         priv_grp=priv_grp)
    #
    if_key, if_vals = \
        __individual_fairness_measures(X, prtc_attr, y_true, y_pred)
    #
    mp_key, mp_vals = \
        __regression_performance_measures(y_true, y_pred)
    # Convert scores to a formatted dataframe and return
    measures = {gf_key: gf_vals, if_key: if_vals, mp_key: mp_vals}
    df = pd.DataFrame.from_dict(measures, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    df['Value'] = df.loc[:, 'Value'].round(4)
    df.fillna("", inplace=True)
    return df


def regression_performance(y_true, y_pred):
    """ Generates a report of the regression metrics, similar to scikit's
        classification_performance

    Args:
        y_true (array): Target values. Must be compatible with model.predict().
        y_pred (array): Prediction values. Must be compatible with model.predict().

    Returns:
        pandas dataframe
    """
    report = {}
    report['Rsqrd'] = sk_metric.r2_score(y_true, y_pred)
    report['MeanAE'] = sk_metric.mean_absolute_error(y_true, y_pred)
    report['MeanSE'] = sk_metric.mean_squared_error(y_true, y_pred)
    report = pd.DataFrame().from_dict(report, orient='index'
                          ).rename(columns={0: 'Score'})
    return report
