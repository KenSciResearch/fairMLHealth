# -*- coding: utf-8 -*-
"""
Tools producing reports of fairness, bias, or model performance measures
Contributors:
    camagallen <ca.magallen@gmail.com>
"""
import aif360.sklearn.metrics as aif_mtrc
from IPython.display import HTML
import logging
import pandas as pd
import numpy as np
import sklearn.metrics as sk_metric
import warnings

# Tutorial Libraries
from . import tutorial_helpers as helpers
from .__fairness_metrics import eq_odds_diff, eq_odds_ratio
from .utils import __preprocess_input


# ToDo: find better solution for warnings
warnings.filterwarnings('ignore', module='pandas')
warnings.filterwarnings('ignore', module='sklearn')


__all__ = ["classification_fairness",
           "classification_performance",
           "regression_fairness",
           "regression_performance",
           "flag_suspicious"]


def get_report_labels(pred_type: str = "binary"):
    """ Returns a dictionary of category labels used by reporting functions
    Args:
        pred_type (b): number of classes in the prediction problem
    """
    valid_pred_types = ["binary", "multiclass", "regression"]
    if pred_type not in valid_pred_types:
        raise ValueError(f"pred_type must be one of {valid_pred_types}")
    c_note = "" if pred_type == "binary" else " (Weighted Avg)"
    report_labels = {'gf_label': "Group Fairness",
                     'if_label': "Individual Fairness",
                     'mp_label': f"Model Performance{c_note}",
                     'dt_label': "Data Metrics"
                     }
    return report_labels


def __binary_group_fairness_measures(X, pa_name, y_true, y_pred, y_prob=None,
                                     priv_grp=1):
    """ Returns a dictionary containing group fairness measures specific
        to binary classification problems
    Args:
        X (pandas DataFrame): Sample features
        pa_name (str):
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
        y_prob (pandas DataFrame, optional): Sample target probabilities.
            Defaults to None.
        priv_grp (int): Specifies which label indicates the privileged
                group. Defaults to 1.
    """
    #
    gf_vals = {}

    gf_vals['Statistical Parity Difference'] = \
        aif_mtrc.statistical_parity_difference(y_true, y_pred,
                                               prot_attr=pa_name)
    gf_vals['Disparate Impact Ratio'] = \
        aif_mtrc.disparate_impact_ratio(y_true, y_pred, prot_attr=pa_name)

    gf_vals['Equalized Odds Difference'] = eq_odds_diff(y_true, y_pred,
                                                        prtc_attr=pa_name)
    gf_vals['Equalized Odds Ratio'] = eq_odds_ratio(y_true, y_pred,
                                                            prtc_attr=pa_name)

    if helpers.is_kdd_tutorial():
        gf_vals['Average Odds Difference'] = \
            aif_mtrc.average_odds_difference(y_true, y_pred, prot_attr=pa_name)
        gf_vals['Equal Opportunity Difference'] = \
            aif_mtrc.equal_opportunity_difference(y_true, y_pred,
                                                  prot_attr=pa_name)

    # Precision
    gf_vals['Positive Predictive Parity Difference'] = \
        aif_mtrc.difference(sk_metric.precision_score, y_true,
                            y_pred, prot_attr=pa_name, priv_group=priv_grp)

    gf_vals['Balanced Accuracy Difference'] = \
        aif_mtrc.difference(sk_metric.balanced_accuracy_score, y_true,
                            y_pred, prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['Balanced Accuracy Ratio'] = \
        aif_mtrc.ratio(sk_metric.balanced_accuracy_score, y_true,
                       y_pred, prot_attr=pa_name, priv_group=priv_grp)
    if y_prob is not None:
        try:
            gf_vals['AUC Difference'] = \
                aif_mtrc.difference(sk_metric.roc_auc_score, y_true, y_prob,
                                prot_attr=pa_name, priv_group=priv_grp)
        except:
            pass
    return gf_vals


def __classification_performance_measures(y_true, y_pred):
    """ Returns a dictionary containing performance measures specific to
        classification problems
    Args:
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
    """
    # Generate a model performance report
    # If more than 2 classes, return the weighted average prediction scores
    n_class = np.unique(np.append(y_true.values, y_pred.values)).shape[0]
    target_labels = [f"target = {t}" for t in set(np.unique(y_true))]
    rprt = classification_performance(y_true.iloc[:, 0], y_pred.iloc[:, 0],
                                      target_labels)
    avg_lbl = "weighted avg" if n_class > 2 else target_labels[-1]
    #
    mp_vals = {}
    for score in ['precision', 'recall', 'f1-score']:
        mp_vals[score.title()] = rprt.loc[avg_lbl, score]
    mp_vals['Accuracy'] = rprt.loc['accuracy', 'accuracy']
    return mp_vals


def __data_metrics(y_true, priv_grp):
    """Returns a dictionary of data metrics applicable to evaluation of
    fairness
    Args:
        y_true (pandas DataFrame): Sample targets
        priv_grp (int): Specifies which label indicates the privileged
                group. Defaults to 1.
    """
    dt_vals = {}
    dt_vals['Prevalence of Privileged Class (%)'] = \
        round(100*y_true[y_true.eq(priv_grp)].sum()/y_true.shape[0])
    return dt_vals


def __individual_fairness_measures(X, pa_name, y_true, y_pred):
    """ Returns a dictionary of individual fairness measures for the data that
        were passed
    Args:
        X (pandas DataFrame): Sample features
        pa_name (str):
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
    """
    # Generate dict of Individual Fairness measures
    if_vals = {}
    # consistency_score raises error if null values are present in X
    if X.notnull().all().all():
        if_vals['Consistency Score'] = \
            aif_mtrc.consistency_score(X, y_pred.iloc[:, 0])
    else:
        msg = "Cannot calculate consistency score. Null values present in data."
        logging.warning(msg)
    # Other aif360 metrics (not consistency) can handle null values
    if_vals['Between-Group Gen. Entropy Error'] = \
        aif_mtrc.between_group_generalized_entropy_error(y_true, y_pred,
                                                         prot_attr=pa_name)
    return if_vals


def __regres_group_fairness_measures(pa_name, y_true, y_pred, priv_grp=1):
    """ Returns a dictionary containing group fairness measures specific
        to regression problems
    Args:
        prtc_attr_name (named array-like): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
        priv_grp (int): Specifies which label indicates the privileged
                group. Defaults to 1.
    """
    def pdmean(y_true, y_pred, *args): return y_pred.mean()
    #
    gf_vals = {}
    gf_vals['Mean Prediction Ratio'] = \
        aif_mtrc.ratio(pdmean, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['MAE Ratio'] = \
        aif_mtrc.ratio(sk_metric.mean_absolute_error, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['R2 Ratio'] = \
        aif_mtrc.ratio(sk_metric.r2_score, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['Mean Prediction Difference'] = \
        aif_mtrc.difference(pdmean, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['MAE Difference'] = \
        aif_mtrc.difference(sk_metric.mean_absolute_error, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['R2 Difference'] = \
        aif_mtrc.difference(sk_metric.r2_score, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    return gf_vals


def __regression_performance_measures(y_true, y_pred):
    """ Returns a dictionary containing performance measures specific to
        classification problems
    Args:
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
    """
    mp_vals = {}
    report = regression_performance(y_true, y_pred)
    for row in report.iterrows():
        mp_vals[row[0]] = row[1]['Score']
    return mp_vals


def classification_fairness(X, prtc_attr, y_true, y_pred, y_prob=None,
                            priv_grp=1, sig_dec=4, **kwargs):
    """ Returns a pandas dataframe containing fairness measures for the model
        results
    Args:
        X (array-like): Sample features
        prtc_attr (array-like, named): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target predictions
        y_prob (array-like, 1-D): Sample target probabilities
        priv_grp (int): Specifies which label indicates the privileged
            group. Defaults to 1.
        sig_dec (int): number of significant decimals to which to round
            measure values. Defaults to 4.
    """
    # Validate and Format Arguments
    if not isinstance(priv_grp, int):
        raise ValueError("priv_grp must be an integer value")
    if not isinstance(sig_dec, int):
        raise ValueError("sig_dec must be an integer value")
    X, prtc_attr, y_true, y_pred, y_prob = \
        __preprocess_input(X, prtc_attr, y_true, y_pred, y_prob, priv_grp)
    pa_name = prtc_attr.columns.tolist()

    # Temporarily prevent processing for more than 2 classes
    # ToDo: enable multiclass
    n_class = np.unique(np.append(y_true.values, y_pred.values)).shape[0]
    if n_class != 2:
        raise ValueError(
            "Reporter cannot yet process multiclass classification models")
    if n_class == 2:
        labels = get_report_labels()
    else:
        labels = get_report_labels("multiclass")
    gfl, ifl, mpl, dtl = labels.values()
    # Generate a dictionary of measure values to be converted t a dataframe
    mv_dict = {}
    mv_dict[gfl] = \
        __binary_group_fairness_measures(X, pa_name, y_true, y_pred, y_prob,
                                         priv_grp)
    mv_dict[dtl] = __data_metrics(y_true, priv_grp)
    if not kwargs.pop('skip_if', False):
        mv_dict[ifl] = __individual_fairness_measures(X, pa_name, y_true, y_pred)
    if not kwargs.pop('skip_performance', False):
        mv_dict[mpl] = __classification_performance_measures(y_true, y_pred)
    # Convert scores to a formatted dataframe and return
    df = pd.DataFrame.from_dict(mv_dict, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    df.loc[:, 'Value'] = df['Value'].astype(float).round(sig_dec)
    # Fix the order in which the metrics appear
    metric_order = {gfl: 0, ifl: 1, mpl: 2, dtl: 3}
    df.reset_index(inplace=True)
    df['sortorder'] = df['level_0'].map(metric_order)
    df = df.sort_values('sortorder').drop('sortorder', axis=1)
    df.set_index(['level_0', 'level_1'], inplace=True)
    df.rename_axis(('Metric', 'Measure'), inplace=True)
    return df


def classification_performance(y_true, y_pred, target_labels=None):
    """ Returns a pandas dataframe of the scikit-learn classification report,
        formatted for use in fairMLHealth tools
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
    return report


def flag(df, caption="", as_styler=False):
    """ Generates embedded html pandas styler table containing a highlighted
        version of a model comparison dataframe
    Args:
        df (pandas dataframe): Model comparison dataframe (see)
        caption (str, optional): Optional caption for table. Defaults to "".
        as_styler (bool, optional): If True, returns a pandas Styler of the
            highlighted table (to which other styles/highlights can be added).
            Otherwise, returns the table as an embedded HTML object. Defaults
            to False .
    Returns:
        Embedded html or pandas.io.formats.style.Styler
    """
    if caption is None:
        caption = "Fairness Measures"
    #
    idx = pd.IndexSlice
    measures = df.index.get_level_values(1)
    ratios = df.loc[idx['Group Fairness',
                    [c.lower().endswith("ratio") for c in measures]], :].index
    difference = df.loc[idx['Group Fairness',
                        [c.lower().endswith("difference")
                         for c in measures]], :].index
    cs_high = df.loc[idx['Individual Fairness',
                     [c.lower().replace(" ", "_") == "consistency_score"
                      for c in measures]], :].index
    cs_low = df.loc[idx['Individual Fairness',
                        [c.lower().replace(" ", "_")
                            == "generalized_entropy_error"
                         for c in measures]], :].index

    #
    def color_diff(row):
        clr = ['color:magenta'
               if (row.name in difference and not -0.1 < i < 0.1)
               else '' for i in row]
        return clr

    def color_if(row):
        clr = ['color:magenta'
               if (row.name in cs_high and i < 0.8) or
                  (row.name in cs_low and i > 0.2)
               else '' for i in row]
        return clr

    def color_ratios(row):
        clr = ['color:magenta'
               if (row.name in ratios and not 0.8 < i < 1.2)
               else '' for i in row]
        return clr

    styled = df.style.set_caption(caption
                                  ).apply(color_diff, axis=1
                                  ).apply(color_ratios, axis=1
                                  ).apply(color_if, axis=1)
    # Correct management of metric difference has yet to be determined for
    #   regression functions. Add style to o.o.r. difference for binary
    #   classification only
    if "MSE Ratio" not in measures:
        styled.apply(color_diff, axis=1)
    # return pandas styler if requested
    if as_styler:
        return styled
    else:
        return HTML(styled.render())


def flag_suspicious(df, caption="", as_styler=False):
    warnings.warn(
            "flag_suspicious will be deprecated in version 2." +
            " Use flag instead.", PendingDeprecationWarning
        )
    return flag(df, caption="", as_styler=False)


def regression_fairness(X, prtc_attr, y_true, y_pred, priv_grp=1, sig_dec=4,
                        **kwargs):
    """ Returns a pandas dataframe containing fairness measures for the model
        results
    Args:
        X (array-like): Sample features
        prtc_attr (array-like, named): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target probabilities
        priv_grp (int): Specifies which label indicates the privileged
            group. Defaults to 1.
        sig_dec (int): number of significant decimals to which to round
            measure values. Defaults to 4.
    """
    # Validate and Format Arguments
    if not isinstance(priv_grp, int):
        raise ValueError("priv_grp must be an integer value")
    if not isinstance(sig_dec, int):
        raise ValueError("sig_dec must be an integer value")
    X, prtc_attr, y_true, y_pred, _ = \
        __preprocess_input(X, prtc_attr, y_true, y_pred, priv_grp)
    pa_name = prtc_attr.columns().tolist()
    #
    gf_vals = \
        __regres_group_fairness_measures(pa_name, y_true, y_pred,
                                         priv_grp=priv_grp)
    #
    if not kwargs.pop('skip_if', False):
        if_vals = __individual_fairness_measures(X, pa_name, y_true, y_pred)
    mp_vals = __regression_performance_measures(y_true, y_pred)
    dt_vals = __data_metrics(y_true, priv_grp)

    # Convert scores to a formatted dataframe and return
    labels = get_report_labels("regression")
    measures = {labels['gf_label']: gf_vals,
                labels['if_label']: if_vals,
                labels['mp_label']: mp_vals,
                labels['dt_label']: dt_vals}
    df = pd.DataFrame.from_dict(measures, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    df.loc[:, 'Value'] = df['Value'].astype(float).round(sig_dec)
    return df


def regression_performance(y_true, y_pred):
    """ Returns a pandas dataframe of the regression performance metrics,
        similar to scikit's classification_performance
    Args:
        y_true (array): Target values. Must be compatible with model.predict().
        y_pred (array): Prediction values. Must be compatible with
            model.predict().
    """
    report = {}
    report['Rsqrd'] = sk_metric.r2_score(y_true, y_pred)
    report['MeanAE'] = sk_metric.mean_absolute_error(y_true, y_pred)
    report['MeanSE'] = sk_metric.mean_squared_error(y_true, y_pred)
    report = pd.DataFrame().from_dict(report, orient='index'
                          ).rename(columns={0: 'Score'})
    return report
