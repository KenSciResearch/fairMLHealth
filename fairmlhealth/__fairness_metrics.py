
from aif360.sklearn.metrics import difference, ratio
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from .__performance_metrics import (
    epsilon, false_positive_rate, true_positive_rate,
    true_negative_rate, false_negative_rate)


# ToDo: find better solution for these warnings
warnings.filterwarnings('ignore', module='aif360')


def eq_odds_diff(y_true, y_pred, prtc_attr=None, priv_grp=1):
    """ Returns the greatest discrepancy between the between-group FPR
        difference and the between-group TPR difference

    Args:
        y_true (1D array-like):
        y_pred (1D array-like):
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.
    """
    fpr_diff = difference(false_positive_rate, y_true, y_pred,
                          prot_attr=prtc_attr, priv_group=priv_grp)
    tpr_diff = difference(true_positive_rate, y_true, y_pred,
                          prot_attr=prtc_attr, priv_group=priv_grp)
    if abs(fpr_diff) > abs(tpr_diff):
        return fpr_diff
    else:
        return tpr_diff


def eq_odds_ratio(y_true, y_pred, prtc_attr=None, priv_grp=1):
    """ Returns the greatest discrepancy between the between-group FPR
        ratio and the between-group TPR ratio

    Args:
        y_true (1D array-like):
        y_pred (1D array-like):
        priv_grp (int, optional):  . Defaults to 1.
    """
    fpr_ratio = ratio(false_positive_rate, y_true, y_pred,
                        prot_attr=prtc_attr, priv_group=priv_grp)
    tpr_ratio = ratio(true_positive_rate, y_true, y_pred,
                        prot_attr=prtc_attr, priv_group=priv_grp)
    if round(abs(fpr_ratio - 1), 6) > abs(tpr_ratio - 1):
        return fpr_ratio
    else:
        return tpr_ratio


''' Simple '''
def ppv_ratio(y_true, y_pred, pa_name, priv_grp):
    return ratio(precision_score, y_true, y_pred,
                     prot_attr=pa_name, priv_group=priv_grp)


def tpr_ratio(y_true, y_pred, pa_name, priv_grp):
    return ratio(true_positive_rate, y_true, y_pred,
                 prot_attr=pa_name, priv_group=priv_grp)


def fpr_ratio(y_true, y_pred, pa_name, priv_grp):
    return ratio(false_positive_rate, y_true, y_pred,
                 prot_attr=pa_name, priv_group=priv_grp)


def tnr_ratio(y_true, y_pred, pa_name, priv_grp):
    return ratio(true_negative_rate, y_true, y_pred,
                 prot_attr=pa_name, priv_group=priv_grp)


def fnr_ratio(y_true, y_pred, pa_name, priv_grp):
    return ratio(false_negative_rate, y_true, y_pred,
                 prot_attr=pa_name, priv_group=priv_grp)


def ppv_diff(y_true, y_pred, pa_name, priv_grp):
    return difference(precision_score, y_true, y_pred,
                      prot_attr=pa_name, priv_group=priv_grp)


def tpr_diff(y_true, y_pred, pa_name, priv_grp):
    return difference(true_positive_rate, y_true, y_pred,
                      prot_attr=pa_name, priv_group=priv_grp)


def fpr_diff(y_true, y_pred, pa_name, priv_grp):
    return difference(false_positive_rate, y_true, y_pred,
                      prot_attr=pa_name, priv_group=priv_grp)


def tnr_diff(y_true, y_pred, pa_name, priv_grp):
    return difference(true_negative_rate, y_true, y_pred,
                      prot_attr=pa_name, priv_group=priv_grp)


def fnr_diff(y_true, y_pred, pa_name, priv_grp):
    return difference(false_negative_rate, y_true, y_pred,
                      prot_attr=pa_name, priv_group=priv_grp)
