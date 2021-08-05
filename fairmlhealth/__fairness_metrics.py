''' Custom Fairness Metrics

    Note that ratio and difference computation is handled by AIF360's
    sklearn.metrics module. As of the V 0.4.0 release, these are calculated as
    [unprivileged/privileged] and [unprivileged - privileged], respectively
'''
from aif360.sklearn.metrics import difference, ratio
import numpy as np
import pandas as pd
from warnings import catch_warnings, filterwarnings

from .__performance_metrics import (
    epsilon, false_positive_rate, true_positive_rate,
    true_negative_rate, false_negative_rate, precision)



def ratio_wrapper(funcname, *args,):
    """ Text used to filter warnings """
    return


def format_undefined(func):
    """ Wraps ratio functions to return NaN values instead of 0.0 in cases
        where the ratio is undefined
    """
    def wrapper(*args, **kwargs):
        funcname = getattr(func, '__name__', 'an unknown function')
        msg = ("The ratio is ill-defined and being set to 0.0 because" +
                f" '{funcname}' for privileged samples is 0.")
        with catch_warnings(record=True) as w:
            filterwarnings("ignore", message=msg)
            res = func(*args, **kwargs)
        if len(w) > 0:
            return np.nan
        else:
            return res
    return wrapper



@format_undefined
def ppv_ratio(y_true, y_pred, pa_name, priv_grp):
    return ratio(precision, y_true, y_pred,
                     prot_attr=pa_name, priv_group=priv_grp)


@format_undefined
def tpr_ratio(y_true, y_pred, pa_name, priv_grp):
    return ratio(true_positive_rate, y_true, y_pred,
                 prot_attr=pa_name, priv_group=priv_grp)


@format_undefined
def fpr_ratio(y_true, y_pred, pa_name, priv_grp):
    return ratio(false_positive_rate, y_true, y_pred,
                 prot_attr=pa_name, priv_group=priv_grp)


@format_undefined
def tnr_ratio(y_true, y_pred, pa_name, priv_grp):
    return ratio(true_negative_rate, y_true, y_pred,
                 prot_attr=pa_name, priv_group=priv_grp)


@format_undefined
def fnr_ratio(y_true, y_pred, pa_name, priv_grp):
    return ratio(false_negative_rate, y_true, y_pred,
                prot_attr=pa_name, priv_group=priv_grp)


def ppv_diff(y_true, y_pred, pa_name, priv_grp):
    return difference(precision, y_true, y_pred,
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


''' Combined Metrics '''

def eq_odds_diff(y_true, y_pred, prtc_attr=None, priv_grp=1):
    """ Returns the greatest discrepancy between the between-group FPR
        difference and the between-group TPR difference

    Args:
        y_true (1D array-like):
        y_pred (1D array-like):
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.
    """
    fprD = fpr_diff(y_true, y_pred, pa_name=prtc_attr, priv_grp=priv_grp)
    tprD = tpr_diff(y_true, y_pred, pa_name=prtc_attr, priv_grp=priv_grp)
    if abs(fprD) > abs(tprD):
        return fprD
    else:
        return tprD


def eq_odds_ratio(y_true, y_pred, prtc_attr=None, priv_grp=1):
    """ Returns the greatest discrepancy between the between-group FPR
        ratio and the between-group TPR ratio

    Args:
        y_true (1D array-like):
        y_pred (1D array-like):
        priv_grp (int, optional):  . Defaults to 1.
    """
    fprR = fpr_ratio( y_true, y_pred, pa_name=prtc_attr, priv_grp=priv_grp)
    tprR = tpr_ratio( y_true, y_pred, pa_name=prtc_attr, priv_grp=priv_grp)
    if np.isnan(fprR) or np.isnan(tprR):
        return np.nan
    elif round(abs(fprR - 1), 6) > round(abs(tprR - 1), 6):
        return fprR
    else:
        return tprR
