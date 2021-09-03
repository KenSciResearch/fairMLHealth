""" Custom Fairness Metrics

    Note that ratio and difference computation is handled by AIF360's
    sklearn.metrics module. As of the V 0.4.0 release, these are calculated as
    [unprivileged/privileged] and [unprivileged - privileged], respectively
"""
from typing import Callable
from aif360.sklearn.metrics import difference, ratio
import numpy as np
import pandas as pd
from warnings import catch_warnings, filterwarnings

from .performance_metrics import (
    false_positive_rate,
    true_positive_rate,
    true_negative_rate,
    false_negative_rate,
    precision,
)


def __manage_undefined_ratios(func: Callable):
    """ Wraps ratio functions to return NaN values instead of 0.0 in cases
        where the ratio is undefined
    """

    def wrapper(*args, **kwargs):
        funcname = getattr(func, "__name__", "an unknown function")
        msg = (
            "The ratio is ill-defined and being set to 0.0 because"
            + f" '{funcname}' for privileged samples is 0."
        )
        with catch_warnings(record=True) as w:
            filterwarnings("ignore", message=msg)
            res = func(*args, **kwargs)
        if len(w) > 0:
            return np.nan
        else:
            return res

    return wrapper


@__manage_undefined_ratios
def ppv_ratio(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the between-group ratio of Postive Predictive Values

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    return ratio(precision, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp)


@__manage_undefined_ratios
def tpr_ratio(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the between-group ratio of True Positive Rates

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    return ratio(
        true_positive_rate, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp
    )


@__manage_undefined_ratios
def fpr_ratio(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the between-group ratio of False Positive Rates

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    return ratio(
        false_positive_rate, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp
    )


@__manage_undefined_ratios
def tnr_ratio(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the between-group ratio of True Negative Rates

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    return ratio(
        true_negative_rate, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp
    )


@__manage_undefined_ratios
def fnr_ratio(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the between-group ratio of False Negative Rates

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    return ratio(
        false_negative_rate, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp
    )


def ppv_diff(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the between-group difference of Positive Predictive Values

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    return difference(precision, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp)


def tpr_diff(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the between-group difference of True Positive Rates

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    return difference(
        true_positive_rate, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp
    )


def fpr_diff(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the between-group difference of False Positive Rates

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    return difference(
        false_positive_rate, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp
    )


def tnr_diff(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the between-group difference of True Negative Rates

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    return difference(
        true_negative_rate, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp
    )


def fnr_diff(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the between-group difference of False Negative Rates

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    return difference(
        false_negative_rate, y_true, y_pred, prot_attr=pa_name, priv_group=priv_grp
    )


""" Combined Metrics """


def eq_odds_diff(y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1):
    """ Returns the greatest discrepancy between the between-group FPR
        difference and the between-group TPR difference

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.

    Returns:
        Number

    """
    fprD = fpr_diff(y_true, y_pred, pa_name=pa_name, priv_grp=priv_grp)
    tprD = tpr_diff(y_true, y_pred, pa_name=pa_name, priv_grp=priv_grp)
    if abs(fprD) > abs(tprD):
        return fprD
    else:
        return tprD


def eq_odds_ratio(
    y_true: pd.Series, y_pred: pd.Series, pa_name: str, priv_grp: int = 1
):
    """ Returns the greatest discrepancy between the between-group FPR
        ratio and the between-group TPR ratio

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values
        priv_grp (int, optional):  . Defaults to 1.
    """
    fprR = fpr_ratio(y_true, y_pred, pa_name=pa_name, priv_grp=priv_grp)
    tprR = tpr_ratio(y_true, y_pred, pa_name=pa_name, priv_grp=priv_grp)
    if np.isnan(fprR) or np.isnan(tprR):
        return np.nan
    elif round(abs(fprR - 1), 6) > round(abs(tprR - 1), 6):
        return fprR
    else:
        return tprR

