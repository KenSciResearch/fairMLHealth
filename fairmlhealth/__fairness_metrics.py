
from aif360.sklearn.metrics import difference, ratio
import pandas as pd
import warnings

from .__classification_metrics import (
    epsilon, false_positive_rate, true_positive_rate )


# ToDo: find better solution for warnings
warnings.filterwarnings('ignore', module='aif360')


def __fpr(y_true, y_pred):
    """ Wrapper forcing numpy-format arguments for internal classification
        metrics
    """
    # AIF360 converts pd.Series or np.array to an indexed pandas dataframe,
    # which requires special formatting to be re-read as numpy array
    y_true = y_true.iloc[:, 0] if isinstance(y_true, pd.DataFrame) else y_true
    y_pred = y_pred.iloc[:, 0] if isinstance(y_pred, pd.DataFrame) else y_pred
    return false_positive_rate(y_true, y_pred)


def __tpr(y_true, y_pred):
    """ Wrapper forcing numpy-format arguments for internal classification
        metrics
    """
    # AIF360 converts pd.Series or np.array to an indexed pandas dataframe,
    # which requires special formatting to be re-read as numpy array
    y_true = y_true.iloc[:, 0] if isinstance(y_true, pd.DataFrame) else y_true
    y_pred = y_pred.iloc[:, 0] if isinstance(y_pred, pd.DataFrame) else y_pred
    return true_positive_rate(y_true, y_pred)


def eq_odds_diff(y_true, y_pred, prtc_attr=None, priv_grp=1):
    """ Returns the greatest discrepancy between the between-group FPR
        difference and the between-group TPR difference

    Args:
        y_true (1D array-like):
        y_pred (1D array-like):
        prtc_attr (str): name of the protected attribute
        priv_grp (int, optional):  . Defaults to 1.
    """
    fpr_diff = difference(__fpr, y_true, y_pred,
                          prot_attr=prtc_attr, priv_group=priv_grp)
    tpr_diff = difference(__tpr, y_true, y_pred,
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
    fpr_ratio = ratio(__fpr, y_true, y_pred,
                        prot_attr=prtc_attr, priv_group=priv_grp)
    tpr_ratio = ratio(__tpr, y_true, y_pred,
                        prot_attr=prtc_attr, priv_group=priv_grp)
    if round(abs(fpr_ratio - 1), 6) > abs(tpr_ratio - 1):
        return fpr_ratio
    else:
        return tpr_ratio

