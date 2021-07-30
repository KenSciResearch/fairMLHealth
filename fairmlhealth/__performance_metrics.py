'''
'''
import copy
from functools import partial
import logging
from numbers import Number
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metric
from .__utils import epsilon
from .__validation import ArrayLike





__all__ = ["accuracy", "binary_prediction_results", "balanced_accuracy",
           "false_negative_rate", "false_positive_rate", "f1_score",
           "negative_predictive_value", "roc_auc_score", "precision",
           "pr_auc_score", "precision", "true_negative_rate",
           "true_positive_rate"]


''' Utilities '''

def binary_prediction_results(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns a dictionary with counts of TP, TN, FP, and FN

    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    # include labels below to avoid errant results where y_true==y_pred
    tn, fp, fn, tp = sk_metric.confusion_matrix(y_true, y_pred,
                                                labels=[0, 1]).ravel()
    counts = {'TP':tp, 'FP':fp, 'TN':tn, 'FN':fn}
    return counts


def check_result(res:Number, metric_name:str,
                 custom_lower:Number=None, custom_upper:Number=None):
    """ Verifies that the result is in the expected range for the metric and
        returns that result if valid

    Args:
        res (int): result to be validated
        metric_name (str): name of metric; to be used in event of error
    """
    if np.isnan(res):
        return res
    else:
        lower = 0 - 100*epsilon() if custom_lower is None else custom_lower
        upper = 1 + 100*epsilon() if custom_upper is None else custom_upper
        if not lower < res < upper:
            raise ValueError(f"{metric_name} result out of range ({res})")
        else:
            return res


def ratio(numerator:Number, denominator:Number):
    ''' Returns numerator/denominator avoiding division-by-zero errors
    '''
    if denominator == 0:
        return numerator/epsilon()
    else:
        return numerator/denominator


''' Metrics '''


def accuracy(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the accuracy value for the prediction.
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TP'] + rprt['TN'], y_true.shape[0])
    return check_result(res, "Accuracy")


def balanced_accuracy(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the balanced accuracy value for the prediction
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    sens = true_positive_rate(y_true, y_pred)
    spec = true_negative_rate(y_true, y_pred)
    res = ratio(sens + spec, 2)
    return check_result(res, "Balanced Accuracy")


def false_negative_rate(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the false negative rate (miss rate) value for the prediction
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['FN'], rprt['FN'] + rprt['TP'])
    return check_result(res, "FNR")


def false_positive_rate(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the false positive rate (false alarm rate) value for the prediction
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['FP'], rprt['FP'] + rprt['TN'])
    return check_result(res, "FPR}")


def f1_score(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the F1 Score value for the prediction
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    pre = precision(y_true, y_pred)
    rec = true_positive_rate(y_true, y_pred)
    res = 2*ratio(pre*rec, pre+rec)
    return check_result(res, "F1 Score")


def negative_predictive_value(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the negative predictive value for the prediction: TN/(TN+FN)
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TN'], rprt['TN'] + rprt['FN'])
    return res


def roc_auc_score(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the Receiver Operating Characteristic Area Under the Curve
    value for the prediction
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    try:
        res = sk_metric.roc_auc_score(y_true, y_pred)
    except ValueError:
        res = 0
    return check_result(res, "ROC AUC Score")


def pr_auc_score(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the Precision-Recall Area Under the Curve value for the
    prediction
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    try:
        prc, rec, _ = sk_metric.precision_recall_curve(y_true, y_pred)
        res = sk_metric.auc(prc, rec)
    except ValueError:
        res = np.nan
    return check_result(res, "PR AUC Score")


def r_squared(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the R-Squared (coefficient of determination) value
    for the prediction:
        1 - (Sum_of_squares_of_residuals/total_sum_of_squares)
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    res = sk_metric.r2_score(y_true, y_pred)
    if not -1 <= res <= 1:
        res = np.nan
    return check_result(res, "R Squared Score", custom_lower=-1)


def precision(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the precision (Positive Predictive Value, PPV) for the
    prediction: TP/(TP+FP)
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TP'], rprt['TP'] + rprt['FP'])
    return check_result(res, "Precision")


def true_negative_rate(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the True Negative Rate (aka. Selectivity, Specificity) for the
    prediction: TN/(TN+FP)
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TN'], rprt['FP'] + rprt['TN'])
    return check_result(res, "TNR")


def true_positive_rate(y_true:ArrayLike, y_pred:ArrayLike):
    """ Returns the True Positive Rate (aka. Recall, Sensitivity) for the
    prediction: TP/(TP+FN)
    Args:
        y_true, y_pred (numpy-compatible, 1D array-like): binary valued
        objects holding the ground truth and predictions (respectively),
        on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TP'], rprt['FN'] + rprt['TP'])
    return check_result(res, "TPR")
