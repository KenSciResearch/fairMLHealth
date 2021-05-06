'''
'''
import copy
import logging
import sklearn.metrics as sk_metric
import numpy as np
import pandas as pd
from functools import partial



__all__ = ["accuracy", "binary_prediction_results", "balanced_accuracy",
           "false_negative_rate", "false_positive_rate", "f1_score",
           "negative_predictive_value", "roc_auc_score", "precision",
           "pr_auc_score", "precision", "true_negative_rate",
           "true_positive_rate"]


''' Utilities '''

def binary_prediction_results(y_true, y_pred):
    """ Returns a dictionary with counts of TP, TN, FP, and FN. Since validaton
        is assumed to have already been run, this should be faster than using
        scikit

        Args:
            y_true, y_pred (numpy-compatible, 1D array-like): binary valued
            objects holding the ground truth and predictions (respectively),
            on which validation has already been run.
    """
    counts = {}
    # Workaround d.t. bug in some versions of numpy causing error when
    # concatenating dataframes
    arr = np.empty(2, dtype=object)
    arr[:]= [y_true, y_pred]
    #
    t_sum = np.sum(arr, axis=0)
    counts['TP'] = np.count_nonzero(t_sum == 2)
    counts['TN'] = np.count_nonzero(t_sum == 0)
    counts['FP'] = np.count_nonzero(np.logical_and(arr[0]==0, arr[1]==1))
    counts['FN'] = np.count_nonzero(np.logical_and(arr[0]==1, arr[1]==0))
    return counts


def check_result(res, metric_name):
    if res > 1 + 100*epsilon() or res < 0 - 100*epsilon():
        raise ValueError(f"{metric_name} result out of range ({res})")
    else:
        return res


def epsilon():
    """ error value used to prevent 'division by 0' errors """
    return np.finfo(np.float64).eps


def ratio(num, den):
    ''' Returns the ratio of num/den, avoiding division by zero errors
    '''
    if den == 0:
        return num/epsilon()
    else:
        return num/den


''' Metrics '''


def accuracy(y_true, y_pred):
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TP'] + rprt['TN'], y_true.shape[0])
    return check_result(res, "Accuracy")


def balanced_accuracy(y_true, y_pred):
    sens = true_positive_rate(y_true, y_pred)
    spec = true_negative_rate(y_true, y_pred)
    res = ratio(sens + spec, 2)
    return check_result(res, "Balanced Accuracy")


def false_negative_rate(y_true, y_pred): # Miss Rate
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['FN'], rprt['FN'] + rprt['TP'])
    return check_result(res, "FNR")


def false_positive_rate(y_true, y_pred): # False Alarm Rate
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['FP'], rprt['FP'] + rprt['TN'])
    return check_result(res, "FPR}")


def f1_score(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = true_positive_rate(y_true, y_pred)
    res = 2*ratio(pre*rec, pre+rec)
    return check_result(res, "F1 Score")


def scMAE(y_true, y_pred, y_range=None):
    """ Scaled MAE, defined here as the MAE scaled by the range of true values.
        Related metrics such as MAPE (Mean Absolute Percentage Error) or SMAPE
        may be invalid for asymmetrical prediction ranges with negative values.

        Using scaled MAE allows for a standardized "fair" range, rather than
        re-defining this range for each individual regression problem.
    """
    if y_range is not None:
        if not isinstance(y_range, (int, float)):
            err = "Invalid y_range. Must be int or float describing true range."
            raise ValueError(err)
    else:
        y_range = max(epsilon(), np.max(y_true) - np.min(y_true))
    if y_range < 1:
        rmae = np.mean(np.abs(y_true - y_pred))
    else:
        rmae = np.mean(np.abs(y_true - y_pred))/y_range
    return rmae


def negative_predictive_value(y_true, y_pred):
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TN'] + rprt['FN'], rprt['TN'] + rprt['FP'])
    return res


def roc_auc_score(y_true, y_pred):
    try:
        res = sk_metric.roc_auc_score(y_true, y_pred)
    except ValueError:
        res = 0
    return check_result(res, "ROC AUC Score")


def pr_auc_score(y_true, y_pred):
    prc, rec, _ = sk_metric.precision_recall_curve(y_true, y_pred)
    res = sk_metric.auc(prc, rec)
    return check_result(res, "PR AUC Score")


def precision(y_true, y_pred): # aka. PPV
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TP'], rprt['TP'] + rprt['FP'])
    return check_result(res, "Precision")


def true_negative_rate(y_true, y_pred): # aka. selectivity, specificity
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TN'], rprt['FP'] + rprt['TN'])
    return check_result(res, "TNR")


def true_positive_rate(y_true, y_pred): # aka. recall, sensitivity
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TP'], rprt['FN'] + rprt['TP'])
    return check_result(res, "TPR")


