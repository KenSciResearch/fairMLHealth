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
    """ Returns a dictionary with counts of TP, TN, FP, and FN. Note that
        y_true and y_pred are assumed to be numpy-compatible 1D array-like,
        binary-valued objects on which validation has already been run.
    """
    counts = {}
    # Using numpy here instead of scikit since validaton is assumed to have
    # already been run
    arr = np.array((y_true, y_pred))
    t_sum = np.sum(arr, axis=0)
    counts['TP'] = np.count_nonzero(t_sum == 2)
    counts['TN'] = np.count_nonzero(t_sum == 0)
    counts['FP'] = np.count_nonzero(np.logical_and(arr[0]==0, arr[1]==1))
    counts['FN'] = np.count_nonzero(np.logical_and(arr[0]==1, arr[1]==0))
    return counts


def epsilon():
    """ error value used to prevent 'division by 0' errors """
    return 0.00000000001


def __formatted_prediction_success(y_true, y_pred):
    rprt = binary_prediction_results(y_true, y_pred)
    for k, v in rprt.items():
        if v == 0:
            rprt[k] = epsilon()
    return rprt


''' Metrics '''


def accuracy(y_true, y_pred):
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['TP'] + rprt['TN'] )/(y_true.shape[0])
    return validate_result(res, "Accuracy")


def balanced_accuracy(y_true, y_pred):
    sens = true_positive_rate(y_true, y_pred)
    spec = true_negative_rate(y_true, y_pred)
    res = (sens + spec)/2
    return validate_result(res, "Balanced Accuracy")


def false_negative_rate(y_true, y_pred): # Miss Rate
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = rprt['FN']/(rprt['FN'] + rprt['TP'])
    return validate_result(res, "FNR")


def false_positive_rate(y_true, y_pred): # False Alarm Rate
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = rprt['FP']/(rprt['FP'] + rprt['TN'])
    return validate_result(res, "FPR}")


def f1_score(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = true_positive_rate(y_true, y_pred)
    try:
        res = 2*(pre*rec)/(pre+rec)
    except ZeroDivisionError:
        res = 0
    return validate_result(res, "F1 Score")


def negative_predictive_value(y_true, y_pred):
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['TN'] + rprt['FN'])/(rprt['TN'] + rprt['FP'])
    return res


def roc_auc_score(y_true, y_pred):
    try:
        res = sk_metric.roc_auc_score(y_true, y_pred)
    except ValueError:
        res = 0
    return validate_result(res, "ROC AUC Score")


def pr_auc_score(y_true, y_pred):
    prc, rec, _ = sk_metric.precision_recall_curve(y_true, y_pred)
    res = sk_metric.auc(prc, rec)
    return validate_result(res, "PR AUC Score")

def precision(y_true, y_pred): # aka. PPV
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = rprt['TP']/(rprt['TP'] + rprt['FP'])
    return validate_result(res, "Precision")


def true_negative_rate(y_true, y_pred): # aka. selectivity, specificity
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = rprt['TN']/(rprt['FP'] + rprt['TN'])
    return validate_result(res, "TNR")


def true_positive_rate(y_true, y_pred): # aka. recall, sensitivity
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = rprt['TP']/(rprt['FN'] + rprt['TP'] )
    return validate_result(res, "TPR")


def validate_result(res, metric_name):
    '''
    '''
    if res > 1 + 100*epsilon() or res < 0 - 100*epsilon():
        raise ValueError(f"{metric_name} result out of range ({res})")
    else:
        res = np.clip(res, 0, 1)
        return res
