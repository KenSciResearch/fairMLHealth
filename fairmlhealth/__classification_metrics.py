import copy
import logging
import sklearn.metrics as sk_metric
from fairmlhealth.utils import *
import numpy as np
import pandas as pd
from functools import partial


# Set E as the error value used to prevent 'division by 0' errors
E = 0.0000000001


def binary_prediction_results(y_true, y_pred):
    """ Returns a dictionary with counts of TP, TN, FP, and FN
    """
    counts = {}
    # Using numpy here instead of scikit since validaton has already been run
    arr = np.array((y_true, y_pred))
    t_sum = np.sum(arr, axis=0)
    counts['TP'] = np.count_nonzero(t_sum == 2)
    counts['TN'] = np.count_nonzero(t_sum == 0)
    counts['FP'] = np.count_nonzero(np.logical_and(arr[0]==0, arr[1]==1))
    counts['FN'] = np.count_nonzero(np.logical_and(arr[0]==1, arr[1]==0))
    return counts


def __formatted_prediction_success(y_true, y_pred):
    rprt = binary_prediction_results(y_true, y_pred)
    for k, v in rprt.items():
        if v == 0:
            rprt[k] = E
    return(rprt)


def validate_result(res, metric_name):
    '''
    '''
    if res > 1+10*E or res < 0-E:
        raise ValueError(f"{metric_name} result out of range ({res})")
    else:
        return(np.clip(res, 0, 1))


def accuracy(y_true, y_pred):
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['TP'] + rprt['TN'] )/(y_true.shape[0])
    return validate_result(res, "Accuracy")


def balanced_accuracy(y_true, y_pred):
    sens = sensitivity(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    res = (sens + spec)/2
    return validate_result(res, "Balanced Accuracy")


def false_alarm_rate(y_true, y_pred): # FPR
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = rprt['FP']/(rprt['FP'] + rprt['TN'])
    return validate_result(res, "False Alarm Rate")


def miss_rate(y_true, y_pred): # FNR
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = rprt['FN']/(rprt['FN'] + rprt['TP'])
    return validate_result(res, "Miss Rate")


def negative_predictive_value(y_true, y_pred):
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['TN'] + rprt['FN'])/(rprt['TN'] + rprt['FP'])
    return res

def precision(y_true, y_pred): # aka. PPV
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = rprt['TP']/(rprt['TP'] + rprt['FP'])
    return validate_result(res, "Precision")


def sensitivity(y_true, y_pred): # aka. recall, TPR
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = rprt['TP']/(rprt['FN'] + rprt['TP'] )
    return validate_result(res, "Sensitivity")


def specificity(y_true, y_pred): # aka. TNR, selectivity
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = rprt['TN']/(rprt['FP'] + rprt['TN'])
    return validate_result(res, "Specificity")


def f1_score(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = sensitivity(y_true, y_pred)
    try:
        res = 2*(pre*rec)/(pre+rec)
    except ZeroDivisionError:
        res = 0
    return validate_result(res, "F1 Score")


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

