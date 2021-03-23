import copy
import logging
import sklearn.metrics as sk_metric
from fairmlhealth.utils import *
import numpy as np
import pandas as pd
from functools import partial


__BPC = None
E = 0.0000000001



def binary_prediction_results(y_true, y_pred):
    """ Returns a dictionary with counts of TP, TN, FP, and FN
    """
    report = {}
    res = pd.concat((y_true, y_pred), axis=1)
    res.columns = ['t','p']
    report['TP'] = (res['t'].eq(1) & res['p'].eq(1)).sum()
    report['TN'] = (res['t'].eq(0) & res['p'].eq(0)).sum()
    report['FP'] = (res['t'].eq(0) & res['p'].eq(1)).sum()
    report['FN'] = (res['t'].eq(1) & res['p'].eq(0)).sum()
    return report


def __formatted_prediction_success(y_true, y_pred):
    rprt = binary_prediction_results(y_true, y_pred)
    for k,v in rprt.items():
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


def accuracy(y_true, y_pred, **kwargs):
    use_cache = kwargs.pop('__use_cache', False)
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['TP'] + rprt['TN'] )/(y_true.shape[0])
    return validate_result(res, "Accuracy")


def balanced_accuracy(y_true, y_pred, **kwargs):
    res = (sensitivity(y_true, y_pred)+specificity(y_true, y_pred))/2
    return validate_result(res, "Balanced Accuracy")


def precision(y_true, y_pred, **kwargs): # aka. PPV
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['TP'])/(rprt['TP'] + rprt['FP'])
    return validate_result(res, "Precision")


def sensitivity(y_true, y_pred, **kwargs): # aka. recall, TPR
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['TP'])/(rprt['FN'] + rprt['TP'] )
    return validate_result(res, "Sensitivity")


def false_alarm_rate(y_true, y_pred, **kwargs): # FPR
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['FP'])/(rprt['FP'] + rprt['TN'])
    return validate_result(res, "False Alarm Rate")


def specificity(y_true, y_pred, **kwargs): # aka. TNR, selectivity
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['TN'])/(rprt['FP'] + rprt['TN'])
    return validate_result(res, "Specificity")


def miss_rate(y_true, y_pred, **kwargs): # FNR
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['FN'])/(rprt['FN'] + rprt['TP'])
    return validate_result(res, "Miss Rate")


def negative_predictive_value(y_true, y_pred, **kwargs):
    rprt = __formatted_prediction_success(y_true, y_pred)
    res = (rprt['TN'] + rprt['FN'])/(rprt['TN'] + rprt['FP'])
    return res


def f1_score(y_true, y_pred, **kwargs):
    pre = precision(y_true, y_pred, **kwargs)
    rec = sensitivity(y_true, y_pred, **kwargs)
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


