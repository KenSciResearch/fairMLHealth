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


def check_result(res, metric_name):
    if res > 1 + 100*epsilon() or res < 0 - 100*epsilon():
        raise ValueError(f"{metric_name} result out of range ({res})")
    else:
        return res


def epsilon():
    """ error value used to prevent 'division by 0' errors """
    return np.finfo(np.float64).eps


def ratio(numerator, denominator):
    ''' Returns numerator/denominator avoiding division-by-zero errors
    '''
    if den == 0:
        return numerator/epsilon()
    else:
        return numerator/denominator


''' Metrics '''


def accuracy(y_true, y_pred):
    """ Returns the accuracy value for the prediction

        Args:
            y_true, y_pred (numpy-compatible, 1D array-like): binary valued
            objects holding the ground truth and predictions (respectively),
            on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TP'] + rprt['TN'], y_true.shape[0])
    return check_result(res, "Accuracy")


def balanced_accuracy(y_true, y_pred):
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


def false_negative_rate(y_true, y_pred):
    """ Returns the false negative rate (miss rate) value for the prediction

        Args:
            y_true, y_pred (numpy-compatible, 1D array-like): binary valued
            objects holding the ground truth and predictions (respectively),
            on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['FN'], rprt['FN'] + rprt['TP'])
    return check_result(res, "FNR")


def false_positive_rate(y_true, y_pred): 
    """ Returns the false positive rate (false alarm rate) value for the prediction

        Args:
            y_true, y_pred (numpy-compatible, 1D array-like): binary valued
            objects holding the ground truth and predictions (respectively),
            on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['FP'], rprt['FP'] + rprt['TN'])
    return check_result(res, "FPR}")


def f1_score(y_true, y_pred):
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


def negative_predictive_value(y_true, y_pred):
    """ Returns the negative predictive value for the prediction: TN/(TN+FN)

        Args:
            y_true, y_pred (numpy-compatible, 1D array-like): binary valued
            objects holding the ground truth and predictions (respectively),
            on which validation has already been run.
    """
    rprt =  binary_prediction_results(y_true, y_pred)
    res = ratio(rprt['TN'], rprt['TN'] + rprt['FN'])
    return res


def roc_auc_score(y_true, y_pred):
    try:
        res = sk_metric.roc_auc_score(y_true, y_pred)
    except ValueError:
        res = 0
    return check_result(res, "ROC AUC Score")


def pr_auc_score(y_true, y_pred):
    try:
        prc, rec, _ = sk_metric.precision_recall_curve(y_true, y_pred)
        res = sk_metric.auc(prc, rec)
    except ValueError:
        res = np.nan
    return check_result(res, "PR AUC Score")


def r_squared(y_true, y_pred):
    res = sk_metric.r2_score(y_true, y_pred)
    if not -1 <= res <= 1:
        res = np.nan
    return check_result(res, "R Squared Score")


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


