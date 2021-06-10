''' Manages data validation tasks across modules
'''
from collections import OrderedDict
import numpy as np
import pandas as pd



ITER_TYPES = (list, tuple, set, dict, OrderedDict)

class ValidationError(Exception):
    pass

def __validate_binVal(arr, arrname="array", fuzzy=True):
    err = None
    binVals = np.array([0, 1])
    if len(np.unique(arr)) > 2:
        err = (f"Multiple labels found in {arrname}. "
                "Expected only 0 or 1.")
    # Protected attribute must have entries for both 0 and 1
    elif not fuzzy and not np.array_equal(np.unique(arr), binVals):
        err = (f"Expected values of [0, 1] in {arrname}." +
                f" Received {np.unique(arr)}")
    # Other arrays may have entries for only but either 0 or 1
    elif fuzzy and not all(v in binVals for v in np.unique(arr)):
        err = (f"Expected values of [0, 1] in {arrname}." +
                f" Received {np.unique(arr)}")
    if err is not None:
        raise ValidationError(err)
    return None
    
    
def __validate_oneDArray(arr, arrname="array"):
    if len(arr.shape) > 1 and arr.shape[1] > 1:
        err = f"This library is not yet compatible with groups of {arrname}"
        raise ValidationError(err)


def __validate_length(arr, arrname="array", expected_len=0):
    llim = 4
    if expected_len < llim:
        raise ValidationError(f"Cannot measure fewer than {llim} observations"
                              + f" (found in {arrname}")
    N = arr.shape[0]
    if not N == expected_len:
        raise ValidationError("All data arguments must be of same length."
                              + f" Only {N} found in {arrname}")


def __validate_type(data, arrname="array"):
    valid_data_types = (pd.DataFrame, pd.Series, np.ndarray)
    if not isinstance(data, valid_data_types):
            err = ("Inputs must be one of the following types:"
                   f" {valid_data_types}. Found: {type(data)} in {arrname}")
            raise TypeError(err)


def validate_prtc_attr(arr, expected_len=0):
    if arr is None:
        raise ValueError("No protected attribute found.")
    __validate_type(arr)
    __validate_oneDArray(arr, "protected attribute")
    __validate_binVal(arr, "protected attribute", fuzzy=False)
    __validate_length(arr, "protected attribute", expected_len)


def validate_targets(arr, expected_len=0):
    if arr is None:
        raise ValueError("No targets found.")
    __validate_type(arr)
    __validate_oneDArray(arr, "targets")
    __validate_binVal(arr, "targets", fuzzy=True)
    __validate_length(arr, "targets", expected_len)


def validate_preds(arr, expected_len=0):
    if arr is None:
        raise ValueError("No predictions found.")
    __validate_type(arr)
    __validate_oneDArray(arr)
    __validate_binVal(arr, "predictions", fuzzy=True)
    __validate_length(arr, "predictions", expected_len)


def validate_priv_grp(priv_grp):
    if priv_grp is None:
        raise ValueError("No privileged group found.")
    if not isinstance(priv_grp, int):
        raise TypeError("priv_grp must be an integer")


def validate_probs(arr, expected_len=0):
    if arr is None:
        raise ValueError("No probabilities found.")
    __validate_oneDArray(arr)


def validate_X(data):
    if data is None:
        raise ValueError("No input data found.")
    __validate_type(data)
    __validate_length(data, "input data", data.shape[0])


def validate_report_input(X, y_true=None, y_pred=None, y_prob=None,
                            prtc_attr=None, priv_grp:int=1):
    """ Raises error if data are of incorrect type or size for processing by
        the fairness or performance reporters

    Args:
        X (array-like): Sample features
        prtc_attr (array-like, named): values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target predictions
        y_prob (array-like, 1-D): Sample target probabilities
    """
    validate_X(X)
    if y_true is not None:
        validate_targets(y_true, X.shape[0])
    if y_pred is not None:
        validate_preds(y_pred, X.shape[0])
    if y_prob is not None:
        validate_probs(y_prob, X.shape[0])
    if prtc_attr is not None:
        validate_prtc_attr(prtc_attr, X.shape[0])
    validate_priv_grp(priv_grp)
    return True