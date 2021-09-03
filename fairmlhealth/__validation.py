""" Manages data validation tasks across modules
"""
from collections import OrderedDict
import configparser
from importlib.util import find_spec
from numbers import Number
import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Type, Union
from warnings import warn


LIST_TYPES = (list, tuple, set)
ITER_TYPES = LIST_TYPES + (dict, OrderedDict)
ArrayLike = Type[Union[List, Tuple, np.ndarray, pd.Series, pd.DataFrame]]
MatrixLike = Type[Union[np.ndarray, pd.DataFrame]]
IterableOfStrings = Type[Union[List[str], Tuple[str], Set[str]]]


MIN_OBS = 5  # The minimum number of observations required for measuring and reporting functions


def is_dictlike(obj):
    dictlike = bool(callable(getattr(obj, "keys", None)) and not hasattr(obj, "size"))
    return dictlike


def is_listlike(obj):
    listlike = bool(obj is not None and isinstance(obj, LIST_TYPES))
    return listlike


def is_dictlike(obj):
    dictlike = bool(callable(getattr(obj, "keys", None)) and not hasattr(obj, "size"))
    return dictlike


def limit_alert(
    items: list = None,
    item_name: str = "",
    limit: int = 100,
    issue: str = "This may slow processing time.",
):
    """ Warns the user if there are too many items due to potentially slowed
        processing time
    """
    if any(items):
        if len(items) > limit:
            msg = f"More than {limit} {item_name} detected. {issue}"
            warn(msg)


def validate_analytical_input(
    X, y_true=None, y_pred=None, y_prob=None, prtc_attr=None, priv_grp: int = 1
):
    """ Raises error if data are of incorrect type or size for processing by
        the fairness or performance tables

    Args:
        X (ArrayLike): Sample features
        prtc_attr (ArrayLike, named): values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (ArrayLike): Sample targets
        y_pred (ArrayLike): Sample target predictions
        y_prob (ArrayLike): Sample target probabilities
    """
    validate_data(X, name="input data")
    if y_true is not None:
        validate_array(y_true, name="targets", expected_len=X.shape[0])
    if y_pred is not None:
        validate_array(y_pred, name="predictions", expected_len=X.shape[0])
    if y_prob is not None:
        validate_array(y_prob, name="probabilities", expected_len=X.shape[0])
    if prtc_attr is not None:
        validate_binary_attr(prtc_attr, "protected attribute", expected_len=X.shape[0])
    validate_priv_grp(priv_grp)
    return True


def validate_array(arr, name="array", expected_len: int = 0):
    if arr is None:
        raise ValueError(f"No {name} found.")
    __validate_type(arr)
    __validate_oneDArray(arr, name)
    expected_len = arr.shape[0] if expected_len is None else expected_len
    __validate_length(arr, name, expected_len)
    __validate_values(arr, name)


def validate_binary_attr(arr, name="attribute", expected_len: int = 0):
    expected_len = arr.shape[0] if expected_len is None else expected_len
    validate_array(arr, name, expected_len)
    __validate_binVal(arr, name, fuzzy=False)
    __validate_values(arr, name)


def validate_data(data, name="data", expected_len: int = None):
    if data is None:
        raise ValueError(f"No {name} found.")
    __validate_type(data)
    expected_len = data.shape[0] if expected_len is None else expected_len
    __validate_length(data, name, expected_len)
    __validate_values(data, name)


def validate_fair_boundaries(boundaries: dict = None, measures: list = None):
    err = None
    #
    if not is_dictlike(boundaries):
        raise ValidationError("boundaries must be contained in a dictionary")
    for k, v in boundaries.items():
        if not isinstance(v, tuple) or not all([isinstance(i, Number) for i in v]):
            raise ValidationError("boundaries must contain tuples of numbers")
        elif not v[0] < v[1]:
            raise ValidationError(
                "Invalid boundary values. must be (lower, higher)" + f" Got {v} for {k}"
            )
    if not is_listlike(measures) or not all([isinstance(s, str) for s in measures]):
        raise ValidationError("measures must be a list of strings")
    if measures is not None:
        # Nonsense keys are acceptable as long as one of they keys is correct
        meas = [m.lower() for m in measures]
        errant_entries = [k for k in boundaries.keys() if k.lower() not in meas]
        if not any(errant_entries):
            return None
        else:
            err = (
                f"Boundary keys must be present among the measures"
                + f" displayed in the table. Found: {errant_entries}"
            )
            raise ValidationError(err)
    else:
        pass


def validate_priv_grp(priv_grp: int = None):
    if priv_grp is None:
        raise ValueError("No privileged group found.")
    if not isinstance(priv_grp, int):
        raise TypeError("priv_grp must be an integer")


def validate_supplemental_requirements():
    """ Package validation specific to the supplemental tutorial notebook
    """
    reqs = ["fairlearn", "lightgbm"]
    for r in reqs:
        if find_spec(r) is None:
            missing_reqs = (
                f"{r} package not found. The following packages are"
                + f"required to run this tutorial: {reqs}. \n"
            )
            raise ValidationError(missing_reqs)
        else:
            pass


def validate_tutorial_requirements():
    """ Alerts the user if they're missing packages required to run extended
        tutorial and example notebooks
    """
    # Load current list of required tutorial packages
    config = configparser.ConfigParser()
    config.read("../setup.cfg")
    reqs = config["options.extras_require"]["tutorials"]
    reqs = reqs.lstrip("\n").rstrip("\n").split("\n")
    for r in reqs:
        if find_spec(r) is None:
            missing_reqs = (
                f"{r} package not found. The following packages are"
                + f"required to run this tutorial: {reqs}. \n"
                + " Run the following from the command line to "
                + "automatically install all tutorial requirements:\n"
                + "python3 -m pip install fairmlhealth[tutorials]"
            )
            raise ValidationError(missing_reqs)
        else:
            pass


class ValidationError(Exception):
    def __init__(self, message):
        self.message = message


def __validate_binVal(arr, name: str = "array", fuzzy: bool = True):
    """ Verifies that the array is binary valued.

    Args:
        arr (ArrayLike): numpy-compatible array
        name (str, optional): Name of array to be displayed in feedback.
            Defaults to "array".
        fuzzy (bool, optional): If False, both values 0 and 1 must be present in
            the array to pass validation. Defaults to True.

    Raises:
        ValidationError
    """
    err = None
    binVals = np.array([0, 1])
    if len(np.unique(arr)) > 2:
        err = f"Expected values of [0, 1] in {name}." + f" Received {np.unique(arr)}"
    # Protected attribute must have entries for both 0 and 1
    elif not fuzzy and not np.array_equal(np.unique(arr), binVals):
        err = f"Expected values of [0, 1] in {name}." + f" Received {np.unique(arr)}"
    # Other arrays may have entries for only but either 0 or 1
    elif fuzzy and not all(v in binVals for v in np.unique(arr)):
        err = f"Expected values of [0, 1] in {name}." + f" Received {np.unique(arr)}"
    if err is not None:
        raise ValidationError(err)
    return None


def __validate_oneDArray(arr, name: str = "array"):
    """ Validates that the array is one-dimensional

    Args:
        arr (ArrayLike): numpy-compatible array
        name (str, optional): Name of array to be displayed in feedback.
            Defaults to "array".

    Raises:
        ValidationError
    """
    if len(arr.shape) > 1 and arr.shape[1] > 1:
        err = f"This library is not yet compatible with groups of {name}"
        raise ValidationError(err)


def __validate_length(data, name: str = "array", expected_len: int = 0):
    """ Verifies that the data meet the minimum length criteria and have the
        number of observations expected.

    Args:
        data (matrix-like): numpy-compatible matrix or array
        name (str, optional): Name of array to be displayed in feedback.
            Defaults to "array".
        expected_len (int): Expected length of the data. Defaults to 0, which
            does not meet the minimum length criterion (will fail).

    Raises:
        ValidationError
    """
    # AIF360's consistency_score defaults to 5 nearest neighbors, thus 5 is
    #   the minimum acceptable length as long as that dependency exists
    if expected_len < MIN_OBS:
        raise ValidationError(
            f"Cannot measure fewer than {MIN_OBS} observations"
            + f" (Only {expected_len} found in {name})"
        )
    N = data.shape[0]
    if not N == expected_len:
        raise ValidationError(
            f"All data arguments must be of same length. Only {N} found in {name}"
        )


def __validate_type(data, name: str = "array"):
    """ Verifies that the data are of a type that can be processed by this
        library
validate_fair_boundaries
    Args:
        data (matrix-like): numpy-compatible matrix or array
        name (str, optional): Name of array to be displayed in feedback.
            Defaults to "array".

    Raises:
        TypeError
    """
    valid_data_types = list, tuple, np.ndarray, pd.Series, pd.DataFrame
    if not isinstance(data, valid_data_types):
        err = (
            "Inputs must be one of the following types:"
            f" {valid_data_types}. Found: {type(data)} in {name}"
        )
        raise TypeError(err)


def __validate_values(data: ArrayLike, name: str = "Series"):
    if isinstance(data, pd.Series):
        if data.isnull().all():
            raise ValidationError(f"{name} contains all missing values")
        else:
            pass
    elif isinstance(data, pd.DataFrame):
        if data.isnull().all().any():
            cols = data.loc[:, data.isnull().all()].columns.tolist()
            raise ValidationError(
                f"Some {name} columns contain only missing values."
                + f"If this is expected, please convert to string type: {cols}."
            )
        else:
            pass
    elif isinstance(data, np.ndarray):
        if np.array_equal(np.unique(data), np.array([np.nan])):
            raise ValidationError(f"All values are missing from {name}.")
    else:
        pass
