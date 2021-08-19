"""

"""
from abc import ABC
import numpy as np
import pandas as pd
from . import __validation as valid
from .__validation import ArrayLike, IterableOfStrings, MatrixLike, ValidationError


class AnalyticalLabels(ABC):
    @classmethod
    def get_dict(cls, pred_type: str = "binary"):
        """ Returns a dictionary of category labels used by analytical functions
        Args:
            pred_type (b): number of classes in the prediction problem
        """
        valid_types = cls.get_valid_types()
        if pred_type not in valid_types:
            raise ValueError(f"pred_type must be one of {valid_types}")
        c_note = " (Weighted Avg)" if pred_type == "multiclass" else ""
        lbls = {
            "gf_label": "Group Fairness",
            "if_label": "Individual Fairness",
            "mp_label": f"Model Performance{c_note}",
            "dt_label": "Data Metrics",
        }
        return lbls

    @classmethod
    def get_labels(cls, pred_type: str = "binary"):
        label_dict = cls.get_dict(pred_type)
        return label_dict.values()

    @staticmethod
    def get_valid_types():
        return ["binary", "multiclass", "regression"]


def prep_arraylike(arr: ArrayLike, name: str = None, expected_len: int = None):
    valid.validate_array(arr, name, expected_len)
    if isinstance(arr, pd.DataFrame):
        raise ValidationError(
            "This function accepts only 1D numpy arrays or pandas Series objects."
        )
    series = pd.Series(arr, name=name).reset_index(drop=True)
    return series


def prep_data(data: MatrixLike):
    """ Ensures that data are in the correct format

    Returns:
        pd.DataFrame: formatted "X" data (test data)
    """
    if not isinstance(data, pd.DataFrame):
        if isinstance(data, pd.Series):
            X = pd.DataFrame(data, columns=[data.name])
        else:
            X = pd.DataFrame(data)
        # Ensure that some unique identifier is present for each/any column(s)
        if not any(X.columns):
            X.columns = [str(i) for i in range(len(X.columns))]
    else:
        X = data.copy(deep=True)
    # Convert columns that do not contain any strings to numeric type
    for col in X.columns:
        X.loc[:, col] = pd.to_numeric(X[col], errors="ignore")
    return X


def prep_prtc_attr(arr: ArrayLike):
    if not isinstance(arr, pd.DataFrame):
        if isinstance(arr, pd.Series):
            prtc_attr = pd.DataFrame(arr, columns=[arr.name])
        else:
            pa_name = "protected_attribute"
            prtc_attr = pd.DataFrame(arr, columns=[pa_name])
    else:
        prtc_attr = arr.copy(deep=True)
    prtc_attr.reset_index(inplace=True, drop=True)
    return prtc_attr


def prep_targets(arr: ArrayLike, prtc_attr: ArrayLike = None):
    if isinstance(arr, (np.ndarray, pd.Series)):
        targets = pd.DataFrame(arr)
    else:
        targets = arr.copy(deep=True)
    if prtc_attr is not None:
        targets = pd.concat([prtc_attr, targets.reset_index(drop=True)], axis=1)
        targets.set_index(prtc_attr.columns.tolist(), inplace=True)
    else:
        targets.reset_index(drop=True, inplace=True)
    return targets


def prep_preds(
    arr: ArrayLike,
    y_col: str = None,
    prtc_attr: ArrayLike = None,
    name: str = "predictions",
):
    if isinstance(arr, np.ndarray):
        preds = pd.DataFrame(arr)
    else:
        preds = arr.copy(deep=True)
    if prtc_attr is not None:
        preds = pd.concat([prtc_attr, preds.reset_index(drop=True)], axis=1)
        preds.set_index(prtc_attr.columns.tolist(), inplace=True)
    else:
        preds.reset_index(drop=True, inplace=True)
    if y_col is None:
        raise ValidationError(f"Cannot evaluate {name} without ground truth")
    else:
        preds.columns = [y_col]
    return preds


def standard_preprocess(
    X: MatrixLike,
    prtc_attr: ArrayLike = None,
    y_true: ArrayLike = None,
    y_pred: ArrayLike = None,
    y_prob: ArrayLike = None,
    priv_grp: int = 1,
):
    """ Formats data for use by fairness analytical functions.
    Args:
        X (ArrayLike): Sample features
        prtc_attr (ArrayLike, named): values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (ArrayLike): Sample targets
        y_pred (ArrayLike): Sample target predictions
        y_prob (ArrayLike, optional): Sample target probabilities. Defaults
            to None.
        priv_grp (int, optional): label of the privileged group. Defaults
            to 1.
    Returns:
        Tuple containing formatted versions of all passed args.
    """
    valid.validate_analytical_input(X, y_true, y_pred, y_prob, prtc_attr, priv_grp)

    # Format inputs to required datatypes
    X = prep_data(X)
    # Format protected attributes
    if prtc_attr is not None:
        prtc_attr = prep_prtc_attr(prtc_attr)
    #
    _y = None
    if y_true is not None:
        y_true = prep_targets(y_true, prtc_attr)
        if not any(y_true.columns) or y_true.columns[0] == 0:
            _y = y_cols()["priv_names"]["yt"]
            y_true.columns = [_y]
        else:
            _y = y_true.columns[0]
    if y_pred is not None:
        y_pred = prep_preds(y_pred, _y, prtc_attr, name="predictions")
    if y_prob is not None:
        y_prob = prep_preds(y_prob, _y, prtc_attr, name="probabilities")
    #
    return (X, prtc_attr, y_true, y_pred, y_prob)


def stratified_preprocess(
    X: MatrixLike,
    y_true: ArrayLike = None,
    y_pred: ArrayLike = None,
    y_prob: ArrayLike = None,
    features: IterableOfStrings = None,
):
    """
    Runs validation and formats data for use in stratified tables

    Args:
        df (pandas dataframe or compatible object): sample data to be assessed
        y_true (ArrayLike): Sample targets
        y_pred (ArrayLike): Sample target predictions
        y_prob (ArrayLike, optional): Sample target probabilities. Defaults
            to None.
        features (list): columns in df to be assessed if not all columns.
            Defaults to None.

    Requirements:
        - Each feature must be discrete to run stratified analysis, and must be
        binary to run the assessment. If any data are not discrete and there
        are more than 11 values, the tool will reformat those data into
        quantiles
    """
    #
    X, _, y_true, y_pred, y_prob = standard_preprocess(
        X, prtc_attr=None, y_true=y_true, y_pred=y_pred, y_prob=y_prob
    )
    # Attach y variables and subset to expected columns
    _y, _yh, _yp = y_cols()["priv_names"].values()
    df = X.copy()
    pred_cols = []
    if y_true is not None:
        df[_y] = y_true.values
        pred_cols.append(_y)
    if y_pred is not None:
        df[_yh] = y_pred.values
        pred_cols.append(_yh)
    if y_prob is not None:
        df[_yp] = y_prob.values
        pred_cols.append(_yp)
    if features is None:
        features = X.columns.tolist()
    stratified_features = [f for f in features if f not in pred_cols]
    df = df.loc[:, stratified_features + pred_cols]
    #
    if len(df.columns) == 0:
        raise ValidationError("Error during preprocessing")
    return df


def y_cols(df: pd.DataFrame = None):
    """ Returns a dict of hidden column names for each
        of the y values used in stratified table functions, the keys for
        which are as follows: "yt"="y true"; "yh"="y predicted";
        "yp"="y probabilities". This allows for consistent references that are
        not likely to be found among the actual columns of the data (e.g., so
        that columns can be added without error).

        Optionally drops name values that are missing from the df argument.

        Args:
            df (pandas DataFrame, optional): dataframe to check for the presence
                of known names; names that are not found will be dropped from
                the results. Defaults to None.
    """
    y_names = {
        # Private names used in processing
        "priv_names": {"yt": "___y_true", "yh": "___y_pred", "yp": "___y_prob"},
        # Display names
        "disp_names": {"yt": "Target", "yh": "Prediction", "yp": "Probability"},
    }
    #
    if df is not None:
        for k in y_names["priv_names"].keys():
            if y_names["priv_names"][k] not in df.columns:
                y_names["priv_names"][k] = None
    return y_names

