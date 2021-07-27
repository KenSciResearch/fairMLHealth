'''

'''
from importlib.util import find_spec
import numpy as np
import pandas as pd
from . import __validation as valid
from .__validation import ValidationError



def analytical_labels(pred_type: str = "binary"):
    """ Returns a dictionary of category labels used by analytical functions
    Args:
        pred_type (b): number of classes in the prediction problem
    """
    valid_pred_types = ["binary", "multiclass", "regression"]
    if pred_type not in valid_pred_types:
        raise ValueError(f"pred_type must be one of {valid_pred_types}")
    c_note = "" if pred_type == "binary" else " (Weighted Avg)"
    lbls = {'gf_label': "Group Fairness",
            'if_label': "Individual Fairness",
            'mp_label': f"Model Performance{c_note}",
            'dt_label': "Data Metrics"
            }
    return lbls


def prep_X(data):
    """ Ensures that data are in the correct format

    Returns:
        pd.DataFrame: formatted "X" data (test data)
    """
    if not isinstance(data, pd.DataFrame):
        if isinstance(data, pd.Series):
            X = pd.DataFrame(data, columns=[data.name])
        else:
            X = pd.DataFrame(data, columns=['X'])
    else:
        X = data.copy(deep=True)
    # Convert columns to numeric type if they do not contain strings
    for col in X.columns:
        X.loc[:, col] = pd.to_numeric(X[col], errors='ignore')
    return X


def prep_prtc_attr(arr):
    if not isinstance(arr, pd.DataFrame):
        if isinstance(arr, pd.Series):
            prtc_attr = pd.DataFrame(arr, columns=[arr.name])
        else:
            pa_name = 'protected_attribute'
            prtc_attr = pd.DataFrame(arr, columns=[pa_name])
    else:
        prtc_attr = arr.copy(deep=True)
    prtc_attr.reset_index(inplace=True, drop=True)
    return prtc_attr


def prep_targets(arr, prtc_attr=None):
    if isinstance(arr, (np.ndarray, pd.Series)):
        y_true = pd.DataFrame(arr)
    else:
        y_true = arr.copy(deep=True)
    if prtc_attr is not None:
        y_true = pd.concat([prtc_attr, y_true.reset_index(drop=True)], axis=1)
        y_true.set_index(prtc_attr.columns.tolist(), inplace=True)
    return y_true


def prep_preds(arr, y_col=None, prtc_attr=None):
    if isinstance(arr, np.ndarray):
        y_pred = pd.DataFrame(arr)
    else:
        y_pred = arr.copy(deep=True)
    if prtc_attr is not None:
        y_pred = pd.concat([prtc_attr, y_pred.reset_index(drop=True)], axis=1)
        y_pred.set_index(prtc_attr.columns.tolist(), inplace=True)
    else:
        pass
    if y_col is None:
        raise ValidationError(
            "Cannot evaluate predictions without ground truth")
    else:
        y_pred.columns = y_col
    return y_pred


def prep_probs(arr, y_col=None, prtc_attr=None):
    if isinstance(arr, np.ndarray):
        y_prob = pd.DataFrame(arr)
    else:
        y_prob = arr.copy(deep=True)
    if prtc_attr is not None:
        y_prob = pd.concat([prtc_attr, y_prob.reset_index(drop=True)], axis=1)
        y_prob.set_index(prtc_attr.columns.tolist(), inplace=True)
    else:
        pass
    if y_col is None:
        raise ValidationError(
            "Cannot evaluate probabilities without ground truth")
    else:
        y_prob.columns = y_col
    return y_prob


def standard_preprocess(X, prtc_attr=None, y_true=None, y_pred=None,
                        y_prob=None, priv_grp=1):
    """ Formats data for use by fairness analytical functions.
    Args:
        X (array-like): Sample features
        prtc_attr (named array-like): values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (1D array-like): Sample targets
        y_pred (1D array-like): Sample target predictions
        y_prob (1D array-like, optional): Sample target probabilities. Defaults
            to None.
        priv_grp (int, optional): label of the privileged group. Defaults
            to 1.
    Returns:
        Tuple containing formatted versions of all passed args.
    """
    valid.validate_analytical_input(X, y_true, y_pred, y_prob, prtc_attr, priv_grp)

    # Format inputs to required datatypes
    X = prep_X(X)
    # Format protected attributes
    if prtc_attr is not None:
        prtc_attr = prep_prtc_attr(prtc_attr)
    #
    y_col = None
    if y_true is not None:
        y_true = prep_targets(y_true, prtc_attr)
        y_col = y_true.columns
    if y_pred is not None:
        y_pred = prep_preds(y_pred, y_col, prtc_attr)
    if y_prob is not None:
        y_prob = prep_probs(y_prob, y_col, prtc_attr)
    #
    return (X, prtc_attr, y_true, y_pred, y_prob)


def stratified_preprocess(X, y_true=None, y_pred=None, y_prob=None,
                          features:list=None):
    """
    Runs validation and formats data for use in stratified tables

    Args:
        df (pandas dataframe or compatible object): sample data to be assessed
        y_true (1D array-like): Sample targets
        y_pred (1D array-like): Sample target predictions
        y_prob (1D array-like, optional): Sample target probabilities. Defaults
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
    X, _, y_true, y_pred, y_prob = \
        standard_preprocess(X, prtc_attr=None, y_true=y_true, y_pred=y_pred,
                           y_prob=y_prob)
    # Attach y variables and subset to expected columns
    yt, yh, yp = y_cols()['col_names'].values()
    df = X.copy()
    pred_cols = []
    if y_true is not None:
        df[yt] = y_true.values
        pred_cols.append(yt)
    if y_pred is not None:
        df[yh] = y_pred.values
        pred_cols.append(yh)
    if y_prob is not None:
        df[yp] = y_prob.values
        pred_cols.append(yp)
    if features is None:
        features = X.columns.tolist()
    stratified_features = [f for f in features if f not in pred_cols]
    df = df.loc[:, stratified_features + pred_cols]
    #
    if len(df.columns) == 0:
        raise ValidationError("Error during preprocessing")
    return df


def y_cols(df=None):
    ''' Returns a dict of hidden column names for each
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
    '''
    y_names = {'col_names': {'yt': '__fairmlhealth_y_true',
                            'yh': '__fairmlhealth_y_pred',
                            'yp': '__fairmlhealth_y_prob'},
              'disp_names': {'yt': 'Target',
                             'yh': 'Pred.',
                             'yp': 'Prob.'}
            }
    #
    if df is not None:
        for k in y_names['col_names'].keys():
            if y_names['col_names'][k] not in df.columns:
                y_names['col_names'][k] = None
    return y_names

