'''

'''
from importlib.util import find_spec
import numpy as np
import pandas as pd
from .utils import ValidationError


def clean_hidden_names(col):
    ''' If the column is a hidden variable, replaces the variable with a
        display name
    '''
    yvars = y_cols()
    if col in yvars['col_names'].values():
        idx = list(yvars['col_names'].values()).index(col)
        key = list(yvars['col_names'].keys())[idx]
        display_name = yvars['disp_names'][key]
    else:
        display_name = col
    return display_name


def report_labels(pred_type: str = "binary"):
    """ Returns a dictionary of category labels used by reporting functions
    Args:
        pred_type (b): number of classes in the prediction problem
    """
    valid_pred_types = ["binary", "multiclass", "regression"]
    if pred_type not in valid_pred_types:
        raise ValueError(f"pred_type must be one of {valid_pred_types}")
    c_note = "" if pred_type == "binary" else " (Weighted Avg)"
    report_labels = {'gf_label': "Group Fairness",
                     'if_label': "Individual Fairness",
                     'mp_label': f"Model Performance{c_note}",
                     'dt_label': "Data Metrics"
                     }
    return report_labels


def standard_preprocess(X, prtc_attr, y_true, y_pred, y_prob=None, priv_grp=1):
    """ Formats data for use by fairness reporting functions.
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
    validate_report_input(X, y_true, y_pred, y_prob, prtc_attr, priv_grp)

    # Format inputs to required datatypes
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X, columns=[X.name])
        else:
            X = pd.DataFrame(X, columns=['X'])
    if isinstance(y_true, (np.ndarray, pd.Series)):
        y_true = pd.DataFrame(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred)
    if isinstance(y_prob, np.ndarray):
        y_prob = pd.DataFrame(y_prob)
    for data in [y_true, y_pred, y_prob]:
        if data is not None and (len(data.shape) > 1 and data.shape[1] > 1):
            raise TypeError("Targets and predictions must be 1-Dimensional")

    # Format protected attributes
    if prtc_attr is not None:
        if not isinstance(prtc_attr, pd.DataFrame):
            if isinstance(prtc_attr, pd.Series):
                prtc_attr = pd.DataFrame(prtc_attr, columns=[prtc_attr.name])
            else:
                pa_name = 'protected_attribute'
                prtc_attr = pd.DataFrame(prtc_attr, columns=[pa_name])
        pa_cols = prtc_attr.columns.tolist()

        # Ensure that protected attributes are integer-valued
        for c in pa_cols:
            binary_boolean = prtc_attr[c].isin([0, 1, False, True]).all()
            two_valued = ((set(prtc_attr[c].astype(int)) == {0, 1}))
            if not two_valued and binary_boolean:
                err = "prtc_attr must be binary or boolean and heterogeneous"
                raise ValueError(err)
            prtc_attr.loc[:, c] = prtc_attr[c].astype(int)
            if isinstance(c, int):
                prtc_attr.rename(columns={c: f"prtc_attr_{c}"}, inplace=True)

        # Attach protected attributes as target data index
        prtc_attr.reset_index(inplace=True, drop=True)
        y_true = pd.concat([prtc_attr, y_true.reset_index(drop=True)], axis=1)
        y_true.set_index(pa_cols, inplace=True)
        y_pred = pd.concat([prtc_attr, y_pred.reset_index(drop=True)], axis=1)
        y_pred.set_index(pa_cols, inplace=True)
        if y_prob is not None:
            y_prob = pd.concat([prtc_attr, y_prob.reset_index(drop=True)],
                                axis=1)
            y_prob.set_index(pa_cols, inplace=True)
            y_prob.columns = y_true.columns

    if y_pred is not None and y_true is not None:
        y_pred.columns = y_true.columns

    return (X, prtc_attr, y_true, y_pred, y_prob)


def stratified_preprocess(X, y_true, y_pred=None, y_prob=None,
                          features:list=None):
    """
    Runs validation and formats data for use in stratified reports

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
        are more than 11 values, the reporter will reformat those data into
        quantiles
    """
    #
    max_cats = 11
    #
    X, _, y_true, y_pred, y_prob = \
        standard_preprocess(X, prtc_attr=None, y_true=y_true, y_pred=y_pred,
                           y_prob=y_prob)
    yt, yh, yp = y_cols()['col_names'].values()
    # Attach y variables and subset to expected columns
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
    over_max_vals = []
    for f in stratified_features:
        if df[f].nunique() > max_cats:
            over_max_vals.append(f)
        else:
            pass
        df[f].fillna(np.nan, inplace=True)
        df[f] = df[f].astype(str)
    if any(over_max_vals):
        print(f"USER ALERT! The following features have more than {max_cats}",
              "values, which will slow processing time. Consider reducing to",
              f"bins or quantiles: {over_max_vals}")
    elif len(df.columns) == 0:
        raise ValidationError("Error during preprocessing")
    return df


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
    valid_data_types = (pd.DataFrame, pd.Series, np.ndarray)

    # input data
    if X is None:
        raise ValueError("No input data ")
    for data in [X, y_true, y_pred]:
        if data is not None:
            if not isinstance(data, valid_data_types):
                err = ("One of X, y_true, or y_pred is invalid type"
                       + str(type(data)))
                raise TypeError(err)
            if not data.shape[0] > 1:
                err = ("One of X, y_true, or y_pred has too few nonmissing"
                       + "observations to measure")
                raise ValueError(err)
    for y in [y_true, y_pred]:
        if y is None:
            continue
        if isinstance(y, pd.DataFrame):
            if len(y.columns) > 1:
                raise ValueError("target data must contain only one column")
            y = y.iloc[:, 0]
    if y_prob is not None:
        if not isinstance(y_prob, valid_data_types):
            raise TypeError(f"y_prob is invalid type {type(y_prob)}")

    # protected attribute
    if prtc_attr is not None:
        if not isinstance(prtc_attr, valid_data_types):
            raise TypeError("input data is invalid type")
        if set(np.unique(prtc_attr)) != {0, 1}:
            err = (f"Invalid values detected in protected attribute(s).",
                   " Must be {0,1}.")
            raise ValueError(err)
    # priv_grp
    if not isinstance(priv_grp, int):
        raise TypeError("priv_grp must be an integer")

    # If all above runs, inputs pass validation
    return True


def y_cols(df=None):
    ''' Returns a dict of hidden column names for each
        of the y values used in stratified reporting functions, the keys for
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
