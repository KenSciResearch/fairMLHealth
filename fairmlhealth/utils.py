'''
Back-end functions used throughout the library
'''
import numpy as np
import pandas as pd


def cb_round(series, base=5, sig_dec=0):
    """ Returns the pandas series (or column) with values rounded per the
            custom base value

        Args:
            series (pd.Series): data to be rounded
            base (float): base value to which data should be rounded (may be
                decimal)
            sig_dec (int): number of significant decimals for the
                custom-rounded value
    """
    if not base >= 0.01:
        raise ValueError(f"cannot round with base {base}."
                         + "cb_round designed for base >= 0.01."
                         )
    result = series.apply(lambda x: round(base * round(float(x)/base), sig_dec))
    return result


def is_dictlike(obj):
    dictlike = all([callable(getattr(obj, "keys", None)),
                    not hasattr(obj, "size")])
    return dictlike


class ValidationError(Exception):
    pass


def __preprocess_input(X, prtc_attr, y_true, y_pred, y_prob=None, priv_grp=1):
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
    __validate_report_input(X, y_true, y_pred, y_prob, prtc_attr, priv_grp)

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
                msg = "prtc_attr must be binary or boolean and heterogeneous"
                raise ValueError(msg)
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

    y_pred.columns = y_true.columns

    return (X, prtc_attr, y_true, y_pred, y_prob)


def __validate_report_input(X, y_true, y_pred=None, y_prob=None, prtc_attr=None,
                            priv_grp:int=1):
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
    for data in [X, y_true, y_pred]:
        if not isinstance(data, valid_data_types) and data is not None:
            raise TypeError(f"input data is invalid type: {type(data)}")
        if not data.shape[0] > 1:
            raise ValueError("input data are too small to measure")
    for y in [y_true, y_pred]:
        if y is None:
            continue
        if isinstance(y, pd.DataFrame):
            if len(y.columns) > 1:
                raise ValueError("target data must contain only one column")
            y = y.iloc[:, 0]
    if y_prob is not None:
        if not isinstance(y_prob, valid_data_types):
            raise TypeError("y_prob is invalid type")

    # protected attribute
    if prtc_attr is not None:
        if not isinstance(prtc_attr, valid_data_types):
            raise TypeError("input data is invalid type")
        if set(np.unique(prtc_attr)) != {0, 1}:
            msg = (f"Invalid values detected in protected attribute(s).",
                   " Must be {0,1}.")
            raise ValueError(msg)
    # priv_grp
    if not isinstance(priv_grp, int):
        raise TypeError("priv_grp must be an integer")

    # If all above runs, inputs pass validation
    return True
