'''
Back-end functions used throughout the library
'''
from importlib.util import find_spec
import pandas as pd
from .__preprocessing import prep_X
from .__validation import validate_X, ValidationError
from warnings import warn


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
        err = (f"cannot round with base {base}." +
               "cb_round designed for base >= 0.01.")
        raise ValueError(err)
    result = series.apply(lambda x: round(base * round(float(x)/base), sig_dec))
    return result



def format_errwarn(func):
    """ Wraps a function returning some result with dictionaries for errors and
        warnings, then formats those errors and warnings as grouped warnings.
        Used for reporting functions to skip errors (and warnings) while
        providing

    Args:
        func (function): any function returning a tuple of format:
            (result, error_dictionary, warning_dictionary). Note that
            dictionaries should be of form {<column or id>:<message>}

    Returns:
        function: the first member of the tuple returned by func
    """
    def format_info(dict):
        info_dict = {}
        for colname, err_wrn in dict.items():
            _ew = list(set(err_wrn)) if isinstance(err_wrn, list) else [err_wrn]
            for m in _ew:
                m = getattr(m, 'message') if 'message' in dir(m) else str(m)
                if m in info_dict.keys():
                    info_dict[m].append(colname)
                else:
                    info_dict[m] = [colname]
        info_dict = {ew:list(set(c)) for ew, c in info_dict.items()}
        return info_dict

    def wrapper(*args, **kwargs):
        res, errs, warns = func(*args, **kwargs)
        if any(errs):
            err_dict = format_info(errs)
            for er, cols in err_dict.items():
                warn(f"Error processing column(s) {cols}. {er}\n")
        if any(warns):
            warn_dict = format_info(warns)
            for wr, cols in warn_dict.items():
                warn(f"Possible error in column(s) {cols}. {wr}\n")
        return res

    return wrapper


def iterate_cohorts(func):
    def prepend_index(df, new_ix):
        idx = df.index.to_frame()
        for l, i in enumerate(new_ix):
            idx.insert(l, i[0], i[1])
        df.index = pd.MultiIndex.from_frame(idx)
        return df

    def wrapper(*args, cohorts=None, **kwargs):
        if len(args) > 0:
            args = [*args]
            X = args.pop(0)
        else:
            X = kwargs.pop('X', None)
        #
        if cohorts is not None:
            validate_X(cohorts)
            cohorts = prep_X(cohorts)
            cols = cohorts.columns
            cgrp = cohorts.groupby(cols)
            results = []
            for k in cgrp.groups.keys():
                df = func(X.iloc[cgrp.groups[k].values, :], *args, **kwargs)
                vals = cgrp.get_group(k)[cols].head(1).values[0]
                ix = [(c, vals[i]) for i, c in enumerate(cols)]
                df = prepend_index(df, ix)
                results.append(df)
            output = pd.concat(results, axis=0)
            return output
        else:
            return func(X, *args, **kwargs)

    return wrapper


def is_dictlike(obj):
    dictlike = all([callable(getattr(obj, "keys", None)),
                    not hasattr(obj, "size")])
    return dictlike


def validate_notebook_requirements():
    """ Alerts the user if they're missing packages required to run extended
        tutorial and example notebooks
    """
    if find_spec('fairlearn') is None:
        err = ("This notebook cannot be re-run witout Fairlearn, available " +
               "via https://github.com/fairlearn/fairlearn. Please install " +
               "Fairlearn to run this notebook.")
        raise ValidationError(err)
    else:
        pass

