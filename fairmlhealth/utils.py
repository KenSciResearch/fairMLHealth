'''
Back-end functions used throughout the library
'''
from importlib.util import find_spec
import pandas as pd
from . import __preprocessing as prep
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
    """ Runs the function for each cohort subset

    Args:
        func (function): the function to iterate

    Returns:
        cohort-iterated version of the output

    """
    def prepend_cohort(df, new_ix):
        idx = df.index.to_frame().rename(columns={0:'__index'})
        for l, i in enumerate(new_ix):
            idx.insert(l, i[0], i[1])
        if '__index' in idx.columns:
            idx.drop('__index', axis=1, inplace=True)
        df.index = pd.MultiIndex.from_frame(idx)
        return df

    def subset(data, idxs):
        if data is not None:
            return data.loc[idxs,]
        else:
            return None

    def wrapper(cohorts=None, **kwargs):
        """ Iterates for each cohort subset

        Args:
            cohorts (array-like or dataframe, optional): Groups by which to
                subset the data. Defaults to None.

        Returns:
            pandas DataFrame
        """
        # Run preprocessing to facilitate subsetting
        X = kwargs.pop('X', None)
        y_true = kwargs.get('y_true', None)
        y_pred = kwargs.get('y_pred', None)
        y_prob = kwargs.get('y_prob', None)
        prtc_attr = kwargs.get('prtc_attr', None)
        X, prtc_attr, y_true, y_pred, y_prob = \
            prep.standard_preprocess(X, prtc_attr, y_true, y_pred, y_prob)
        #
        if cohorts is not None:
            #
            validate_X(cohorts, name="cohorts", expected_len=X.shape[0])
            cohorts = prep.prep_X(cohorts)
            #
            cix = cohorts.index
            cols = cohorts.columns.tolist()
            cgrp = cohorts.groupby(cols)
            limit_alert(cgrp, "permutations of cohorts", 8)
            #
            results = []
            for k in cgrp.groups.keys():
                ixs = cix.astype('int64').isin(cgrp.groups[k])
                yt = subset(y_true, ixs)
                yh = subset(y_pred, ixs)
                yp = subset(y_prob, ixs)
                pa = subset(prtc_attr, ixs)
                new_args = ['prtc_attr', 'y_true', 'y_pred', 'y_prob']
                sub_args = {k:v for k, v in kwargs.items() if k not in new_args}
                df = func(X=X.iloc[ixs, :], y_true=yt, y_pred=yh, y_prob=yp,
                          prtc_attr=pa, **sub_args)
                vals = cgrp.get_group(k)[cols].head(1).values[0]
                ix = [(c, vals[i]) for i, c in enumerate(cols)]
                df = prepend_cohort(df, ix)
                results.append(df)
            output = pd.concat(results, axis=0)
            return output
        else:
            return func(X=X, **kwargs)

    return wrapper


def is_dictlike(obj):
    dictlike = all([callable(getattr(obj, "keys", None)),
                    not hasattr(obj, "size")])
    return dictlike


def limit_alert(items:list=None, item_name="", limit:int=100,
                issue:str="This may slow processing time."):
    """ Warns the user if there are too many items due to potentially slowed
        processing time
    """
    if any(items):
        if len(items) > limit:
            msg = f"More than {limit} {item_name} detected. {issue}"
            warn(msg)


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

