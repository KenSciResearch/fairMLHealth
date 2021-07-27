'''
Back-end functions used throughout the library
'''
from importlib.util import find_spec
import numpy as np
import pandas as pd
from . import __preprocessing as prep
from .__validation import validate_data, ValidationError
from warnings import warn


def format_errwarn(func):
    """ Wraps a function returning some result with dictionaries for errors and
        warnings, then formats those errors and warnings as grouped warnings.
        Used for analytical functions to skip errors (and warnings) while
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


def is_dictlike(obj):
    dictlike = all([callable(getattr(obj, "keys", None)),
                    not hasattr(obj, "size")])
    return dictlike


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
            validate_data(cohorts, name="cohorts", expected_len=X.shape[0])
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


class Flagger():
    """ Manages flag functionality
    """
    diffs = ["auc difference" , "balanced accuracy difference",
            "equalized odds difference", "positive predictive parity difference",
            "statistical parity difference", "fpr diff", "tpr diff", "ppv diff"]
            # not yet enabled: "mean prediction difference", "mae difference", "r2 difference"
    ratios = ["balanced accuracy ratio", "disparate impact ratio ",
              "equalized odds ratio", "fpr ratio", "tpr ratio", "ppv ratio"]
              # not yet enabled "mean prediction ratio", "mae ratio", "r2 ratio"
    stats_high = ["consistency score"]
    stats_low =["between-group gen. entropy error"]

    def __init__(self):
        self.reset()

    def apply_flag(self, df, caption="", sig_fig=4, as_styler=True):
        """ Generates embedded html pandas styler table containing a highlighted
            version of a model comparison dataframe
        Args:
            df (pandas dataframe): model_comparison.compare_models or
                model_comparison.measure_model dataframe
            caption (str, optional): Optional caption for table. Defaults to "".
            as_styler (bool, optional): If True, returns a pandas Styler of the
                highlighted table (to which other styles/highlights can be added).
                Otherwise, returns the table as an embedded HTML object.
        Returns:
            Embedded html or pandas.io.formats.style.Styler
        """
        if caption is None:
            caption = "Fairness Measures"
        # bools are treated as a subclass of int, so must test for both
        if not isinstance(sig_fig, int) or isinstance(sig_fig, bool):
            raise ValueError(f"Invalid value of significant figure: {sig_fig}")
        #
        self.reset()
        if isinstance(df, pd.DataFrame):
            self.df = df.copy()
        else:
            err = "df must be a pandas DataFrame or Styler"
            try:
                if isinstance(df, pd.io.formats.style.Styler):
                    self.df = df.data.copy()
                else: raise ValidationError(err)
            except:
                raise ValidationError(err)
        self.labels, self.label_type = self.set_measure_labels(df)
        #
        if self.label_type == "index":
            styled = self.df.style.set_caption(caption
                                    ).apply(self.color_diff, axis=1
                                    ).apply(self.color_ratio, axis=1
                                    ).apply(self.color_st, axis=1)
        else:
            # pd.Styler doesn't support non-unique indices
            if len(self.df.index.unique()) <  len(self.df):
                self.df.reset_index(inplace=True)
            styled = self.df.style.set_caption(caption
                                    ).apply(self.color_diff, axis=0
                                    ).apply(self.color_ratio, axis=0
                                    ).apply(self.color_st, axis=0)
        # Styler will reset precision to 6 sig figs
        styled = styled.set_precision(sig_fig)
        # return pandas styler if requested
        if as_styler:
            return styled
        else:
            return HTML(styled.render())

    def color_diff(self, s):
        """ Returns a list containing the color settings for difference
            measures found to be OOR
        """
        def is_oor(i): return bool(not -0.1 < i < 0.1 and not np.isnan(i))
        if self.label_type == "index":
            idx = pd.IndexSlice
            lbls = self.df.loc[idx['Group Fairness',
                        [c.lower() in self.diffs for c in self.labels]], :].index
            clr = [f'{self.flag_type}:{self.flag_color}'
                   if (s.name in lbls and is_oor(i)) else '' for i in s]
        else:
            lbls = self.diffs
            clr = [f'{self.flag_type}:{self.flag_color}'
                   if (s.name.lower() in lbls and is_oor(i)) else '' for i in s]
        return clr

    def color_st(self, s):
        """ Returns a list containing the color settings for statistical
            measures found to be OOR
        """
        def is_oor(n, i):
            res = bool((n in lb_low and i > 0.2)
                        or (n in lb_high and i < 0.8) and not np.isnan(i))
            return res
        if self.label_type == "index":
            idx = pd.IndexSlice
            lb_high = self.df.loc[idx['Individual Fairness',
                        [c.lower() in self.stats_high
                        for c in self.labels]], :].index
            lb_low = self.df.loc[idx['Individual Fairness',
                        [c.lower() in self.stats_low
                                for c in self.labels]], :].index
            clr = [f'{self.flag_type}:{self.flag_color}'
                    if is_oor(s.name, i) else '' for i in s]
        else:
            lb_high = self.stats_high
            lb_low = self.stats_low
            clr = [f'{self.flag_type}:{self.flag_color}'
                   if is_oor(s.name.lower(), i) else '' for i in s]
        return clr

    def color_ratio(self, s):
        """ Returns a list containing the color settings for ratio
            measures found to be OOR
        """
        def is_oor(i): return bool(not 0.8 < i < 1.2 and not np.isnan(i))
        if self.label_type == "index":
            idx = pd.IndexSlice
            lbls = self.df.loc[idx['Group Fairness',
                        [c.lower() in self.ratios for c in self.labels]], :].index
            clr = [f'{self.flag_type}:{self.flag_color}'
                   if (s.name in lbls and is_oor(i)) else '' for i in s]
        else:
            lbls = self.ratios
            clr = [f'{self.flag_type}:{self.flag_color}'
                   if (s.name.lower() in lbls and is_oor(i)) else '' for i in s]
        return clr

    def reset(self):
        """ Clears the __Flagger settings
        """
        self.df = None
        self.labels = None
        self.label_type = None
        self.flag_type = "background-color"
        self.flag_color = "magenta"

    def set_measure_labels(self, df):
        """ Determines the locations of the strings containing measure names
            (within df); then reutrns a list of those measure names along
            with their location (one of ["columns", "index"]).
        """
        try:
            labels = df.index.get_level_values(1)
            if type(labels) == pd.core.indexes.numeric.Int64Index:
                label_type = "columns"
            else:
                label_type = "index"
        except:
            label_type = "columns"
        if label_type == "columns":
            labels = df.columns.tolist()
        return labels, label_type


