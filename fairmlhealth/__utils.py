'''
Back-end functions used throughout the library, many of which assume that inputs
have been validated
'''
from numbers import Number
from typing import Callable
import numpy as np
import pandas as pd
from . import __preprocessing as prep, __validation as valid
from .__validation import ValidationError
from warnings import warn



def epsilon():
    """ error value used to prevent 'division by 0' errors """
    return np.finfo(np.float64).eps


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


def iterate_cohorts(func:Callable):
    """ Runs the function for each cohort subset

    Args:
        func (function): the function to iterate

    Returns:
        cohort-iterated version of the output

    """
    def prepend_cohort(df:valid.MatrixLike, new_ix:valid.ArrayLike):
        idx = df.index.to_frame().rename(columns={0:'__index'})
        for l, i in enumerate(new_ix):
            idx.insert(l, i[0], i[1])
        if '__index' in idx.columns:
            idx.drop('__index', axis=1, inplace=True)
        df.index = pd.MultiIndex.from_frame(idx)
        return df

    def subset(data:valid.MatrixLike, idxs:valid.ArrayLike):
        if data is not None:
            return data.loc[idxs,]
        else:
            return None

    def wrapper(cohorts:valid.MatrixLike=None, **kwargs):
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
            valid.validate_data(cohorts, name="cohorts", expected_len=X.shape[0])
            cohorts = prep.prep_data(cohorts)
            #
            cix = cohorts.index
            cols = cohorts.columns.tolist()
            cgrp = cohorts.groupby(cols)
            valid.limit_alert(cgrp, "permutations of cohorts", 8,
                        issue="This may slow processing time and reduce utility.")
            minobs = valid.MIN_OBS
            if cohorts.reset_index().groupby(cols)['index'].count().lt(minobs).any():
                err = ("Some cohort groups have too few observations to be measured."
                       + f" At least {minobs} are required for each group.")
                raise ValidationError(err)
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


class FairRanges():
    __diffs = ["auc difference" , "balanced accuracy difference",
                "equalized odds difference", "fpr diff", "tpr diff", "ppv diff"
                "positive predictive parity difference",
                "statistical parity difference", "mean prediction difference",
                "mae difference", "r2 difference"]
    __ratios = ["balanced accuracy ratio", "disparate impact ratio ",
                "equalized odds ratio", "fpr ratio", "tpr ratio", "ppv ratio",
                "mean prediction ratio", "mae ratio", "r2 ratio"]
    __stats =  ['consistency score']

    def __init__(self):
        pass

    def mad(self, arr):
        return np.median(np.abs(arr - np.median(arr)))

    def load_fair_ranges(self, custom_ranges:"dict[str, tuple[Number, Number]]"=None,
                         y_true:valid.ArrayLike=None, y_pred:valid.ArrayLike=None):
        """
        Args:
            custom_ranges (dict): a  dict whose keys are present among the measures
                in df and whose values are tuples containing the (lower, upper)
                bounds to the "fair" range. If None, uses default boundaries and
                will skip difference measures for regressions models. Default is None.
            y_true (array-like 1D, optional): Sample targets. Defaults to None.
            y_pred (array-like 1D, optional): Sample target predictions. Defaults to None.
        """
        #Load generic defaults
        bnds = self.default_boundaries()

        # Update with specific regression boundaries if possible
        if y_true is not None and y_pred is not None:
            y = prep.prep_arraylike(y_true, "y")
            yh = prep.prep_arraylike(y_pred, "y")
            aerr = (y - yh).abs()
            # Use np.concatenate to combine values s.t. ignore column names
            all_vals = y.append(yh)
            bnds['mean prediction difference'] = self.__calc_diff_range(all_vals)
            bnds['mae difference'] = self.__calc_diff_range(aerr)
            #import pdb; pdb.set_trace()
        else:
            bnds.pop('mean prediction difference')
            bnds.pop('mae difference')
        #
        if custom_ranges is not None:
            if valid.is_dictlike(custom_ranges):
                for k, v in custom_ranges.items():
                    bnds[str(k).lower()] = v
            else:
                raise TypeError(
                    "custom boundaries must be dict-like object if defined")
        #
        available_measures = self.__diffs + self.__ratios + self.__stats
        valid.validate_fair_boundaries(bnds, available_measures)
        return bnds

    def __calc_diff_range(self, ser:pd.Series):
        s_range = np.max(ser) - np.min(ser)
        s_bnd = 0.1*s_range
        if s_bnd == 0 or np.isnan(s_bnd):
            raise ValidationError(
                "Error computing fair boundaries. Verify targets and predictions.")
        return (-s_bnd, s_bnd)

    def default_boundaries(self, diff_bnd=(-0.1, 0.1), rto_bnd=(0.8, 1.2)):
        default = {"consistency score": (0.7, 1)}
        for d in self.__diffs:
            default[d] = diff_bnd
        for r in self.__ratios:
            default[r] = rto_bnd
        return default


class Flagger():

    def __init__(self):
        self.reset()

    def apply_flag(self, df, caption="", sig_fig=4, as_styler=True,
                   boundaries=None):
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
        self.reset()
        self.__set_df(df)
        self.__set_labels()
        self.__set_boundaries(boundaries)
        #
        if caption is None:
            caption = "Fairness Measures"

        # bools are treated as a subclass of int, so must test for both
        if not isinstance(sig_fig, int) or isinstance(sig_fig, bool):
            raise ValueError(f"Invalid value of significant figure: {sig_fig}")
        if self.label_type == "index":
            styled = self.df.style.set_caption(caption
                                 ).apply(self.__colors, axis=1)
        else:
            # pd.Styler doesn't support non-unique indices
            if len(self.df.index.unique()) <  len(self.df):
                self.df.reset_index(inplace=True)
            styled = self.df.style.set_caption(caption
                                 ).apply(self.__colors, axis=0)
        # Styler will reset precision to 6 sig figs
        styled = styled.set_precision(sig_fig)
        #
        setattr(styled, "fair_ranges", self.boundaries)
        # return pandas styler if requested
        if as_styler:
            return styled
        else:
            return HTML(styled.render())

    def reset(self):
        """ Clears the __Flagger settings
        """
        self.boundaries = None
        self.df = None
        self.flag_type = "background-color"
        self.flag_color = "magenta"
        self.labels = None
        self.label_type = None

    def __colors(self, vals):
        """ Returns a list containing the color settings for difference
            measures found to be OOR
        """
        def is_oor(name, val):
            low = self.boundaries[name.lower()][0]
            high = self.boundaries[name.lower()][1]
            return bool(not low < val < high and not np.isnan(val))
        #
        clr = f'{self.flag_type}:{self.flag_color}'
        if self.label_type == "index":
            name = vals.name[1].lower()
        else:
            name = vals.name.lower()
        #
        if name not in self.boundaries.keys():
            return ['' for v in  vals]
        else:
            clr = f'{self.flag_type}:{self.flag_color}'
            return [clr if is_oor(name, v) else ""  for v in vals]

    def __set_boundaries(self, boundaries):
        lbls = [l.lower() for l in self.labels]
        if boundaries is None:
            bnd = FairRanges().load_fair_ranges()
            # Mismatched keys may lead to errant belief that a measure is within
            # the fair range when actually there was a mistake (eg. key was
            # misspelled)
            boundaries = {k:v for k,v in bnd.items() if k.lower() in lbls}
        valid.validate_fair_boundaries(boundaries, lbls)
        self.boundaries = boundaries

    def __set_df(self, df):
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

    def __set_labels(self):
        """ Determines the locations of the strings containing measure names
            (within df); then reutrns a list of those measure names along
            with their location (one of ["columns", "index"]).
        """
        if self.df is None:
            pass
        else:
            try:
                labels = self.df.index.get_level_values(1)
                if type(labels) == pd.core.indexes.numeric.Int64Index:
                    label_type = "columns"
                else:
                    label_type = "index"
            except:
                label_type = "columns"
                labels = self.df.columns.tolist()
            self.label_type, self.labels = label_type, labels

