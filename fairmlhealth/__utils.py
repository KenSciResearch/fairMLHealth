"""
Back-end functions used throughout the library, many of which assume that inputs
have been validated
"""
from numbers import Number
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Tuple
from warnings import warn, catch_warnings, filterwarnings

from . import __preprocessing as prep, __validation as valid
from .__validation import ArrayLike, MatrixLike, ValidationError


def epsilon():
    """ error value used to prevent 'division by 0' errors """
    return np.finfo(np.float64).eps


def format_errwarn(func: Callable[[], Tuple[Any, Dict, Dict]]):
    """ Wraps a function returning some result with dictionaries for errors and
        warnings, then formats those errors and warnings as grouped warnings.
        Used for analytical functions to skip errors (and warnings) while
        providing

    Args:
        func (function): any function returning a tuple of format:
            (result, error_dictionary, warning_dictionary). Note that
            dictionaries should be of form {<column or id>:<message>}

    Returns:
        Any (result): the first member of the tuple returned by func
    """

    def format_info(dict):
        info_dict = {}
        for colname, err_wrn in dict.items():
            _ew = list(set(err_wrn)) if isinstance(err_wrn, list) else [err_wrn]
            for m in _ew:
                m = getattr(m, "message") if "message" in dir(m) else str(m)
                if m in info_dict.keys():
                    info_dict[m].append(colname)
                else:
                    info_dict[m] = [colname]
        info_dict = {ew: list(set(c)) for ew, c in info_dict.items()}
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


def iterate_cohorts(func: Callable[[], pd.DataFrame]):
    """ Runs the function for each cohort subset

    Args:
        func (function): the function to iterate

    Returns:
        pd.DataFrame: cohort-iterated version of the output

    """

    def prepend_cohort(df: pd.DataFrame, new_ix: ArrayLike):
        """ Attaches cohort information to the far left of the dataframe, adjusting
            by type. Cohorts are added to the index to enable flagging for
            cohorted summary tables (pd.Styler fails with non-unique indices)

        Returns:
            pd.DataFrame
        """
        idx = df.index.to_frame().rename(columns={0: "__index"})
        # add padding to shift start location if prepending a summary table
        ixpad = 0
        if df.index.names == ["Metric", "Measure"]:
            ixpad = 2
        # Attach new_ix data to the index
        if idx.any().any():
            for l, i in enumerate(new_ix):
                idx.insert(l + ixpad, i[0], i[1])
            if "__index" in idx.columns:
                idx.drop("__index", axis=1, inplace=True)
        else:
            pass  # No data in dataframe
        df.index = pd.MultiIndex.from_frame(idx)
        return df

    def subset(data: pd.DataFrame, idxs: ArrayLike):
        if data is not None:
            return data.loc[
                idxs,
            ]
        else:
            return None

    def add_group_info(errant_list, grp_cols, grp_vals):
        grpname = ""
        for i, c in enumerate(grp_cols):
            if len(grpname) > 0:
                grpname += " & "
            grpname += f"{c} {grp_vals[i]}"
        errant_list.append(grpname)
        return None

    def wrapper(cohort_labels: MatrixLike = None, **kwargs):
        """ Iterates for each cohort subset

        Args:
            cohorts (array-like or dataframe, optional): Groups by which to
                subset the data. Defaults to None.

        Returns:
            pandas DataFrame
        """
        # Run preprocessing to facilitate subsetting
        X = kwargs.get("X", None)
        Y = kwargs.get("Y", None)
        y_true = kwargs.get("y_true", None)
        y_pred = kwargs.get("y_pred", None)
        y_prob = kwargs.get("y_prob", None)
        prtc_attr = kwargs.get("prtc_attr", None)
        X, prtc_attr, y_true, y_pred, y_prob = prep.standard_preprocess(
            X, prtc_attr, y_true, y_pred, y_prob
        )
        #
        if cohort_labels is not None:
            #
            valid.validate_data(
                cohort_labels, name="cohort_labels", expected_len=X.shape[0]
            )
            cohorts = prep.prep_data(cohort_labels)
            #
            cix = cohorts.index
            grp_cols = cohorts.columns.tolist()
            cgrp = cohorts.groupby(grp_cols)
            valid.limit_alert(
                cgrp,
                "permutations of cohorts",
                8,
                issue="This may slow processing time and reduce utility.",
            )
            #
            results = []
            errant_list = []
            minobs = valid.MIN_OBS
            for k in cgrp.groups.keys():
                grp_vals = cgrp.get_group(k)[grp_cols].head(1).values[0]
                # Subset each argument to those observations matching the group
                ixs = cix.astype("int64").isin(cgrp.groups[k])
                # Test subset for sufficent data. cohorts with too few observations
                # will cause terminating errors.
                x = subset(X, ixs)
                if x.shape[0] < minobs:
                    add_group_info(errant_list, grp_cols, grp_vals)
                    continue
                # subset remaining arguments
                y = subset(Y, ixs)
                yt = subset(y_true, ixs)
                yh = subset(y_pred, ixs)
                yp = subset(y_prob, ixs)
                pa = subset(prtc_attr, ixs)
                #
                new_args = ["X", "Y", "y_true", "y_pred", "y_prob", "prtc_attr"]
                sub_args = {k: v for k, v in kwargs.items() if k not in new_args}
                try:
                    df = func(
                        X=x,
                        Y=y,
                        y_true=yt,
                        y_pred=yh,
                        y_prob=yp,
                        prtc_attr=pa,
                        **sub_args,
                    )
                except ValidationError as e:
                    # Skip groups for which subsetting led to insufficient target representation
                    m = getattr(e, "message") if "message" in dir(e) else None
                    if m == "Only one target classification found.":
                        df = pd.DataFrame()
                    else:
                        raise ValidationError(m)
                # Empty dataframes indicate issues with evaluation of the function,
                # likely caused by presence of i.e. only one feature-value pair.
                if len(df) == 0:
                    add_group_info(errant_list, grp_cols, grp_vals)
                ix = [(c, grp_vals[i]) for i, c in enumerate(grp_cols)]
                df = prepend_cohort(df, ix)
                results.append(df)
            # Report issues to the user
            if len(errant_list) == len(cgrp.groups.keys()):
                msg = (
                    "Invalid cohort specification. None of the cohort subsets "
                    + f"could be processed. least {minobs} observations."
                )
                raise ValidationError(msg)
            elif any(errant_list):
                msg = (
                    "Could not evaluate function for group(s): "
                    + "{errant_list}. This is commonly caused when there is "
                    + " too little data or there is only a single "
                    + "feature-value pair is available in a given cohort. "
                    + f"Each cohort must have  {minobs} observations."
                )
                warn(msg)
            else:
                pass
            # Combine results and return
            output = pd.concat(results, axis=0)
            return output
        else:
            return func(**kwargs)

    return wrapper


class FairRanges:
    __diffs = [
        "auc difference",
        "balanced accuracy difference",
        "equal odds difference",
        "fpr diff",
        "tpr diff",
        "ppv diff" "positive predictive parity difference",
        "statistical parity difference",
        "mean prediction difference",
        "mae difference",
        "r2 difference",
    ]
    __ratios = [
        "balanced accuracy ratio",
        "disparate impact ratio",
        "selection ratio",
        "equal odds ratio",
        "fpr ratio",
        "tpr ratio",
        "ppv ratio",
        "mean prediction ratio",
        "mae ratio",
        "r2 ratio",
    ]
    __stats = ["consistency score"]

    def __init__(self):
        pass

    def mad(self, arr: ArrayLike):
        return np.median(np.abs(arr - np.median(arr)))

    def load_fair_ranges(
        self,
        custom_ranges: Dict[str, Tuple[Number, Number]] = None,
        y_true: ArrayLike = None,
        y_pred: ArrayLike = None,
    ):
        """
        Args:
            custom_ranges (dict): a  dict whose keys are present among the measures
                in df and whose values are tuples containing the (lower, upper)
                bounds to the "fair" range. If None, uses default boundaries and
                will skip difference measures for regressions models. Default is None.
            y_true (ArrayLike, optional): Sample targets. Defaults to None.
            y_pred (ArrayLike, optional): Sample target predictions. Defaults to None.
        """
        # Load generic defaults
        bnds = self.default_boundaries()

        # Update with specific regression boundaries if possible
        if y_true is not None and y_pred is not None:
            y = prep.prep_arraylike(y_true, "y")
            yh = prep.prep_arraylike(y_pred, "y")
            aerr = (y - yh).abs()
            # Use np.concatenate to combine values s.t. ignore column names
            all_vals = y.append(yh)
            bnds["mean prediction difference"] = self.__calc_diff_range(all_vals)
            bnds["mae difference"] = self.__calc_diff_range(aerr)
        else:
            bnds.pop("mean prediction difference")
            bnds.pop("mae difference")
        #
        if custom_ranges is not None:
            if valid.is_dictlike(custom_ranges):
                for k, v in custom_ranges.items():
                    bnds[str(k).lower().strip()] = v
            else:
                raise TypeError("custom boundaries must be dict-like object if defined")
        #
        available_measures = self.__diffs + self.__ratios + self.__stats
        valid.validate_fair_boundaries(bnds, available_measures)
        return bnds

    def __calc_diff_range(self, ser: pd.Series):
        s_range = np.max(ser) - np.min(ser)
        s_bnd = 0.1 * s_range
        if s_bnd == 0 or np.isnan(s_bnd):
            raise ValidationError(
                "Error computing fair boundaries. Verify targets and predictions."
            )
        return (-s_bnd, s_bnd)

    def default_boundaries(
        self,
        diff_bnd: Tuple[Number, Number] = (-0.1, 0.1),
        rto_bnd: Tuple[Number, Number] = (0.8, 1.2),
    ):
        default = {}
        for d in self.__diffs:
            default[d] = diff_bnd
        for r in self.__ratios:
            default[r] = rto_bnd
        return default


class Flagger:
    __hex = {
        "magenta": "#d00095",
        "magenta_lt": "#ff05b8",
        "purple": "#947fed",
        "purple_lt": "#c2bae3",
    }

    def __init__(self):
        self.reset()

    def apply_flag(
        self,
        df: pd.DataFrame,
        caption: str = "",
        sig_fig: int = 4,
        as_styler: bool = True,
        boundaries: Dict[str, Tuple[Number, Number]] = None,
    ):
        """ Generates embedded html pandas styler table containing a highlighted
            version of a model comparison dataframe
        Args:
            df (pandas dataframe): report.compare dataframe
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
            styled = self.df.style.set_caption(caption).apply(self.__colors, axis=1)
        else:
            # pd.Styler doesn't support non-unique indices
            if len(self.df.index.unique()) < len(self.df):
                self.df.reset_index(inplace=True)
            styled = self.df.style.set_caption(caption).apply(self.__colors, axis=0)
        # Styler automatically sets precision to 6 sig figs
        with catch_warnings(record=False):
            filterwarnings("ignore", category=DeprecationWarning)
            styled = styled.set_precision(sig_fig)

        #
        setattr(styled, "fair_ranges", self.boundaries)
        # return pandas styler if requested
        if as_styler:
            return styled
        else:
            return styled.render()

    def reset(self):
        """ Clears the __Flagger settings
        """
        self.boundaries = None
        self.df = None
        self.flag_type = "background-color"
        self.flag_color = self.__hex["purple_lt"]
        self.labels = None
        self.label_type = None

    def __colors(self, vals: pd.Series):
        """ Returns a list containing the color settings for difference
            measures found to be OOR
        """

        def is_oor(name, val):
            low = self.boundaries[name.lower()][0]
            high = self.boundaries[name.lower()][1]
            return bool(not low < val < high and not np.isnan(val))

        #
        clr = f"{self.flag_type}:{self.flag_color}"
        if self.label_type == "index":
            name = vals.name[1].lower().strip()
        else:
            name = vals.name.lower().strip()
        #
        if name not in self.boundaries.keys():
            return ["" for v in vals]
        else:
            clr = f"{self.flag_type}:{self.flag_color}"
            return [clr if is_oor(name, v) else "" for v in vals]

    def __set_boundaries(
        self, custom_boundaries: Dict[str, Tuple[Number, Number]] = None
    ):
        lbls = [str(l).lower() for l in self.labels]
        if custom_boundaries is None:
            custom_boundaries = {}
        # FairRanges will automatically join defaults with custom boundaries
        bnd = FairRanges().load_fair_ranges(custom_boundaries)
        # Mismatched keys may lead to errant belief that a measure is within
        # the fair range when actually there was a mistake (eg. key was
        # misspelled)
        boundaries = {k: v for k, v in bnd.items() if k.lower().strip() in lbls}
        valid.validate_fair_boundaries(boundaries, lbls)
        self.boundaries = boundaries

    def __set_df(self, df: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            self.df = df.copy()
        else:
            err = "df must be a pandas DataFrame or Styler"
            try:
                if isinstance(df, pd.io.formats.style.Styler):
                    self.df = df.data.copy()
                else:
                    raise ValidationError(err)
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
            if isinstance(self.df.index, pd.MultiIndex):
                if "Measure" in self.df.index.names:
                    label_type = "index"
                    labels = self.df.index.get_level_values("Measure")
                else:
                    label_type = "columns"
                    labels = self.df.columns.tolist()
            else:
                label_type = "columns"
                labels = self.df.columns.tolist()

            self.label_type, self.labels = label_type, labels

