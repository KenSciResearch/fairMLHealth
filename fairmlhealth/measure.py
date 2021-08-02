# -*- coding: utf-8 -*-
"""
Tools producing analytical tables of fairness, bias, or model performance measures
Contributors:
    camagallen <ca.magallen@gmail.com>
"""


import aif360.sklearn.metrics as aif
from functools import reduce
from IPython.display import HTML
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                            balanced_accuracy_score)
from scipy import stats
from warnings import catch_warnings, simplefilter, warn, filterwarnings

# Tutorial Libraries
from . import __performance_metrics as pmtrc, __fairness_metrics as fcmtrc
from .__fairness_metrics import eq_odds_diff, eq_odds_ratio
from .__preprocessing import (standard_preprocess, stratified_preprocess,
                              analytical_labels, y_cols)
from .__validation import ValidationError
from .__utils import format_errwarn, iterate_cohorts, limit_alert, Flagger


# ToDo: find better solution for these warnings
filterwarnings('ignore', module='pandas')
filterwarnings('ignore', module='sklearn')


def bias(X, y_true, y_pred, features:list=None, pred_type="classification",
                sig_fig:int=4, flag_oor=False, **kwargs):
    """ Generates a table of stratified bias metrics

    Args:
        X (array-like): Sample features
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target predictions
        features (list): columns in X to be assessed if not all columns.
            Defaults to None (i.e. all columns).
        pred_type (str, optional): One of "classification" or "regression".
            Defaults to "classification".
        flag_oor (bool): if true, will apply flagging function to highlight
            fairness metrics which are considered to be outside the "fair" range
            (Out Of Range). Defaults to False.
        priv_grp (int): Specifies which label indicates the privileged
            group. Defaults to 1.

    Raises:
        ValueError

    Returns:
        pandas Data Frame
    """
    validtypes = ["classification", "regression"]
    if pred_type not in validtypes:
        raise ValueError(f"Summary table type must be one of {validtypes}")
    if pred_type == "classification":
        df = __classification_bias(X=X, y_true=y_true, y_pred=y_pred,
                                          features=features, **kwargs)
    elif pred_type == "regression":
        df = __regression_bias(X=X, y_true=y_true, y_pred=y_pred,
                                      features=features, **kwargs)
    #
    if flag_oor:
        df = flag(df, sig_fig=sig_fig)
    else:
        df = df.round(sig_fig)
    return df


def data(X, Y, features:list=None, targets:list=None, add_overview=True,
                sig_fig:int=4):
    """
    Generates a table of stratified data metrics

    Args:
        X (pandas dataframe or compatible object): sample data to be assessed
        Y (pandas dataframe or compatible object): sample targets to be
            assessed. Note that any observations with missing targets will be
            ignored.
        features (list): columns in X to be assessed if not all columns.
            Defaults to None (i.e. all columns).
        targets (list): columns in Y to be assessed if not all columns.
            Defaults to None (i.e. all columns).
        add_overview (bool): whether to add a summary row with metrics for
            "ALL FEATURES" and "ALL VALUES" as a single group. Defaults to True.

    Requirements:
        Each feature must be discrete to run stratified analysis. If any data
        are not discrete and there are more than 11 values, the tool will
        reformat those data into quantiles

    Returns:
        pandas Data Frame
    """
    #
    def entropy(x):
        # use float type for x to avoid boolean interpretation issues if any
        #   pd.NA (integer na) values are prent
        try:
            _x = x.astype(float)
        except ValueError: # convert strings to numeric categories
            _x = pd.Categorical(x).codes
        return stats.entropy(np.unique(_x, return_counts=True)[1], base=2)

    def __data_dict(x, col):
        ''' Generates a dictionary of statistics '''
        res = {'Obs.': x.shape[0]}
        if not x[col].isna().all():
            res[f"{col} Mean"] = x[col].mean()
            res[f"{col} Median"] = x[col].median()
            res[f"{col} Std. Dev."] = x[col].std()
        else:
            # Force addition of second column to ensure proper formatting
            # as pandas series
            for c in [f"{col} Mean", f"{col} Median", f"{col} Std. Dev."]:
                res[c] = np.nan
        return res
    #
    X_df = stratified_preprocess(X=X, features=features)
    Y_df = stratified_preprocess(X=Y, features=targets)
    if X_df.shape[0] != Y_df.shape[0]:
        raise ValidationError("Number of observations mismatch between X and Y")
    #
    if features is None:
        features = X_df.columns.tolist()
    strat_feats = [f for f in features if f in X_df.columns]
    limit_alert(strat_feats, item_name="features")
    #
    if targets is None:
        targets = Y_df.columns.tolist()
    strat_targs = [t for t in targets if t in Y_df.columns]
    limit_alert(strat_targs, item_name="targets", limit=3,
                issue="This may make the output difficult to read.")
    #
    res = []
    # "Obs."" included in index for ease of calculation
    ix_cols = ['Feature Name', 'Feature Value', 'Obs.']
    for t in strat_targs:
        X_df[t] = Y_df[t]
        feat_subset = [f for f in strat_feats if f != t]
        if not any(feat_subset):
            continue
        res_t = __apply_featureGroups(feat_subset, X_df, __data_dict, t)
        # convert id columns to strings to work around bug in pd.concat
        for m in ix_cols:
            res_t[m] = res_t[m].astype(str)
        res.append(res_t.set_index(ix_cols))
    results = pd.concat(res, axis=1).reset_index()
    #
    results['Obs.'] = results['Obs.'].astype(float).astype(int)
    results['Value Prevalence'] = results['Obs.']/X_df.shape[0]
    n_missing = X_df[strat_feats].replace('nan', np.nan).isna().sum().reset_index()
    n_missing.columns = ['Feature Name', 'Missing Values']
    entropy = X_df[strat_feats].apply(axis=0, func=entropy).reset_index()
    entropy.columns = ['Feature Name', 'Entropy']
    results = results.merge(n_missing, how='left', on='Feature Name'
                    ).merge(entropy, how='left', on='Feature Name')
    #
    if add_overview:
        res = []
        for i, t in enumerate(strat_targs):
            res_t = pd.DataFrame(__data_dict(X_df, t), index=[0])
            res.append(res_t.set_index('Obs.'))
        overview = pd.concat(res, axis=1).reset_index()
        N_feat = len(strat_feats)
        N_missing = n_missing['Missing Values'].sum()
        N_obs = X_df.shape[0]
        overview['Feature Name'] = "ALL FEATURES"
        overview['Feature Value'] = "ALL VALUES"
        overview['Missing Values'] = N_missing,
        overview['Value Prevalence'] = (N_obs*N_feat-N_missing)/(N_obs*N_feat)
        rprt = pd.concat([overview, results], axis=0, ignore_index=True)
    else:
        rprt = results
    #
    rprt = __sort_table(rprt)
    rprt = rprt.round(sig_fig)
    return rprt


def flag(df, caption:str="", sig_fig:int=4, as_styler:bool=True):
    """ Generates embedded html pandas styler table containing a highlighted
        version of a model comparison dataframe

    Args:
        df (pandas dataframe): Model comparison dataframe (see)
        caption (str, optional): Optional caption for table. Defaults to "".
        as_styler (bool, optional): If True, returns a pandas Styler of the
            highlighted table (to which other styles/highlights can be added).
            Otherwise, returns the table as an embedded HTML object. Defaults
            to False .

    Returns:
        Embedded html or pandas.io.formats.style.Styler
    """
    return Flagger().apply_flag(df, caption, sig_fig, as_styler)


def performance(X, y_true, y_pred, y_prob=None, features:list=None,
                      pred_type="classification", sig_fig:int=4,
                      add_overview=True):
    """ Generates a table of stratified performance metrics

    Args:
        X (pandas dataframe or compatible object): sample data to be assessed
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target predictions
        y_prob (array-like, 1-D): Sample target probabilities. Defaults to None.
        features (list): columns in X to be assessed if not all columns.
            Defaults to None (i.e. all columns).
        pred_type (str, optional): One of "classification" or "regression".
            Defaults to "classification".
        add_overview (bool): whether to add a summary row with metrics for
            "ALL FEATURES" and "ALL VALUES" as a single group. Defaults to True.

    Raises:
        ValueError

    Returns:
        pandas DataFrame
    """
    validtypes = ["classification", "regression"]
    if pred_type not in validtypes:
        raise ValueError(f"Summary table type must be one of {validtypes}")
    if pred_type == "classification":
        df = __strat_class_performance(X, y_true, y_pred, y_prob,
                                                   features, add_overview)
    elif pred_type == "regression":
        df = __strat_reg_performance(X, y_true, y_pred,
                                               features, add_overview)
    #
    df = df.round(sig_fig)
    return df


def summary(X, prtc_attr, y_true, y_pred, y_prob=None, flag_oor=True,
                   pred_type="classification", priv_grp=1, sig_fig:int=4,
                   **kwargs):
    """ Generates a summary of fairness measures for a set of predictions
    relative to their input data

    Args:
        X (array-like): Sample features
        prtc_attr (array-like, named): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target predictions
        y_prob (array-like, 1-D): Sample target probabilities. Defaults to None.
        flag_oor (bool): if true, will apply flagging function to highlight
            fairness metrics which are considered to be outside the "fair" range
            (Out Of Range). Defaults to False.
        pred_type (str, optional): One of "classification" or "regression".
            Defaults to "classification".
        priv_grp (int): Specifies which label indicates the privileged
            group. Defaults to 1.

    Raises:
        ValueError

    Returns:
        pandas DataFrame
    """
    validtypes = ["classification", "regression"]
    if pred_type not in validtypes:
        raise ValueError(f"Summary table type must be one of {validtypes}")
    if pred_type == "classification":
        df = __classification_summary(X=X, prtc_attr=prtc_attr, y_true=y_true,
                                      y_pred=y_pred, y_prob=y_prob,
                                        priv_grp=priv_grp, **kwargs)
    elif pred_type == "regression":
        df = __regression_summary(X=X, prtc_attr=prtc_attr, y_true=y_true,
                                  y_pred=y_pred, priv_grp=priv_grp, **kwargs)
    #
    if flag_oor:
        df = flag(df, sig_fig=sig_fig)
    else:
        df = df.round(sig_fig)
    return df


''' Private Functions '''


@format_errwarn
def __apply_featureGroups(features, df, func, *args):
    """ Iteratively applies a function across groups of each stratified feature,
    collecting errors and warnings to be displayed succinctly after processing

    Args:
        features (list): columns of df to be iteratively measured
        df (pd.DataFrame): data to be measured
        func (function): a function accepting *args and returning a dictionary

    Returns:
        pandas DataFrame: set of results for each feature-value
    """
    #
    errs = {}
    warns = {}
    res = []
    for f in features:
        # Data are expected in string format
        with catch_warnings(record=True) as w:
            simplefilter("always")
            try:
                grp = df.groupby(f)
                grp_res = grp.apply(lambda x: pd.Series(func(x, *args)))
            except BaseException as e:
                errs[f] = e
                continue
            if len(w) > 0:
                warns[f] = w
        grp_res = grp_res.reset_index().rename(columns={f: 'Feature Value'})
        grp_res.insert(0, 'Feature Name', f)
        res.append(grp_res)
    if len(res) == 0:
        results = pd.DataFrame(columns=['Feature Name', 'Feature Value'])
    else:
        results = pd.concat(res, ignore_index=True)
    return results, errs, warns


@format_errwarn
def __apply_biasGroups(features, df, func, yt, yh):
    """ Iteratively applies a function across groups of each stratified feature,
        collecting errors and warnings to be displayed succinctly after processing.

    Args:
        features (list): columns of df to be iteratively measured
        df (pd.DataFrame): data to be measured
        func (function): a function accepting two array arguments for comparison
            (selected from df as yt and yh), as well as a pa_name (str) and
            priv_grp (int) which will be set by __apply_biasGroups. This function
            must return a dictionary.
        yt (string): name of column found in df containing target values
        yh (string): name of column found in df containing predicted values

    Returns:
        pandas DataFrame: set of results for each feature-value
    """
    #
    errs = {}
    warns = {}
    pa_name = 'prtc_attr'
    res = []
    for f in features:
        df[f] = df[f].astype(str)
        vals = sorted(df[f].unique().tolist())
        # AIF360 can't handle float types
        for v in vals:
            df[pa_name] = 0
            df.loc[df[f].eq(v), pa_name] = 1
            if v != "nan":
                df.loc[df[f].eq("nan"), pa_name] = np.nan
            # Nothing to measure if only one value is present (other than nan)
            if df[pa_name].nunique() == 1:
                continue
            # Data are expected in string format
            with catch_warnings(record=True) as w:
                simplefilter("always")
                subset = df.loc[df[pa_name].notnull(),
                                    [pa_name, yt, yh]].set_index(pa_name)
                try:
                    #
                    grp_res = func(subset[yt], subset[yh], pa_name, priv_grp=1)
                except BaseException as e:
                    errs[f] = e
                    continue
                if len(w) > 0:
                    warns[f] = w
            grp_res = pd.DataFrame(grp_res, index=[0])
            grp_res.insert(0, 'Feature Name', f)
            grp_res.insert(1, 'Feature Value', v)
            res.append(grp_res)
    if len(res) == 0:
        results = pd.DataFrame(columns=['Feature Name', 'Feature Value'])
    else:
        results = pd.concat(res, ignore_index=True)
    return results, errs, warns


@iterate_cohorts
def __classification_bias(*, X, y_true, y_pred, features:list=None, **kwargs):
    """ Generates a table of stratified fairness metrics metrics for each specified
        feature

        Note: named arguments are enforced to enable use of iterate_cohorts

    Args:
        df (pandas dataframe or compatible object): data to be assessed
        y_true (1D array-like): Sample target true values; must be binary values
        y_pred (1D array-like): Sample target predictions; must be binary values
        features (list): columns in df to be assessed if not all columns.
            Defaults to None.

    Requirements:
        Each feature must be discrete to run stratified analysis. If any data
        are not discrete and there are more than 11 values, the tool will
        reformat those data into quantiles
    """
    #
    if y_true is None or y_pred is None:
        msg = "Cannot assess fairness without both y_true and y_pred"
        raise ValueError(msg)
    #
    df = stratified_preprocess(X, y_true, y_pred, features=features)
    _y, _yh, _yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [_y, _yh, _yp] if n is not None]
    strat_feats = [f for f in df.columns.tolist() if f not in pred_cols]
    if any(y is None for y in [_y, _yh]):
        raise ValidationError("Cannot measure with undefined targets")
    limit_alert(strat_feats, item_name="features", limit=200)
    #
    results = __apply_biasGroups(strat_feats, df,
                                 __fair_classification_measures, _y, _yh)
    rprt = __sort_table(results)
    return rprt


def __classification_performance(x:pd.DataFrame, y:str, yh:str, yp:str=None):
    res = {'Obs.': x.shape[0],
        f'{y} Mean': x[y].mean(),
        f'{yh} Mean': x[yh].mean(),
        'TPR': pmtrc.true_positive_rate(x[y], x[yh]),
        'FPR': pmtrc.false_positive_rate(x[y], x[yh]),
        'Accuracy': pmtrc.accuracy(x[y], x[yh]),
        'Precision': pmtrc.precision(x[y], x[yh]),  # PPV
        'F1-Score': pmtrc.f1_score(x[y], x[yh])
        }
    if yp is not None:
        res['ROC AUC'] = pmtrc.roc_auc_score(x[y], x[yp])
        res['PR AUC'] = pmtrc.pr_auc_score(x[y], x[yp])
    return res


@iterate_cohorts
def __classification_summary(*, X, prtc_attr, y_true, y_pred, y_prob=None,
                             priv_grp=1, **kwargs):
    """ Returns a pandas dataframe containing fairness measures for the model
        results

        Note: named arguments are enforced to enable use of iterate_cohorts

    Args:
        X (array-like): Sample features
        prtc_attr (array-like, named): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target predictions
        y_prob (array-like, 1-D): Sample target probabilities
        priv_grp (int): Specifies which label indicates the privileged
            group. Defaults to 1.
    """
    #
    def update_summary(summary_dict, pa_name, y_true, y_pred, y_prob, priv_grp):
        """ Adds replaces measure keys with the names found in the literature

        Args:
            X (pandas DataFrame): Sample features
            pa_name (str):
            y_true (pandas DataFrame): Sample targets
            y_pred (pandas DataFrame): Sample target predictions
            y_prob (pandas DataFrame, optional): Sample target probabilities.
                Defaults to None.
            priv_grp (int): Specifies which label indicates the privileged
                    group. Defaults to 1.
        """
        name_update = {"Selection Diff":"Statistical Parity Difference",
                       "Selection Ratio": "Disparate Impact Ratio",
                       "PPV Diff": "Positive Predictive Parity Difference",
                       "PPV Ratio": "Positive Predictive Parity Ratio"
                       }
        drop_keys = ['TPR Ratio', 'TPR Diff', 'FPR Ratio', 'FPR Diff']
        for k in name_update.keys():
            val = summary_dict.pop(k)
            summary_dict[name_update[k]] = val
        for k in drop_keys:
            summary_dict.pop(k)
        summary_dict['Equalized Odds Difference'] = eq_odds_diff(y_true, y_pred,
                                                            prtc_attr=pa_name)
        summary_dict['Equalized Odds Ratio'] = eq_odds_ratio(y_true, y_pred,
                                                        prtc_attr=pa_name)
        if y_prob is not None:
            try:
                summary_dict['AUC Difference'] = \
                    aif.difference(pmtrc.roc_auc_score, y_true, y_prob,
                                    prot_attr=pa_name, priv_group=priv_grp)
            except:
                pass
        return summary_dict

    # Validate and Format Arguments
    if not isinstance(priv_grp, int):
        raise ValueError("priv_grp must be an integer value")
    X, prtc_attr, y_true, y_pred, y_prob = \
        standard_preprocess(X, prtc_attr, y_true, y_pred, y_prob, priv_grp)
    pa_name = prtc_attr.columns.tolist()[0]

    # Temporarily prevent processing for more than 2 classes
    # ToDo: enable multiclass
    n_class = np.unique(np.append(y_true.values, y_pred.values)).shape[0]
    if n_class == 2:
        summary_type = "binary"
    else:
        summary_type = "multiclass"
        raise ValueError(
            "tool cannot yet process multiclass classification models")
    # Generate a dictionary of measure values to be converted t a dataframe
    labels = analytical_labels(summary_type)
    summary = __fair_classification_measures(y_true, y_pred, pa_name, priv_grp)
    measures = {labels['gf_label']: update_summary(summary, pa_name, y_true,
                                                   y_pred, y_prob, priv_grp),
                labels['dt_label']: __value_prevalence(y_true, priv_grp)
                }
    if not kwargs.pop('skip_if', False):
        measures[labels['if_label']] = \
            __similarity_measures(X, pa_name, y_true, y_pred)
    if not kwargs.pop('skip_performance', False):
        _y, _yh= y_true.columns[0], y_pred.columns[0]
        X[_y], X[_yh] = y_true.values, y_pred.values
        measures[labels['mp_label']] = __classification_performance(X, _y, _yh)
    # Convert scores to a formatted dataframe and return
    df = pd.DataFrame.from_dict(measures, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    output = __format_summary(df, summary_type)
    return output


def __fair_classification_measures(y_true, y_pred, pa_name, priv_grp=1):
    """ Returns a dict of measure values
    """
    def predmean(_, y_pred, *args): return np.mean(y_pred.values)
    #
    measures = {}
    measures['Selection Ratio'] = aif.ratio(predmean, y_true, y_pred,
                                            prot_attr=pa_name,
                                            priv_group=priv_grp)
    measures['PPV Ratio'] = \
        fcmtrc.ppv_ratio(y_true, y_pred, pa_name, priv_grp)
    measures['TPR Ratio'] =  \
        fcmtrc.tpr_ratio(y_true, y_pred, pa_name, priv_grp)
    measures['FPR Ratio'] =  \
        fcmtrc.fpr_ratio(y_true, y_pred, pa_name, priv_grp)
    #
    measures['Selection Diff'] = aif.difference(predmean, y_true, y_pred,
                                                prot_attr=pa_name,
                                                priv_group=priv_grp)
    measures['PPV Diff'] = fcmtrc.ppv_diff(y_true, y_pred, pa_name, priv_grp)
    measures['TPR Diff'] = fcmtrc.tpr_diff(y_true, y_pred, pa_name, priv_grp)
    measures['FPR Diff'] = fcmtrc.fpr_diff(y_true, y_pred, pa_name, priv_grp)
    measures['Balanced Accuracy Difference'] = \
            aif.difference(balanced_accuracy_score, y_true,
                           y_pred, prot_attr=pa_name, priv_group=priv_grp)
    measures['Balanced Accuracy Ratio'] = \
            aif.ratio(balanced_accuracy_score, y_true,
                       y_pred, prot_attr=pa_name, priv_group=priv_grp)
    return measures


def __fair_regression_measures(y_true, y_pred, pa_name, priv_grp=1):
    """ Returns dict of regression-specific fairness measures
    """
    def predmean(_, y_pred, *args): return np.mean(y_pred.values)
    def meanerr(y_true, y_pred, *args): return np.mean((y_pred - y_true).values)
    #
    measures = {}
    # Ratios
    measures['Mean Prediction Ratio'] = \
        aif.ratio(predmean, y_true, y_pred,prot_attr=pa_name, priv_group=priv_grp)
    measures['MAE Ratio'] = aif.ratio(mean_absolute_error, y_true, y_pred,
                                     prot_attr=pa_name, priv_group=priv_grp)
    # Differences
    measures['Mean Prediction Difference'] = \
        aif.difference(predmean, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    measures['MAE Difference'] = \
        aif.difference(mean_absolute_error, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    return measures


def __format_summary(df, summary_type):
    df.columns = ['Value']
    # Fix the order in which the metrics appear
    gfl, ifl, mpl, dtl = analytical_labels(summary_type)
    metric_order = {gfl: 0, ifl: 1, mpl: 2, dtl: 3}
    df.reset_index(inplace=True)
    df['sortorder'] = df['level_0'].map(metric_order)
    df = df.sort_values('sortorder').drop('sortorder', axis=1)
    # Fix Display Names
    df.set_index(['level_0', 'level_1'], inplace=True)
    df.rename_axis(('Metric', 'Measure'), inplace=True)
    return df


def __regression_performance(x:pd.DataFrame, y:str, yh:str):
    res = {'Obs.': x.shape[0],
            f'{y} Mean': x[y].mean(),
            f'{y} Std. Dev.': x[y].std(),
            f'{yh} Mean': x[yh].mean(),
            f'{yh} Std. Dev.': x[yh].std(),
            'Error Mean': (x[yh] - x[y]).mean(),
            'Error Std. Dev.': (x[yh] - x[y]).std(),
            'MAE': mean_absolute_error(x[y], x[yh]),
            'MSE': mean_squared_error(x[y], x[yh]),
            'Rsqrd': pmtrc.r_squared(x[y], x[yh])
            }
    return res


def __strat_class_performance(X, y_true, y_pred, y_prob=None,
                                        features:list=None, add_overview=True):
    """Generates a table of stratified performance metrics for each specified
        feature

    Args:
        df (pandas dataframe or compatible object): data to be assessed
        y_true (1D array-like): Sample target true values; must be binary values
        y_pred (1D array-like): Sample target predictions; must be binary values
        y_prob (1D array-like, optional): Sample target probabilities. Defaults
            to None.
        features (list): columns in df to be assessed if not all columns.
            Defaults to None.

    Returns:
        pandas DataFrame
    """
    #
    if y_true is None or y_pred is None:
        msg = "Cannot assess performance without both y_true and y_pred"
        raise ValueError(msg)
    #
    df = stratified_preprocess(X, y_true, y_pred, y_prob, features=features)
    _y, _yh, _yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [_y, _yh, _yp] if n is not None]
    strat_feats = [f for f in df.columns.tolist() if f not in pred_cols]
    if any(y is None for y in [_y, _yh]):
        raise ValidationError("Cannot measure with undefined targets")
    limit_alert(strat_feats, item_name="features")
    #
    results = __apply_featureGroups(strat_feats, df,
                                    __classification_performance, _y, _yh, _yp)
    if add_overview:
        overview = {'Feature Name': "ALL FEATURES",
                    'Feature Value': "ALL VALUES"}
        o_dict = __classification_performance(df, _y, _yh, _yp)
        for k, v in o_dict.items():
            overview[k] = v
        overview_df = pd.DataFrame(overview, index=[0])
        rprt = pd.concat([overview_df, results], axis=0, ignore_index=True)
    else:
        rprt = results
    rprt = __sort_table(rprt)
    return rprt


def __strat_reg_performance(X, y_true, y_pred, features:list=None,
                                    add_overview=True):
    """
    Generates a table of stratified performance metrics for each specified
    feature

    Args:
        df (pandas dataframe or compatible object): data to be assessed
        y_true (1D array-like): Sample target true values
        y_pred (1D array-like): Sample target predictions
        features (list): columns in df to be assessed if not all columns.
            Defaults to None.

    Requirements:
        Each feature must be discrete to run stratified analysis. If any data
        are not discrete and there are more than 11 values, the tool will
        reformat those data into quantiles
    """
    #
    if y_true is None or y_pred is None:
        msg = "Cannot assess performance without both y_true and y_pred"
        raise ValueError(msg)
    #
    df = stratified_preprocess(X, y_true, y_pred, features=features)
    _y, _yh, _yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [_y, _yh, _yp] if n is not None]
    strat_feats = [f for f in df.columns.tolist() if f not in pred_cols]
    if any(y is None for y in [_y, _yh]):
        raise ValidationError("Cannot measure with undefined targets")
    limit_alert(strat_feats, item_name="features")
    #
    results = __apply_featureGroups(strat_feats, df,
                                    __regression_performance, _y, _yh)
    if add_overview:
        overview = {'Feature Name': "ALL FEATURES",
                    'Feature Value': "ALL VALUES"}
        o_dict = __regression_performance(df, _y, _yh)
        for k, v in o_dict.items():
            overview[k] = v
        overview_df = pd.DataFrame(overview, index=[0])
        rprt = pd.concat([overview_df, results], axis=0, ignore_index=True)
    else:
        rprt = results
    rprt = __sort_table(rprt)
    return rprt


@iterate_cohorts
def __regression_bias(*, X, y_true, y_pred, features:list=None, **kwargs):
    """
    Generates a table of stratified fairness metrics metrics for each specified
    feature

    Note: named arguments are enforced to enable use of iterate_cohorts

    Args:
        df (pandas dataframe or compatible object): data to be assessed
        y_true (1D array-like): Sample target true values
        y_pred (1D array-like): Sample target predictions
        features (list): columns in df to be assessed if not all columns.
            Defaults to None.

    """
    if y_true is None or y_pred is None:
        msg = "Cannot assess fairness without both y_true and y_pred"
        raise ValueError(msg)
    #
    df = stratified_preprocess(X, y_true, y_pred, features=features)
    _y, _yh, _yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [_y, _yh, _yp] if n is not None]
    strat_feats = [f for f in df.columns.tolist() if f not in pred_cols]
    if any(y is None for y in [_y, _yh]):
        raise ValidationError("Cannot measure with undefined targets")
    limit_alert(strat_feats, item_name="features", limit=200)
    #
    results = __apply_biasGroups(strat_feats, df,
                                 __fair_regression_measures, _y, _yh)
    rprt = __sort_table(results)
    return rprt


@iterate_cohorts
def __regression_summary(*, X, prtc_attr, y_true, y_pred, priv_grp=1,
                         **kwargs):
    """ Returns a pandas dataframe containing fairness measures for the model
        results

        Note: named arguments are enforced to enable @iterate_cohorts

    Args:
        X (array-like): Sample features
        prtc_attr (array-like, named): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (array-like, 1-D): Sample targets
        y_pred (array-like, 1-D): Sample target probabilities
        priv_grp (int): Specifies which label indicates the privileged
            group. Defaults to 1.
    """
    #
    # Validate and Format Arguments
    if not isinstance(priv_grp, int):
        raise ValueError("priv_grp must be an integer value")
    X, prtc_attr, y_true, y_pred, _ = \
        standard_preprocess(X, prtc_attr, y_true, y_pred, priv_grp=priv_grp)
    pa_name = prtc_attr.columns.tolist()[0]
    #
    grp_vals = __fair_regression_measures(y_true, y_pred, pa_name, priv_grp=priv_grp)
    #
    dt_vals = __value_prevalence(y_true, priv_grp)
    if not kwargs.pop('skip_if', False):
        if_vals = __similarity_measures(X, pa_name, y_true, y_pred)

    mp_vals = {}
    if not kwargs.pop('skip_performance', False):
        # y_true and y_pred will have the same name after going through prep
        rprt_input = pd.concat([y_true, y_pred], axis=1)
        colnames = [rprt_input.columns[0], "Prediction"]
        rprt_input.columns = colnames
        perf_rep = __regression_performance(rprt_input, colnames[0], colnames[1])
        strat_tbl = pd.DataFrame().from_dict(perf_rep, orient='index'
                          ).rename(columns={0: 'Score'})
        for row in strat_tbl.iterrows():
            mp_vals[row[0]] = row[1]['Score']
    # Convert scores to a formatted dataframe and return
    labels = analytical_labels("regression")
    measures = {labels['gf_label']: grp_vals,
                labels['if_label']: if_vals,
                labels['mp_label']: mp_vals,
                labels['dt_label']: dt_vals}
    df = pd.DataFrame.from_dict(measures, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    output = __format_summary(df, "regression")
    return output


def __similarity_measures(X, pa_name:str, y_true:pd.Series, y_pred:pd.Series):
    """ Returns dict of similarity-based fairness measures
    """
    if_vals = {}
    # consistency_score raises error if null values are present in X
    if X.notnull().all().all():
        if_vals['Consistency Score'] = \
            aif.consistency_score(X, y_pred.iloc[:, 0])
    else:
        msg = "Cannot calculate consistency score. Null values present in data."
        logging.warning(msg)
    # Other aif360 metrics (not consistency) can handle null values
    if_vals['Between-Group Gen. Entropy Error'] = \
        aif.between_group_generalized_entropy_error(y_true, y_pred,
                                                        prot_attr=pa_name)
    return if_vals


def __sort_table(strat_tbl):
    """ Sorts columns in standardized order

    Args:
        strat_tbl (pd.DataFrame): any of the stratified tables produced by this
        module

    Returns:
        pandas DataFrame: sorted strat_tbl
    """
    _y = y_cols()['disp_names']['yt']
    _yh = y_cols()['disp_names']['yh']
    head_names = ['Feature Name', 'Feature Value', 'Obs.',
                 f'{_y} Mean', f'{_yh} Mean']
    head_cols = [c for c in head_names if c in strat_tbl.columns]
    tail_cols = sorted([c for c in strat_tbl.columns if c not in head_cols])
    return strat_tbl[head_cols + tail_cols]


def __value_prevalence(y_true, priv_grp):
    """ Returns a dictionary of data metrics applicable to evaluation of
        fairness

    Args:
        y_true (pandas DataFrame): Sample targets
        priv_grp (int): Specifies which label indicates the privileged
                group. Defaults to 1.
    """
    dt_vals = {}
    prev = round(100*y_true[y_true.eq(priv_grp)].sum()/y_true.shape[0])
    if not isinstance(prev, float):
        prev = prev[0]
    dt_vals['Prevalence of Privileged Class (%)'] = prev
    return dt_vals
