# -*- coding: utf-8 -*-
"""
Tools producing reports of fairness, bias, or model performance measures
Contributors:
    camagallen <ca.magallen@gmail.com>
"""


import aif360.sklearn.metrics as aif
from functools import reduce
from IPython.display import HTML
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, roc_auc_score,
                             balanced_accuracy_score, classification_report)
from scipy import stats
from warnings import catch_warnings, simplefilter, warn, filterwarnings

# Tutorial Libraries
from . import __performance_metrics as pmtrc, __fairness_metrics as fcmtrc
from .__fairness_metrics import eq_odds_diff, eq_odds_ratio
from .__preprocessing import (standard_preprocess, stratified_preprocess,
                              report_labels, y_cols)
from . import tutorial_helpers as helpers
from .__validation import ValidationError
from .utils import format_errwarn, iterate_cohorts


# ToDo: find better solution for these warnings
filterwarnings('ignore', module='pandas')
filterwarnings('ignore', module='sklearn')


''' Deprecated Public Functions '''

def classification_fairness(X, prtc_attr, y_true, y_pred, y_prob=None,
                            priv_grp=1, **kwargs):
    warn(
            "classification_fairness function will be deprecated in version " +
            "2.0. Use summary_report instead.",
            PendingDeprecationWarning
        )
    return __classification_summary(X, prtc_attr, y_true, y_pred, y_prob,
                                    priv_grp, **kwargs)


def regression_fairness(X, prtc_attr, y_true, y_pred, priv_grp=1, **kwargs):
    warn(
            "regression_fairness function will be deprecated in version " +
            "2.0. Use summary_report instead.", PendingDeprecationWarning
        )
    return __regression_summary(X, prtc_attr, y_true, y_pred, priv_grp, **kwargs)


''' Mini Reports '''

def classification_performance(y_true, y_pred, target_labels=None):
    """ Returns a pandas dataframe of the scikit-learn classification report,
        formatted for use in fairMLHealth tools
    Args:
        y_true (array): Target values. Must be compatible with model.predict().
        y_pred (array): Prediction values. Must be compatible with
            model.predict().
        target_labels (list of str): Optional labels for target values.
    """
    if target_labels is None:
        target_labels = [f"target = {t}" for t in set(y_true)]
    report = classification_report(y_true, y_pred, output_dict=True,
                                             target_names=target_labels)
    report = pd.DataFrame(report).transpose()
    # Move accuracy to separate row
    accuracy = report.loc['accuracy', :]
    report.drop('accuracy', inplace=True)
    report.loc['accuracy', 'accuracy'] = accuracy[0]
    return report


def regression_performance(y_true, y_pred):
    """ Returns a pandas dataframe of the regression performance metrics,
        similar to scikit's classification_performance
    Args:
        y_true (array): Target values. Must be compatible with model.predict().
        y_pred (array): Prediction values. Must be compatible with
            model.predict().
    """
    report = {}
    y = y_cols()['disp_names']['yt']
    yh = y_cols()['disp_names']['yh']
    report[f'{y} Mean'] = np.mean(y_true)
    report[f'{yh} Mean'] = np.mean(y_pred)
    report['scMAE'] = pmtrc.scMAE(y_true, y_pred)
    report['MSE'] = mean_squared_error(y_true, y_pred)
    report['MAE'] = mean_absolute_error(y_true, y_pred)
    report['Rsqrd'] = r2_score(y_true, y_pred)
    report = pd.DataFrame().from_dict(report, orient='index'
                          ).rename(columns={0: 'Score'})
    return report



''' Main Reports '''


def flag(df, caption="", as_styler=True):
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
    return __Flagger().apply_flag(df, caption, as_styler)


def bias_report(X, y_true, y_pred, features:list=None,
                pred_type="classification", flag_oor=False, priv_grp=1):
    """
    """
    validtypes = ["classification", "regression"]
    if pred_type not in validtypes:
        raise ValueError(f"Summary report type must be one of {validtypes}")
    if pred_type == "classification":
        df = __classification_bias_report(X, y_true, y_pred, features, priv_grp)
    elif pred_type == "regression":
        msg = "Regression reporting will be available in version 2.0"
        raise ValueError(msg)
        # df = __regression_bias_report(X, y_true, y_pred, features, priv_grp)
    if flag_oor:
        df = flag(df)
    return df


def data_report(X, Y, features:list=None, targets:list=None, add_overview=True):
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

    Requirements:
        Each feature must be discrete to run stratified analysis. If any data
        are not discrete and there are more than 11 values, the reporter will
        reformat those data into quantiles
    """
    #
    def entrp(x):
        ''' Calculates the entropy of a series '''
        return stats.entropy(np.unique(x, return_counts=True)[1], base=2)

    def __data_dict(x, col):
        ''' Generates a dictionary of statistics '''
        res = {'Obs.': x.shape[0]}
        if not x[col].isna().all():
            res[col + " Mean"] = x[col].mean()
            res[col + " Median"] = x[col].median()
            res[col + " Std. Dev."] = x[col].std()
        else:
            # Force addition of second column to ensure proper formatting
            # as pandas series
            for c in [col + " Mean", col + " Median", col + " Std. Dev."]:
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
    #
    if targets is None:
        targets = Y_df.columns.tolist()
    strat_targs = [t for t in targets if t in Y_df.columns]
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
    n_missing = X_df[strat_feats].replace('nan', np.nan
                                ).isna().sum().reset_index()
    n_missing.columns = ['Feature Name', 'Missing Values']
    entropy = X_df[strat_feats].apply(axis=0, func=entrp).reset_index()
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
    rprt = sort_report(rprt)
    return rprt



def performance_report(X, y_true, y_pred, y_prob=None, features:list=None,
                      pred_type="classification", add_overview=True):
    """
    """
    validtypes = ["classification", "regression"]
    if pred_type not in validtypes:
        raise ValueError(f"Summary report type must be one of {validtypes}")
    if pred_type == "classification":
        return __classification_performance_report(X, y_true, y_pred, y_prob,
                                                   features, add_overview)
    elif pred_type == "regression":
        msg = "Regression reporting will be available in version 2.0"
        raise ValueError(msg)
        #return __regression_performance_report(X, y_true, y_pred, features, add_overview)


def sort_report(report):
    """ Sorts columns in standardized order

    Args:
        report (pd.DataFrame): any of the stratified reports produced by this
        module

    Returns:
        pd.DataFrame: sorted report
    """
    yname = y_cols()['disp_names']['yt']
    yhname = y_cols()['disp_names']['yh']
    head_names = ['Feature Name', 'Feature Value', 'Obs.',
                 f'{yname} Mean', f'{yhname} Mean']
    head_cols = [c for c in head_names if c in report.columns]
    tail_cols = sorted([c for c in report.columns if c not in head_cols])
    return report[head_cols + tail_cols]


def summary_report(X, prtc_attr, y_true, y_pred, y_prob=None, flag_oor=False,
                   pred_type="classification", priv_grp=1, **kwargs):
    """
    """
    validtypes = ["classification", "regression"]
    if pred_type not in validtypes:
        raise ValueError(f"Summary report type must be one of {validtypes}")
    if pred_type == "classification":
        df = __classification_summary(X, prtc_attr, y_true, y_pred, y_prob,
                                        priv_grp, **kwargs)
    elif pred_type == "regression":
        msg = "Regression reporting will be available in version 2.0"
        raise ValueError(msg)
        #df = __regression_summary(X, prtc_attr, y_true, y_pred, priv_grp, **kwargs)
    if flag_oor:
        df = flag(df)
    return df


''' Private Functions '''

@format_errwarn
def __apply_featureGroups(features, df, func, *args):
    """ Iteratively applies a function across groups of each stratified feature,
    collecting errors and warnings to be displayed succinctly after processing

    Args:
        features (list): columns of df to be iteratively analyzed
        df (pd.DataFrame): data to be analyzed
        func (function): a function accepting *args and returning a dictionary

    Returns:
        pd.DataFrame: set of results for each feature-value
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
        features (list): columns of df to be iteratively analyzed
        df (pd.DataFrame): data to be analyzed
        func (function): a function accepting two array arguments for comparison
            (selected from df as yt and yh), as well as a pa_name (str) and
            priv_grp (int) which will be set by __apply_biasGroups. This function
            must return a dictionary.
        yt (string): name of column found in df containing target values
        yh (string): name of column found in df containing predicted values

    Returns:
        pd.DataFrame: set of results for each feature-value
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
            res.append(grp_res)
    if len(res) == 0:
        results = pd.DataFrame(columns=['Feature Name', 'Feature Value'])
    else:
        results = pd.concat(res, ignore_index=True)
    return results, errs, warns


def __class_prevalence(y_true, priv_grp):
    """Returns a dictionary of data metrics applicable to evaluation of
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


def __classification_performance_report(X, y_true, y_pred, y_prob=None,
                                        features:list=None, add_overview=True):
    """
    Generates a table of stratified performance metrics for each specified
    feature

    Args:
        df (pandas dataframe or compatible object): data to be assessed
        y_true (1D array-like): Sample target true values; must be binary values
        y_pred (1D array-like): Sample target predictions; must be binary values
        y_prob (1D array-like, optional): Sample target probabilities. Defaults
            to None.
        features (list): columns in df to be assessed if not all columns.
            Defaults to None.
    """
    #
    def __perf_rep(x, y, yh, yp):
        _y = y_cols()['disp_names']['yt']
        _yh = y_cols()['disp_names']['yh']
        res = {'Obs.': x.shape[0],
            f'{_y} Mean': x[y].mean(),
            f'{_yh} Mean': x[yh].mean(),
            'TPR': pmtrc.true_positive_rate(x[y], x[yh]),
            'FPR': pmtrc.false_positive_rate(x[y], x[yh]),
            'Accuracy': pmtrc.accuracy(x[y], x[yh]),
            'Precision': pmtrc.precision(x[y], x[yh])  # PPV
            }
        if yp is not None:
            res['ROC AUC'] = pmtrc.roc_auc_score(x[y], x[yp])
            res['PR AUC'] = pmtrc.pr_auc_score(x[y], x[yp])
        return res
    #
    if y_true is None or y_pred is None:
        msg = "Cannot assess performance without both y_true and y_pred"
        raise ValueError(msg)
    #
    df = stratified_preprocess(X, y_true, y_pred, y_prob, features=features)
    yt, yh, yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    strat_feats = [f for f in df.columns.tolist() if f not in pred_cols]
    if any(y is None for y in [yt, yh]):
        raise ValidationError("Cannot generate report with undefined targets")
    #
    results = __apply_featureGroups(strat_feats, df, __perf_rep, yt, yh, yp)
    if add_overview:
        overview = {'Feature Name': "ALL FEATURES",
                    'Feature Value': "ALL VALUES"}
        ov_dict = __perf_rep(df, yt, yh, yp)
        for k, v in ov_dict.items():
            overview[k] = v
        overview_df = pd.DataFrame(overview, index=[0])
        rprt = pd.concat([overview_df, results], axis=0, ignore_index=True)
    else:
        rprt = results
    rprt = sort_report(rprt)
    return rprt


def __regression_performance_report(X, y_true, y_pred, features:list=None,
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
        are not discrete and there are more than 11 values, the reporter will
        reformat those data into quantiles
    """
    #
    def __perf_rep(x, y, yh):
        _y = y_cols()['disp_names']['yt']
        _yh = y_cols()['disp_names']['yh']
        res = {'Obs.': x.shape[0],
                f'{_y} Mean': x[y].mean(),
                f'{_yh} Mean': x[yh].mean(),
                f'{_yh} Median': x[yh].median(),
                f'{_yh} Std. Dev.': x[yh].std(),
                'Error Mean': (x[yh] - x[y]).mean(),
                'Error Std. Dev.': (x[yh] - x[y]).std(),
                'scMAE': pmtrc.scMAE(x[y], x[yh]),
                'MAE': mean_absolute_error(x[y], x[yh]),
                'MSE': mean_squared_error(x[y], x[yh])
                }
        return res
    #
    if y_true is None or y_pred is None:
        msg = "Cannot assess performance without both y_true and y_pred"
        raise ValueError(msg)
    #
    df = stratified_preprocess(X, y_true, y_pred, features=features)
    yt, yh, yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    strat_feats = [f for f in df.columns.tolist() if f not in pred_cols]
    if any(y is None for y in [yt, yh]):
        raise ValidationError("Cannot generate report with undefined targets")
    #
    results = __apply_featureGroups(strat_feats, df, __perf_rep, yt, yh)
    if add_overview:
        overview = {'Feature Name': "ALL FEATURES",
                    'Feature Value': "ALL VALUES"}
        ov_dict = __perf_rep(df, yt, yh)
        for k, v in ov_dict.items():
            overview[k] = v
        overview_df = pd.DataFrame(overview, index=[0])
        rprt = pd.concat([overview_df, results], axis=0, ignore_index=True)
    else:
        rprt = results
    rprt = sort_report(rprt)
    return rprt


@iterate_cohorts
def __classification_bias_report(X, y_true, y_pred, features:list=None,
                                 priv_grp=1):
    """
    Generates a table of stratified fairness metrics metrics for each specified
    feature

    Args:
        df (pandas dataframe or compatible object): data to be assessed
        y_true (1D array-like): Sample target true values; must be binary values
        y_pred (1D array-like): Sample target predictions; must be binary values
        features (list): columns in df to be assessed if not all columns.
            Defaults to None.

    Requirements:
        Each feature must be discrete to run stratified analysis. If any data
        are not discrete and there are more than 11 values, the reporter will
        reformat those data into quantiles
    """
    #
    def __bias_rep(y_true, y_pred, pa_name, priv_grp=1):
        gf_vals = {}
        gf_vals['PPV Ratio'] = \
            fcmtrc.ppv_ratio(y_true, y_pred, pa_name, priv_grp)
        gf_vals['TPR Ratio'] =  \
            fcmtrc.tpr_ratio(y_true, y_pred, pa_name, priv_grp)
        gf_vals['FPR Ratio'] =  \
            fcmtrc.fpr_ratio(y_true, y_pred, pa_name, priv_grp)
        #
        gf_vals['PPV Diff'] = fcmtrc.ppv_diff(y_true, y_pred, pa_name, priv_grp)
        gf_vals['TPR Diff'] = fcmtrc.tpr_diff(y_true, y_pred, pa_name, priv_grp)
        gf_vals['FPR Diff'] = fcmtrc.fpr_diff(y_true, y_pred, pa_name, priv_grp)
        return gf_vals
    #
    if y_true is None or y_pred is None:
        msg = "Cannot assess fairness without both y_true and y_pred"
        raise ValueError(msg)
    #
    df = stratified_preprocess(X, y_true, y_pred, features=features)
    yt, yh, yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    strat_feats = [f for f in df.columns.tolist() if f not in pred_cols]
    if any(y is None for y in [yt, yh]):
        raise ValidationError("Cannot generate report with undefined targets")
    #
    results = __apply_biasGroups(strat_feats, df, __bias_rep, yt, yh)
    rprt = sort_report(results)
    return rprt


@iterate_cohorts
def __regression_bias_report(X, y_true, y_pred, features:list=None, priv_grp=1):
    """
    Generates a table of stratified fairness metrics metrics for each specified
    feature

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
    yt, yh, yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    strat_feats = [f for f in df.columns.tolist() if f not in pred_cols]
    if any(y is None for y in [yt, yh]):
        raise ValidationError("Cannot generate report with undefined targets")
    #
    results = __apply_biasGroups(strat_feats, df, __regression_bias, yt, yh)
    rprt = sort_report(results)
    return rprt


def __similarity_measures(X, pa_name, y_true, y_pred):
    # Generate dict of Similarity-Based Fairness measures
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


@iterate_cohorts
def __classification_summary(X, prtc_attr, y_true, y_pred, y_prob=None,
                             priv_grp=1, **kwargs):
    """ Returns a pandas dataframe containing fairness measures for the model
        results
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
    def __summary(X, pa_name, y_true, y_pred, y_prob=None,
                                        priv_grp=1):
        """ Returns a dictionary containing group fairness measures specific
            to binary classification problems
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
        #
        gf_vals = {}

        gf_vals['Statistical Parity Difference'] = \
            aif.statistical_parity_difference(y_true, y_pred,
                                                prot_attr=pa_name)
        gf_vals['Disparate Impact Ratio'] = \
            aif.disparate_impact_ratio(y_true, y_pred, prot_attr=pa_name)

        gf_vals['Equalized Odds Difference'] = eq_odds_diff(y_true, y_pred,
                                                            prtc_attr=pa_name)
        gf_vals['Equalized Odds Ratio'] = eq_odds_ratio(y_true, y_pred,
                                                        prtc_attr=pa_name)

        if helpers.is_kdd_tutorial():
            gf_vals['Average Odds Difference'] = \
                aif.average_odds_difference(y_true, y_pred, prot_attr=pa_name)
            gf_vals['Equal Opportunity Difference'] = \
                aif.equal_opportunity_difference(y_true, y_pred,
                                                    prot_attr=pa_name)

        # Precision
        gf_vals['Positive Predictive Parity Difference'] = \
            aif.difference(precision_score, y_true,
                                y_pred, prot_attr=pa_name, priv_group=priv_grp)
        gf_vals['Balanced Accuracy Difference'] = \
            aif.difference(balanced_accuracy_score, y_true,
                                y_pred, prot_attr=pa_name, priv_group=priv_grp)
        gf_vals['Balanced Accuracy Ratio'] = \
            aif.ratio(balanced_accuracy_score, y_true,
                        y_pred, prot_attr=pa_name, priv_group=priv_grp)
        if y_prob is not None:
            try:
                gf_vals['AUC Difference'] = \
                    aif.difference(roc_auc_score, y_true, y_prob,
                                    prot_attr=pa_name, priv_group=priv_grp)
            except:
                pass
        return gf_vals

    def __m_p_c(y, yh, yp=None):
        # Returns a dict containing classification performance measure values for
        # non-stratified reports
        res = {'Accuracy': pmtrc.accuracy(y, yh),
            'Balanced Accuracy': pmtrc.balanced_accuracy(y, yh),
            'F1-Score': pmtrc.f1_score(y, yh),
            'Recall': pmtrc.true_positive_rate(y, yh),
            'Precision': pmtrc.precision(y, yh)
            }
        if yp is not None:
            res['ROC_AUC'] = pmtrc.roc_auc_score(y, yp)
            res['PR_AUC'] = pmtrc.pr_auc_score(y, yp)
        return res
    #
    # Validate and Format Arguments
    if not isinstance(priv_grp, int):
        raise ValueError("priv_grp must be an integer value")
    X, prtc_attr, y_true, y_pred, y_prob = \
        standard_preprocess(X, prtc_attr, y_true, y_pred, y_prob, priv_grp)
    pa_name = prtc_attr.columns.tolist()[0]

    # Temporarily prevent processing for more than 2 classes
    # ToDo: enable multiclass
    n_class = np.unique(np.append(y_true.values, y_pred.values)).shape[0]
    if n_class != 2:
        raise ValueError(
            "Reporter cannot yet process multiclass classification models")
    if n_class == 2:
        labels = report_labels()
    else:
        labels = report_labels("multiclass")
    gfl, ifl, mpl, dtl = labels.values()
    # Generate a dictionary of measure values to be converted t a dataframe
    mv_dict = {}
    mv_dict[gfl] =  __summary(X, pa_name, y_true, y_pred, y_prob, priv_grp)
    mv_dict[dtl] = __class_prevalence(y_true, priv_grp)
    if not kwargs.pop('skip_if', False):
        mv_dict[ifl] = __similarity_measures(X, pa_name, y_true, y_pred)
    if not kwargs.pop('skip_performance', False):
        mv_dict[mpl] = __m_p_c(y_true, y_pred)
    # Convert scores to a formatted dataframe and return
    df = pd.DataFrame.from_dict(mv_dict, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    # Fix the order in which the metrics appear
    metric_order = {gfl: 0, ifl: 1, mpl: 2, dtl: 3}
    df.reset_index(inplace=True)
    df['sortorder'] = df['level_0'].map(metric_order)
    df = df.sort_values('sortorder').drop('sortorder', axis=1)
    df.set_index(['level_0', 'level_1'], inplace=True)
    df.rename_axis(('Metric', 'Measure'), inplace=True)
    return df


def __regression_bias(y_true, y_pred, pa_name, priv_grp=1):
    def pdmean(y_true, y_pred, *args): return y_pred.mean()
    def meanerr(y_true, y_pred, *args): return (y_pred - y_true).mean()
    #
    gf_vals = {}
    # Ratios
    gf_vals['Mean Prediction Ratio'] = \
        aif.ratio(pdmean, y_true, y_pred,
                  prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['scMAE Ratio'] = \
        aif.ratio(pmtrc.scMAE, y_true, y_pred,
                  prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['MAE Ratio'] = \
        aif.ratio(mean_absolute_error, y_true, y_pred,
                  prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['Mean Error Ratio'] = \
        aif.ratio(meanerr, y_true, y_pred,
                  prot_attr=pa_name, priv_group=priv_grp)
    # Differences
    gf_vals['Mean Prediction Difference'] = \
        aif.difference(pdmean, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['scMAE Difference'] = \
        aif.difference(pmtrc.scMAE, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['MAE Difference'] = \
        aif.difference(mean_absolute_error, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['Mean Error Difference'] = \
        aif.difference(meanerr, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    return gf_vals


@iterate_cohorts
def __regression_summary(X, prtc_attr, y_true, y_pred, priv_grp=1,
                         **kwargs):
    """ Returns a pandas dataframe containing fairness measures for the model
        results
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
    bias_df = pd.concat([y_true.reset_index(), y_pred], axis=1)
    gf_vals = __regression_bias(y_true, y_pred, pa_name, priv_grp=priv_grp)
    #
    if not kwargs.pop('skip_if', False):
        if_vals = __similarity_measures(X, pa_name, y_true, y_pred)

    dt_vals = __class_prevalence(y_true, priv_grp)
    #
    mp_vals = {}
    report = regression_performance(y_true, y_pred)
    for row in report.iterrows():
        mp_vals[row[0]] = row[1]['Score']
    # Convert scores to a formatted dataframe and return
    labels = report_labels("regression")
    measures = {labels['gf_label']: gf_vals,
                labels['if_label']: if_vals,
                labels['mp_label']: mp_vals,
                labels['dt_label']: dt_vals}
    df = pd.DataFrame.from_dict(measures, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    return df


def __clean_names(col):
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


class __Flagger():
    """
    """
    diffs = ["auc difference" , "balanced accuracy difference",
            "equalized odds difference", "positive predictive parity difference",
            "Statistical Parity Difference", "fpr diff", "tpr diff", "ppv diff"]
    ratios = ["balanced accuracy ratio", "disparate impact ratio ",
                "equalized odds ratio", "fpr ratio", "tpr ratio", "ppv ratio"]
    stats_high = ["consistency score"]
    stats_low =["between-group gen. entropy error"]

    def __init__(self):
        self.reset()

    def reset(self):
        self.df = None
        self.labels = None
        self.label_type = None
        self.flag_type = "background-color"
        self.flag_color = "magenta"

    def color_diff(self, s):
        if self.label_type == "index":
            idx = pd.IndexSlice
            lbls = self.df.loc[idx['Group Fairness',
                        [c.lower() in self.diffs for c in self.labels]], :].index
            clr = [f'{self.flag_type}:{self.flag_color}'
                    if (s.name in lbls and not -0.1 < i < 0.1)
                    else '' for i in s]
        else:
            lbls = self.diffs
            clr = [f'{self.flag_type}:{self.flag_color}'
                    if (s.name.lower() in lbls and not -0.1 < i < 0.1)
                    else '' for i in s]
        return clr

    def color_st(self, s):
        if self.label_type == "index":
            idx = pd.IndexSlice
            lb_high = self.df.loc[idx['Individual Fairness',
                        [c.lower() in self.stats_high
                        for c in self.labels]], :].index
            lb_low = self.df.loc[idx['Individual Fairness',
                        [c.lower() in self.stats_low
                                for c in self.labels]], :].index
            clr = [f'{self.flag_type}:{self.flag_color}'
                    if (s.name in lb_high and i < 0.8)
                        or (s.name  in lb_low and i > 0.2)
                    else '' for i in s]
        else:
            lb_high = self.stats_high
            lb_low = self.stats_low
            clr = [f'{self.flag_type}:{self.flag_color}'
                    if (s.name.lower() in lb_high and i < 0.8)
                        or (s.name.lower() in lb_low and i > 0.2)
                    else '' for i in s]
        return clr

    def color_ratio(self, s):
        if self.label_type == "index":
            idx = pd.IndexSlice
            lbls = self.df.loc[idx['Group Fairness',
                        [c.lower() in self.ratios for c in self.labels]], :].index
            clr = [f'{self.flag_type}:{self.flag_color}'
                    if (s.name in lbls and not 0.8 < i < 1.2)
                    else '' for i in s]
        else:
            lbls = self.ratios
            clr = [f'{self.flag_type}:{self.flag_color}'
               if (s.name.lower() in lbls and not 0.8 < i < 1.2)
               else '' for i in s]
        return clr

    def set_measure_labels(self, df):
        try:
            labels = df.index.get_level_values(1)
            label_type = "index"
        except:
            labels = df.columns.tolist()
            label_type = "columns"
        return df, labels, label_type

    def apply_flag(self, df, caption="", as_styler=True):
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
        if caption is None:
            caption = "Fairness Measures"
        #
        self.reset()
        self.df, self.labels, self.label_type = self.set_measure_labels(df)
        if self.label_type == "index":
            styled = self.df.style.set_caption(caption
                                    ).apply(self.color_diff, axis=1
                                    ).apply(self.color_ratio, axis=1
                                    ).apply(self.color_st, axis=1)
        else:
            styled = self.df.style.set_caption(caption
                                    ).apply(self.color_diff, axis=0
                                    ).apply(self.color_ratio, axis=0
                                    ).apply(self.color_st, axis=0)
        # return pandas styler if requested
        if as_styler:
            return styled
        else:
            return HTML(styled.render())



