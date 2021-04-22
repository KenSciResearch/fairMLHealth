# -*- coding: utf-8 -*-
"""
Tools producing reports of fairness, bias, or model performance measures
Contributors:
    camagallen <ca.magallen@gmail.com>
"""

# A note about the naming schema for private functions:
#    Private functions are formatted in the following format:
#        __[report type]_[metric type]_[prediction type]
#   Sub-function names may be abbreviated as follows
#        report types: m = "measures", s = "stratified"
#        metric types: d = "data", f = "fairness", p = "performance",
#           gf = "group fairness", sf = "similarity-based fairness"
#        prediction types: c = "classification", r = "regression",
#            bc = "binary classification", mc = "multiclass"
#    Example: __s_p_c <--> __stratified_performance_classification



import aif360.sklearn.metrics as aif
from IPython.display import HTML
import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import sklearn.metrics as sk_metric
from scipy import stats
import warnings

# Tutorial Libraries
from . import __classification_metrics as clmtrc, __fairness_metrics as fcmtrc
from .__fairness_metrics import eq_odds_diff, eq_odds_ratio
from .__preprocessing import standard_preprocess, stratified_preprocess, y_cols
from . import tutorial_helpers as helpers
from .utils import ValidationError



# ToDo: find better solution for these warnings
warnings.filterwarnings('ignore', module='pandas')
warnings.filterwarnings('ignore', module='sklearn')




''' Deprecated Public Functions '''

def flag_suspicious(df, caption="", as_styler=False):
    warnings.warn(
            "flag_suspicious function will be deprecated in version 2.0" +
            " Use flag instead.", PendingDeprecationWarning
        )
    return flag(df, caption="", as_styler=False)


def classification_fairness(X, prtc_attr, y_true, y_pred, y_prob=None,
                            priv_grp=1, sig_dec=4, **kwargs):
    warnings.warn(
            "classification_fairness function will be deprecated in version " +
            "2.0. Use flag instead.", PendingDeprecationWarning
        )
    return __measures_fairness_classification(X, prtc_attr, y_true, y_pred,
                                              y_prob=None,priv_grp=1, sig_dec=4,
                                              **kwargs)


def regression_fairness(X, prtc_attr, y_true, y_pred, priv_grp=1, sig_dec=4,
                        **kwargs):
    warnings.warn(
            "regression_fairness function will be deprecated in version " +
            "2.0. Use flag instead.", PendingDeprecationWarning
        )
    return __measures_fairness_regression(X, prtc_attr, y_true, y_pred,
                                          priv_grp=1, sig_dec=4, **kwargs)






''' mini reports '''
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
    report = sk_metric.classification_report(y_true, y_pred, output_dict=True,
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
    report['True Mean'] = np.mean(y_true)
    report['Mean Prediction'] = np.mean(y_pred)
    report['Rsqrd'] = sk_metric.r2_score(y_true, y_pred)
    report['Mean Absolute Error'] = sk_metric.mean_absolute_error(y_true, y_pred)
    report['Mean Square Error'] = sk_metric.mean_squared_error(y_true, y_pred)
    report = pd.DataFrame().from_dict(report, orient='index'
                          ).rename(columns={0: 'Score'})
    return report







''' Public Functions '''

def flag(df, caption="", as_styler=False):
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
    idx = pd.IndexSlice
    measures = df.index.get_level_values(1)
    ratios = df.loc[idx['Group Fairness',
                    [c.lower().endswith("ratio") for c in measures]], :].index
    difference = df.loc[idx['Group Fairness',
                        [c.lower().endswith("difference")
                         for c in measures]], :].index
    cs_high = df.loc[idx['Individual Fairness',
                     [c.lower().replace(" ", "_") == "consistency_score"
                      for c in measures]], :].index
    cs_low = df.loc[idx['Individual Fairness',
                        [c.lower().replace(" ", "_")
                            == "generalized_entropy_error"
                         for c in measures]], :].index
    #
    def color_diff(row):
        clr = ['color:magenta'
               if (row.name in difference and not -0.1 < i < 0.1)
               else '' for i in row]
        return clr

    def color_if(row):
        clr = ['color:magenta'
               if (row.name in cs_high and i < 0.8) or
                  (row.name in cs_low and i > 0.2)
               else '' for i in row]
        return clr

    def color_ratios(row):
        clr = ['color:magenta'
               if (row.name in ratios and not 0.8 < i < 1.2)
               else '' for i in row]
        return clr

    styled = df.style.set_caption(caption
                                  ).apply(color_diff, axis=1
                                  ).apply(color_ratios, axis=1
                                  ).apply(color_if, axis=1)
    # Correct management of metric difference has yet to be determined for
    #   regression functions. Add style to o.o.r. difference for binary
    #   classification only
    if "MSE Ratio" not in measures:
        styled.apply(color_diff, axis=1)
    # return pandas styler if requested
    if as_styler:
        return styled
    else:
        return HTML(styled.render())


def data_report(X, y_true, features:list=None):
    """
    Generates a table of stratified data metrics

    Args:
        df (pandas dataframe or compatible object): sample data to be assessed
        y_true (1D array-like): Sample target true values
        features (list): columns in df to be assessed if not all columns.
            Defaults to None.

    Requirements:
        Each feature must be discrete to run stratified analysis. If any data
        are not discrete and there are more than 11 values, the reporter will
        reformat those data into quantiles
    """
    #
    df = stratified_preprocess(X, y_true, features=features)
    yt, yh, yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    stratified_features = [f for f in df.columns.tolist() if f not in pred_cols]
    # For the data report it does not matter if y_true is defined. Other metrics
    #   can still be generated
    if yt is None:
        df["__fairmlhealth_standin"] = 1
        yt = ["__fairmlhealth_standin"]
    #
    res = []
    N_missing = 0
    N_obs = df.shape[0]
    skipped_vars = []
    for f in stratified_features:
        if df[f].nunique() == 1:
            skipped_vars.append(f)
            continue
        n_missing = df.loc[df[f].isna() | df[f].eq('nan'), f].count()
        # Add feature-specific statistics for each group in the feature
        grp = df.groupby(f)[pred_cols]
        try:
            r = grp.apply(lambda x: pd.Series(__stratified_data(x, yt)))
        except BaseException as e:
            raise ValueError(f"Error processing {f}. {e}\n")
        r = r.reset_index().rename(columns={f: 'FEATURE VALUE'})
        #
        r.insert(0, 'FEATURE', f)
        r['N MISSING'] = n_missing
        N_missing += n_missing
        _, feat_count = np.unique(df[f], return_counts=True)
        r['FEATURE ENTROPY'] = stats.entropy(feat_count, base=2)
        res.append(r)
    full_res = pd.concat(res, ignore_index=True)
    full_res['VALUE PREVALENCE'] = full_res['N OBS']/N_obs

    #
    N_feat = len(stratified_features)
    overview = {'FEATURE': "ALL_FEATURES",
                'FEATURE VALUE': "ALL_VALUES",
                'N MISSING': N_missing,
                'VALUE PREVALENCE': (N_obs*N_feat - N_missing)/(N_obs*N_feat)
                }
    ov_dict = __stratified_data(df, yt)
    for k, v in ov_dict.items():
        overview[k] = v
    overview_df = pd.DataFrame(overview, index=[0])
    # Combine and format
    rprt = pd.concat([overview_df, full_res], axis=0, ignore_index=True)
    head_cols = ['FEATURE', 'FEATURE VALUE', 'N OBS', 'N MISSING']
    tail_cols = sorted([c for c in rprt.columns if c not in head_cols])
    rprt = rprt[head_cols + tail_cols]
    rprt = rprt.round(4)
    return rprt


def __measures_data(y_true, priv_grp):
    """Returns a dictionary of data metrics applicable to evaluation of
    fairness
    Args:
        y_true (pandas DataFrame): Sample targets
        priv_grp (int): Specifies which label indicates the privileged
                group. Defaults to 1.
    """
    dt_vals = {}
    dt_vals['Prevalence of Privileged Class (%)'] = \
        round(100*y_true[y_true.eq(priv_grp)].sum()/y_true.shape[0])
    return dt_vals


def __stratified_data(x, col):
    """
    Returns a dict of statistics. Intended for use with pandas .apply()
    function.

    Args:
        x (pandas DataFrame)
        col (string): name of column in dataframe
    """
    # If column is a hidden variable, replace it with a user-friendly name
    yvars = y_cols()
    if col in yvars['col_names'].values():
        idx = list(yvars['col_names'].values()).index(col)
        key = list(yvars['col_names'].keys())[idx]
        display_name = yvars['disp_names'][key]
    else:
        display_name = col
    # Generate dictionary of statistics
    res = {'N OBS': x.shape[0]}
    if not x[col].isna().all():
        res[f'{display_name} MEAN'] = x[col].mean()
        res[f'{display_name} MEDIAN'] = x[col].median()
        res[f'{display_name} STDV'] = x[col].std()
        res[f'{display_name} MIN'] = x[col].min()
        res[f'{display_name} MAX'] = x[col].max()
    return res






def summary_report(X, prtc_attr, y_true, y_pred, y_prob=None,
                      type="classification", priv_grp=1, sig_dec=4, **kwargs):

    if type == "classification":
        __measures_fairness_classification(X, prtc_attr, y_true, y_pred,
                                           y_prob=None, priv_grp=1, sig_dec=4,
                                           **kwargs)
    elif type == "regression":
        __measures_fairness_regression(X, prtc_attr, y_true, y_pred, priv_grp=1,
                                       sig_dec=4, **kwargs)


# data_report
# performance_report
# fairness_report







''' __measures_fairness_classification '''
def __measures_fairness_classification(X, prtc_attr, y_true, y_pred, y_prob=None,
                            priv_grp=1, sig_dec=4, **kwargs):
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
        sig_dec (int): number of significant decimals to which to round
            measure values. Defaults to 4.
    """
    # Validate and Format Arguments
    if not isinstance(priv_grp, int):
        raise ValueError("priv_grp must be an integer value")
    if not isinstance(sig_dec, int):
        raise ValueError("sig_dec must be an integer value")
    X, prtc_attr, y_true, y_pred, y_prob = \
        standard_preprocess(X, prtc_attr, y_true, y_pred, y_prob, priv_grp)
    pa_name = prtc_attr.columns.tolist()

    # Temporarily prevent processing for more than 2 classes
    # ToDo: enable multiclass
    n_class = np.unique(np.append(y_true.values, y_pred.values)).shape[0]
    if n_class != 2:
        raise ValueError(
            "Reporter cannot yet process multiclass classification models")
    if n_class == 2:
        labels = __report_labels()
    else:
        labels = __report_labels("multiclass")
    gfl, ifl, mpl, dtl = labels.values()
    # Generate a dictionary of measure values to be converted t a dataframe
    mv_dict = {}
    mv_dict[gfl] = \
        __m_gf_bc(X, pa_name, y_true, y_pred, y_prob, priv_grp)
    mv_dict[dtl] = __measures_data(y_true, priv_grp)
    if not kwargs.pop('skip_if', False):
        mv_dict[ifl] = __m_sf(X, pa_name, y_true, y_pred)
    if not kwargs.pop('skip_performance', False):
        mv_dict[mpl] = __m_p_c(y_true, y_pred)
    # Convert scores to a formatted dataframe and return
    df = pd.DataFrame.from_dict(mv_dict, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    df.loc[:, 'Value'] = df['Value'].astype(float).round(sig_dec)
    # Fix the order in which the metrics appear
    metric_order = {gfl: 0, ifl: 1, mpl: 2, dtl: 3}
    df.reset_index(inplace=True)
    df['sortorder'] = df['level_0'].map(metric_order)
    df = df.sort_values('sortorder').drop('sortorder', axis=1)
    df.set_index(['level_0', 'level_1'], inplace=True)
    df.rename_axis(('Metric', 'Measure'), inplace=True)
    return df


def __m_gf_bc(X, pa_name, y_true, y_pred, y_prob=None,
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
        aif.difference(sk_metric.precision_score, y_true,
                            y_pred, prot_attr=pa_name, priv_group=priv_grp)

    gf_vals['Balanced Accuracy Difference'] = \
        aif.difference(sk_metric.balanced_accuracy_score, y_true,
                            y_pred, prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['Balanced Accuracy Ratio'] = \
        aif.ratio(sk_metric.balanced_accuracy_score, y_true,
                       y_pred, prot_attr=pa_name, priv_group=priv_grp)
    if y_prob is not None:
        try:
            gf_vals['AUC Difference'] = \
                aif.difference(sk_metric.roc_auc_score, y_true, y_prob,
                                prot_attr=pa_name, priv_group=priv_grp)
        except:
            pass
    return gf_vals


def __m_p_c(y, yh, yp=None):
    # Returns a dict containing classification performance measure values for
    # non-stratified reports
    res = {'Accuracy': clmtrc.accuracy(y, yh),
           'Balanced Accuracy': clmtrc.balanced_accuracy(y, yh),
           'F1-Score': clmtrc.f1_score(y, yh),
           'Recall': clmtrc.true_positive_rate(y, yh),
           'Precision': clmtrc.precision(y, yh)
          }
    if yp is not None:
        res['ROC_AUC'] = clmtrc.roc_auc_score(y, yp)
        res['PR_AUC'] = clmtrc.pr_auc_score(y, yp)
    return res




''' __measures_fairness_regression '''

def __measures_fairness_regression(X, prtc_attr, y_true, y_pred, priv_grp=1, sig_dec=4,
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
        sig_dec (int): number of significant decimals to which to round
            measure values. Defaults to 4.
    """
    # Validate and Format Arguments
    if not isinstance(priv_grp, int):
        raise ValueError("priv_grp must be an integer value")
    if not isinstance(sig_dec, int):
        raise ValueError("sig_dec must be an integer value")
    X, prtc_attr, y_true, y_pred, _ = \
        standard_preprocess(X, prtc_attr, y_true, y_pred, priv_grp)
    pa_name = prtc_attr.columns().tolist()
    #
    gf_vals = \
        __m_gf_r(pa_name, y_true, y_pred,
                                         priv_grp=priv_grp)
    #
    if not kwargs.pop('skip_if', False):
        if_vals = __m_sf(X, pa_name, y_true, y_pred)

    dt_vals = __measures_data(y_true, priv_grp)

    #
    mp_vals = {}
    report = regression_performance(y_true, y_pred)
    for row in report.iterrows():
        mp_vals[row[0]] = row[1]['Score']


    # Convert scores to a formatted dataframe and return
    labels = __report_labels("regression")
    measures = {labels['gf_label']: gf_vals,
                labels['if_label']: if_vals,
                labels['mp_label']: mp_vals,
                labels['dt_label']: dt_vals}
    df = pd.DataFrame.from_dict(measures, orient="index").stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index)
    df.columns = ['Value']
    df.loc[:, 'Value'] = df['Value'].astype(float).round(sig_dec)
    return df


def __m_gf_r(pa_name, y_true, y_pred, priv_grp=1):
    """ Returns a dictionary containing group fairness measures specific
        to regression problems
    Args:
        prtc_attr_name (named array-like): Values for the protected attribute
            (note: protected attribute may also be present in X)
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
        priv_grp (int): Specifies which label indicates the privileged
                group. Defaults to 1.
    """
    def pdmean(y_true, y_pred, *args): return y_pred.mean()
    #
    gf_vals = {}
    gf_vals['Mean Prediction Ratio'] = \
        aif.ratio(pdmean, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['MAE Ratio'] = \
        aif.ratio(sk_metric.mean_absolute_error, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['R2 Ratio'] = \
        aif.ratio(sk_metric.r2_score, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['Mean Prediction Difference'] = \
        aif.difference(pdmean, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['MAE Difference'] = \
        aif.difference(sk_metric.mean_absolute_error, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['R2 Difference'] = \
        aif.difference(sk_metric.r2_score, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    return gf_vals



''' Similarity-based Bias Measures '''
def __m_sf(X, pa_name, y_true, y_pred):
    """ Returns a dictionary of individual fairness measures for the data that
        were passed
    Args:
        X (pandas DataFrame): Sample features
        pa_name (str):
        y_true (pandas DataFrame): Sample targets
        y_pred (pandas DataFrame): Sample target predictions
    """
    # Generate dict of Individual Fairness measures
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







''' Stratified Reporting '''






def classification_performance_report(X, y_true, y_pred, y_prob=None,
                               features:list=None):
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
    if y_true is None or y_pred is None:
        msg = "Cannot assess performance without both y_true and y_pred"
        raise ValueError(msg)
    #
    df = stratified_preprocess(X, y_true, y_pred, y_prob, features=features)
    yt, yh, yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    stratified_features = [f for f in df.columns.tolist() if f not in pred_cols]
    #
    res = []
    errs = {}
    for f in stratified_features:
        if not df[f].astype(str).eq(df[f]).all():
            assert TypeError(f, "data are expected in string format")
        # Add feature-specific performance values for each group in the feature
        grp = df.groupby(f)[pred_cols]
        try:
            r = grp.apply(lambda x: pd.Series(__s_p_c(x, yt, yh, yp)))
        except BaseException as e:
            errs[f] = e
            continue
        r = r.reset_index().rename(columns={f: 'FEATURE VALUE'})
        r.insert(0, 'FEATURE', f)
        res.append(r)
    if any(errs):
        for k, v in errs.items():
            print(f"Error processing column(s) {k}. {v}\n")
    full_res = pd.concat(res, ignore_index=True)
    #
    overview = {'FEATURE': "ALL_FEATURES",
                'FEATURE VALUE': "ALL_VALUES"}
    ov_dict = __s_p_c(df, yt, yh, yp)
    for k, v in ov_dict.items():
        overview[k] = v
    overview_df = pd.DataFrame(overview, index=[0])
    # Combine and format
    rprt = pd.concat([overview_df, full_res], axis=0, ignore_index=True)
    head_cols = ['FEATURE', 'FEATURE VALUE', 'N OBS', 'TRUE MEAN', 'PRED MEAN']
    tail_cols = sorted([c for c in rprt.columns if c not in head_cols])
    rprt = rprt[head_cols + tail_cols]
    rprt = rprt.round(4)
    return rprt

def __s_p_c(x, y, yh, yp):
    """
    Returns a dict containing classification performance values. Intended for
    use with pandas .apply() function.
    """
    res = {'N OBS': x.shape[0],
           'TRUE MEAN': x[y].mean(),
           'PRED MEAN': x[yh].mean(),
           'TPR': clmtrc.true_positive_rate(x[y], x[yh]),
           'TNR': clmtrc.true_negative_rate(x[y], x[yh]),
           'FPR': clmtrc.false_positive_rate(x[y], x[yh]),
           'FNR': clmtrc.false_negative_rate(x[y], x[yh]),
           'ACCURACY': clmtrc.accuracy(x[y], x[yh]),
           'PRECISION (PPV)': clmtrc.precision(x[y], x[yh])  # PPV
           }
    if yp is not None:
        res['ROC_AUC'] = clmtrc.roc_auc_score(x[y], x[yp])
        res['PR_AUC'] = clmtrc.pr_auc_score(x[y], x[yp])
    return res










def regression_performance_report(X, y_true, y_pred, features:list=None):
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
    if y_true is None or y_pred is None:
        msg = "Cannot assess performance without both y_true and y_pred"
        raise ValueError(msg)
    #
    df = stratified_preprocess(X, y_true, y_pred, features=features)
    yt, yh, yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    stratified_features = [f for f in df.columns.tolist() if f not in pred_cols]
    #
    res = []
    skipped_vars = []
    errs = {}
    for f in stratified_features:
        if df[f].nunique() == 1:
            skipped_vars.append(f)
            continue
        # Data are expected in string format
        assert df[f].astype(str).eq(df[f]).all()
        # Add feature-specific performance values for each group in the feature
        grp = df.groupby(f)[pred_cols]
        try:
            r = grp.apply(lambda x: pd.Series(__s_p_r(x, yt, yh)))
        except BaseException as e:
            errs[f] = e
            continue
        r = r.reset_index().rename(columns={f: 'FEATURE VALUE'})
        r.insert(0, 'FEATURE', f)
        res.append(r)
    if any(errs):
        for k, v in errs.items():
            print(f"Error processing column(s) {k}. {v}\n")
    full_res = pd.concat(res, ignore_index=True)
    #
    overview = {'FEATURE': "ALL_FEATURES",
                'FEATURE VALUE': "ALL_VALUES"}
    ov_dict = __s_p_r(df, yt, yh)
    for k, v in ov_dict.items():
        overview[k] = v
    overview_df = pd.DataFrame(overview, index=[0])
    #
    rprt = pd.concat([overview_df, full_res], axis=0, ignore_index=True)
    head_cols = ['FEATURE', 'FEATURE VALUE', 'N OBS', 'TRUE MEAN', 'PRED MEAN']
    tail_cols = sorted([c for c in rprt.columns if c not in head_cols])
    rprt = rprt[head_cols + tail_cols]
    rprt = rprt.round(4)
    return rprt



def __s_p_r(x, y, yh):
    """
    Returns a dict containing regression performance values. Intended for use
    with pandas .apply() function
    """
    res = {'N OBS': x.shape[0],
           'TRUE MEAN': x[y].mean(),
           'PRED MEAN': x[yh].mean(),
           'PRED MEDIAN': x[yh].median(),
           'PRED STD': x[yh].std(),
           'ERROR MEAN': (x[yh] - x[y]).mean(),
           'ERROR STD': (x[yh] - x[y]).std(),
           'MAE': sk_metric.mean_absolute_error(x[y], x[yh]),
           'MSE': sk_metric.mean_squared_error(x[y], x[yh]),
           'Rsqrd': sk_metric.r2_score(x[y], x[yh])
           }
    return res











def classification_fairness_report(X, y_true, y_pred, features:list=None, **kwargs):
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
    if y_true is None or y_pred is None:
        msg = "Cannot assess fairness without both y_true and y_pred"
        raise ValueError(msg)
    #
    df = stratified_preprocess(X, y_true, y_pred, features=features)
    yt, yh, yp = y_cols(df)['col_names'].values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    stratified_features = [f for f in df.columns.tolist() if f not in pred_cols]
    #
    res = []
    pa_name = 'prtc_attr'
    errs = {}
    for f in stratified_features:
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
            try:
                subset = df.loc[df[pa_name].notnull(),
                                [pa_name, yt, yh]].set_index(pa_name)
                meas = __s_gf_c(pa_name, subset[yt], subset[yh], priv_grp=1)
            except BaseException as e:
                errs[f] = e
                continue
            r = pd.DataFrame(meas, index=[0])
            r['N OBS'] = df.loc[df[f].eq(v), pa_name].sum()
            r['FEATURE'] = f
            r['FEATURE VALUE'] = v
            res.append(r)
    if any(errs):
        for k, v in errs.items():
            print(f"Error processing column(s) {k}. {v}\n")
    # Combine and format
    full_res = pd.concat(res, ignore_index=True)
    head_cols = ['FEATURE', 'FEATURE VALUE', 'N OBS']
    tail_cols = sorted([c for c in full_res.columns if c not in head_cols])
    rprt = full_res[head_cols + tail_cols]
    rprt = rprt.round(4)
    #
    return rprt


def __s_gf_c(pa_name, y_true, y_pred, priv_grp=1):
    """
    Returns a dict containing classification fairness measure values. Intended
    for use with pandas .apply() function.
    """
    gf_vals = {}
    gf_vals['PPV Ratio'] = fcmtrc.ppv_ratio(y_true, y_pred, pa_name, priv_grp)
    gf_vals['TPR Ratio'] = fcmtrc.tpr_ratio(y_true, y_pred, pa_name, priv_grp)
    gf_vals['FPR Ratio'] = fcmtrc.fpr_ratio(y_true, y_pred, pa_name, priv_grp)
    gf_vals['TNR Ratio'] = fcmtrc.tnr_ratio(y_true, y_pred, pa_name, priv_grp)
    gf_vals['FNR Ratio'] = fcmtrc.fnr_ratio(y_true, y_pred, pa_name, priv_grp)
    #
    gf_vals['PPV Diff'] = fcmtrc.ppv_diff(y_true, y_pred, pa_name, priv_grp)
    gf_vals['TPR Diff'] = fcmtrc.tpr_diff(y_true, y_pred, pa_name, priv_grp)
    gf_vals['FPR Diff'] = fcmtrc.fpr_diff(y_true, y_pred, pa_name, priv_grp)
    gf_vals['TNR Diff'] = fcmtrc.tnr_diff(y_true, y_pred, pa_name, priv_grp)
    gf_vals['FNR Diff'] = fcmtrc.fnr_diff(y_true, y_pred, pa_name, priv_grp)
    return gf_vals








def regression_fairness_report(X, y_true, y_pred, features:list=None, **kwargs):
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
    stratified_features = [f for f in df.columns.tolist() if f not in pred_cols]
    #
    res = []
    for f in stratified_features:
        # Data are expected in string format
        assert df[f].astype(str).eq(df[f]).all()
        grp = df.groupby(f, as_index=False)[yt].count()
        grp.rename(columns={f: 'FEATURE VALUE', yt: 'N OBS'}, inplace=True)
        grp['FEATURE'] = f
        res.append(grp)
    rprt = pd.concat(res, axis=0, ignore_index=True)
    #
    res_f = []
    rprt = rprt[['FEATURE', 'FEATURE VALUE', 'N OBS']]
    pa_name = 'prtc_attr'
    errs = {}
    for _, row in rprt.iterrows():
        f = row['FEATURE']
        v = row['FEATURE VALUE']
        df[pa_name] = 0
        df.loc[df[f].eq(v), pa_name] = 1
        if v != "nan":
            df.loc[df[f].eq("nan"), pa_name] = np.nan
        # Nothing to measure if only one value is present (other than nan)
        if df[pa_name].nunique() == 1:
            continue
        try:
            subset = df.loc[df[pa_name].notnull(),
                            [pa_name, yt, yh]].set_index(pa_name)
            meas = __m_gf_r(pa_name, subset[yt], subset[yh], priv_grp=1)
        except BaseException as e:
            errs[f] = e
            continue
        r = pd.DataFrame(meas, index=[0])
        r['FEATURE'] = f
        r['FEATURE VALUE'] = v
        res_f.append(r)
    if any(errs):
        for k, v in errs.items():
            print(f"Error processing column(s) {k}. {v}\n")
    # Combine and format
    rprt_update = pd.concat(res_f, ignore_index=True)
    rprt_update = rprt_update[sorted(rprt_update.columns, key=lambda x: x[-5:])]
    rprt = rprt.merge(rprt_update, on=['FEATURE', 'FEATURE VALUE'], how='left')
    head_cols = ['FEATURE', 'FEATURE VALUE', 'N OBS']
    tail_cols = sorted([c for c in rprt.columns if c not in head_cols])
    rprt = rprt[head_cols + tail_cols]
    rprt = rprt.round(4)
    #
    return rprt






def __report_labels(pred_type: str = "binary"):
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







