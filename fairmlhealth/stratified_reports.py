import aif360.sklearn.metrics as aif_mtrc
from fairmlhealth import reports
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from scipy import stats
import sklearn.metrics as sk_metric
import warnings


from . import __classification_metrics as clmtrc
from .utils import __preprocess_input



''' Utility Functions '''


def __get_yname_dict(df=None):
    ''' Returns a dictionary containing the expected column names for each
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
    names = {'yt': '__fairmlhealth_y_true',
             'yh': '__fairmlhealth_y_pred',
             'yp': '__fairmlhealth_y_prob'}
    #
    if df is not None:
        for k in names.keys():
            if names[k] not in df.columns:
                names[k] = None
    return names


''' Data Reporting '''


def __preprocess_stratified(X, y_true, y_pred=None, y_prob=None,
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
        __preprocess_input(X, prtc_attr=None, y_true=y_true, y_pred=y_pred,
                           y_prob=y_prob)
    yt, yh, yp = __get_yname_dict().values()
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
    return df


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
    df = __preprocess_stratified(X, y_true, features=features)
    yt, yh, yp = __get_yname_dict(df).values()
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
            r = grp.apply(lambda x: pd.Series(__data_grp(x, yt)))
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
    ov_dict = __data_grp(df, yt)
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


def __data_grp(x, col):
    """
    Returns a dict of statistics. Intended for use with pandas .apply()
    function.

    Args:
        x (pandas DataFrame)
    """
    # If column is a hidden variable, replace it with a user-friendly name
    yvars = __get_yname_dict()
    if col in yvars.values():
        colname = list(yvars.keys())[list(yvars.values()).index(col)]
        colname = "Y" if colname == "yt" else colname.title()
    else:
        colname = col
    # Generate dictionary of statistics
    res = {'N OBS': x.shape[0]}
    if not x[col].isna().all():
        res[f'{colname} MEAN'] = x[col].mean()
        res[f'{colname} MEDIAN'] = x[col].median()
        res[f'{colname} STDV'] = x[col].std()
        res[f'{colname} MIN'] = x[col].min()
        res[f'{colname} MAX'] = x[col].max()
    return res


''' Performance Reporting '''


def classification_performance(X, y_true, y_pred, y_prob=None,
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
    df = __preprocess_stratified(X, y_true, y_pred, y_prob, features=features)
    yt, yh, yp = __get_yname_dict(df).values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    stratified_features = [f for f in df.columns.tolist() if f not in pred_cols]
    #
    res = []
    for f in stratified_features:
        if not df[f].astype(str).eq(df[f]).all():
            assert TypeError(f, "data are expected in string format")
        # Add feature-specific performance values for each group in the feature
        grp = df.groupby(f)[pred_cols]
        try:
            r = grp.apply(lambda x: pd.Series(__cp_group(x, yt, yh, yp)))
        except BaseException as e:
            # raise ValueError(f"Error processing {f}. {e}\n")
            print(f"Error processing {f}. {e}\n")
            continue
        r = r.reset_index().rename(columns={f: 'FEATURE VALUE'})
        r.insert(0, 'FEATURE', f)
        res.append(r)
    full_res = pd.concat(res, ignore_index=True)
    #
    overview = {'FEATURE': "ALL_FEATURES",
                'FEATURE VALUE': "ALL_VALUES"}
    ov_dict = __cp_group(df, yt, yh, yp)
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


def __cp_group(x, y, yh, yp):
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


def regression_performance(X, y_true, y_pred, features:list=None):
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
    df = __preprocess_stratified(X, y_true, y_pred, features=features)
    yt, yh, yp = __get_yname_dict(df).values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    stratified_features = [f for f in df.columns.tolist() if f not in pred_cols]
    #
    res = []
    skipped_vars = []
    for f in stratified_features:
        if df[f].nunique() == 1:
            skipped_vars.append(f)
            continue
        # Data are expected in string format
        assert df[f].astype(str).eq(df[f]).all()
        # Add feature-specific performance values for each group in the feature
        grp = df.groupby(f)[pred_cols]
        try:
            r = grp.apply(lambda x: pd.Series(__rp_grp(x, yt, yh)))
        except BaseException as e:
            print(f"Error processing {f}. {e}\n")
            continue
        r = r.reset_index().rename(columns={f: 'FEATURE VALUE'})
        r.insert(0, 'FEATURE', f)
        res.append(r)
    full_res = pd.concat(res, ignore_index=True)
    #
    overview = {'FEATURE': "ALL_FEATURES",
                'FEATURE VALUE': "ALL_VALUES"}
    ov_dict = __rp_grp(df, yt, yh)
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


def __rp_grp(x, y, yh):
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


''' Fairness Reporting '''


def classification_fairness(X, y_true, y_pred, features:list=None, **kwargs):
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
    df = __preprocess_stratified(X, y_true, y_pred, features=features)
    yt, yh, yp = __get_yname_dict(df).values()
    pred_cols = [n for n in [yt, yh, yp] if n is not None]
    stratified_features = [f for f in df.columns.tolist() if f not in pred_cols]
    #
    res = []
    pa_name = 'prtc_attr'
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
                meas = __cf_group(pa_name, subset[yt], subset[yh], priv_grp=1)
            except BaseException as e:
                print(f"Error processing {f}. {e}\n")
                continue
            r = pd.DataFrame(meas, index=[0])
            r['N OBS'] = df.loc[df[f].eq(v), pa_name].sum()
            r['FEATURE'] = f
            r['FEATURE VALUE'] = v
            res.append(r)
    # Combine and format
    full_res = pd.concat(res, ignore_index=True)
    head_cols = ['FEATURE', 'FEATURE VALUE', 'N OBS']
    tail_cols = sorted([c for c in full_res.columns if c not in head_cols])
    rprt = full_res[head_cols + tail_cols]
    rprt = rprt.round(4)
    #
    return rprt


def __cf_group(pa_name, y_true, y_pred, priv_grp=1):
    """
    Returns a dict containing classification fairness measure values. Intended
    for use with pandas .apply() function.
    """
    gf_vals = {}
    gf_vals['PPV Ratio'] = \
        aif_mtrc.difference(sk_metric.precision_score, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['TPR Ratio'] = \
        aif_mtrc.ratio(clmtrc.true_positive_rate, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['FPR Ratio'] = \
        aif_mtrc.ratio(clmtrc.false_positive_rate, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['TNR Ratio'] = \
        aif_mtrc.ratio(clmtrc.true_negative_rate, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['FNR Ratio'] = \
        aif_mtrc.ratio(clmtrc.false_negative_rate, y_true, y_pred,
                       prot_attr=pa_name, priv_group=priv_grp)
    #
    gf_vals['PPV Diff'] = \
        aif_mtrc.difference(sk_metric.precision_score, y_true,
                            y_pred, prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['TPR Diff'] = \
        aif_mtrc.difference(clmtrc.true_positive_rate, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['FPR Diff'] = \
        aif_mtrc.difference(clmtrc.false_positive_rate, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['TNR Diff'] = \
        aif_mtrc.difference(clmtrc.true_negative_rate, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    gf_vals['FNR Diff'] = \
        aif_mtrc.difference(clmtrc.false_negative_rate, y_true, y_pred,
                            prot_attr=pa_name, priv_group=priv_grp)
    return gf_vals


def regression_fairness(X, y_true, y_pred, features:list=None, **kwargs):
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
    df = __preprocess_stratified(X, y_true, y_pred, features=features)
    yt, yh, yp = __get_yname_dict(df).values()
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
            meas = __rf_group(pa_name, subset[yt], subset[yh], priv_grp=1)
        except BaseException as e:
            # raise ValueError(f"Error processing {f}. {e}\n")
            print(f"Error processing {f}. {e}\n")
            continue
        r = pd.DataFrame(meas, index=[0])
        r['FEATURE'] = f
        r['FEATURE VALUE'] = v
        res_f.append(r)
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


def __rf_group(pa_name, y_true, y_pred, priv_grp=1):
    """
    Returns a dict containing regression fairness measure values. Intended for
    use with pandas .apply() function.
    """
    res = reports.__regres_group_fairness_measures(pa_name, y_true, y_pred,
                                                   priv_grp)
    return res
